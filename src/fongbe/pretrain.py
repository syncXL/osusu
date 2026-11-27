from __future__ import annotations

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.strategies import DDPStrategy
from nemo.utils.exp_manager import exp_manager
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from fongbe import callbacks as f_call
from fongbe.recipes.pretrain_fongbe import DEFAULT_DATASET_ID, build_pretrain_streams
from fongbe.ssl import EncoderFreezeWarmupCallback, FongbeNESTModel, SSLAudioDataset, setup_ssl_model
from fongbe.train import enable_gradient_checkpointing


def pretrain_fongbe(
    base_model: str = "nvidia/parakeet-tdt-0.6b-v3",
    exp_dir: str = "/tmp/osusu_ssl",
    repo_id: str = "nolimitsxl/osusu-asr",
    run_name: str = "parakeet-tdt-0.6b-osusu-ssl",
    path_in_repo: str = "ssl",
    dataset_id: str = DEFAULT_DATASET_ID,
    encoder_lr: float = 4e-5,
    head_lr: float = 5e-4,
    freeze_encoder_steps: int = 500,
    warmup_steps: int = 500,
    min_lr: float = 1e-6,
    global_batch_size: int = 64,
    max_batch_size: int = 4,
    max_batch_duration: float = 80.0,
    min_duration: float = 0.5,
    max_duration: float = 60.0,
    crop_duration_sec: float = 20.0,
    sort_buffer_size: int = 32,
    shuffle_buffer_size: int = 256,
    beta_corpus: float = 0.5,
    duration_adjusted: bool = True,
    max_epochs: int = 4,
    steps_per_epoch: int | None = None,
    val_check_interval: int | None = None,
    limit_val_batches: int = 200,
    seed: int = 42,
    upload_every_n_epochs: int = 1,
    smoke_test: bool = False,
) -> tuple[Trainer, FongbeNESTModel]:
    """Configure and prepare a NEST masked-prediction SSL run on ``nolimitsxl/osusu``.

    Follows the same ergonomics and Kaggle 2x T4 memory/disk invariants as ``train_fongbe``:
    ``num_workers=0``, ``pin_memory=False``, ``16-mixed`` precision, infinite stream
    cycling during training, ``CheckpointImpl.NO_REENTRANT`` activation checkpointing,
    and ``HFHubUploadCallback`` appended *after* ``exp_manager``.
    """
    has_cuda = torch.cuda.is_available()
    num_devices = max(1, torch.cuda.device_count()) if has_cuda else 1

    if smoke_test:
        max_epochs = 1
        steps_per_epoch = steps_per_epoch or 10
        val_check_interval = val_check_interval or 5
        limit_val_batches = min(limit_val_batches, 2)
        warmup_steps = min(warmup_steps, 2)
        freeze_encoder_steps = min(freeze_encoder_steps, 2)
        log_every_n_steps = 5
    else:
        steps_per_epoch = steps_per_epoch or 4000
        log_every_n_steps = 100

    accumulate_grad_batches = max(1, int(global_batch_size) // (num_devices * int(max_batch_size)))
    max_steps = int(max_epochs) * int(steps_per_epoch)
    limit_train_batches = int(steps_per_epoch) * int(accumulate_grad_batches)
    resolved_val_check = (
        int(val_check_interval) if val_check_interval is not None else limit_train_batches
    )

    strategy = (
        DDPStrategy(
            gradient_as_bucket_view=True,
            find_unused_parameters=freeze_encoder_steps > 0,
        )
        if num_devices > 1
        else "auto"
    )

    callbacks = [
        f_call.HFStreamCheckpointCallback(),
        EncoderFreezeWarmupCallback(freeze_steps=freeze_encoder_steps),
    ]
    trainer = Trainer(
        devices=-1 if has_cuda else 1,
        num_nodes=1,
        max_epochs=int(max_epochs),
        max_steps=max_steps,
        limit_train_batches=limit_train_batches,
        val_check_interval=resolved_val_check,
        check_val_every_n_epoch=1,
        limit_val_batches=int(limit_val_batches),
        accelerator="gpu" if has_cuda else "cpu",
        strategy=strategy,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=1.0,
        precision="16-mixed" if has_cuda else "32-true",
        log_every_n_steps=log_every_n_steps,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        sync_batchnorm=False,
        enable_checkpointing=False,
        logger=False,
        benchmark=False,
        callbacks=callbacks,
    )

    exp_cfg = OmegaConf.create(
        {
            "exp_dir": None,
            "explicit_log_dir": exp_dir,
            "name": run_name,
            "create_tensorboard_logger": True,
            "create_checkpoint_callback": True,
            "checkpoint_callback_params": {
                "monitor": "val_loss",
                "mode": "min",
                "save_top_k": 1,
                "save_last": True,
                "always_save_nemo": False,
                "save_nemo_on_train_end": True,
                "filename": f"{run_name}--{{val_loss:.4f}}-{{step}}",
            },
            "resume_if_exists": True,
            "resume_ignore_no_checkpoint": True,
            "create_wandb_logger": False,
        }
    )
    exp_manager(trainer, exp_cfg)

    # Must come AFTER exp_manager so NeMoModelCheckpoint saves & frees RAM first
    trainer.callbacks.append(
        f_call.HFHubUploadCallback(
            repo_id=repo_id,
            exp_dir=exp_dir,
            path_in_repo=path_in_repo,
            upload_every_n_epochs=upload_every_n_epochs,
        )
    )

    train_stream, val_stream, _ = build_pretrain_streams(
        dataset_id=dataset_id,
        beta_corpus=beta_corpus,
        duration_adjusted=duration_adjusted,
        crop_duration_sec=crop_duration_sec,
        seed=seed,
        shuffle_buffer_size=shuffle_buffer_size,
        rank=trainer.global_rank,
        world_size=trainer.world_size,
    )

    train_ds = SSLAudioDataset(
        train_stream,
        sample_rate=16000,
        min_duration=min_duration,
        max_duration=max_duration,
        crop_duration_sec=crop_duration_sec,
        crop_mode="random",
        max_batch_duration=max_batch_duration,
        max_batch_size=max_batch_size,
        sort_buffer_size=sort_buffer_size,
        cycle=True,
        seed=seed + trainer.global_rank,
    )
    val_ds = SSLAudioDataset(
        val_stream,
        sample_rate=16000,
        min_duration=min_duration,
        max_duration=max_duration,
        crop_duration_sec=None,
        crop_mode="none",
        max_batch_duration=max_batch_duration,
        max_batch_size=max_batch_size,
        sort_buffer_size=sort_buffer_size,
        cycle=False,
        seed=seed,
    )

    train_dl = DataLoader(train_ds, batch_size=None, num_workers=0, pin_memory=False)
    val_dl = DataLoader(val_ds, batch_size=None, num_workers=0, pin_memory=False)

    ssl_model = setup_ssl_model(
        base_model=base_model,
        trainer=trainer,
        encoder_lr=encoder_lr,
        head_lr=head_lr,
        warmup_steps=warmup_steps,
        min_lr=min_lr,
        train_dl=train_dl,
        val_dl=val_dl,
    )
    enable_gradient_checkpointing(ssl_model)
    return trainer, ssl_model
