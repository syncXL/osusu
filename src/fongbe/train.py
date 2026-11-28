import datetime
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import hydra
from omegaconf import OmegaConf
from lightning.pytorch import Trainer
from nemo.utils.exp_manager import exp_manager
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    checkpoint_wrapper,
)

from fongbe import callbacks as f_call
from fongbe.recipes import fine_tune_fongbe
from fongbe.model import setup_model_for_training


def enable_gradient_checkpointing(asr_model):
    for i, layer in enumerate(asr_model.encoder.layers):
        asr_model.encoder.layers[i] = checkpoint_wrapper(
            layer,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )


def resolve_head_lr(stage: str, encoder_lr: float, head_lr: float | None) -> float | None:
    """Resolve effective head LR; returns None when uniform LR should be used."""
    if head_lr is not None:
        resolved = float(head_lr)
        return None if resolved == float(encoder_lr) else resolved
    return 5e-4 if stage == "stage1" else None


def train_fongbe(
    model_name: str,
    exp_dir: str,
    repo_id: str,
    stage: str = "stage2",
    tokenizer_dir: str | None = None,
    run_name: str | None = None,
    path_in_repo: str = "",
    smoke_test: bool = False,
    upload_every_n_epochs: int = 1,
    upload_on_val: bool = True,
    upload_every_n_val_checks: int = 1,
    encoder_lr: float | None = None,
    head_lr: float | None = None,
    warmup_steps: int | None = None,
    init_encoder_from: str | None = None,
):
    config = fine_tune_fongbe.load_fongbe_cfg(
        run_name=run_name,
        model_name=model_name,
        exp_dir=exp_dir,
        stage=stage,
        tokenizer_dir=tokenizer_dir,
        smoke_test=smoke_test,
        encoder_lr=encoder_lr,
        warmup_steps=warmup_steps,
    )
    if run_name:
        config.name = run_name
        config.exp_manager.name = run_name
        config.exp_manager.checkpoint_callback_params.filename = (
            f"{run_name}--{{val_wer:.4f}}-{{step}}"
        )
    trainer_kwargs = OmegaConf.to_container(config.trainer, resolve=True)
    if isinstance(trainer_kwargs.get("strategy"), dict):
        trainer_kwargs["strategy"] = hydra.utils.instantiate(
            config.trainer.strategy,
            timeout=datetime.timedelta(minutes=5),
        )

    callbacks = [f_call.HFStreamCheckpointCallback()]
    trainer = Trainer(**trainer_kwargs, callbacks=callbacks)
    exp_manager(trainer, config.exp_manager)

    # Must come AFTER exp_manager so NeMoModelCheckpoint saves & frees RAM first
    trainer.callbacks.append(
        f_call.HFHubUploadCallback(
            repo_id=repo_id,
            exp_dir=exp_dir,
            path_in_repo=path_in_repo or stage,
            upload_every_n_epochs=upload_every_n_epochs,
            upload_on_val=upload_on_val,
            upload_every_n_val_checks=upload_every_n_val_checks,
        )
    )

    resolved_head_lr = resolve_head_lr(
        stage=stage,
        encoder_lr=float(config.model.optim.lr),
        head_lr=head_lr,
    )
    setup_kwargs = {"trainer": trainer, "head_lr": resolved_head_lr}
    if init_encoder_from is not None:
        setup_kwargs["init_encoder_from"] = init_encoder_from
    asr_model = setup_model_for_training(config, **setup_kwargs)
    enable_gradient_checkpointing(asr_model)

    return trainer, asr_model