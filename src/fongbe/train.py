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


def train_fongbe(
    model_name: str,
    exp_dir: str,
    repo_id: str,
    stage: str = "stage2",
    tokenizer_dir: str | None = None,
    run_name: str | None = None,
    path_in_repo: str = "",
    smoke_test: bool = False,
    upload_every_n_epochs: int = 1
):
    config = fine_tune_fongbe.load_fongbe_cfg(
        run_name=run_name,
        model_name=model_name,
        exp_dir=exp_dir,
        stage=stage,
        tokenizer_dir=tokenizer_dir,
        smoke_test=smoke_test,
    )
    if run_name:
        config.name = run_name
        config.exp_manager.name = run_name
        config.exp_manager.checkpoint_callback_params.filename = (
            f"{run_name}--{{val_wer:.4f}}-{{step}}"
        )
    trainer_kwargs = OmegaConf.to_container(config.trainer, resolve=True)
    if isinstance(trainer_kwargs.get("strategy"), dict):
        trainer_kwargs["strategy"] = hydra.utils.instantiate(config.trainer.strategy)

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
            )
    )


    head_lr = 5e-4 if stage == "stage1" else None
    asr_model = setup_model_for_training(config, trainer=trainer, head_lr=head_lr)
    enable_gradient_checkpointing(asr_model)

    return trainer, asr_model