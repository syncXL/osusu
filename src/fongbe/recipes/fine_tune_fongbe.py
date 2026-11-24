from pathlib import Path
from fongbe.utils import load_config


def load_fongbe_cfg(
    model_name: str | None = None,
    exp_dir: str | None = None,
    stage: str = "stage2",
    tokenizer_dir: str | None = None,
    smoke_test: bool = False,
):
    repo_root = Path(__file__).parents[3].resolve()
    default_tok = (
        "tokenizers/tokenizer_spe_unigram_v1024"
        if stage == "stage1"
        else "tokenizers/tokenizer_spe_unigram_v512"
    )
    tok_path = Path(tokenizer_dir) if tokenizer_dir else (repo_root / default_tok)

    config = load_config.load_finetune_config()
    config = load_config.add_tokenizer(config, str(tok_path), "bpe")
    config = load_config.add_data(config, "nolimitsxl/osusu", ["zindi_fon"])

    config.model.train_ds.max_duration = 60.0
    config.model.train_ds.min_duration = 0.3

    # 1. Name the model & checkpoint files cleanly
    resolved_name = f"parakeet-tdt-0.6b-fongbe-{stage}"
    config.name = resolved_name
    config.exp_manager.name = resolved_name

    # 2. Configure checkpoint rotation & filename pattern
    ckpt_params = config.exp_manager.checkpoint_callback_params
    ckpt_params.filename = f"{resolved_name}--{{val_wer:.4f}}-{{step}}"
    ckpt_params.always_save_nemo = True

    if stage == "stage1":
        config.model.train_ds.batch_size = 8
        config.trainer.accumulate_grad_batches = 4   # 8 * 2 GPUs * 4 = 64 global batch
        config.model.optim.lr = 1e-4
        config.model.optim.sched.warmup_steps = None
        config.model.optim.sched.warmup_ratio = 0.08
        config.model.optim.sched.min_lr = 1e-6
        config.model.spec_augment.freq_width = 20
        config.model.spec_augment.time_masks = 5
        config.model.spec_augment.time_width = 0.04
    else:
        config.model.train_ds.batch_size = 8
        config.trainer.accumulate_grad_batches = 2   # 8 * 2 GPUs * 2 = 32 global batch
        config.model.optim.lr = 4e-5
        config.model.optim.sched.warmup_steps = None
        config.model.optim.sched.warmup_ratio = 0.10
        config.model.optim.sched.min_lr = 1e-6

    if smoke_test:
        config.trainer.max_epochs = 1
        config.trainer.max_steps = 20
        config.trainer.val_check_interval = 10
        config.trainer.log_every_n_steps = 5
        config.trainer.limit_val_batches = 5

    # Required integer batch limit for NeMo HF streaming
    config.trainer.limit_train_batches = (
        config.trainer.max_steps * config.trainer.accumulate_grad_batches
    )

    config.exp_manager.exp_dir = None
    config.exp_manager.explicit_log_dir = exp_dir
    config.exp_manager.checkpoint_callback_params.save_top_k = 1
    config.exp_manager.checkpoint_callback_params.save_last = True

    if model_name:
        config.init_from_pretrained_model = model_name

    return config