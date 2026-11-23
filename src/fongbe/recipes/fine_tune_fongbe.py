from pathlib import Path
from fongbe.utils import load_config


def load_fongbe_cfg(
    model_name: str | None = None,
    exp_dir: str | None = None,
    stage: str = "stage2",
    tokenizer_dir: str | None = None,
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
    config = load_config.add_data(config, "nolimitsxl/fongbe-challenge", ["zindi_fon"])

    # Audio duration bounds (prevents T4 VRAM spikes)
    config.model.train_ds.max_duration = 60
    config.model.train_ds.min_duration = 0.3

    # Stage-specific hyperparameters
    if stage == "stage1":
        config.trainer.accumulate_grad_batches = 2  # Global batch = 64 on 2x T4
        config.model.optim.lr = 1e-4
        config.model.optim.sched.warmup_steps = None
        config.model.optim.sched.warmup_ratio = 0.08
        config.model.optim.sched.min_lr = 1e-6
        config.model.spec_augment.freq_width = 20
        config.model.spec_augment.time_masks = 5
        config.model.spec_augment.time_width = 0.04
    else:
        config.trainer.accumulate_grad_batches = 1  # Global batch = 32 on 2x T4
        config.model.optim.lr = 4e-5
        config.model.optim.sched.warmup_steps = None
        config.model.optim.sched.warmup_ratio = 0.10
        config.model.optim.sched.min_lr = 1e-6

    # Fixed log dir so Kaggle sessions resume cleanly
    config.exp_manager.exp_dir = None
    config.exp_manager.explicit_log_dir = exp_dir
    config.exp_manager.checkpoint_callback_params.save_top_k = 1
    config.exp_manager.checkpoint_callback_params.save_last = True

    if model_name:
        config.init_from_pretrained_model = model_name

    return config