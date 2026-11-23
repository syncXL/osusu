from nemo.collections.asr.models import ASRModel
from nemo.utils import model_utils


def setup_differential_lr(asr_model, encoder_lr: float, head_lr: float):
    enc_params = list(asr_model.encoder.parameters())
    enc_ids = {id(p) for p in enc_params}
    head_params = [p for p in asr_model.parameters() if id(p) not in enc_ids]
    asr_model._optimizer_param_groups = [
        {"params": enc_params, "lr": encoder_lr},
        {"params": head_params, "lr": head_lr},
    ]


def setup_model_for_training(config, trainer=None, head_lr: float | None = None):
    cfg = model_utils.convert_model_config_to_dict_config(config)
    asr_model = ASRModel.from_pretrained(model_name=cfg.init_from_pretrained_model)

    if trainer is not None:
        asr_model.set_trainer(trainer)

    decoder_state = asr_model.decoder.state_dict()
    joint_state = asr_model.joint.state_dict()
    prev_vocab_size = asr_model.tokenizer.vocab_size

    asr_model.change_vocabulary(
        new_tokenizer_dir=cfg.model.tokenizer.dir,
        new_tokenizer_type=cfg.model.tokenizer.type,
    )
    if asr_model.tokenizer.vocab_size == prev_vocab_size:
        asr_model.decoder.load_state_dict(decoder_state)
        asr_model.joint.load_state_dict(joint_state)

    if "spec_augment" in cfg.model and cfg.model.spec_augment is not None:
        asr_model.spec_augmentation = asr_model.from_config_dict(cfg.model.spec_augment)

    asr_model.setup_training_data(cfg.model.train_ds)
    asr_model.setup_validation_data(cfg.model.validation_ds)

    if head_lr is not None:
        setup_differential_lr(asr_model, encoder_lr=cfg.model.optim.lr, head_lr=head_lr)

    asr_model.setup_optimization(cfg.model.optim)
    return asr_model