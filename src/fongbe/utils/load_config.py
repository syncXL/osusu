from omegaconf import OmegaConf, open_dict, ListConfig,DictConfig
from pathlib import Path
## Finetune_fongbe config:

# def 

# Fongbe Finetune 
# Data support  DOne
# Tokenizer # Done
# Noise Augmentation
# Spec Augment/Data Augment#
# load from ckpt

REPOSITORY_ROOT = Path(__file__).parents[3].resolve()
CONFIG_ROOT = REPOSITORY_ROOT / "configs"

def add_tokenizer(config: ListConfig | DictConfig, tokenizer: str, t_type: str):
    config.model.tokenizer.update_tokenizer = True
    config.model.tokenizer.dir = tokenizer
    config.model.tokenizer.type = t_type
    return config

# - path: ${model.data_path}
# name: ${model.data_name}
# split: 'train.clean.360'
# streaming: ${model.streaming}
# trust_remote_code: true

def add_data(config: DictConfig | ListConfig, source: str, train_sets: list[str], validation_sets: list[str] | None = None):
    config.model.data_path = source
    config.model.streaming = True
    train_configs = [{"path": source, "name": name, "split": "train", "streaming": True} for name in train_sets]
    if validation_sets is None:
        validation_sets = train_sets
    validation_configs = [{"path": source, "name": name, "split": "validation", "streaming": True} for name in validation_sets]
    config.model.text_key = "text"
    config.model.train_ds.hf_data_cfg = train_configs
    config.model.validation_ds.hf_data_cfg = validation_configs
    return config


def load_finetune_config() -> ListConfig | DictConfig:
    config = OmegaConf.load(CONFIG_ROOT / "speech_to_text_hf_finetune.yaml")
    config.model.normalize_text = False
    config.model.symbols_to_keep = []
    return config


if __name__ == '__main__':
    print(REPOSITORY_ROOT)