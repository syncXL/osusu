import io
from typing import Any

import soundfile as sf
from omegaconf import open_dict
from torch.utils.data import DataLoader, IterableDataset
from nemo.collections.asr.data.huggingface import hf_audio_to_text
from nemo.collections.asr.models import ASRModel
from nemo.utils import model_utils

_orig_get_nested = hf_audio_to_text.get_nested_dict_value


def _decode_bytes_aware_nested_get(dictionary, key):
    # Decode raw audio bytes lazily when HF dataset uses Audio(decode=False)
    if key.startswith("audio.") and isinstance(dictionary.get("audio"), dict):
        audio = dictionary["audio"]
        if "array" not in audio and ("bytes" in audio or "path" in audio):
            raw_bytes = audio.get("bytes")
            wav, sr = sf.read(io.BytesIO(raw_bytes) if raw_bytes is not None else audio["path"], dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            audio["array"] = wav
            audio["sampling_rate"] = sr
    return _orig_get_nested(dictionary, key)


hf_audio_to_text.get_nested_dict_value = _decode_bytes_aware_nested_get


class DynamicDurationBatchDataset(IterableDataset):
    """Packs streaming utterances into length-sorted, padded-duration-bounded batches.

    Sorts small local windows (`sort_buffer_size`) by utterance duration so similar lengths
    pair together while keeping `max_dur_in_batch * len(batch) <= max_batch_duration` and
    `len(batch) <= max_batch_size` (emitting `bs=1` when a single clip exceeds half the budget).
    """

    def __init__(
        self,
        dataset,
        max_batch_duration: float = 60.0,
        max_batch_size: int = 2,
        sample_rate: int = 16000,
        sort_buffer_size: int = 32,
        min_duration: float | None = None,
        max_duration: float | None = None,
        collate_fn=None,
    ):
        super().__init__()
        self.dataset = dataset
        self.max_batch_duration = float(max_batch_duration)
        self.max_batch_size = int(max_batch_size)
        self.sample_rate = int(sample_rate)
        self.sort_buffer_size = max(1, int(sort_buffer_size))
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.collate_fn = collate_fn if collate_fn is not None else getattr(dataset, "collate_fn", None)

    def _duration_sec(self, sample: Any) -> float:
        if isinstance(sample, dict):
            sig_len = sample.get("a_sig_length")
            if sig_len is None and "audio_signal" in sample:
                sig_len = len(sample["audio_signal"])
        else:
            sig_len = sample[1] if len(sample) > 1 else len(sample[0])
        if hasattr(sig_len, "item"):
            sig_len = sig_len.item()
        return float(sig_len) / self.sample_rate

    def _emit_sorted_batches(self, buf: list[tuple[float, Any]]):
        buf.sort(key=lambda pair: pair[0])
        batch: list[Any] = []
        batch_max_dur = 0.0
        for dur, sample in buf:
            new_max = max(batch_max_dur, dur)
            new_len = len(batch) + 1
            if batch and (new_len > self.max_batch_size or new_max * new_len > self.max_batch_duration):
                yield self.collate_fn(batch) if self.collate_fn is not None else batch
                batch = [sample]
                batch_max_dur = dur
            else:
                batch.append(sample)
                batch_max_dur = new_max
        if batch:
            yield self.collate_fn(batch) if self.collate_fn is not None else batch

    def __iter__(self):
        buf: list[tuple[float, Any]] = []
        for sample in self.dataset:
            dur = self._duration_sec(sample)
            if self.min_duration is not None and dur < self.min_duration:
                continue
            if self.max_duration is not None and dur > self.max_duration:
                continue
            buf.append((dur, sample))
            if len(buf) >= self.sort_buffer_size:
                yield from self._emit_sorted_batches(buf)
                buf.clear()
        if buf:
            yield from self._emit_sorted_batches(buf)

    def _resolve_stateful_dataset(self):
        ds = self.dataset
        while ds is not None and not (hasattr(ds, "state_dict") and hasattr(ds, "load_state_dict")):
            nxt = getattr(ds, "dataset", None)
            if nxt is ds:
                return None
            ds = nxt
        return ds

    def state_dict(self):
        ds = self._resolve_stateful_dataset()
        if ds is not None:
            return ds.state_dict()
        return {}

    def load_state_dict(self, state_dict):
        ds = self._resolve_stateful_dataset()
        if ds is not None:
            ds.load_state_dict(state_dict)


def _apply_joint_config(asr_model, joint_cfg) -> None:
    if joint_cfg is None:
        return
    fused_bs = joint_cfg.get("fused_batch_size")
    fuse_loss_wer = joint_cfg.get("fuse_loss_wer")
    for target in (
        getattr(asr_model.cfg, "joint", None),
        getattr(getattr(asr_model, "joint", None), "_cfg", None),
    ):
        if target is None:
            continue
        with open_dict(target):
            if fused_bs is not None:
                target.fused_batch_size = fused_bs
            if fuse_loss_wer is not None:
                target.fuse_loss_wer = fuse_loss_wer
    if fused_bs is not None:
        decoding_cfg = getattr(asr_model.cfg, "decoding", None)
        if decoding_cfg is not None and decoding_cfg.get("fused_batch_size") is not None:
            with open_dict(decoding_cfg):
                decoding_cfg.fused_batch_size = fused_bs


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

    # Copy fused_batch_size onto joint config before change_vocabulary rebuilds joint
    _apply_joint_config(asr_model, cfg.model.get("joint"))

    asr_model.change_vocabulary(
        new_tokenizer_dir=cfg.model.tokenizer.dir,
        new_tokenizer_type=cfg.model.tokenizer.type,
    )
    if asr_model.tokenizer.vocab_size == prev_vocab_size:
        asr_model.decoder.load_state_dict(decoder_state)
        asr_model.joint.load_state_dict(joint_state)

    if hasattr(asr_model, "change_subsampling_conv_chunking_factor"):
        asr_model.change_subsampling_conv_chunking_factor(1)

    if "spec_augment" in cfg.model and cfg.model.spec_augment is not None:
        asr_model.spec_augmentation = asr_model.from_config_dict(cfg.model.spec_augment)

    asr_model.setup_training_data(cfg.model.train_ds)
    batch_duration = cfg.model.train_ds.get("batch_duration")
    if batch_duration is not None and asr_model._train_dl is not None:
        base_ds = asr_model._train_dl.dataset
        collate_fn = getattr(asr_model._train_dl, "collate_fn", None) or getattr(base_ds, "collate_fn", None)
        dynamic_ds = DynamicDurationBatchDataset(
            dataset=base_ds,
            max_batch_duration=float(batch_duration),
            max_batch_size=int(cfg.model.train_ds.get("batch_size", 2)),
            sample_rate=int(cfg.model.train_ds.get("sample_rate", cfg.model.get("sample_rate", 16000))),
            sort_buffer_size=int(cfg.model.train_ds.get("sort_buffer_size", 32)),
            min_duration=cfg.model.train_ds.get("min_duration"),
            max_duration=cfg.model.train_ds.get("max_duration"),
            collate_fn=collate_fn,
        )
        asr_model._train_dl = DataLoader(
            dynamic_ds,
            batch_size=None,
            num_workers=0,
            pin_memory=False,
        )

    asr_model.setup_validation_data(cfg.model.validation_ds)

    if head_lr is not None:
        setup_differential_lr(asr_model, encoder_lr=cfg.model.optim.lr, head_lr=head_lr)

    asr_model.setup_optimization(cfg.model.optim)
    return asr_model