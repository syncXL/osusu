from __future__ import annotations

import copy
import os
import random
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from nemo.collections.asr.data.ssl_dataset import AudioNoiseBatch
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.models.ssl_models import EncDecDenoiseMaskedTokenPredModel
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import IterableDataset

from fongbe.model import extract_encoder_state_dict, setup_differential_lr
from fongbe.recipes.pretrain_fongbe import decode_audio_sample


def crop_audio_waveform(
    wav: np.ndarray,
    *,
    sample_rate: int = 16000,
    crop_duration_sec: float | None = 20.0,
    crop_mode: str = "random",
    rng: random.Random | None = None,
) -> np.ndarray:
    """Crop an audio waveform if it exceeds ``crop_duration_sec``."""
    if crop_duration_sec is None or crop_mode == "none":
        return wav
    max_samples = int(round(float(crop_duration_sec) * int(sample_rate)))
    if max_samples <= 0 or len(wav) <= max_samples:
        return wav

    if crop_mode == "center":
        start = (len(wav) - max_samples) // 2
    elif crop_mode == "random":
        r = rng if rng is not None else random
        start = r.randint(0, len(wav) - max_samples)
    else:
        raise ValueError(f"Unsupported crop_mode={crop_mode!r}; expected 'random', 'center', or 'none'.")
    return np.ascontiguousarray(wav[start : start + max_samples])


def collate_ssl_audio_batch(samples: list[tuple[np.ndarray, int]]) -> AudioNoiseBatch:
    """Collate variable-length 1D audio arrays into a zero-noise ``AudioNoiseBatch``."""
    lengths = torch.tensor([len(wav) for wav, _ in samples], dtype=torch.long)
    max_len = int(lengths.max().item())
    batch_size = len(samples)
    audio = torch.zeros((batch_size, max_len), dtype=torch.float32)
    for i, (wav, _) in enumerate(samples):
        wav_t = torch.as_tensor(wav, dtype=torch.float32)
        audio[i, : len(wav)] = wav_t
    return AudioNoiseBatch(
        audio=audio,
        audio_len=lengths,
        noise=None,
        noise_len=None,
        noisy_audio=audio,
        noisy_audio_len=lengths,
    )


class SSLAudioDataset(IterableDataset):
    """Streaming audio dataset with duration filtering, optional cropping, and duration-bounded batching."""

    def __init__(
        self,
        dataset: Any,
        *,
        sample_rate: int = 16000,
        min_duration: float = 0.5,
        max_duration: float = 60.0,
        crop_duration_sec: float | None = 20.0,
        crop_mode: str = "random",
        max_batch_duration: float = 80.0,
        max_batch_size: int = 4,
        sort_buffer_size: int = 32,
        cycle: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.dataset = dataset
        self.sample_rate = int(sample_rate)
        self.min_duration = float(min_duration)
        self.max_duration = float(max_duration)
        self.crop_duration_sec = crop_duration_sec
        self.crop_mode = crop_mode
        self.max_batch_duration = float(max_batch_duration)
        self.max_batch_size = int(max_batch_size)
        self.sort_buffer_size = max(1, int(sort_buffer_size))
        self.cycle = bool(cycle)
        self.seed = int(seed)
        self._pass = 0
        self._rng = random.Random(self.seed)

    def state_dict(self) -> dict[str, Any]:
        if hasattr(self.dataset, "state_dict"):
            return self.dataset.state_dict()
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict and hasattr(self.dataset, "load_state_dict"):
            self.dataset.load_state_dict(state_dict)

    def _emit_sorted_batches(self, buf: list[tuple[float, np.ndarray]]) -> Iterator[AudioNoiseBatch]:
        buf.sort(key=lambda x: x[0])
        batch: list[tuple[np.ndarray, int]] = []
        max_dur_in_batch = 0.0

        for dur, wav in buf:
            cand_max = max(max_dur_in_batch, dur)
            cand_padded_dur = cand_max * (len(batch) + 1)
            if batch and (
                len(batch) >= self.max_batch_size or cand_padded_dur > self.max_batch_duration
            ):
                yield collate_ssl_audio_batch(batch)
                batch = [(wav, self.sample_rate)]
                max_dur_in_batch = dur
            else:
                batch.append((wav, self.sample_rate))
                max_dur_in_batch = cand_max

        if batch:
            yield collate_ssl_audio_batch(batch)

    def __iter__(self) -> Iterator[AudioNoiseBatch]:
        while True:
            if self._pass > 0 and hasattr(self.dataset, "set_epoch"):
                self.dataset.set_epoch(self._pass)
            self._pass += 1

            buf: list[tuple[float, np.ndarray]] = []
            yielded_any = False

            for raw_sample in self.dataset:
                try:
                    wav, sr = decode_audio_sample(raw_sample, target_sample_rate=self.sample_rate)
                except Exception:
                    continue

                raw_dur = len(wav) / float(sr)
                if raw_dur < self.min_duration or raw_dur > self.max_duration:
                    continue

                cropped = crop_audio_waveform(
                    wav,
                    sample_rate=sr,
                    crop_duration_sec=self.crop_duration_sec,
                    crop_mode=self.crop_mode,
                    rng=self._rng,
                )
                eff_dur = len(cropped) / float(sr)
                buf.append((eff_dur, cropped))

                if len(buf) >= self.sort_buffer_size:
                    for batch in self._emit_sorted_batches(buf):
                        yielded_any = True
                        yield batch
                    buf = []

            if buf:
                for batch in self._emit_sorted_batches(buf):
                    yielded_any = True
                    yield batch

            if not self.cycle or not yielded_any:
                break


class EncoderFreezeWarmupCallback(Callback):
    """Freeze ``pl_module.encoder`` during head warmup and unfreeze cleanly at ``freeze_steps``."""

    def __init__(self, freeze_steps: int = 500):
        super().__init__()
        self.freeze_steps = max(0, int(freeze_steps))
        self._encoder_frozen: bool | None = None

    @staticmethod
    def _set_encoder_frozen(pl_module: Any, freeze: bool) -> None:
        encoder = getattr(pl_module, "encoder", None)
        if encoder is None:
            return
        for param in encoder.parameters():
            param.requires_grad = not freeze
            if freeze:
                param.grad = None
        encoder.eval() if freeze else encoder.train()

    def _sync_freeze_state(self, trainer: Any, pl_module: Any) -> None:
        if self.freeze_steps <= 0:
            if self._encoder_frozen is not False:
                self._set_encoder_frozen(pl_module, False)
                self._encoder_frozen = False
            return

        should_freeze = int(getattr(trainer, "global_step", 0)) < self.freeze_steps
        if self._encoder_frozen is not should_freeze:
            self._set_encoder_frozen(pl_module, should_freeze)
            self._encoder_frozen = should_freeze

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        self._sync_freeze_state(trainer, pl_module)

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        self._sync_freeze_state(trainer, pl_module)

    def on_train_batch_start(self, trainer: Any, pl_module: Any, batch: Any, batch_idx: int) -> None:
        self._sync_freeze_state(trainer, pl_module)


def _compute_masked_token_accuracy(
    log_probs: torch.Tensor,
    masks: torch.Tensor,
    tokens: torch.Tensor,
    *,
    combine_time_steps: int = 8,
    mask_threshold: float = 0.8,
) -> tuple[torch.Tensor, int]:
    """Compute `(num_correct, num_total)` masked-token predictions matching `MLMLoss`."""
    t_masks = masks.transpose(1, 2)
    valid_t = (t_masks.shape[1] // combine_time_steps) * combine_time_steps
    t_masks = t_masks[:, :valid_t, :]
    t_masks = t_masks.reshape(t_masks.shape[0], valid_t // combine_time_steps, -1)
    bool_mask = t_masks.mean(-1) > mask_threshold

    if bool_mask.sum().item() == 0:
        return torch.tensor(0, device=log_probs.device, dtype=torch.long), 0

    aligned_probs = log_probs[:, : bool_mask.shape[1]]
    aligned_tokens = tokens[:, : bool_mask.shape[1]]
    if aligned_tokens.shape[1] < bool_mask.shape[1]:
        aligned_tokens = torch.nn.functional.pad(
            aligned_tokens, (0, bool_mask.shape[1] - aligned_tokens.shape[1])
        )

    preds = aligned_probs[bool_mask].argmax(dim=-1)
    targets = aligned_tokens[bool_mask]
    return (preds == targets).sum(), int(targets.numel())


class FongbeNESTModel(EncDecDenoiseMaskedTokenPredModel):
    """Pure masked-prediction NEST SSL model logging `val_loss` and masked-token `val_acc`."""

    def __init__(self, cfg: DictConfig, trainer=None):
        super().__init__(cfg=cfg, trainer=trainer)
        # NeMo initializes quantizer.codebooks in float64 while proj output is float32;
        # align both to float32 so einsum works cleanly on CPU and under 16-mixed AMP.
        if hasattr(self, "quantizer") and hasattr(self.quantizer, "codebooks"):
            self.quantizer.codebooks.data = self.quantizer.codebooks.data.float()

    def setup_training_data(self, train_data_config):
        # DataLoaders are injected cleanly via streaming datasets.
        return

    def setup_validation_data(self, val_data_config):
        return

    def _run_ssl_forward_and_loss(
        self,
        batch: AudioNoiseBatch,
        *,
        apply_mask: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if batch.noisy_audio is not None and batch.noisy_audio is not batch.audio:
            log_probs, encoded_len, masks, tokens = self.forward(
                input_signal=batch.audio,
                input_signal_length=batch.audio_len,
                noisy_input_signal=batch.noisy_audio,
                noisy_input_signal_length=batch.noisy_audio_len,
                apply_mask=apply_mask,
            )
        else:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=batch.audio,
                length=batch.audio_len,
            )
            with torch.no_grad(), torch.amp.autocast(
                device_type=processed_signal.device.type, enabled=False
            ):
                _, tokens = self.quantizer(input_signal=processed_signal.float())
            if apply_mask:
                masked_signal, masks = self.mask_processor(
                    input_feats=processed_signal, input_lengths=processed_signal_length
                )
            else:
                masked_signal = processed_signal
                masks = torch.zeros_like(processed_signal)
            encoded, encoded_len = self.encoder(
                audio_signal=masked_signal, length=processed_signal_length
            )
            log_probs = self.decoder(encoder_output=encoded)

        loss_value = self.loss(
            masks=masks,
            decoder_outputs=log_probs,
            targets=tokens,
            decoder_lengths=encoded_len,
        )
        return loss_value, log_probs, encoded_len, masks, tokens

    def training_step(self, batch: AudioNoiseBatch, batch_idx: int):
        loss_value, _, _, _, _ = self._run_ssl_forward_and_loss(batch, apply_mask=True)
        self.log("train_loss", loss_value, prog_bar=True, on_step=True, on_epoch=False)
        tensorboard_logs = {
            "learning_rate": self._optimizer.param_groups[0]["lr"],
            "global_step": self.trainer.global_step,
            "train_loss": loss_value,
        }
        return {"loss": loss_value, "log": tensorboard_logs}

    def inference_pass(
        self,
        batch: AudioNoiseBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
        mode: str = "val",
        apply_mask: bool = True,
    ):
        loss_value, log_probs, _, masks, tokens = self._run_ssl_forward_and_loss(
            batch, apply_mask=apply_mask
        )
        combine_steps = int(self.cfg.loss.get("combine_time_steps", 8))
        mask_threshold = float(self.cfg.loss.get("mask_threshold", 0.8))
        correct, total = _compute_masked_token_accuracy(
            log_probs,
            masks,
            tokens,
            combine_time_steps=combine_steps,
            mask_threshold=mask_threshold,
        )
        acc = correct.float() / max(1, total)
        return {
            f"{mode}_loss": loss_value,
            f"{mode}_acc": acc,
            f"{mode}_correct": correct,
            f"{mode}_total": torch.tensor(total, device=loss_value.device, dtype=torch.long),
        }

    def multi_validation_epoch_end(self, outputs: list, dataloader_idx: int = 0):
        loss_list = [x["val_loss"] for x in outputs if isinstance(x, dict) and "val_loss" in x]
        if not loss_list:
            return {}
        val_loss_mean = torch.stack(loss_list).mean()

        if all("val_correct" in x and "val_total" in x for x in outputs if isinstance(x, dict)):
            total_correct = torch.stack([x["val_correct"] for x in outputs]).sum().float()
            total_count = torch.stack([x["val_total"] for x in outputs]).sum().clamp_min(1).float()
            val_acc_mean = total_correct / total_count
        else:
            acc_list = [x["val_acc"] for x in outputs if isinstance(x, dict) and "val_acc" in x]
            val_acc_mean = torch.stack(acc_list).mean() if acc_list else torch.tensor(0.0, device=val_loss_mean.device)

        logs = {"val_loss": val_loss_mean, "val_acc": val_acc_mean}
        return {"val_loss": val_loss_mean, "val_acc": val_acc_mean, "log": logs}


def _load_base_asr_model(base_model: str):
    if os.path.exists(base_model) and base_model.endswith(".nemo"):
        return ASRModel.restore_from(base_model, map_location="cpu")
    return ASRModel.from_pretrained(model_name=base_model, map_location="cpu")


def build_nest_model_config(
    *,
    preprocessor_cfg: dict[str, Any] | DictConfig | None = None,
    encoder_cfg: dict[str, Any] | DictConfig | None = None,
    optim_cfg: dict[str, Any] | DictConfig | None = None,
    num_classes: int = 8192,
    code_dim: int = 16,
    mask_prob: float = 0.5,
    block_size: int = 48,
) -> DictConfig:
    """Build a NeMo NEST config matching Parakeet-TDT 0.6B's preprocessor and FastConformer."""
    if preprocessor_cfg is not None:
        pre_dict = copy.deepcopy(OmegaConf.to_container(OmegaConf.create(preprocessor_cfg), resolve=True))
    else:
        pre_dict = {
            "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
            "sample_rate": 16000,
            "normalize": "per_feature",
            "window_size": 0.025,
            "window_stride": 0.01,
            "window": "hann",
            "features": 128,
            "n_fft": 512,
            "log": True,
            "frame_splicing": 1,
            "dither": 1e-5,
            "pad_value": 0.0,
        }
    # Ensure spectrogram time axis always divides cleanly by subsampling_factor=8
    pre_dict["pad_to"] = 16

    if encoder_cfg is not None:
        enc_dict = copy.deepcopy(OmegaConf.to_container(OmegaConf.create(encoder_cfg), resolve=True))
    else:
        enc_dict = {
            "_target_": "nemo.collections.asr.modules.ConformerEncoder",
            "feat_in": 128,
            "feat_out": -1,
            "n_layers": 24,
            "d_model": 1024,
            "use_bias": False,
            "subsampling": "dw_striding",
            "subsampling_factor": 8,
            "subsampling_conv_channels": 256,
            "causal_downsampling": False,
            "ff_expansion_factor": 4,
            "self_attention_model": "rel_pos",
            "n_heads": 8,
            "att_context_size": [-1, -1],
            "att_context_style": "regular",
            "xscaling": False,
            "untie_biases": True,
            "pos_emb_max_len": 5000,
            "conv_kernel_size": 9,
            "conv_norm_type": "batch_norm",
            "dropout": 0.1,
            "dropout_pre_encoder": 0.1,
            "dropout_emb": 0.0,
            "dropout_att": 0.1,
            "stochastic_depth_drop_prob": 0.0,
        }
    enc_dict["subsampling_conv_chunking_factor"] = 1

    feat_in = int(enc_dict.get("feat_in", 128))
    d_model = int(enc_dict.get("d_model", 1024))
    combine_steps = int(enc_dict.get("subsampling_factor", 8))

    default_optim = {
        "name": "adamw",
        "lr": 4e-5,
        "betas": [0.9, 0.98],
        "weight_decay": 1e-3,
        "sched": {
            "name": "CosineAnnealing",
            "warmup_steps": 500,
            "warmup_ratio": None,
            "min_lr": 1e-6,
        },
    }
    if optim_cfg is not None:
        resolved_optim = copy.deepcopy(OmegaConf.to_container(OmegaConf.create(optim_cfg), resolve=True))
    else:
        resolved_optim = default_optim

    return OmegaConf.create(
        {
            "num_classes": num_classes,
            "num_books": 1,
            "code_dim": code_dim,
            "squeeze_single": True,
            "mask_position": "pre_conv",
            "preprocessor": pre_dict,
            "encoder": enc_dict,
            "quantizer": {
                "_target_": "nemo.collections.asr.modules.ssl_modules.RandomProjectionVectorQuantizer",
                "feat_in": feat_in,
                "code_dim": code_dim,
                "num_classes": num_classes,
                "num_books": 1,
                "dist_fn": "cosine",
                "freeze": True,
                "squeeze_single": True,
                "combine_time_steps": combine_steps,
            },
            "masking": {
                "_target_": "nemo.collections.asr.modules.ssl_modules.RandomBlockMasking",
                "feat_in": feat_in,
                "mask_prob": mask_prob,
                "block_size": block_size,
                "mask_value": None,
                "freeze": True,
                "allow_overlap": False,
            },
            "decoder": {
                "_target_": "nemo.collections.asr.modules.ssl_modules.MultiSoftmaxDecoder",
                "feat_in": d_model,
                "num_classes": num_classes,
                "num_decoders": 1,
                "use_bias": False,
                "squeeze_single": True,
            },
            "loss": {
                "_target_": "nemo.collections.asr.losses.ssl_losses.mlm.MLMLoss",
                "combine_time_steps": combine_steps,
                "mask_threshold": 0.8,
            },
            "spec_augment": {
                "_target_": "nemo.collections.asr.modules.SpectrogramAugmentation",
                "freq_masks": 0,
                "time_masks": 0,
            },
            "optim": resolved_optim,
        }
    )


def setup_ssl_model(
    *,
    base_model: str = "nvidia/parakeet-tdt-0.6b-v3",
    trainer: Any = None,
    encoder_lr: float = 4e-5,
    head_lr: float = 5e-4,
    warmup_steps: int = 500,
    min_lr: float = 1e-6,
    train_dl: Any = None,
    val_dl: Any = None,
) -> FongbeNESTModel:
    """Instantiate ``FongbeNESTModel``, copy Parakeet encoder weights, and set up optimizer."""
    base_asr = _load_base_asr_model(base_model)
    optim_cfg = {
        "name": "adamw",
        "lr": float(encoder_lr),
        "betas": [0.9, 0.98],
        "weight_decay": 1e-3,
        "sched": {
            "name": "CosineAnnealing",
            "warmup_steps": int(warmup_steps),
            "warmup_ratio": None,
            "min_lr": float(min_lr),
        },
    }
    ssl_cfg = build_nest_model_config(
        preprocessor_cfg=base_asr.cfg.preprocessor,
        encoder_cfg=base_asr.cfg.encoder,
        optim_cfg=optim_cfg,
    )
    ssl_model = FongbeNESTModel(cfg=ssl_cfg, trainer=trainer)

    enc_sd = extract_encoder_state_dict({"state_dict": base_asr.state_dict()})
    ssl_model.encoder.load_state_dict(enc_sd, strict=True)
    ssl_model.preprocessor.load_state_dict(base_asr.preprocessor.state_dict(), strict=False)
    del base_asr

    if hasattr(ssl_model, "change_subsampling_conv_chunking_factor"):
        ssl_model.change_subsampling_conv_chunking_factor(1)

    if train_dl is not None:
        ssl_model._train_dl = train_dl
    if val_dl is not None:
        ssl_model._validation_dl = val_dl

    if float(head_lr) != float(encoder_lr):
        setup_differential_lr(ssl_model, encoder_lr=float(encoder_lr), head_lr=float(head_lr))

    ssl_model.setup_optimization(ssl_cfg.optim)
    return ssl_model
