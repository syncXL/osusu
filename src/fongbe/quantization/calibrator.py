from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, Iterable

import librosa
import numpy as np
import soundfile as sf
import torch
from compressed_tensors.quantization.utils import is_module_quantized
from datasets import Audio, load_dataset
from torch import nn

from fongbe.quantization.schemes import QuantizationSpec


def decode_hf_audio(audio_obj: dict[str, Any], target_sr: int = 16000) -> np.ndarray:
    """Decode HF Audio(decode=False) dict or already-decoded audio dict to mono float32 at target_sr."""
    if "array" in audio_obj and audio_obj["array"] is not None:
        wav = np.asarray(audio_obj["array"], dtype=np.float32)
        sr = int(audio_obj.get("sampling_rate", target_sr))
    else:
        raw_bytes = audio_obj.get("bytes")
        src = io.BytesIO(raw_bytes) if raw_bytes is not None else audio_obj["path"]
        wav, sr = sf.read(src, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return np.asarray(wav, dtype=np.float32)


def iter_hf_audio_samples(
    dataset_id: str = "nolimitsxl/osusu",
    dataset_config: str = "zindi_fon",
    split: str = "validation",
    max_samples: int = 64,
    target_sr: int = 16000,
    target_key: str = "syllables",
) -> list[dict[str, Any]]:
    """Stream up to `max_samples` decoded audio utterances from a Hugging Face dataset."""
    ds = load_dataset(dataset_id, dataset_config, split=split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    samples: list[dict[str, Any]] = []
    for row in ds:
        wav = decode_hf_audio(row["audio"], target_sr=target_sr)
        if wav.size == 0:
            continue
        text = row.get(target_key) or row.get("text") or ""
        samples.append({"array": wav, "sampling_rate": target_sr, "text": str(text)})
        if len(samples) >= max_samples:
            break
    return samples


@dataclass
class LayerActivationStats:
    """Collected activation statistics for a single quantized Linear layer."""

    min_val: float = float("inf")
    max_val: float = float("-inf")
    absmax: float = 0.0
    channel_absmax: torch.Tensor | None = None
    channel_abs_sum: torch.Tensor | None = None
    num_tokens: int = 0
    hessian: torch.Tensor | None = None
    reservoir: list[torch.Tensor] = field(default_factory=list)
    sample_inputs: list[torch.Tensor] = field(default_factory=list)
    _sample_tokens: int = 0

    def update(
        self,
        x: torch.Tensor,
        *,
        collect_hessian: bool = False,
        collect_samples: bool = False,
        reservoir_per_batch: int = 32768,
        max_reservoir_batches: int = 16,
        max_sample_tokens: int = 512,
    ) -> None:
        if x.numel() == 0:
            return
        x_2d = x.detach().reshape(-1, x.shape[-1]).to(device="cpu", dtype=torch.float32)
        n_tok = x_2d.shape[0]

        batch_min = float(x_2d.min().item())
        batch_max = float(x_2d.max().item())
        self.min_val = min(self.min_val, batch_min)
        self.max_val = max(self.max_val, batch_max)
        self.absmax = max(self.absmax, abs(batch_min), abs(batch_max))

        abs_x = x_2d.abs()
        ch_max = abs_x.amax(dim=0)
        ch_sum = abs_x.sum(dim=0)
        if self.channel_absmax is None:
            self.channel_absmax = ch_max
            self.channel_abs_sum = ch_sum
        else:
            self.channel_absmax = torch.maximum(self.channel_absmax, ch_max)
            self.channel_abs_sum = self.channel_abs_sum + ch_sum
        self.num_tokens += n_tok

        # Subsample flat activations for percentile estimation without unbounded RAM
        flat = x_2d.reshape(-1)
        if flat.numel() > reservoir_per_batch:
            step = max(1, flat.numel() // reservoir_per_batch)
            flat = flat[::step][:reservoir_per_batch]
        if len(self.reservoir) < max_reservoir_batches:
            self.reservoir.append(flat.clone())

        if collect_hessian:
            x64 = x_2d.to(torch.float64)
            gram = x64.T @ x64
            if self.hessian is None:
                self.hessian = gram
            else:
                self.hessian = self.hessian + gram

        if collect_samples and self._sample_tokens < max_sample_tokens:
            take = min(n_tok, max_sample_tokens - self._sample_tokens)
            self.sample_inputs.append(x_2d[:take].clone())
            self._sample_tokens += take

    def channel_mean_abs(self) -> torch.Tensor | None:
        if self.channel_abs_sum is None or self.num_tokens == 0:
            return None
        return self.channel_abs_sum / float(self.num_tokens)

    def get_hessian(self) -> torch.Tensor | None:
        if self.hessian is None or self.num_tokens == 0:
            return None
        return (self.hessian / float(self.num_tokens)).to(torch.float32)

    def get_sample_matrix(self) -> torch.Tensor | None:
        if not self.sample_inputs:
            return None
        return torch.cat(self.sample_inputs, dim=0)

    def activation_bounds(
        self,
        *,
        symmetric: bool,
        calib_method: str = "percentile",
        calib_percentile: float = 99.9,
    ) -> tuple[float, float]:
        if self.num_tokens == 0:
            return (-1.0, 1.0) if symmetric else (0.0, 1.0)

        if calib_method == "absmax" or not self.reservoir:
            if symmetric:
                return -self.absmax, self.absmax
            return self.min_val, self.max_val

        pooled = torch.cat(self.reservoir, dim=0)
        q = min(1.0, max(0.0, calib_percentile / 100.0))
        if symmetric:
            q_abs = float(torch.quantile(pooled.abs(), q).item())
            return -q_abs, q_abs

        tail = max(0.0, (1.0 - q) / 2.0)
        low = float(torch.quantile(pooled, tail).item())
        high = float(torch.quantile(pooled, 1.0 - tail).item())
        return low, high


def set_quantization_enabled(model: nn.Module, enabled: bool) -> None:
    """Toggle QDQ execution on all compressed-tensors quantized submodules."""
    for mod in model.modules():
        if is_module_quantized(mod):
            mod.quantization_enabled = enabled


def collect_activation_stats(
    model: nn.Module,
    spec: QuantizationSpec,
    calib_batches: Iterable[dict[str, torch.Tensor]],
    target_modules: dict[str, nn.Linear] | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, LayerActivationStats]:
    """Run calibration forward passes with static activation hooks and collect stats per layer."""
    if target_modules is None:
        target_modules = {
            name: mod
            for name, mod in model.named_modules()
            if isinstance(mod, nn.Linear) and is_module_quantized(mod)
        }

    stats: dict[str, LayerActivationStats] = {name: LayerActivationStats() for name in target_modules}
    collect_hessian = spec.algorithm == "gptq"
    collect_samples = spec.algorithm == "awq"

    hooks = []
    for name, mod in target_modules.items():
        def _make_hook(layer_name: str):
            def _hook(_mod, inputs):
                if inputs and isinstance(inputs[0], torch.Tensor):
                    stats[layer_name].update(
                        inputs[0],
                        collect_hessian=collect_hessian,
                        collect_samples=collect_samples,
                    )
            return _hook

        hooks.append(mod.register_forward_pre_hook(_make_hook(name)))

    run_full_decode = any(
        n.startswith("decoder.") or n.startswith("joint.") for n in target_modules
    )

    was_training = model.training
    model.eval()
    set_quantization_enabled(model, False)

    try:
        with torch.no_grad():
            for batch in calib_batches:
                moved = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                if run_full_decode and hasattr(model, "generate"):
                    gen_kwargs = {}
                    if getattr(model.generation_config, "decoder_start_token_id", None) is None:
                        blank_id = getattr(model.config, "blank_token_id", None)
                        if blank_id is not None:
                            gen_kwargs["decoder_start_token_id"] = int(blank_id)
                    model.generate(**moved, **gen_kwargs)
                elif hasattr(model, "get_audio_features"):
                    model.get_audio_features(
                        input_features=moved["input_features"],
                        attention_mask=moved.get("attention_mask"),
                    )
                else:
                    model(**moved)
    finally:
        for h in hooks:
            h.remove()
        set_quantization_enabled(model, True)
        if was_training:
            model.train()

    return stats
