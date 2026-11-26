from __future__ import annotations

from pathlib import Path

import torch
from compressed_tensors.compressors import ModelCompressor
from huggingface_hub import HfApi
from torch import nn


def save_quantized_hf_model(
    model: nn.Module,
    processor,
    output_dir: str | Path,
    push_to_hub: bool = False,
    hub_repo_id: str | None = None,
) -> tuple[Path, float]:
    """Compress and serialize a quantized HF Parakeet model + processor in safetensors format.

    Uses compressed-tensors `ModelCompressor` (`from_pretrained_model`, `compress_model`,
    `save_pretrained`, `update_config`) so `AutoModelForTDT.from_pretrained(output_dir)` loads it directly.
    Returns `(output_path, size_mb_quant)`.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    compressor = ModelCompressor.from_pretrained_model(model)
    compressor.compress_model(model)

    model.save_pretrained(str(out), safe_serialization=True)
    compressor.update_config(str(out))
    if processor is not None:
        processor.save_pretrained(str(out))

    safetensors_files = list(out.glob("*.safetensors"))
    total_bytes = sum(f.stat().st_size for f in safetensors_files)
    if total_bytes == 0:
        bin_files = list(out.glob("*.bin"))
        total_bytes = sum(f.stat().st_size for f in bin_files)
    size_mb = total_bytes / (1024.0 * 1024.0)

    if push_to_hub:
        if not hub_repo_id:
            raise ValueError("--hub_repo_id is required when --push_to_hub=True.")
        api = HfApi()
        api.upload_folder(
            folder_path=str(out),
            repo_id=hub_repo_id,
            repo_type="model",
        )

    return out, size_mb


def state_dict_size_mb(model: nn.Module) -> float:
    """Compute in-memory parameter/buffer tensor footprint in MiB."""
    seen: set[int] = set()
    total_bytes = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        ptr = tensor.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024.0 * 1024.0)
