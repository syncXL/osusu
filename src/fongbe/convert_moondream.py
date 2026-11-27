from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file as load_safetensors


_ATTN_HF_TO_NEMO = (
    (".self_attn.q_proj.", ".self_attn.linear_q."),
    (".self_attn.k_proj.", ".self_attn.linear_k."),
    (".self_attn.v_proj.", ".self_attn.linear_v."),
    (".self_attn.o_proj.", ".self_attn.linear_out."),
    (".self_attn.relative_k_proj.", ".self_attn.linear_pos."),
    (".self_attn.bias_u", ".self_attn.pos_bias_u"),
    (".self_attn.bias_v", ".self_attn.pos_bias_v"),
    (".conv.norm.", ".conv.batch_norm."),
)


def _detect_joint_head_index(target_state_dict: Mapping[str, torch.Tensor]) -> int:
    indices: list[int] = []
    for key in target_state_dict:
        if key.startswith("joint.joint_net.") and key.endswith(".weight"):
            parts = key.split(".")
            if len(parts) == 4 and parts[2].isdigit():
                indices.append(int(parts[2]))
    return max(indices) if indices else 2


def map_hf_key_to_nemo(hf_key: str, *, joint_head_idx: int = 2) -> str | None:
    """Map a single HF `ParakeetForTDT` state dict key to its NeMo `EncDecRNNTBPEModel` key.

    Returns ``None`` if the key does not match any supported mapping rule.
    Moondream's auxiliary ``vad_head.*`` keys should be filtered by the caller before calling this.
    """
    if hf_key.startswith("encoder.subsampling.layers."):
        suffix = hf_key[len("encoder.subsampling.layers.") :]
        return f"encoder.pre_encode.conv.{suffix}"
    if hf_key.startswith("encoder.subsampling.linear."):
        suffix = hf_key[len("encoder.subsampling.linear.") :]
        return f"encoder.pre_encode.out.{suffix}"
    if hf_key.startswith("encoder.layers."):
        mapped = hf_key
        for src, dst in _ATTN_HF_TO_NEMO:
            if src in mapped:
                mapped = mapped.replace(src, dst, 1)
                break
        return mapped
    if hf_key == "decoder.embedding.weight":
        return "decoder.prediction.embed.weight"
    if hf_key.startswith("decoder.lstm."):
        suffix = hf_key[len("decoder.lstm.") :]
        return f"decoder.prediction.dec_rnn.lstm.{suffix}"
    if hf_key.startswith("encoder_projector."):
        suffix = hf_key[len("encoder_projector.") :]
        return f"joint.enc.{suffix}"
    if hf_key.startswith("decoder.decoder_projector."):
        suffix = hf_key[len("decoder.decoder_projector.") :]
        return f"joint.pred.{suffix}"
    if hf_key.startswith("joint.head."):
        suffix = hf_key[len("joint.head.") :]
        return f"joint.joint_net.{joint_head_idx}.{suffix}"
    return None


def map_hf_to_nemo_state_dict(
    hf_state_dict: Mapping[str, torch.Tensor],
    target_state_dict: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, list[str]]]:
    """Map HF `ParakeetForTDT` tensors to a NeMo `EncDecRNNTBPEModel` state dict.

    Explicitly drops ONLY `vad_head.*` keys from the HF checkpoint, preserves non-weight
    `preprocessor.*` buffers (e.g. `featurizer.fb`, `featurizer.window`) from
    `target_state_dict`, casts weights to match target tensor dtypes, and verifies
    there are 0 missing keys and 0 unmapped non-VAD keys.
    """
    joint_head_idx = _detect_joint_head_index(target_state_dict)
    mapped: dict[str, torch.Tensor] = {}
    dropped_vad_keys: list[str] = []
    unmapped_non_vad_keys: list[str] = []

    for hf_key, tensor in hf_state_dict.items():
        if hf_key.startswith("vad_head."):
            dropped_vad_keys.append(hf_key)
            continue

        nemo_key = map_hf_key_to_nemo(hf_key, joint_head_idx=joint_head_idx)
        if nemo_key is None or nemo_key not in target_state_dict:
            unmapped_non_vad_keys.append(hf_key)
            continue

        target_tensor = target_state_dict[nemo_key]
        if tuple(tensor.shape) != tuple(target_tensor.shape):
            raise ValueError(
                f"Shape mismatch for '{hf_key}' -> '{nemo_key}': "
                f"HF {tuple(tensor.shape)} vs NeMo {tuple(target_tensor.shape)}"
            )
        mapped[nemo_key] = tensor.detach().to(device="cpu", dtype=target_tensor.dtype).contiguous()

    # Copy preprocessor spectral filterbank/window buffers from the NeMo skeleton
    # (HF ParakeetForTDT relies on an external ParakeetFeatureExtractor instead).
    preserved_preprocessor_keys: list[str] = []
    for nemo_key, target_tensor in target_state_dict.items():
        if nemo_key.startswith("preprocessor.") and nemo_key not in mapped:
            mapped[nemo_key] = target_tensor.detach().clone()
            preserved_preprocessor_keys.append(nemo_key)

    missing_keys = sorted(k for k in target_state_dict if k not in mapped)
    assert len(unmapped_non_vad_keys) == 0, (
        f"Unmapped non-VAD HF keys found ({len(unmapped_non_vad_keys)}): {unmapped_non_vad_keys[:10]}"
    )
    assert len(missing_keys) == 0, (
        f"Missing NeMo target keys after mapping ({len(missing_keys)}): {missing_keys[:10]}"
    )

    stats = {
        "dropped_vad_keys": sorted(dropped_vad_keys),
        "unmapped_non_vad_keys": unmapped_non_vad_keys,
        "missing_keys": missing_keys,
        "preserved_preprocessor_keys": sorted(preserved_preprocessor_keys),
    }
    return mapped, stats


def convert_moondream_to_nemo(
    hf_repo: str = "moondream/parakeet-ultra",
    base_nemo_model: str = "nvidia/parakeet-tdt-0.6b-v3",
    output_path: str = "/tmp/parakeet-ultra.nemo",
    push_to_hub_repo: str | None = None,
    hub_path_in_repo: str = "checkpoints/parakeet-ultra.nemo",
) -> str:
    """Convert Hugging Face ``moondream/parakeet-ultra`` weights into a NeMo ``.nemo`` archive.

    Uses ``base_nemo_model`` as the NeMo skeleton (same 0.6B FastConformer-TDT architecture
    and 8,192 vocab), maps all HF weights onto it while dropping only ``vad_head.*``,
    validates ``load_state_dict(..., strict=True)``, and saves to ``output_path``.
    """
    from nemo.collections.asr.models import ASRModel

    if os.path.exists(base_nemo_model) and base_nemo_model.endswith(".nemo"):
        asr_model = ASRModel.restore_from(restore_path=base_nemo_model, map_location="cpu")
    else:
        asr_model = ASRModel.from_pretrained(model_name=base_nemo_model, map_location="cpu")

    safetensors_candidate = Path(hf_repo) / "model.safetensors"
    if os.path.isfile(hf_repo):
        safetensors_path = hf_repo
    elif safetensors_candidate.is_file():
        safetensors_path = str(safetensors_candidate)
    else:
        safetensors_path = hf_hub_download(repo_id=hf_repo, filename="model.safetensors")

    hf_state_dict = load_safetensors(safetensors_path, device="cpu")
    target_state_dict = asr_model.state_dict()
    mapped_sd, stats = map_hf_to_nemo_state_dict(hf_state_dict, target_state_dict)

    incompatible = asr_model.load_state_dict(mapped_sd, strict=True)
    assert len(incompatible.missing_keys) == 0, f"Unexpected missing keys: {incompatible.missing_keys}"
    assert len(incompatible.unexpected_keys) == 0, f"Unexpected extra keys: {incompatible.unexpected_keys}"

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    asr_model.save_to(str(out))

    print(
        f"[convert_moondream] Converted {len(mapped_sd)} tensors to '{out}' "
        f"(dropped {len(stats['dropped_vad_keys'])} vad_head keys, "
        f"preserved {len(stats['preserved_preprocessor_keys'])} preprocessor buffers)."
    )

    if push_to_hub_repo:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(out),
            path_in_repo=hub_path_in_repo,
            repo_id=push_to_hub_repo,
            repo_type="model",
        )
        print(f"[convert_moondream] Uploaded '{out}' -> '{push_to_hub_repo}/{hub_path_in_repo}'.")

    return str(out)
