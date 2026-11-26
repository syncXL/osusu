from __future__ import annotations

import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch
import yaml
from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import Unigram
from transformers import (
    AutoModelForTDT,
    AutoProcessor,
    ParakeetFeatureExtractor,
    ParakeetForTDT,
    ParakeetProcessor,
    ParakeetTDTConfig,
    ParakeetTokenizer,
)


def build_parakeet_processor_from_spm(
    spm_model_path: str | Path,
    *,
    feature_size: int = 80,
    sampling_rate: int = 16000,
    hop_length: int = 160,
    win_length: int = 400,
    n_fft: int = 512,
    preemphasis: float = 0.97,
    blank_token: str = "<blank>",
) -> tuple[ParakeetProcessor, int]:
    """Build a Hugging Face ParakeetProcessor directly from a SentencePiece `.model` file.

    Returns `(processor, base_vocab_size)` where `blank_token_id = pad_token_id = base_vocab_size`
    and `len(processor.tokenizer) == base_vocab_size + 1`.
    """
    sp = spm.SentencePieceProcessor(model_file=str(spm_model_path))
    base_vocab_size = sp.get_piece_size()
    vocab = [(sp.id_to_piece(i), float(sp.get_score(i))) for i in range(base_vocab_size)]
    unk_id = sp.unk_id()
    unk_token = sp.id_to_piece(unk_id) if 0 <= unk_id < base_vocab_size else "<unk>"

    fast_tok = Tokenizer(Unigram(vocab, unk_id=max(0, unk_id)))
    fast_tok.normalizer = normalizers.Sequence([normalizers.Replace(" ", "▁")])
    fast_tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
    fast_tok.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")

    hf_tokenizer = ParakeetTokenizer(tokenizer_object=fast_tok, unk_token=unk_token)
    hf_tokenizer.add_special_tokens(
        {
            "pad_token": blank_token,
            "additional_special_tokens": [blank_token],
        }
    )

    feature_extractor = ParakeetFeatureExtractor(
        feature_size=feature_size,
        sampling_rate=sampling_rate,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        preemphasis=preemphasis,
    )
    processor = ParakeetProcessor(
        feature_extractor=feature_extractor,
        tokenizer=hf_tokenizer,
        blank_token=blank_token,
        decoder_type="tdt",
    )
    return processor, base_vocab_size


def build_parakeet_tdt_config_from_nemo(
    nemo_cfg: dict[str, Any],
    base_vocab_size: int,
) -> ParakeetTDTConfig:
    """Build a ParakeetTDTConfig matching a NeMo FastConformer-TDT config."""
    enc_cfg = nemo_cfg.get("encoder", {})
    dec_cfg = nemo_cfg.get("decoder", {})
    prednet_cfg = dec_cfg.get("prednet", {})
    joint_cfg = nemo_cfg.get("joint", {})
    jointnet_cfg = joint_cfg.get("jointnet", {})
    pre_cfg = nemo_cfg.get("preprocessor", {})
    defaults_cfg = nemo_cfg.get("model_defaults", {})

    d_model = int(enc_cfg.get("d_model", 1024))
    ff_mult = int(enc_cfg.get("ff_expansion_factor", 4))
    durations = list(defaults_cfg.get("tdt_durations", [0, 1, 2, 3, 4]))

    encoder_config = {
        "hidden_size": d_model,
        "num_hidden_layers": int(enc_cfg.get("n_layers", 24)),
        "num_attention_heads": int(enc_cfg.get("n_heads", 8)),
        "intermediate_size": d_model * ff_mult,
        "hidden_act": "silu",
        "attention_bias": bool(enc_cfg.get("use_bias", True)),
        "convolution_bias": bool(enc_cfg.get("use_bias", True)),
        "conv_kernel_size": int(enc_cfg.get("conv_kernel_size", 9)),
        "subsampling_factor": int(enc_cfg.get("subsampling_factor", 8)),
        "subsampling_conv_channels": int(enc_cfg.get("subsampling_conv_channels", 256)),
        "num_mel_bins": int(pre_cfg.get("features", 80)),
        "max_position_embeddings": int(enc_cfg.get("pos_emb_max_len", 5000)),
        "scale_input": bool(enc_cfg.get("xscaling", True)),
    }

    # In HF ParakeetTDTConfig, vocab_size includes the appended <blank> token at index base_vocab_size,
    # and joint.head outputs vocab_size + len(durations) logits.
    return ParakeetTDTConfig(
        vocab_size=base_vocab_size + 1,
        blank_token_id=base_vocab_size,
        pad_token_id=base_vocab_size,
        decoder_start_token_id=base_vocab_size,
        decoder_hidden_size=int(prednet_cfg.get("pred_hidden", 640)),
        num_decoder_layers=int(prednet_cfg.get("pred_rnn_layers", 2)),
        hidden_act=str(jointnet_cfg.get("activation", "relu")),
        durations=durations,
        encoder_config=encoder_config,
    )


def convert_nemo_state_dict_to_hf(nemo_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map NeMo EncDecRNNTBPEModel (FastConformer-TDT) state_dict keys to HF ParakeetForTDT keys."""
    hf_sd: dict[str, torch.Tensor] = {}

    joint_net_indices: list[int] = []
    for k in nemo_state_dict:
        if k.startswith("joint.joint_net.") and k.endswith(".weight"):
            parts = k.split(".")
            if len(parts) == 4 and parts[2].isdigit():
                joint_net_indices.append(int(parts[2]))
    joint_head_idx = max(joint_net_indices) if joint_net_indices else 1
    joint_head_prefix = f"joint.joint_net.{joint_head_idx}."

    attn_renames = (
        (".self_attn.linear_q.", ".self_attn.q_proj."),
        (".self_attn.linear_k.", ".self_attn.k_proj."),
        (".self_attn.linear_v.", ".self_attn.v_proj."),
        (".self_attn.linear_out.", ".self_attn.o_proj."),
        (".self_attn.linear_pos.", ".self_attn.relative_k_proj."),
        (".self_attn.pos_bias_u", ".self_attn.bias_u"),
        (".self_attn.pos_bias_v", ".self_attn.bias_v"),
        (".conv.batch_norm.", ".conv.norm."),
    )

    for key, tensor in nemo_state_dict.items():
        if key.startswith("preprocessor.") or key.startswith("spec_augmentation."):
            continue
        if key.startswith("encoder.pre_encode.conv."):
            suffix = key[len("encoder.pre_encode.conv.") :]
            hf_sd[f"encoder.subsampling.layers.{suffix}"] = tensor
        elif key.startswith("encoder.pre_encode.out."):
            suffix = key[len("encoder.pre_encode.out.") :]
            hf_sd[f"encoder.subsampling.linear.{suffix}"] = tensor
        elif key.startswith("encoder.layers."):
            mapped = key
            for src, dst in attn_renames:
                if src in mapped:
                    mapped = mapped.replace(src, dst, 1)
                    break
            hf_sd[mapped] = tensor
        elif key == "decoder.prediction.embed.weight":
            hf_sd["decoder.embedding.weight"] = tensor
        elif key.startswith("decoder.prediction.dec_rnn.lstm."):
            suffix = key[len("decoder.prediction.dec_rnn.lstm.") :]
            hf_sd[f"decoder.lstm.{suffix}"] = tensor
        elif key.startswith("joint.enc."):
            suffix = key[len("joint.enc.") :]
            hf_sd[f"encoder_projector.{suffix}"] = tensor
        elif key.startswith("joint.pred."):
            suffix = key[len("joint.pred.") :]
            hf_sd[f"decoder.decoder_projector.{suffix}"] = tensor
        elif key.startswith(joint_head_prefix):
            suffix = key[len(joint_head_prefix) :]
            hf_sd[f"joint.head.{suffix}"] = tensor

    return hf_sd


def convert_nemo_to_hf(
    nemo_path: str | Path,
    dtype: torch.dtype = torch.float32,
) -> tuple[ParakeetForTDT, ParakeetProcessor]:
    """Convert a local `.nemo` FastConformer-TDT archive into HF `(ParakeetForTDT, ParakeetProcessor)`."""
    nemo_path = Path(nemo_path)
    if not nemo_path.is_file():
        raise FileNotFoundError(f"NeMo checkpoint not found: {nemo_path}")

    with tarfile.open(nemo_path, "r:*") as tar:
        members = {os.path.basename(m.name): m for m in tar.getmembers() if m.isfile()}
        if "model_config.yaml" not in members or "model_weights.ckpt" not in members:
            raise ValueError(
                f"Invalid .nemo archive '{nemo_path}': expected model_config.yaml and model_weights.ckpt."
            )

        cfg_bytes = tar.extractfile(members["model_config.yaml"]).read()
        nemo_cfg = yaml.safe_load(cfg_bytes)

        tok_cfg = nemo_cfg.get("tokenizer", {})
        tok_model_ref = str(tok_cfg.get("model", "")).removeprefix("nemo:")
        tok_basename = os.path.basename(tok_model_ref) if tok_model_ref else ""

        spm_member = members.get(tok_basename)
        if spm_member is None:
            spm_candidates = [m for name, m in members.items() if name.endswith(".model")]
            if not spm_candidates:
                raise ValueError(f"No SentencePiece `.model` file found inside '{nemo_path}'.")
            spm_member = spm_candidates[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            spm_path = tmp / "tokenizer.model"
            spm_path.write_bytes(tar.extractfile(spm_member).read())

            ckpt_path = tmp / "model_weights.ckpt"
            ckpt_path.write_bytes(tar.extractfile(members["model_weights.ckpt"]).read())

            pre_cfg = nemo_cfg.get("preprocessor", {})
            sr = int(pre_cfg.get("sample_rate", 16000))
            processor, base_vocab_size = build_parakeet_processor_from_spm(
                spm_path,
                feature_size=int(pre_cfg.get("features", 80)),
                sampling_rate=sr,
                hop_length=int(round(float(pre_cfg.get("window_stride", 0.01)) * sr)),
                win_length=int(round(float(pre_cfg.get("window_size", 0.025)) * sr)),
                n_fft=int(pre_cfg.get("n_fft", 512)),
                preemphasis=float(pre_cfg.get("preemph", 0.97)),
            )

            nemo_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if "state_dict" in nemo_sd and isinstance(nemo_sd["state_dict"], dict):
                nemo_sd = nemo_sd["state_dict"]

    hf_config = build_parakeet_tdt_config_from_nemo(nemo_cfg, base_vocab_size=base_vocab_size)
    hf_model = ParakeetForTDT(hf_config)
    hf_sd = convert_nemo_state_dict_to_hf(nemo_sd)
    missing, unexpected = hf_model.load_state_dict(hf_sd, strict=False)

    # Allow only non-persistent buffers (e.g. inv_freq) to be absent; fail on missing weights.
    persistent_missing = [k for k in missing if not k.endswith("encode_positions.inv_freq")]
    if persistent_missing:
        raise RuntimeError(f"Missing keys when converting .nemo to ParakeetForTDT: {persistent_missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys when converting .nemo to ParakeetForTDT: {unexpected}")

    hf_model.to(dtype=dtype)
    hf_model.eval()
    return hf_model, processor


def load_parakeet_model_and_processor(
    model_path: str,
    dtype: torch.dtype = torch.float32,
) -> tuple[ParakeetForTDT, ParakeetProcessor]:
    """Load `(ParakeetForTDT, AutoProcessor)` from either a `.nemo` file or a HF repo/directory."""
    path = Path(model_path)
    if path.suffix == ".nemo" or (path.is_file() and tarfile.is_tarfile(path)):
        return convert_nemo_to_hf(path, dtype=dtype)

    model = AutoModelForTDT.from_pretrained(model_path, dtype=dtype)
    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()
    return model, processor
