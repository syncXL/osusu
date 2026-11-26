from __future__ import annotations

import csv
import json
import tarfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml
from compressed_tensors.quantization import QuantizationStatus
from compressed_tensors.quantization.lifecycle.forward import fake_quantize
from compressed_tensors.quantization.utils import is_module_quantized
from torch import nn
from transformers import AutoModelForTDT, AutoProcessor, ParakeetForTDT, ParakeetTDTConfig

from fongbe.quantization.algorithms import (
    apply_gptq,
    apply_quantization_algorithm,
    apply_smoothquant_to_pair,
    compute_weight_qparams,
)
from fongbe.quantization.calibrator import LayerActivationStats, collect_activation_stats
from fongbe.quantization.cli import quantize
from fongbe.quantization.converter import build_parakeet_processor_from_spm, convert_nemo_to_hf
from fongbe.quantization.evaluate import (
    build_evaluation_report,
    compute_wer_and_cer,
    print_qat_decision_banner,
    write_tsv_reports,
)
from fongbe.quantization.exporter import save_quantized_hf_model
from fongbe.quantization.schemes import normalize_ignore_modules, validate_quant_args

SPM_V512 = Path(__file__).resolve().parents[1] / "tokenizers" / "tokenizer_spe_unigram_v512" / "tokenizer.model"


def _make_tiny_parakeet(vocab_size: int = 32) -> tuple[ParakeetForTDT, object]:
    cfg = ParakeetTDTConfig(
        vocab_size=vocab_size + 1,
        blank_token_id=vocab_size,
        pad_token_id=vocab_size,
        decoder_hidden_size=16,
        num_decoder_layers=1,
        durations=[0, 1, 2, 3, 4],
        encoder_config={
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 64,
            "subsampling_conv_channels": 16,
            "num_mel_bins": 16,
            "conv_kernel_size": 3,
        },
    )
    model = ParakeetForTDT(cfg)
    model.eval()
    return model, cfg


def _make_dummy_calib_batches(n_batches: int = 2, batch_size: int = 2, seq_len: int = 32, n_mels: int = 16):
    torch.manual_seed(0)
    batches = []
    for _ in range(n_batches):
        batches.append(
            {
                "input_features": torch.randn(batch_size, seq_len, n_mels),
                "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
            }
        )
    return batches


def test_validate_quant_args_rejects_invalid_combinations():
    # Valid specs
    s1 = validate_quant_args(algorithm="absmax", bits=8, scheme="symmetric", granularity="per_channel")
    assert s1.requires_calibration is False
    s2 = validate_quant_args(algorithm="smoothquant", bits=8, static_ptq=True, granularity="per_channel")
    assert s2.requires_calibration is True
    s3 = validate_quant_args(algorithm="awq", bits=4, granularity="per_block", block_size=128)
    assert s3.bits == 4
    s4 = validate_quant_args(algorithm="gptq", bits=3, granularity="per_channel")
    assert s4.bits == 3

    # Invalid combos
    with pytest.raises(ValueError, match="smoothquant.*bits=8"):
        validate_quant_args(algorithm="smoothquant", bits=4, static_ptq=True)
    with pytest.raises(ValueError, match="smoothquant.*static_ptq=True"):
        validate_quant_args(algorithm="smoothquant", bits=8, static_ptq=False)
    with pytest.raises(ValueError, match="smoothquant.*per_block"):
        validate_quant_args(algorithm="smoothquant", bits=8, static_ptq=True, granularity="per_block")
    with pytest.raises(ValueError, match="low-bit weights"):
        validate_quant_args(algorithm="awq", bits=8, granularity="per_channel")
    with pytest.raises(ValueError, match="low-bit weights"):
        validate_quant_args(algorithm="gptq", bits=8, granularity="per_channel")
    with pytest.raises(ValueError, match="per_tensor"):
        validate_quant_args(algorithm="awq", bits=4, granularity="per_tensor")
    with pytest.raises(ValueError, match="per_tensor"):
        validate_quant_args(algorithm="gptq", bits=3, granularity="per_tensor")
    with pytest.raises(ValueError, match="static_ptq=True"):
        validate_quant_args(algorithm="absmax", bits=4, static_ptq=True)
    with pytest.raises(ValueError, match="calib_samples"):
        validate_quant_args(algorithm="awq", bits=4, granularity="per_channel", calib_samples=0)
    with pytest.raises(ValueError, match="scheme"):
        validate_quant_args(scheme="fp8")


def test_ignore_modules_normalization():
    patterns = normalize_ignore_modules("pre_encode, decoder, joint, encoder_projector")
    assert r"re:^encoder\.subsampling\..*" in patterns
    assert r"re:^decoder\..*" in patterns
    assert r"re:^joint\..*" in patterns
    assert "encoder_projector" in patterns


def test_schemes_and_granularities_weight_math():
    torch.manual_seed(42)
    layer = nn.Linear(32, 16, bias=False)
    for scheme in ("symmetric", "affine"):
        for bits in (8, 4, 3):
            for gran in ("per_tensor", "per_channel", "per_block"):
                spec = validate_quant_args(
                    scheme=scheme,
                    bits=bits,
                    granularity=gran,
                    block_size=8,
                    algorithm="absmax",
                )
                model = nn.Sequential(nn.Linear(32, 16, bias=False))
                model[0].weight.data.copy_(layer.weight.data)
                _, diags = apply_quantization_algorithm(model, spec)
                assert len(diags) == 1
                assert diags[0].weight_mse >= 0.0
                assert diags[0].weight_sqnr_db > 10.0
                if gran == "per_tensor":
                    assert model[0].weight_scale.shape == (1,)
                elif gran == "per_channel":
                    assert model[0].weight_scale.shape == (16, 1)
                else:
                    assert model[0].weight_scale.shape == (16, 4)


def test_calibrator_absmax_vs_percentile_observer():
    model, _ = _make_tiny_parakeet()
    # Inject a deterministic large activation outlier into one batch
    batches = _make_dummy_calib_batches(n_batches=2, batch_size=2, seq_len=16, n_mels=16)
    batches[0]["input_features"][0, 0, :] = 100.0

    stats = collect_activation_stats(
        model=model,
        spec=validate_quant_args(algorithm="absmax", bits=8, static_ptq=True, calib_method="percentile"),
        calib_batches=batches,
        target_modules={"encoder.layers.0.feed_forward1.linear1": model.encoder.layers[0].feed_forward1.linear1},
    )
    st = stats["encoder.layers.0.feed_forward1.linear1"]
    assert st.num_tokens > 0

    low_abs, high_abs = st.activation_bounds(symmetric=True, calib_method="absmax")
    low_pct, high_pct = st.activation_bounds(symmetric=True, calib_method="percentile", calib_percentile=99.0)
    assert high_abs > high_pct > 0.0


def test_smoothquant_preserves_output_and_roundtrips(tmp_path):
    torch.manual_seed(7)
    norm = nn.LayerNorm(32)
    lin = nn.Linear(32, 16)
    x = torch.randn(4, 32)
    # Give channel 0 a 50x outlier
    x[:, 0] *= 50.0
    with torch.no_grad():
        ref_out = lin(norm(x))

    act_absmax = norm(x).abs().amax(dim=0)
    apply_smoothquant_to_pair([("lin", lin)], act_absmax=act_absmax, prev_module=norm, alpha=0.5)
    with torch.no_grad():
        smoothed_out = lin(norm(x))
    assert torch.allclose(ref_out, smoothed_out, atol=1e-4)

    # Full Parakeet W8A8 SmoothQuant + save + AutoModelForTDT reload
    model, _ = _make_tiny_parakeet()
    batches = _make_dummy_calib_batches()
    spec = validate_quant_args(
        scheme="symmetric",
        bits=8,
        granularity="per_channel",
        algorithm="smoothquant",
        smooth_alpha=0.5,
        static_ptq=True,
        calib_samples=4,
        calib_method="percentile",
        calib_percentile=99.9,
    )
    _, diags = apply_quantization_algorithm(model, spec, calib_batches=batches)
    assert len(diags) == 18  # 2 encoder layers * 9 linears; pre_encode/decoder/joint ignored
    assert not is_module_quantized(model.decoder.decoder_projector)
    assert not is_module_quantized(model.joint.head)
    assert is_module_quantized(model.encoder.layers[0].feed_forward1.linear1)
    assert model.encoder.layers[0].feed_forward1.linear1.quantization_status == QuantizationStatus.FROZEN

    out_dir = tmp_path / "w8a8_smoothquant"
    save_quantized_hf_model(model, processor=None, output_dir=out_dir)
    assert (out_dir / "model.safetensors").exists()
    cfg_json = json.loads((out_dir / "config.json").read_text())
    assert "quantization_config" in cfg_json

    reloaded = AutoModelForTDT.from_pretrained(out_dir)
    with torch.no_grad():
        out = reloaded(
            input_features=batches[0]["input_features"],
            attention_mask=batches[0]["attention_mask"],
            decoder_input_ids=torch.tensor([[32], [32]]),
        )
    assert out.logits.shape[-1] == 38


def test_awq_and_gptq_algorithms_and_roundtrip(tmp_path):
    torch.manual_seed(11)
    # 1. Direct GPTQ error reduction vs naive absmax at 3 bits
    x = torch.randn(128, 32)
    x[:, :4] *= 15.0  # salient input channels
    hessian = (x.T @ x) / x.shape[0]
    w_init = torch.randn(16, 32)

    lin_naive = nn.Linear(32, 16, bias=False)
    lin_gptq = nn.Linear(32, 16, bias=False)
    lin_naive.weight.data.copy_(w_init)
    lin_gptq.weight.data.copy_(w_init)

    spec_gptq = validate_quant_args(algorithm="gptq", bits=3, granularity="per_block", block_size=8)
    from fongbe.quantization.schemes import build_weight_quant_args

    w_args = build_weight_quant_args(spec_gptq, [lin_gptq])
    s_naive, zp_naive = compute_weight_qparams(lin_naive.weight, w_args)
    wq_naive = fake_quantize(lin_naive.weight, s_naive, zp_naive, w_args)

    apply_gptq(lin_gptq, hessian, w_args)
    wq_gptq = lin_gptq.weight.data

    with torch.no_grad():
        ref_y = x @ w_init.T
        err_naive = torch.mean((ref_y - x @ wq_naive.T).pow(2)).item()
        err_gptq = torch.mean((ref_y - x @ wq_gptq.T).pow(2)).item()
    assert err_gptq < err_naive

    # 2. Full Parakeet AWQ (W4 per_block) -> save -> AutoModelForTDT reload
    model, _ = _make_tiny_parakeet()
    batches = _make_dummy_calib_batches()
    spec_awq = validate_quant_args(
        scheme="affine",
        bits=4,
        granularity="per_block",
        block_size=8,
        algorithm="awq",
        calib_samples=4,
    )
    apply_quantization_algorithm(model, spec_awq, calib_batches=batches)
    out_dir = tmp_path / "w4_awq"
    save_quantized_hf_model(model, processor=None, output_dir=out_dir)
    reloaded = AutoModelForTDT.from_pretrained(out_dir)
    with torch.no_grad():
        out = reloaded(
            input_features=batches[0]["input_features"],
            attention_mask=batches[0]["attention_mask"],
            decoder_input_ids=torch.tensor([[32], [32]]),
        )
    assert out.logits.shape[-1] == 38


def test_nemo_conversion_custom_512_vocab(tmp_path):
    assert SPM_V512.is_file(), f"Missing tokenizer file: {SPM_V512}"
    # Build tiny HF model with 512 SP vocab (+1 blank = 513), convert its weights into NeMo naming,
    # pack into a .nemo tarfile, and verify convert_nemo_to_hf recovers exact config/vocab/logits.
    proc_ref, base_vocab = build_parakeet_processor_from_spm(SPM_V512)
    assert base_vocab == 512
    assert proc_ref.blank_token_id == 512
    assert proc_ref.tokenizer.pad_token_id == 512

    hf_ref, _ = _make_tiny_parakeet(vocab_size=base_vocab)
    hf_sd = hf_ref.state_dict()
    nemo_sd = {}
    for k, v in hf_sd.items():
        if k.startswith("encoder.subsampling.layers."):
            nemo_sd["encoder.pre_encode.conv." + k[len("encoder.subsampling.layers.") :]] = v
        elif k.startswith("encoder.subsampling.linear."):
            nemo_sd["encoder.pre_encode.out." + k[len("encoder.subsampling.linear.") :]] = v
        elif k.startswith("encoder.layers."):
            nk = (
                k.replace(".self_attn.q_proj.", ".self_attn.linear_q.")
                .replace(".self_attn.k_proj.", ".self_attn.linear_k.")
                .replace(".self_attn.v_proj.", ".self_attn.linear_v.")
                .replace(".self_attn.o_proj.", ".self_attn.linear_out.")
                .replace(".self_attn.relative_k_proj.", ".self_attn.linear_pos.")
                .replace(".self_attn.bias_u", ".self_attn.pos_bias_u")
                .replace(".self_attn.bias_v", ".self_attn.pos_bias_v")
                .replace(".conv.norm.", ".conv.batch_norm.")
            )
            nemo_sd[nk] = v
        elif k == "decoder.embedding.weight":
            nemo_sd["decoder.prediction.embed.weight"] = v
        elif k.startswith("decoder.lstm."):
            nemo_sd["decoder.prediction.dec_rnn.lstm." + k[len("decoder.lstm.") :]] = v
        elif k.startswith("encoder_projector."):
            nemo_sd["joint.enc." + k[len("encoder_projector.") :]] = v
        elif k.startswith("decoder.decoder_projector."):
            nemo_sd["joint.pred." + k[len("decoder.decoder_projector.") :]] = v
        elif k.startswith("joint.head."):
            nemo_sd["joint.joint_net.1." + k[len("joint.head.") :]] = v

    nemo_cfg = {
        "model_defaults": {"tdt_durations": [0, 1, 2, 3, 4]},
        "preprocessor": {"sample_rate": 16000, "features": 16, "window_stride": 0.01, "window_size": 0.025, "n_fft": 512, "preemph": 0.97},
        "encoder": {
            "d_model": 32,
            "n_layers": 2,
            "n_heads": 2,
            "ff_expansion_factor": 2,
            "conv_kernel_size": 3,
            "subsampling_factor": 8,
            "subsampling_conv_channels": 16,
            "pos_emb_max_len": 5000,
            "xscaling": True,
            "use_bias": True,
        },
        "decoder": {"prednet": {"pred_hidden": 16, "pred_rnn_layers": 1}},
        "joint": {"jointnet": {"activation": "relu"}},
        "tokenizer": {"model": "nemo:tokenizer.model"},
    }

    cfg_file = tmp_path / "model_config.yaml"
    cfg_file.write_text(yaml.safe_dump(nemo_cfg))
    ckpt_file = tmp_path / "model_weights.ckpt"
    torch.save(nemo_sd, ckpt_file)
    nemo_archive = tmp_path / "tiny_fongbe.nemo"
    with tarfile.open(nemo_archive, "w") as tar:
        tar.add(cfg_file, arcname="model_config.yaml")
        tar.add(ckpt_file, arcname="model_weights.ckpt")
        tar.add(SPM_V512, arcname="tokenizer.model")

    conv_model, conv_proc = convert_nemo_to_hf(nemo_archive)
    assert conv_model.config.vocab_size == 513
    assert conv_model.config.blank_token_id == 512
    assert conv_model.config.pad_token_id == 512
    assert conv_model.joint.head.out_features == 518
    assert conv_proc.blank_token_id == 512

    x = torch.randn(1, 32, 16)
    mask = torch.ones(1, 32, dtype=torch.bool)
    dec_in = torch.tensor([[512]])
    with torch.no_grad():
        ref_logits = hf_ref(input_features=x, attention_mask=mask, decoder_input_ids=dec_in).logits
        conv_logits = conv_model(input_features=x, attention_mask=mask, decoder_input_ids=dec_in).logits
    assert torch.allclose(ref_logits, conv_logits, atol=1e-5)


def test_evaluation_tsv_report_and_cli_end_to_end(tmp_path, capsys):
    wer, cer = compute_wer_and_cer([("mǐ ná yi", "mǐ na yi")])
    assert wer == pytest.approx(1 / 3)
    assert cer > 0.0

    spec = validate_quant_args(algorithm="absmax", bits=8, granularity="per_channel")
    rep_pass = build_evaluation_report(spec, 10, wer_fp32=0.10, wer_quant=0.105, cer_fp32=0.05, cer_quant=0.052, size_mb_fp32=100.0, size_mb_quant=25.0, wer_threshold=0.01)
    assert rep_pass.needs_qat is False
    rep_boundary = build_evaluation_report(spec, 10, wer_fp32=0.10, wer_quant=0.11, cer_fp32=0.05, cer_quant=0.055, size_mb_fp32=100.0, size_mb_quant=25.0, wer_threshold=0.01)
    assert rep_boundary.needs_qat is False
    rep_qat = build_evaluation_report(spec, 10, wer_fp32=0.10, wer_quant=0.12, cer_fp32=0.05, cer_quant=0.06, size_mb_fp32=100.0, size_mb_quant=25.0, wer_threshold=0.01)
    assert rep_qat.needs_qat is True

    # Prepare local HF model + processor directory for end-to-end CLI execution
    proc, base_vocab = build_parakeet_processor_from_spm(SPM_V512)
    # Match tiny model mel bins = 80 so ParakeetProcessor defaults work cleanly
    cfg = ParakeetTDTConfig(
        vocab_size=base_vocab + 1,
        blank_token_id=base_vocab,
        pad_token_id=base_vocab,
        decoder_start_token_id=base_vocab,
        decoder_hidden_size=16,
        num_decoder_layers=1,
        durations=[0, 1, 2, 3, 4],
        encoder_config={
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "intermediate_size": 64,
            "subsampling_conv_channels": 16,
            "num_mel_bins": 80,
            "conv_kernel_size": 3,
        },
    )
    src_model = ParakeetForTDT(cfg)
    src_dir = tmp_path / "fp_hf_model"
    src_model.save_pretrained(src_dir)
    proc.save_pretrained(src_dir)

    fake_eval_samples = [
        {"array": np.zeros(3200, dtype=np.float32), "sampling_rate": 16000, "text": "mǐ ná yi"},
        {"array": np.ones(3200, dtype=np.float32) * 0.01, "sampling_rate": 16000, "text": "un nɔ ɖo"},
    ]
    out_dir = tmp_path / "quant_out"
    tsv_path = tmp_path / "report.tsv"

    with patch("fongbe.quantization.cli.iter_hf_audio_samples", return_value=fake_eval_samples):
        result = quantize(
            model_path=str(src_dir),
            output_dir=str(out_dir),
            scheme="symmetric",
            bits=8,
            granularity="per_channel",
            algorithm="absmax",
            static_ptq=False,
            eval_samples=2,
            wer_threshold=0.01,
            report_tsv=str(tsv_path),
            device="cpu",
        )

    captured = capsys.readouterr().out
    assert "Fongbe Parakeet Quantization Summary" in captured
    assert (out_dir / "model.safetensors").exists()
    assert (out_dir / "config.json").exists()
    assert (out_dir / "tokenizer.json").exists()
    assert (out_dir / "processor_config.json").exists()
    assert tsv_path.exists()
    layers_tsv = Path(result["layers_tsv"])
    assert layers_tsv.exists()

    with tsv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert len(rows) == 1
    expected_cols = {
        "wer_fp32", "wer_quant", "wer_delta",
        "cer_fp32", "cer_quant", "cer_delta",
        "size_mb_fp32", "size_mb_quant", "compression_ratio", "needs_qat",
    }
    assert expected_cols.issubset(rows[0].keys())

    with layers_tsv.open("r", encoding="utf-8") as f:
        layer_rows = list(csv.DictReader(f, delimiter="\t"))
    assert len(layer_rows) == 9
    assert {"layer_name", "weight_mse", "weight_sqnr_db", "max_abs_err"}.issubset(layer_rows[0].keys())

    # Verify AutoModelForTDT and AutoProcessor load directly from output_dir
    loaded_model = AutoModelForTDT.from_pretrained(out_dir)
    loaded_proc = AutoProcessor.from_pretrained(out_dir)
    assert loaded_proc.blank_token_id == 512
    from fongbe.quantization.evaluate import transcribe_samples

    decoded = transcribe_samples(loaded_model, loaded_proc, fake_eval_samples[:1], device="cpu")
    assert len(decoded) == 1
