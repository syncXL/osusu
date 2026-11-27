import io
import math
import os
import tarfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn as nn

# Match CPU-only local test env guard used across fongbe tests
os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "0")

from fongbe.convert_moondream import map_hf_to_nemo_state_dict  # noqa: E402
from fongbe.eval_significance import compare_predictions_bootstrap  # noqa: E402
from fongbe.model import extract_encoder_state_dict, load_encoder_state_dict_from_source  # noqa: E402
from fongbe.recipes.pretrain_fongbe import (  # noqa: E402
    PRETRAIN_CORPORA,
    CorpusSpec,
    VAL_DATASET_CONFIG,
    VAL_DATASET_SPLIT,
    build_pretrain_streams,
    compute_corpus_probabilities,
    decode_audio_sample,
)
from fongbe.ssl import (  # noqa: E402
    EncoderFreezeWarmupCallback,
    FongbeNESTModel,
    SSLAudioDataset,
    build_nest_model_config,
)


def _wav_bytes(waveform: np.ndarray, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, waveform, sample_rate, format="WAV")
    return buf.getvalue()


class _FakeHFStream:
    def __init__(self, items, name=""):
        self.items = list(items)
        self.name = name
        self.column_names = ["audio", "text"] if self.items else ["audio"]
        self.saved_state = {"cursor": 0, "name": name}
        self.epoch = 0

    def remove_columns(self, cols):
        self.column_names = [c for c in self.column_names if c not in cols]
        return self

    def cast_column(self, col, spec):
        return self

    def shuffle(self, seed=42, buffer_size=256):
        return self

    def set_epoch(self, epoch):
        self.epoch = epoch

    def state_dict(self):
        return dict(self.saved_state)

    def load_state_dict(self, state):
        self.saved_state = dict(state)

    def __iter__(self):
        return iter(self.items)


# ---------------------------------------------------------------------------
# 1. OmniASR temperature probability math & split hygiene
# ---------------------------------------------------------------------------


def test_omniasr_temperature_probabilities_and_split_exclusion():
    assert len(PRETRAIN_CORPORA) == 7
    configs = {c.config_name: c for c in PRETRAIN_CORPORA}
    assert set(configs.keys()) == {
        "ffstc_fon",
        "fon_hau_fon",
        "fongbe-speech-zenodo_fon",
        "zindi_fon",
        "waxalnlp_ewe",
        "open-bible-speech-african_ewe",
        "youversion-african-speech_ewe",
    }
    assert configs["zindi_fon"].splits == ("train",)
    assert configs["waxalnlp_ewe"].splits == ("train", "unlabeled")
    assert VAL_DATASET_CONFIG == "zindi_fon"
    assert VAL_DATASET_SPLIT == "validation"

    probs_raw = compute_corpus_probabilities(
        PRETRAIN_CORPORA, beta_corpus=0.5, duration_adjusted=False
    )
    assert math.isclose(sum(probs_raw), 1.0, rel_tol=1e-9, abs_tol=1e-9)
    assert all(p > 0 for p in probs_raw)
    # Without duration adjustment, largest hours (waxalnlp_ewe) gets highest probability
    assert probs_raw[4] == max(probs_raw)

    probs_adj = compute_corpus_probabilities(
        PRETRAIN_CORPORA,
        beta_corpus=0.5,
        duration_adjusted=True,
        crop_duration_sec=20.0,
    )
    assert math.isclose(sum(probs_adj), 1.0, rel_tol=1e-9, abs_tol=1e-9)
    # Check manual formula for zindi_fon (index 3) and fon_hau_fon (index 1, mean_dur=24.19 clipped to 20.0)
    raw_weights = [
        (c.hours ** 0.5) / min(c.mean_duration_sec, 20.0) for c in PRETRAIN_CORPORA
    ]
    expected = [w / sum(raw_weights) for w in raw_weights]
    for actual, exp in zip(probs_adj, expected):
        assert math.isclose(actual, exp, rel_tol=1e-9, abs_tol=1e-9)


def test_build_pretrain_streams_rejects_zindi_val_or_test_and_interleaves():
    bad_corpora = [CorpusSpec("zindi_fon", ("train", "validation"), 338.55, 9.12, "fon")]
    with pytest.raises(ValueError, match="Forbidden split"):
        build_pretrain_streams(corpora=bad_corpora)

    bad_test = [CorpusSpec("zindi_fon", ("test",), 10.0, 5.0, "fon")]
    with pytest.raises(ValueError, match="Forbidden split"):
        build_pretrain_streams(corpora=bad_test)

    calls = []

    def fake_load(dataset_id, config_name, split, streaming=True):
        calls.append((config_name, split))
        return _FakeHFStream(
            [{"audio": {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000}}],
            name=f"{config_name}/{split}",
        )

    with (
        patch("fongbe.recipes.pretrain_fongbe.concatenate_datasets", side_effect=lambda ds: ds[0]),
        patch("fongbe.recipes.pretrain_fongbe.interleave_datasets", side_effect=lambda ds, probabilities, seed, stopping_strategy: (ds, probabilities, stopping_strategy)),
    ):
        train_stream, val_stream, probs = build_pretrain_streams(load_dataset_fn=fake_load)

    assert math.isclose(sum(probs), 1.0, rel_tol=1e-9)
    assert train_stream[2] == "all_exhausted"
    assert ("zindi_fon", "train") in calls
    assert ("waxalnlp_ewe", "train") in calls
    assert ("waxalnlp_ewe", "unlabeled") in calls
    # zindi_fon validation loaded strictly once as val_stream, never in train corpora
    assert calls.count(("zindi_fon", "validation")) == 1
    assert ("zindi_fon", "test") not in calls
    assert val_stream.name == "zindi_fon/validation"


# ---------------------------------------------------------------------------
# 2. Cropping, 60s filtering, mono/16kHz resampling fallback, cycle=True
# ---------------------------------------------------------------------------


def test_decode_audio_sample_mono_and_conditional_resample():
    mono_16k = _wav_bytes(np.linspace(-0.5, 0.5, 16000, dtype=np.float32), sample_rate=16000)
    with patch("fongbe.recipes.pretrain_fongbe._resample_waveform") as mock_resamp:
        wav, sr = decode_audio_sample({"audio": {"bytes": mono_16k}}, target_sample_rate=16000)
    assert sr == 16000
    assert wav.ndim == 1
    assert len(wav) == 16000
    mock_resamp.assert_not_called()

    # Stereo at 24 kHz should downmix to mono and resample to 16 kHz
    t = np.linspace(0, 1.0, 24000, endpoint=False, dtype=np.float32)
    stereo_24k = np.stack([np.sin(2 * np.pi * 220 * t), np.cos(2 * np.pi * 220 * t)], axis=1)
    raw_stereo_24k = _wav_bytes(stereo_24k, sample_rate=24000)
    wav2, sr2 = decode_audio_sample({"audio": {"bytes": raw_stereo_24k}}, target_sample_rate=16000)
    assert sr2 == 16000
    assert wav2.ndim == 1
    assert abs(len(wav2) - 16000) <= 10


def test_ssl_audio_dataset_filtering_cropping_cycling_and_state():
    samples = [
        {"audio": {"bytes": _wav_bytes(np.zeros(4000, dtype=np.float32), 16000)}},   # 0.25s -> filtered out
        {"audio": {"bytes": _wav_bytes(np.zeros(61 * 16000, dtype=np.float32), 16000)}},  # 61s -> filtered out
        {"audio": {"bytes": _wav_bytes(np.ones(30 * 16000, dtype=np.float32), 16000)}},   # 30s -> cropped to 20s
        {"audio": {"bytes": _wav_bytes(np.ones(4 * 16000, dtype=np.float32), 16000)}},    # 4s -> kept as 4s
    ]
    underlying = _FakeHFStream(samples, name="train")

    # Non-cycling single pass first to check filtering, random 20s cropping & AudioNoiseBatch contract
    ds_once = SSLAudioDataset(
        underlying,
        sample_rate=16000,
        min_duration=0.5,
        max_duration=60.0,
        crop_duration_sec=20.0,
        crop_mode="random",
        max_batch_duration=22.0,
        max_batch_size=4,
        sort_buffer_size=32,
        cycle=False,
    )
    batches = list(ds_once)
    # Kept 4s and 20s (cropped from 30s); 4s + 20s padded -> 40s > 22s max_batch_duration -> 2 batches
    assert len(batches) == 2
    assert batches[0].audio_len.tolist() == [4 * 16000]
    assert batches[1].audio_len.tolist() == [20 * 16000]
    assert batches[1].noisy_audio is batches[1].audio
    assert batches[1].noisy_audio_len is batches[1].audio_len

    # State dict delegation
    assert ds_once.state_dict() == {"cursor": 0, "name": "train"}
    ds_once.load_state_dict({"cursor": 7, "name": "train"})
    assert underlying.state_dict() == {"cursor": 7, "name": "train"}

    # Cycling mode wraps infinitely and calls set_epoch cleanly
    ds_cycle = SSLAudioDataset(
        underlying,
        min_duration=0.5,
        max_duration=60.0,
        crop_duration_sec=20.0,
        crop_mode="random",
        max_batch_duration=80.0,
        max_batch_size=4,
        sort_buffer_size=32,
        cycle=True,
    )
    it = iter(ds_cycle)
    seen = [next(it) for _ in range(4)]
    assert len(seen) == 4
    assert underlying.epoch >= 3


# ---------------------------------------------------------------------------
# 3. EncoderFreezeWarmupCallback, NEST forward/val_acc & encoder handoff
# ---------------------------------------------------------------------------


def test_encoder_freeze_warmup_callback_and_nest_step():
    cfg = build_nest_model_config(
        preprocessor_cfg={
            "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
            "sample_rate": 16000,
            "normalize": "per_feature",
            "window_size": 0.025,
            "window_stride": 0.01,
            "window": "hann",
            "features": 32,
            "n_fft": 512,
            "log": True,
            "frame_splicing": 1,
            "dither": 0.0,
            "pad_value": 0.0,
        },
        encoder_cfg={
            "_target_": "nemo.collections.asr.modules.ConformerEncoder",
            "feat_in": 32,
            "feat_out": -1,
            "n_layers": 1,
            "d_model": 32,
            "use_bias": False,
            "subsampling": "dw_striding",
            "subsampling_factor": 8,
            "subsampling_conv_channels": 16,
            "ff_expansion_factor": 2,
            "self_attention_model": "rel_pos",
            "n_heads": 2,
            "conv_kernel_size": 5,
            "conv_norm_type": "batch_norm",
        },
        num_classes=64,
        code_dim=8,
        block_size=16,
    )
    model = FongbeNESTModel(cfg=cfg)
    trainer = SimpleNamespace(global_step=0)
    cb = EncoderFreezeWarmupCallback(freeze_steps=500)

    cb.on_train_start(trainer, model)
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.decoder.parameters())

    trainer.global_step = 499
    cb.on_train_batch_start(trainer, model, None, 0)
    assert all(not p.requires_grad for p in model.encoder.parameters())

    trainer.global_step = 500
    cb.on_train_batch_start(trainer, model, None, 1)
    assert all(p.requires_grad for p in model.encoder.parameters())

    # Verify pad_to=16 keeps mel frames aligned with combine_time_steps=8 and produces val_loss + val_acc
    from nemo.collections.asr.data.ssl_dataset import AudioNoiseBatch

    wav = torch.randn(2, 16000 * 2 + 333)
    wav_len = torch.tensor([16000 + 111, 16000 * 2 + 333])
    batch = AudioNoiseBatch(audio=wav, audio_len=wav_len, noisy_audio=wav, noisy_audio_len=wav_len)
    model.eval()
    val_out = model.inference_pass(batch, batch_idx=0, mode="val", apply_mask=True)
    assert "val_loss" in val_out and torch.isfinite(val_out["val_loss"])
    assert "val_acc" in val_out and 0.0 <= float(val_out["val_acc"]) <= 1.0
    epoch_out = model.multi_validation_epoch_end([val_out])
    assert "val_loss" in epoch_out and "val_acc" in epoch_out


def test_encoder_handoff_strips_checkpoint_wrapper_and_loads_from_ckpt_and_nemo(tmp_path):
    enc = nn.Sequential(nn.Linear(8, 8), nn.ReLU())
    raw_sd = {
        "model.encoder.layers.0._checkpoint_wrapped_module.0.weight": enc[0].weight.clone() + 1.0,
        "encoder.layers.0._checkpoint_wrapped_module.0.bias": enc[0].bias.clone() + 2.0,
        "decoder.weight": torch.zeros(4, 8),
    }
    stripped = extract_encoder_state_dict({"state_dict": raw_sd})
    assert set(stripped.keys()) == {"layers.0.0.weight", "layers.0.0.bias"}

    ckpt_path = tmp_path / "ssl_last.ckpt"
    torch.save({"state_dict": raw_sd}, ckpt_path)
    loaded_ckpt = load_encoder_state_dict_from_source(str(ckpt_path))
    assert torch.allclose(loaded_ckpt["layers.0.0.weight"], raw_sd["model.encoder.layers.0._checkpoint_wrapped_module.0.weight"])

    weights_buf = io.BytesIO()
    torch.save(raw_sd, weights_buf)
    weights_bytes = weights_buf.getvalue()
    nemo_path = tmp_path / "ssl_model.nemo"
    with tarfile.open(nemo_path, "w") as tar:
        info = tarfile.TarInfo("model_weights.ckpt")
        info.size = len(weights_bytes)
        tar.addfile(info, io.BytesIO(weights_bytes))

    loaded_nemo = load_encoder_state_dict_from_source(str(nemo_path))
    assert set(loaded_nemo.keys()) == {"layers.0.0.weight", "layers.0.0.bias"}


# ---------------------------------------------------------------------------
# 4. convert_moondream key remapping
# ---------------------------------------------------------------------------


def test_convert_moondream_key_remapping_drops_only_vad_head():
    target_sd = {
        "preprocessor.featurizer.fb": torch.ones(1, 128, 257, dtype=torch.float32),
        "preprocessor.featurizer.window": torch.ones(400, dtype=torch.float32),
        "encoder.pre_encode.conv.0.weight": torch.empty(256, 1, 3, 3, dtype=torch.float32),
        "encoder.pre_encode.out.bias": torch.empty(1024, dtype=torch.float32),
        "encoder.layers.0.self_attn.linear_q.weight": torch.empty(1024, 1024, dtype=torch.float32),
        "encoder.layers.0.self_attn.linear_pos.weight": torch.empty(1024, 1024, dtype=torch.float32),
        "encoder.layers.0.self_attn.pos_bias_u": torch.empty(8, 128, dtype=torch.float32),
        "encoder.layers.0.conv.batch_norm.num_batches_tracked": torch.empty((), dtype=torch.int64),
        "decoder.prediction.embed.weight": torch.empty(8193, 640, dtype=torch.float32),
        "decoder.prediction.dec_rnn.lstm.weight_ih_l0": torch.empty(2560, 640, dtype=torch.float32),
        "joint.enc.weight": torch.empty(640, 1024, dtype=torch.float32),
        "joint.pred.weight": torch.empty(640, 640, dtype=torch.float32),
        "joint.joint_net.2.weight": torch.empty(8198, 640, dtype=torch.float32),
    }

    hf_sd = {
        "encoder.subsampling.layers.0.weight": torch.randn(256, 1, 3, 3, dtype=torch.bfloat16),
        "encoder.subsampling.linear.bias": torch.randn(1024, dtype=torch.bfloat16),
        "encoder.layers.0.self_attn.q_proj.weight": torch.randn(1024, 1024, dtype=torch.bfloat16),
        "encoder.layers.0.self_attn.relative_k_proj.weight": torch.randn(1024, 1024, dtype=torch.bfloat16),
        "encoder.layers.0.self_attn.bias_u": torch.randn(8, 128, dtype=torch.bfloat16),
        "encoder.layers.0.conv.norm.num_batches_tracked": torch.tensor(42, dtype=torch.int32),
        "decoder.embedding.weight": torch.randn(8193, 640, dtype=torch.bfloat16),
        "decoder.lstm.weight_ih_l0": torch.randn(2560, 640, dtype=torch.bfloat16),
        "encoder_projector.weight": torch.randn(640, 1024, dtype=torch.bfloat16),
        "decoder.decoder_projector.weight": torch.randn(640, 640, dtype=torch.bfloat16),
        "joint.head.weight": torch.randn(8198, 640, dtype=torch.bfloat16),
        "vad_head.proj.weight": torch.randn(128, 1024, dtype=torch.float16),
        "vad_head.out.bias": torch.randn(1, dtype=torch.float16),
    }

    mapped, stats = map_hf_to_nemo_state_dict(hf_sd, target_sd)
    assert stats["dropped_vad_keys"] == ["vad_head.out.bias", "vad_head.proj.weight"]
    assert stats["unmapped_non_vad_keys"] == []
    assert stats["missing_keys"] == []
    assert set(mapped.keys()) == set(target_sd.keys())
    for k, target_tensor in target_sd.items():
        assert mapped[k].dtype == target_tensor.dtype

    # Any unexpected non-VAD key must fail cleanly
    hf_bad = dict(hf_sd)
    hf_bad["unknown_module.weight"] = torch.zeros(2, 2)
    with pytest.raises(AssertionError, match="Unmapped non-VAD"):
        map_hf_to_nemo_state_dict(hf_bad, target_sd)


# ---------------------------------------------------------------------------
# 5. compare_predictions_bootstrap() statistical significance & TSV export
# ---------------------------------------------------------------------------


def test_compare_predictions_bootstrap_detects_significant_improvement(tmp_path):
    refs = [
        "ún ɖò xwégbè égbé",
        "mɛ̌ wɛ wá fí tɔ́n",
        "è ɖò azɔ̌ wà wɛ̀",
        "nyɛ̀ ɖò wěma xà wɛ̀",
        "xɔ́ntɔn cè yì axì mɛ̀",
    ] * 6  # 30 utterances
    pred_a = ["un do xwegbe egbe" if i % 2 == 0 else "foo bar baz qux" for i in range(len(refs))]
    pred_b = list(refs)  # Exact tonal predictions -> strictly lower error everywhere

    out_tsv = tmp_path / "sig.tsv"
    report = compare_predictions_bootstrap(
        refs, pred_a, pred_b, n_bootstrap=500, seed=42, output_tsv=out_tsv
    )

    assert report["n_utterances"] == 30
    assert out_tsv.is_file()
    tsv_text = out_tsv.read_text(encoding="utf-8")
    assert "weighted_wer" in tsv_text

    for metric in ("wer", "weighted_wer", "competition_score", "cer"):
        m = report[metric]
        assert m["model_b"] < m["model_a"]
        assert m["delta"] < 0.0
        assert m["ci_low"] <= m["delta"] <= m["ci_high"] < 0.0
        assert m["p_value"] < 0.05

    tie_report = compare_predictions_bootstrap(refs, pred_a, pred_a, n_bootstrap=200, seed=42)
    assert tie_report["wer"]["delta"] == 0.0
    assert tie_report["wer"]["p_value"] == 1.0
