from __future__ import annotations

import csv
import io
import json
import os
import random
import tarfile
import unicodedata
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import sentencepiece as spm
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Audio, load_dataset

from fongbe.model import extract_encoder_state_dict, load_encoder_state_dict_from_source
from fongbe.recipes.pretrain_fongbe import decode_audio_sample


def _edit_distance(a: Sequence[Any], b: Sequence[Any]) -> int:
    try:
        from rapidfuzz.distance import Levenshtein

        return int(Levenshtein.distance(a, b))
    except ImportError:
        n, m = len(a), len(b)
        dp = list(range(m + 1))
        for i in range(1, n + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, m + 1):
                cur = dp[j]
                dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
                prev = cur
        return dp[m]


def _normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFD", text or "").lower().split())


def _load_competition_scorer():
    """Load `measure(pairs)` from `data/score.py` if available in the repository."""
    import importlib.util
    import sys

    score_path = Path(__file__).resolve().parents[2] / "data" / "score.py"
    if not score_path.is_file():
        return None
    data_dir = str(score_path.parent)
    if data_dir not in sys.path:
        sys.path.insert(0, data_dir)
    spec = importlib.util.spec_from_file_location("fongbe_official_score", score_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _per_utterance_stats(
    ref_texts: Sequence[str],
    pred_texts: Sequence[str],
    score_mod: Any | None,
) -> list[dict[str, float]]:
    from rapidfuzz.distance import Levenshtein

    rows: list[dict[str, float]] = []
    for ref, hyp in zip(ref_texts, pred_texts):
        r_norm = _normalize_text(ref)
        h_norm = _normalize_text(hyp)
        r_words = r_norm.split()
        h_words = h_norm.split()
        w_edits = float(_edit_distance(r_words, h_words))
        r_w_len = float(len(r_words))
        c_edits = float(_edit_distance(list(r_norm), list(h_norm)))
        r_c_len = float(len(r_norm))

        if score_mod is not None:
            r_syl = score_mod.syllables(ref)
            h_syl = score_mod.syllables(hyp)
            syl_ref_n = float(len(r_syl))
            syl_tonal_dist = float(
                _edit_distance([b + "|" + t for b, t in r_syl], [b + "|" + t for b, t in h_syl])
            )
            rb = [b for b, _ in r_syl]
            hb = [b for b, _ in h_syl]
            matched = 0.0
            wrong_tone = 0.0
            for op in Levenshtein.opcodes(rb, hb):
                if op.tag == "equal":
                    for k in range(op.src_end - op.src_start):
                        matched += 1.0
                        if r_syl[op.src_start + k][1] != h_syl[op.dest_start + k][1]:
                            wrong_tone += 1.0
        else:
            syl_ref_n = r_w_len
            syl_tonal_dist = w_edits
            matched = 0.0
            wrong_tone = 0.0

        rows.append(
            {
                "word_edits": w_edits,
                "ref_words": r_w_len,
                "utt_wer": w_edits / max(1.0, r_w_len),
                "char_edits": c_edits,
                "ref_chars": r_c_len,
                "syl_tonal_dist": syl_tonal_dist,
                "syl_ref_n": syl_ref_n,
                "matched_syl": matched,
                "wrong_tone": wrong_tone,
            }
        )
    return rows


def _aggregate_metrics(stats: Sequence[dict[str, float]], indices: np.ndarray | None = None) -> dict[str, float]:
    if indices is None:
        subset = stats
    else:
        subset = [stats[int(i)] for i in indices]

    w_edits = sum(r["word_edits"] for r in subset)
    r_words = sum(r["ref_words"] for r in subset)
    utt_wer_mean = sum(r["utt_wer"] for r in subset) / max(1, len(subset))
    c_edits = sum(r["char_edits"] for r in subset)
    r_chars = sum(r["ref_chars"] for r in subset)

    syl_dist = sum(r["syl_tonal_dist"] for r in subset)
    syl_ref = sum(r["syl_ref_n"] for r in subset)
    matched = sum(r["matched_syl"] for r in subset)
    wrong_tone = sum(r["wrong_tone"] for r in subset)

    corpus_wer = w_edits / max(1.0, r_words)
    cer = c_edits / max(1.0, r_chars)
    tser = min(1.0, syl_dist / max(1.0, syl_ref))
    terc = wrong_tone / max(1.0, matched)
    competition_score = 0.7 * tser + 0.3 * terc

    return {
        "wer": float(utt_wer_mean),
        "weighted_wer": float(corpus_wer),
        "competition_score": float(competition_score),
        "cer": float(cer),
    }


def compare_predictions_bootstrap(
    ref_texts: Sequence[str],
    pred_a: Sequence[str],
    pred_b: Sequence[str],
    n_bootstrap: int = 1000,
    seed: int = 42,
    output_tsv: str | Path | None = None,
) -> dict[str, Any]:
    """Compare models A and B using paired bootstrap resampling.

    Computes unweighted utterance WER (``wer``), length-weighted corpus WER
    (``weighted_wer``), competition tonal score (``competition_score``), and ``cer``,
    alongside paired deltas (``B - A``, negative favors B), 95% confidence intervals,
    and two-sided paired bootstrap p-values. Optionally writes a summary ``.tsv``.
    """
    n = len(ref_texts)
    if n == 0:
        raise ValueError("ref_texts must not be empty.")
    if len(pred_a) != n or len(pred_b) != n:
        raise ValueError(
            f"Length mismatch: refs={n}, pred_a={len(pred_a)}, pred_b={len(pred_b)}."
        )
    if n_bootstrap <= 0:
        raise ValueError(f"n_bootstrap must be positive, got {n_bootstrap}.")

    score_mod = _load_competition_scorer()
    stats_a = _per_utterance_stats(ref_texts, pred_a, score_mod)
    stats_b = _per_utterance_stats(ref_texts, pred_b, score_mod)

    obs_a = _aggregate_metrics(stats_a)
    obs_b = _aggregate_metrics(stats_b)

    metrics_keys = ("wer", "weighted_wer", "competition_score", "cer")
    boot_deltas: dict[str, list[float]] = {k: [] for k in metrics_keys}

    rng = np.random.default_rng(int(seed))
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        m_a = _aggregate_metrics(stats_a, idx)
        m_b = _aggregate_metrics(stats_b, idx)
        for k in metrics_keys:
            boot_deltas[k].append(m_b[k] - m_a[k])

    summary: dict[str, Any] = {
        "n_utterances": n,
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
    }

    for k in metrics_keys:
        deltas = np.asarray(boot_deltas[k], dtype=np.float64)
        obs_delta = obs_b[k] - obs_a[k]
        ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
        n_pos = int(np.sum(deltas >= 0.0))
        n_neg = int(np.sum(deltas <= 0.0))
        p_val = min(1.0, 2.0 * (min(n_pos, n_neg) + 1.0) / (len(deltas) + 1.0))

        summary[k] = {
            "model_a": float(obs_a[k]),
            "model_b": float(obs_b[k]),
            "delta": float(obs_delta),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "p_value": float(p_val),
        }

    if output_tsv is not None:
        tsv_path = Path(output_tsv)
        tsv_path.parent.mkdir(parents=True, exist_ok=True)
        with tsv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(
                [
                    "metric",
                    "model_a",
                    "model_b",
                    "delta_b_minus_a",
                    "ci_95_low",
                    "ci_95_high",
                    "p_value",
                    "n_utterances",
                    "n_bootstrap",
                ]
            )
            for k in metrics_keys:
                row = summary[k]
                writer.writerow(
                    [
                        k,
                        f"{row['model_a']:.6f}",
                        f"{row['model_b']:.6f}",
                        f"{row['delta']:.6f}",
                        f"{row['ci_low']:.6f}",
                        f"{row['ci_high']:.6f}",
                        f"{row['p_value']:.6f}",
                        n,
                        int(n_bootstrap),
                    ]
                )
        summary["summary_tsv"] = str(tsv_path)

    return summary


def _ctc_greedy_decode(
    token_ids: Sequence[int],
    *,
    blank_id: int,
    sp: spm.SentencePieceProcessor,
) -> str:
    collapsed: list[int] = []
    prev: int | None = None
    for tid in token_ids:
        t = int(tid)
        if t != prev and t != blank_id:
            collapsed.append(t)
        prev = t
    return sp.decode(collapsed) if collapsed else ""


def _load_encoder_and_preprocessor(encoder_source: str, base_nemo_model: str):
    from nemo.collections.asr.models import ASRModel

    if os.path.exists(encoder_source):
        base_asr = (
            ASRModel.restore_from(base_nemo_model, map_location="cpu")
            if os.path.exists(base_nemo_model) and base_nemo_model.endswith(".nemo")
            else ASRModel.from_pretrained(model_name=base_nemo_model, map_location="cpu")
        )
        encoder_sd = load_encoder_state_dict_from_source(encoder_source)
        base_asr.encoder.load_state_dict(encoder_sd, strict=True)
        preprocessor = base_asr.preprocessor
        encoder = base_asr.encoder
        d_model = int(base_asr.cfg.encoder.d_model)
    else:
        asr = ASRModel.from_pretrained(model_name=encoder_source, map_location="cpu")
        preprocessor = asr.preprocessor
        encoder = asr.encoder
        d_model = int(asr.cfg.encoder.d_model)

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    preprocessor.eval()
    return preprocessor, encoder, d_model


def run_frozen_ctc_probe(
    encoder_source: str,
    tokenizer_dir: str,
    *,
    dataset_id: str = "nolimitsxl/osusu",
    dataset_config: str = "zindi_fon",
    target_key: str = "syllables",
    base_nemo_model: str = "nvidia/parakeet-tdt-0.6b-v3",
    output_dir: str = "/tmp/ctc_probe",
    steps: int = 1500,
    lr: float = 1e-3,
    batch_size: int = 4,
    max_duration: float = 30.0,
    min_duration: float = 0.3,
    max_val_samples: int | None = None,
    seed: int = 42,
    device: str | None = None,
) -> dict[str, Any]:
    """Train a tiny frozen-encoder linear CTC probe on ``zindi_fon/train`` and evaluate on validation.

    Saves per-utterance predictions to ``<output_dir>/predictions.tsv`` so
    ``compare_predictions_bootstrap()`` can statistically test two encoders fairly.
    """
    torch.manual_seed(int(seed))
    random.seed(int(seed))

    tok_model = Path(tokenizer_dir) / "tokenizer.model"
    if not tok_model.is_file():
        raise FileNotFoundError(f"SentencePiece tokenizer not found at '{tok_model}'.")
    sp = spm.SentencePieceProcessor(model_file=str(tok_model))
    vocab_size = int(sp.get_piece_size())
    blank_id = vocab_size

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    preprocessor, encoder, d_model = _load_encoder_and_preprocessor(encoder_source, base_nemo_model)
    preprocessor = preprocessor.to(resolved_device)
    encoder = encoder.to(resolved_device)

    ctc_head = nn.Linear(d_model, vocab_size + 1).to(resolved_device)
    optimizer = torch.optim.AdamW(ctc_head.parameters(), lr=float(lr), weight_decay=1e-4)

    train_stream = load_dataset(dataset_id, dataset_config, split="train", streaming=True)
    train_stream = train_stream.cast_column("audio", Audio(decode=False)).shuffle(seed=seed, buffer_size=256)

    def _collate_ctc_items(items: list[tuple[np.ndarray, list[int]]]):
        lengths = torch.tensor([len(w) for w, _ in items], dtype=torch.long)
        sig = torch.zeros((len(items), int(lengths.max().item())), dtype=torch.float32)
        flat_targets: list[int] = []
        target_lens: list[int] = []
        for i, (w, ids) in enumerate(items):
            sig[i, : len(w)] = torch.from_numpy(w)
            flat_targets.extend(ids)
            target_lens.append(len(ids))
        return (
            sig.to(resolved_device),
            lengths.to(resolved_device),
            torch.tensor(flat_targets, dtype=torch.long, device=resolved_device),
            torch.tensor(target_lens, dtype=torch.long, device=resolved_device),
        )

    ctc_head.train()
    step = 0
    buf: list[tuple[np.ndarray, list[int]]] = []
    epoch = 0
    while step < int(steps):
        if epoch > 0 and hasattr(train_stream, "set_epoch"):
            train_stream.set_epoch(epoch)
        epoch += 1
        progressed = False
        for sample in train_stream:
            text = str(sample.get(target_key) or sample.get("text") or "").strip()
            if not text:
                continue
            try:
                wav, sr = decode_audio_sample(sample, target_sample_rate=16000)
            except Exception:
                continue
            dur = len(wav) / float(sr)
            if dur < min_duration or dur > max_duration:
                continue
            ids = sp.encode(text, out_type=int)
            if not ids:
                continue
            buf.append((wav, ids))
            progressed = True
            if len(buf) < batch_size:
                continue

            sig, sig_len, targets, target_lens = _collate_ctc_items(buf)
            buf = []
            with torch.no_grad():
                mel, mel_len = preprocessor(input_signal=sig, length=sig_len)
                encoded, enc_len = encoder(audio_signal=mel, length=mel_len)
            logits = ctc_head(encoded.transpose(1, 2).float())
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            loss = F.ctc_loss(log_probs, targets, enc_len, target_lens, blank=blank_id, zero_infinity=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step >= int(steps):
                break
        if not progressed:
            break

    # Evaluate on zindi_fon validation
    ctc_head.eval()
    val_stream = load_dataset(dataset_id, dataset_config, split="validation", streaming=True)
    val_stream = val_stream.cast_column("audio", Audio(decode=False))

    utterance_ids: list[str] = []
    references: list[str] = []
    predictions: list[str] = []

    with torch.no_grad():
        for idx, sample in enumerate(val_stream):
            if max_val_samples is not None and idx >= max_val_samples:
                break
            ref = str(sample.get(target_key) or sample.get("text") or "").strip()
            try:
                wav, sr = decode_audio_sample(sample, target_sample_rate=16000)
            except Exception:
                continue
            sig = torch.from_numpy(wav).unsqueeze(0).to(resolved_device)
            sig_len = torch.tensor([len(wav)], dtype=torch.long, device=resolved_device)
            mel, mel_len = preprocessor(input_signal=sig, length=sig_len)
            encoded, enc_len = encoder(audio_signal=mel, length=mel_len)
            logits = ctc_head(encoded.transpose(1, 2).float())
            best_path = logits[0, : int(enc_len[0].item())].argmax(dim=-1).tolist()
            pred = _ctc_greedy_decode(best_path, blank_id=blank_id, sp=sp)
            utterance_ids.append(str(sample.get("id", idx)))
            references.append(ref)
            predictions.append(pred)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_tsv = out_dir / "predictions.tsv"
    with pred_tsv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["id", "reference", "prediction"])
        for uid, ref, pred in zip(utterance_ids, references, predictions):
            writer.writerow([uid, ref, pred])

    score_mod = _load_competition_scorer()
    stats = _per_utterance_stats(references, predictions, score_mod)
    metrics = _aggregate_metrics(stats) if stats else {}

    summary_json = out_dir / "metrics.json"
    summary_json.write_text(json.dumps({"encoder_source": encoder_source, **metrics}, indent=2))

    return {
        "encoder_source": encoder_source,
        "predictions_tsv": str(pred_tsv),
        "metrics_json": str(summary_json),
        "references": references,
        "predictions": predictions,
        "metrics": metrics,
    }
