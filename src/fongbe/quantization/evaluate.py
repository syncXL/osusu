from __future__ import annotations

import csv
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from compressed_tensors.compressors import ModelCompressor
from torch import nn

from fongbe.quantization.algorithms import LayerQuantDiagnostics
from fongbe.quantization.schemes import QuantizationSpec


def _levenshtein_distance(a: Sequence[Any], b: Sequence[Any]) -> int:
    """Compute edit distance between two sequences using rapidfuzz when available."""
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
                if a[i - 1] == b[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = cur
        return dp[m]


def _normalize_for_eval(text: str) -> str:
    normed = unicodedata.normalize("NFD", text or "").lower()
    return " ".join(normed.split())


def compute_wer_and_cer(pairs: Sequence[tuple[str, str]]) -> tuple[float, float]:
    """Compute corpus-level WER and CER over `(reference, hypothesis)` pairs."""
    word_edits = 0
    ref_words = 0
    char_edits = 0
    ref_chars = 0

    for ref, hyp in pairs:
        r_norm = _normalize_for_eval(ref)
        h_norm = _normalize_for_eval(hyp)

        r_w = r_norm.split()
        h_w = h_norm.split()
        word_edits += _levenshtein_distance(r_w, h_w)
        ref_words += len(r_w)

        r_c = list(r_norm)
        h_c = list(h_norm)
        char_edits += _levenshtein_distance(r_c, h_c)
        ref_chars += len(r_c)

    wer = word_edits / max(1, ref_words)
    cer = char_edits / max(1, ref_chars)
    return float(wer), float(cer)


@dataclass
class EvaluationReport:
    algorithm: str
    scheme: str
    bits: int
    granularity: str
    block_size: int
    static_ptq: bool
    calib_method: str
    calib_samples: int
    eval_samples: int
    wer_fp32: float
    wer_quant: float
    wer_delta: float
    cer_fp32: float
    cer_quant: float
    cer_delta: float
    size_mb_fp32: float
    size_mb_quant: float
    compression_ratio: float
    wer_threshold: float
    needs_qat: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "scheme": self.scheme,
            "bits": self.bits,
            "granularity": self.granularity,
            "block_size": self.block_size,
            "static_ptq": self.static_ptq,
            "calib_method": self.calib_method,
            "calib_samples": self.calib_samples,
            "eval_samples": self.eval_samples,
            "wer_fp32": f"{self.wer_fp32:.6f}",
            "wer_quant": f"{self.wer_quant:.6f}",
            "wer_delta": f"{self.wer_delta:.6f}",
            "cer_fp32": f"{self.cer_fp32:.6f}",
            "cer_quant": f"{self.cer_quant:.6f}",
            "cer_delta": f"{self.cer_delta:.6f}",
            "size_mb_fp32": f"{self.size_mb_fp32:.2f}",
            "size_mb_quant": f"{self.size_mb_quant:.2f}",
            "compression_ratio": f"{self.compression_ratio:.2f}",
            "wer_threshold": f"{self.wer_threshold:.6f}",
            "needs_qat": self.needs_qat,
        }


def build_evaluation_report(
    spec: QuantizationSpec,
    eval_samples: int,
    wer_fp32: float,
    wer_quant: float,
    cer_fp32: float,
    cer_quant: float,
    size_mb_fp32: float,
    size_mb_quant: float,
    wer_threshold: float = 0.01,
) -> EvaluationReport:
    """Create an EvaluationReport and compute deltas, compression ratio, and QAT gate."""
    wer_delta = float(wer_quant - wer_fp32)
    cer_delta = float(cer_quant - cer_fp32)
    compression_ratio = float(size_mb_fp32 / max(size_mb_quant, 1e-9))
    needs_qat = bool(wer_delta > float(wer_threshold))
    return EvaluationReport(
        algorithm=spec.algorithm,
        scheme=spec.scheme,
        bits=spec.bits,
        granularity=spec.granularity,
        block_size=spec.block_size,
        static_ptq=spec.static_ptq,
        calib_method=spec.calib_method,
        calib_samples=spec.calib_samples if spec.requires_calibration else 0,
        eval_samples=eval_samples,
        wer_fp32=float(wer_fp32),
        wer_quant=float(wer_quant),
        wer_delta=wer_delta,
        cer_fp32=float(cer_fp32),
        cer_quant=float(cer_quant),
        cer_delta=cer_delta,
        size_mb_fp32=float(size_mb_fp32),
        size_mb_quant=float(size_mb_quant),
        compression_ratio=compression_ratio,
        wer_threshold=float(wer_threshold),
        needs_qat=needs_qat,
    )


@torch.no_grad()
def transcribe_samples(
    model: nn.Module,
    processor,
    samples: Sequence[dict[str, Any]],
    device: torch.device | str = "cpu",
    batch_size: int = 4,
) -> list[str]:
    """Run greedy decoding on HF ParakeetForTDT and return decoded transcripts."""
    if not samples:
        return []
    # ParakeetForTDT.generate() calls get_audio_features() -> encoder() before forward(),
    # bypassing top-level ct_decompress_hook; decompress explicitly if still compressed.
    if hasattr(model, "ct_decompress_hook"):
        ModelCompressor.from_pretrained_model(model).decompress_model(model)
    model.eval()
    preds: list[str] = []
    gen_kwargs: dict[str, Any] = {}
    if getattr(model.generation_config, "decoder_start_token_id", None) is None:
        blank_id = getattr(model.config, "blank_token_id", None)
        if blank_id is not None:
            gen_kwargs["decoder_start_token_id"] = int(blank_id)
    if getattr(model.generation_config, "pad_token_id", None) is None:
        pad_id = getattr(model.config, "pad_token_id", getattr(model.config, "blank_token_id", None))
        if pad_id is not None:
            gen_kwargs["pad_token_id"] = int(pad_id)

    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        arrays = [s["array"] for s in chunk]
        sr = int(chunk[0].get("sampling_rate", 16000))
        inputs = processor(arrays, sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        gen_out = model.generate(**inputs, **gen_kwargs)
        sequences = gen_out.sequences if hasattr(gen_out, "sequences") else gen_out
        decoded = processor.batch_decode(sequences, skip_special_tokens=True)
        preds.extend(decoded)
    return preds


def write_tsv_reports(
    report: EvaluationReport,
    layer_diagnostics: Sequence[LayerQuantDiagnostics],
    report_tsv: str | Path,
) -> tuple[Path, Path]:
    """Write model-level `.tsv` report and per-layer `.layers.tsv` diagnostic report."""
    report_path = Path(report_tsv)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    layers_path = report_path.with_suffix(".layers.tsv")

    row = report.to_dict()
    with report_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), delimiter="\t")
        writer.writeheader()
        writer.writerow(row)

    layer_fields = ["layer_name", "weight_mse", "weight_sqnr_db", "max_abs_err"]
    with layers_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=layer_fields, delimiter="\t")
        writer.writeheader()
        for item in layer_diagnostics:
            writer.writerow(
                {
                    "layer_name": item.layer_name,
                    "weight_mse": f"{item.weight_mse:.8e}",
                    "weight_sqnr_db": f"{item.weight_sqnr_db:.3f}",
                    "max_abs_err": f"{item.max_abs_err:.6f}",
                }
            )

    return report_path, layers_path


def print_qat_decision_banner(
    report: EvaluationReport,
    report_path: Path,
    layers_path: Path,
) -> None:
    """Print a concise terminal summary and QAT recommendation banner."""
    verdict = (
        f"QAT RECOMMENDED (wer_delta={report.wer_delta:+.4f} > threshold={report.wer_threshold:.4f})"
        if report.needs_qat
        else f"PTQ PASSED — QAT NOT NEEDED (wer_delta={report.wer_delta:+.4f} <= threshold={report.wer_threshold:.4f})"
    )
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"Fongbe Parakeet Quantization Summary ({report.algorithm.upper()}, {report.bits}-bit {report.scheme}, {report.granularity})")
    print(sep)
    print(f"Eval Samples      : {report.eval_samples}")
    print(f"WER (FP32 / Quant): {report.wer_fp32:.4f} / {report.wer_quant:.4f} (delta {report.wer_delta:+.4f})")
    print(f"CER (FP32 / Quant): {report.cer_fp32:.4f} / {report.cer_quant:.4f} (delta {report.cer_delta:+.4f})")
    print(f"Size (FP32 / Q)   : {report.size_mb_fp32:.2f} MB / {report.size_mb_quant:.2f} MB ({report.compression_ratio:.2f}x)")
    print(f"Decision          : {verdict}")
    print(f"Report TSV        : {report_path}")
    print(f"Layer Diag TSV    : {layers_path}")
    print(f"{sep}\n")
