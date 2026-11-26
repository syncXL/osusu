from __future__ import annotations

import copy
from pathlib import Path
from typing import Iterable

import fire
import torch

from fongbe.quantization.algorithms import apply_quantization_algorithm
from fongbe.quantization.calibrator import iter_hf_audio_samples
from fongbe.quantization.converter import load_parakeet_model_and_processor
from fongbe.quantization.evaluate import (
    build_evaluation_report,
    compute_wer_and_cer,
    print_qat_decision_banner,
    transcribe_samples,
    write_tsv_reports,
)
from fongbe.quantization.exporter import save_quantized_hf_model, state_dict_size_mb
from fongbe.quantization.schemes import DEFAULT_IGNORE_MODULES, validate_quant_args


def _prepare_batches(processor, samples: list[dict], batch_size: int = 4) -> list[dict[str, torch.Tensor]]:
    batches: list[dict[str, torch.Tensor]] = []
    for i in range(0, len(samples), batch_size):
        chunk = samples[i : i + batch_size]
        arrays = [s["array"] for s in chunk]
        sr = int(chunk[0].get("sampling_rate", 16000))
        feat = processor(arrays, sampling_rate=sr, return_tensors="pt", padding=True)
        batches.append(dict(feat))
    return batches


def quantize(
    model_path: str,
    output_dir: str = "outputs/quantized_parakeet",
    scheme: str = "symmetric",
    bits: int = 8,
    granularity: str = "per_channel",
    block_size: int = 128,
    algorithm: str = "absmax",
    smooth_alpha: float = 0.5,
    static_ptq: bool = False,
    calib_samples: int = 64,
    calib_method: str = "percentile",
    calib_percentile: float = 99.9,
    calib_split: str = "validation",
    dataset_id: str = "nolimitsxl/osusu",
    dataset_config: str = "zindi_fon",
    ignore_modules: str | Iterable[str] | None = DEFAULT_IGNORE_MODULES,
    eval_samples: int = 50,
    eval_target_key: str = "syllables",
    wer_threshold: float = 0.01,
    report_tsv: str | None = None,
    batch_size: int = 4,
    device: str | None = None,
    push_to_hub: bool = False,
    hub_repo_id: str | None = None,
) -> dict:
    """Quantize a Hugging Face ParakeetForTDT model or local `.nemo` checkpoint and evaluate PTQ impact.

    Args:
        model_path: Local `.nemo` checkpoint path, HF directory, or HF Hub repo ID.
        output_dir: Output directory for quantized HF model (`model.safetensors` + `config.json`).
        scheme: Quantization symmetry (`symmetric` or `affine`).
        bits: Weight bit-width (`8`, `4`, or `3`).
        granularity: Weight granularity (`per_tensor`, `per_channel`, or `per_block`).
        block_size: Block/group size when `granularity="per_block"`.
        algorithm: Weight quantization algorithm (`absmax`, `awq`, `gptq`, or `smoothquant`).
        smooth_alpha: Migration strength for SmoothQuant (`[0.0, 1.0]`).
        static_ptq: Enable static W8A8 activation quantization (requires `--bits=8`).
        calib_samples: Number of HF utterances streamed for calibration.
        calib_method: Activation calibration observer (`absmax` or `percentile`).
        calib_percentile: Quantile percentage when `calib_method="percentile"`.
        calib_split: HF split used for calibration (`validation` or `train`).
        dataset_id: HF dataset repository ID (`nolimitsxl/osusu`).
        dataset_config: HF dataset configuration (`zindi_fon`).
        ignore_modules: Comma-separated or iterable module exclusions (`pre_encode,decoder,joint,encoder_projector`).
        eval_samples: Number of `zindi_fon` validation utterances to evaluate FP32 vs quantized WER/CER (0 to skip).
        eval_target_key: Reference transcript column (`syllables` or `text`).
        wer_threshold: Absolute WER degradation threshold above which QAT is flagged as needed.
        report_tsv: Output path for TSV metrics report (defaults to `<output_dir>/quantization_report.tsv`).
        batch_size: Micro-batch size for calibration and evaluation forward passes.
        device: Torch device (`cuda` or `cpu`; auto-detected if omitted).
        push_to_hub: Upload saved HF quantized directory to Hugging Face Hub.
        hub_repo_id: Target Hugging Face Hub repo ID when `push_to_hub=True`.
    """
    spec = validate_quant_args(
        scheme=scheme,
        bits=bits,
        granularity=granularity,
        block_size=block_size,
        algorithm=algorithm,
        smooth_alpha=smooth_alpha,
        static_ptq=static_ptq,
        calib_samples=calib_samples,
        calib_method=calib_method,
        calib_percentile=calib_percentile,
        ignore_modules=ignore_modules,
    )

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    target_dtype = torch.float16 if resolved_device.type == "cuda" else torch.float32

    model, processor = load_parakeet_model_and_processor(model_path, dtype=target_dtype)
    model.to(resolved_device)
    size_mb_fp32 = state_dict_size_mb(model)

    sr = int(getattr(processor.feature_extractor, "sampling_rate", 16000))

    eval_data: list[dict] = []
    wer_fp32 = 0.0
    cer_fp32 = 0.0
    if eval_samples > 0:
        eval_data = iter_hf_audio_samples(
            dataset_id=dataset_id,
            dataset_config=dataset_config,
            split="validation",
            max_samples=eval_samples,
            target_sr=sr,
            target_key=eval_target_key,
        )
        fp32_preds = transcribe_samples(model, processor, eval_data, device=resolved_device, batch_size=batch_size)
        fp32_pairs = [(s["text"], p) for s, p in zip(eval_data, fp32_preds)]
        wer_fp32, cer_fp32 = compute_wer_and_cer(fp32_pairs)

    calib_batches = None
    if spec.requires_calibration:
        if eval_data and calib_split == "validation" and len(eval_data) >= spec.calib_samples:
            calib_data = eval_data[: spec.calib_samples]
        else:
            calib_data = iter_hf_audio_samples(
                dataset_id=dataset_id,
                dataset_config=dataset_config,
                split=calib_split,
                max_samples=spec.calib_samples,
                target_sr=sr,
                target_key=eval_target_key,
            )
        calib_batches = _prepare_batches(processor, calib_data, batch_size=batch_size)

    _, layer_diagnostics = apply_quantization_algorithm(
        model=model,
        spec=spec,
        calib_batches=calib_batches,
        device=resolved_device,
    )

    out_path, size_mb_quant = save_quantized_hf_model(
        model=model,
        processor=processor,
        output_dir=output_dir,
        push_to_hub=push_to_hub,
        hub_repo_id=hub_repo_id,
    )

    wer_quant = 0.0
    cer_quant = 0.0
    if eval_samples > 0 and eval_data:
        quant_preds = transcribe_samples(model, processor, eval_data, device=resolved_device, batch_size=batch_size)
        quant_pairs = [(s["text"], p) for s, p in zip(eval_data, quant_preds)]
        wer_quant, cer_quant = compute_wer_and_cer(quant_pairs)

    report = build_evaluation_report(
        spec=spec,
        eval_samples=len(eval_data),
        wer_fp32=wer_fp32,
        wer_quant=wer_quant,
        cer_fp32=cer_fp32,
        cer_quant=cer_quant,
        size_mb_fp32=size_mb_fp32,
        size_mb_quant=size_mb_quant,
        wer_threshold=wer_threshold,
    )

    tsv_target = Path(report_tsv) if report_tsv else (out_path / "quantization_report.tsv")
    report_path, layers_path = write_tsv_reports(report, layer_diagnostics, tsv_target)
    print_qat_decision_banner(report, report_path, layers_path)

    return {
        "output_dir": str(out_path),
        "report_tsv": str(report_path),
        "layers_tsv": str(layers_path),
        "report": report.to_dict(),
    }


def main():
    fire.Fire(quantize)


if __name__ == "__main__":
    main()
