from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)

VALID_SCHEMES = ("symmetric", "affine")
VALID_BITS = (8, 4, 3)
VALID_GRANULARITIES = ("per_tensor", "per_channel", "per_block")
VALID_ALGORITHMS = ("absmax", "awq", "gptq", "smoothquant")
VALID_CALIB_METHODS = ("absmax", "percentile")

DEFAULT_IGNORE_MODULES = ("pre_encode", "decoder", "joint", "encoder_projector")

_IGNORE_ALIASES = {
    "pre_encode": r"re:^encoder\.subsampling\..*",
    "encoder.subsampling": r"re:^encoder\.subsampling\..*",
    "decoder": r"re:^decoder\..*",
    "joint": r"re:^joint\..*",
    "encoder_projector": "encoder_projector",
}


def normalize_ignore_modules(ignore_modules: str | Iterable[str] | None) -> list[str]:
    """Normalize user-supplied module exclusions into compressed-tensors ignore patterns."""
    if ignore_modules is None:
        return []
    if isinstance(ignore_modules, str):
        raw_items = [item.strip() for item in ignore_modules.split(",") if item.strip()]
    else:
        raw_items = [str(item).strip() for item in ignore_modules if str(item).strip()]

    patterns: list[str] = []
    for item in raw_items:
        if item in _IGNORE_ALIASES:
            pat = _IGNORE_ALIASES[item]
        elif item.startswith("re:") or item in ("Linear", "Embedding"):
            pat = item
        elif "." in item:
            escaped = item.replace(".", r"\.")
            pat = rf"re:^{escaped}(\..*)?$"
        else:
            pat = item
        if pat not in patterns:
            patterns.append(pat)
    return patterns


@dataclass(frozen=True)
class QuantizationSpec:
    """Validated specification for Parakeet quantization."""

    scheme: str = "symmetric"
    bits: int = 8
    granularity: str = "per_channel"
    block_size: int = 128
    algorithm: str = "absmax"
    smooth_alpha: float = 0.5
    static_ptq: bool = False
    calib_samples: int = 64
    calib_method: str = "percentile"
    calib_percentile: float = 99.9
    ignore_patterns: list[str] = field(default_factory=lambda: normalize_ignore_modules(DEFAULT_IGNORE_MODULES))

    @property
    def symmetric(self) -> bool:
        return self.scheme == "symmetric"

    @property
    def requires_calibration(self) -> bool:
        return self.static_ptq or self.algorithm in ("awq", "gptq", "smoothquant")


def validate_quant_args(
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
    ignore_modules: str | Iterable[str] | None = DEFAULT_IGNORE_MODULES,
) -> QuantizationSpec:
    """Validate quantization CLI parameters and reject incompatible flag combinations."""
    scheme = str(scheme).lower()
    granularity = str(granularity).lower()
    algorithm = str(algorithm).lower()
    calib_method = str(calib_method).lower()
    bits = int(bits)
    block_size = int(block_size)

    if scheme not in VALID_SCHEMES:
        raise ValueError(f"Unsupported --scheme '{scheme}'. Expected one of {VALID_SCHEMES}.")
    if bits not in VALID_BITS:
        raise ValueError(f"Unsupported --bits={bits}. Expected one of {VALID_BITS}.")
    if granularity not in VALID_GRANULARITIES:
        raise ValueError(f"Unsupported --granularity '{granularity}'. Expected one of {VALID_GRANULARITIES}.")
    if algorithm not in VALID_ALGORITHMS:
        raise ValueError(f"Unsupported --algorithm '{algorithm}'. Expected one of {VALID_ALGORITHMS}.")
    if calib_method not in VALID_CALIB_METHODS:
        raise ValueError(f"Unsupported --calib_method '{calib_method}'. Expected one of {VALID_CALIB_METHODS}.")
    if block_size <= 0:
        raise ValueError(f"--block_size must be a positive integer, got {block_size}.")
    if not (0.0 <= float(smooth_alpha) <= 1.0):
        raise ValueError(f"--smooth_alpha must be in [0.0, 1.0], got {smooth_alpha}.")
    if not (0.0 < float(calib_percentile) <= 100.0):
        raise ValueError(f"--calib_percentile must be in (0.0, 100.0], got {calib_percentile}.")

    if algorithm == "smoothquant":
        if bits != 8:
            raise ValueError(
                f"--algorithm='smoothquant' requires --bits=8 (W8A8), got --bits={bits}."
            )
        if not static_ptq:
            raise ValueError(
                "--algorithm='smoothquant' requires --static_ptq=True (W8A8 static activation quantization)."
            )
        if granularity == "per_block":
            raise ValueError(
                "--algorithm='smoothquant' is for W8A8 and does not support --granularity='per_block'; "
                "use 'per_channel' or 'per_tensor'."
            )

    if algorithm in ("awq", "gptq"):
        if bits not in (3, 4):
            raise ValueError(
                f"--algorithm='{algorithm}' is designed for low-bit weights (--bits=4 or --bits=3), "
                f"got --bits={bits}."
            )
        if granularity == "per_tensor":
            raise ValueError(
                f"--algorithm='{algorithm}' is incompatible with --granularity='per_tensor'; "
                "use 'per_channel' or 'per_block'."
            )

    if static_ptq and bits != 8:
        raise ValueError(
            f"--static_ptq=True (W8A8 static activation quantization) requires --bits=8, got --bits={bits}."
        )

    spec = QuantizationSpec(
        scheme=scheme,
        bits=bits,
        granularity=granularity,
        block_size=block_size,
        algorithm=algorithm,
        smooth_alpha=float(smooth_alpha),
        static_ptq=bool(static_ptq),
        calib_samples=int(calib_samples),
        calib_method=calib_method,
        calib_percentile=float(calib_percentile),
        ignore_patterns=normalize_ignore_modules(ignore_modules),
    )
    if spec.requires_calibration and spec.calib_samples <= 0:
        raise ValueError(
            f"--calib_samples must be > 0 when calibration is required (algorithm='{algorithm}', static_ptq={static_ptq})."
        )
    return spec


def build_weight_quant_args(
    spec: QuantizationSpec,
    target_modules: list[torch.nn.Linear] | None = None,
) -> QuantizationArgs:
    """Build compressed-tensors QuantizationArgs for weights."""
    if spec.granularity == "per_tensor":
        strategy = QuantizationStrategy.TENSOR
        extra = {}
    elif spec.granularity == "per_channel":
        strategy = QuantizationStrategy.CHANNEL
        extra = {}
    else:
        # Prefer 1D group quantization along in_features when divisible (standard for W4/W3 AWQ/GPTQ);
        # fall back to 2D block quantization if any targeted layer has non-divisible in_features.
        divisible = target_modules is None or all(
            m.in_features % spec.block_size == 0 for m in target_modules
        )
        if divisible:
            strategy = QuantizationStrategy.GROUP
            extra = {"group_size": spec.block_size}
        else:
            strategy = QuantizationStrategy.BLOCK
            extra = {"block_structure": [spec.block_size, spec.block_size]}

    return QuantizationArgs(
        num_bits=spec.bits,
        type=QuantizationType.INT,
        symmetric=spec.symmetric,
        strategy=strategy,
        dynamic=False,
        **extra,
    )


def build_activation_quant_args(spec: QuantizationSpec) -> QuantizationArgs | None:
    """Build compressed-tensors QuantizationArgs for static activation PTQ, or None."""
    if not spec.static_ptq:
        return None
    return QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        symmetric=spec.symmetric,
        strategy=QuantizationStrategy.TENSOR,
        dynamic=False,
    )


def build_compressed_tensors_config(
    spec: QuantizationSpec,
    target_modules: list[torch.nn.Linear] | None = None,
) -> QuantizationConfig:
    """Create a compressed-tensors QuantizationConfig from a validated QuantizationSpec."""
    w_args = build_weight_quant_args(spec, target_modules=target_modules)
    a_args = build_activation_quant_args(spec)
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=w_args,
        input_activations=a_args,
    )
    return QuantizationConfig(
        config_groups={"group_0": scheme},
        ignore=list(spec.ignore_patterns),
    )
