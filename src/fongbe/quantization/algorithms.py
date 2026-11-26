from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStatus,
    QuantizationStrategy,
    apply_quantization_config,
)
from compressed_tensors.quantization.lifecycle.forward import fake_quantize
from compressed_tensors.quantization.utils import (
    calculate_qparams,
    is_module_quantized,
    maybe_pad_tensor_for_block_quant,
)
from compressed_tensors.utils.match import match_named_modules
from torch import nn

from fongbe.quantization.calibrator import LayerActivationStats
from fongbe.quantization.schemes import QuantizationSpec, build_compressed_tensors_config


@dataclass(frozen=True)
class LayerQuantDiagnostics:
    layer_name: str
    weight_mse: float
    weight_sqnr_db: float
    max_abs_err: float


def compute_weight_qparams(
    weight: torch.Tensor,
    w_args: QuantizationArgs,
    clip_ratio: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute `(weight_scale, weight_zero_point)` matching compressed-tensors strategy."""
    w = weight.detach().to(torch.float32)
    strategy = w_args.strategy

    if strategy == QuantizationStrategy.TENSOR:
        min_vals = w.amin().reshape(1) * clip_ratio
        max_vals = w.amax().reshape(1) * clip_ratio
    elif strategy == QuantizationStrategy.CHANNEL:
        min_vals = w.amin(dim=1, keepdim=True) * clip_ratio
        max_vals = w.amax(dim=1, keepdim=True) * clip_ratio
    elif strategy == QuantizationStrategy.GROUP:
        gs = int(w_args.group_size)
        if w.shape[-1] % gs != 0:
            raise ValueError(
                f"Weight in_features={w.shape[-1]} must be divisible by group_size={gs}."
            )
        w_grp = w.unflatten(-1, (w.shape[-1] // gs, gs))
        min_vals = w_grp.amin(dim=-1) * clip_ratio
        max_vals = w_grp.amax(dim=-1) * clip_ratio
    elif strategy == QuantizationStrategy.BLOCK:
        bs = list(w_args.block_structure)
        w_pad = maybe_pad_tensor_for_block_quant(w, bs)
        nr = w_pad.shape[0] // bs[0]
        nc = w_pad.shape[1] // bs[1]
        w_blk = w_pad.reshape(nr, bs[0], nc, bs[1]).transpose(1, 2)
        min_vals = w_blk.amin(dim=(-2, -1)) * clip_ratio
        max_vals = w_blk.amax(dim=(-2, -1)) * clip_ratio
    else:
        raise ValueError(f"Unsupported weight quantization strategy: {strategy}")

    scale, zp = calculate_qparams(min_vals, max_vals, w_args)
    return scale.to(dtype=weight.dtype, device=weight.device), zp.to(device=weight.device)


def assign_weight_qparams(
    module: nn.Linear,
    scale: torch.Tensor,
    zp: torch.Tensor,
) -> None:
    """Write computed scale/zero-point parameters onto an initialized quantized module."""
    if hasattr(module, "weight_scale") and isinstance(module.weight_scale, torch.Tensor):
        module.weight_scale.data.copy_(scale.to(dtype=module.weight_scale.dtype, device=module.weight_scale.device))
    if hasattr(module, "weight_zero_point") and isinstance(module.weight_zero_point, torch.Tensor):
        module.weight_zero_point.data.copy_(zp.to(dtype=module.weight_zero_point.dtype, device=module.weight_zero_point.device))


def assign_activation_qparams(
    module: nn.Linear,
    stats_obj: LayerActivationStats,
    a_args: QuantizationArgs,
    calib_method: str = "percentile",
    calib_percentile: float = 99.9,
) -> None:
    """Compute and assign static activation scale/zero-point from calibration statistics."""
    low, high = stats_obj.activation_bounds(
        symmetric=a_args.symmetric,
        calib_method=calib_method,
        calib_percentile=calib_percentile,
    )
    dtype = module.input_scale.dtype if hasattr(module, "input_scale") else module.weight.dtype
    device = module.input_scale.device if hasattr(module, "input_scale") else module.weight.device
    min_t = torch.tensor([low], dtype=dtype, device=device)
    max_t = torch.tensor([high], dtype=dtype, device=device)
    scale, zp = calculate_qparams(min_t, max_t, a_args)
    if hasattr(module, "input_scale") and isinstance(module.input_scale, torch.Tensor):
        module.input_scale.data.copy_(scale.to(dtype=module.input_scale.dtype, device=module.input_scale.device))
    if hasattr(module, "input_zero_point") and isinstance(module.input_zero_point, torch.Tensor):
        module.input_zero_point.data.copy_(zp.to(dtype=module.input_zero_point.dtype, device=module.input_zero_point.device))


def find_fusable_pairs(
    model: nn.Module,
    target_modules: dict[str, nn.Linear],
) -> list[tuple[nn.Module | None, list[tuple[str, nn.Linear]]]]:
    """Group Parakeet/generic Linear modules with preceding fusable LayerNorm or v_proj layers."""
    named_mods = dict(model.named_modules())
    paired: set[str] = set()
    groups: list[tuple[nn.Module | None, list[tuple[str, nn.Linear]]]] = []

    # 1. ParakeetEncoderBlock / ConformerLayer structured pairings
    for prefix, mod in named_mods.items():
        if hasattr(mod, "norm_feed_forward1") and hasattr(mod, "feed_forward1"):
            lin_name = f"{prefix}.feed_forward1.linear1"
            if lin_name in target_modules and lin_name not in paired:
                groups.append((mod.norm_feed_forward1, [(lin_name, target_modules[lin_name])]))
                paired.add(lin_name)

        if hasattr(mod, "norm_feed_forward2") and hasattr(mod, "feed_forward2"):
            lin_name = f"{prefix}.feed_forward2.linear1"
            if lin_name in target_modules and lin_name not in paired:
                groups.append((mod.norm_feed_forward2, [(lin_name, target_modules[lin_name])]))
                paired.add(lin_name)

        if hasattr(mod, "norm_self_att") and hasattr(mod, "self_attn"):
            qkv = []
            for proj in ("q_proj", "k_proj", "v_proj"):
                lin_name = f"{prefix}.self_attn.{proj}"
                if lin_name in target_modules and lin_name not in paired:
                    qkv.append((lin_name, target_modules[lin_name]))
                    paired.add(lin_name)
            if qkv:
                groups.append((mod.norm_self_att, qkv))

            # v_proj -> o_proj when not already consumed or when v_proj is unquantized?
            # Note: scaling v_proj output by 1/s and o_proj input by s preserves output even when
            # v_proj is quantized afterwards (computed before weight quantization!).
            v_name = f"{prefix}.self_attn.v_proj"
            o_name = f"{prefix}.self_attn.o_proj"
            if o_name in target_modules and o_name not in paired and v_name in named_mods:
                v_mod = named_mods[v_name]
                if isinstance(v_mod, nn.Linear) and v_mod.out_features == target_modules[o_name].in_features:
                    groups.append((v_mod, [(o_name, target_modules[o_name])]))
                    paired.add(o_name)

    # 2. Generic nn.Sequential(LayerNorm, Linear) support (helpful for modular models & unit tests)
    for prefix, mod in named_mods.items():
        if isinstance(mod, nn.Sequential):
            children = list(mod.named_children())
            for idx in range(len(children) - 1):
                prev_sub = children[idx][1]
                cur_name, cur_sub = children[idx + 1]
                full_name = f"{prefix}.{cur_name}" if prefix else cur_name
                if (
                    isinstance(prev_sub, nn.LayerNorm)
                    and isinstance(cur_sub, nn.Linear)
                    and full_name in target_modules
                    and full_name not in paired
                ):
                    groups.append((prev_sub, [(full_name, cur_sub)]))
                    paired.add(full_name)

    # 3. Remaining standalone Linear layers (no fusable predecessor)
    for name, mod in target_modules.items():
        if name not in paired:
            groups.append((None, [(name, mod)]))

    return groups


def _scale_previous_module(prev_module: nn.Module, inv_scales: torch.Tensor) -> None:
    """Fold inverse channel scales `1 / s` into a preceding LayerNorm or Linear module."""
    with torch.no_grad():
        if isinstance(prev_module, nn.LayerNorm):
            if prev_module.weight is not None:
                prev_module.weight.data.mul_(inv_scales.to(dtype=prev_module.weight.dtype, device=prev_module.weight.device))
            if prev_module.bias is not None:
                prev_module.bias.data.mul_(inv_scales.to(dtype=prev_module.bias.dtype, device=prev_module.bias.device))
        elif isinstance(prev_module, nn.Linear):
            prev_module.weight.data.mul_(
                inv_scales.to(dtype=prev_module.weight.dtype, device=prev_module.weight.device).unsqueeze(1)
            )
            if prev_module.bias is not None:
                prev_module.bias.data.mul_(
                    inv_scales.to(dtype=prev_module.bias.dtype, device=prev_module.bias.device)
                )
        else:
            raise TypeError(f"Cannot fold scales into module of type {type(prev_module).__name__}.")


def compute_smoothquant_scales(
    weights: Iterable[torch.Tensor],
    act_absmax: torch.Tensor,
    alpha: float = 0.5,
    min_scale: float = 1e-4,
) -> torch.Tensor:
    """Compute SmoothQuant per-input-channel migration scales:
        s_j = max(|X_j|)^alpha / max(|W_j|)^(1 - alpha).
    """
    weight_absmax = None
    for w in weights:
        cur = w.detach().to(torch.float32).abs().amax(dim=0)
        weight_absmax = cur if weight_absmax is None else torch.maximum(weight_absmax, cur)

    if weight_absmax is None:
        raise ValueError("Expected at least one weight tensor for SmoothQuant.")

    act_max = act_absmax.detach().to(device=weight_absmax.device, dtype=torch.float32).clamp(min=1e-5)
    w_max = weight_absmax.clamp(min=1e-5)
    scales = (act_max.pow(alpha)) / (w_max.pow(1.0 - alpha))
    return scales.clamp(min=min_scale)


def apply_smoothquant_to_pair(
    linears: list[tuple[str, nn.Linear]],
    act_absmax: torch.Tensor,
    prev_module: nn.Module | None = None,
    alpha: float = 0.5,
    stats_map: dict[str, LayerActivationStats] | None = None,
) -> torch.Tensor | None:
    """Apply SmoothQuant scaling to a group of Linear layers sharing an input and optional predecessor."""
    if prev_module is None or not linears:
        return None

    scales = compute_smoothquant_scales([m.weight for _, m in linears], act_absmax, alpha=alpha)
    inv_scales = 1.0 / scales

    _scale_previous_module(prev_module, inv_scales)
    with torch.no_grad():
        for name, lin in linears:
            s_dev = scales.to(dtype=lin.weight.dtype, device=lin.weight.device)
            lin.weight.data.mul_(s_dev.unsqueeze(0))
            if stats_map is not None and name in stats_map:
                st = stats_map[name]
                inv_cpu = inv_scales.to(device="cpu", dtype=torch.float32)
                if st.channel_absmax is not None:
                    st.channel_absmax = st.channel_absmax * inv_cpu
                if st.channel_abs_sum is not None:
                    st.channel_abs_sum = st.channel_abs_sum * inv_cpu
                if st.sample_inputs:
                    st.sample_inputs = [x * inv_cpu.unsqueeze(0) for x in st.sample_inputs]
                    all_x = torch.cat(st.sample_inputs, dim=0)
                    st.min_val = float(all_x.min().item())
                    st.max_val = float(all_x.max().item())
                    st.absmax = max(abs(st.min_val), abs(st.max_val))
                    st.reservoir = [all_x.reshape(-1)]
    return scales


def apply_awq_to_pair(
    linears: list[tuple[str, nn.Linear]],
    stats_map: dict[str, LayerActivationStats],
    w_args: QuantizationArgs,
    prev_module: nn.Module | None = None,
) -> None:
    """Apply Activation-Aware Weight Quantization (AWQ) salience scaling or clipping search."""
    first_name, _ = linears[0]
    st = stats_map.get(first_name)
    act_mean = st.channel_mean_abs() if st is not None else None
    sample_x = st.get_sample_matrix() if st is not None else None

    if prev_module is not None and act_mean is not None:
        act_salience = act_mean.clamp(min=1e-5)
        best_alpha = 0.0
        best_scales = torch.ones_like(act_salience)
        best_err = float("inf")

        with torch.no_grad():
            for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
                raw_s = act_salience.pow(alpha)
                norm_factor = math.sqrt(float(raw_s.max().item()) * float(raw_s.min().item()))
                s = (raw_s / max(norm_factor, 1e-6)).clamp(min=1e-4, max=1e4)

                total_err = 0.0
                for _, lin in linears:
                    w = lin.weight.detach().to(torch.float32)
                    s_w = s.to(device=w.device, dtype=torch.float32)
                    w_scaled = w * s_w.unsqueeze(0)
                    scale, zp = compute_weight_qparams(w_scaled, w_args)
                    w_q = fake_quantize(w_scaled, scale, zp, w_args) / s_w.unsqueeze(0)
                    diff = w - w_q
                    if sample_x is not None:
                        x_dev = sample_x.to(device=w.device, dtype=torch.float32)
                        out_diff = x_dev @ diff.T
                        total_err += float(torch.mean(out_diff.pow(2)).item())
                    else:
                        sal = act_salience.to(device=w.device, dtype=torch.float32)
                        total_err += float(torch.mean((diff * sal.unsqueeze(0)).pow(2)).item())

                if total_err < best_err:
                    best_err = total_err
                    best_alpha = alpha
                    best_scales = s

            if best_alpha > 0.0:
                _scale_previous_module(prev_module, 1.0 / best_scales)
                for _, lin in linears:
                    lin.weight.data.mul_(
                        best_scales.to(dtype=lin.weight.dtype, device=lin.weight.device).unsqueeze(0)
                    )

        for _, lin in linears:
            scale, zp = compute_weight_qparams(lin.weight, w_args)
            assign_weight_qparams(lin, scale, zp)
        return

    # Non-fusable layers: salience-weighted clipping search over weight range
    with torch.no_grad():
        for name, lin in linears:
            st_cur = stats_map.get(name)
            x_cur = st_cur.get_sample_matrix() if st_cur is not None else None
            w = lin.weight.detach().to(torch.float32)
            best_ratio = 1.0
            best_err = float("inf")
            for ratio in (1.0, 0.95, 0.9, 0.85, 0.8):
                scale, zp = compute_weight_qparams(lin.weight, w_args, clip_ratio=ratio)
                w_q = fake_quantize(w, scale.to(torch.float32), zp, w_args)
                diff = w - w_q
                if x_cur is not None:
                    x_dev = x_cur.to(device=w.device, dtype=torch.float32)
                    err = float(torch.mean((x_dev @ diff.T).pow(2)).item())
                else:
                    err = float(torch.mean(diff.pow(2)).item())
                if err < best_err:
                    best_err = err
                    best_ratio = ratio
            scale, zp = compute_weight_qparams(lin.weight, w_args, clip_ratio=best_ratio)
            assign_weight_qparams(lin, scale, zp)


@torch.no_grad()
def apply_gptq(
    module: nn.Linear,
    hessian: torch.Tensor | None,
    w_args: QuantizationArgs,
    damp_percent: float = 0.01,
    col_blocksize: int = 128,
) -> None:
    """Apply Hessian-based column-wise GPTQ with error compensation to `module.weight`."""
    w_orig = module.weight.data
    w = w_orig.detach().to(torch.float32).clone()
    out_features, in_features = w.shape

    if hessian is None:
        scale, zp = compute_weight_qparams(w, w_args)
        assign_weight_qparams(module, scale, zp)
        return

    h = hessian.detach().to(device=w.device, dtype=torch.float32).clone()
    dead = torch.diag(h) == 0
    h[dead, dead] = 1.0
    w[:, dead] = 0.0

    # Compute initial scales/zero-points according to target strategy
    scale, zp = compute_weight_qparams(w, w_args)

    diag_mean = torch.mean(torch.diag(h))
    h_inv_chol = None
    for factor in (1.0, 10.0, 100.0):
        h_damped = h.clone()
        damp = max(damp_percent * factor * float(diag_mean.item()), 1e-6)
        idx = torch.arange(in_features, device=w.device)
        h_damped[idx, idx] += damp
        try:
            l_chol = torch.linalg.cholesky(h_damped)
            h_inv = torch.cholesky_inverse(l_chol)
            h_inv_chol = torch.linalg.cholesky(h_inv, upper=True)
            break
        except RuntimeError:
            continue

    if h_inv_chol is None:
        assign_weight_qparams(module, scale, zp)
        return

    # Scalar per-tensor args for per-column fake quantization
    col_args = QuantizationArgs(
        num_bits=w_args.num_bits,
        type=w_args.type,
        symmetric=w_args.symmetric,
        strategy=QuantizationStrategy.CHANNEL,
        dynamic=False,
    )

    q_weight = torch.zeros_like(w)
    for i1 in range(0, in_features, col_blocksize):
        i2 = min(i1 + col_blocksize, in_features)
        count = i2 - i1

        w1 = w[:, i1:i2].clone()
        q1 = torch.zeros_like(w1)
        err1 = torch.zeros_like(w1)
        h_inv1 = h_inv_chol[i1:i2, i1:i2]

        for i in range(count):
            col_idx = i1 + i
            col_w = w1[:, i].unsqueeze(1)
            d = h_inv1[i, i]

            if w_args.strategy == QuantizationStrategy.TENSOR:
                col_s = scale.reshape(1, 1).expand(out_features, 1)
                col_zp = zp.reshape(1, 1).expand(out_features, 1)
            elif w_args.strategy == QuantizationStrategy.CHANNEL:
                col_s = scale.reshape(out_features, 1)
                col_zp = zp.reshape(out_features, 1)
            elif w_args.strategy == QuantizationStrategy.GROUP:
                g_idx = col_idx // int(w_args.group_size)
                col_s = scale[:, g_idx : g_idx + 1]
                col_zp = zp[:, g_idx : g_idx + 1]
            elif w_args.strategy == QuantizationStrategy.BLOCK:
                bh, bw = w_args.block_structure
                gc = col_idx // bw
                row_groups = torch.arange(out_features, device=w.device) // bh
                col_s = scale[row_groups, gc].unsqueeze(1)
                col_zp = zp[row_groups, gc].unsqueeze(1)
            else:
                raise ValueError(f"Unsupported strategy for GPTQ: {w_args.strategy}")

            q_col = fake_quantize(col_w, col_s.to(torch.float32), col_zp, col_args).squeeze(1)
            q1[:, i] = q_col
            err = (w1[:, i] - q_col) / d
            w1[:, i:] -= err.unsqueeze(1) * h_inv1[i, i:].unsqueeze(0)
            err1[:, i] = err

        q_weight[:, i1:i2] = q1
        if i2 < in_features:
            w[:, i2:] -= err1 @ h_inv_chol[i1:i2, i2:]

    module.weight.data.copy_(q_weight.to(dtype=w_orig.dtype, device=w_orig.device))
    assign_weight_qparams(module, scale.to(dtype=w_orig.dtype), zp)


def apply_quantization_algorithm(
    model: nn.Module,
    spec: QuantizationSpec,
    stats_map: dict[str, LayerActivationStats] | None = None,
    calib_batches: Iterable[dict[str, torch.Tensor]] | None = None,
    device: torch.device | str = "cpu",
) -> tuple[dict[str, torch.Tensor], list[LayerQuantDiagnostics]]:
    """Apply configured quantization algorithm (`absmax`, `awq`, `gptq`, `smoothquant`)
    and optional static activation PTQ scales, returning `(ref_weights, layer_diagnostics)`.
    """
    from fongbe.quantization.calibrator import collect_activation_stats

    # Discover matching Linear modules first so per_block group/block strategy is chosen once
    matched_linears = [
        mod
        for _, mod in match_named_modules(model, ["Linear"], spec.ignore_patterns, warn_on_fail=False)
        if isinstance(mod, nn.Linear)
    ]
    q_config = build_compressed_tensors_config(spec, target_modules=matched_linears)
    apply_quantization_config(model, q_config, run_compressed=False, show_progress=False)

    target_modules: dict[str, nn.Linear] = {
        name: mod
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and is_module_quantized(mod)
    }

    ref_weights = {
        name: mod.weight.detach().to(device="cpu", dtype=torch.float32).clone()
        for name, mod in target_modules.items()
    }

    if spec.requires_calibration and stats_map is None:
        if calib_batches is None:
            raise ValueError(
                f"Calibration batches or stats_map required for algorithm='{spec.algorithm}', static_ptq={spec.static_ptq}."
            )
        stats_map = collect_activation_stats(
            model=model,
            spec=spec,
            calib_batches=calib_batches,
            target_modules=target_modules,
            device=device,
        )

    stats_map = stats_map or {}
    pairs = find_fusable_pairs(model, target_modules)

    if spec.algorithm == "smoothquant":
        for prev_mod, linears in pairs:
            if prev_mod is not None:
                first_name = linears[0][0]
                st = stats_map.get(first_name)
                if st is not None and st.channel_absmax is not None:
                    apply_smoothquant_to_pair(
                        linears=linears,
                        act_absmax=st.channel_absmax,
                        prev_module=prev_mod,
                        alpha=spec.smooth_alpha,
                        stats_map=stats_map,
                    )
        # Re-calibrate smoothed activations if batches are available and percentile calibration is requested
        if calib_batches is not None and spec.calib_method == "percentile":
            stats_map = collect_activation_stats(
                model=model,
                spec=spec,
                calib_batches=calib_batches,
                target_modules=target_modules,
                device=device,
            )

        for name, mod in target_modules.items():
            scheme = mod.quantization_scheme
            scale, zp = compute_weight_qparams(mod.weight, scheme.weights)
            assign_weight_qparams(mod, scale, zp)

    elif spec.algorithm == "awq":
        for prev_mod, linears in pairs:
            scheme = linears[0][1].quantization_scheme
            apply_awq_to_pair(
                linears=linears,
                stats_map=stats_map,
                w_args=scheme.weights,
                prev_module=prev_mod,
            )

    elif spec.algorithm == "gptq":
        for name, mod in target_modules.items():
            st = stats_map.get(name)
            h = st.get_hessian() if st is not None else None
            apply_gptq(mod, h, mod.quantization_scheme.weights)

    else:  # absmax
        for name, mod in target_modules.items():
            scale, zp = compute_weight_qparams(mod.weight, mod.quantization_scheme.weights)
            assign_weight_qparams(mod, scale, zp)

    # Static activation quantization (W8A8)
    for name, mod in target_modules.items():
        scheme = mod.quantization_scheme
        if spec.static_ptq and scheme.input_activations is not None:
            st = stats_map.get(name)
            if st is None:
                raise ValueError(f"Missing activation calibration stats for layer '{name}'.")
            assign_activation_qparams(
                mod,
                st,
                scheme.input_activations,
                calib_method=spec.calib_method,
                calib_percentile=spec.calib_percentile,
            )
        mod.quantization_status = QuantizationStatus.FROZEN

    diagnostics = compute_layer_diagnostics(ref_weights, target_modules)
    return ref_weights, diagnostics


@torch.no_grad()
def compute_layer_diagnostics(
    ref_weights: dict[str, torch.Tensor],
    target_modules: dict[str, nn.Linear],
) -> list[LayerQuantDiagnostics]:
    """Compute per-layer weight MSE, SQNR (dB), and max absolute error against FP reference weights."""
    diagnostics: list[LayerQuantDiagnostics] = []
    for name, w_ref in ref_weights.items():
        mod = target_modules[name]
        w_cur = mod.weight.detach().to(device="cpu", dtype=torch.float32)
        scheme = getattr(mod, "quantization_scheme", None)
        if scheme is not None and scheme.weights is not None:
            scale = mod.weight_scale.detach().to(device="cpu", dtype=torch.float32)
            zp = (
                mod.weight_zero_point.detach().to(device="cpu")
                if hasattr(mod, "weight_zero_point") and mod.weight_zero_point is not None
                else None
            )
            w_q = fake_quantize(w_cur, scale, zp, scheme.weights)
        else:
            w_q = w_cur

        diff = w_ref.to(torch.float32) - w_q
        mse = float(torch.mean(diff.pow(2)).item())
        sig_power = float(torch.mean(w_ref.to(torch.float32).pow(2)).item())
        sqnr_db = 10.0 * math.log10(sig_power / max(mse, 1e-12)) if sig_power > 0.0 else 0.0
        max_err = float(torch.max(torch.abs(diff)).item())
        diagnostics.append(
            LayerQuantDiagnostics(
                layer_name=name,
                weight_mse=mse,
                weight_sqnr_db=sqnr_db,
                max_abs_err=max_err,
            )
        )
    return diagnostics
