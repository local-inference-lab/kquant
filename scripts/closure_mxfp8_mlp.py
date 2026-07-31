#!/usr/bin/env python
"""Identical-input logical-TP closure for K3 serialized MXFP8 MLPs.

This exercises the production ``sparkinfer.gemm.mxfp8_linear`` path while
running every logical TP rank sequentially on one GPU.  It separates:

* column-parallel gate/up sharding;
* row-parallel down-projection sharding;
* the BF16 rounding of each local down-projection partial; and
* an FP32 dequantized oracle using the exact dynamically quantized operands.

The primary purpose is to determine whether cross-TP drift comes from
checkpoint slicing, the production kernel, or rank-boundary-dependent
rounding/reduction.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kquant.correctness import write_json
from kquant.tp_simulator import comparison_metrics, situ

MXFP8_BLOCK_SIZE = 32


class IndexedTensorStore:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]

    def get(self, name: str) -> torch.Tensor:
        filename = self.weight_map.get(name)
        if filename is None:
            raise KeyError(name)
        with safe_open(str(self.model_dir / filename), framework="pt") as handle:
            return handle.get_tensor(name)


def _tensor_from_trace(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(
        payload.get("tensor"), torch.Tensor
    ):
        raise TypeError(f"{path} is not a Kimi correctness trace payload")
    return payload["tensor"]


def _sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(raw).hexdigest()


def _stats(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "norm": float(value.norm()),
        "max_abs": float(value.abs().max()),
        "sha256": _sha256(tensor),
    }


def _load_config(model_dir: Path) -> dict[str, Any]:
    config = json.loads((model_dir / "config.json").read_text())
    return config.get("text_config", config)


def _projection_prefix(layer: int, kind: str) -> str:
    if kind == "dense":
        return f"language_model.model.layers.{layer}.mlp"
    if kind == "shared":
        return f"language_model.model.layers.{layer}.block_sparse_moe.shared_experts"
    raise ValueError(kind)


def _load_projection(
    store: IndexedTensorStore,
    prefix: str,
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        projection: {
            "weight": store.get(f"{prefix}.{projection}_proj.weight"),
            "scale": store.get(f"{prefix}.{projection}_proj.weight_scale"),
        }
        for projection in ("gate", "up", "down")
    }


def _validate_serialized_mxfp8(
    projection: dict[str, dict[str, torch.Tensor]],
) -> tuple[int, int]:
    gate = projection["gate"]["weight"]
    up = projection["up"]["weight"]
    down = projection["down"]["weight"]
    if gate.dtype != torch.float8_e4m3fn:
        raise ValueError(f"gate weight must be e4m3, got {gate.dtype}")
    if up.dtype != torch.float8_e4m3fn or down.dtype != torch.float8_e4m3fn:
        raise ValueError("all MXFP8 weights must be float8_e4m3fn")
    if any(value["scale"].dtype != torch.uint8 for value in projection.values()):
        raise ValueError("all MXFP8 scales must be uint8")
    if gate.shape != up.shape:
        raise ValueError(
            f"gate/up shapes differ: {tuple(gate.shape)} vs {tuple(up.shape)}"
        )
    intermediate_size, hidden_size = map(int, gate.shape)
    if tuple(down.shape) != (hidden_size, intermediate_size):
        raise ValueError(
            "down shape does not transpose gate geometry: "
            f"{tuple(down.shape)} vs {(hidden_size, intermediate_size)}"
        )
    for name, value in projection.items():
        weight = value["weight"]
        scale = value["scale"]
        expected = (int(weight.shape[0]), int(weight.shape[1]) // 32)
        if tuple(scale.shape) != expected:
            raise ValueError(f"{name} scale shape {tuple(scale.shape)} != {expected}")
    return hidden_size, intermediate_size


def _pad_last(tensor: torch.Tensor, width: int) -> torch.Tensor:
    if int(tensor.shape[-1]) == width:
        return tensor.contiguous()
    if int(tensor.shape[-1]) > width:
        raise ValueError("cannot pad to a smaller width")
    return torch.nn.functional.pad(
        tensor, (0, width - int(tensor.shape[-1]))
    ).contiguous()


def _reference_from_packed(
    source: torch.Tensor,
    packed_weight: Any,
    *,
    block_fp8_linear: Any,
    dequantize_rows: Any,
) -> torch.Tensor:
    """FP32 oracle over the exact MXFP8 operands consumed by the kernel."""
    padded_source = _pad_last(source, int(packed_weight.padded_in_features))
    source_quant = block_fp8_linear.quantize_input(padded_source)
    source_dequant = dequantize_rows(source_quant.values, source_quant.scale_rows)
    weight_dequant = dequantize_rows(
        packed_weight.weight.values, packed_weight.weight.scale_rows
    )
    return source_dequant @ weight_dequant.T


def _production_linear(
    source: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    mxfp8_linear: Any,
    block_fp8_linear: Any,
    dequantize_rows: Any,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    packed = mxfp8_linear.pack_weight(weight.contiguous(), scale.contiguous())
    actual = mxfp8_linear.mm(
        source.contiguous(),
        packed,
        expected_m=int(source.shape[0]),
    )
    reference_fp32 = _reference_from_packed(
        source,
        packed,
        block_fp8_linear=block_fp8_linear,
        dequantize_rows=dequantize_rows,
    )
    rounded_reference = reference_fp32.to(actual.dtype)
    metrics = comparison_metrics(rounded_reference, actual)
    metrics["bitwise_equal"] = bool(torch.equal(rounded_reference, actual))
    return actual, reference_fp32, metrics


def _max_kernel_metrics(
    values: list[dict[str, float]],
) -> dict[str, Any]:
    return {
        "calls": len(values),
        "all_bitwise_equal": all(bool(value["bitwise_equal"]) for value in values),
        "worst_relative_l2": max(float(value["relative_l2"]) for value in values),
        "worst_max_abs": max(float(value["max_abs"]) for value in values),
        "minimum_cosine": min(float(value["cosine"]) for value in values),
    }


def _rank_trace_path(
    trace_root: Path,
    rank: int,
    layer: int,
    stage: str,
) -> Path:
    return (
        trace_root / f"tp-rank-{rank:03d}" / f"layer-{layer:03d}.call-000000.{stage}.pt"
    )


def _trace_rank_count(trace_root: Path) -> int:
    ranks = sorted(trace_root.glob("tp-rank-*"))
    if not ranks:
        raise ValueError(f"no tp-rank-* directories under {trace_root}")
    return len(ranks)


def _run_tp(
    *,
    tp_size: int,
    source: torch.Tensor,
    projection: dict[str, dict[str, torch.Tensor]],
    intermediate_size: int,
    beta: float,
    linear_beta: float,
    device: torch.device,
    mxfp8_linear: Any,
    block_fp8_linear: Any,
    dequantize_rows: Any,
    trace_root: Path | None,
    layer: int,
    trace_stage: str,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if intermediate_size % tp_size != 0:
        raise ValueError(
            f"intermediate size {intermediate_size} is not divisible by TP{tp_size}"
        )
    local_size = intermediate_size // tp_size
    if local_size % MXFP8_BLOCK_SIZE != 0:
        raise ValueError(f"local intermediate {local_size} is not MXFP8-group aligned")

    gate_outputs = []
    up_outputs = []
    activations = []
    down_partials = []
    down_reference_partials = []
    gate_kernel_metrics = []
    down_kernel_metrics = []
    trace_metrics = []

    for rank in range(tp_size):
        row_start = rank * local_size
        row_end = row_start + local_size
        scale_start = row_start // MXFP8_BLOCK_SIZE
        scale_end = row_end // MXFP8_BLOCK_SIZE

        gate_weight = projection["gate"]["weight"][row_start:row_end].to(device)
        up_weight = projection["up"]["weight"][row_start:row_end].to(device)
        gate_scale = projection["gate"]["scale"][row_start:row_end].to(device)
        up_scale = projection["up"]["scale"][row_start:row_end].to(device)
        fused_weight = torch.cat((gate_weight, up_weight), dim=0)
        fused_scale = torch.cat((gate_scale, up_scale), dim=0)
        gate_up, _, gate_metrics = _production_linear(
            source,
            fused_weight,
            fused_scale,
            mxfp8_linear=mxfp8_linear,
            block_fp8_linear=block_fp8_linear,
            dequantize_rows=dequantize_rows,
        )
        gate_kernel_metrics.append(gate_metrics)
        gate, up = gate_up.split(local_size, dim=-1)
        activation = situ(gate, up, beta=beta, linear_beta=linear_beta)

        down_weight = projection["down"]["weight"][:, row_start:row_end].to(device)
        down_scale = projection["down"]["scale"][:, scale_start:scale_end].to(device)
        down, down_reference, down_metrics = _production_linear(
            activation,
            down_weight,
            down_scale,
            mxfp8_linear=mxfp8_linear,
            block_fp8_linear=block_fp8_linear,
            dequantize_rows=dequantize_rows,
        )
        down_kernel_metrics.append(down_metrics)

        if trace_root is not None:
            traced = _tensor_from_trace(
                _rank_trace_path(trace_root, rank, layer, trace_stage)
            ).to(device)
            trace_metrics.append(
                {
                    "rank": rank,
                    **comparison_metrics(traced, down),
                    "bitwise_equal": bool(torch.equal(traced, down)),
                }
            )

        gate_outputs.append(gate)
        up_outputs.append(up)
        activations.append(activation)
        down_partials.append(down)
        down_reference_partials.append(down_reference)

        del (
            gate_weight,
            up_weight,
            gate_scale,
            up_scale,
            fused_weight,
            fused_scale,
            down_weight,
            down_scale,
        )

    gate_full = torch.cat(gate_outputs, dim=-1)
    up_full = torch.cat(up_outputs, dim=-1)
    activation_full = torch.cat(activations, dim=-1)
    down_kernel_fp32_sum = torch.stack([value.float() for value in down_partials]).sum(
        0
    )
    down_kernel_bf16_sum = down_kernel_fp32_sum.to(torch.bfloat16)
    down_reference_fp32_sum = torch.stack(down_reference_partials).sum(0)
    down_reference_bf16_sum = down_reference_fp32_sum.to(torch.bfloat16)

    partial_norm_sum = sum(float(value.float().norm()) for value in down_partials)
    reduced_norm = float(down_kernel_fp32_sum.norm())
    result: dict[str, Any] = {
        "tp_size": tp_size,
        "local_intermediate_size": local_size,
        "gate_up_kernel": _max_kernel_metrics(gate_kernel_metrics),
        "down_kernel": _max_kernel_metrics(down_kernel_metrics),
        "down_partial_rounding": comparison_metrics(
            down_reference_fp32_sum, down_kernel_fp32_sum
        ),
        "down_partial_cancellation": {
            "sum_of_partial_norms": partial_norm_sum,
            "reduced_norm": reduced_norm,
            "sum_norm_over_reduced_norm": (
                partial_norm_sum / reduced_norm if reduced_norm != 0.0 else float("inf")
            ),
        },
        "gate": _stats(gate_full),
        "up": _stats(up_full),
        "activation": _stats(activation_full),
        "down_kernel_fp32_sum": _stats(down_kernel_fp32_sum),
        "down_kernel_bf16_sum": _stats(down_kernel_bf16_sum),
        "down_reference_fp32_sum": _stats(down_reference_fp32_sum),
        "down_reference_bf16_sum": _stats(down_reference_bf16_sum),
    }
    if trace_metrics:
        result["trace_partial_validation"] = {
            "all_bitwise_equal": all(
                bool(value["bitwise_equal"]) for value in trace_metrics
            ),
            "worst_relative_l2": max(
                float(value["relative_l2"]) for value in trace_metrics
            ),
            "worst_max_abs": max(float(value["max_abs"]) for value in trace_metrics),
            "ranks": trace_metrics,
        }

    tensors = {
        "gate": gate_full,
        "up": up_full,
        "activation": activation_full,
        "down_kernel_fp32_sum": down_kernel_fp32_sum,
        "down_kernel_bf16_sum": down_kernel_bf16_sum,
        "down_reference_fp32_sum": down_reference_fp32_sum,
        "down_reference_bf16_sum": down_reference_bf16_sum,
    }
    return result, tensors


def _cross_tp(
    reference: dict[str, torch.Tensor],
    actual: dict[str, torch.Tensor],
) -> dict[str, dict[str, Any]]:
    stages = (
        "gate",
        "up",
        "activation",
        "down_kernel_fp32_sum",
        "down_kernel_bf16_sum",
        "down_reference_fp32_sum",
        "down_reference_bf16_sum",
    )
    return {
        stage: {
            **comparison_metrics(reference[stage], actual[stage]),
            "bitwise_equal": bool(torch.equal(reference[stage], actual[stage])),
        }
        for stage in stages
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--kind", choices=("dense", "shared"), required=True)
    parser.add_argument(
        "--input-trace",
        type=Path,
        required=True,
        help="Kimi trace payload containing the identical MLP input",
    )
    parser.add_argument(
        "--tp",
        type=int,
        nargs="+",
        default=[1, 4, 12, 16],
    )
    parser.add_argument(
        "--trace-root",
        type=Path,
        help=(
            "Optional physical trace root. Its rank count must match one "
            "requested TP and local down partials are checked against it."
        ),
    )
    parser.add_argument(
        "--trace-stage",
        default="shared_output_partial",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("the production MXFP8 closure requires CUDA")

    from sparkinfer.gemm import block_fp8_linear, mxfp8_linear
    from sparkinfer.gemm._shared.wo_mxfp8 import (
        dequantize_mxfp8_rows_torch,
    )

    config = _load_config(args.model)
    store = IndexedTensorStore(args.model)
    prefix = _projection_prefix(args.layer, args.kind)
    projection = _load_projection(store, prefix)
    hidden_size, intermediate_size = _validate_serialized_mxfp8(projection)
    source = _tensor_from_trace(args.input_trace).to(device).contiguous()
    if int(source.shape[-1]) != hidden_size:
        raise ValueError(f"input width {source.shape[-1]} != hidden size {hidden_size}")

    beta = float(config.get("activation_situ_beta", 1.0))
    linear_beta = float(config.get("activation_situ_linear_beta", 25.0))
    requested_tp = list(dict.fromkeys(args.tp))
    trace_tp = (
        _trace_rank_count(args.trace_root) if args.trace_root is not None else None
    )
    if trace_tp is not None and trace_tp not in requested_tp:
        raise ValueError(
            f"trace root has TP{trace_tp}, not in requested {requested_tp}"
        )

    report: dict[str, Any] = {
        "status": "running",
        "model": str(args.model.resolve()),
        "layer": args.layer,
        "kind": args.kind,
        "prefix": prefix,
        "device": str(device),
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "activation": {
            "name": "situ",
            "beta": beta,
            "linear_beta": linear_beta,
        },
        "input": {
            "path": str(args.input_trace.resolve()),
            **_stats(source),
        },
        "tp_sizes": requested_tp,
        "runs": {},
        "comparisons": {},
    }

    tensors_by_tp: dict[int, dict[str, torch.Tensor]] = {}
    with torch.inference_mode():
        for tp_size in requested_tp:
            result, tensors = _run_tp(
                tp_size=tp_size,
                source=source,
                projection=projection,
                intermediate_size=intermediate_size,
                beta=beta,
                linear_beta=linear_beta,
                device=device,
                mxfp8_linear=mxfp8_linear,
                block_fp8_linear=block_fp8_linear,
                dequantize_rows=dequantize_mxfp8_rows_torch,
                trace_root=(args.trace_root if trace_tp == tp_size else None),
                layer=args.layer,
                trace_stage=args.trace_stage,
            )
            report["runs"][str(tp_size)] = result
            tensors_by_tp[tp_size] = tensors
            torch.cuda.synchronize(device)

        for reference_tp, actual_tp in itertools.combinations(requested_tp, 2):
            report["comparisons"][f"tp{reference_tp}_vs_tp{actual_tp}"] = _cross_tp(
                tensors_by_tp[reference_tp], tensors_by_tp[actual_tp]
            )

    report["status"] = "pass"
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
