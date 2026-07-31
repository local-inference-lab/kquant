"""Server-log evidence that a correctness run exercised the intended kernels."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

STOCK_W4A16 = "stock-w4a16"
HYBRID_EXL3 = "hybrid-exl3"
KERNEL_PATHS = (STOCK_W4A16, HYBRID_EXL3)

_REQUIRED: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    STOCK_W4A16: {
        "b12x_experts": ("Using B12xExperts",),
        "forced_w4a16": (
            "B12X MoE force-A16 enabled: using quant_mode=w4a16",
        ),
        "ordinary_w4a16_kernel": ("W4A16FusedMoeKernel",),
        "mma_packed_weights": (
            "weight_layout': 'packed'",
            '"weight_layout":"packed"',
        ),
        "repeat_check_w4a16": (
            "implementation=w4a16",
        ),
    },
    HYBRID_EXL3: {
        "hybrid_quantization": ("quantization=nvfp4_nf3_hybrid",),
        "exl3_checkpoint_tensors": ("w13_exl3_trellis",),
        "w4a16_kernel_family": ("W4A16FusedMoeKernel",),
        "exl3_trellis_layout": ("trellis3_t256",),
        "repeat_check_w4a16": ("implementation=w4a16",),
    },
}

_FORBIDDEN: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    STOCK_W4A16: {
        "hybrid_quantization": ("quantization=nvfp4_nf3_hybrid",),
        "exl3_trellis_layout": ("trellis3_t256",),
        "hybrid_one_grid": (
            "executing hybrid one-grid",
            "executing K3 TP16 one-grid",
        ),
    },
    HYBRID_EXL3: {},
}


def audit_kernel_path(log_text: str, required_path: str) -> dict[str, Any]:
    """Return machine-readable positive and negative kernel-path evidence."""
    if required_path not in KERNEL_PATHS:
        raise ValueError(
            f"unknown kernel path {required_path!r}; expected one of {KERNEL_PATHS}"
        )
    required = {}
    for label, alternatives in _REQUIRED[required_path].items():
        matches = {
            value: log_text.count(value)
            for value in alternatives
            if value in log_text
        }
        required[label] = {
            "pass": bool(matches),
            "matches": matches,
        }
    forbidden = {}
    for label, alternatives in _FORBIDDEN[required_path].items():
        matches = {
            value: log_text.count(value)
            for value in alternatives
            if value in log_text
        }
        forbidden[label] = {
            "pass": not matches,
            "matches": matches,
        }
    missing = [label for label, value in required.items() if not value["pass"]]
    unexpected = [
        label for label, value in forbidden.items() if not value["pass"]
    ]
    return {
        "required_path": required_path,
        "pass": not missing and not unexpected,
        "required_evidence": required,
        "forbidden_evidence": forbidden,
        "missing_evidence": missing,
        "unexpected_evidence": unexpected,
    }
