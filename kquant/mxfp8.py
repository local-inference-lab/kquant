"""CPU helpers for vLLM's serialized MXFP8 dense-weight format."""

from __future__ import annotations

import torch

MXFP8_BLOCK_SIZE = 32
MXFP8_E4M3_MAX = 448.0


def mxfp8_quantize_cpu(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a rank-2 weight to E4M3 values and per-32 E8M0 scales.

    The returned layout is the one consumed by vLLM's serialized MXFP8
    linear method: ``weight`` remains ``[out_features, in_features]`` and
    ``weight_scale`` is ``uint8[out_features, in_features / 32]``.  Each
    scale byte stores the power-of-two exponent with a bias of 127.
    """

    if weight.ndim != 2:
        raise ValueError(
            f"MXFP8 requires a rank-2 weight, got shape={tuple(weight.shape)}"
        )
    if not weight.dtype.is_floating_point:
        raise TypeError(f"MXFP8 requires a floating-point weight, got {weight.dtype}")

    out_features, in_features = weight.shape
    if in_features % MXFP8_BLOCK_SIZE:
        raise ValueError(
            "MXFP8 input dimension must be divisible by "
            f"{MXFP8_BLOCK_SIZE}, got {in_features}"
        )

    values = weight.detach().to(device="cpu", dtype=torch.float32).contiguous()
    blocks = values.view(
        out_features, in_features // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE
    )
    amax = blocks.abs().amax(dim=-1)

    # Smallest power of two for which amax / 2**exponent fits in E4M3.
    exponent = torch.ceil(torch.log2(amax.clamp(min=1e-30) / MXFP8_E4M3_MAX)).clamp(
        min=-127, max=128
    )
    scale = (exponent + 127.0).to(torch.uint8)
    quantized = (
        blocks.div(torch.exp2(exponent).unsqueeze(-1))
        .clamp(-MXFP8_E4M3_MAX, MXFP8_E4M3_MAX)
        .view(out_features, in_features)
        .to(torch.float8_e4m3fn)
    )
    return quantized, scale


def mxfp8_dequantize_cpu(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize the serialized MXFP8 layout, primarily for validation."""

    if weight.ndim != 2:
        raise ValueError(
            f"MXFP8 requires a rank-2 weight, got shape={tuple(weight.shape)}"
        )
    out_features, in_features = weight.shape
    expected_scale_shape = (out_features, in_features // MXFP8_BLOCK_SIZE)
    if in_features % MXFP8_BLOCK_SIZE or tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            f"invalid MXFP8 scale shape {tuple(scale.shape)} for weight "
            f"shape {tuple(weight.shape)}; expected {expected_scale_shape}"
        )
    if scale.dtype != torch.uint8:
        raise TypeError(f"MXFP8 scales must be uint8, got {scale.dtype}")

    exponent = scale.to(device="cpu", dtype=torch.int16).sub(127).to(torch.float32)
    blocks = weight.to(device="cpu", dtype=torch.float32).view(
        out_features, in_features // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE
    )
    return (
        blocks.mul(torch.exp2(exponent).unsqueeze(-1))
        .view(out_features, in_features)
        .to(dtype)
    )
