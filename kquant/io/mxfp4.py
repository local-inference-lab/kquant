"""Codec for compressed-tensors ``mxfp4-pack-quantized`` expert weights.

Storage (verified against the real checkpoint):
  weight_packed: uint8 [out, K/2], two E2M1 codes per byte, low nibble = even
                 K index (compressed-tensors pack_fp4 convention);
  weight_scale:  uint8 [out, K/32], E8M0 exponent (value = 2^(byte-127)),
                 one scale per 32 consecutive K elements.
"""

from __future__ import annotations

import torch

from kquant import constants as C

E2M1_VALUES = torch.tensor(C.E2M1_LUT, dtype=torch.float32)


def unpack_codes(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [..., K/2] -> uint8 codes [..., K] (values 0..15)."""
    if packed.dtype != torch.uint8:
        raise TypeError(f"expected uint8, got {packed.dtype}")
    lo = packed & 0x0F
    hi = packed >> 4
    out = torch.empty(*packed.shape[:-1], packed.shape[-1] * 2,
                      dtype=torch.uint8, device=packed.device)
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return out


def pack_codes(codes: torch.Tensor) -> torch.Tensor:
    """uint8 codes [..., K] -> uint8 [..., K/2] (inverse of unpack_codes)."""
    if codes.shape[-1] % 2:
        raise ValueError("K must be even")
    lo = codes[..., 0::2]
    hi = codes[..., 1::2]
    return (lo | (hi << 4)).to(torch.uint8)


def scale_factors(scale: torch.Tensor) -> torch.Tensor:
    """uint8 E8M0 [..., K/32] -> float32 2^(e-127)."""
    return torch.exp2(scale.to(torch.float32) - C.E8M0_BIAS)


def code_values(codes: torch.Tensor) -> torch.Tensor:
    """uint8 codes -> float32 E2M1 values (unscaled)."""
    lut = E2M1_VALUES.to(codes.device)
    return lut[codes.long()]


def dequant(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Packed codes + E8M0 scales -> float32 weights [..., K]."""
    codes = unpack_codes(packed)
    vals = code_values(codes)
    s = scale_factors(scale)
    k = vals.shape[-1]
    if s.shape[-1] * C.MXFP4_BLOCK != k:
        raise ValueError(
            f"scale blocks {s.shape[-1]} do not cover K={k} at block {C.MXFP4_BLOCK}"
        )
    vals = vals.view(*vals.shape[:-1], s.shape[-1], C.MXFP4_BLOCK)
    out = vals * s.unsqueeze(-1)
    return out.view(*out.shape[:-2], k)


def dequant_blocked(packed: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (code_values [..., nb, 32] float32, scale_factors [..., nb]).

    Callers that re-quantize per block want the unscaled grid values and the
    block scales separately; this avoids materializing the dequantized weights
    twice.
    """
    codes = unpack_codes(packed)
    vals = code_values(codes)
    s = scale_factors(scale)
    nb = s.shape[-1]
    vals = vals.view(*vals.shape[:-1], nb, C.MXFP4_BLOCK)
    return vals, s
