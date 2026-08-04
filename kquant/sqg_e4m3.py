"""Stratified-quantile-graph (SQG) E4M3 reconstruction tables.

This module implements the independently proposed L16 SQG labelling.  It
deliberately separates the rolling trellis graph from its numerical labels:
the encoder still stores K branch bits per coefficient, while a deterministic
K-specific mapping turns each 16-bit transition state into an E4M3 value.

The CUDA validation path consumes the raw FP8 bytes returned by
``sqg_e4m3_bytes``.  ``sqg_e4m3_codebook`` widens the same bytes back to FP16
for reference reconstruction and closure checks.
"""

from __future__ import annotations

from functools import lru_cache
import math

import torch


SQG_NORMAL_E4M3 = "sqg-normal-e4m3"
SQG_TAIL_E4M3 = "sqg-tail-e4m3"
SQG_E4M3_CODEBOOKS = (SQG_NORMAL_E4M3, SQG_TAIL_E4M3)

_TRANSITIONS = 1 << 16
_CLIP = 1.0 / 2048.0
_P = (1.25667142, 2.87422731, -9.02398882, 5.36810336, -0.46703015)
_Q = (1.0, 2.07630930, -8.08332684, 6.32135736, -1.31208298)


def _validate(bits: int, mode: str) -> None:
    if isinstance(bits, bool) or not isinstance(bits, int) or bits not in (2, 3, 4, 5):
        raise ValueError("SQG supports integer K2, K3, K4, or K5")
    if mode not in ("normal", "tail"):
        raise ValueError("SQG mode must be 'normal' or 'tail'")


def _mix_width(
    values: torch.Tensor,
    *,
    width: int,
    multiplier_a: int,
    multiplier_b: int,
    shift_a: int,
    shift_b: int,
    shift_c: int,
) -> torch.Tensor:
    """Apply the report's bijective XOR/multiply mixer modulo ``2**width``."""

    mask = (1 << width) - 1
    result = values & mask
    for multiplier, shift in (
        (multiplier_a | 1, shift_a),
        (multiplier_b | 1, shift_b),
    ):
        result ^= result >> min(max(shift, 1), width - 1)
        result = (result * multiplier) & mask
    result ^= result >> min(max(shift_c, 1), width - 1)
    return result & mask


def _reverse_low_bits(values: torch.Tensor, bits: int) -> torch.Tensor:
    result = torch.zeros_like(values)
    for index in range(bits):
        result |= ((values >> index) & 1) << (bits - 1 - index)
    return result


def _r44_inverse_normal(probability: torch.Tensor) -> torch.Tensor:
    a = 2.0 * probability - 1.0
    x = a.square()
    numerator = torch.full_like(x, _P[-1])
    denominator = torch.full_like(x, _Q[-1])
    for coefficient in reversed(_P[:-1]):
        numerator = numerator * x + coefficient
    for coefficient in reversed(_Q[:-1]):
        denominator = denominator * x + coefficient
    return a * numerator / denominator


@lru_cache(maxsize=None)
def _sqg_cpu_bytes(bits: int, mode: str) -> torch.Tensor:
    _validate(bits, mode)
    width = 16 - bits
    branches = 1 << bits
    states = torch.arange(_TRANSITIONS, dtype=torch.int64)
    history = states >> bits
    branch = states & (branches - 1)

    phase = _mix_width(
        history,
        width=width,
        multiplier_a=0x65AF,
        multiplier_b=0x16BF,
        shift_a=6,
        shift_b=4,
        shift_c=5,
    )
    syndrome_hash = _mix_width(
        history ^ 0x5105,
        width=width,
        multiplier_a=0x8693,
        multiplier_b=0x2A21,
        shift_a=2,
        shift_b=4,
        shift_c=4,
    )
    syndrome = syndrome_hash & (branches - 1)
    stratum = (
        7 * (_reverse_low_bits(branch, bits) ^ syndrome)
    ) & (branches - 1)
    rank = (stratum << width) | phase
    probability = ((rank.double() + 0.5) / _TRANSITIONS).clamp(
        _CLIP, 1.0 - _CLIP
    )
    gaussian = _r44_inverse_normal(probability)
    if mode == "tail":
        # The supplied construction uses 5/32 for K3/K4.  K2 is needed by
        # TrellisShift's donor arm; retaining the same compander is the least
        # assumptive extrapolation and is reported separately in validation.
        beta = 9.0 / 32.0 if bits == 5 else 5.0 / 32.0
        gaussian = gaussian * (1.0 + beta * gaussian.square())
    values = (1.5 * gaussian).float()
    if not bool(torch.isfinite(values).all()):
        raise RuntimeError("SQG compander produced a non-finite value")
    return values.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()


def sqg_e4m3_bytes(
    bits: int,
    mode: str,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return the 65,536 raw finite-E4M3 labels for an L16 SQG graph."""

    _validate(bits, mode)
    return _sqg_cpu_bytes(bits, mode).to(device=device).contiguous()


def sqg_e4m3_codebook(
    bits: int,
    mode: str,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Return the SQG table widened from its exact E4M3 wire values."""

    if not dtype.is_floating_point:
        raise TypeError("SQG codebook dtype must be floating point")
    raw = sqg_e4m3_bytes(bits, mode, device=device)
    return raw.view(torch.float8_e4m3fn).to(dtype=dtype).contiguous()


def sqg_e4m3_bytes_from_rank_lut(
    bits: int,
    rank_lut: torch.Tensor,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Map a shared rank-indexed E4M3 law onto the unchanged SQG graph.

    ``rank_lut`` is deliberately independent of ``bits``.  K2/K3/K4 may
    expose different outgoing menus because their stratum widths differ, but
    they retain the same history mixer, syndrome mixer, branch permutation,
    and scalar reconstruction law.  This is the controlled interface used to
    evaluate alternative companders without introducing rate-specific graph
    edits.
    """

    _validate(bits, "normal")
    if rank_lut.dtype != torch.uint8:
        raise TypeError("SQG rank LUT must contain raw uint8 E4M3 labels")
    if rank_lut.ndim != 1 or rank_lut.numel() != _TRANSITIONS:
        raise ValueError("SQG rank LUT must contain exactly 65,536 labels")
    rank_lut = rank_lut.detach().to(device="cpu").contiguous()
    values = rank_lut.view(torch.float8_e4m3fn).float()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("SQG rank LUT must contain only finite E4M3 labels")
    return rank_lut.index_select(0, sqg_rank_permutation(bits)).to(
        device=device
    ).contiguous()


def sqg_e4m3_codebook_from_rank_lut(
    bits: int,
    rank_lut: torch.Tensor,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Widen a shared rank-indexed E4M3 law after SQG graph labelling."""

    if not dtype.is_floating_point:
        raise TypeError("SQG codebook dtype must be floating point")
    raw = sqg_e4m3_bytes_from_rank_lut(bits, rank_lut, device=device)
    return raw.view(torch.float8_e4m3fn).to(dtype=dtype).contiguous()


def sqg_rank_permutation(bits: int) -> torch.Tensor:
    """Return the pre-projection quantile rank for structural validation."""

    _validate(bits, "normal")
    width = 16 - bits
    branches = 1 << bits
    transitions = torch.arange(_TRANSITIONS, dtype=torch.int64)
    history = transitions >> bits
    branch = transitions & (branches - 1)
    phase = _mix_width(
        history,
        width=width,
        multiplier_a=0x65AF,
        multiplier_b=0x16BF,
        shift_a=6,
        shift_b=4,
        shift_c=5,
    )
    syndrome = _mix_width(
        history ^ 0x5105,
        width=width,
        multiplier_a=0x8693,
        multiplier_b=0x2A21,
        shift_a=2,
        shift_b=4,
        shift_c=4,
    ) & (branches - 1)
    stratum = (7 * (_reverse_low_bits(branch, bits) ^ syndrome)) & (branches - 1)
    return ((stratum << width) | phase).contiguous()


def sqg_mode_from_codebook(codebook: str) -> str:
    if codebook == SQG_NORMAL_E4M3:
        return "normal"
    if codebook == SQG_TAIL_E4M3:
        return "tail"
    raise ValueError(f"not an SQG codebook: {codebook!r}")
