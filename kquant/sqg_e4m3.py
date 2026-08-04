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
SQG_CHEB_NORMAL_E4M3 = "sqg-cheb-normal-e4m3"
SQG_TAIL_E4M3 = "sqg-tail-e4m3"
SQG_E4M3_CODEBOOKS = (
    SQG_NORMAL_E4M3,
    SQG_CHEB_NORMAL_E4M3,
    SQG_TAIL_E4M3,
)

SQG_K2_LAW_NORMAL = "normal"
SQG_K2_LAW_ASINH = "asinh"
SQG_K2_LAW_RATIONAL = "rational"
SQG_K2_LAW_OUTER_SLOPE = "outer-slope"
SQG_K2_LAW_NORMAL_CLIPPED = "normal-clipped"
SQG_K2_LAW_R44_FULL = "r44-full"
SQG_K2_LAW_R44_CLIPPED = "r44-clipped"
SQG_K2_LAWS = (
    SQG_K2_LAW_NORMAL,
    SQG_K2_LAW_ASINH,
    SQG_K2_LAW_RATIONAL,
    SQG_K2_LAW_OUTER_SLOPE,
    SQG_K2_LAW_NORMAL_CLIPPED,
    SQG_K2_LAW_R44_FULL,
    SQG_K2_LAW_R44_CLIPPED,
)

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
        # QSRT's donor arm; retaining the same compander is the least
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


def _finite_e4m3_rank_lut(values: torch.Tensor) -> torch.Tensor:
    """Round a monotone real rank law to canonical finite E4M3 bytes."""

    if values.ndim != 1 or values.numel() != _TRANSITIONS:
        raise ValueError("SQG rank-law values must contain exactly 65,536 entries")
    if not values.is_floating_point() or not bool(torch.isfinite(values).all()):
        raise ValueError("SQG rank-law values must be finite floating point")
    raw = values.float().to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    # E4M3 has signed zero.  The numerical law uses one canonical zero byte so
    # its exact representation is stable across CPU/GPU conversion paths.
    raw = raw.clone()
    raw[(raw & 0x7F) == 0] = 0
    decoded = raw.view(torch.float8_e4m3fn).float()
    if not bool(torch.isfinite(decoded).all()):
        raise RuntimeError("SQG rank law rounded to a non-finite E4M3 value")
    if not bool(torch.all(decoded[1:] >= decoded[:-1])):
        raise RuntimeError("SQG rank law is not monotone after E4M3 rounding")
    return raw


@lru_cache(maxsize=1)
def sqg_cheb_normal_rank_e4m3_bytes() -> torch.Tensor:
    """Return the exact full-tail normal target used by SQG-Cheb.

    The Chebyshev evaluator is an implementation of this finite staircase,
    not a different reconstruction law.  Computing the midpoint quantiles
    here gives a compact scientific reference for candidate generation and
    reproduces the exhaustively validated L16 SQG-Cheb table byte-for-byte.
    """

    ranks = torch.arange(_TRANSITIONS, dtype=torch.float64)
    probability = (ranks + 0.5) / _TRANSITIONS
    values = 1.5 * torch.special.ndtri(probability)
    return _finite_e4m3_rank_lut(values)


@lru_cache(maxsize=None)
def sqg_cheb_normal_e4m3_bytes(bits: int) -> torch.Tensor:
    """Map the exact SQG-Cheb normal staircase onto the frozen SQG graph."""

    return sqg_e4m3_bytes_from_rank_lut(
        bits, sqg_cheb_normal_rank_e4m3_bytes()
    ).contiguous()


def sqg_codebook_bytes(
    bits: int,
    codebook: str,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return exact state-indexed E4M3 labels for a named SQG codebook."""

    if codebook == SQG_NORMAL_E4M3:
        return sqg_e4m3_bytes(bits, "normal", device=device)
    if codebook == SQG_CHEB_NORMAL_E4M3:
        return sqg_cheb_normal_e4m3_bytes(bits).to(device=device).contiguous()
    if codebook == SQG_TAIL_E4M3:
        return sqg_e4m3_bytes(bits, "tail", device=device)
    raise ValueError(f"unsupported SQG codebook: {codebook!r}")


def sqg_codebook(
    bits: int,
    codebook: str,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Return a named SQG codebook widened from its exact E4M3 labels."""

    if not dtype.is_floating_point:
        raise TypeError("SQG codebook dtype must be floating point")
    return (
        sqg_codebook_bytes(bits, codebook, device=device)
        .view(torch.float8_e4m3fn)
        .to(dtype=dtype)
        .contiguous()
    )


@lru_cache(maxsize=None)
def sqg_k2_tail_compressed_rank_e4m3_bytes(
    law: str,
    parameter: float | None = None,
) -> torch.Tensor:
    """Return a K2 candidate staircase derived from SQG-Cheb normal.

    ``asinh`` uses ``a * asinh(z / a)``.  ``rational`` uses the globally
    monotone ``z / sqrt(1 + lambda * z**2)``; the tempting alternative
    ``z / (1 + lambda * z**2)`` is deliberately excluded because its
    derivative becomes negative in the tails.  Both candidates preserve unit
    slope at zero and modify only the scalar rank-to-value law, never SQG's
    state graph, phase, syndrome, or branch permutation. ``outer-slope`` is
    the investigator's Q33 family: it leaves the central two K2 quartiles
    unchanged and multiplies distance beyond ``normal_ppf(0.75)`` by the
    supplied slope.  The remaining fixed laws isolate endpoint clipping from
    the R44-versus-exact-normal approximation.
    """

    if law not in SQG_K2_LAWS:
        raise ValueError(f"unsupported SQG K2 law {law!r}; expected {SQG_K2_LAWS}")
    fixed_laws = {
        SQG_K2_LAW_NORMAL,
        SQG_K2_LAW_NORMAL_CLIPPED,
        SQG_K2_LAW_R44_FULL,
        SQG_K2_LAW_R44_CLIPPED,
    }
    if law in fixed_laws:
        if parameter is not None:
            raise ValueError(f"the {law} SQG K2 law has no parameter")
        if law == SQG_K2_LAW_NORMAL:
            return sqg_cheb_normal_rank_e4m3_bytes()
        ranks = torch.arange(_TRANSITIONS, dtype=torch.float64)
        probability = (ranks + 0.5) / _TRANSITIONS
        if law in (SQG_K2_LAW_NORMAL_CLIPPED, SQG_K2_LAW_R44_CLIPPED):
            probability = probability.clamp(_CLIP, 1.0 - _CLIP)
        if law in (SQG_K2_LAW_R44_FULL, SQG_K2_LAW_R44_CLIPPED):
            values = 1.5 * _r44_inverse_normal(probability)
        else:
            values = 1.5 * torch.special.ndtri(probability)
        return _finite_e4m3_rank_lut(values)
    if (
        parameter is None
        or isinstance(parameter, bool)
        or not isinstance(parameter, (int, float))
        or not math.isfinite(float(parameter))
        or float(parameter) <= 0.0
    ):
        raise ValueError(f"the {law} SQG K2 law requires a positive parameter")

    ranks = torch.arange(_TRANSITIONS, dtype=torch.float64)
    probability = (ranks + 0.5) / _TRANSITIONS
    gaussian = torch.special.ndtri(probability)
    value = float(parameter)
    if law == SQG_K2_LAW_ASINH:
        values = 1.5 * gaussian
        values = value * torch.asinh(values / value)
    elif law == SQG_K2_LAW_RATIONAL:
        values = 1.5 * gaussian
        values = values / torch.sqrt(1.0 + value * values.square())
    else:
        # Q33 is ``outer-slope:1.03125``.  Parameterizing the slope directly
        # keeps the exact searched E4 staircase reproducible.
        quartile = float(torch.special.ndtri(torch.tensor(0.75, dtype=torch.float64)))
        magnitude = gaussian.abs()
        adjusted = magnitude + (value - 1.0) * torch.clamp(
            magnitude - quartile, min=0.0
        )
        values = 1.5 * gaussian.sign() * adjusted
    return _finite_e4m3_rank_lut(values)


def sqg_k4_eight_stratum_rank_permutation() -> torch.Tensor:
    """Return the K3-aligned eight-stratum labelling for a K4 trellis.

    The physical trellis still consumes four new branch bits and therefore
    retains K4 path rate and state transitions.  For numerical labelling, the
    most-significant new bit extends a virtual K3 history and the remaining
    three bits select one of eight K3 strata.  Each K4 state consequently
    exposes two history-conditioned choices in every octile.

    Transition codewords are already laid out as ``history12 | branch4``;
    reinterpreting them as ``history13 | branch3`` is therefore exactly the
    existing K3 SQG rank permutation over the same 16-bit index space.
    """

    return sqg_rank_permutation(3)


def sqg_k4_eight_stratum_e4m3_bytes_from_rank_lut(
    rank_lut: torch.Tensor,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Map a rank staircase onto the K3-aligned eight-stratum K4 labels."""

    if rank_lut.dtype != torch.uint8:
        raise TypeError("SQG rank LUT must contain raw uint8 E4M3 labels")
    if rank_lut.ndim != 1 or rank_lut.numel() != _TRANSITIONS:
        raise ValueError("SQG rank LUT must contain exactly 65,536 labels")
    rank_lut = rank_lut.detach().to(device="cpu").contiguous()
    values = rank_lut.view(torch.float8_e4m3fn).float()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("SQG rank LUT must contain only finite E4M3 labels")
    return rank_lut.index_select(
        0, sqg_k4_eight_stratum_rank_permutation()
    ).to(device=device).contiguous()


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
