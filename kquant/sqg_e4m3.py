"""QSRT and research-control SQG E4M3 reconstruction tables.

This module implements the independently proposed L16 SQG labelling.  It
deliberately separates the rolling trellis graph from its numerical labels:
the encoder still stores K branch bits per coefficient, while a deterministic
K-specific mapping turns each 16-bit transition state into an E4M3 value.

The primary ``qsrt_sqg_e4m3`` profile composes the carry-mixed bijective SQG rank
map with a modal 12-bit compression of the Chebyshev-derived finite-E4M3
staircase.  The older exact-graph and R44 mappings remain explicit research
controls.  All paths return raw E4M3 bytes so the offline encoder and serving
kernel can share one byte-exact numerical contract.
"""

from __future__ import annotations

from functools import lru_cache

import torch


SQG_NORMAL_E4M3 = "sqg-normal-e4m3"
SQG_CHEB_NORMAL_E4M3 = "sqg-cheb-normal-e4m3"
SQG_CHEB_NORMAL_K2_Q8H4_W2_E4M3 = (
    "sqg-cheb-normal-k2-q8h4-w2-e4m3"
)
QSRT_E4M3 = "qsrt_sqg_e4m3"

_TRANSITIONS = 1 << 16
_CLIP = 1.0 / 2048.0
_P = (1.25667142, 2.87422731, -9.02398882, 5.36810336, -0.46703015)
_Q = (1.0, 2.07630930, -8.08332684, 6.32135736, -1.31208298)


def _validate(bits: int, mode: str) -> None:
    if isinstance(bits, bool) or not isinstance(bits, int) or bits not in (2, 3, 4):
        raise ValueError("SQG supports integer K2, K3, or K4")
    if mode != "normal":
        raise ValueError("the supported R44 mode is 'normal'")


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


@lru_cache(maxsize=1)
def qsrt_e4m3_rank_lut_bytes() -> torch.Tensor:
    """Return the frozen 4 KiB modal T12 reconstruction staircase.

    Each byte represents sixteen consecutive ranks of the exact normal
    staircase.  The modal byte is selected per block; ties select the lower
    raw byte.  This is the same immutable construction used by the B12X
    ``qsrt_sqg_e4m3`` decoder.
    """

    exact = sqg_cheb_normal_rank_e4m3_bytes().reshape(1 << 12, 16)
    result = torch.empty(1 << 12, dtype=torch.uint8)
    for index, block in enumerate(exact):
        labels, counts = torch.unique(block, return_counts=True)
        result[index] = labels[counts == counts.max()].min()
    return result.contiguous()


@lru_cache(maxsize=None)
def qsrt_e4m3_rank_permutation(bits: int) -> torch.Tensor:
    """Return the primary carry-mixed SQG rank for every L16 codeword.

    The two triangular xorshifts are bijections over the retained history,
    the multiplier is odd, and bit reversal followed by XOR is a branch
    permutation.  The resulting map is therefore bijective over all 65,536
    ``(stratum, phase)`` coordinates while preserving one outgoing edge per
    stratum in every state.
    """

    _validate(bits, "normal")
    width = 16 - bits
    history_mask = (1 << width) - 1
    branch_mask = (1 << bits) - 1
    codeword = torch.arange(_TRANSITIONS, dtype=torch.int64)
    history = codeword >> bits
    branch = codeword & branch_mask

    mixed = history ^ (history >> 11)
    mixed ^= (mixed << 11) & history_mask
    product = (0x3FA7D929 * mixed + 0xC928FD8E) & 0xFFFFFFFF
    phase = product & history_mask
    syndrome = product >> (32 - bits)
    stratum = _reverse_low_bits(branch, bits) ^ syndrome
    return ((stratum << width) | phase).contiguous()


@lru_cache(maxsize=None)
def _qsrt_e4m3_cpu_bytes(bits: int) -> torch.Tensor:
    ranks = qsrt_e4m3_rank_permutation(bits)
    return qsrt_e4m3_rank_lut_bytes().index_select(0, ranks >> 4).contiguous()


def qsrt_e4m3_bytes(
    bits: int,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return the primary QSRT-E4M3 labels for all L16 codewords."""

    _validate(bits, "normal")
    return _qsrt_e4m3_cpu_bytes(bits).to(device=device).contiguous()


def sqg_codebook_bytes(
    bits: int,
    codebook: str,
    *,
    rate_axis: str | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return exact state-indexed E4M3 labels for a named SQG codebook."""

    if codebook == QSRT_E4M3:
        return qsrt_e4m3_bytes(bits, device=device)
    if codebook == SQG_NORMAL_E4M3:
        return sqg_e4m3_bytes(bits, "normal", device=device)
    if codebook == SQG_CHEB_NORMAL_E4M3:
        return sqg_cheb_normal_e4m3_bytes(bits).to(device=device).contiguous()
    if codebook == SQG_CHEB_NORMAL_K2_Q8H4_W2_E4M3:
        if rate_axis not in ("k", "n"):
            raise ValueError(
                "the w2-only K2-Q8H4 profile requires rate_axis 'k' or 'n'"
            )
        rank_lut = sqg_cheb_normal_rank_e4m3_bytes()
        if bits == 2 and rate_axis == "k":
            return sqg_k2_eight_stratum_e4m3_bytes_from_rank_lut(
                rank_lut, history_bit=4, device=device
            )
        return sqg_e4m3_bytes_from_rank_lut(
            bits, rank_lut, device=device
        ).contiguous()
    raise ValueError(f"unsupported SQG codebook: {codebook!r}")


def sqg_codebook(
    bits: int,
    codebook: str,
    *,
    rate_axis: str | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Return a named SQG codebook widened from its exact E4M3 labels."""

    if not dtype.is_floating_point:
        raise TypeError("SQG codebook dtype must be floating point")
    return (
        sqg_codebook_bytes(bits, codebook, rate_axis=rate_axis, device=device)
        .view(torch.float8_e4m3fn)
        .to(dtype=dtype)
        .contiguous()
    )


@lru_cache(maxsize=None)
def sqg_k2_eight_stratum_rank_permutation(
    history_bit: int = 2,
) -> torch.Tensor:
    """Return a state-conditioned eight-stratum labelling for a K2 trellis.

    The physical graph still appends two branch bits and retains fourteen
    state bits.  Numerical labelling reinterprets the newest retained history
    bit plus the two new branch bits as a virtual three-bit stratum selector;
    the remaining thirteen bits select phase.  A K2 state therefore exposes
    four of eight strata, while its retained history chooses which four.
    Globally this is the same bijective rank law used by K3.  ``history_bit``
    selects which retained physical codeword bit becomes the most-significant
    virtual stratum bit.  Bit 2 is the newest retained bit and reproduces the
    exact K3 transition-label function over the unmodified codeword.
    """

    if isinstance(history_bit, bool) or not isinstance(history_bit, int):
        raise TypeError("K2 eight-stratum history bit must be an integer")
    if not 2 <= history_bit < 16:
        raise ValueError("K2 eight-stratum history bit must be in [2, 15]")
    k3_ranks = sqg_rank_permutation(3)
    if history_bit == 2:
        return k3_ranks

    transitions = torch.arange(_TRANSITIONS, dtype=torch.int64)
    bit_2 = (transitions >> 2) & 1
    bit_h = (transitions >> history_bit) & 1
    swap = bit_2 ^ bit_h
    permuted = transitions ^ (swap << 2) ^ (swap << history_bit)
    return k3_ranks.index_select(0, permuted).contiguous()


def sqg_k2_eight_stratum_e4m3_bytes_from_rank_lut(
    rank_lut: torch.Tensor,
    *,
    history_bit: int = 2,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Map one shared rank staircase onto state-conditioned K2 octiles."""

    if rank_lut.dtype != torch.uint8:
        raise TypeError("SQG rank LUT must contain raw uint8 E4M3 labels")
    if rank_lut.ndim != 1 or rank_lut.numel() != _TRANSITIONS:
        raise ValueError("SQG rank LUT must contain exactly 65,536 labels")
    rank_lut = rank_lut.detach().to(device="cpu").contiguous()
    values = rank_lut.view(torch.float8_e4m3fn).float()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("SQG rank LUT must contain only finite E4M3 labels")
    return rank_lut.index_select(
        0, sqg_k2_eight_stratum_rank_permutation(history_bit)
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
