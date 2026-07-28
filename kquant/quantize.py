"""Blockwise codebook quantization shared by the distortion engine and packers.

Scale convention follows b12x's nf3_2p1: per-32-block scale = block absmax /
max|codebook|, rounded to 3 mantissa bits (E4M3-style), floored at 2^-10.
The same routine serves any symmetric codebook (NF3, NF2, refits, E2M1
subsets); an optional multiplicative scale search refines absmax scaling.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from kquant import constants as C


@dataclass(frozen=True)
class CodebookFormat:
    name: str
    levels: tuple[float, ...]  # ascending, symmetric
    bits_effective: float  # storage bpw including scale bytes
    code_bits: int

    def levels_tensor(self, device) -> torch.Tensor:
        return torch.tensor(self.levels, dtype=torch.float32, device=device)


KEEP_FORMAT = "keep_mxfp4"

BUILTIN_FORMATS: dict[str, CodebookFormat] = {
    "nf3": CodebookFormat("nf3", C.NF3_CODEBOOK, C.NF3_BITS_EFFECTIVE, 3),
    "nf2": CodebookFormat("nf2", C.NF2_CODEBOOK, C.NF2_BITS_EFFECTIVE, 2),
}

FORMAT_BITS: dict[str, float] = {
    KEEP_FORMAT: C.MXFP4_BITS_EFFECTIVE,
    "exl3_3": C.EXL3_BITS_EFFECTIVE,
    **{k: f.bits_effective for k, f in BUILTIN_FORMATS.items()},
}


def format_bits(name: str) -> float:
    """Effective bpw for a format name; refit variants share their base's."""
    if name in FORMAT_BITS:
        return FORMAT_BITS[name]
    base = name.removesuffix("_refit")
    if base in FORMAT_BITS:
        return FORMAT_BITS[base]
    raise KeyError(f"unknown format {name}")


def round_scale_e4m3(t: torch.Tensor,
                     mantissa_bits: int = C.NF3_SCALE_MANTISSA_BITS,
                     floor: float = C.NF3_SCALE_FLOOR) -> torch.Tensor:
    """Round positive scales to `mantissa_bits` mantissa bits, floored."""
    t = t.clamp(min=floor)
    e = torch.floor(torch.log2(t))
    step = torch.exp2(e - mantissa_bits)
    return torch.round(t / step) * step


@dataclass
class BlockQuantResult:
    codes: torch.Tensor  # int64 [..., nb, 32] indices into levels
    scales: torch.Tensor  # float32 [..., nb]
    recon: torch.Tensor  # float32 [..., nb, 32]
    sq_err: torch.Tensor  # float64 [..., nb] per-block squared error


def quantize_blocks(
    w: torch.Tensor,  # float32 [..., nb, 32]
    fmt: CodebookFormat,
    scale_search: tuple[float, ...] = (1.0,),
) -> BlockQuantResult:
    levels = fmt.levels_tensor(w.device)
    mids = (levels[1:] + levels[:-1]) / 2
    absmax = w.abs().amax(dim=-1)  # [..., nb]
    base = absmax / float(levels.abs().max())

    best: BlockQuantResult | None = None
    best_block_err: torch.Tensor | None = None
    for mult in scale_search:
        t = round_scale_e4m3(base * mult)
        wn = w / t.unsqueeze(-1)
        idx = torch.bucketize(wn, mids)
        recon = levels[idx] * t.unsqueeze(-1)
        err = ((w - recon).double() ** 2).sum(dim=-1)  # [..., nb]
        if best is None:
            best = BlockQuantResult(idx, t, recon, err)
            best_block_err = err
        else:
            better = err < best_block_err
            best_block_err = torch.where(better, err, best_block_err)
            bb = better.unsqueeze(-1)
            best.codes = torch.where(bb, idx, best.codes)
            best.recon = torch.where(bb, recon, best.recon)
            best.scales = torch.where(better, t, best.scales)
    if len(scale_search) > 1:
        best.sq_err = best_block_err
    return best


def pack_3bit(codes: torch.Tensor) -> torch.Tensor:
    """int codes 0..7 [..., K] -> uint8 [..., K*3/8], little-endian bit order."""
    K = codes.shape[-1]
    if K % 8:
        raise ValueError("K must be divisible by 8 for 3-bit packing")
    c = codes.to(torch.int64).reshape(*codes.shape[:-1], K // 8, 8)
    shifts = torch.arange(8, device=codes.device, dtype=torch.int64) * 3
    word = (c << shifts).sum(dim=-1)  # 24 bits used
    out = torch.stack(
        [(word >> (8 * i)) & 0xFF for i in range(3)], dim=-1
    ).to(torch.uint8)
    return out.reshape(*codes.shape[:-1], K // 8 * 3)


def unpack_3bit(packed: torch.Tensor, K: int) -> torch.Tensor:
    """Inverse of pack_3bit -> int64 codes [..., K]."""
    if packed.shape[-1] * 8 != K * 3:
        raise ValueError("packed size does not match K")
    b = packed.to(torch.int64).reshape(*packed.shape[:-1], K // 8, 3)
    word = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16)
    shifts = torch.arange(8, device=packed.device, dtype=torch.int64) * 3
    codes = (word.unsqueeze(-1) >> shifts) & 0x7
    return codes.reshape(*packed.shape[:-1], K)


# --- scale byte storage (uint8 = E4M3 bits of t * 2^SCALE_BIAS_LOG2) --------
SCALE_BIAS_LOG2 = 4  # 2^-10 floor * 2^4 = 2^-6 = E4M3 min normal


def encode_scale_bytes(t: torch.Tensor) -> torch.Tensor:
    biased = (t * float(2**SCALE_BIAS_LOG2)).to(torch.float8_e4m3fn)
    return biased.view(torch.uint8)


def decode_scale_bytes(b: torch.Tensor) -> torch.Tensor:
    return b.view(torch.float8_e4m3fn).to(torch.float32) * float(2**-SCALE_BIAS_LOG2)
