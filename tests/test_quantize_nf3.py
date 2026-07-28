"""NF3 semantics cross-checked against the b12x reference conventions
(sparkinfer w4a16 test: _round_to_e4m3_scale + codebook dequant)."""

import torch

from kquant import constants as C
from kquant.quantize import (
    BUILTIN_FORMATS,
    decode_scale_bytes,
    encode_scale_bytes,
    pack_3bit,
    quantize_blocks,
    round_scale_e4m3,
    unpack_3bit,
)


def _b12x_round_to_e4m3_scale(t_s: torch.Tensor) -> torch.Tensor:
    """Verbatim port of the reference from b12x tests/moe/test_w4a16_nf3.py
    (modulo the floor constant, which prepare.py sets to 2^-10)."""
    t_s = t_s.to(torch.float32).clamp(min=2.0**-10)
    e = torch.floor(torch.log2(t_s))
    step = torch.pow(2.0, e - 3)
    return torch.round(t_s / step) * step


def test_scale_rounding_matches_b12x_reference():
    t = torch.rand(4096) * 0.5 + 1e-4
    ours = round_scale_e4m3(t)
    ref = _b12x_round_to_e4m3_scale(t)
    assert torch.equal(ours, ref)


def test_codebook_matches_constants():
    fmt = BUILTIN_FORMATS["nf3"]
    assert fmt.levels == C.NF3_CODEBOOK
    assert fmt.bits_effective == 3.25


def test_quantize_blocks_exact_on_codebook_points():
    fmt = BUILTIN_FORMATS["nf3"]
    g = torch.Generator().manual_seed(7)
    scales = torch.exp2(torch.randint(-6, 3, (2, 3, 4), generator=g).float())
    idx = torch.randint(0, 8, (2, 3, 4, 32), generator=g)
    levels = torch.tensor(fmt.levels)
    w = levels[idx] * scales.unsqueeze(-1)
    q = quantize_blocks(w, fmt)
    # every input is exactly representable at its block's absmax scale
    assert torch.allclose(q.recon, w, atol=0.0)
    assert (q.sq_err == 0).all()


def test_quantize_blocks_nearest_assignment():
    fmt = BUILTIN_FORMATS["nf3"]
    w = torch.tensor([[[[0.9, -0.9, 0.05, 0.5] + [0.0] * 28]]])
    q = quantize_blocks(w, fmt)
    t = q.scales[0, 0, 0].item()
    lv = torch.tensor(fmt.levels)
    expect = lv[(w / t - lv.view(-1, 1, 1, 1, 1)).abs().argmin(dim=0)] * t
    assert torch.allclose(q.recon, expect)


def test_pack3bit_roundtrip():
    g = torch.Generator().manual_seed(3)
    codes = torch.randint(0, 8, (5, 96), generator=g)
    packed = pack_3bit(codes)
    assert packed.shape == (5, 36)
    assert torch.equal(unpack_3bit(packed, 96), codes)


def test_scale_byte_roundtrip_exact_for_3mantissa():
    # scales produced by round_scale_e4m3 must survive the u8 storage exactly
    t = round_scale_e4m3(torch.rand(2048) * 2 + 2**-10)
    rt = decode_scale_bytes(encode_scale_bytes(t))
    assert torch.equal(rt, t)
