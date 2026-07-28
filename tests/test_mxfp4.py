import torch

from kquant import constants as C
from kquant.io import mxfp4


def test_pack_unpack_roundtrip():
    g = torch.Generator().manual_seed(0)
    codes = torch.randint(0, 16, (7, 64), dtype=torch.uint8, generator=g)
    assert torch.equal(mxfp4.unpack_codes(mxfp4.pack_codes(codes)), codes)


def test_nibble_order_low_first():
    packed = torch.tensor([[0x21]], dtype=torch.uint8)  # low=1, high=2
    codes = mxfp4.unpack_codes(packed)
    assert codes.tolist() == [[1, 2]]


def test_dequant_values():
    # one row, one block: code 7 (=6.0) with scale exponent 127 (=1.0)
    codes = torch.zeros(1, 32, dtype=torch.uint8)
    codes[0, 0] = 7  # 6.0
    codes[0, 1] = 15  # -6.0
    packed = mxfp4.pack_codes(codes)
    scale = torch.full((1, 1), C.E8M0_BIAS + 1, dtype=torch.uint8)  # 2.0
    w = mxfp4.dequant(packed, scale)
    assert w[0, 0].item() == 12.0
    assert w[0, 1].item() == -12.0
    assert w[0, 2].item() == 0.0


def test_dequant_blocked_matches_dequant():
    g = torch.Generator().manual_seed(1)
    codes = torch.randint(0, 16, (3, 5, 96), dtype=torch.uint8, generator=g)
    packed = mxfp4.pack_codes(codes)
    scale = torch.randint(100, 140, (3, 5, 3), dtype=torch.uint8, generator=g)
    w1 = mxfp4.dequant(packed, scale)
    vals, s = mxfp4.dequant_blocked(packed, scale)
    w2 = (vals * s.unsqueeze(-1)).reshape(w1.shape)
    assert torch.equal(w1, w2)
