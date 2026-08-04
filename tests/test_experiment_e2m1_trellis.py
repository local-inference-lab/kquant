import torch

from kquant.e2m1_trellis import opposite_zero_transition_codes
from kquant.exl3_reference import tensor_core_permutation
from kquant.io.mxfp4 import pack_codes
from kquant.source_weights import PackedMXFP4Matrix
from scripts.experiment_e2m1_trellis import (
    _parse_experts,
    _random_opposite_zero_mapping,
    sample_mxfp4_tiles,
)


def test_parse_experts_supports_ranges_and_all():
    assert _parse_experts("1,3:7:2") == (1, 3, 5)
    assert len(_parse_experts("all")) == 896


def test_sample_tiles_preserves_codes_and_source_scales():
    codes = torch.arange(16, dtype=torch.uint8).repeat(16, 2)
    packed = pack_codes(codes)
    scale = torch.full((16, 1), 127, dtype=torch.uint8)
    raw = PackedMXFP4Matrix(packed=packed, scale=scale)
    tiles, weights, energy = sample_mxfp4_tiles(
        raw, count=1, seed=1, device=torch.device("cpu")
    )
    assert tiles.shape == weights.shape == (1, 256)
    assert torch.equal(torch.sort(tiles[0]).values, torch.arange(16, dtype=torch.uint8).repeat_interleave(16))
    assert torch.equal(weights, torch.ones_like(weights))
    assert torch.allclose(
        energy,
        torch.tensor([sum(value * value for value in (0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6)) * 16]),
    )


def test_sample_tiles_uses_logical_gemm_k_n_orientation():
    codes = (
        torch.arange(16, dtype=torch.uint8)[:, None] * 1
        + torch.arange(32, dtype=torch.uint8)[None, :] % 16
    ) % 16
    packed = pack_codes(codes)
    scale = torch.full((16, 1), 127, dtype=torch.uint8)
    raw = PackedMXFP4Matrix(packed=packed, scale=scale)
    tiles, _, _ = sample_mxfp4_tiles(
        raw, count=1, seed=1, device=torch.device("cpu")
    )
    expected = codes[:, :16].T.reshape(-1).index_select(0, tensor_core_permutation())
    assert torch.equal(tiles[0], expected)


def test_random_mapping_keeps_zero_labels_at_opposite_corners():
    import random

    mapping = _random_opposite_zero_mapping(random.Random(7))
    assert mapping[0] == 0
    assert mapping[-1] == 8
    assert sorted(mapping) == list(range(16))
    assert opposite_zero_transition_codes()[0] == mapping[0]
