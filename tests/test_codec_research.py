from __future__ import annotations

import math

import torch

from kquant.codec_research import (
    affine_prediction,
    apply_residual_vq,
    contextual_index_rate,
    endpoint_ternary,
    gather_tiles,
    index_entropy,
    nearest_neuron_prediction,
    normalize_matrix_gains,
    normalized_mse,
    packed_mxfp4_zero_statistics,
    pvq_codebook_size,
    pvq_gain_shape,
    quantized_low_rank,
    rank_contexts,
    sample_tile_coordinates,
    train_residual_vq,
)
from kquant.io.mxfp4 import pack_codes
from scripts.experiment_context_exl3 import (
    _context_block_order,
)
from scripts.experiment_context_exl3 import (
    _parse_allocations as parse_exl_allocations,
)
from scripts.experiment_hessian_vq import (
    _inverse_transform_weight,
    _transform_factors,
    _transform_weight,
)
from scripts.experiment_shared_rvq import (
    _pvq_configurations,
    _rvq_configurations,
)
from scripts.experiment_sparse_enhancement_vq import (
    _matrix_tiles,
    _tile_vectors,
    _unvectorize,
    _vectorize,
)


def test_affine_prediction_recovers_scale_and_offset() -> None:
    reference = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4)
    target = (
        reference * torch.tensor([2.0, -0.5])[:, None, None]
        + torch.tensor([3.0, 7.0])[:, None, None]
    )
    predicted = affine_prediction(target, reference)
    torch.testing.assert_close(predicted, target)


def test_endpoint_ternary_has_sub_three_bit_rate_and_fits_line_block() -> None:
    endpoint0 = torch.linspace(-1.0, 1.0, 16)
    endpoint1 = torch.linspace(1.0, 2.0, 16)
    alpha = torch.linspace(0.0, 1.0, 16)[:, None]
    tile = endpoint0 + alpha * (endpoint1 - endpoint0)
    result = endpoint_ternary(tile[None])
    assert result.bits_per_weight < 3.0
    assert normalized_mse(tile[None], result.reconstruction) < 1e-3


def test_quantized_low_rank_preserves_low_rank_tile() -> None:
    left = torch.randn(16, 2, generator=torch.Generator().manual_seed(1))
    right = torch.randn(2, 16, generator=torch.Generator().manual_seed(2))
    tile = left @ right
    result = quantized_low_rank(tile[None], rank=2, factor_bits=8)
    assert result.bits_per_weight < 2.2
    assert normalized_mse(tile[None], result.reconstruction) < 5e-4


def test_nearest_neuron_prediction_finds_permutation() -> None:
    gate = torch.eye(8)
    signature = torch.cat((gate, gate * 2), dim=1)
    permutation = torch.tensor([3, 0, 6, 1, 7, 4, 2, 5])
    result = nearest_neuron_prediction(
        gate[permutation], gate, signature[permutation], signature
    )
    assert result["best_index"].tolist() == permutation.tolist()
    assert result["mutual_fraction"] == 1.0
    assert result["scaled_residual_ratio"] == 0.0


def test_tile_coordinate_sampling_and_gather_are_deterministic() -> None:
    matrix = torch.arange(32 * 48).reshape(32, 48)
    coordinates = sample_tile_coordinates(
        tuple(matrix.shape), tile_size=16, count=4, seed=7
    )
    repeated = sample_tile_coordinates(
        tuple(matrix.shape), tile_size=16, count=4, seed=7
    )
    torch.testing.assert_close(coordinates, repeated)
    tiles = gather_tiles(matrix, coordinates, tile_size=16)
    assert tiles.shape == (4, 16, 16)


def test_pvq_gain_shape_has_exact_three_bit_rate() -> None:
    vectors = torch.randn(64, 8, generator=torch.Generator().manual_seed(3))
    result = pvq_gain_shape(vectors, pulses=6, gain_bits=9)
    assert pvq_codebook_size(8, 6) <= 1 << 15
    assert result.bits_per_weight == 3.0
    assert torch.isfinite(result.reconstruction).all()
    assert normalized_mse(vectors, result.reconstruction) < 0.2


def test_residual_vq_learns_shared_clustered_vectors() -> None:
    generator = torch.Generator().manual_seed(4)
    centers = torch.tensor([[-2.0, -1.0], [-1.0, 2.0], [1.0, -2.0], [2.0, 1.0]])
    labels = torch.arange(256) % len(centers)
    vectors = centers[labels] + 0.05 * torch.randn(256, 2, generator=generator)
    model = train_residual_vq(
        vectors,
        stages=2,
        codebook_size=4,
        iterations=6,
        batch_size=64,
        seed=5,
    )
    reconstruction, stage_labels = apply_residual_vq(vectors, model, batch_size=64)
    assert model.bits_per_weight == 2.0
    assert len(stage_labels) == 2
    assert normalized_mse(vectors, reconstruction) < 0.01
    assert 0.0 <= index_entropy(stage_labels[0], 4) <= 2.0


def test_fixed_rate_vq_configurations_are_three_bits_per_weight() -> None:
    for dimension in (4, 8, 16):
        for stages, size in _rvq_configurations(dimension):
            assert stages * int(math.log2(size)) == 3 * dimension
        for _, shape_bits, gain_bits in _pvq_configurations(dimension):
            assert shape_bits + gain_bits == 3 * dimension


def test_separable_matrix_gains_round_trip_and_equilibrate() -> None:
    generator = torch.Generator().manual_seed(6)
    row = torch.tensor([0.1, 0.5, 2.0, 7.0])
    column = torch.tensor([0.2, 1.0, 3.0, 9.0, 20.0])
    matrix = torch.randn(4, 5, generator=generator) * row[:, None] * column[None]
    normalized, gains = normalize_matrix_gains(
        matrix, mode="separable", vector_orientation="column", iterations=10
    )
    torch.testing.assert_close(gains.apply(normalized), matrix.float())
    assert gains.stored_bits == 16 * (matrix.shape[0] + matrix.shape[1])
    torch.testing.assert_close(
        normalized.square().mean(dim=1),
        torch.ones(matrix.shape[0]),
        atol=3e-3,
        rtol=3e-3,
    )


def test_sparse_enhancement_layout_round_trips_both_orientations() -> None:
    matrix = torch.arange(16 * 32).reshape(16, 32)
    for orientation in ("row", "column"):
        vectors = _vectorize(matrix, orientation)
        torch.testing.assert_close(
            _unvectorize(vectors, tuple(matrix.shape), orientation), matrix
        )
        tiles = _matrix_tiles(matrix)
        assert _tile_vectors(tiles, orientation).shape == (128, 4)


def test_packed_mxfp4_zero_statistics_counts_positive_and_negative_zero() -> None:
    codes = torch.tensor([[0, 8, 1, 0], [8, 0, 2, 8], [0, 8, 3, 0]], dtype=torch.uint8)
    statistics = packed_mxfp4_zero_statistics(pack_codes(codes))
    assert statistics["zero_columns"] == 3
    assert statistics["zero_elements"] == 9
    assert statistics["logical_columns"] == 4


def test_hessian_block_transforms_round_trip() -> None:
    generator = torch.Generator().manual_seed(7)
    basis = torch.randn(8, 8, generator=generator)
    hessian = basis @ basis.T
    weight = torch.randn(6, 8, generator=generator)
    for mode in ("none", "diagonal", "block4_cholesky"):
        factor, _ = _transform_factors(hessian, mode)
        transformed = _transform_weight(weight, factor)
        reconstructed = _inverse_transform_weight(transformed, factor)
        torch.testing.assert_close(reconstructed, weight, atol=1e-5, rtol=1e-5)


def test_rank_contexts_are_stable_and_balanced() -> None:
    scores = torch.tensor([4.0, 1.0, 3.0, 1.0, 2.0, 8.0, 7.0, 6.0])
    contexts = rank_contexts(scores, 4)

    assert contexts.tolist() == [2, 0, 1, 0, 1, 3, 3, 2]
    assert torch.bincount(contexts, minlength=4).tolist() == [2, 2, 2, 2]


def test_contextual_index_rate_uses_context_population() -> None:
    assignments = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

    assert (
        contextual_index_rate(assignments, (10, 11, 13, 14), vector_dimension=4) == 3.0
    )


def test_context_exl_allocations_hold_three_bit_average() -> None:
    configurations = parse_exl_allocations("3,3,3,3;2,3,3,4;2,2,4,4")

    assert all(sum(bits) / len(bits) == 3.0 for bits in configurations)


def test_context_exl_tp12_layout_balances_paired_rates() -> None:
    assignments = torch.arange(4).repeat_interleave(192)
    order = _context_block_order(assignments, "tp12_balanced")
    channel_contexts = assignments[order].repeat_interleave(4).reshape(12, 256)
    rates = torch.tensor([2, 3, 3, 4])[channel_contexts]

    torch.testing.assert_close(rates.float().mean(dim=1), torch.full((12,), 3.0))
    assert torch.all(channel_contexts.reshape(-1, 128).diff(dim=1) == 0)


def test_context_exl_importance_order_can_be_postpacked_for_tp12() -> None:
    assignments = torch.arange(8).repeat_interleave(96)
    importance_order = _context_block_order(assignments, "importance_ordered")
    postpack_order = _context_block_order(assignments, "tp12_balanced")
    importance_channels = assignments[importance_order].repeat_interleave(4)
    postpack_channels = assignments[postpack_order].repeat_interleave(4)
    bit_widths = torch.tensor([2, 3, 3, 3, 3, 3, 3, 4])

    assert sorted(importance_order.tolist()) == sorted(postpack_order.tolist())
    assert torch.all(importance_channels.reshape(-1, 128).diff(dim=1) == 0)
    assert torch.all(postpack_channels.reshape(-1, 128).diff(dim=1) == 0)
    postpack_rates = bit_widths[postpack_channels].reshape(12, 256)
    torch.testing.assert_close(
        postpack_rates.float().mean(dim=1), torch.full((12,), 3.0)
    )
