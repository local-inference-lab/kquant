import pytest
import torch

from kquant.e4m3_subcode import (
    decode_mcg_anchor_plus_alternative,
    decode_predictor_anchor_plus_alternative,
    e4m3_codes,
    e4m3_values,
    expand_conditional_binary_e4m3,
    fit_conditional_binary_e4m3,
    logical_conditional_binary_e4m3_bytes,
    pack_transition_bits,
    unpack_transition_bits,
)
from kquant.exl3_reference import (
    decode_mcg_states,
    decode_mul1_e4m3_states,
    decode_mul1_states,
    mul1_byte_sums,
)
from kquant.state_palette import (
    expand_state_palette,
    expand_state_scaled_palette,
    expand_state_subpalettes,
    fit_state_palette,
    fit_state_scaled_palette,
    fit_state_subpalettes,
    logical_state_palette_bytes,
    logical_state_scaled_palette_bytes,
    logical_state_subpalette_bytes,
)
from kquant.trellis_window import (
    encode_trellis,
    fp8_constrained_codebook,
    geometry,
    mcg_codebook,
    refit_transition_codebook,
    reconstruct_window_states,
    tie_codebook_values,
    trellis_alias_metrics,
)


@pytest.mark.parametrize("bits", (2, 3, 4, 5))
@pytest.mark.parametrize("window_bits", (10, 12, 14, 16))
def test_geometry_has_constant_transition_count(bits: int, window_bits: int):
    spec = geometry(bits, window_bits)
    assert spec.branches * spec.states == 1 << window_bits
    assert spec.predecessor_shift == window_bits - 2 * bits


def test_mcg_codebook_is_zero_extended_prefix():
    actual = mcg_codebook(12, device="cpu")
    expected = decode_mcg_states(torch.arange(1 << 12)).float()
    assert torch.equal(actual, expected)


def test_mul1_reference_matches_exl_literal_half_fma():
    states = torch.tensor([0, 1, 2, 3, 0x1234, 0x7FFF, 0xFFFF])
    expected_bits = torch.tensor(
        [-15640, 14625, -18643, 12857, -18274, 16330, -18122],
        dtype=torch.int16,
    )
    assert torch.equal(decode_mul1_states(states).view(torch.int16), expected_bits)


def test_mul1_e4m3_reference_is_exact_fp8_projection():
    states = torch.arange(1 << 16, dtype=torch.int32)
    expected = decode_mul1_states(states).to(torch.float8_e4m3fn).to(torch.float16)
    actual = decode_mul1_e4m3_states(states)
    assert torch.equal(actual, expected)
    assert torch.equal(actual, actual.to(torch.float8_e4m3fn).to(torch.float16))


def test_mul1_byte_sum_accepts_a_learned_uint32_multiplier():
    states = torch.tensor([0, 1, 0x1234, 0xFFFF])
    multiplier = 0x01020305
    products = ((states.to(torch.int64) & 0xFFFF) * multiplier) & 0xFFFFFFFF
    expected = sum((products >> shift) & 0xFF for shift in (0, 8, 16, 24))
    assert torch.equal(
        mul1_byte_sums(states, multiplier=multiplier), expected
    )
    assert not torch.equal(
        decode_mul1_states(states, multiplier=multiplier),
        decode_mul1_states(states),
    )


@pytest.mark.parametrize("multiplier", (-1, 1 << 32, 0.5, True))
def test_mul1_multiplier_must_be_uint32(multiplier):
    with pytest.raises(ValueError):
        mul1_byte_sums(torch.arange(4), multiplier=multiplier)


@pytest.mark.parametrize("dtype", (torch.float8_e4m3fn, torch.float8_e5m2))
def test_fp8_codebook_is_exactly_in_requested_alphabet(dtype: torch.dtype):
    source = mcg_codebook(12, device="cpu")
    scale = 0.875
    constrained = fp8_constrained_codebook(source, dtype=dtype, alphabet_scale=scale)
    assert torch.equal((constrained / scale).to(dtype).float(), constrained / scale)


def test_e4m3_byte_helpers_preserve_signed_zero_and_values():
    values = torch.tensor([-4.5, -0.0, 0.0, 0.125, 448.0])
    codes = e4m3_codes(values)
    decoded = e4m3_values(codes)
    assert torch.equal(decoded, values)
    assert int(codes[1]) != int(codes[2])


def test_conditional_binary_e4m3_expands_transition_bits():
    predictor = torch.tensor([-1.0, -1.0, 2.0, 2.0])
    anchors = e4m3_codes(predictor)
    pairs = torch.zeros((256, 2), dtype=torch.uint8)
    pairs[:] = e4m3_codes(torch.zeros((256, 2)))
    pairs[int(anchors[0]), :] = e4m3_codes(torch.tensor([-1.0, -2.0]))
    pairs[int(anchors[2]), :] = e4m3_codes(torch.tensor([2.0, 3.0]))
    expanded = expand_conditional_binary_e4m3(
        predictor, torch.tensor([0, 1, 1, 0]), pairs
    )
    assert torch.equal(expanded, torch.tensor([-1.0, -2.0, 3.0, 2.0]))


def test_conditional_binary_e4m3_fit_recovers_two_outputs_per_anchor():
    predictor = torch.ones(8)
    targets = torch.tensor([0.5, 0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 2.0])
    fit = fit_conditional_binary_e4m3(
        targets,
        torch.ones_like(targets),
        predictor,
        iterations=8,
    )
    assert fit.objective == 0.0
    assert torch.equal(fit.codebook, targets)
    assert fit.transition_bits.unique().numel() == 2


def test_conditional_binary_e4m3_logical_bytes():
    assert logical_conditional_binary_e4m3_bytes(
        65536, fixed_primary=False
    ) == 8 * 1024 + 512
    assert logical_conditional_binary_e4m3_bytes(
        65536, fixed_primary=True
    ) == 8 * 1024 + 256


def test_transition_bitplane_pack_unpack_and_fixed_primary_decode():
    bits = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=torch.uint8)
    packed = pack_transition_bits(bits)
    assert torch.equal(packed, torch.tensor([0b10010110, 0b00000001], dtype=torch.uint8))
    assert torch.equal(unpack_transition_bits(packed, bits.numel()), bits)

    predictor = torch.tensor([-1.0, -1.0, 2.0, 2.0, 0.5, 0.5, 1.0, 1.0, 3.0])
    alternatives = torch.arange(256, dtype=torch.uint8)
    alternatives[int(e4m3_codes(torch.tensor([-1.0]))[0])] = e4m3_codes(
        torch.tensor([-2.0])
    )[0]
    alternatives[int(e4m3_codes(torch.tensor([2.0]))[0])] = e4m3_codes(
        torch.tensor([3.0])
    )[0]
    alternatives[int(e4m3_codes(torch.tensor([0.5]))[0])] = e4m3_codes(
        torch.tensor([0.75])
    )[0]
    alternatives[int(e4m3_codes(torch.tensor([1.0]))[0])] = e4m3_codes(
        torch.tensor([1.5])
    )[0]
    alternatives[int(e4m3_codes(torch.tensor([3.0]))[0])] = e4m3_codes(
        torch.tensor([4.0])
    )[0]
    decoded = decode_mcg_anchor_plus_alternative(
        predictor, packed, alternatives
    )
    assert torch.equal(
        decoded,
        torch.tensor([-1.0, -2.0, 3.0, 2.0, 0.75, 0.5, 1.0, 1.5, 4.0]),
    )


def test_fixed_primary_decoder_accepts_any_procedural_predictor():
    predictor = decode_mul1_states(torch.arange(256)).float()
    alternatives = torch.arange(256, dtype=torch.uint8)
    bits = (torch.arange(256) % 3 == 0).to(torch.uint8)
    packed = pack_transition_bits(bits)
    assert torch.equal(
        decode_predictor_anchor_plus_alternative(predictor, packed, alternatives),
        decode_mcg_anchor_plus_alternative(predictor, packed, alternatives),
    )


@pytest.mark.parametrize("bits", (2, 3, 4, 5))
@pytest.mark.parametrize("window_bits", (10, 12, 14, 16))
def test_encoder_closes_and_reconstructs_a_cyclic_codebook_path(bits: int, window_bits: int):
    generator = torch.Generator().manual_seed(1000 + bits * 100 + window_bits)
    edges = torch.randint(0, 1 << bits, (2, 16), dtype=torch.uint8, generator=generator)
    states = reconstruct_window_states(edges, bits, window_bits)
    codebook = mcg_codebook(window_bits, device="cpu")
    source = codebook[states]
    result = encode_trellis(
        source,
        bits,
        window_bits,
        codebook,
        max_trace_bytes=64 << 20,
    )
    assert torch.count_nonzero(result.squared_error) == 0
    assert result.transition_indices is not None
    assert torch.equal(codebook[result.transition_indices.long()], result.reconstructed)


def test_encoder_applies_position_dependent_reconstruction_scales_inside_search():
    bits = 3
    window_bits = 12
    generator = torch.Generator().manual_seed(20260802)
    edges = torch.randint(0, 1 << bits, (2, 16), dtype=torch.uint8, generator=generator)
    transitions = reconstruct_window_states(edges, bits, window_bits)
    codebook = mcg_codebook(window_bits, device="cpu")
    scale = torch.linspace(0.5, 2.0, 16).repeat(2, 1)
    source = codebook[transitions] * scale
    result = encode_trellis(
        source,
        bits,
        window_bits,
        codebook,
        reconstruction_scale=scale,
        max_trace_bytes=64 << 20,
    )
    assert torch.count_nonzero(result.squared_error) == 0
    assert torch.equal(result.reconstructed, source)


def test_encoder_rejects_invalid_reconstruction_scale():
    source = torch.zeros((1, 16))
    codebook = mcg_codebook(12, device="cpu")
    with pytest.raises(ValueError, match="source shape"):
        encode_trellis(
            source,
            3,
            12,
            codebook,
            reconstruction_scale=torch.ones((1, 8)),
        )
    with pytest.raises(ValueError, match="finite and positive"):
        encode_trellis(
            source,
            3,
            12,
            codebook,
            reconstruction_scale=torch.zeros_like(source),
        )


def test_encoder_rejects_wrong_codebook_size():
    source = torch.zeros((1, 16))
    with pytest.raises(ValueError, match="4096"):
        encode_trellis(source, 3, 12, torch.zeros(128))


def test_alias_metrics_measure_output_preserving_successors():
    codebook = torch.arange(16, dtype=torch.float32)
    codebook[:4] = torch.tensor([0.0, 0.0, 1.0, 2.0])
    metrics = trellis_alias_metrics(
        codebook,
        bits=2,
        window_bits=4,
        transition_indices=torch.tensor([0, 1, 2, 3]),
    )
    outgoing = metrics["outgoing"]
    selected = metrics["selected"]
    assert metrics["numeric_labels"] == 15
    assert outgoing["states_with_alias_fraction"] == pytest.approx(0.25)
    assert outgoing["edges_with_alias_fraction"] == pytest.approx(0.125)
    assert outgoing["conditional_alias_bits_uniform"] == pytest.approx(0.125)
    assert outgoing["max_alias_multiplicity"] == 2
    assert selected["edges_with_alias_available_fraction"] == pytest.approx(0.5)
    assert selected["mean_available_alias_bits"] == pytest.approx(0.5)
    future = metrics["right_context_distinguishability"][0]
    assert future["edges_with_distinguishable_alias_fraction"] == pytest.approx(0.125)
    assert future["distinguishable_fraction_among_aliased_edges"] == pytest.approx(1.0)


def test_alias_metrics_do_not_count_equivalent_zero_futures_as_steering():
    metrics = trellis_alias_metrics(torch.zeros(16), bits=2, window_bits=4)
    outgoing = metrics["outgoing"]
    future = metrics["right_context_distinguishability"][0]
    assert outgoing["edges_with_alias_fraction"] == 1.0
    assert outgoing["conditional_alias_bits_uniform"] == 2.0
    assert future["right_context_classes"] == 1
    assert future["edges_with_distinguishable_alias_fraction"] == 0.0
    assert future["conditional_distinguishable_alias_bits_uniform"] == 0.0


def test_alias_metrics_reject_out_of_range_selected_transition():
    with pytest.raises(ValueError, match="out-of-range"):
        trellis_alias_metrics(
            torch.arange(16, dtype=torch.float32),
            bits=2,
            window_bits=4,
            transition_indices=torch.tensor([16]),
        )


def test_refit_transition_codebook_uses_weighted_centroids_and_keeps_unvisited():
    source = torch.tensor([[1.0, 3.0, 8.0, 10.0]])
    transitions = torch.tensor([[0, 0, 1, 1]])
    weights = torch.tensor([[1.0, 3.0, 1.0, 1.0]])
    initial = torch.tensor([100.0, 200.0, 300.0])
    fitted = refit_transition_codebook(
        source,
        transitions,
        initial,
        weights=weights,
    )
    assert torch.equal(fitted, torch.tensor([2.5, 9.0, 300.0]))


def test_refit_transition_codebook_supports_ridge_and_e4m3_constraint():
    source = torch.tensor([1.0, 3.0])
    transitions = torch.tensor([0, 0])
    initial = torch.tensor([4.0, 7.0])
    fitted = refit_transition_codebook(
        source,
        transitions,
        initial,
        ridge_pseudocount=2.0,
        reconstruction_dtype=torch.float8_e4m3fn,
    )
    # (1 + 3 + 2*4) / 4 = 3 is exactly representable in E4M3.
    assert torch.equal(fitted, torch.tensor([3.0, 7.0]))


def test_tie_codebook_values_learns_weighted_duplicate_labels():
    codebook = torch.tensor([0.0, 2.0, 8.0, 10.0])
    tied, centroids = tie_codebook_values(
        codebook,
        2,
        weights=torch.tensor([1.0, 3.0, 1.0, 1.0]),
        initial_centroids=torch.tensor([0.0, 10.0]),
        reconstruction_dtype=torch.float16,
    )
    assert torch.equal(centroids, torch.tensor([1.5, 9.0]))
    assert torch.equal(tied, torch.tensor([1.5, 1.5, 9.0, 9.0]))


def test_state_palette_expands_state_major_branch_rows():
    state_to_palette = torch.tensor([1, 0, 1, 0])
    palettes = torch.tensor([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]])
    expanded = expand_state_palette(
        state_to_palette, palettes, bits=2, window_bits=4
    )
    assert torch.equal(
        expanded,
        torch.tensor(
            [4.0, 5.0, 6.0, 7.0, 0.0, 1.0, 2.0, 3.0] * 2
        ),
    )


def test_state_palette_fit_recovers_shared_branch_rows():
    palettes = torch.tensor(
        [[-4.0, -2.0, 0.0, 2.0], [1.0, 3.0, 5.0, 7.0]]
    )
    assignments = torch.tensor([0, 1, 0, 1])
    targets = palettes.index_select(0, assignments).reshape(-1)
    fit = fit_state_palette(
        targets,
        torch.ones_like(targets),
        bits=2,
        window_bits=4,
        classes=2,
        reconstruction_dtype=torch.float8_e4m3fn,
    )
    assert fit.objective == 0.0
    assert torch.equal(fit.codebook, targets)


def test_state_palette_farthest_initialization_runs():
    palettes = torch.tensor(
        [[-2.0, -1.0, 0.0, 1.0], [2.0, 1.0, 0.0, -1.0]], dtype=torch.float32
    )
    targets = palettes.index_select(0, torch.tensor([0, 1, 0, 1])).flatten()
    fit = fit_state_palette(
        targets,
        torch.ones_like(targets),
        bits=2,
        window_bits=4,
        classes=2,
        reconstruction_dtype=torch.float16,
    )
    assert torch.equal(fit.codebook, targets)
    assert fit.state_to_palette.unique().numel() == 2


def test_state_palette_logical_bytes_include_map_and_packed_rows():
    assert logical_state_palette_bytes(
        states=8192, branches=8, classes=64, packed_classes=False
    ) == 8192 + 64 * 8
    assert logical_state_palette_bytes(
        states=8192, branches=8, classes=64, packed_classes=True
    ) == 8192 * 6 // 8 + 64 * 8
    assert logical_state_palette_bytes(
        states=8192, branches=8, classes=512, packed_classes=False
    ) == 8192 * 2 + 512 * 8


def test_state_scaled_palette_expands_power_of_two_rows():
    expanded = expand_state_scaled_palette(
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([1, 2, 0, 1]),
        torch.tensor(
            [[-1.0, -0.5, 0.0, 0.5], [1.0, 2.0, 3.0, 4.0]]
        ),
        torch.tensor([-1.0, 0.0, 1.0]),
        bits=2,
        window_bits=4,
    )
    assert torch.equal(
        expanded.reshape(4, 4),
        torch.tensor(
            [
                [-1.0, -0.5, 0.0, 0.5],
                [-2.0, -1.0, 0.0, 1.0],
                [0.5, 1.0, 1.5, 2.0],
                [1.0, 2.0, 3.0, 4.0],
            ]
        ),
    )


def test_state_scaled_palette_fit_reuses_one_shape_at_multiple_scales():
    palette = torch.tensor([[-1.0, -0.5, 0.5, 1.0]])
    exponent_values = (-1, 0, 1, 2)
    exponent_codes = torch.tensor([1, 2, 0, 3])
    targets = expand_state_scaled_palette(
        torch.zeros(4, dtype=torch.long),
        exponent_codes,
        palette,
        torch.tensor(exponent_values, dtype=torch.float32),
        bits=2,
        window_bits=4,
    )
    fit = fit_state_scaled_palette(
        targets,
        torch.ones_like(targets),
        bits=2,
        window_bits=4,
        classes=1,
        exponent_values=exponent_values,
        reconstruction_dtype=torch.float8_e4m3fn,
        initial_state_to_palette=torch.zeros(4, dtype=torch.long),
        initial_state_to_exponent=exponent_codes,
        initial_palettes=palette,
    )
    assert fit.objective == 0.0
    assert torch.equal(fit.codebook, targets)


def test_state_scaled_palette_bytes_charge_class_and_exponent_codes():
    assert logical_state_scaled_palette_bytes(
        states=8192,
        branches=8,
        classes=512,
        exponent_levels=16,
        packed_descriptor=False,
    ) == 20 * 1024
    assert logical_state_scaled_palette_bytes(
        states=8192,
        branches=8,
        classes=512,
        exponent_levels=16,
        packed_descriptor=True,
    ) == 17 * 1024
    assert logical_state_scaled_palette_bytes(
        states=8192,
        branches=8,
        classes=512,
        exponent_levels=8,
        packed_descriptor=True,
    ) == 16 * 1024


def test_state_subpalettes_recover_composite_rows():
    state_to_palette = torch.tensor([[0, 1, 0, 1], [1, 1, 0, 0]])
    palettes = torch.tensor(
        [
            [[-4.0, -2.0], [2.0, 4.0]],
            [[-1.0, 0.0], [1.0, 3.0]],
        ]
    )
    targets = expand_state_subpalettes(
        state_to_palette, palettes, bits=2, window_bits=4
    )
    fit = fit_state_subpalettes(
        targets,
        torch.ones_like(targets),
        bits=2,
        window_bits=4,
        classes=2,
        branch_groups=2,
        reconstruction_dtype=torch.float16,
    )
    assert fit.objective == 0.0
    assert torch.equal(fit.codebook, targets)


def test_state_subpalette_bytes_replicate_only_the_state_map():
    assert logical_state_subpalette_bytes(
        states=8192,
        branches=8,
        classes=256,
        branch_groups=2,
        packed_classes=True,
    ) == 2 * 8192 + 256 * 8
    assert logical_state_subpalette_bytes(
        states=8192,
        branches=8,
        classes=128,
        branch_groups=2,
        packed_classes=True,
    ) == 2 * 8192 * 7 // 8 + 128 * 8
