from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from kquant.exl3_reference import reconstruct_trellis_states
from kquant.qsrt import (
    EXPERT_TRELLIS_BYTES,
    EXPERTS_PER_LAYER,
    FORMAT_SECTION_BYTES,
    FORMAT_TABLE_BYTES,
    HOT_METADATA_BYTES,
    INTERMEDIATE_CHANNELS,
    LAYER_HEADER_BYTES,
    MATRIX_TRELLIS_BYTES,
    PAIRS_PER_EXPERT,
    PAIR_BYTES,
    PHASE1_H13_EXPERT_LOCAL_ALPHA,
    PHASE1_H2_EXPERT_LOCAL_ALPHA,
    PHASE1_H2_LOCAL_BASIS,
    PHASE1_H2_SHRINKAGE_POLICY,
    PHASE1_MODE_CANDIDATES,
    PHASE1_MODE_IDS,
    PHASE1_SHARED_RATE_MODE,
    RECORDS_PER_EXPERT,
    RATE_TRANSFER_MODES,
    R0,
    R1,
    R2,
    SCHEMA,
    SHARED_SCALE_BYTES,
    SHARED_SCALE_SECTION_BYTES,
    STORAGE_ALIGNMENT,
    ExpertFormatSpec,
    ModeSpec,
    PackedQSRTTrellis,
    QSRTTrellisDescriptor,
    coupled_intermediate_permutation,
    decode_qsrt_exl3_weight,
    expand_group_order,
    expand_qsrt_hot_metadata,
    expand_record_order,
    importance_group_order,
    matrix_rate_axis,
    mode_from_context_bits,
    pack_qsrt_format_table,
    pack_qsrt_format_section,
    pack_qsrt_trellis,
    pack_qsrt_shared_scale_section,
    pack_trellis_edges,
    rate_transfer_record_order,
    record_repack_order,
    relabel_block_contexts_for_record_order,
    repack_encoded_records,
    repack_rate_axis,
    pair_bits,
    pair_kinds,
    logical_pair_for_slot,
    physical_pair_slot,
    record_bits,
    record_contexts,
    storage_group_order,
    unpack_qsrt_trellis_edges,
    unpack_qsrt_format_section,
    unpack_qsrt_shared_scale_section,
    unpack_qsrt_format_table,
    unpack_qsrt_trellis_states,
    unpack_trellis_edges,
    unpack_trellis_states,
)
from kquant.tp_simulator import situ


def test_mode_schedules_are_fixed_rate_pairs() -> None:
    assert pair_kinds(R0) == ("P33",) * PAIRS_PER_EXPERT
    assert pair_kinds(R1) == ("P24",) + ("P33",) * 11
    assert pair_kinds(R2) == ("P24",) * 2 + ("P33",) * 10


def test_pair_rotation_is_a_per_expert_bijection() -> None:
    for layer, expert in ((1, 0), (1, 17), (40, 285), (92, 895)):
        physical = [
            physical_pair_slot(layer, expert, logical_pair)
            for logical_pair in range(PAIRS_PER_EXPERT)
        ]
        assert sorted(physical) == list(range(PAIRS_PER_EXPERT))
        assert [
            logical_pair_for_slot(layer, expert, rank)
            for rank in physical
        ] == list(range(PAIRS_PER_EXPERT))


def test_pair_rotation_balances_p24_ownership_across_experts() -> None:
    counts = [0] * PAIRS_PER_EXPERT
    r = 2
    for expert in range(EXPERTS_PER_LAYER):
        for rank in range(PAIRS_PER_EXPERT):
            counts[rank] += logical_pair_for_slot(24, expert, rank) < r

    assert max(counts) - min(counts) <= 1
    for r, mode in enumerate(RATE_TRANSFER_MODES):
        assert mode.mode_id == r
        assert mode.name == f"R{r}"
        assert len(record_bits(mode)) == RECORDS_PER_EXPERT
        assert len(pair_bits(mode)) == PAIRS_PER_EXPERT
        assert all(sum(pair) == 6 for pair in pair_bits(mode))
        assert sum(record_bits(mode)) / RECORDS_PER_EXPERT == 3
        assert pair_kinds(mode) == ("P24",) * r + ("P33",) * (12 - r)


def test_qsrt_encoder_policy_is_frozen_to_r012() -> None:
    assert PHASE1_MODE_IDS == (0, 1, 2)
    assert tuple(mode.mode_id for mode in PHASE1_MODE_CANDIDATES) == PHASE1_MODE_IDS
    assert not PHASE1_SHARED_RATE_MODE
    assert PHASE1_H13_EXPERT_LOCAL_ALPHA == 0.0
    assert PHASE1_H2_EXPERT_LOCAL_ALPHA == 0.75
    assert PHASE1_H2_LOCAL_BASIS == "decoded_candidate_post_situ"
    assert PHASE1_H2_SHRINKAGE_POLICY == "weighted_oas_scaled_identity"
    assert len(RATE_TRANSFER_MODES) == 3


def test_context_allocations_map_to_the_rate_transfer_ladder() -> None:
    assert mode_from_context_bits((3, 3, 3, 3)) is R0
    assert mode_from_context_bits((2,) + (3,) * 22 + (4,)) is R1
    assert mode_from_context_bits((2, 2) + (3,) * 20 + (4, 4)) is R2
    with pytest.raises(ValueError, match="monotone"):
        mode_from_context_bits((2, 4) + (3,) * 20 + (2, 4))
    with pytest.raises(ValueError, match="rate-transfer count 6"):
        ModeSpec(1, "bad", (2, 3, 3, 4))


def test_candidate_descriptor_contains_no_tensor_parallel_geometry() -> None:
    descriptor = QSRTTrellisDescriptor(
        mode_id=R2.mode_id,
        rate_axis="k",
        k_tiles=192,
        n_tiles=224,
    )
    assert "tp_size" not in descriptor.to_manifest()


def test_provisional_format_table_supports_separate_r13_r2() -> None:
    formats = ["X4T", R0, R1, (1, 2)] * (EXPERTS_PER_LAYER // 4)
    packed = pack_qsrt_format_table(formats)

    assert packed.dtype == torch.uint8
    assert packed.shape == (FORMAT_TABLE_BYTES,)
    assert packed[:4].tolist() == [0xFF, 0x00, 0x11, 0x12]
    assert unpack_qsrt_format_table(packed) == tuple(
        ["X4T", "R0", "R1", "R1/R2"] * (EXPERTS_PER_LAYER // 4)
    )

    malformed = packed.clone()
    malformed[0] = 0xD0
    with pytest.raises(ValueError, match="invalid rate code"):
        unpack_qsrt_format_table(malformed)
    assert ExpertFormatSpec.compressed(1, 2).code == 0x12
    assert ExpertFormatSpec.from_code(0x12).name == "R1/R2"
    assert ExpertFormatSpec.from_code(0xFF).is_x4t
    with pytest.raises(ValueError, match="896 experts"):
        pack_qsrt_format_table([R0])
    with pytest.raises(TypeError, match="torch.uint8"):
        unpack_qsrt_format_table(packed.int())


def test_format_section_padding_and_hot_slot_expansion_are_canonical() -> None:
    formats = ["X4T", R0, R1, (1, 2)] * (EXPERTS_PER_LAYER // 4)
    section = pack_qsrt_format_section(formats)
    codes, slots = expand_qsrt_hot_metadata(formats)

    assert section.shape == (FORMAT_SECTION_BYTES,)
    assert torch.count_nonzero(section[FORMAT_TABLE_BYTES:]) == 0
    assert unpack_qsrt_format_section(section) == tuple(
        ["X4T", "R0", "R1", "R1/R2"] * (EXPERTS_PER_LAYER // 4)
    )
    assert codes[:8].tolist() == [255, 0, 17, 18, 255, 0, 17, 18]
    assert slots[:8].tolist() == [0, 0, 1, 2, 1, 3, 4, 5]
    assert codes.numel() + slots.numel() * slots.element_size() == HOT_METADATA_BYTES

    malformed = section.clone()
    malformed[-1] = 1
    with pytest.raises(ValueError, match="nonzero alignment padding"):
        unpack_qsrt_format_section(malformed)


def test_shared_scale_section_round_trips_exactly() -> None:
    shared = tuple(
        torch.linspace(-1 + index, 1 + index, 3584, dtype=torch.float16)
        for index in range(3)
    )
    section = pack_qsrt_shared_scale_section(*shared)
    actual = unpack_qsrt_shared_scale_section(section)
    assert all(
        torch.equal(expected, observed)
        for expected, observed in zip(shared, actual, strict=True)
    )

    malformed = section.clone()
    malformed[-1] = 1
    with pytest.raises(ValueError, match="nonzero alignment padding"):
        unpack_qsrt_shared_scale_section(malformed)


def test_matrix_rate_axis_tracks_shared_intermediate_dimension() -> None:
    assert matrix_rate_axis("w1") == "n"
    assert matrix_rate_axis("w3") == "n"
    assert matrix_rate_axis("w2") == "k"
    with pytest.raises(ValueError, match="unknown expert matrix"):
        matrix_rate_axis("router")


@pytest.mark.parametrize(
    ("bits", "expected_prefix"),
    [
        (2, [6939] * 6),
        (3, [30469, 1337, 1337, 14711, 14711, 30469]),
        (4, [17767, 291, -12817, -30293, 17767, 291]),
    ],
)
def test_trellis_edge_pack_matches_native_word_order(
    bits: int, expected_prefix: list[int]
) -> None:
    indices = (torch.arange(256, dtype=torch.int16) % (1 << bits))[None]
    packed = pack_trellis_edges(indices, bits)

    assert packed.shape == (1, 16 * bits)
    assert packed[0, : len(expected_prefix)].tolist() == expected_prefix
    torch.testing.assert_close(unpack_trellis_edges(packed, bits), indices)


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_trellis_edge_pack_round_trips_only_stored_low_bits(bits: int) -> None:
    generator = torch.Generator().manual_seed(100 + bits)
    indices = torch.randint(-32768, 32768, (7, 256), generator=generator)
    packed = pack_trellis_edges(indices, bits)
    unpacked = unpack_trellis_edges(packed, bits)

    torch.testing.assert_close(unpacked.int(), indices.int() & ((1 << bits) - 1))


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_cyclic_trellis_states_round_trip_exactly(bits: int) -> None:
    generator = torch.Generator().manual_seed(150 + bits)
    edges = torch.randint(0, 1 << bits, (5, 256), generator=generator)
    states = reconstruct_trellis_states(edges, bits)
    packed = pack_trellis_edges(states, bits)

    torch.testing.assert_close(unpack_trellis_states(packed, bits), states)


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_cyclic_trellis_state_closed_form_matches_scalar_recurrence(bits: int) -> None:
    generator = torch.Generator().manual_seed(175 + bits)
    edges = torch.randint(0, 1 << bits, (3, 5, 256), generator=generator)

    state = torch.zeros(edges.shape[:-1], dtype=torch.int64)
    for edge in edges[..., -((16 + bits - 1) // bits) :].unbind(dim=-1):
        state = ((state << bits) | edge) & 0xFFFF
    expected = []
    for edge in edges.unbind(dim=-1):
        state = ((state << bits) | edge) & 0xFFFF
        expected.append(state)
    expected_states = torch.stack(expected, dim=-1).to(torch.int16)

    assert torch.equal(reconstruct_trellis_states(edges, bits), expected_states)


def _expected_edge_symbols(
    encoded: torch.Tensor, record_bits: tuple[int, ...], rate_axis: str
) -> torch.Tensor:
    tile_bits = torch.tensor(record_bits).repeat_interleave(8)
    masks = (1 << tile_bits) - 1
    if rate_axis == "k":
        masks = masks[:, None, None]
    else:
        masks = masks[None, :, None]
    return (encoded.to(torch.int64) & masks).to(torch.int16)


def _legal_qsrt_states(
    shape: tuple[int, int, int], record_bits: tuple[int, ...], rate_axis: str, seed: int
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    edges = torch.empty(shape, dtype=torch.int16)
    states = torch.empty_like(edges)
    for record, bits in enumerate(record_bits):
        begin = record * 8
        if rate_axis == "k":
            record_edges = torch.randint(
                0, 1 << bits, (8, shape[1], 256), generator=generator
            ).to(torch.int16)
            edges[begin : begin + 8] = record_edges
            states[begin : begin + 8] = reconstruct_trellis_states(
                record_edges, bits
            )
        else:
            record_edges = torch.randint(
                0, 1 << bits, (shape[0], 8, 256), generator=generator
            ).to(torch.int16)
            edges[:, begin : begin + 8] = record_edges
            states[:, begin : begin + 8] = reconstruct_trellis_states(
                record_edges, bits
            )
    return states


@pytest.mark.parametrize(
    ("rate_axis", "shape"),
    [("k", (192, 224, 256)), ("n", (224, 192, 256))],
)
@pytest.mark.parametrize("mode", [R0, R1, R2])
def test_pair_payload_closes_on_both_logical_axes(
    rate_axis: str, shape: tuple[int, int, int], mode
) -> None:
    encoded = _legal_qsrt_states(
        shape, record_bits(mode), rate_axis, 200 + mode.mode_id
    )
    packed = pack_qsrt_trellis(encoded, mode, rate_axis=rate_axis)
    unpacked = unpack_qsrt_trellis_edges(packed)

    assert packed.descriptor.schema == SCHEMA
    assert packed.payload.numel() == encoded.numel() * 3 // 16
    assert packed.descriptor.payload_bytes == packed.payload.numel() * 2
    assert (
        packed.descriptor.words_per_pair * PAIRS_PER_EXPERT
        == packed.payload.numel()
    )
    assert packed.descriptor.to_manifest()["pair_kinds"] == list(
        pair_kinds(mode)
    )
    torch.testing.assert_close(
        unpacked, _expected_edge_symbols(encoded, record_bits(mode), rate_axis)
    )
    torch.testing.assert_close(unpack_qsrt_trellis_states(packed), encoded)


def test_packer_rejects_noncyclic_states() -> None:
    malformed = torch.zeros((192, 224, 256), dtype=torch.int16)
    malformed[..., 0] = 1

    with pytest.raises(ValueError, match="closed cyclic trellis path"):
        pack_qsrt_trellis(malformed, R2, rate_axis="k")


def test_payload_rejects_wrong_shape_dtype_and_length() -> None:
    encoded = torch.zeros((192, 224, 256), dtype=torch.int16)
    packed = pack_qsrt_trellis(encoded, R2, rate_axis="k")
    suh = torch.ones(3072, dtype=torch.float16)
    svh = torch.ones(3584, dtype=torch.float16)

    with pytest.raises(ValueError, match="length"):
        PackedQSRTTrellis(packed.descriptor, packed.payload[:-1]).validate()
    with pytest.raises(TypeError, match="torch.int16"):
        PackedQSRTTrellis(packed.descriptor, packed.payload.int()).validate()
    with pytest.raises(ValueError, match="3072"):
        pack_qsrt_trellis(encoded[:184], R2, rate_axis="k")
    with pytest.raises(ValueError, match="rate_axis"):
        replace(packed.descriptor, rate_axis="x")
    with pytest.raises(TypeError, match="suh must use torch.float16"):
        decode_qsrt_exl3_weight(packed, suh.float(), svh)
    with pytest.raises(ValueError, match="exactly 3072"):
        decode_qsrt_exl3_weight(packed, suh[:-1], svh)
    nonfinite = svh.clone()
    nonfinite[0] = torch.inf
    with pytest.raises(ValueError, match="non-finite"):
        decode_qsrt_exl3_weight(packed, suh, nonfinite)


def test_context_orders_form_homogeneous_records() -> None:
    block_contexts = torch.arange(24).repeat_interleave(32)
    importance = importance_group_order(block_contexts, R2)
    storage = storage_group_order(block_contexts, R2)
    storage_channels = block_contexts.index_select(0, storage).repeat_interleave(4)

    assert expand_group_order(storage).shape == (INTERMEDIATE_CHANNELS,)
    assert torch.all(storage_channels.reshape(RECORDS_PER_EXPERT, 128).diff(dim=1) == 0)
    assert (
        storage_channels.reshape(RECORDS_PER_EXPERT, 128)[:, 0].tolist()
        == list(record_contexts(R2))
    )

    order = record_repack_order(importance, storage)
    encoded = (
        torch.arange(RECORDS_PER_EXPERT)
        .repeat_interleave(8 * 256)
        .reshape(192, 1, 256)
    )
    repacked = repack_encoded_records(encoded, order, "k")
    assert repacked.reshape(RECORDS_PER_EXPERT, -1)[:, 0].tolist() == order.tolist()


def test_record_repack_rejects_a_split_record() -> None:
    block_contexts = torch.arange(24).repeat_interleave(32)
    source = importance_group_order(block_contexts, R2)
    malformed = source.clone()
    malformed[31], malformed[32] = malformed[32].clone(), malformed[31].clone()

    with pytest.raises(ValueError, match="splits or changes"):
        record_repack_order(source, malformed)


def test_arbitrary_shared_record_map_folds_into_common_permutation() -> None:
    # Start from a deliberately non-contiguous semantic partition so the test
    # exercises relabeling rather than relying on identity group order.
    generator = torch.Generator().manual_seed(47)
    base_contexts = torch.arange(RECORDS_PER_EXPERT).repeat_interleave(32)
    base_contexts = base_contexts[torch.randperm(base_contexts.numel(), generator=generator)]
    donors = (7, 2)
    recipients = (19, 11)
    order = rate_transfer_record_order(donors, recipients)
    relabeled = relabel_block_contexts_for_record_order(base_contexts, order)
    importance = importance_group_order(relabeled, R2)
    ordered_semantic_records = base_contexts.index_select(0, importance).reshape(
        RECORDS_PER_EXPERT, 32
    )[:, 0]

    assert order[:2] == donors
    assert order[-2:] == tuple(reversed(recipients))
    assert ordered_semantic_records.tolist() == list(order)

    storage = storage_group_order(relabeled, R2)
    physical_records = base_contexts.index_select(0, storage).reshape(
        RECORDS_PER_EXPERT, 32
    )[:, 0]
    assert list(zip(physical_records[0:4:2].tolist(), physical_records[1:4:2].tolist())) == list(
        zip(donors, recipients, strict=True)
    )


@pytest.mark.parametrize(
    "donors,recipients,middle,error",
    [
        ((1,), (), None, "counts must match"),
        ((1,), (1,), None, "unique and disjoint"),
        ((24,), (1,), None, "record IDs"),
        ((1,), (2,), tuple(range(21)), "middle_records"),
    ],
)
def test_arbitrary_record_map_rejects_malformed_assignments(
    donors, recipients, middle, error
) -> None:
    with pytest.raises(ValueError, match=error):
        rate_transfer_record_order(
            donors, recipients, middle_records=middle
        )


def test_record_repack_moves_matching_scale_or_weight_axis() -> None:
    order = torch.arange(RECORDS_PER_EXPERT - 1, -1, -1)
    channels = expand_record_order(order)
    vector = torch.arange(INTERMEDIATE_CHANNELS)
    matrix = vector.repeat(3, 1)

    torch.testing.assert_close(repack_rate_axis(vector, order, 0), vector[channels])
    torch.testing.assert_close(
        repack_rate_axis(matrix, order, 1), matrix.index_select(1, channels)
    )


def test_coupled_permutation_preserves_situ_expert_function() -> None:
    generator = torch.Generator().manual_seed(300)
    hidden = 7
    intermediate = 64
    inputs = torch.randn(5, hidden, dtype=torch.float64, generator=generator)
    w1 = torch.randn(intermediate, hidden, dtype=torch.float64, generator=generator)
    w3 = torch.randn(intermediate, hidden, dtype=torch.float64, generator=generator)
    w2 = torch.randn(hidden, intermediate, dtype=torch.float64, generator=generator)
    permutation = torch.randperm(intermediate, generator=generator)

    reference_middle = situ(F.linear(inputs, w1), F.linear(inputs, w3))
    reference = F.linear(reference_middle, w2)
    permuted_w1, permuted_w3, permuted_w2 = coupled_intermediate_permutation(
        w1, w3, w2, permutation
    )
    permuted_middle = situ(
        F.linear(inputs, permuted_w1), F.linear(inputs, permuted_w3)
    )
    actual = F.linear(permuted_middle, permuted_w2)

    torch.testing.assert_close(permuted_middle, reference_middle[:, permutation])
    torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
