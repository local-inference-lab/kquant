from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from kquant.qsrt import (
    EXPERT_TRELLIS_BYTES,
    INTERMEDIATE_CHANNELS,
    LOCAL_SCALE_VECTORS,
    R0,
    R1,
    R2,
    ExpertFormatSpec,
    QSRTTrellisDescriptor,
    PackedQSRTTrellis,
    SCALE_BYTES,
    STORAGE_ALIGNMENT,
)
from kquant.pack.qsrt_atoms import (
    QSRTAtomLayerReader,
    QSRTAtomLayerSpec,
    assemble_candidate_atoms,
    materialize_atom_layer,
)
from kquant.pack.qsrt_candidates import candidate_tensor_name
from kquant.qsrt_storage import (
    ATOMS_PER_EXPERT,
    ATOMS_PER_RECORD_PAIR,
    ATOM_BUNDLE_BYTES,
    ATOM_CHANNELS,
    EQUAL_SHARD_COUNTS,
    QSRTLayerLayout,
    QSRTLayerHeader,
    atom_components,
    atom_rates,
    balanced_atom_partition,
    logical_atom_for_slot,
    logical_atom_index,
    pack_local_scale_atoms,
    pack_matrix_atoms,
    pair_rotation,
    physical_atom_slot,
    rotate_expert_atoms,
    unpack_local_scale_atoms,
    unpack_matrix_atoms,
    unrotate_expert_atoms,
)


def test_balanced_atoms_cover_every_stripe_once() -> None:
    atoms = [
        logical_atom_index(pair, stripe)
        for pair in range(12)
        for stripe in range(ATOMS_PER_RECORD_PAIR)
    ]
    assert atoms == list(range(ATOMS_PER_EXPERT))
    assert [atom_components(atom) for atom in atoms] == [
        (pair, stripe)
        for pair in range(12)
        for stripe in range(ATOMS_PER_RECORD_PAIR)
    ]


def test_every_atom_has_a_fixed_six_bit_rate_sum() -> None:
    for mode, shifted_pairs in ((R0, 0), (R1, 1), (R2, 2)):
        pairs = [atom_rates(mode, pair * ATOMS_PER_RECORD_PAIR) for pair in range(12)]
        assert pairs[:shifted_pairs] == [(2, 4)] * shifted_pairs
        assert pairs[shifted_pairs:] == [(3, 3)] * (12 - shifted_pairs)
        assert all(sum(atom_rates(mode, atom)) == 6 for atom in range(96))


def test_physical_atom_rotation_is_bijective_and_keeps_pair_stripes_contiguous() -> None:
    for layer, expert in ((1, 0), (1, 17), (40, 285), (92, 895)):
        physical = [
            physical_atom_slot(layer, expert, logical) for logical in range(96)
        ]
        assert sorted(physical) == list(range(96))
        assert [
            logical_atom_for_slot(layer, expert, slot) for slot in physical
        ] == list(range(96))

        for shard in range(12):
            logical = [
                logical_atom_for_slot(
                    layer,
                    expert,
                    shard * ATOMS_PER_RECORD_PAIR + stripe,
                )
                for stripe in range(ATOMS_PER_RECORD_PAIR)
            ]
            assert [atom_components(atom)[0] for atom in logical] == [
                (shard - pair_rotation(layer, expert)) % 12
            ] * ATOMS_PER_RECORD_PAIR
            assert [atom_components(atom)[1] for atom in logical] == list(
                range(ATOMS_PER_RECORD_PAIR)
            )


def test_atomization_preserves_exact_compressed_expert_bytes() -> None:
    expected = (
        EXPERT_TRELLIS_BYTES
        + LOCAL_SCALE_VECTORS * INTERMEDIATE_CHANNELS * SCALE_BYTES
    )
    assert ATOMS_PER_EXPERT * ATOM_BUNDLE_BYTES == expected == 12_404_736


@pytest.mark.parametrize("compressed", (1, 512, 700, 800, 896))
def test_equal_shards_are_contiguous_aligned_extents(compressed: int) -> None:
    layout = QSRTLayerLayout(compressed, 896 - compressed)
    for shard_count in EQUAL_SHARD_COUNTS:
        extents = [
            layout.shard_extent(shard_count, shard_index, require_equal=True)
            for shard_index in range(shard_count)
        ]
        assert {extent.atom_slots for extent in extents} == {96 // shard_count}
        assert {extent.intermediate_channels for extent in extents} == {
            INTERMEDIATE_CHANNELS // shard_count
        }
        assert {extent.extent_bytes for extent in extents} == {
            (96 // shard_count) * layout.atom_slot_stride_bytes
        }
        assert extents[0].offset_bytes % STORAGE_ALIGNMENT == 0
        assert all(
            left.offset_bytes + left.extent_bytes == right.offset_bytes
            for left, right in zip(extents, extents[1:])
        )
        assert sum(extent.payload_bytes for extent in extents) == (
            layout.compressed_payload_bytes
        )
        assert sum(extent.extent_bytes for extent in extents) == (
            96 * layout.atom_slot_stride_bytes
        )


def test_twelve_way_view_is_eight_atoms_and_256_channels_per_shard() -> None:
    layout = QSRTLayerLayout(800, 96)
    extent = layout.shard_extent(12, 7)
    assert extent.first_atom_slot == 56
    assert extent.atom_slots == 8
    assert extent.intermediate_channels == 256


def test_uneven_whole_atom_partitions_cover_every_atom_once() -> None:
    layout = QSRTLayerLayout(800, 96)
    for shard_count in (5, 17, 64):
        extents = [
            layout.shard_extent(shard_count, shard_index)
            for shard_index in range(shard_count)
        ]
        assert [(item.first_atom_slot, item.atom_slots) for item in extents] == [
            (*balanced_atom_partition(shard_count, shard_index),)
            for shard_index in range(shard_count)
        ]
        assert sum(item.atom_slots for item in extents) == ATOMS_PER_EXPERT
        assert max(item.atom_slots for item in extents) - min(
            item.atom_slots for item in extents
        ) <= 1
        assert all(
            left.offset_bytes + left.extent_bytes == right.offset_bytes
            for left, right in zip(extents, extents[1:])
        )
    with pytest.raises(ValueError, match="do not divide"):
        layout.shard_extent(64, 0, require_equal=True)


def test_tp_free_layer_header_round_trip_and_offsets() -> None:
    layout = QSRTLayerLayout(700, 196)
    header = QSRTLayerHeader(24, layout)
    decoded = QSRTLayerHeader.from_bytes(header.to_bytes())
    assert decoded == header
    assert "tp_size" not in decoded.layout.to_manifest()
    assert layout.atom_bundle_offset(0, 0) % STORAGE_ALIGNMENT == 0
    assert layout.atom_bundle_offset(1, 0) % STORAGE_ALIGNMENT == 0
    assert layout.atom_bundle_offset(4, 17) == (
        layout.shard_extent(24, 1).offset_bytes + 17 * ATOM_BUNDLE_BYTES
    )


@pytest.mark.parametrize("mode", (R0, R1, R2))
@pytest.mark.parametrize("rate_axis", ("k", "n"))
def test_matrix_atom_transpose_is_bit_exact(mode, rate_axis: str) -> None:
    descriptor = QSRTTrellisDescriptor(
        mode_id=mode.mode_id,
        rate_axis=rate_axis,
        k_tiles=192 if rate_axis == "k" else 224,
        n_tiles=224 if rate_axis == "k" else 192,
    )
    generator = torch.Generator().manual_seed(1000 + mode.mode_id)
    payload = torch.randint(
        -(1 << 15),
        1 << 15,
        (descriptor.payload_words,),
        dtype=torch.int16,
        generator=generator,
    )
    packed = PackedQSRTTrellis(descriptor, payload)
    atoms = pack_matrix_atoms(packed)
    restored = unpack_matrix_atoms(atoms, descriptor)
    assert torch.equal(restored.payload, payload)


def test_local_scale_atom_transpose_is_bit_exact() -> None:
    scale = torch.arange(INTERMEDIATE_CHANNELS, dtype=torch.int16).view(torch.float16)
    atoms = pack_local_scale_atoms(scale)
    restored = unpack_local_scale_atoms(atoms)
    assert torch.equal(restored.view(torch.int16), scale.view(torch.int16))


def test_physical_atom_rotation_round_trip_preserves_arbitrary_payload() -> None:
    logical = torch.arange(96 * 7, dtype=torch.int64).reshape(96, 7)
    physical = rotate_expert_atoms(logical, 24, 285)
    restored = unrotate_expert_atoms(physical, 24, 285)
    assert torch.equal(restored, logical)


def test_atom_layer_materialization_has_tp_free_bytes_and_direct_views(tmp_path) -> None:
    layer = 24
    expert = 0
    format_spec = ExpertFormatSpec.compressed(1, 2)
    formats = (format_spec,) + (ExpertFormatSpec.x4t(),) * 895
    candidate_root = tmp_path / "pool"
    candidate_dir = candidate_root / "candidates"
    candidate_dir.mkdir(parents=True)
    tensors: dict[str, torch.Tensor] = {}
    nested: dict[str, dict[str, torch.Tensor]] = {}
    for matrix in ("w1", "w3", "w2"):
        parts = {
            "trellis": torch.zeros(
                EXPERT_TRELLIS_BYTES // 3 // torch.int16.itemsize,
                dtype=torch.int16,
            ),
            "suh": torch.ones(
                3072 if matrix == "w2" else 3584, dtype=torch.float16
            ),
            "svh": torch.ones(
                3584 if matrix == "w2" else 3072, dtype=torch.float16
            ),
        }
        nested[matrix] = parts
        for part, value in parts.items():
            tensors[candidate_tensor_name(layer, expert, matrix, part)] = value
    save_file(tensors, candidate_dir / f"qsrt-layer-{layer:05d}.safetensors")

    expected, _ = assemble_candidate_atoms(
        layer=layer,
        expert=expert,
        format_spec=format_spec,
        tensors=nested,
    )
    destination = tmp_path / "layer.safetensors"
    materialize_atom_layer(
        candidate_root,
        destination,
        QSRTAtomLayerSpec(layer, formats, (expert,), tuple(range(1, 896))),
    )
    with QSRTAtomLayerReader(destination) as reader:
        assert reader.header.layout == QSRTLayerLayout(1, 895)
        assert reader.formats[0] == "R1/R2"
        assert reader.formats[1:] == ("X4T",) * 895
        assert all(
            torch.equal(scale, torch.ones(3584, dtype=torch.float16))
            for scale in reader.shared_scales
        )
        assert torch.equal(reader.read_expert_atoms(0), expected)
        shards = [
            reader.read_shard(12, shard_index, require_equal=True)
            for shard_index in range(12)
        ]
        assert torch.equal(torch.cat(shards, dim=0)[:, 0], expected)

        uneven = [reader.read_shard(5, shard_index) for shard_index in range(5)]
        assert torch.equal(torch.cat(uneven, dim=0)[:, 0], expected)
