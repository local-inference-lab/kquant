from __future__ import annotations

import pytest
import torch

from kquant.exl3_reference import reconstruct_trellis_states
from kquant.fruit_qsrt import (
    FRUIT_QSRT_ARTIFACT_TENSORS,
    FRUIT_QSRT_ATOM_BUNDLE_BYTES,
    FRUIT_QSRT_ATOM_TENSOR,
    FRUIT_QSRT_ATOMS_PER_EXPERT,
    FRUIT_QSRT_FORMAT_TENSOR,
    FRUIT_QSRT_MATRIX_BYTES,
    FRUIT_QSRT_PAIR_COUNT,
    FRUIT_QSRT_PAIR_WORDS,
    FRUIT_QSRT_SCHEMA,
    FruitExpertEncoding,
    FruitMatrixEncoding,
    decode_fruit_matrix,
    fruit_pair_modes,
    fruit_transform_seeds,
    fruit_weight_permutations,
    pack_fruit_atom_layer,
    pack_fruit_expert_atoms,
    pack_fruit_local_scale_atoms,
    pack_fruit_matrix_atoms,
    pack_fruit_trellis,
    rotate_fruit_expert_atoms,
    stack_fruit_layer,
    unpack_fruit_local_scale_atoms,
    unpack_fruit_matrix_atoms,
    unpack_fruit_trellis,
    unrotate_fruit_expert_atoms,
)
from kquant.logical_qsrt import (
    FRUIT_QSRT_GEOMETRY,
    R0,
    R1,
    R2,
    fruit_rate_modes,
    paired_record_bits,
)


def _closed_states(mode, rate_axis: str) -> torch.Tensor:
    shape = (32, 64, 256) if rate_axis == "k" else (64, 32, 256)
    states = torch.empty(shape, dtype=torch.int16)
    generator = torch.Generator().manual_seed(17 + mode.transfer)
    for record, bits in enumerate(paired_record_bits(mode)):
        begin = record * 8
        edge_shape = (8, 64, 256) if rate_axis == "k" else (64, 8, 256)
        edges = torch.randint(
            0,
            1 << bits,
            edge_shape,
            dtype=torch.int16,
            generator=generator,
        )
        reconstructed = reconstruct_trellis_states(edges, bits)
        states.narrow(0 if rate_axis == "k" else 1, begin, 8).copy_(reconstructed)
    return states


@pytest.mark.parametrize("mode", fruit_rate_modes())
@pytest.mark.parametrize("rate_axis", ("k", "n"))
def test_fixed_pair_payload_round_trips_every_fruit_mode(mode, rate_axis: str) -> None:
    states = _closed_states(mode, rate_axis)
    payload = pack_fruit_trellis(states, mode, rate_axis=rate_axis)

    assert payload.dtype == torch.int16
    assert payload.is_contiguous()
    assert payload.shape == (FRUIT_QSRT_PAIR_COUNT, FRUIT_QSRT_PAIR_WORDS)
    assert payload.numel() * payload.element_size() == FRUIT_QSRT_MATRIX_BYTES
    assert torch.equal(unpack_fruit_trellis(payload, mode, rate_axis=rate_axis), states)


def test_pair_mode_metadata_is_addressable_per_physical_pair() -> None:
    assert torch.equal(fruit_pair_modes(R0), torch.tensor([0, 0], dtype=torch.int32))
    assert torch.equal(fruit_pair_modes(R1), torch.tensor([1, 0], dtype=torch.int32))
    assert torch.equal(fruit_pair_modes(R2), torch.tensor([1, 1], dtype=torch.int32))

    with pytest.raises(ValueError, match="four-record"):
        fruit_pair_modes(type(R0)(6, 0))


def test_decode_fruit_matrix_accepts_both_rate_axes() -> None:
    for matrix, rate_axis, shape in (
        ("w1", "n", (1024, 512)),
        ("w3", "n", (1024, 512)),
        ("w2", "k", (512, 1024)),
    ):
        states = _closed_states(R1, rate_axis)
        payload = pack_fruit_trellis(states, R1, rate_axis=rate_axis)
        decoded = decode_fruit_matrix(
            payload,
            torch.ones(shape[0], dtype=torch.float16),
            torch.ones(shape[1], dtype=torch.float16),
            matrix=matrix,
            mode=R1,
        )
        assert decoded.shape == shape
        assert decoded.dtype == torch.float32
        assert bool(torch.all(torch.isfinite(decoded)))


def test_weight_permutation_is_shared_and_moves_whole_four_channel_groups() -> None:
    generator = torch.Generator().manual_seed(29)
    w1 = torch.randn((512, 1024), generator=generator)
    w3 = torch.randn((512, 1024), generator=generator)
    w2 = torch.randn((1024, 512), generator=generator)

    encoder, physical, scores = fruit_weight_permutations(w1, w3, w2)

    assert encoder.shape == physical.shape == (512,)
    assert scores.shape == (128,)
    assert torch.equal(torch.sort(encoder).values, torch.arange(512))
    assert torch.equal(torch.sort(physical).values, torch.arange(512))
    assert torch.equal(
        encoder.reshape(-1, 4)[:, 1:] - encoder.reshape(-1, 4)[:, :-1],
        torch.ones((128, 3), dtype=torch.long),
    )
    assert torch.equal(
        physical.reshape(-1, 4)[:, 1:] - physical.reshape(-1, 4)[:, :-1],
        torch.ones((128, 3), dtype=torch.long),
    )

    inputs = torch.randn((3, 1024), generator=generator)
    canonical = torch.nn.functional.silu(inputs @ w1.T) * (inputs @ w3.T)
    expected = canonical @ w2.T
    permuted = torch.nn.functional.silu(inputs @ w1[physical].T) * (
        inputs @ w3[physical].T
    )
    actual = permuted @ w2[:, physical].T
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=3e-2)


def _matrix_encoding(matrix: str, mode) -> FruitMatrixEncoding:
    rate_axis = "k" if matrix == "w2" else "n"
    states = _closed_states(mode, rate_axis)
    shape = (512, 1024) if matrix == "w2" else (1024, 512)
    return FruitMatrixEncoding(
        matrix=matrix,
        mode=mode,
        trellis=pack_fruit_trellis(states, mode, rate_axis=rate_axis),
        suh=torch.ones(shape[0], dtype=torch.float16),
        svh=torch.ones(shape[1], dtype=torch.float16),
        reconstruction=torch.zeros((1024, 512) if matrix == "w2" else (512, 1024)),
        coding={"matrix": matrix},
    )


def _expert(
    expert: int, *, layer: int = 3, r13: int = 1, r2: int = 2
) -> FruitExpertEncoding:
    modes = fruit_rate_modes()
    permutation = torch.arange(FRUIT_QSRT_GEOMETRY.intermediate_channels)
    return FruitExpertEncoding(
        layer=layer,
        expert=expert,
        r13=r13,
        r2=r2,
        encoder_permutation=permutation,
        physical_permutation=permutation,
        matrices={
            "w1": _matrix_encoding("w1", modes[r13]),
            "w3": _matrix_encoding("w3", modes[r13]),
            "w2": _matrix_encoding("w2", modes[r2]),
        },
        selection={"policy": "test"},
    )


def test_layer_artifact_stacks_runtime_tensors_without_losing_pair_dimension() -> None:
    artifact = stack_fruit_layer(3, (_expert(0), _expert(255, r13=0, r2=1)))

    assert artifact.expert_ids == (0, 255)
    assert tuple(artifact.tensors) == FRUIT_QSRT_ARTIFACT_TENSORS
    assert artifact.tensors["w13_trellis"].shape == (
        2,
        2,
        FRUIT_QSRT_PAIR_COUNT,
        FRUIT_QSRT_PAIR_WORDS,
    )
    assert artifact.tensors["w2_trellis"].shape == (
        2,
        FRUIT_QSRT_PAIR_COUNT,
        FRUIT_QSRT_PAIR_WORDS,
    )
    assert artifact.tensors["fc1_pair_modes"].tolist() == [[1, 0], [0, 0]]
    assert artifact.tensors["fc2_pair_modes"].tolist() == [[1, 1], [1, 0]]
    assert artifact.tensors["intermediate_rotations"].shape == (2, 1536)
    assert artifact.manifest()["schema"] == FRUIT_QSRT_SCHEMA


def test_layer_artifact_fails_closed_on_mixed_or_unsorted_assignments() -> None:
    with pytest.raises(ValueError, match="cannot mix"):
        stack_fruit_layer(3, (_expert(0), _expert(1, layer=4)))
    with pytest.raises(ValueError, match="sorted"):
        stack_fruit_layer(3, (_expert(2), _expert(1)))
    with pytest.raises(ValueError, match="unique"):
        stack_fruit_layer(3, (_expert(1), _expert(1)))


def test_transform_seed_contract_is_stable_and_rejects_wrong_geometry() -> None:
    assert fruit_transform_seeds(3, 0, "w1") == fruit_transform_seeds(3, 0, "w1")
    assert fruit_transform_seeds(3, 0, "w1") != fruit_transform_seeds(3, 1, "w1")
    assert fruit_transform_seeds(3, 0, "w1") != fruit_transform_seeds(3, 0, "w2")
    with pytest.raises(ValueError, match="layer"):
        fruit_transform_seeds(2, 0, "w1")
    with pytest.raises(ValueError, match="expert"):
        fruit_transform_seeds(3, 256, "w1")


def test_trellis_payload_rejects_wrong_dtype_shape_and_geometry() -> None:
    states = _closed_states(R0, "n")
    payload = pack_fruit_trellis(states, R0, rate_axis="n")
    with pytest.raises(TypeError, match="int16"):
        unpack_fruit_trellis(payload.int(), R0, rate_axis="n")
    with pytest.raises(ValueError, match="shape"):
        unpack_fruit_trellis(payload.flatten(), R0, rate_axis="n")
    with pytest.raises(ValueError, match="geometry"):
        pack_fruit_trellis(states[:, :-1], R0, rate_axis="n")


@pytest.mark.parametrize("mode", fruit_rate_modes())
@pytest.mark.parametrize("matrix,rate_axis", (("w1", "n"), ("w3", "n"), ("w2", "k")))
def test_matrix_atoms_round_trip_every_mode_and_orientation(
    mode, matrix: str, rate_axis: str
) -> None:
    payload = pack_fruit_trellis(
        _closed_states(mode, rate_axis), mode, rate_axis=rate_axis
    )

    atoms = pack_fruit_matrix_atoms(payload, mode, matrix=matrix)

    assert atoms.shape[0] == FRUIT_QSRT_ATOMS_PER_EXPERT
    assert torch.equal(unpack_fruit_matrix_atoms(atoms, mode, matrix=matrix), payload)


def test_local_scale_and_physical_rotation_atoms_round_trip() -> None:
    scale = torch.arange(FRUIT_QSRT_GEOMETRY.intermediate_channels, dtype=torch.float16)
    scale_atoms = pack_fruit_local_scale_atoms(scale)
    logical = torch.arange(FRUIT_QSRT_ATOMS_PER_EXPERT * 7, dtype=torch.int32).reshape(
        FRUIT_QSRT_ATOMS_PER_EXPERT, 7
    )

    assert torch.equal(unpack_fruit_local_scale_atoms(scale_atoms), scale)
    assert torch.equal(
        unrotate_fruit_expert_atoms(
            rotate_fruit_expert_atoms(logical, layer=13, expert=255),
            layer=13,
            expert=255,
        ),
        logical,
    )


def test_atom_layer_matches_independently_packed_experts() -> None:
    encodings = (_expert(0, r13=0, r2=1), _expert(1, r13=2, r2=2))
    artifact = stack_fruit_layer(3, encodings)

    packed = pack_fruit_atom_layer(artifact.tensors, layer=3)

    assert packed[FRUIT_QSRT_FORMAT_TENSOR][:2].tolist() == [0x01, 0x22]
    atom_slab = packed[FRUIT_QSRT_ATOM_TENSOR][
        :, : len(encodings) * FRUIT_QSRT_ATOM_BUNDLE_BYTES
    ].reshape(
        FRUIT_QSRT_ATOMS_PER_EXPERT,
        len(encodings),
        FRUIT_QSRT_ATOM_BUNDLE_BYTES,
    )
    for index, encoding in enumerate(encodings):
        expected = pack_fruit_expert_atoms(
            layer=3,
            expert=encoding.expert,
            r13=encoding.r13,
            r2=encoding.r2,
            w13_trellis=artifact.tensors["w13_trellis"][:, index].contiguous(),
            w2_trellis=artifact.tensors["w2_trellis"][index].contiguous(),
            intermediate_rotations=artifact.tensors["intermediate_rotations"][
                index
            ].contiguous(),
        )
        assert torch.equal(atom_slab[:, index], expected)
