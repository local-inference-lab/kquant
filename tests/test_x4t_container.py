from __future__ import annotations

import pytest
import torch
from safetensors import safe_open

from kquant.x4t import (
    X4T_LAYER_FIXED_BYTES,
    X4T_MATRIX_ORDER,
    X4T_SAFETENSORS_SCHEMA,
    X4TLayerReader,
    X4TLayerWriter,
    X4TMatrix,
    pack_x4t_scale_components,
    partition_x4t_components,
    x4t_expert_storage_bytes,
    x4t_layer_path,
    x4t_scale_storage_bytes,
)


def _production_matrix(
    matrix: str, *, packed_seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    from kquant import constants as C

    out_features, in_features = C.EXPERT_SHAPES[matrix]
    generator = torch.Generator().manual_seed(packed_seed)
    packed = torch.randint(
        0,
        256,
        (out_features, in_features // 2),
        dtype=torch.uint8,
        generator=generator,
    )
    scale = torch.full(
        (out_features, in_features // C.MXFP4_BLOCK), 120, dtype=torch.uint8
    )
    scale[:, 1::2] = 121
    scale[0, 0] = 123
    return packed, scale


def test_x4t_layer_is_canonical_safetensors_and_round_trips(tmp_path) -> None:
    destination = x4t_layer_path(tmp_path, 24)
    source = {
        matrix: _production_matrix(matrix, packed_seed=index + 1)
        for index, matrix in enumerate(X4T_MATRIX_ORDER)
    }
    with X4TLayerWriter(destination, layer=24) as writer:
        for matrix in X4T_MATRIX_ORDER:
            writer.add(6, matrix, *source[matrix])

    assert destination.suffix == ".safetensors"
    with safe_open(destination, framework="pt", device="cpu") as handle:
        assert handle.metadata()["schema"] == X4T_SAFETENSORS_SCHEMA
        assert tuple(handle.get_tensor("expert_ids").tolist()) == (6,)
        assert set(handle.keys()) == {
            "expert_ids",
            *{
                f"{matrix}.{part}"
                for matrix in X4T_MATRIX_ORDER
                for part in (
                    "packed",
                    "scale_fixed",
                    "scale_exceptions",
                    "scale_exception_offsets",
                )
            },
        }

    reader = X4TLayerReader(destination)
    assert reader.layer == 24
    assert reader.matrix_count == 3
    assert not reader.has(5, "w1")
    assert reader.has(6, "w2")
    with pytest.raises(KeyError):
        reader.read(5, "w1")
    for matrix in X4T_MATRIX_ORDER:
        decoded = reader.read(6, matrix)
        assert torch.equal(decoded.packed, source[matrix][0])
        assert torch.equal(decoded.scale, source[matrix][1])

    expected = X4T_LAYER_FIXED_BYTES + x4t_expert_storage_bytes(
        {matrix: source[matrix][1] for matrix in X4T_MATRIX_ORDER}
    )
    assert destination.stat().st_size == expected


def test_x4t_empty_layer_is_valid_safetensors(tmp_path) -> None:
    destination = x4t_layer_path(tmp_path, 1)
    with X4TLayerWriter(destination, layer=1):
        pass
    reader = X4TLayerReader(destination)
    assert reader.expert_ids == ()
    assert reader.file_bytes == X4T_LAYER_FIXED_BYTES


def test_x4t_layer_rejects_corrupt_safetensors_header(tmp_path) -> None:
    destination = x4t_layer_path(tmp_path, 1)
    with X4TLayerWriter(destination, layer=1):
        pass
    with destination.open("r+b") as handle:
        handle.seek(8)
        handle.write(b"!")
    with pytest.raises(Exception):
        X4TLayerReader(destination)


def test_x4t_scale_accounting_matches_safetensors_components() -> None:
    _, scale = _production_matrix("w2", packed_seed=9)
    fixed, exceptions = pack_x4t_scale_components(scale)
    assert x4t_scale_storage_bytes(scale) == fixed.numel() + 4 * exceptions.numel()


def test_x4t_layer_writer_requires_complete_canonical_triplets(tmp_path) -> None:
    destination = x4t_layer_path(tmp_path, 1)
    packed, scale = _production_matrix("w1", packed_seed=5)
    with pytest.raises(ValueError, match="w1, w3, and w2"):
        with X4TLayerWriter(destination, layer=1) as writer:
            writer.add(0, "w1", packed, scale)
            writer.add(1, "w1", packed, scale)


@pytest.mark.parametrize("matrix", ("w1", "w2"))
@pytest.mark.parametrize("shard_count", (5, 12))
def test_x4t_matrix_partitions_cover_canonical_intermediate_axis(
    matrix: str,
    shard_count: int,
) -> None:
    packed, scale = _production_matrix(matrix, packed_seed=17)
    source = X4TMatrix(matrix=matrix, packed=packed, scale=scale)
    pieces = [
        partition_x4t_components(source, matrix, shard_count, shard_index)
        for shard_index in range(shard_count)
    ]
    dimension = 0 if matrix == "w1" else 1

    assert torch.equal(torch.cat([part[0] for part in pieces], dim=dimension), packed)
    assert torch.equal(torch.cat([part[1] for part in pieces], dim=dimension), scale)
    widths = [part[0].shape[dimension] for part in pieces]
    assert max(widths) - min(widths) <= (32 if matrix == "w1" else 16)
    if shard_count == 12:
        assert len(set(widths)) == 1
