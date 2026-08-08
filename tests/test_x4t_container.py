from __future__ import annotations

import struct

import pytest
import torch

from kquant.x4t import (
    X4TMatrix,
    X4TLayerReader,
    X4TLayerWriter,
    pack_x4t_matrix_record,
    partition_x4t_components,
    unpack_x4t_matrix_record,
    x4t_matrix_storage_bytes,
    x4t_record_bpw,
    x4t_scale_storage_bytes,
)


def _matrix(rows: int = 16, scale_columns: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(123)
    packed = torch.randint(
        0,
        256,
        (rows, scale_columns * 16),
        dtype=torch.uint8,
        generator=generator,
    )
    scale = torch.full((rows, scale_columns), 120, dtype=torch.uint8)
    scale[:, 1::2] = 121
    scale[0, 0] = 123
    return packed, scale


def test_x4t_matrix_record_is_exact_and_deterministic() -> None:
    packed, scale = _matrix()

    payload = pack_x4t_matrix_record(
        "w1", packed, scale, production_shape=False
    )
    decoded = unpack_x4t_matrix_record(
        payload, expected_matrix="w1", production_shape=False
    )

    assert torch.equal(decoded.packed, packed)
    assert torch.equal(decoded.scale, scale)
    assert payload == pack_x4t_matrix_record(
        "w1", packed.clone(), scale.clone(), production_shape=False
    )
    assert x4t_record_bpw(payload) > 4.0


def test_x4t_record_rejects_matrix_slot_mismatch() -> None:
    packed, scale = _matrix()
    payload = pack_x4t_matrix_record(
        "w1", packed, scale, production_shape=False
    )

    with pytest.raises(ValueError, match="directory slot"):
        unpack_x4t_matrix_record(
            payload, expected_matrix="w2", production_shape=False
        )


def test_x4t_record_rejects_noncanonical_reserved_bytes() -> None:
    packed, scale = _matrix()
    payload = bytearray(
        pack_x4t_matrix_record("w1", packed, scale, production_shape=False)
    )
    payload[63] = 1

    with pytest.raises(ValueError, match="noncanonical"):
        unpack_x4t_matrix_record(bytes(payload), production_shape=False)


def test_x4t_layer_round_trip_and_sparse_lookup(tmp_path) -> None:
    packed, scale = _matrix()
    destination = tmp_path / "x4t-layer.bin"
    with X4TLayerWriter(destination, layer=24) as writer:
        # Canonical matrix order is w1, w3, w2.
        writer.add(6, "w1", *_production_matrix("w1", packed_seed=1))
        writer.add(6, "w3", *_production_matrix("w3", packed_seed=2))
        writer.add(6, "w2", *_production_matrix("w2", packed_seed=3))

    reader = X4TLayerReader(destination)
    assert reader.layer == 24
    assert reader.record_count == 3
    assert not reader.has(5, "w1")
    assert reader.has(6, "w2")
    with pytest.raises(KeyError):
        reader.read(5, "w1")
    decoded = reader.read(6, "w1")
    expected_packed, expected_scale = _production_matrix("w1", packed_seed=1)
    assert torch.equal(decoded.packed, expected_packed)
    assert torch.equal(decoded.scale, expected_scale)


def _production_matrix(matrix: str, *, packed_seed: int) -> tuple[torch.Tensor, torch.Tensor]:
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
    # Keep the fixture compressible while exercising both adjacent palettes.
    scale = torch.full(
        (out_features, in_features // C.MXFP4_BLOCK), 120, dtype=torch.uint8
    )
    scale[:, 1::2] = 121
    scale[0, 0] = 123
    return packed, scale


def test_x4t_layer_rejects_directory_corruption(tmp_path) -> None:
    destination = tmp_path / "x4t-layer.bin"
    with X4TLayerWriter(destination, layer=1):
        pass
    payload = bytearray(destination.read_bytes())
    payload[4096] = 1
    destination.write_bytes(payload)

    with pytest.raises(ValueError, match="directory checksum"):
        X4TLayerReader(destination)


def test_x4t_layer_rejects_record_corruption(tmp_path) -> None:
    destination = tmp_path / "x4t-layer.bin"
    packed, scale = _production_matrix("w1", packed_seed=4)
    with X4TLayerWriter(destination, layer=1) as writer:
        entry = writer.add(0, "w1", packed, scale)
    with destination.open("r+b") as handle:
        handle.seek(entry.offset + entry.length - 1)
        byte = handle.read(1)
        handle.seek(entry.offset + entry.length - 1)
        handle.write(bytes([byte[0] ^ 1]))

    with pytest.raises(ValueError, match="record checksum"):
        X4TLayerReader(destination)


def test_x4t_scale_only_accounting_matches_written_record() -> None:
    packed, scale = _production_matrix("w2", packed_seed=9)
    record = pack_x4t_matrix_record("w2", packed, scale)

    # The record contains a fixed 64-byte matrix header and the unchanged
    # packed nibble plane before the X4T scale payload.
    assert x4t_scale_storage_bytes(scale) == len(record) - 64 - packed.numel()
    assert x4t_matrix_storage_bytes("w2", scale) == (
        (len(record) + 4095) // 4096 * 4096
    )


def test_x4t_layer_writer_requires_canonical_order(tmp_path) -> None:
    destination = tmp_path / "x4t-layer.bin"
    packed, scale = _production_matrix("w1", packed_seed=5)
    with pytest.raises(ValueError, match="canonical order"):
        with X4TLayerWriter(destination, layer=1) as writer:
            writer.add(1, "w1", packed, scale)
            writer.add(0, "w1", packed, scale)


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
