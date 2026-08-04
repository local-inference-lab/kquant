from __future__ import annotations

import struct

import pytest
import torch

from kquant.x4 import (
    X4LayerReader,
    X4LayerWriter,
    pack_x4_matrix_record,
    unpack_x4_matrix_record,
    x4_matrix_storage_bytes,
    x4_record_bpw,
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


def test_x4_matrix_record_is_exact_and_deterministic() -> None:
    packed, scale = _matrix()

    payload = pack_x4_matrix_record(
        "w1", packed, scale, production_shape=False
    )
    decoded = unpack_x4_matrix_record(
        payload, expected_matrix="w1", production_shape=False
    )

    assert torch.equal(decoded.packed, packed)
    assert torch.equal(decoded.scale, scale)
    assert payload == pack_x4_matrix_record(
        "w1", packed.clone(), scale.clone(), production_shape=False
    )
    assert x4_record_bpw(payload) > 4.0


def test_x4_record_rejects_matrix_slot_mismatch() -> None:
    packed, scale = _matrix()
    payload = pack_x4_matrix_record(
        "w1", packed, scale, production_shape=False
    )

    with pytest.raises(ValueError, match="directory slot"):
        unpack_x4_matrix_record(
            payload, expected_matrix="w2", production_shape=False
        )


def test_x4_record_rejects_noncanonical_reserved_bytes() -> None:
    packed, scale = _matrix()
    payload = bytearray(
        pack_x4_matrix_record("w1", packed, scale, production_shape=False)
    )
    payload[63] = 1

    with pytest.raises(ValueError, match="noncanonical"):
        unpack_x4_matrix_record(bytes(payload), production_shape=False)


def test_x4_layer_round_trip_and_sparse_lookup(tmp_path) -> None:
    packed, scale = _matrix()
    destination = tmp_path / "x4-layer.bin"
    with X4LayerWriter(destination, layer=24) as writer:
        # Canonical matrix order is w1, w3, w2.
        writer.add(6, "w1", *_production_matrix("w1", packed_seed=1))
        writer.add(6, "w3", *_production_matrix("w3", packed_seed=2))
        writer.add(6, "w2", *_production_matrix("w2", packed_seed=3))

    reader = X4LayerReader(destination)
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


def test_x4_layer_rejects_directory_corruption(tmp_path) -> None:
    destination = tmp_path / "x4-layer.bin"
    with X4LayerWriter(destination, layer=1):
        pass
    payload = bytearray(destination.read_bytes())
    payload[4096] = 1
    destination.write_bytes(payload)

    with pytest.raises(ValueError, match="directory checksum"):
        X4LayerReader(destination)


def test_x4_layer_rejects_record_corruption(tmp_path) -> None:
    destination = tmp_path / "x4-layer.bin"
    packed, scale = _production_matrix("w1", packed_seed=4)
    with X4LayerWriter(destination, layer=1) as writer:
        entry = writer.add(0, "w1", packed, scale)
    with destination.open("r+b") as handle:
        handle.seek(entry.offset + entry.length - 1)
        byte = handle.read(1)
        handle.seek(entry.offset + entry.length - 1)
        handle.write(bytes([byte[0] ^ 1]))

    with pytest.raises(ValueError, match="record checksum"):
        X4LayerReader(destination)


def test_x4_scale_only_accounting_matches_written_record() -> None:
    packed, scale = _production_matrix("w2", packed_seed=9)
    record = pack_x4_matrix_record("w2", packed, scale)

    assert x4_matrix_storage_bytes("w2", scale) == (
        (len(record) + 4095) // 4096 * 4096
    )


def test_x4_layer_writer_requires_canonical_order(tmp_path) -> None:
    destination = tmp_path / "x4-layer.bin"
    packed, scale = _production_matrix("w1", packed_seed=5)
    with pytest.raises(ValueError, match="canonical order"):
        with X4LayerWriter(destination, layer=1) as writer:
            writer.add(1, "w1", packed, scale)
            writer.add(0, "w1", packed, scale)
