from __future__ import annotations

import struct

import pytest
import torch

from kquant.x4t import (
    X4T_POSITION_BITS,
    effective_x4t_bpw,
    pack_x4t_scale_plane,
    unpack_x4t_scale_plane,
    x4t_scale_components,
)


def _scale(rows: int = 32, columns: int = 17) -> torch.Tensor:
    result = torch.empty((rows, columns), dtype=torch.uint8)
    for row in range(rows):
        base = 110 + row % 7
        result[row] = torch.tensor(
            [base + (column & 1) for column in range(columns)], dtype=torch.uint8
        )
    result[0, 0] = 140
    result[17, columns - 1] = 3
    return result.contiguous()


def test_x4t_round_trip_is_exact_and_deterministic() -> None:
    scale = _scale()
    payload = pack_x4t_scale_plane(scale)

    assert torch.equal(unpack_x4t_scale_plane(payload), scale)
    assert payload == pack_x4t_scale_plane(scale.clone())
    assert effective_x4t_bpw(scale, payload) < 4.25


def test_x4t_components_close_over_fixed_and_exception_streams() -> None:
    scale = _scale(columns=8)
    payload = pack_x4t_scale_plane(scale)

    fixed, exceptions, rows, columns = x4t_scale_components(payload)

    assert (rows, columns) == tuple(scale.shape)
    assert fixed.dtype == torch.uint8
    assert exceptions.dtype == torch.uint32
    assert int(exceptions.numel()) == 2
    assert int(exceptions[0]) & ((1 << X4T_POSITION_BITS) - 1) == 0


def test_x4t_rejects_redundant_exception() -> None:
    payload = bytearray(pack_x4t_scale_plane(_scale(columns=8)))
    # The first exception is the final eight bytes; replace its value with the
    # row's adjacent-palette base while retaining its logical position.
    entry_offset = len(payload) - 8
    entry = struct.unpack_from("<I", payload, entry_offset)[0]
    position = entry & ((1 << X4T_POSITION_BITS) - 1)
    struct.pack_into("<I", payload, entry_offset, position | (110 << X4T_POSITION_BITS))

    with pytest.raises(ValueError, match="redundantly"):
        unpack_x4t_scale_plane(bytes(payload))


def test_x4t_requires_complete_16_row_tiles() -> None:
    with pytest.raises(ValueError, match="multiple of 16"):
        pack_x4t_scale_plane(torch.zeros((15, 8), dtype=torch.uint8))
