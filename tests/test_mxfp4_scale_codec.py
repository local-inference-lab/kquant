from __future__ import annotations

import struct

import pytest
import torch

from kquant.mxfp4_scale_codec import (
    compact_scale_plane_stats,
    effective_mxfp4_bpw,
    pack_scale_plane,
    pack_scale_plane_compact,
    unpack_scale_plane,
    unpack_scale_plane_compact,
)


def _example_scale() -> torch.Tensor:
    scale = torch.tensor(
        [
            [120, 121, 120, 121, 122, 120, 121, 120, 119, 121],
            [117, 117, 117, 117, 117, 117, 117, 117, 117, 117],
            [3, 9, 3, 9, 42, 3, 9, 3, 9, 3],
        ],
        dtype=torch.uint8,
    )
    return scale


def test_scale_plane_round_trip_with_palettes_and_exceptions() -> None:
    scale = _example_scale()

    payload = pack_scale_plane(scale, tile_rows=2)
    reconstruction = unpack_scale_plane(payload)

    assert torch.equal(reconstruction, scale)
    assert effective_mxfp4_bpw(scale, payload) > 4.0


def test_scale_plane_pack_is_deterministic() -> None:
    scale = _example_scale()

    assert pack_scale_plane(scale, tile_rows=2) == pack_scale_plane(
        scale.clone(), tile_rows=2
    )


def test_scale_plane_rejects_bad_terminal_offset() -> None:
    payload = bytearray(pack_scale_plane(_example_scale(), tile_rows=2))
    # The final uint32 offset is the third entry for two tiles.
    prior = struct.unpack_from("<I", payload, 16 + 4)[0]
    struct.pack_into("<I", payload, 16 + 2 * 4, prior)

    with pytest.raises(ValueError, match="terminal offset"):
        unpack_scale_plane(bytes(payload))


def test_scale_plane_rejects_non_uint8_input() -> None:
    with pytest.raises(ValueError, match="CPU uint8"):
        pack_scale_plane(torch.ones((2, 8), dtype=torch.int16))


def test_compact_scale_plane_round_trip() -> None:
    scale = torch.tensor(
        [[120 + (column & 1) for column in range(112)] for _ in range(16)],
        dtype=torch.uint8,
    )
    scale[0, 0] = 122

    payload = pack_scale_plane_compact(scale, tile_rows=16)
    reconstruction = unpack_scale_plane_compact(payload)

    assert torch.equal(reconstruction, scale)
    assert len(payload) < len(pack_scale_plane(scale, tile_rows=16))
    assert compact_scale_plane_stats(payload) == {
        "rows": 16,
        "tiles": 1,
        "palette_escapes": 0,
        "count_escapes": 0,
        "exceptions": 1,
    }


def test_compact_scale_plane_handles_adjacent_boundary_escape() -> None:
    scale = torch.tensor(
        [[0, 0, 0, 0], [255, 255, 255, 255]], dtype=torch.uint8
    )

    payload = pack_scale_plane_compact(scale)

    assert torch.equal(unpack_scale_plane_compact(payload), scale)
