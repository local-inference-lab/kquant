"""GPU-tile-friendly exact coding for MXFP4 UE8M0 scale planes.

X4T keeps the E2M1 nibble plane unchanged.  It represents each 16-row scale
slab with one adjacent two-value palette per row and a fixed-stride selector
bitmap.  Values outside that adjacent pair are carried by a sorted uint32
exception stream::

    bits  0..23  logical row-major scale index
    bits 24..31  exact UE8M0 byte

The fixed stream is deliberately trivial for a GPU CTA to decode and the
exception stream is suitable for a second parallel scatter.  Unlike the X4
row-palette format, hot-path decoding needs no variable tile offsets, prefix
sums, or exception searches.
"""

from __future__ import annotations

import math
import struct

import torch


X4T_MAGIC = b"KQX4T\0\0\0"
X4T_VERSION = 1
X4T_TILE_ROWS = 16
X4T_POSITION_BITS = 24
X4T_POSITION_MASK = (1 << X4T_POSITION_BITS) - 1

_HEADER = struct.Struct("<8sBBH I H H I I Q Q 20s")

if _HEADER.size != 64:
    raise AssertionError("X4T header layout drifted")


def _validate_scale(scale: torch.Tensor) -> tuple[int, int]:
    if (
        scale.dtype != torch.uint8
        or scale.ndim != 2
        or scale.device.type != "cpu"
        or not scale.is_contiguous()
    ):
        raise ValueError("X4T scale must be a contiguous two-dimensional CPU uint8 tensor")
    rows, columns = map(int, scale.shape)
    if not rows or rows % X4T_TILE_ROWS:
        raise ValueError("X4T scale rows must be a nonzero multiple of 16")
    if not 1 <= columns <= 255:
        raise ValueError("X4T scale columns must lie in 1..255")
    if rows * columns > X4T_POSITION_MASK:
        raise ValueError("X4T logical scale plane exceeds its 24-bit position field")
    return rows, columns


def _adjacent_bases(scale: torch.Tensor) -> torch.Tensor:
    """Choose the consecutive byte pair covering the most values per row."""

    rows = int(scale.shape[0])
    histogram = torch.zeros((rows, 256), dtype=torch.int16)
    histogram.scatter_add_(
        1,
        scale.to(torch.int64),
        torch.ones_like(scale, dtype=torch.int16),
    )
    # torch.argmax supplies the canonical lowest-base tie break.  Base 254 is
    # the last legal pair and therefore also represents constant-255 rows.
    return (histogram[:, :-1] + histogram[:, 1:]).argmax(dim=1).to(torch.uint8)


def pack_x4t_scale_plane(scale: torch.Tensor) -> bytes:
    """Pack an exact UE8M0 scale plane into the fixed-stream X4T format."""

    rows, columns = _validate_scale(scale)
    selector_bytes = math.ceil(columns / 8)
    tile_count = rows // X4T_TILE_ROWS
    tile_bytes = X4T_TILE_ROWS + X4T_TILE_ROWS * selector_bytes

    bases = _adjacent_bases(scale)
    base_i16 = bases.to(torch.int16)
    source_i16 = scale.to(torch.int16)
    low = source_i16 == base_i16[:, None]
    high = source_i16 == (base_i16[:, None] + 1)
    exceptions = ~(low | high)

    padded_columns = selector_bytes * 8
    selector_bits = torch.zeros((rows, padded_columns), dtype=torch.uint8)
    selector_bits[:, :columns] = high.to(torch.uint8)
    bit_weights = (1 << torch.arange(8, dtype=torch.int16)).view(1, 1, 8)
    selectors = (
        selector_bits.view(rows, selector_bytes, 8).to(torch.int16)
        * bit_weights
    ).sum(dim=2).to(torch.uint8)

    fixed = torch.empty((tile_count, tile_bytes), dtype=torch.uint8)
    fixed[:, :X4T_TILE_ROWS] = bases.view(tile_count, X4T_TILE_ROWS)
    fixed[:, X4T_TILE_ROWS:] = selectors.view(tile_count, -1)
    fixed_payload = fixed.numpy().tobytes()

    coordinates = torch.nonzero(exceptions, as_tuple=False)
    if int(coordinates.numel()):
        positions = coordinates[:, 0] * columns + coordinates[:, 1]
        values = scale[coordinates[:, 0], coordinates[:, 1]].to(torch.int64)
        entries = positions | (values << X4T_POSITION_BITS)
        exception_payload = struct.pack(
            f"<{int(entries.numel())}I", *map(int, entries.tolist())
        )
    else:
        entries = torch.empty((0,), dtype=torch.int64)
        exception_payload = b""

    header = _HEADER.pack(
        X4T_MAGIC,
        X4T_VERSION,
        X4T_TILE_ROWS,
        0,
        rows,
        columns,
        selector_bytes,
        tile_bytes,
        tile_count,
        len(fixed_payload),
        int(entries.numel()),
        bytes(20),
    )
    return header + fixed_payload + exception_payload


def _parse_header(payload: bytes) -> tuple[int, int, int, int, int, int]:
    if len(payload) < _HEADER.size:
        raise ValueError("X4T payload is truncated before its header")
    (
        magic,
        version,
        tile_rows,
        flags,
        rows,
        columns,
        selector_bytes,
        tile_bytes,
        tile_count,
        fixed_bytes,
        exception_count,
        reserved,
    ) = _HEADER.unpack_from(payload)
    if magic != X4T_MAGIC or version != X4T_VERSION:
        raise ValueError("X4T payload has an unsupported magic or version")
    if tile_rows != X4T_TILE_ROWS or flags or any(reserved):
        raise ValueError("X4T payload header is noncanonical")
    if not rows or rows % X4T_TILE_ROWS or not 1 <= columns <= 255:
        raise ValueError("X4T payload dimensions are invalid")
    if rows * columns > X4T_POSITION_MASK:
        raise ValueError("X4T payload exceeds its 24-bit position field")
    expected_selector_bytes = math.ceil(columns / 8)
    expected_tile_count = rows // X4T_TILE_ROWS
    expected_tile_bytes = X4T_TILE_ROWS * (1 + expected_selector_bytes)
    expected_fixed_bytes = expected_tile_count * expected_tile_bytes
    if (
        selector_bytes != expected_selector_bytes
        or tile_count != expected_tile_count
        or tile_bytes != expected_tile_bytes
        or fixed_bytes != expected_fixed_bytes
    ):
        raise ValueError("X4T fixed-stream geometry is noncanonical")
    expected_length = _HEADER.size + fixed_bytes + 4 * exception_count
    if len(payload) != expected_length:
        raise ValueError("X4T component lengths disagree with its total length")
    return rows, columns, selector_bytes, tile_bytes, fixed_bytes, exception_count


def unpack_x4t_scale_plane(payload: bytes) -> torch.Tensor:
    """Decode and fully validate an exact X4T scale plane."""

    rows, columns, selector_bytes, tile_bytes, fixed_bytes, exception_count = (
        _parse_header(payload)
    )
    fixed_start = _HEADER.size
    fixed = torch.frombuffer(
        bytearray(payload[fixed_start : fixed_start + fixed_bytes]),
        dtype=torch.uint8,
    ).reshape(rows // X4T_TILE_ROWS, tile_bytes)
    bases = fixed[:, :X4T_TILE_ROWS].reshape(rows)
    selectors = fixed[:, X4T_TILE_ROWS:].reshape(rows, selector_bytes)
    column = torch.arange(columns, dtype=torch.int64)
    selected = (
        selectors[:, column // 8].to(torch.int16)
        >> (column % 8).to(torch.int16)
    ) & 1
    result = (bases.to(torch.int16)[:, None] + selected).to(torch.uint8)
    if columns % 8 and bool((selectors[:, -1] >> (columns % 8)).any()):
        raise ValueError("X4T selector has nonzero padding bits")

    exception_start = fixed_start + fixed_bytes
    entries = (
        struct.unpack_from(f"<{exception_count}I", payload, exception_start)
        if exception_count
        else ()
    )
    previous = -1
    flat = result.view(-1)
    for entry in entries:
        position = entry & X4T_POSITION_MASK
        value = entry >> X4T_POSITION_BITS
        if position >= flat.numel() or position <= previous:
            raise ValueError("X4T exception positions must be valid and strictly increasing")
        previous = position
        row = position // columns
        base = int(bases[row])
        if value in (base, base + 1):
            raise ValueError("X4T exception redundantly names an adjacent-palette value")
        flat[position] = value

    # This catches non-optimal bases, redundant exceptions, and alternate tie
    # breaks, keeping one byte representation for every scale plane.
    if pack_x4t_scale_plane(result.contiguous()) != payload:
        raise ValueError("X4T payload is not canonical")
    return result


def x4t_scale_components(payload: bytes) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Return validated fixed bytes and uint32 exceptions for GPU upload."""

    reconstructed = unpack_x4t_scale_plane(payload)
    rows, columns, _, _, fixed_bytes, exception_count = _parse_header(payload)
    fixed_start = _HEADER.size
    fixed = torch.frombuffer(
        bytearray(payload[fixed_start : fixed_start + fixed_bytes]),
        dtype=torch.uint8,
    ).contiguous()
    exception_start = fixed_start + fixed_bytes
    exceptions = torch.frombuffer(
        bytearray(payload[exception_start:]), dtype=torch.uint32
    ).contiguous()
    if int(exceptions.numel()) != exception_count:
        raise AssertionError("validated X4T exception accounting drifted")
    del reconstructed
    return fixed, exceptions, rows, columns


def effective_x4t_bpw(scale: torch.Tensor, payload: bytes) -> float:
    """Return nibble plane plus X4T scale bytes in bits per weight."""

    rows, columns = _validate_scale(scale)
    weights = rows * columns * 32
    return 4.0 + len(payload) * 8 / weights
