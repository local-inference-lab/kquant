"""Exact row-palette coding for the UE8M0 scale plane of MXFP4 weights."""

from __future__ import annotations

import math
import struct
from collections import Counter

import torch


MAGIC = b"KQSC"
VERSION = 1
COMPACT_VERSION = 2
DEFAULT_TILE_ROWS = 16
_HEADER = struct.Struct("<4sBBH I H H")


def _row_palette(row: list[int]) -> tuple[int, int]:
    counts = Counter(row)
    ordered = sorted(counts, key=lambda value: (-counts[value], value))
    first = ordered[0]
    second = first if len(ordered) == 1 else ordered[1]
    return first, second


def pack_scale_plane(
    scale: torch.Tensor, *, tile_rows: int = DEFAULT_TILE_ROWS
) -> bytes:
    """Pack a two-dimensional uint8 scale plane without numerical loss."""

    if scale.dtype != torch.uint8 or scale.ndim != 2 or scale.device.type != "cpu":
        raise ValueError("scale must be a two-dimensional CPU uint8 tensor")
    rows, columns = map(int, scale.shape)
    if not rows or not columns or columns > 255:
        raise ValueError("scale dimensions must be nonzero and columns must fit uint8")
    if not 1 <= tile_rows <= 255:
        raise ValueError("tile_rows must fit uint8")
    selector_bytes = math.ceil(columns / 8)
    tiles: list[bytes] = []
    contiguous = scale.contiguous()
    for start in range(0, rows, tile_rows):
        end = min(start + tile_rows, rows)
        tile = contiguous[start:end]
        palette0 = bytearray()
        palette1 = bytearray()
        counts = bytearray()
        selectors = bytearray()
        exceptions = bytearray()
        for tensor_row in tile:
            row = list(map(int, tensor_row.tolist()))
            first, second = _row_palette(row)
            palette0.append(first)
            palette1.append(second)
            selector = bytearray(selector_bytes)
            row_exceptions: list[tuple[int, int]] = []
            for position, value in enumerate(row):
                if value == second and second != first:
                    selector[position // 8] |= 1 << (position % 8)
                elif value != first:
                    row_exceptions.append((position, value))
            if len(row_exceptions) > 255:
                raise ValueError("row exception count does not fit uint8")
            counts.append(len(row_exceptions))
            selectors.extend(selector)
            for position, value in row_exceptions:
                exceptions.extend((position, value))
        tiles.append(bytes(palette0 + palette1 + counts + selectors + exceptions))

    offsets = [0]
    for tile in tiles:
        offsets.append(offsets[-1] + len(tile))
    if offsets[-1] >= 2**32:
        raise ValueError("scale payload exceeds uint32 offset range")
    header = _HEADER.pack(
        MAGIC,
        VERSION,
        tile_rows,
        0,
        rows,
        columns,
        0,
    )
    offset_table = struct.pack(f"<{len(offsets)}I", *offsets)
    return header + offset_table + b"".join(tiles)


def unpack_scale_plane(payload: bytes) -> torch.Tensor:
    """Decode and validate a row-palette UE8M0 scale plane."""

    if len(payload) < _HEADER.size:
        raise ValueError("scale payload is truncated before its header")
    magic, version, tile_rows, reserved0, rows, columns, reserved1 = _HEADER.unpack_from(
        payload
    )
    if magic != MAGIC or version != VERSION:
        raise ValueError("scale payload has an unsupported magic or version")
    if reserved0 or reserved1:
        raise ValueError("scale payload reserved fields must be zero")
    if not rows or not columns or columns > 255 or not tile_rows:
        raise ValueError("scale payload dimensions are invalid")
    tile_count = math.ceil(rows / tile_rows)
    table_bytes = 4 * (tile_count + 1)
    payload_start = _HEADER.size + table_bytes
    if len(payload) < payload_start:
        raise ValueError("scale payload is truncated in its offset table")
    offsets = struct.unpack_from(f"<{tile_count + 1}I", payload, _HEADER.size)
    if offsets[0] != 0 or any(left > right for left, right in zip(offsets, offsets[1:])):
        raise ValueError("scale payload offsets are not monotone from zero")
    if offsets[-1] != len(payload) - payload_start:
        raise ValueError("scale payload terminal offset disagrees with its length")

    selector_bytes = math.ceil(columns / 8)
    result = torch.empty((rows, columns), dtype=torch.uint8)
    for tile_index in range(tile_count):
        start_row = tile_index * tile_rows
        row_count = min(tile_rows, rows - start_row)
        body = memoryview(payload)[
            payload_start + offsets[tile_index] : payload_start + offsets[tile_index + 1]
        ]
        fixed_bytes = row_count * (3 + selector_bytes)
        if len(body) < fixed_bytes:
            raise ValueError("scale tile is truncated before its fixed fields")
        palette0 = body[:row_count]
        palette1 = body[row_count : 2 * row_count]
        counts = body[2 * row_count : 3 * row_count]
        selector_start = 3 * row_count
        exception_start = selector_start + row_count * selector_bytes
        expected_exceptions = sum(map(int, counts))
        if len(body) != exception_start + 2 * expected_exceptions:
            raise ValueError("scale tile exception counts disagree with its length")
        cursor = exception_start
        for local_row in range(row_count):
            first = int(palette0[local_row])
            second = int(palette1[local_row])
            selector = body[
                selector_start + local_row * selector_bytes :
                selector_start + (local_row + 1) * selector_bytes
            ]
            row = torch.full((columns,), first, dtype=torch.uint8)
            for position in range(columns):
                if int(selector[position // 8]) & (1 << (position % 8)):
                    row[position] = second
            if columns % 8:
                unused = int(selector[-1]) >> (columns % 8)
                if unused:
                    raise ValueError("scale selector has nonzero padding bits")
            seen: set[int] = set()
            for _ in range(int(counts[local_row])):
                position = int(body[cursor])
                value = int(body[cursor + 1])
                cursor += 2
                if position >= columns or position in seen:
                    raise ValueError("scale exception position is invalid or duplicated")
                if value in (first, second):
                    raise ValueError("scale exception redundantly names a palette value")
                seen.add(position)
                row[position] = value
            result[start_row + local_row] = row
        if cursor != len(body):
            raise ValueError("scale tile decoder did not consume its complete body")
    return result


def effective_mxfp4_bpw(scale: torch.Tensor, payload: bytes) -> float:
    """Return exact nibble-plus-compressed-scale storage in bits per weight."""

    if scale.ndim != 2:
        raise ValueError("scale must be two-dimensional")
    weights = int(scale.numel()) * 32
    packed_weight_bytes = weights // 2
    return (packed_weight_bytes + len(payload)) * 8 / weights


def _packed_low_bits(values: list[int], *, bits: int) -> bytes:
    if bits not in (1, 2):
        raise ValueError("only one- and two-bit fields are supported")
    payload = bytearray(math.ceil(len(values) * bits / 8))
    mask = (1 << bits) - 1
    for index, value in enumerate(values):
        if value & ~mask:
            raise ValueError("bit-packed field value is out of range")
        bit = index * bits
        payload[bit // 8] |= value << (bit % 8)
    return bytes(payload)


def _unpacked_low_bits(
    payload: memoryview, *, count: int, bits: int, field: str
) -> list[int]:
    expected = math.ceil(count * bits / 8)
    if len(payload) != expected:
        raise ValueError(f"compact scale {field} has the wrong length")
    mask = (1 << bits) - 1
    values = [
        (int(payload[(index * bits) // 8]) >> ((index * bits) % 8)) & mask
        for index in range(count)
    ]
    used_bits = count * bits
    if used_bits % 8 and int(payload[-1]) >> (used_bits % 8):
        raise ValueError(f"compact scale {field} has nonzero padding bits")
    return values


def pack_scale_plane_compact(
    scale: torch.Tensor, *, tile_rows: int = DEFAULT_TILE_ROWS
) -> bytes:
    """Pack a scale plane with adjacent-palette and sparse-count coding.

    The two most common values in a row are normally adjacent UE8M0
    exponents.  Version 2 stores the first value plus a direction bit; an
    explicit second value is used as a lossless escape.  Exception counts 0,
    1, and 2 are represented directly with two bits, with one byte only for
    the uncommon count of three or more.
    """

    if scale.dtype != torch.uint8 or scale.ndim != 2 or scale.device.type != "cpu":
        raise ValueError("scale must be a two-dimensional CPU uint8 tensor")
    rows, columns = map(int, scale.shape)
    if not rows or not columns or columns > 255:
        raise ValueError("scale dimensions must be nonzero and columns must fit uint8")
    if not 1 <= tile_rows <= 255:
        raise ValueError("tile_rows must fit uint8")
    selector_bytes = math.ceil(columns / 8)
    tiles: list[bytes] = []
    contiguous = scale.contiguous()
    for start in range(0, rows, tile_rows):
        tile = contiguous[start : min(start + tile_rows, rows)]
        palette0 = bytearray()
        directions: list[int] = []
        explicit_flags: list[int] = []
        explicit_palette1 = bytearray()
        count_codes: list[int] = []
        extended_counts = bytearray()
        selectors = bytearray()
        exceptions = bytearray()
        for tensor_row in tile:
            row = list(map(int, tensor_row.tolist()))
            first, second = _row_palette(row)
            palette0.append(first)
            delta = second - first
            if delta in (-1, 1):
                directions.append(int(delta > 0))
                explicit_flags.append(0)
            else:
                directions.append(0)
                explicit_flags.append(1)
                explicit_palette1.append(second)
            selector = bytearray(selector_bytes)
            row_exceptions: list[tuple[int, int]] = []
            for position, value in enumerate(row):
                if value == second and second != first:
                    selector[position // 8] |= 1 << (position % 8)
                elif value != first:
                    row_exceptions.append((position, value))
            count = len(row_exceptions)
            if count > 255:
                raise ValueError("row exception count does not fit uint8")
            if count < 3:
                count_codes.append(count)
            else:
                count_codes.append(3)
                extended_counts.append(count)
            selectors.extend(selector)
            for position, value in row_exceptions:
                exceptions.extend((position, value))
        tiles.append(
            bytes(palette0)
            + _packed_low_bits(directions, bits=1)
            + _packed_low_bits(explicit_flags, bits=1)
            + _packed_low_bits(count_codes, bits=2)
            + bytes(explicit_palette1)
            + bytes(extended_counts)
            + bytes(selectors)
            + bytes(exceptions)
        )

    offsets = [0]
    for tile in tiles:
        offsets.append(offsets[-1] + len(tile))
    if offsets[-1] >= 2**32:
        raise ValueError("scale payload exceeds uint32 offset range")
    header = _HEADER.pack(
        MAGIC,
        COMPACT_VERSION,
        tile_rows,
        0,
        rows,
        columns,
        0,
    )
    return header + struct.pack(f"<{len(offsets)}I", *offsets) + b"".join(tiles)


def unpack_scale_plane_compact(payload: bytes) -> torch.Tensor:
    """Decode and validate a version-2 compact UE8M0 scale plane."""

    if len(payload) < _HEADER.size:
        raise ValueError("compact scale payload is truncated before its header")
    magic, version, tile_rows, reserved0, rows, columns, reserved1 = _HEADER.unpack_from(
        payload
    )
    if magic != MAGIC or version != COMPACT_VERSION:
        raise ValueError("compact scale payload has an unsupported magic or version")
    if reserved0 or reserved1:
        raise ValueError("compact scale payload reserved fields must be zero")
    if not rows or not columns or columns > 255 or not tile_rows:
        raise ValueError("compact scale payload dimensions are invalid")
    tile_count = math.ceil(rows / tile_rows)
    table_bytes = 4 * (tile_count + 1)
    payload_start = _HEADER.size + table_bytes
    if len(payload) < payload_start:
        raise ValueError("compact scale payload is truncated in its offset table")
    offsets = struct.unpack_from(f"<{tile_count + 1}I", payload, _HEADER.size)
    if offsets[0] != 0 or any(left > right for left, right in zip(offsets, offsets[1:])):
        raise ValueError("compact scale payload offsets are not monotone from zero")
    if offsets[-1] != len(payload) - payload_start:
        raise ValueError("compact scale payload terminal offset disagrees with its length")

    selector_bytes = math.ceil(columns / 8)
    result = torch.empty((rows, columns), dtype=torch.uint8)
    for tile_index in range(tile_count):
        start_row = tile_index * tile_rows
        row_count = min(tile_rows, rows - start_row)
        body = memoryview(payload)[
            payload_start + offsets[tile_index] : payload_start + offsets[tile_index + 1]
        ]
        flag_bytes = math.ceil(row_count / 8)
        count_bytes = math.ceil(row_count * 2 / 8)
        fixed_prefix = row_count + 2 * flag_bytes + count_bytes
        if len(body) < fixed_prefix:
            raise ValueError("compact scale tile is truncated before its row fields")
        cursor = 0
        palette0 = body[cursor : cursor + row_count]
        cursor += row_count
        directions = _unpacked_low_bits(
            body[cursor : cursor + flag_bytes],
            count=row_count,
            bits=1,
            field="direction bits",
        )
        cursor += flag_bytes
        explicit_flags = _unpacked_low_bits(
            body[cursor : cursor + flag_bytes],
            count=row_count,
            bits=1,
            field="explicit-palette bits",
        )
        cursor += flag_bytes
        count_codes = _unpacked_low_bits(
            body[cursor : cursor + count_bytes],
            count=row_count,
            bits=2,
            field="count codes",
        )
        cursor += count_bytes
        explicit_count = sum(explicit_flags)
        extension_count = sum(code == 3 for code in count_codes)
        metadata_end = cursor + explicit_count + extension_count
        if len(body) < metadata_end + row_count * selector_bytes:
            raise ValueError("compact scale tile is truncated in its row escapes")
        explicit_values = body[cursor : cursor + explicit_count]
        cursor += explicit_count
        extended_counts = body[cursor : cursor + extension_count]
        cursor += extension_count
        counts = []
        extension_cursor = 0
        for code in count_codes:
            if code < 3:
                counts.append(code)
            else:
                count = int(extended_counts[extension_cursor])
                extension_cursor += 1
                if count < 3:
                    raise ValueError("compact scale count escape is noncanonical")
                counts.append(count)
        selector_start = cursor
        exception_start = selector_start + row_count * selector_bytes
        expected_length = exception_start + 2 * sum(counts)
        if len(body) != expected_length:
            raise ValueError("compact scale exception counts disagree with its length")

        explicit_cursor = 0
        exception_cursor = exception_start
        for local_row in range(row_count):
            first = int(palette0[local_row])
            if explicit_flags[local_row]:
                second = int(explicit_values[explicit_cursor])
                explicit_cursor += 1
                if second - first in (-1, 1):
                    raise ValueError("compact scale palette escape is noncanonical")
            else:
                second = first + (1 if directions[local_row] else -1)
                if not 0 <= second <= 255:
                    raise ValueError("compact scale adjacent palette is out of range")
            selector = body[
                selector_start + local_row * selector_bytes :
                selector_start + (local_row + 1) * selector_bytes
            ]
            if columns % 8 and int(selector[-1]) >> (columns % 8):
                raise ValueError("compact scale selector has nonzero padding bits")
            row = torch.full((columns,), first, dtype=torch.uint8)
            for position in range(columns):
                if int(selector[position // 8]) & (1 << (position % 8)):
                    row[position] = second
            seen: set[int] = set()
            for _ in range(counts[local_row]):
                position = int(body[exception_cursor])
                value = int(body[exception_cursor + 1])
                exception_cursor += 2
                if position >= columns or position in seen:
                    raise ValueError(
                        "compact scale exception position is invalid or duplicated"
                    )
                if value in (first, second):
                    raise ValueError(
                        "compact scale exception redundantly names a palette value"
                    )
                seen.add(position)
                row[position] = value
            result[start_row + local_row] = row
        if explicit_cursor != explicit_count or exception_cursor != len(body):
            raise ValueError("compact scale decoder did not consume its complete tile")
    return result


def compact_scale_plane_stats(payload: bytes) -> dict[str, int]:
    """Return structural counts from a validated version-2 payload.

    Callers that need correctness should first decode the payload.  This
    helper then obtains accounting evidence without rescanning the source
    tensor row by row.
    """

    if len(payload) < _HEADER.size:
        raise ValueError("compact scale payload is truncated before its header")
    magic, version, tile_rows, reserved0, rows, columns, reserved1 = _HEADER.unpack_from(
        payload
    )
    if magic != MAGIC or version != COMPACT_VERSION or reserved0 or reserved1:
        raise ValueError("payload is not a canonical compact scale plane")
    if not rows or not columns or columns > 255 or not tile_rows:
        raise ValueError("compact scale payload dimensions are invalid")
    tile_count = math.ceil(rows / tile_rows)
    table_bytes = 4 * (tile_count + 1)
    payload_start = _HEADER.size + table_bytes
    if len(payload) < payload_start:
        raise ValueError("compact scale payload is truncated in its offset table")
    offsets = struct.unpack_from(f"<{tile_count + 1}I", payload, _HEADER.size)
    if offsets[0] != 0 or offsets[-1] != len(payload) - payload_start:
        raise ValueError("compact scale payload offsets disagree with its length")

    selector_bytes = math.ceil(columns / 8)
    palette_escapes = count_escapes = exceptions = 0
    for tile_index in range(tile_count):
        row_count = min(tile_rows, rows - tile_index * tile_rows)
        body = memoryview(payload)[
            payload_start + offsets[tile_index] : payload_start + offsets[tile_index + 1]
        ]
        flag_bytes = math.ceil(row_count / 8)
        count_bytes = math.ceil(row_count * 2 / 8)
        prefix = row_count
        explicit_flags = _unpacked_low_bits(
            body[prefix + flag_bytes : prefix + 2 * flag_bytes],
            count=row_count,
            bits=1,
            field="explicit-palette bits",
        )
        count_codes = _unpacked_low_bits(
            body[
                prefix + 2 * flag_bytes : prefix + 2 * flag_bytes + count_bytes
            ],
            count=row_count,
            bits=2,
            field="count codes",
        )
        tile_palette_escapes = sum(explicit_flags)
        tile_count_escapes = sum(code == 3 for code in count_codes)
        fixed = (
            row_count
            + 2 * flag_bytes
            + count_bytes
            + tile_palette_escapes
            + tile_count_escapes
            + row_count * selector_bytes
        )
        exception_bytes = len(body) - fixed
        if exception_bytes < 0 or exception_bytes % 2:
            raise ValueError("compact scale payload byte accounting did not close")
        palette_escapes += tile_palette_escapes
        count_escapes += tile_count_escapes
        exceptions += exception_bytes // 2
    return {
        "rows": rows,
        "tiles": tile_count,
        "palette_escapes": palette_escapes,
        "count_escapes": count_escapes,
        "exceptions": exceptions,
    }
