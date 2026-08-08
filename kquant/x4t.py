"""GPU-tile-friendly exact coding for MXFP4 UE8M0 scale planes.

X4T keeps the E2M1 nibble plane unchanged.  It represents each 16-row scale
slab with one adjacent two-value palette per row and a fixed-stride selector
bitmap.  Values outside that adjacent pair are carried by a sorted uint32
exception stream::

    bits  0..23  logical row-major scale index
    bits 24..31  exact UE8M0 byte

The fixed stream is deliberately trivial for a GPU CTA to decode and the
exception stream is suitable for a second parallel scatter. Hot-path decoding
needs no variable tile offsets, prefix sums, or exception searches.
"""

from __future__ import annotations

import math
import os
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from kquant import constants as C


X4T_MAGIC = b"KQX4T\0\0\0"
X4T_VERSION = 1
X4T_TILE_ROWS = 16
X4T_POSITION_BITS = 24
X4T_POSITION_MASK = (1 << X4T_POSITION_BITS) - 1

X4T_LAYER_MAGIC = b"KQX4TLY\0"
X4T_RECORD_MAGIC = b"KQ4T"
X4T_LAYER_HEADER_BYTES = 4096
X4T_DIRECTORY_BYTES = 65536
X4T_DIRECTORY_ENTRY_BYTES = 24
X4T_DATA_OFFSET = X4T_LAYER_HEADER_BYTES + X4T_DIRECTORY_BYTES
X4T_RECORD_ALIGNMENT = 4096
X4T_EXPERTS_PER_LAYER = 896
X4T_MATRIX_ORDER = ("w1", "w3", "w2")

_HEADER = struct.Struct("<8sBBH I H H I I Q Q 20s")
_LAYER_HEADER = struct.Struct("<8sHHIHHHHQQQQIIII")
_DIRECTORY_ENTRY = struct.Struct("<QQII")
_RECORD_HEADER = struct.Struct("<4sHHBBHIIQQ28s")

if _HEADER.size != 64:
    raise AssertionError("X4T header layout drifted")
if _DIRECTORY_ENTRY.size != X4T_DIRECTORY_ENTRY_BYTES:
    raise AssertionError("X4T directory entry layout drifted")
if _RECORD_HEADER.size != 64:
    raise AssertionError("X4T record header layout drifted")
if (
    X4T_EXPERTS_PER_LAYER
    * len(X4T_MATRIX_ORDER)
    * X4T_DIRECTORY_ENTRY_BYTES
    > X4T_DIRECTORY_BYTES
):
    raise AssertionError("X4T fixed directory is too small")


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


def _adjacent_bases_numpy(source: np.ndarray) -> np.ndarray:
    """Choose each row's best adjacent byte pair with a lowest-base tie break."""

    if source.dtype != np.uint8 or source.ndim != 2 or not source.flags.c_contiguous:
        raise ValueError("X4T source array must be contiguous two-dimensional uint8")
    rows = int(source.shape[0])
    indexed = (np.arange(rows, dtype=np.int64)[:, None] << 8) + source
    histogram = np.bincount(indexed.ravel(), minlength=rows * 256).reshape(rows, 256)
    # np.argmax supplies the canonical lowest-base tie break. Base 254 is the
    # last legal pair and therefore also represents constant-255 rows.
    return np.argmax(histogram[:, :-1] + histogram[:, 1:], axis=1).astype(
        np.uint8
    )


def _adjacent_bases(scale: torch.Tensor) -> torch.Tensor:
    return torch.from_numpy(_adjacent_bases_numpy(scale.numpy()))


def pack_x4t_scale_components(
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return X4T's GPU-facing fixed stream and exception words.

    This is the payload behind the 64-byte standalone scale header. Exposing
    it directly lets checkpoint preparation preserve X4T all the way to the
    GPU instead of reconstructing a dense scale plane at model load.
    """

    rows, columns = _validate_scale(scale)
    selector_bytes = math.ceil(columns / 8)
    tile_count = rows // X4T_TILE_ROWS
    tile_bytes = X4T_TILE_ROWS * (1 + selector_bytes)

    source = scale.numpy()
    bases = _adjacent_bases_numpy(source)
    source_i16 = source.astype(np.int16)
    base_i16 = bases.astype(np.int16)
    low = source_i16 == base_i16[:, None]
    high = source_i16 == (base_i16[:, None] + 1)
    selectors = np.packbits(high, axis=1, bitorder="little")

    fixed = np.empty((tile_count, tile_bytes), dtype=np.uint8)
    fixed[:, :X4T_TILE_ROWS] = bases.reshape(tile_count, X4T_TILE_ROWS)
    fixed[:, X4T_TILE_ROWS:] = selectors.reshape(tile_count, -1)

    coordinates = np.argwhere(~(low | high))
    if coordinates.size:
        positions = (
            coordinates[:, 0].astype(np.uint32) * np.uint32(columns)
            + coordinates[:, 1].astype(np.uint32)
        )
        values = source[coordinates[:, 0], coordinates[:, 1]].astype(np.uint32)
        exceptions = positions | (values << np.uint32(X4T_POSITION_BITS))
    else:
        exceptions = np.empty((0,), dtype=np.uint32)
    return (
        torch.from_numpy(fixed.reshape(-1).copy()),
        torch.from_numpy(exceptions.copy()),
    )


def pack_x4t_scale_plane(scale: torch.Tensor) -> bytes:
    """Pack an exact UE8M0 scale plane into the fixed-stream X4T format."""

    rows, columns = _validate_scale(scale)
    selector_bytes = math.ceil(columns / 8)
    tile_count = rows // X4T_TILE_ROWS
    tile_bytes = X4T_TILE_ROWS + X4T_TILE_ROWS * selector_bytes
    fixed, exceptions = pack_x4t_scale_components(scale)
    fixed_payload = fixed.numpy().tobytes()
    exception_payload = exceptions.numpy().astype("<u4", copy=False).tobytes()

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
        int(exceptions.numel()),
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


def _align_up(value: int, alignment: int = X4T_RECORD_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def _matrix_id(matrix: str) -> int:
    try:
        return X4T_MATRIX_ORDER.index(matrix)
    except ValueError as exc:
        raise ValueError(f"unsupported X4T matrix: {matrix}") from exc


def _matrix_shapes(matrix: str) -> tuple[tuple[int, int], tuple[int, int]]:
    if matrix not in C.EXPERT_SHAPES:
        raise ValueError(f"unsupported X4T matrix: {matrix}")
    out_features, in_features = C.EXPERT_SHAPES[matrix]
    return (
        (out_features, in_features // 2),
        (out_features, in_features // C.MXFP4_BLOCK),
    )


def _validate_matrix_tensors(
    matrix: str,
    packed: torch.Tensor,
    scale: torch.Tensor,
    *,
    production_shape: bool,
) -> None:
    for name, value in (("packed", packed), ("scale", scale)):
        if value.dtype != torch.uint8 or value.ndim != 2 or value.device.type != "cpu":
            raise ValueError(f"X4T {name} must be a two-dimensional CPU uint8 tensor")
        if not value.is_contiguous():
            raise ValueError(f"X4T {name} tensor must be contiguous")
    packed_rows, packed_columns = map(int, packed.shape)
    scale_rows, scale_columns = map(int, scale.shape)
    if not packed_rows or not packed_columns or not scale_rows or not scale_columns:
        raise ValueError("X4T matrices must have nonzero dimensions")
    if packed_rows != scale_rows or packed_columns * 2 != scale_columns * C.MXFP4_BLOCK:
        raise ValueError("X4T packed and scale shapes do not describe the same matrix")
    if production_shape:
        expected_packed, expected_scale = _matrix_shapes(matrix)
        if tuple(packed.shape) != expected_packed or tuple(scale.shape) != expected_scale:
            raise ValueError(
                f"X4T {matrix} shape mismatch: expected packed {expected_packed} and "
                f"scale {expected_scale}"
            )


@dataclass(frozen=True)
class X4TMatrix:
    matrix: str
    packed: torch.Tensor
    scale: torch.Tensor

    def __post_init__(self) -> None:
        _matrix_id(self.matrix)
        _validate_matrix_tensors(
            self.matrix,
            self.packed,
            self.scale,
            production_shape=False,
        )


@dataclass(frozen=True)
class X4TDirectoryEntry:
    offset: int = 0
    length: int = 0
    crc32: int = 0
    flags: int = 0

    @property
    def present(self) -> bool:
        return bool(self.flags & 1)

    def to_bytes(self) -> bytes:
        return _DIRECTORY_ENTRY.pack(self.offset, self.length, self.crc32, self.flags)

    @classmethod
    def from_bytes(cls, payload: bytes | memoryview) -> "X4TDirectoryEntry":
        if len(payload) != _DIRECTORY_ENTRY.size:
            raise ValueError("X4T directory entry has the wrong length")
        return cls(*_DIRECTORY_ENTRY.unpack(payload))


def pack_x4t_matrix_record(
    matrix: str,
    packed: torch.Tensor,
    scale: torch.Tensor,
    *,
    production_shape: bool = True,
) -> bytes:
    """Encode one exact MXFP4 matrix as an independently decodable X4T record."""

    matrix_id = _matrix_id(matrix)
    _validate_matrix_tensors(
        matrix,
        packed,
        scale,
        production_shape=production_shape,
    )
    out_features = int(packed.shape[0])
    in_features = int(packed.shape[1]) * 2
    packed_payload = memoryview(packed.numpy()).cast("B").tobytes()
    scale_payload = pack_x4t_scale_plane(scale)
    header = _RECORD_HEADER.pack(
        X4T_RECORD_MAGIC,
        X4T_VERSION,
        _RECORD_HEADER.size,
        matrix_id,
        X4T_TILE_ROWS,
        0,
        out_features,
        in_features,
        len(packed_payload),
        len(scale_payload),
        bytes(28),
    )
    return header + packed_payload + scale_payload


def unpack_x4t_matrix_record(
    payload: bytes,
    *,
    expected_matrix: str | None = None,
    production_shape: bool = True,
) -> X4TMatrix:
    """Decode one X4T record with canonicality and exact-byte validation."""

    if len(payload) < _RECORD_HEADER.size:
        raise ValueError("X4T record is truncated before its header")
    (
        magic,
        version,
        header_bytes,
        matrix_id,
        tile_rows,
        flags,
        out_features,
        in_features,
        packed_bytes,
        scale_bytes,
        reserved,
    ) = _RECORD_HEADER.unpack_from(payload)
    if magic != X4T_RECORD_MAGIC or version != X4T_VERSION:
        raise ValueError("X4T record has an unsupported magic or version")
    if header_bytes != _RECORD_HEADER.size or flags or any(reserved):
        raise ValueError("X4T record header is noncanonical")
    if matrix_id >= len(X4T_MATRIX_ORDER):
        raise ValueError("X4T record matrix ID is invalid")
    matrix = X4T_MATRIX_ORDER[matrix_id]
    if expected_matrix is not None and matrix != expected_matrix:
        raise ValueError("X4T record matrix ID disagrees with its directory slot")
    if tile_rows != X4T_TILE_ROWS:
        raise ValueError("X4T record uses an unsupported scale tile height")
    if not out_features or not in_features or in_features % (2 * C.MXFP4_BLOCK):
        raise ValueError("X4T record dimensions are invalid")
    expected_packed_bytes = out_features * in_features // 2
    if packed_bytes != expected_packed_bytes:
        raise ValueError("X4T record packed payload length disagrees with its shape")
    if len(payload) != header_bytes + packed_bytes + scale_bytes:
        raise ValueError("X4T record component lengths disagree with its total length")

    packed_start = header_bytes
    scale_start = packed_start + packed_bytes
    packed = torch.frombuffer(
        bytearray(payload[packed_start:scale_start]), dtype=torch.uint8
    ).reshape(out_features, in_features // 2)
    scale_payload = payload[scale_start:]
    scale = unpack_x4t_scale_plane(scale_payload)
    if tuple(scale.shape) != (out_features, in_features // C.MXFP4_BLOCK):
        raise ValueError("X4T record scale payload shape disagrees with its matrix shape")
    if pack_x4t_scale_plane(scale) != scale_payload:
        raise ValueError("X4T record scale payload is not canonical")
    _validate_matrix_tensors(
        matrix,
        packed,
        scale,
        production_shape=production_shape,
    )
    return X4TMatrix(matrix=matrix, packed=packed, scale=scale)


def x4t_record_bpw(record: bytes, *, alignment: bool = False) -> float:
    """Return record-only storage in bits per represented matrix weight."""

    decoded = unpack_x4t_matrix_record(record, production_shape=False)
    weights = int(decoded.packed.numel()) * 2
    stored = _align_up(len(record)) if alignment else len(record)
    return stored * 8 / weights


def x4t_matrix_storage_bytes(matrix: str, scale: torch.Tensor) -> int:
    """Return exact aligned X4T record bytes without reading the nibble plane."""

    expected_packed, expected_scale = _matrix_shapes(matrix)
    if (
        scale.dtype != torch.uint8
        or scale.device.type != "cpu"
        or not scale.is_contiguous()
        or tuple(scale.shape) != expected_scale
    ):
        raise ValueError(
            f"X4T {matrix} scale must be contiguous CPU uint8 {expected_scale}"
        )
    packed_bytes = expected_packed[0] * expected_packed[1]
    scale_bytes = x4t_scale_storage_bytes(scale)
    return _align_up(_RECORD_HEADER.size + packed_bytes + scale_bytes)


def x4t_scale_storage_bytes(scale: torch.Tensor) -> int:
    """Return the exact X4T scale payload size without serializing it.

    The fixed stream length depends only on the plane geometry.  Every value
    outside its row's best adjacent pair contributes one uint32 exception, so
    counting those values closes the payload size without allocating selector
    or exception byte streams.  The all-expert cost-index pass uses this path.
    """

    rows, columns = _validate_scale(scale)
    selector_bytes = math.ceil(columns / 8)
    fixed_bytes = (rows // X4T_TILE_ROWS) * X4T_TILE_ROWS * (1 + selector_bytes)
    source = scale.numpy()
    bases = _adjacent_bases_numpy(source).astype(np.int16)
    source_i16 = source.astype(np.int16)
    covered = (source_i16 == bases[:, None]) | (
        source_i16 == bases[:, None] + 1
    )
    exception_count = int(np.count_nonzero(~covered))
    return _HEADER.size + fixed_bytes + 4 * exception_count


def x4t_expert_storage_bytes(scales: dict[str, torch.Tensor]) -> int:
    """Return the exact sum of the three aligned X4T matrix records."""

    if set(scales) != set(X4T_MATRIX_ORDER):
        raise ValueError(f"X4T expert scales must contain {X4T_MATRIX_ORDER}")
    return sum(
        x4t_matrix_storage_bytes(matrix, scales[matrix])
        for matrix in X4T_MATRIX_ORDER
    )


def partition_x4t_components(
    raw: "PackedMXFP4Matrix | X4TMatrix",
    matrix: str,
    shard_count: int,
    shard_index: int,
    *,
    require_equal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice an X4T-exact matrix on 32-channel storage groups.

    Equal divisors yield identical local operand shapes. Other shard counts
    cover the same canonical matrix with widths differing by one 32-channel
    group, suitable for offline resharding or padded load-time preparation.
    """

    from kquant.qsrt import INTERMEDIATE_CHANNELS, LATENT_CHANNELS
    from kquant.source_weights import PackedMXFP4Matrix

    if isinstance(raw, X4TMatrix):
        if raw.matrix != matrix:
            raise ValueError("X4T matrix identity does not match the requested matrix")
    elif not isinstance(raw, PackedMXFP4Matrix):
        raise TypeError("raw must be a PackedMXFP4Matrix or X4TMatrix")
    _validate_matrix_tensors(
        matrix,
        raw.packed,
        raw.scale,
        production_shape=True,
    )
    groups = INTERMEDIATE_CHANNELS // C.MXFP4_BLOCK
    if isinstance(shard_count, bool) or not isinstance(shard_count, int):
        raise TypeError("shard_count must be an integer")
    if not 1 <= shard_count <= groups:
        raise ValueError(f"shard_count must be in 1..{groups}")
    if isinstance(shard_index, bool) or not isinstance(shard_index, int):
        raise TypeError("shard_index must be an integer")
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"shard_index must be in 0..{shard_count - 1}")
    if require_equal and groups % shard_count:
        raise ValueError("shard_count must divide the 32-channel X4T group axis")
    quotient, remainder = divmod(groups, shard_count)
    group_count = quotient + int(shard_index < remainder)
    first_group = shard_index * quotient + min(shard_index, remainder)
    first = first_group * C.MXFP4_BLOCK
    width = group_count * C.MXFP4_BLOCK
    if matrix in ("w1", "w3"):
        packed = raw.packed.narrow(0, first, width).contiguous()
        scale = raw.scale.narrow(0, first, width).contiguous()
    else:
        packed_width = width // 2
        scale_width = width // C.MXFP4_BLOCK
        packed = raw.packed.narrow(1, first // 2, packed_width).contiguous()
        scale = raw.scale.narrow(1, first_group, scale_width).contiguous()
    shard_weights = width * LATENT_CHANNELS
    if packed.numel() != shard_weights // 2 or scale.numel() != (
        shard_weights // C.MXFP4_BLOCK
    ):
        raise AssertionError("X4T shard byte accounting drifted")
    return packed, scale


def _entry_index(expert: int, matrix: str) -> int:
    if isinstance(expert, bool) or not isinstance(expert, int):
        raise TypeError("X4T expert ID must be an integer")
    if not 0 <= expert < X4T_EXPERTS_PER_LAYER:
        raise ValueError(
            f"X4T expert ID must be in 0..{X4T_EXPERTS_PER_LAYER - 1}"
        )
    return expert * len(X4T_MATRIX_ORDER) + _matrix_id(matrix)


def _canonical_layer_header(
    *,
    layer: int,
    file_bytes: int,
    record_count: int,
    directory_crc32: int,
    header_crc32: int,
) -> bytes:
    prefix = _LAYER_HEADER.pack(
        X4T_LAYER_MAGIC,
        X4T_VERSION,
        X4T_LAYER_HEADER_BYTES,
        layer,
        X4T_EXPERTS_PER_LAYER,
        len(X4T_MATRIX_ORDER),
        X4T_DIRECTORY_ENTRY_BYTES,
        0,
        X4T_LAYER_HEADER_BYTES,
        X4T_DIRECTORY_BYTES,
        X4T_DATA_OFFSET,
        file_bytes,
        record_count,
        directory_crc32,
        header_crc32,
        0,
    )
    return prefix + bytes(X4T_LAYER_HEADER_BYTES - len(prefix))


class X4TLayerWriter:
    """Atomic streaming writer for one sparse X4T MoE-layer sidecar."""

    def __init__(self, destination: str | Path, *, layer: int) -> None:
        if layer not in C.MOE_LAYERS:
            raise ValueError("X4T sidecar layer must be a Kimi-K3 MoE layer")
        self.destination = Path(destination)
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        if self.destination.exists():
            raise FileExistsError(self.destination)
        self.partial = self.destination.with_name(f".{self.destination.name}.partial")
        if self.partial.exists():
            raise FileExistsError(self.partial)
        self.layer = layer
        self._file = self.partial.open("x+b")
        self._file.write(bytes(X4T_DATA_OFFSET))
        self._entries = [
            X4TDirectoryEntry()
            for _ in range(X4T_EXPERTS_PER_LAYER * len(X4T_MATRIX_ORDER))
        ]
        self._cursor = X4T_DATA_OFFSET
        self._last_index = -1
        self._closed = False

    def add(
        self,
        expert: int,
        matrix: str,
        packed: torch.Tensor,
        scale: torch.Tensor,
    ) -> X4TDirectoryEntry:
        if self._closed:
            raise RuntimeError("X4T layer writer is already closed")
        index = _entry_index(expert, matrix)
        if index <= self._last_index:
            raise ValueError("X4T records must be added in expert-major canonical order")
        record = pack_x4t_matrix_record(matrix, packed, scale)
        if self._cursor % X4T_RECORD_ALIGNMENT:
            raise AssertionError("X4T writer cursor lost record alignment")
        self._file.seek(self._cursor)
        self._file.write(record)
        padded_end = _align_up(self._cursor + len(record))
        self._file.write(bytes(padded_end - self._cursor - len(record)))
        entry = X4TDirectoryEntry(
            offset=self._cursor,
            length=len(record),
            crc32=zlib.crc32(record),
            flags=1,
        )
        self._entries[index] = entry
        self._cursor = padded_end
        self._last_index = index
        return entry

    def close(self) -> None:
        if self._closed:
            return
        directory_entries = b"".join(entry.to_bytes() for entry in self._entries)
        directory = directory_entries + bytes(
            X4T_DIRECTORY_BYTES - len(directory_entries)
        )
        directory_crc32 = zlib.crc32(directory)
        record_count = sum(entry.present for entry in self._entries)
        header_zero_crc = _canonical_layer_header(
            layer=self.layer,
            file_bytes=self._cursor,
            record_count=record_count,
            directory_crc32=directory_crc32,
            header_crc32=0,
        )
        header = _canonical_layer_header(
            layer=self.layer,
            file_bytes=self._cursor,
            record_count=record_count,
            directory_crc32=directory_crc32,
            header_crc32=zlib.crc32(header_zero_crc),
        )
        self._file.seek(0)
        self._file.write(header)
        self._file.write(directory)
        self._file.truncate(self._cursor)
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()
        os.replace(self.partial, self.destination)
        self._closed = True

    def abort(self) -> None:
        if self._closed:
            return
        self._file.close()
        self.partial.unlink(missing_ok=True)
        self._closed = True

    def __enter__(self) -> "X4TLayerWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


class X4TLayerReader:
    """Validated random-access reader for one X4T MoE-layer sidecar."""

    def __init__(self, path: str | Path, *, verify_payloads: bool = True) -> None:
        self.path = Path(path)
        with self.path.open("rb") as handle:
            header = handle.read(X4T_LAYER_HEADER_BYTES)
            directory = handle.read(X4T_DIRECTORY_BYTES)
        if len(header) != X4T_LAYER_HEADER_BYTES or len(directory) != X4T_DIRECTORY_BYTES:
            raise ValueError("X4T layer sidecar is truncated before its data section")
        (
            magic,
            version,
            header_bytes,
            layer,
            experts,
            matrices,
            entry_bytes,
            flags,
            directory_offset,
            directory_bytes,
            data_offset,
            file_bytes,
            record_count,
            directory_crc32,
            header_crc32,
            reserved,
        ) = _LAYER_HEADER.unpack_from(header)
        if magic != X4T_LAYER_MAGIC or version != X4T_VERSION:
            raise ValueError("X4T layer sidecar has an unsupported magic or version")
        if (
            header_bytes != X4T_LAYER_HEADER_BYTES
            or experts != X4T_EXPERTS_PER_LAYER
            or matrices != len(X4T_MATRIX_ORDER)
            or entry_bytes != X4T_DIRECTORY_ENTRY_BYTES
            or flags
            or directory_offset != X4T_LAYER_HEADER_BYTES
            or directory_bytes != X4T_DIRECTORY_BYTES
            or data_offset != X4T_DATA_OFFSET
            or reserved
        ):
            raise ValueError("X4T layer header is noncanonical")
        if layer not in C.MOE_LAYERS:
            raise ValueError("X4T layer ID is invalid")
        if file_bytes != self.path.stat().st_size or file_bytes < data_offset:
            raise ValueError("X4T layer file length disagrees with its header")
        canonical_zero_crc = _canonical_layer_header(
            layer=layer,
            file_bytes=file_bytes,
            record_count=record_count,
            directory_crc32=directory_crc32,
            header_crc32=0,
        )
        canonical_header = _canonical_layer_header(
            layer=layer,
            file_bytes=file_bytes,
            record_count=record_count,
            directory_crc32=directory_crc32,
            header_crc32=header_crc32,
        )
        if header != canonical_header or zlib.crc32(canonical_zero_crc) != header_crc32:
            raise ValueError("X4T layer header checksum or padding is invalid")
        if zlib.crc32(directory) != directory_crc32:
            raise ValueError("X4T layer directory checksum is invalid")

        entry_count = X4T_EXPERTS_PER_LAYER * len(X4T_MATRIX_ORDER)
        used_directory = entry_count * X4T_DIRECTORY_ENTRY_BYTES
        if any(directory[used_directory:]):
            raise ValueError("X4T layer directory padding is nonzero")
        self.entries = tuple(
            X4TDirectoryEntry.from_bytes(
                memoryview(directory)[
                    index * X4T_DIRECTORY_ENTRY_BYTES : (index + 1)
                    * X4T_DIRECTORY_ENTRY_BYTES
                ]
            )
            for index in range(entry_count)
        )
        if sum(entry.present for entry in self.entries) != record_count:
            raise ValueError("X4T layer record count disagrees with its directory")
        cursor = data_offset
        handle = self.path.open("rb") if verify_payloads else None
        try:
            for entry in self.entries:
                if not entry.present:
                    if entry != X4TDirectoryEntry():
                        raise ValueError("absent X4T directory entry is noncanonical")
                    continue
                if (
                    entry.flags != 1
                    or entry.offset != cursor
                    or entry.length < _RECORD_HEADER.size
                ):
                    raise ValueError("present X4T directory entry is noncanonical")
                end = entry.offset + entry.length
                padded_end = _align_up(end)
                if padded_end > file_bytes:
                    raise ValueError("X4T directory entry extends beyond the layer file")
                if handle is not None:
                    handle.seek(entry.offset)
                    record = handle.read(entry.length)
                    padding = handle.read(padded_end - end)
                    if (
                        len(record) != entry.length
                        or zlib.crc32(record) != entry.crc32
                    ):
                        raise ValueError("X4T matrix record checksum is invalid")
                    if any(padding):
                        raise ValueError("X4T matrix record padding is nonzero")
                cursor = padded_end
        finally:
            if handle is not None:
                handle.close()
        if cursor != file_bytes:
            raise ValueError("X4T layer data section has unreferenced trailing bytes")
        self.layer = layer
        self.file_bytes = file_bytes
        self.record_count = record_count

    def has(self, expert: int, matrix: str) -> bool:
        return self.entries[_entry_index(expert, matrix)].present

    def record_bytes(self, expert: int, matrix: str) -> int:
        """Return the serialized record length, excluding alignment padding."""

        entry = self.entries[_entry_index(expert, matrix)]
        if not entry.present:
            raise KeyError((expert, matrix))
        return entry.length

    def read(self, expert: int, matrix: str) -> X4TMatrix:
        entry = self.entries[_entry_index(expert, matrix)]
        if not entry.present:
            raise KeyError((expert, matrix))
        with self.path.open("rb") as handle:
            handle.seek(entry.offset)
            payload = handle.read(entry.length)
        if len(payload) != entry.length or zlib.crc32(payload) != entry.crc32:
            raise ValueError("X4T matrix record changed after layer validation")
        return unpack_x4t_matrix_record(payload, expected_matrix=matrix)


def x4t_layer_path(root: str | Path, layer: int) -> Path:
    if layer not in C.MOE_LAYERS:
        raise ValueError("X4T sidecar layer must be a Kimi-K3 MoE layer")
    return Path(root) / f"x4t-layer-{layer:05d}.bin"
