"""Exact X4 records and layer containers for official MXFP4 experts.

X4 preserves the packed E2M1 nibble plane byte-for-byte and losslessly codes
the UE8M0 scale plane.  A layer container is sparse over experts, supports
constant-time directory lookup, and keeps every matrix independently
decodable.  Compression is deliberately performed before TP12 sharding so
the scale codec retains the complete row/column context of the source matrix.
"""

from __future__ import annotations

import os
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import torch

from kquant import constants as C
from kquant.mxfp4_scale_codec import (
    DEFAULT_TILE_ROWS,
    pack_scale_plane_compact,
    unpack_scale_plane_compact,
)


X4_LAYER_MAGIC = b"KQX4LYR\0"
X4_RECORD_MAGIC = b"KQX4"
X4_VERSION = 1
X4_LAYER_HEADER_BYTES = 4096
X4_DIRECTORY_BYTES = 65536
X4_DIRECTORY_ENTRY_BYTES = 24
X4_DATA_OFFSET = X4_LAYER_HEADER_BYTES + X4_DIRECTORY_BYTES
X4_RECORD_ALIGNMENT = 4096
X4_EXPERTS_PER_LAYER = 896
X4_MATRIX_ORDER = ("w1", "w3", "w2")

_LAYER_HEADER = struct.Struct("<8sHHIHHHHQQQQIIII")
_DIRECTORY_ENTRY = struct.Struct("<QQII")
_RECORD_HEADER = struct.Struct("<4sHHBBHIIQQ28s")

if _DIRECTORY_ENTRY.size != X4_DIRECTORY_ENTRY_BYTES:
    raise AssertionError("X4 directory entry layout drifted")
if _RECORD_HEADER.size != 64:
    raise AssertionError("X4 record header layout drifted")
if X4_EXPERTS_PER_LAYER * len(X4_MATRIX_ORDER) * X4_DIRECTORY_ENTRY_BYTES > X4_DIRECTORY_BYTES:
    raise AssertionError("X4 fixed directory is too small")


def _align_up(value: int, alignment: int = X4_RECORD_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def _matrix_id(matrix: str) -> int:
    try:
        return X4_MATRIX_ORDER.index(matrix)
    except ValueError as exc:
        raise ValueError(f"unsupported X4 matrix: {matrix}") from exc


def _matrix_shapes(matrix: str) -> tuple[tuple[int, int], tuple[int, int]]:
    if matrix not in C.EXPERT_SHAPES:
        raise ValueError(f"unsupported X4 matrix: {matrix}")
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
            raise ValueError(f"X4 {name} must be a two-dimensional CPU uint8 tensor")
        if not value.is_contiguous():
            raise ValueError(f"X4 {name} tensor must be contiguous")
    packed_rows, packed_columns = map(int, packed.shape)
    scale_rows, scale_columns = map(int, scale.shape)
    if not packed_rows or not packed_columns or not scale_rows or not scale_columns:
        raise ValueError("X4 matrices must have nonzero dimensions")
    if packed_rows != scale_rows or packed_columns * 2 != scale_columns * C.MXFP4_BLOCK:
        raise ValueError("X4 packed and scale shapes do not describe the same matrix")
    if production_shape:
        expected_packed, expected_scale = _matrix_shapes(matrix)
        if tuple(packed.shape) != expected_packed or tuple(scale.shape) != expected_scale:
            raise ValueError(
                f"X4 {matrix} shape mismatch: expected packed {expected_packed} and "
                f"scale {expected_scale}"
            )


@dataclass(frozen=True)
class X4Matrix:
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
class X4DirectoryEntry:
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
    def from_bytes(cls, payload: bytes | memoryview) -> "X4DirectoryEntry":
        if len(payload) != _DIRECTORY_ENTRY.size:
            raise ValueError("X4 directory entry has the wrong length")
        return cls(*_DIRECTORY_ENTRY.unpack(payload))


def pack_x4_matrix_record(
    matrix: str,
    packed: torch.Tensor,
    scale: torch.Tensor,
    *,
    production_shape: bool = True,
) -> bytes:
    """Encode one exact MXFP4 matrix as an independently decodable X4 record."""

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
    scale_payload = pack_scale_plane_compact(
        scale,
        tile_rows=DEFAULT_TILE_ROWS,
    )
    header = _RECORD_HEADER.pack(
        X4_RECORD_MAGIC,
        X4_VERSION,
        _RECORD_HEADER.size,
        matrix_id,
        DEFAULT_TILE_ROWS,
        0,
        out_features,
        in_features,
        len(packed_payload),
        len(scale_payload),
        bytes(28),
    )
    return header + packed_payload + scale_payload


def unpack_x4_matrix_record(
    payload: bytes,
    *,
    expected_matrix: str | None = None,
    production_shape: bool = True,
) -> X4Matrix:
    """Decode one X4 record with canonicality and exact-byte validation."""

    if len(payload) < _RECORD_HEADER.size:
        raise ValueError("X4 record is truncated before its header")
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
    if magic != X4_RECORD_MAGIC or version != X4_VERSION:
        raise ValueError("X4 record has an unsupported magic or version")
    if header_bytes != _RECORD_HEADER.size or flags or any(reserved):
        raise ValueError("X4 record header is noncanonical")
    if matrix_id >= len(X4_MATRIX_ORDER):
        raise ValueError("X4 record matrix ID is invalid")
    matrix = X4_MATRIX_ORDER[matrix_id]
    if expected_matrix is not None and matrix != expected_matrix:
        raise ValueError("X4 record matrix ID disagrees with its directory slot")
    if tile_rows != DEFAULT_TILE_ROWS:
        raise ValueError("X4 record uses an unsupported scale tile height")
    if not out_features or not in_features or in_features % (2 * C.MXFP4_BLOCK):
        raise ValueError("X4 record dimensions are invalid")
    expected_packed_bytes = out_features * in_features // 2
    if packed_bytes != expected_packed_bytes:
        raise ValueError("X4 record packed payload length disagrees with its shape")
    if len(payload) != header_bytes + packed_bytes + scale_bytes:
        raise ValueError("X4 record component lengths disagree with its total length")

    packed_start = header_bytes
    scale_start = packed_start + packed_bytes
    packed = torch.frombuffer(
        bytearray(payload[packed_start:scale_start]), dtype=torch.uint8
    ).reshape(out_features, in_features // 2)
    scale_payload = payload[scale_start:]
    scale = unpack_scale_plane_compact(scale_payload)
    if tuple(scale.shape) != (out_features, in_features // C.MXFP4_BLOCK):
        raise ValueError("X4 record scale payload shape disagrees with its matrix shape")
    if pack_scale_plane_compact(scale, tile_rows=tile_rows) != scale_payload:
        raise ValueError("X4 record scale payload is not canonical")
    _validate_matrix_tensors(
        matrix,
        packed,
        scale,
        production_shape=production_shape,
    )
    return X4Matrix(matrix=matrix, packed=packed, scale=scale)


def x4_record_bpw(record: bytes, *, alignment: bool = False) -> float:
    """Return record-only storage in bits per represented matrix weight."""

    decoded = unpack_x4_matrix_record(record, production_shape=False)
    weights = int(decoded.packed.numel()) * 2
    stored = _align_up(len(record)) if alignment else len(record)
    return stored * 8 / weights


def x4_matrix_storage_bytes(matrix: str, scale: torch.Tensor) -> int:
    """Return exact aligned record bytes without reading the nibble plane."""

    expected_packed, expected_scale = _matrix_shapes(matrix)
    if (
        scale.dtype != torch.uint8
        or scale.device.type != "cpu"
        or not scale.is_contiguous()
        or tuple(scale.shape) != expected_scale
    ):
        raise ValueError(
            f"X4 {matrix} scale must be contiguous CPU uint8 {expected_scale}"
        )
    packed_bytes = expected_packed[0] * expected_packed[1]
    scale_bytes = len(
        pack_scale_plane_compact(scale, tile_rows=DEFAULT_TILE_ROWS)
    )
    return _align_up(_RECORD_HEADER.size + packed_bytes + scale_bytes)


def x4_expert_storage_bytes(scales: dict[str, torch.Tensor]) -> int:
    """Return the exact sum of the three aligned X4 matrix records."""

    if set(scales) != set(X4_MATRIX_ORDER):
        raise ValueError(f"X4 expert scales must contain {X4_MATRIX_ORDER}")
    return sum(x4_matrix_storage_bytes(matrix, scales[matrix]) for matrix in X4_MATRIX_ORDER)


def _entry_index(expert: int, matrix: str) -> int:
    if isinstance(expert, bool) or not isinstance(expert, int):
        raise TypeError("X4 expert ID must be an integer")
    if not 0 <= expert < X4_EXPERTS_PER_LAYER:
        raise ValueError(f"X4 expert ID must be in 0..{X4_EXPERTS_PER_LAYER - 1}")
    return expert * len(X4_MATRIX_ORDER) + _matrix_id(matrix)


def _canonical_layer_header(
    *,
    layer: int,
    file_bytes: int,
    record_count: int,
    directory_crc32: int,
    header_crc32: int,
) -> bytes:
    prefix = _LAYER_HEADER.pack(
        X4_LAYER_MAGIC,
        X4_VERSION,
        X4_LAYER_HEADER_BYTES,
        layer,
        X4_EXPERTS_PER_LAYER,
        len(X4_MATRIX_ORDER),
        X4_DIRECTORY_ENTRY_BYTES,
        0,
        X4_LAYER_HEADER_BYTES,
        X4_DIRECTORY_BYTES,
        X4_DATA_OFFSET,
        file_bytes,
        record_count,
        directory_crc32,
        header_crc32,
        0,
    )
    return prefix + bytes(X4_LAYER_HEADER_BYTES - len(prefix))


class X4LayerWriter:
    """Atomic streaming writer for one sparse X4 MoE-layer sidecar."""

    def __init__(self, destination: str | Path, *, layer: int) -> None:
        if layer not in C.MOE_LAYERS:
            raise ValueError("X4 sidecar layer must be a Kimi-K3 MoE layer")
        self.destination = Path(destination)
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        if self.destination.exists():
            raise FileExistsError(self.destination)
        self.partial = self.destination.with_name(f".{self.destination.name}.partial")
        if self.partial.exists():
            raise FileExistsError(self.partial)
        self.layer = layer
        self._file = self.partial.open("x+b")
        self._file.write(bytes(X4_DATA_OFFSET))
        self._entries = [
            X4DirectoryEntry()
            for _ in range(X4_EXPERTS_PER_LAYER * len(X4_MATRIX_ORDER))
        ]
        self._cursor = X4_DATA_OFFSET
        self._last_index = -1
        self._closed = False

    def add(
        self,
        expert: int,
        matrix: str,
        packed: torch.Tensor,
        scale: torch.Tensor,
    ) -> X4DirectoryEntry:
        if self._closed:
            raise RuntimeError("X4 layer writer is already closed")
        index = _entry_index(expert, matrix)
        if index <= self._last_index:
            raise ValueError("X4 records must be added in expert-major canonical order")
        record = pack_x4_matrix_record(matrix, packed, scale)
        if self._cursor % X4_RECORD_ALIGNMENT:
            raise AssertionError("X4 writer cursor lost record alignment")
        self._file.seek(self._cursor)
        self._file.write(record)
        padded_end = _align_up(self._cursor + len(record))
        self._file.write(bytes(padded_end - self._cursor - len(record)))
        entry = X4DirectoryEntry(
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
        directory = directory_entries + bytes(X4_DIRECTORY_BYTES - len(directory_entries))
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

    def __enter__(self) -> "X4LayerWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


class X4LayerReader:
    """Validated random-access reader for one X4 MoE-layer sidecar."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with self.path.open("rb") as handle:
            header = handle.read(X4_LAYER_HEADER_BYTES)
            directory = handle.read(X4_DIRECTORY_BYTES)
        if len(header) != X4_LAYER_HEADER_BYTES or len(directory) != X4_DIRECTORY_BYTES:
            raise ValueError("X4 layer sidecar is truncated before its data section")
        values = _LAYER_HEADER.unpack_from(header)
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
        ) = values
        if magic != X4_LAYER_MAGIC or version != X4_VERSION:
            raise ValueError("X4 layer sidecar has an unsupported magic or version")
        if (
            header_bytes != X4_LAYER_HEADER_BYTES
            or experts != X4_EXPERTS_PER_LAYER
            or matrices != len(X4_MATRIX_ORDER)
            or entry_bytes != X4_DIRECTORY_ENTRY_BYTES
            or flags
            or directory_offset != X4_LAYER_HEADER_BYTES
            or directory_bytes != X4_DIRECTORY_BYTES
            or data_offset != X4_DATA_OFFSET
            or reserved
        ):
            raise ValueError("X4 layer header is noncanonical")
        if layer not in C.MOE_LAYERS:
            raise ValueError("X4 layer ID is invalid")
        if file_bytes != self.path.stat().st_size or file_bytes < data_offset:
            raise ValueError("X4 layer file length disagrees with its header")
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
            raise ValueError("X4 layer header checksum or padding is invalid")
        if zlib.crc32(directory) != directory_crc32:
            raise ValueError("X4 layer directory checksum is invalid")

        entry_count = X4_EXPERTS_PER_LAYER * len(X4_MATRIX_ORDER)
        used_directory = entry_count * X4_DIRECTORY_ENTRY_BYTES
        if any(directory[used_directory:]):
            raise ValueError("X4 layer directory padding is nonzero")
        self.entries = tuple(
            X4DirectoryEntry.from_bytes(
                memoryview(directory)[
                    index * X4_DIRECTORY_ENTRY_BYTES : (index + 1)
                    * X4_DIRECTORY_ENTRY_BYTES
                ]
            )
            for index in range(entry_count)
        )
        if sum(entry.present for entry in self.entries) != record_count:
            raise ValueError("X4 layer record count disagrees with its directory")
        cursor = data_offset
        with self.path.open("rb") as handle:
            for entry in self.entries:
                if not entry.present:
                    if entry != X4DirectoryEntry():
                        raise ValueError("absent X4 directory entry is noncanonical")
                    continue
                if entry.flags != 1 or entry.offset != cursor or entry.length < _RECORD_HEADER.size:
                    raise ValueError("present X4 directory entry is noncanonical")
                end = entry.offset + entry.length
                padded_end = _align_up(end)
                if padded_end > file_bytes:
                    raise ValueError("X4 directory entry extends beyond the layer file")
                handle.seek(entry.offset)
                record = handle.read(entry.length)
                padding = handle.read(padded_end - end)
                if len(record) != entry.length or zlib.crc32(record) != entry.crc32:
                    raise ValueError("X4 matrix record checksum is invalid")
                if any(padding):
                    raise ValueError("X4 matrix record padding is nonzero")
                cursor = padded_end
        if cursor != file_bytes:
            raise ValueError("X4 layer data section has unreferenced trailing bytes")
        self.layer = layer
        self.file_bytes = file_bytes
        self.record_count = record_count

    def has(self, expert: int, matrix: str) -> bool:
        return self.entries[_entry_index(expert, matrix)].present

    def read(self, expert: int, matrix: str) -> X4Matrix:
        entry = self.entries[_entry_index(expert, matrix)]
        if not entry.present:
            raise KeyError((expert, matrix))
        with self.path.open("rb") as handle:
            handle.seek(entry.offset)
            payload = handle.read(entry.length)
        if len(payload) != entry.length or zlib.crc32(payload) != entry.crc32:
            raise ValueError("X4 matrix record changed after layer validation")
        return unpack_x4_matrix_record(payload, expected_matrix=matrix)


def x4_layer_path(root: str | Path, layer: int) -> Path:
    if layer not in C.MOE_LAYERS:
        raise ValueError("X4 sidecar layer must be a Kimi-K3 MoE layer")
    return Path(root) / f"x4-layer-{layer:05d}.bin"
