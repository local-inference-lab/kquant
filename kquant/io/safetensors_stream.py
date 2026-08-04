"""Bounded-memory writer for safetensors files with known tensor geometry."""

from __future__ import annotations

import json
import fcntl
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import torch


_DTYPE_CODES = {
    torch.bool: "BOOL",
    torch.uint8: "U8",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.float16: "F16",
    torch.float32: "F32",
    torch.float64: "F64",
}


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.name or self.name == "__metadata__":
            raise ValueError("tensor name must be nonempty and non-reserved")
        if self.dtype not in _DTYPE_CODES:
            raise TypeError(f"unsupported streaming safetensors dtype: {self.dtype}")
        if any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 0
            for dimension in self.shape
        ):
            raise ValueError("tensor dimensions must be non-negative integers")

    @property
    def bytes(self) -> int:
        return math.prod(self.shape) * self.dtype.itemsize


def _pwrite_exact(
    descriptor: int,
    payload: bytes | bytearray | memoryview,
    offset: int,
) -> None:
    view = memoryview(payload).cast("B")
    cursor = 0
    while cursor < len(view):
        written = os.pwrite(descriptor, view[cursor:], offset + cursor)
        if written <= 0:
            raise OSError("pwrite made no forward progress")
        cursor += written


class AtomicSafetensorsWriter:
    """Write predeclared tensors one at a time and publish only on closure."""

    def __init__(
        self,
        path: str | Path,
        specs: Iterable[TensorSpec],
        *,
        metadata: Mapping[str, str] | None = None,
    ):
        self.path = Path(path)
        self.specs = tuple(specs)
        if not self.specs:
            raise ValueError("a safetensors stream must contain at least one tensor")
        names = [spec.name for spec in self.specs]
        if len(set(names)) != len(names):
            raise ValueError("safetensors stream contains duplicate tensor names")
        if metadata is not None and any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in metadata.items()
        ):
            raise TypeError("safetensors metadata keys and values must be strings")
        self._by_name = {spec.name: spec for spec in self.specs}
        self._offsets: dict[str, tuple[int, int]] = {}
        cursor = 0
        header: dict[str, object] = {}
        if metadata is not None:
            header["__metadata__"] = dict(metadata)
        for spec in self.specs:
            end = cursor + spec.bytes
            self._offsets[spec.name] = (cursor, end)
            header[spec.name] = {
                "dtype": _DTYPE_CODES[spec.dtype],
                "shape": list(spec.shape),
                "data_offsets": [cursor, end],
            }
            cursor = end
        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        encoded += b" " * (-len(encoded) % 8)
        self._prefix = struct.pack("<Q", len(encoded)) + encoded
        self._data_offset = len(self._prefix)
        self.total_bytes = self._data_offset + cursor
        self._written: set[str] = set()
        self._descriptor: int | None = None
        self._temporary = self.path.with_name(f".{self.path.name}.partial")

    @staticmethod
    def partial_path(path: str | Path) -> Path:
        path = Path(path)
        return path.with_name(f".{path.name}.partial")

    @classmethod
    def discard_stale_partial(cls, path: str | Path) -> bool:
        """Remove an abandoned partial while refusing to disturb a live writer."""

        temporary = cls.partial_path(path)
        try:
            descriptor = os.open(temporary, os.O_WRONLY)
        except FileNotFoundError:
            return False
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(
                    f"safetensors partial is still owned by a live writer: {temporary}"
                ) from exc
            try:
                temporary.unlink()
            except FileNotFoundError:
                return False
        finally:
            os.close(descriptor)
        return True

    def __enter__(self) -> "AtomicSafetensorsWriter":
        if self.path.exists():
            raise FileExistsError(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._descriptor = os.open(
            self._temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
        try:
            fcntl.flock(self._descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.ftruncate(self._descriptor, self.total_bytes)
            _pwrite_exact(self._descriptor, self._prefix, 0)
        except BaseException:
            self.abort()
            raise
        return self

    def write(self, name: str, tensor: torch.Tensor) -> None:
        if self._descriptor is None:
            raise RuntimeError("safetensors writer is not open")
        try:
            spec = self._by_name[name]
        except KeyError as exc:
            raise KeyError(f"undeclared safetensors tensor: {name}") from exc
        if name in self._written:
            raise ValueError(f"safetensors tensor was written twice: {name}")
        if tensor.dtype != spec.dtype or tuple(tensor.shape) != spec.shape:
            raise ValueError(
                f"{name} has {tensor.dtype} {tuple(tensor.shape)}, expected "
                f"{spec.dtype} {spec.shape}"
            )
        if tensor.device.type != "cpu":
            raise ValueError(f"{name} must be on CPU")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        begin, end = self._offsets[name]
        view = memoryview(tensor.numpy()).cast("B")
        if len(view) != end - begin:
            raise AssertionError("safetensors tensor byte accounting drifted")
        _pwrite_exact(self._descriptor, view, self._data_offset + begin)
        self._written.add(name)

    def write_many(self, tensors: Mapping[str, torch.Tensor]) -> None:
        for name, tensor in tensors.items():
            self.write(name, tensor)

    def finalize(self) -> Path:
        if self._descriptor is None:
            raise RuntimeError("safetensors writer is not open")
        missing = set(self._by_name).difference(self._written)
        if missing:
            preview = sorted(missing)[:4]
            raise ValueError(
                f"safetensors stream is missing {len(missing)} tensors: {preview}"
            )
        descriptor = self._descriptor
        self._descriptor = None
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if self._temporary.stat().st_size != self.total_bytes:
            self._temporary.unlink(missing_ok=True)
            raise ValueError("streamed safetensors file has the wrong final size")
        self._temporary.replace(self.path)
        directory = os.open(
            self.path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return self.path

    def abort(self) -> None:
        if self._descriptor is not None:
            descriptor = self._descriptor
            self._descriptor = None
            os.close(descriptor)
        self._temporary.unlink(missing_ok=True)

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_type is None:
            try:
                self.finalize()
            except BaseException:
                self.abort()
                raise
        else:
            self.abort()
        return False
