"""GPU-tile-friendly exact coding for MXFP4 UE8M0 scale planes.

X4T keeps the E2M1 nibble plane unchanged. FC1 nibbles are serialized in their
natural output-row order; FC2 nibbles are serialized by 32-channel input groups
so any tensor-parallel whole-atom extent is contiguous on disk. It represents
each 16-row scale slab with one adjacent two-value palette per row and a fixed-stride selector
bitmap.  Values outside that adjacent pair are carried by a sorted uint32
exception stream::

    bits  0..23  logical row-major scale index
    bits 24..31  exact UE8M0 byte

The fixed stream is deliberately trivial for a GPU CTA to decode and the
exception stream is suitable for a second parallel scatter. Hot-path decoding
needs no variable tile offsets, prefix sums, or exception searches.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from kquant import constants as C


X4T_VERSION = 2
X4T_TILE_ROWS = 16
X4T_POSITION_BITS = 24
X4T_POSITION_MASK = (1 << X4T_POSITION_BITS) - 1

X4T_SAFETENSORS_SCHEMA = "kquant_x4t_layer_v1"
X4T_SAFETENSORS_HEADER_BYTES = 4096
# Three exception-offset vectors each contain one leading int64. The remaining
# 28 bytes per expert are charged in x4t_expert_storage_bytes(). Keeping the
# standards-valid safetensors JSON header padded to 4 KiB makes exact allocation
# additive without reintroducing a private container or record format.
X4T_LAYER_FIXED_BYTES = 8 + X4T_SAFETENSORS_HEADER_BYTES + 3 * 8
X4T_EXPERTS_PER_LAYER = 896
X4T_MATRIX_ORDER = ("w1", "w3", "w2")


def _validate_scale(scale: torch.Tensor) -> tuple[int, int]:
    if (
        scale.dtype != torch.uint8
        or scale.ndim != 2
        or scale.device.type != "cpu"
        or not scale.is_contiguous()
    ):
        raise ValueError(
            "X4T scale must be a contiguous two-dimensional CPU uint8 tensor"
        )
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

    These tensors are stored directly in the canonical X4T safetensors layer,
    so checkpoint preparation can preserve X4T all the way to the GPU instead
    of reconstructing a dense scale plane at model load.
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


def effective_x4t_bpw(scale: torch.Tensor) -> float:
    """Return nibble plane plus X4T scale-tensor bytes in bits per weight."""

    rows, columns = _validate_scale(scale)
    weights = rows * columns * 32
    return 4.0 + x4t_scale_storage_bytes(scale) * 8 / weights


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


def x4t_scale_storage_bytes(scale: torch.Tensor) -> int:
    """Return fixed-stream plus exception bytes without serializing them.

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
    return fixed_bytes + 4 * exception_count


def x4t_scale_exception_count(scale: torch.Tensor) -> int:
    """Return the exact number of uint32 exceptions in a scale plane."""

    rows, columns = _validate_scale(scale)
    del rows, columns
    source = scale.numpy()
    bases = _adjacent_bases_numpy(source).astype(np.int16)
    source_i16 = source.astype(np.int16)
    covered = (source_i16 == bases[:, None]) | (
        source_i16 == bases[:, None] + 1
    )
    return int(np.count_nonzero(~covered))


def _scale_fixed_bytes(shape: tuple[int, int]) -> int:
    rows, columns = shape
    selector_bytes = math.ceil(columns / 8)
    return (rows // X4T_TILE_ROWS) * X4T_TILE_ROWS * (1 + selector_bytes)


def x4t_matrix_storage_bytes(matrix: str, scale: torch.Tensor) -> int:
    """Return one matrix's additive safetensors data contribution."""

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
    return x4t_matrix_storage_bytes_from_exception_count(
        matrix, x4t_scale_exception_count(scale)
    )


def x4t_matrix_storage_bytes_from_exception_count(
    matrix: str, exception_count: int
) -> int:
    """Return additive matrix bytes from an already-counted scale stream."""

    expected_packed, expected_scale = _matrix_shapes(matrix)
    if isinstance(exception_count, bool) or not isinstance(exception_count, int):
        raise TypeError("X4T exception count must be an integer")
    if exception_count < 0:
        raise ValueError("X4T exception count must be non-negative")
    return (
        math.prod(expected_packed)
        + _scale_fixed_bytes(expected_scale)
        + 4 * exception_count
    )


def x4t_expert_storage_bytes(scales: dict[str, torch.Tensor]) -> int:
    """Return one expert's exact additive safetensors contribution."""

    if set(scales) != set(X4T_MATRIX_ORDER):
        raise ValueError(f"X4T expert scales must contain {X4T_MATRIX_ORDER}")
    # Three int64 offset entries plus one int32 expert ID.
    return 28 + sum(
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


def _validate_expert(expert: int) -> None:
    if isinstance(expert, bool) or not isinstance(expert, int):
        raise TypeError("X4T expert ID must be an integer")
    if not 0 <= expert < X4T_EXPERTS_PER_LAYER:
        raise ValueError(
            f"X4T expert ID must be in 0..{X4T_EXPERTS_PER_LAYER - 1}"
        )


def _tensor_name(matrix: str, part: str) -> str:
    _matrix_id(matrix)
    return f"{matrix}.{part}"


def _packed_storage_shape(matrix: str, experts: int) -> tuple[int, ...]:
    packed, _ = _matrix_shapes(matrix)
    if matrix == "w2":
        out_features, packed_columns = packed
        return (
            experts,
            packed_columns * 2 // C.MXFP4_BLOCK,
            out_features,
            C.MXFP4_BLOCK // 2,
        )
    return (experts, *packed)


def _x4t_tensor_layout(
    *,
    layer: int,
    experts: int,
    exception_counts: dict[str, int],
) -> tuple[bytes, tuple[tuple[str, str, tuple[int, ...], int], ...], int]:
    """Build the canonical fixed-header safetensors layout."""

    if layer not in C.MOE_LAYERS:
        raise ValueError("X4T safetensors layer must be a Kimi-K3 MoE layer")
    if not 0 <= experts <= X4T_EXPERTS_PER_LAYER:
        raise ValueError("X4T safetensors expert count is invalid")
    if set(exception_counts) != set(X4T_MATRIX_ORDER):
        raise ValueError("X4T exception inventory is incomplete")
    if any(value < 0 for value in exception_counts.values()):
        raise ValueError("X4T exception counts must be non-negative")

    specs: list[tuple[str, str, tuple[int, ...], int]] = []
    for matrix in X4T_MATRIX_ORDER:
        specs.append(
            (
                _tensor_name(matrix, "scale_exception_offsets"),
                "I64",
                (experts + 1,),
                8 * (experts + 1),
            )
        )
    specs.append(("expert_ids", "I32", (experts,), 4 * experts))
    for matrix in X4T_MATRIX_ORDER:
        _, scale_shape = _matrix_shapes(matrix)
        specs.extend(
            (
                (
                    _tensor_name(matrix, "packed"),
                    "U8",
                    _packed_storage_shape(matrix, experts),
                    math.prod(_packed_storage_shape(matrix, experts)),
                ),
                (
                    _tensor_name(matrix, "scale_fixed"),
                    "U8",
                    (experts, _scale_fixed_bytes(scale_shape)),
                    experts * _scale_fixed_bytes(scale_shape),
                ),
                (
                    _tensor_name(matrix, "scale_exceptions"),
                    "U8",
                    (4 * exception_counts[matrix],),
                    4 * exception_counts[matrix],
                ),
            )
        )

    metadata = {
        "format": "pt",
        "schema": X4T_SAFETENSORS_SCHEMA,
        "version": str(X4T_VERSION),
        "layer": str(layer),
        "experts": str(experts),
        "expert_capacity": str(X4T_EXPERTS_PER_LAYER),
        "matrix_order": ",".join(X4T_MATRIX_ORDER),
        "scale_codec": "x4t-adjacent-pair-fixed-stream-v1",
        "w2_packed_layout": "group-major-32-channel-v1",
        "exact_mxfp4_reconstruction": "true",
    }
    header: dict[str, object] = {"__metadata__": metadata}
    offset = 0
    for name, dtype, shape, size in specs:
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    raw = json.dumps(header, separators=(",", ":")).encode()
    if len(raw) > X4T_SAFETENSORS_HEADER_BYTES:
        raise ValueError("X4T safetensors tensor directory exceeds its 4 KiB header")
    padded = raw + bytes([32]) * (X4T_SAFETENSORS_HEADER_BYTES - len(raw))
    return struct.pack("<Q", len(padded)) + padded, tuple(specs), offset


def x4t_layer_storage_bytes(
    layer: int,
    expert_storage_bytes: np.ndarray | list[int] | tuple[int, ...],
) -> int:
    """Return exact layer bytes from additive per-expert costs."""

    if layer not in C.MOE_LAYERS:
        raise ValueError("X4T safetensors layer must be a Kimi-K3 MoE layer")
    costs = np.asarray(expert_storage_bytes, dtype=np.int64)
    if (
        costs.ndim != 1
        or costs.size > X4T_EXPERTS_PER_LAYER
        or np.any(costs <= 0)
    ):
        raise ValueError("X4T layer expert costs must be a positive vector")
    return X4T_LAYER_FIXED_BYTES + int(costs.sum())


def _decode_scale_components(
    fixed: torch.Tensor,
    exceptions: torch.Tensor,
    *,
    rows: int,
    columns: int,
) -> torch.Tensor:
    selector_bytes = math.ceil(columns / 8)
    tile_bytes = X4T_TILE_ROWS * (1 + selector_bytes)
    if (
        fixed.dtype != torch.uint8
        or fixed.numel() != (rows // X4T_TILE_ROWS) * tile_bytes
    ):
        raise ValueError("X4T fixed scale tensor has invalid geometry")
    if exceptions.dtype != torch.uint32 or exceptions.ndim != 1:
        raise ValueError("X4T exception tensor must be one-dimensional uint32")
    fixed = fixed.reshape(rows // X4T_TILE_ROWS, tile_bytes)
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
    previous = -1
    flat = result.view(-1)
    for entry in exceptions.tolist():
        position = entry & X4T_POSITION_MASK
        value = entry >> X4T_POSITION_BITS
        if position >= flat.numel() or position <= previous:
            raise ValueError(
                "X4T exception positions must be valid and strictly increasing"
            )
        previous = position
        row = position // columns
        base = int(bases[row])
        if value in (base, base + 1):
            raise ValueError("X4T exception redundantly names an adjacent-palette value")
        flat[position] = value
    expected_fixed, expected_exceptions = pack_x4t_scale_components(result.contiguous())
    if not torch.equal(expected_fixed, fixed.reshape(-1)) or not torch.equal(
        expected_exceptions, exceptions
    ):
        raise ValueError("X4T safetensors scale stream is not canonical")
    return result


class X4TLayerWriter:
    """Atomic streaming writer for one sparse X4T safetensors layer."""

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
        self._spool_paths = {
            (matrix, part): self.destination.with_name(
                f".{self.destination.name}.{matrix}.{part}.partial"
            )
            for matrix in X4T_MATRIX_ORDER
            for part in ("packed", "scale_fixed", "scale_exceptions")
        }
        if any(path.exists() for path in self._spool_paths.values()):
            raise FileExistsError("an X4T safetensors spool file already exists")
        self._spools = {
            key: path.open("xb") for key, path in self._spool_paths.items()
        }
        self._experts: list[int] = []
        self._exception_offsets = {matrix: [0] for matrix in X4T_MATRIX_ORDER}
        self._last_index = -1
        self._adds = 0
        self._closed = False

    def add(
        self,
        expert: int,
        matrix: str,
        packed: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        if self._closed:
            raise RuntimeError("X4T layer writer is already closed")
        _validate_expert(expert)
        matrix_id = _matrix_id(matrix)
        index = expert * len(X4T_MATRIX_ORDER) + matrix_id
        if index <= self._last_index:
            raise ValueError("X4T tensors must be added in expert-major canonical order")
        expected_matrix_id = self._adds % len(X4T_MATRIX_ORDER)
        if matrix_id != expected_matrix_id:
            raise ValueError("every X4T expert must contain w1, w3, and w2 in order")
        if matrix_id == 0:
            self._experts.append(expert)
        elif not self._experts or self._experts[-1] != expert:
            raise ValueError("X4T matrix triplets must belong to one expert")
        _validate_matrix_tensors(matrix, packed, scale, production_shape=True)
        if matrix == "w2":
            out_features, in_features = C.EXPERT_SHAPES[matrix]
            packed_storage = (
                packed.reshape(
                    out_features,
                    in_features // C.MXFP4_BLOCK,
                    C.MXFP4_BLOCK // 2,
                )
                .permute(1, 0, 2)
                .contiguous()
            )
        else:
            packed_storage = packed
        fixed, exceptions = pack_x4t_scale_components(scale)
        self._spools[(matrix, "packed")].write(packed_storage.numpy().tobytes())
        self._spools[(matrix, "scale_fixed")].write(fixed.numpy().tobytes())
        self._spools[(matrix, "scale_exceptions")].write(
            exceptions.numpy().astype("<u4", copy=False).tobytes()
        )
        self._exception_offsets[matrix].append(
            self._exception_offsets[matrix][-1] + int(exceptions.numel())
        )
        self._last_index = index
        self._adds += 1

    def close(self) -> None:
        if self._closed:
            return
        if self._adds % len(X4T_MATRIX_ORDER):
            raise ValueError("X4T safetensors layer ends with an incomplete expert")
        for handle in self._spools.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        counts = {
            matrix: self._exception_offsets[matrix][-1]
            for matrix in X4T_MATRIX_ORDER
        }
        header, specs, data_bytes = _x4t_tensor_layout(
            layer=self.layer,
            experts=len(self._experts),
            exception_counts=counts,
        )
        with self.partial.open("xb") as output:
            output.write(header)
            for name, _, _, expected_bytes in specs:
                if name.endswith("scale_exception_offsets"):
                    matrix = name.split(".", 1)[0]
                    payload = np.asarray(
                        self._exception_offsets[matrix], dtype="<i8"
                    ).tobytes()
                    output.write(payload)
                elif name == "expert_ids":
                    output.write(np.asarray(self._experts, dtype="<i4").tobytes())
                else:
                    matrix, part = name.split(".", 1)
                    path = self._spool_paths[(matrix, part)]
                    if path.stat().st_size != expected_bytes:
                        raise AssertionError(f"X4T spool byte count drifted for {name}")
                    with path.open("rb") as source:
                        shutil.copyfileobj(source, output, length=8 << 20)
            if output.tell() != len(header) + data_bytes:
                raise AssertionError("X4T safetensors byte accounting drifted")
            output.flush()
            os.fsync(output.fileno())
        os.replace(self.partial, self.destination)
        for path in self._spool_paths.values():
            path.unlink(missing_ok=True)
        self._closed = True

    def abort(self) -> None:
        if self._closed:
            return
        for handle in self._spools.values():
            if not handle.closed:
                handle.close()
        for path in self._spool_paths.values():
            path.unlink(missing_ok=True)
        self.partial.unlink(missing_ok=True)
        self._closed = True

    def __enter__(self) -> "X4TLayerWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            try:
                self.close()
            except BaseException:
                self.abort()
                raise
        else:
            self.abort()


class X4TLayerReader:
    """Validated random-access reader for one X4T safetensors layer."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with safe_open(self.path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            if metadata is None or metadata.get("schema") != X4T_SAFETENSORS_SCHEMA:
                raise ValueError("X4T safetensors schema is unsupported")
            if metadata.get("version") != str(X4T_VERSION):
                raise ValueError("X4T safetensors version is unsupported")
            try:
                layer = int(metadata["layer"])
                experts = int(metadata["experts"])
            except (KeyError, ValueError) as exc:
                raise ValueError("X4T safetensors metadata is invalid") from exc
            if layer not in C.MOE_LAYERS or not 0 <= experts <= X4T_EXPERTS_PER_LAYER:
                raise ValueError("X4T safetensors layer or expert count is invalid")
            if (
                metadata.get("expert_capacity") != str(X4T_EXPERTS_PER_LAYER)
                or metadata.get("matrix_order") != ",".join(X4T_MATRIX_ORDER)
                or metadata.get("scale_codec") != "x4t-adjacent-pair-fixed-stream-v1"
                or metadata.get("w2_packed_layout") != "group-major-32-channel-v1"
                or metadata.get("exact_mxfp4_reconstruction") != "true"
            ):
                raise ValueError("X4T safetensors metadata is noncanonical")
            expected_keys = {"expert_ids"}
            for matrix in X4T_MATRIX_ORDER:
                expected_keys.update(
                    _tensor_name(matrix, part)
                    for part in (
                        "packed",
                        "scale_fixed",
                        "scale_exceptions",
                        "scale_exception_offsets",
                    )
                )
            if set(handle.keys()) != expected_keys:
                raise ValueError("X4T safetensors tensor inventory is noncanonical")
            expert_ids_tensor = handle.get_tensor("expert_ids")
            if expert_ids_tensor.dtype != torch.int32 or tuple(
                expert_ids_tensor.shape
            ) != (experts,):
                raise ValueError("X4T expert_ids tensor is invalid")
            expert_ids = tuple(map(int, expert_ids_tensor.tolist()))
            if expert_ids != tuple(sorted(expert_ids)) or len(set(expert_ids)) != experts:
                raise ValueError("X4T expert IDs must be unique and sorted")
            if any(not 0 <= expert < X4T_EXPERTS_PER_LAYER for expert in expert_ids):
                raise ValueError("X4T expert ID lies outside the layer capacity")
            self._offsets: dict[str, tuple[int, ...]] = {}
            for matrix in X4T_MATRIX_ORDER:
                offsets = handle.get_tensor(
                    _tensor_name(matrix, "scale_exception_offsets")
                )
                values = tuple(map(int, offsets.tolist()))
                if offsets.dtype != torch.int64 or tuple(offsets.shape) != (
                    experts + 1,
                ):
                    raise ValueError("X4T exception offsets tensor is invalid")
                if (
                    not values
                    or values[0] != 0
                    or any(a > b for a, b in zip(values, values[1:]))
                ):
                    raise ValueError("X4T exception offsets are noncanonical")
                exception_shape = tuple(
                    handle.get_slice(
                        _tensor_name(matrix, "scale_exceptions")
                    ).get_shape()
                )
                if exception_shape != (4 * values[-1],):
                    raise ValueError("X4T exception bytes disagree with their offsets")
                _, scale_shape = _matrix_shapes(matrix)
                if tuple(
                    handle.get_slice(_tensor_name(matrix, "packed")).get_shape()
                ) != _packed_storage_shape(matrix, experts):
                    raise ValueError("X4T packed tensor shape is invalid")
                if tuple(
                    handle.get_slice(_tensor_name(matrix, "scale_fixed")).get_shape()
                ) != (experts, _scale_fixed_bytes(scale_shape)):
                    raise ValueError("X4T fixed scale tensor shape is invalid")
                self._offsets[matrix] = values
        self.layer = layer
        self.file_bytes = self.path.stat().st_size
        self.expert_ids = expert_ids
        self._expert_to_slot = {expert: slot for slot, expert in enumerate(expert_ids)}
        self.matrix_count = experts * len(X4T_MATRIX_ORDER)

    def has(self, expert: int, matrix: str) -> bool:
        _validate_expert(expert)
        _matrix_id(matrix)
        return expert in self._expert_to_slot

    def matrix_payload_bytes(self, expert: int, matrix: str) -> int:
        """Return matrix tensor bytes attributable to one expert."""

        if not self.has(expert, matrix):
            raise KeyError((expert, matrix))
        slot = self._expert_to_slot[expert]
        packed_shape, scale_shape = _matrix_shapes(matrix)
        return (
            math.prod(packed_shape)
            + _scale_fixed_bytes(scale_shape)
            + 4 * (self._offsets[matrix][slot + 1] - self._offsets[matrix][slot])
        )

    def read(self, expert: int, matrix: str) -> X4TMatrix:
        if not self.has(expert, matrix):
            raise KeyError((expert, matrix))
        slot = self._expert_to_slot[expert]
        with safe_open(self.path, framework="pt", device="cpu") as handle:
            packed_storage = handle.get_slice(_tensor_name(matrix, "packed"))[
                slot
            ].contiguous()
            fixed = handle.get_slice(_tensor_name(matrix, "scale_fixed"))[
                slot
            ].contiguous()
            first = self._offsets[matrix][slot]
            end = self._offsets[matrix][slot + 1]
            exception_bytes = handle.get_slice(_tensor_name(matrix, "scale_exceptions"))[
                4 * first : 4 * end
            ].contiguous()
        exceptions = exception_bytes.view(torch.uint32)
        out_features, in_features = C.EXPERT_SHAPES[matrix]
        if matrix == "w2":
            packed = (
                packed_storage.permute(1, 0, 2)
                .reshape(out_features, in_features // 2)
                .contiguous()
            )
        else:
            packed = packed_storage
        scale = _decode_scale_components(
            fixed,
            exceptions,
            rows=out_features,
            columns=in_features // C.MXFP4_BLOCK,
        )
        return X4TMatrix(matrix=matrix, packed=packed, scale=scale)


def x4t_layer_path(root: str | Path, layer: int) -> Path:
    if layer not in C.MOE_LAYERS:
        raise ValueError("X4T sidecar layer must be a Kimi-K3 MoE layer")
    return Path(root) / f"x4t-layer-{layer:05d}.safetensors"
