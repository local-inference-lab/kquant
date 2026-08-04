"""Persistent rank-major TP12 checkpoint shards for the X4T tier.

The canonical X4T sidecar is independent of tensor parallelism.  Its full
matrix row-major ordering is appropriate for archival/random expert access,
but it makes every TP12 process traverse the same full ``w2`` nibble planes at
startup.  This module losslessly compiles that artifact into one persistent
checkpoint shard per layer/rank.  The stored values remain X4T: packed E2M1
nibbles, fixed adjacent-pair scale streams, and sparse exception words.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from kquant import constants as C
from kquant.pack.package_helpers import _atomic_json, _fsync_directory
from kquant.qsrt import TP_SIZE
from kquant.x4t import (
    X4T_POSITION_BITS,
    X4T_POSITION_MASK,
    X4T_TILE_ROWS,
    X4TLayerReader,
    pack_x4t_scale_components,
    x4t_layer_path,
)


X4T_TP12_FORMAT = "kquant-x4t-tp12"
X4T_TP12_VERSION = 1
X4T_TP12_MANIFEST = "x4t-tp12-manifest.json"
X4T_TP12_MANIFEST_KIND = "kquant_kimi_k3_qsrt_x4t_tp12_checkpoint"
X4T_TP12_MANIFEST_SCHEMA_VERSION = 1
X4T_TP12_KEYS = frozenset(
    {
        "expert_ids",
        "w13_packed",
        "w2_packed",
        "w13_fixed",
        "w13_exception_offsets",
        "w13_exceptions",
        "w2_fixed",
        "w2_exception_offsets",
        "w2_exceptions",
    }
)

_LOCAL_INTERMEDIATE = C.EXPERT_INTER // TP_SIZE
_W13_PACKED_SHAPE = (2 * _LOCAL_INTERMEDIATE, C.LATENT // 2)
_W2_PACKED_SHAPE = (C.LATENT, _LOCAL_INTERMEDIATE // 2)
_W13_SCALE_SHAPE = (2 * _LOCAL_INTERMEDIATE, C.LATENT // C.MXFP4_BLOCK)
_W2_SCALE_SHAPE = (C.LATENT, _LOCAL_INTERMEDIATE // C.MXFP4_BLOCK)


def x4t_tp12_rank_filename(layer: int, rank: int) -> str:
    if layer not in C.MOE_LAYERS:
        raise ValueError("X4T TP12 layer must lie in 1..92")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < TP_SIZE:
        raise ValueError("X4T TP12 rank must lie in 0..11")
    return f"x4t-tp12-layer-{layer:05d}-rank-{rank:02d}.safetensors"


def x4t_tp12_rank_filenames() -> tuple[str, ...]:
    return tuple(
        x4t_tp12_rank_filename(layer, rank)
        for layer in C.MOE_LAYERS
        for rank in range(TP_SIZE)
    )


@dataclass(frozen=True)
class X4TScaleComponents:
    fixed: torch.Tensor
    exceptions: torch.Tensor
    rows: int
    columns: int

    def concatenate_rows(self, other: "X4TScaleComponents") -> "X4TScaleComponents":
        if self.columns != other.columns:
            raise ValueError("X4T row concatenation requires equal columns")
        if other.exceptions.numel():
            words = other.exceptions.to(torch.int64)
            positions = words & X4T_POSITION_MASK
            values = words >> X4T_POSITION_BITS
            shifted = (
                (values << X4T_POSITION_BITS)
                | (positions + self.rows * self.columns)
            ).to(torch.uint32)
        else:
            shifted = torch.empty((0,), dtype=torch.uint32)
        return X4TScaleComponents(
            fixed=torch.cat((self.fixed, other.fixed)).contiguous(),
            exceptions=torch.cat((self.exceptions, shifted)).contiguous(),
            rows=self.rows + other.rows,
            columns=self.columns,
        )


def _rank_scale_components(
    matrix: str,
    fixed: torch.Tensor,
    exceptions: torch.Tensor,
    rank: int,
) -> X4TScaleComponents:
    rows, source_columns = (
        C.EXPERT_SHAPES[matrix][0],
        C.EXPERT_SHAPES[matrix][1] // C.MXFP4_BLOCK,
    )
    selector_bytes = (source_columns + 7) // 8
    tile_bytes = X4T_TILE_ROWS * (1 + selector_bytes)
    fixed_2d = fixed.reshape(rows // X4T_TILE_ROWS, tile_bytes)
    words = exceptions.to(torch.int64)
    positions = words & X4T_POSITION_MASK
    values = words >> X4T_POSITION_BITS

    if matrix in ("w1", "w3"):
        rank_rows = rows // TP_SIZE
        row_begin = rank * rank_rows
        tile_begin = row_begin // X4T_TILE_ROWS
        tile_end = tile_begin + rank_rows // X4T_TILE_ROWS
        rank_fixed = fixed_2d[tile_begin:tile_end].reshape(-1).contiguous()
        source_rows = positions // source_columns
        keep = (source_rows >= row_begin) & (source_rows < row_begin + rank_rows)
        local_positions = positions[keep] - row_begin * source_columns
        rank_exceptions = (
            (values[keep] << X4T_POSITION_BITS) | local_positions
        ).to(torch.uint32)
        return X4TScaleComponents(
            fixed=rank_fixed,
            exceptions=rank_exceptions.contiguous(),
            rows=rank_rows,
            columns=source_columns,
        )

    rank_columns = source_columns // TP_SIZE
    column_begin = rank * rank_columns
    selectors = fixed_2d[:, X4T_TILE_ROWS:].reshape(
        rows // X4T_TILE_ROWS,
        X4T_TILE_ROWS,
        selector_bytes,
    )
    rank_fixed_2d = torch.empty(
        (rows // X4T_TILE_ROWS, 2 * X4T_TILE_ROWS),
        dtype=torch.uint8,
    )
    rank_fixed_2d[:, :X4T_TILE_ROWS].copy_(fixed_2d[:, :X4T_TILE_ROWS])
    rank_fixed_2d[:, X4T_TILE_ROWS:].copy_(selectors[:, :, rank])
    source_rows = positions // source_columns
    source_cols = positions % source_columns
    keep = (source_cols >= column_begin) & (
        source_cols < column_begin + rank_columns
    )
    local_positions = (
        source_rows[keep] * rank_columns + source_cols[keep] - column_begin
    )
    rank_exceptions = (
        (values[keep] << X4T_POSITION_BITS) | local_positions
    ).to(torch.uint32)
    return X4TScaleComponents(
        fixed=rank_fixed_2d.reshape(-1).contiguous(),
        exceptions=rank_exceptions.contiguous(),
        rows=rows,
        columns=rank_columns,
    )


def _fixed_bytes(rows: int, columns: int) -> int:
    return (rows // X4T_TILE_ROWS) * X4T_TILE_ROWS * (1 + (columns + 7) // 8)


@dataclass
class _RankBuffers:
    experts: int
    w13_packed: torch.Tensor = field(init=False)
    w2_packed: torch.Tensor = field(init=False)
    w13_fixed: torch.Tensor = field(init=False)
    w2_fixed: torch.Tensor = field(init=False)
    w13_exception_parts: list[torch.Tensor] = field(default_factory=list)
    w2_exception_parts: list[torch.Tensor] = field(default_factory=list)
    w13_exception_offsets: list[int] = field(default_factory=lambda: [0])
    w2_exception_offsets: list[int] = field(default_factory=lambda: [0])

    def __post_init__(self) -> None:
        self.w13_packed = torch.empty(
            (self.experts, *_W13_PACKED_SHAPE), dtype=torch.uint8
        )
        self.w2_packed = torch.empty(
            (self.experts, *_W2_PACKED_SHAPE), dtype=torch.uint8
        )
        self.w13_fixed = torch.empty(
            (self.experts, _fixed_bytes(*_W13_SCALE_SHAPE)), dtype=torch.uint8
        )
        self.w2_fixed = torch.empty(
            (self.experts, _fixed_bytes(*_W2_SCALE_SHAPE)), dtype=torch.uint8
        )

    def add_scales(
        self,
        slot: int,
        w13: X4TScaleComponents,
        w2: X4TScaleComponents,
    ) -> None:
        self.w13_fixed[slot].copy_(w13.fixed)
        self.w2_fixed[slot].copy_(w2.fixed)
        self.w13_exception_parts.append(w13.exceptions)
        self.w2_exception_parts.append(w2.exceptions)
        self.w13_exception_offsets.append(
            self.w13_exception_offsets[-1] + w13.exceptions.numel()
        )
        self.w2_exception_offsets.append(
            self.w2_exception_offsets[-1] + w2.exceptions.numel()
        )

    def tensors(self, expert_ids: tuple[int, ...]) -> dict[str, torch.Tensor]:
        empty = torch.empty((0,), dtype=torch.uint32)
        return {
            "expert_ids": torch.tensor(expert_ids, dtype=torch.int32),
            "w13_packed": self.w13_packed,
            "w2_packed": self.w2_packed,
            "w13_fixed": self.w13_fixed,
            "w13_exception_offsets": torch.tensor(
                self.w13_exception_offsets, dtype=torch.int64
            ),
            "w13_exceptions": (
                torch.cat(self.w13_exception_parts).contiguous()
                if self.w13_exception_parts
                else empty
            ),
            "w2_fixed": self.w2_fixed,
            "w2_exception_offsets": torch.tensor(
                self.w2_exception_offsets, dtype=torch.int64
            ),
            "w2_exceptions": (
                torch.cat(self.w2_exception_parts).contiguous()
                if self.w2_exception_parts
                else empty.clone()
            ),
        }


def _validate_slice(
    handle,
    name: str,
    *,
    dtype: str,
    shape: tuple[int, ...],
) -> None:
    value = handle.get_slice(name)
    if value.get_dtype() != dtype or tuple(value.get_shape()) != shape:
        raise ValueError(
            f"X4T TP12 tensor {name} must be {dtype} {shape}, got "
            f"{value.get_dtype()} {tuple(value.get_shape())}"
        )


def validate_x4t_tp12_rank_shard(
    path: str | Path,
    *,
    layer: int,
    rank: int,
    expected_expert_ids: tuple[int, ...],
) -> dict[str, int | str]:
    path = Path(path)
    experts = len(expected_expert_ids)
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        expected_metadata = {
            "format": X4T_TP12_FORMAT,
            "version": str(X4T_TP12_VERSION),
            "layer": str(layer),
            "rank": str(rank),
        }
        for name, expected in expected_metadata.items():
            if metadata.get(name) != expected:
                raise ValueError(f"X4T TP12 metadata {name} mismatch")
        if set(handle.keys()) != X4T_TP12_KEYS:
            raise ValueError("X4T TP12 tensor inventory is noncanonical")
        _validate_slice(handle, "expert_ids", dtype="I32", shape=(experts,))
        _validate_slice(
            handle,
            "w13_packed",
            dtype="U8",
            shape=(experts, *_W13_PACKED_SHAPE),
        )
        _validate_slice(
            handle,
            "w2_packed",
            dtype="U8",
            shape=(experts, *_W2_PACKED_SHAPE),
        )
        _validate_slice(
            handle,
            "w13_fixed",
            dtype="U8",
            shape=(experts, _fixed_bytes(*_W13_SCALE_SHAPE)),
        )
        _validate_slice(
            handle,
            "w2_fixed",
            dtype="U8",
            shape=(experts, _fixed_bytes(*_W2_SCALE_SHAPE)),
        )
        _validate_slice(
            handle,
            "w13_exception_offsets",
            dtype="I64",
            shape=(experts + 1,),
        )
        _validate_slice(
            handle,
            "w2_exception_offsets",
            dtype="I64",
            shape=(experts + 1,),
        )
        expert_ids = handle.get_tensor("expert_ids")
        if tuple(expert_ids.tolist()) != expected_expert_ids:
            raise ValueError("X4T TP12 expert order disagrees with the allocation")
        for family in ("w13", "w2"):
            offsets = handle.get_tensor(f"{family}_exception_offsets")
            exception_shape = tuple(
                handle.get_slice(f"{family}_exceptions").get_shape()
            )
            if len(exception_shape) != 1:
                raise ValueError(f"X4T TP12 {family} exceptions must be a vector")
            _validate_slice(
                handle,
                f"{family}_exceptions",
                dtype="U32",
                shape=exception_shape,
            )
            if (
                int(offsets[0]) != 0
                or int(offsets[-1]) != exception_shape[0]
                or bool((offsets[1:] < offsets[:-1]).any())
            ):
                raise ValueError(f"X4T TP12 {family} exception offsets are invalid")
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "layer": layer,
        "rank": rank,
        "experts": experts,
    }


def build_x4t_tp12_layer(
    artifact: str | Path,
    destination: str | Path,
    *,
    layer: int,
    expert_ids: tuple[int, ...],
    resume: bool = False,
) -> list[dict[str, int | str]]:
    """Compile one canonical X4T layer into twelve persistent rank shards."""

    artifact = Path(artifact)
    destination = Path(destination)
    if tuple(sorted(set(expert_ids))) != expert_ids:
        raise ValueError("X4T TP12 expert IDs must be sorted and unique")
    reader = X4TLayerReader(
        x4t_layer_path(artifact, layer),
        verify_payloads=False,
    )
    if reader.layer != layer:
        raise ValueError("X4T sidecar layer ID mismatch")
    expert_set = set(expert_ids)
    for expert in range(C.NUM_EXPERTS):
        expected = expert in expert_set
        for matrix in C.EXPERT_MATRICES:
            if reader.has(expert, matrix) != expected:
                raise ValueError(
                    f"X4T inventory mismatch at layer {layer} expert {expert} {matrix}"
                )

    existing: list[dict[str, int | str] | None] = []
    for rank in range(TP_SIZE):
        path = destination / x4t_tp12_rank_filename(layer, rank)
        partial = path.with_name(f".{path.name}.partial")
        if partial.exists():
            if not resume:
                raise FileExistsError(partial)
            partial.unlink()
        if path.exists():
            if not resume:
                raise FileExistsError(path)
            existing.append(
                validate_x4t_tp12_rank_shard(
                    path,
                    layer=layer,
                    rank=rank,
                    expected_expert_ids=expert_ids,
                )
            )
        else:
            existing.append(None)
    if all(item is not None for item in existing):
        return [item for item in existing if item is not None]

    buffers = [_RankBuffers(len(expert_ids)) for _ in range(TP_SIZE)]
    for slot, expert in enumerate(expert_ids):
        w1 = reader.read(expert, "w1")
        w3 = reader.read(expert, "w3")
        w2 = reader.read(expert, "w2")
        w1_fixed, w1_exceptions = pack_x4t_scale_components(w1.scale)
        w3_fixed, w3_exceptions = pack_x4t_scale_components(w3.scale)
        w2_fixed, w2_exceptions = pack_x4t_scale_components(w2.scale)
        for rank, target in enumerate(buffers):
            row_begin = rank * _LOCAL_INTERMEDIATE
            row_end = row_begin + _LOCAL_INTERMEDIATE
            packed_begin = rank * (_LOCAL_INTERMEDIATE // 2)
            packed_end = packed_begin + _LOCAL_INTERMEDIATE // 2
            target.w13_packed[slot, :_LOCAL_INTERMEDIATE].copy_(
                w1.packed[row_begin:row_end]
            )
            target.w13_packed[slot, _LOCAL_INTERMEDIATE:].copy_(
                w3.packed[row_begin:row_end]
            )
            target.w2_packed[slot].copy_(w2.packed[:, packed_begin:packed_end])
            w13_scale = _rank_scale_components(
                "w1", w1_fixed, w1_exceptions, rank
            ).concatenate_rows(
                _rank_scale_components("w3", w3_fixed, w3_exceptions, rank)
            )
            w2_scale = _rank_scale_components(
                "w2", w2_fixed, w2_exceptions, rank
            )
            target.add_scales(slot, w13_scale, w2_scale)

    results: list[dict[str, int | str]] = []
    for rank, target in enumerate(buffers):
        path = destination / x4t_tp12_rank_filename(layer, rank)
        if existing[rank] is not None:
            results.append(existing[rank])
            continue
        partial = path.with_name(f".{path.name}.partial")
        save_file(
            target.tensors(expert_ids),
            str(partial),
            metadata={
                "format": X4T_TP12_FORMAT,
                "version": str(X4T_TP12_VERSION),
                "layer": str(layer),
                "rank": str(rank),
            },
        )
        descriptor = os.open(partial, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        partial.replace(path)
        results.append(
            validate_x4t_tp12_rank_shard(
                path,
                layer=layer,
                rank=rank,
                expected_expert_ids=expert_ids,
            )
        )
    _fsync_directory(destination)
    return results


def load_x4t_tp12_manifest(root: str | Path) -> dict:
    root = Path(root).resolve()
    path = root / X4T_TP12_MANIFEST
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    expected = {
        "kind": X4T_TP12_MANIFEST_KIND,
        "schema_version": X4T_TP12_MANIFEST_SCHEMA_VERSION,
        "format": X4T_TP12_FORMAT,
        "format_version": X4T_TP12_VERSION,
        "tp_size": TP_SIZE,
        "complete": True,
    }
    for name, expected_value in expected.items():
        if value.get(name) != expected_value:
            raise ValueError(f"X4T TP12 manifest {name} mismatch")
    files = value.get("files")
    if files != list(x4t_tp12_rank_filenames()):
        raise ValueError("X4T TP12 manifest file inventory is noncanonical")
    layers = value.get("layers")
    if not isinstance(layers, dict) or set(layers) != {
        str(layer) for layer in C.MOE_LAYERS
    }:
        raise ValueError("X4T TP12 manifest layer inventory is incomplete")
    for name in files:
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    return value


def write_x4t_tp12_manifest(
    destination: str | Path,
    *,
    artifact: str | Path,
    layers: dict[str, dict],
) -> dict:
    destination = Path(destination).resolve()
    if set(layers) != {str(layer) for layer in C.MOE_LAYERS}:
        raise ValueError("cannot finalize an incomplete X4T TP12 checkpoint")
    files = list(x4t_tp12_rank_filenames())
    document = {
        "kind": X4T_TP12_MANIFEST_KIND,
        "schema_version": X4T_TP12_MANIFEST_SCHEMA_VERSION,
        "format": X4T_TP12_FORMAT,
        "format_version": X4T_TP12_VERSION,
        "tp_size": TP_SIZE,
        "source_artifact": str(Path(artifact).resolve()),
        "layers": layers,
        "files": files,
        "total_bytes": sum((destination / name).stat().st_size for name in files),
        "complete": True,
    }
    _atomic_json(destination / X4T_TP12_MANIFEST, document)
    load_x4t_tp12_manifest(destination)
    return document


__all__ = [
    "X4T_TP12_FORMAT",
    "X4T_TP12_MANIFEST",
    "X4T_TP12_MANIFEST_KIND",
    "X4T_TP12_MANIFEST_SCHEMA_VERSION",
    "X4T_TP12_VERSION",
    "build_x4t_tp12_layer",
    "load_x4t_tp12_manifest",
    "validate_x4t_tp12_rank_shard",
    "write_x4t_tp12_manifest",
    "x4t_tp12_rank_filename",
    "x4t_tp12_rank_filenames",
]
