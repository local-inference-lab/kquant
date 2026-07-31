"""Utilities for streaming Kimi-K3 through the official PyTorch layers.

The full Kimi-K3 checkpoint is too large to materialize as an ordinary
``transformers`` model.  A fixed prefill, however, only carries two tensors
between decoder layers:

* the layer-boundary hidden state; and
* Kimi's attention-residual block accumulator.

This module provides the checkpoint and state-file primitives used by
``scripts/stream_k3_pytorch.py``.  It intentionally has no dependency on
``transformers`` or ``compressed_tensors`` so its file/state contracts can be
unit-tested in the lightweight kquant environment.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

TRACE_SCHEMA_VERSION = 1
STREAM_STATE_SCHEMA_VERSION = 1
MODEL_TENSOR_PREFIX = "language_model.model"


def tensor_digest(tensor: torch.Tensor) -> str:
    """Return a stable SHA-256 digest of a contiguous CPU tensor."""
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


class IndexedSafetensors:
    """Read individual tensors and row ranges from an indexed checkpoint."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir.expanduser().resolve()
        index_path = self.model_dir / "model.safetensors.index.json"
        try:
            document = json.loads(index_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read checkpoint index {index_path}: {exc}") from exc
        weight_map = document.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"{index_path}: missing or empty weight_map")
        self.weight_map: dict[str, str] = {
            str(name): str(filename) for name, filename in weight_map.items()
        }

    def path_for(self, name: str) -> Path:
        filename = self.weight_map.get(name)
        if filename is None:
            raise KeyError(name)
        path = self.model_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def get(self, name: str) -> torch.Tensor:
        """Load one complete tensor onto CPU."""
        with safe_open(str(self.path_for(name)), framework="pt") as handle:
            return handle.get_tensor(name)

    def get_many(
        self,
        names: list[str] | tuple[str, ...],
        *,
        clone: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Load tensors while opening each backing shard only once.

        ``clone=True`` eagerly faults the mapped tensor pages into ordinary
        CPU storage.  This is useful for background prefetch: returning a
        lazily mapped tensor would defer the actual NVMe read until the later
        host-to-device copy and defeat the overlap.
        """
        requested = list(dict.fromkeys(str(name) for name in names))
        grouped: dict[str, list[str]] = defaultdict(list)
        for name in requested:
            filename = self.weight_map.get(name)
            if filename is None:
                raise KeyError(name)
            grouped[filename].append(name)

        loaded: dict[str, torch.Tensor] = {}
        for filename, shard_names in grouped.items():
            path = self.model_dir / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            with safe_open(str(path), framework="pt") as handle:
                for name in shard_names:
                    value = handle.get_tensor(name)
                    loaded[name] = value.clone() if clone else value
        return {name: loaded[name] for name in requested}

    def get_range(self, name: str, start: int, end: int) -> torch.Tensor:
        """Load ``tensor[start:end]`` without reading the other rows."""
        if start < 0 or end < start:
            raise ValueError(f"invalid row range [{start}, {end})")
        with safe_open(str(self.path_for(name)), framework="pt") as handle:
            value = handle.get_slice(name)
            shape = value.get_shape()
            if not shape:
                raise ValueError(f"{name} is scalar; row slicing is not defined")
            if end > shape[0]:
                raise IndexError(
                    f"{name}: row range [{start}, {end}) exceeds {shape[0]}"
                )
            return value[start:end]

    def get_rows(self, name: str, rows: torch.Tensor | list[int]) -> torch.Tensor:
        """Load arbitrary rows while reading each contiguous range only once."""
        requested = [int(value) for value in torch.as_tensor(rows).view(-1).tolist()]
        if not requested:
            raise ValueError("at least one row is required")
        unique = sorted(set(requested))
        if unique[0] < 0:
            raise IndexError(f"{name}: negative row {unique[0]}")

        ranges: list[tuple[int, int]] = []
        start = previous = unique[0]
        for row in unique[1:]:
            if row == previous + 1:
                previous = row
                continue
            ranges.append((start, previous + 1))
            start = previous = row
        ranges.append((start, previous + 1))

        loaded: dict[int, torch.Tensor] = {}
        for start, end in ranges:
            block = self.get_range(name, start, end)
            loaded.update(
                {row: block[offset] for offset, row in enumerate(range(start, end))}
            )
        return torch.stack([loaded[row] for row in requested])


def assign_parameter(
    module: torch.nn.Module,
    name: str,
    tensor: torch.Tensor,
    *,
    requires_grad: bool = False,
) -> None:
    """Replace a possibly-meta parameter addressed by its dotted name."""
    parent_name, _, leaf = name.rpartition(".")
    parent = module.get_submodule(parent_name) if parent_name else module
    if leaf not in parent._parameters:
        raise KeyError(f"{name} is not a direct parameter")
    parent._parameters[leaf] = torch.nn.Parameter(
        tensor, requires_grad=requires_grad
    )


def fit_checkpoint_parameter(
    name: str,
    value: torch.Tensor,
    expected_shape: tuple[int, ...],
) -> tuple[torch.Tensor, str | None]:
    """Apply narrowly scoped compatibility fixes for official checkpoint tensors.

    Kimi-K3 serializes KDA ``A_log`` as 128 values even though the configured
    model has 96 heads.  The final 32 values are zero padding.  No other
    implicit narrowing or reshaping is accepted.
    """
    if tuple(value.shape) == expected_shape:
        return value, None
    if (
        name.endswith("self_attn.A_log")
        and value.ndim == 1
        and len(expected_shape) == 1
        and value.shape[0] > expected_shape[0]
    ):
        tail = value[expected_shape[0] :]
        if torch.count_nonzero(tail).item() != 0:
            raise ValueError(
                f"{name}: refusing to discard nonzero padded A_log values"
            )
        return (
            value[: expected_shape[0]].contiguous(),
            f"narrowed padded A_log {tuple(value.shape)} -> {expected_shape}",
        )
    raise ValueError(
        f"{name}: checkpoint shape {tuple(value.shape)} != module shape "
        f"{expected_shape}"
    )


@dataclass
class StreamState:
    """The complete inter-layer state for a fixed, cache-free prefill."""

    next_layer: int
    input_ids: torch.Tensor
    hidden_states: torch.Tensor
    block_residual: torch.Tensor

    def validate(self) -> None:
        if self.next_layer < 0:
            raise ValueError("next_layer must be non-negative")
        if self.input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if self.hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape [batch, sequence, hidden]"
            )
        if self.block_residual.ndim != 3:
            raise ValueError(
                "block_residual must have shape [batch*sequence, blocks, hidden]"
            )
        batch, sequence, hidden = self.hidden_states.shape
        if tuple(self.input_ids.shape) != (batch, sequence):
            raise ValueError("input_ids and hidden_states batch/sequence differ")
        if tuple(self.block_residual.shape[::2]) != (batch * sequence, hidden):
            raise ValueError(
                "block_residual token/hidden dimensions do not match hidden_states"
            )

    def cpu(self) -> "StreamState":
        self.validate()
        return StreamState(
            next_layer=self.next_layer,
            input_ids=self.input_ids.detach().contiguous().cpu(),
            hidden_states=self.hidden_states.detach().contiguous().cpu(),
            block_residual=self.block_residual.detach().contiguous().cpu(),
        )


def atomic_torch_save(value: Any, path: Path) -> None:
    """Save with an atomic rename so an interrupted layer cannot corrupt state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def save_stream_state(
    path: Path,
    state: StreamState,
    *,
    checkpoint: str,
    prompt: str | None,
) -> None:
    value = state.cpu()
    atomic_torch_save(
        {
            "schema_version": STREAM_STATE_SCHEMA_VERSION,
            "checkpoint": checkpoint,
            "prompt": prompt,
            "next_layer": value.next_layer,
            "input_ids": value.input_ids,
            "hidden_states": value.hidden_states,
            "block_residual": value.block_residual,
        },
        path,
    )


def load_stream_state(path: Path) -> tuple[StreamState, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: stream state must be a dictionary")
    if payload.get("schema_version") != STREAM_STATE_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: unsupported schema_version "
            f"{payload.get('schema_version')!r}"
        )
    required = ("input_ids", "hidden_states", "block_residual")
    if any(not isinstance(payload.get(name), torch.Tensor) for name in required):
        raise ValueError(f"{path}: stream state is missing required tensors")
    state = StreamState(
        next_layer=int(payload["next_layer"]),
        input_ids=payload["input_ids"],
        hidden_states=payload["hidden_states"],
        block_residual=payload["block_residual"],
    )
    state.validate()
    metadata = {
        "checkpoint": payload.get("checkpoint"),
        "prompt": payload.get("prompt"),
    }
    return state, metadata


def write_trace_tensor(
    trace_root: Path,
    *,
    layer: int,
    stage: str,
    tensor: torch.Tensor,
    call: int = 0,
) -> Path:
    """Write a TP1 tensor in the same schema as vLLM's Kimi trace."""
    saved = tensor.detach().contiguous().cpu()
    metadata = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "layer": layer,
        "stage": stage,
        "call": call,
        "tp_rank": 0,
        "tp_world_size": 1,
        "original_shape": list(tensor.shape),
        "saved_shape": list(saved.shape),
        "dtype": str(saved.dtype),
        "source_device": str(tensor.device),
        "sha256": tensor_digest(saved),
    }
    rank_dir = trace_root / "tp-rank-000"
    destination = (
        rank_dir / f"layer-{layer:03d}.call-{call:06d}.{stage}.pt"
    )
    atomic_torch_save({"metadata": metadata, "tensor": saved}, destination)
    return destination


def write_trace_manifest(
    trace_root: Path,
    *,
    layers: list[int],
    max_tokens: int,
    checkpoint: str,
) -> None:
    rank_dir = trace_root / "tp-rank-000"
    rank_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "pid": os.getpid(),
        "tp_rank": 0,
        "tp_world_size": 1,
        "layers": layers,
        "ranks": [0],
        "start_call": 0,
        "max_calls": 1,
        "max_tokens": max_tokens,
        "source": "official_pytorch_stream",
        "checkpoint": checkpoint,
    }
    path = rank_dir / "manifest.json"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
