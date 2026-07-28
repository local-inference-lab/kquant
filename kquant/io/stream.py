"""Shard-centric streaming over the (partial) checkpoint.

Shards are visited once each, sequentially; expert tensors are parsed from
tensor names and yielded in batches stacked per matrix type so downstream
engines can run vectorized (optionally on GPU). Non-expert tensors are
loadable by name on demand.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

import torch
from safetensors import safe_open

from kquant import constants as C
from kquant.io.hf_cache import CheckpointCache

_EXPERT_RE = re.compile(
    r"^language_model\.model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)"
    r"\.(w1|w2|w3)\.weight_packed$"
)


@dataclass
class ExpertBatch:
    matrix: str  # w1 | w3 | w2
    layers: torch.Tensor  # int32 [B]
    experts: torch.Tensor  # int32 [B]
    packed: torch.Tensor  # uint8 [B, out, K/2]
    scale: torch.Tensor  # uint8 [B, out, K/32]


def load_tensor(cache: CheckpointCache, name: str) -> torch.Tensor:
    path = cache.tensor_path(name)
    if path is None:
        raise FileNotFoundError(f"shard for {name} not downloaded yet")
    with safe_open(str(path), framework="pt") as f:
        return f.get_tensor(name)


def iter_expert_batches(
    cache: CheckpointCache,
    batch_size: int = 128,
    layers: set[int] | None = None,
    matrices: tuple[str, ...] = C.EXPERT_MATRICES,
    shards: list[str] | None = None,
) -> Iterator[ExpertBatch]:
    """Yield stacked expert (packed, scale) batches, one matrix type per batch.

    Iterates shards in order; batches never mix matrix types (shapes are
    uniform per type). Scale tensors always travel with their packed tensor.
    """
    shard_list = shards if shards is not None else sorted(cache.shard_paths)
    pending: dict[str, list[tuple[int, int, torch.Tensor, torch.Tensor]]] = {
        m: [] for m in matrices
    }

    def flush(matrix: str) -> ExpertBatch:
        items = pending[matrix]
        pending[matrix] = []
        return ExpertBatch(
            matrix=matrix,
            layers=torch.tensor([i[0] for i in items], dtype=torch.int32),
            experts=torch.tensor([i[1] for i in items], dtype=torch.int32),
            packed=torch.stack([i[2] for i in items]),
            scale=torch.stack([i[3] for i in items]),
        )

    for shard in shard_list:
        path = cache.shard_paths.get(shard)
        if path is None:
            continue
        with safe_open(str(path), framework="pt") as f:
            keys = list(f.keys())
            for name in keys:
                m = _EXPERT_RE.match(name)
                if not m:
                    continue
                layer, expert, matrix = int(m[1]), int(m[2]), m[3]
                if matrix not in matrices:
                    continue
                if layers is not None and layer not in layers:
                    continue
                scale_name = name.replace("weight_packed", "weight_scale")
                packed = f.get_tensor(name)
                # scale lives in the same shard in practice; fall back if not
                if scale_name in keys:
                    scale = f.get_tensor(scale_name)
                else:
                    scale = load_tensor(cache, scale_name)
                pending[matrix].append((layer, expert, packed, scale))
                if len(pending[matrix]) >= batch_size:
                    yield flush(matrix)
    for matrix in matrices:
        if pending[matrix]:
            yield flush(matrix)
