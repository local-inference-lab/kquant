"""Synthetic mini-checkpoint fixture in the real cache layout.

Small geometry (2 MoE layers x 4 experts, latent 64, inter 96) but the real
tensor-name schema, compressed-tensors packing, and index format, so io /
inventory / stats / pack code paths run unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import kquant.constants as C
from kquant.io import mxfp4

MINI = dict(layers=(1, 2), experts=4, latent=64, inter=96)


def _rand_expert(rng: torch.Generator, out: int, k: int):
    codes = torch.randint(0, 16, (out, k), dtype=torch.uint8, generator=rng)
    packed = mxfp4.pack_codes(codes)
    scale = torch.randint(105, 130, (out, k // 32), dtype=torch.uint8,
                          generator=rng)
    return packed, scale


@pytest.fixture(scope="session")
def mini_ckpt(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("mini") / "repo"
    snap = root / "snapshots" / C.REVISION
    snap.mkdir(parents=True)
    rng = torch.Generator().manual_seed(1234)

    weight_map: dict[str, str] = {}
    shard_tensors: dict[str, dict[str, torch.Tensor]] = {}

    def put(shard: str, name: str, t: torch.Tensor):
        weight_map[name] = shard
        shard_tensors.setdefault(shard, {})[name] = t

    L, E = MINI["layers"], MINI["experts"]
    lat, inter = MINI["latent"], MINI["inter"]
    for i, layer in enumerate(L):
        shard = f"model-{i + 1:05d}-of-{len(L):06d}.safetensors"
        for e in range(E):
            for m in C.EXPERT_MATRICES:
                out, k = (inter, lat) if m in ("w1", "w3") else (lat, inter)
                packed, scale = _rand_expert(rng, out, k)
                put(shard, C.expert_tensor(layer, e, m, "weight_packed"), packed)
                put(shard, C.expert_tensor(layer, e, m, "weight_scale"), scale)
        put(shard, C.router_bias_tensor(layer),
            torch.randn(E, generator=rng).float())
        put(shard, C.router_weight_tensor(layer),
            torch.randn(E, 32, generator=rng).bfloat16())
        put(shard, C.latent_up_proj_tensor(layer),
            torch.randn(48, lat, generator=rng).bfloat16())

    for shard, tensors in shard_tensors.items():
        save_file(tensors, str(snap / shard))
    total = sum(
        t.numel() * t.element_size()
        for ts in shard_tensors.values() for t in ts.values()
    )
    (snap / C.INDEX_FILE).write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map})
    )
    return root


@pytest.fixture()
def mini_cache(mini_ckpt, monkeypatch):
    """CheckpointCache over the mini checkpoint, with geometry monkeypatched
    so [92, 896] accumulators still index correctly (mini layers/experts are a
    subset of the real ranges)."""
    from kquant.io.hf_cache import resolve

    return resolve(mini_ckpt, C.REVISION)
