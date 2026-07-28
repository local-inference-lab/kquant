"""Checkpoint inventory: categorize every tensor, measure bytes, emit ledger.

Works on a partial cache: tensors living in missing shards get shapes/dtypes
extrapolated from the per-pattern census of present shards (all patterns are
shape-uniform in this checkpoint); such rows are flagged ``extrapolated``.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from kquant import constants as C
from kquant.io.hf_cache import CheckpointCache, read_safetensors_header

_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "U8": 1, "I8": 1,
                "F8_E4M3": 1, "I32": 4, "I64": 8}

_CATEGORIES = [
    ("routed_experts", re.compile(r"\.block_sparse_moe\.experts\.\d+\.w[123]\.")),
    ("latent_proj", re.compile(r"\.block_sparse_moe\.routed_expert_(down|up)_proj\.|\.block_sparse_moe\.routed_expert_norm\.")),
    ("router", re.compile(r"\.block_sparse_moe\.gate\.")),
    ("shared_experts", re.compile(r"\.block_sparse_moe\.shared_experts\.")),
    ("attention", re.compile(r"\.self_attn\.")),
    ("attn_residual", re.compile(r"(self_attention_res_|mlp_res_|output_attn_res_)")),
    ("dense_mlp", re.compile(r"\.layers\.0\.mlp\.")),
    ("embeddings", re.compile(r"embed_tokens")),
    ("lm_head", re.compile(r"lm_head")),
    ("vision", re.compile(r"^(vision_tower|mm_projector)\.")),
    ("norms_other", re.compile(r"(layernorm|\.norm\.)")),
]


def categorize(name: str) -> str:
    for cat, rx in _CATEGORIES:
        if rx.search(name):
            return cat
    return "uncategorized"


def _pattern(name: str) -> str:
    return re.sub(r"(?<=\.)\d+(?=\.)", "N", name)


# Config-derived shapes for tensors whose pattern may have no downloaded
# example (they are unique singletons living in late shards).
_FALLBACK_SHAPES: dict[str, tuple[str, tuple[int, ...]]] = {
    "language_model.model.embed_tokens.weight": ("BF16", (163840, 7168)),
    "language_model.lm_head.weight": ("BF16", (163840, 7168)),
    "language_model.model.norm.weight": ("BF16", (7168,)),
    "language_model.model.output_attn_res_norm.weight": ("BF16", (7168,)),
    "language_model.model.output_attn_res_proj.weight": ("BF16", (1, 7168)),
}


def build_inventory(cache: CheckpointCache) -> dict:
    # dtype/shape census from present shard headers
    seen: dict[str, tuple[str, tuple[int, ...]]] = {}
    pattern_info: dict[str, tuple[str, tuple[int, ...]]] = {}
    for shard, path in cache.shard_paths.items():
        header = read_safetensors_header(path)
        for name, info in header.items():
            if name == "__metadata__":
                continue
            dt, shape = info["dtype"], tuple(info["shape"])
            seen[name] = (dt, shape)
            pattern_info.setdefault(_pattern(name), (dt, shape))

    cats: dict[str, dict] = defaultdict(
        lambda: {"tensors": 0, "bytes": 0, "params": 0, "extrapolated_tensors": 0}
    )
    unresolved: list[str] = []
    for name, shard in cache.weight_map.items():
        if name in seen:
            dt, shape = seen[name]
            extrapolated = False
        else:
            hit = pattern_info.get(_pattern(name)) or _FALLBACK_SHAPES.get(name)
            if hit is None:
                unresolved.append(name)
                continue
            dt, shape = hit
            extrapolated = True
        cat = categorize(name)
        n_elem = 1
        for d in shape:
            n_elem *= d
        nbytes = n_elem * _DTYPE_BYTES[dt]
        row = cats[cat]
        row["tensors"] += 1
        row["bytes"] += nbytes
        row["extrapolated_tensors"] += int(extrapolated)
        # logical params: packed U8 nibbles hold 2 params/byte; scales hold none
        if name.endswith("weight_scale"):
            pass
        elif name.endswith("weight_packed"):
            row["params"] += n_elem * 2
        else:
            row["params"] += n_elem

    complete_layers = cache.complete_moe_layers()
    latent = pattern_info.get(
        "language_model.model.layers.N.block_sparse_moe.routed_expert_down_proj.weight"
    )
    inv = {
        "model": C.MODEL_ID,
        "revision": C.REVISION,
        "total_size_index": cache.index["metadata"]["total_size"],
        "shards_present": len(cache.shard_paths),
        "shards_missing": len(cache.missing_shards),
        "complete_moe_layers": complete_layers,
        "num_complete_moe_layers": len(complete_layers),
        "latent_proj_dtype": latent[0] if latent else None,
        "categories": dict(sorted(cats.items())),
        "unresolved_tensors": unresolved,
    }
    total_bytes = sum(r["bytes"] for r in cats.values())
    inv["total_bytes_accounted"] = total_bytes
    inv["accounting_gap_bytes"] = cache.index["metadata"]["total_size"] - total_bytes
    return inv


def write_inventory(inv: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "inventory.json"
    p.write_text(json.dumps(inv, indent=1))
    return p


def format_summary(inv: dict) -> str:
    lines = [
        f"model {inv['model']} @ {inv['revision'][:8]}",
        f"shards {inv['shards_present']}/{inv['shards_present'] + inv['shards_missing']}"
        f" | complete MoE layers {inv['num_complete_moe_layers']}/{C.NUM_MOE_LAYERS}",
        f"latent projections dtype: {inv['latent_proj_dtype']}",
        f"{'category':16s} {'tensors':>8s} {'params':>16s} {'GiB':>9s} {'extrap':>7s}",
    ]
    for cat, r in inv["categories"].items():
        lines.append(
            f"{cat:16s} {r['tensors']:8d} {r['params']:16,d} "
            f"{r['bytes'] / 2**30:9.2f} {r['extrapolated_tensors']:7d}"
        )
    lines.append(
        f"accounted {inv['total_bytes_accounted'] / 2**40:.4f} TiB"
        f" vs index {inv['total_size_index'] / 2**40:.4f} TiB"
        f" (gap {inv['accounting_gap_bytes']} B)"
    )
    return "\n".join(lines)
