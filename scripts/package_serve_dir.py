"""Package a kquant NF3 artifact + source checkpoint into a vLLM-servable dir.

Builds <dest>/ with:
  - nonexpert-*.safetensors: all non-expert tensors extracted from the source
    shards (original dtypes), so no source file leaks its MXFP4 expert tensors
    into the load (vLLM yields every tensor of every indexed file).
  - symlinks to the kquant artifact shards (NF3 expert codes + scales).
  - model.safetensors.index.json covering exactly the union.
  - config.json with the nvfp4_nf3_hybrid quantization_config (all-NF3 bit
    map, kquant refit levels, scale bias) replacing the source
    compressed-tensors block.
  - symlinks for tokenizer/aux files.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from kquant.io.hf_cache import resolve

ARTIFACT = Path("/models/Kimi-K3-NF3R-Uniform-3p25")
DEST = Path("/models/Kimi-K3-NF3R-Uniform-3p25-serve")
SHARD_BYTES = 8 << 30
EXPERT_RE = re.compile(r"\.experts\.\d+\.(w1|w2|w3)\.")

EXCLUDE_MODULES = [
    "*self_attn*",
    "*linear_attn*",
    "*shared_experts*",
    "*mlp*",
    "*block_sparse_moe.gate*",
    "*routed_expert*",
    "*routed_output*",
    "*embed_tokens*",
    "lm_head",
    "*vision*",
    "*mm_projector*",
]


def main() -> None:
    cache = resolve()
    snap = Path(cache.snapshot_dir)
    DEST.mkdir(parents=True, exist_ok=True)

    kq_cfg = json.loads((ARTIFACT / "kquant_config.json").read_text())
    nf3 = kq_cfg["formats"]["nf3_refit"]
    layers = kq_cfg["layers_packed"]

    src_index = json.loads((snap / "model.safetensors.index.json").read_text())
    src_map = src_index["weight_map"]
    nonexpert = [n for n in src_map if not EXPERT_RE.search(n)]
    print(f"source tensors {len(src_map)}, non-expert kept {len(nonexpert)}")

    # Group by source shard so each file is opened once.
    by_shard: dict[str, list[str]] = {}
    for name in nonexpert:
        by_shard.setdefault(src_map[name], []).append(name)

    weight_map: dict[str, str] = {}
    total = 0
    buf: dict[str, torch.Tensor] = {}
    buf_bytes = 0
    out_idx = 0

    def flush() -> None:
        nonlocal buf, buf_bytes, out_idx
        if not buf:
            return
        fname = f"nonexpert-{out_idx:05d}.safetensors"
        save_file(buf, str(DEST / fname))
        for n in buf:
            weight_map[n] = fname
        print(f"  wrote {fname} ({buf_bytes / 2**30:.2f} GiB, {len(buf)} tensors)")
        out_idx += 1
        buf, buf_bytes = {}, 0

    for shard, names in sorted(by_shard.items()):
        with safe_open(str(cache.shard_paths[shard]), framework="pt") as f:
            for name in names:
                t = f.get_tensor(name)
                buf[name] = t
                buf_bytes += t.numel() * t.element_size()
                total += t.numel() * t.element_size()
                if buf_bytes >= SHARD_BYTES:
                    flush()
    flush()
    print(f"non-expert total {total / 2**30:.2f} GiB")

    # kquant expert shards: symlink + absorb their index.
    kq_index = json.loads((ARTIFACT / "model-kquant.index.json").read_text())
    for name, fname in kq_index["weight_map"].items():
        weight_map[name] = fname
    for f in sorted(ARTIFACT.glob("model-kquant-*.safetensors")):
        link = DEST / f.name
        if not link.exists():
            link.symlink_to(f)
    total += kq_index.get("metadata", {}).get("total_size", 0)

    (DEST / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=1
        )
    )
    print(f"index: {len(weight_map)} tensors")

    # config.json: swap the quantization config.
    cfg = json.loads((snap / "config.json").read_text())
    qcfg = {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "group_size": 16,
        "exclude_modules": EXCLUDE_MODULES,
        "hybrid_bit_map": {str(layer): [3] * 896 for layer in layers},
        "kept_format": "mxfp4_e8m0k32",
        "nf3_levels": nf3["levels"],
        "nf3_scale_bias_log2": nf3["scale_bias_log2"],
    }
    cfg["text_config"]["quantization_config"] = qcfg
    cfg.pop("quantization_config", None)
    (DEST / "config.json").write_text(json.dumps(cfg, indent=1))

    for f in snap.iterdir():
        if f.suffix == ".safetensors" or f.name in (
            "config.json",
            "model.safetensors.index.json",
        ):
            continue
        link = DEST / f.name
        if not link.exists():
            link.symlink_to(f)
    print(f"done: {DEST}")


if __name__ == "__main__":
    sys.exit(main())
