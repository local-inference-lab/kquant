"""Package the EXL3 artifact + keep tier + non-expert shards for vLLM.

Builds /models/Kimi-K3-EXL3-3p19-serve with:
  - symlinks: exl3-layer-*.safetensors, keep-mxfp4-*.safetensors
  - symlinks: 00-nonexpert-* side-shards (reused from the Phase A serve dir;
    same source tensors)
  - merged model.safetensors.index.json over exactly those files
  - config.json with the exl3_3 hybrid quantization_config
  - tokenizer/aux symlinks
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from safetensors import safe_open

ART = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/models/Kimi-K3-EXL3-3p19")
NONEXPERT_SRC = Path("/models/Kimi-K3-mxfp8-nonexpert")
DEST = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(str(ART) + "-serve")


def artifact_is_shared_su(art: Path, alloc: dict) -> bool:
    """True if the artifact stores identical H-side su rows across experts
    (kquant shared-su pack); verified against real tensors, not metadata."""
    import torch

    layer = sorted(alloc, key=int)[0]
    dem = alloc[layer]["exl3"]
    base = (
        f"language_model.model.layers.{layer}.block_sparse_moe.experts"
    )
    with safe_open(
        str(art / f"exl3-layer-{int(layer):05d}.safetensors"), framework="pt"
    ) as sf:
        a = sf.get_tensor(f"{base}.{dem[0]}.w1.exl3_suh")
        b = sf.get_tensor(f"{base}.{dem[1]}.w1.exl3_suh")
    return torch.equal(a, b)


def main() -> None:
    from kquant.io.hf_cache import resolve

    cache = resolve()
    snap = Path(cache.snapshot_dir)
    DEST.mkdir(parents=True, exist_ok=True)
    alloc_doc = json.loads((ART / "allocation-exl3.json").read_text())
    alloc = alloc_doc["layers"]

    weight_map: dict[str, str] = {}
    total = 0
    groups = [
        sorted(ART.glob("exl3-layer-*.safetensors")),
        sorted(ART.glob("keep-mxfp4-*.safetensors")),
        sorted(NONEXPERT_SRC.glob("00-nonexpert-*.safetensors")),
    ]
    assert all(groups), "missing a shard group"
    for files in groups:
        for f in files:
            link = DEST / f.name
            if not link.exists():
                link.symlink_to(f)
            with safe_open(str(f), framework="pt") as sf:
                for name in sf.keys():
                    weight_map[name] = f.name
            total += f.stat().st_size
    (DEST / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map})
    )
    print(f"index: {len(weight_map)} tensors, {total/2**30:.1f} GiB", flush=True)

    manifest = json.loads((ART / "kquant_exl3_manifest.json").read_text())
    bit_map = {
        layer: [4 if e in set(v["keep"]) else 3 for e in range(896)]
        for layer, v in ((k, alloc[k]) for k in sorted(alloc, key=int))
    }
    cfg = json.loads((snap / "config.json").read_text())
    cfg["text_config"]["quantization_config"] = {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "group_size": 16,
        "hybrid_bit_map": bit_map,
        "kept_format": "mxfp4_e8m0k32",
        "demoted_format": "exl3_3",
        "trellis": {"bits": 3, "codebook": "mcg",
                    "mcg_mult": manifest["mcg_mult"],
                    "shared_su": artifact_is_shared_su(ART, alloc)},
        # Non-expert linears are offline-baked MXFP8 (fp8 + e8m0 scales);
        # the listed modules stay BF16.
        "dense_format": "mxfp8",
        "ignored_layers": [
            "kv_b_proj", "g_proj", "f_a_proj", "f_b_proj", "b_proj",
            "vision_tower", "mm_projector",
        ],
    }
    cfg.pop("quantization_config", None)
    (DEST / "config.json").write_text(json.dumps(cfg, indent=1))

    for f in snap.iterdir():
        if f.suffix == ".safetensors" or f.name in (
            "config.json", "model.safetensors.index.json",
        ):
            continue
        link = DEST / f.name
        if not link.exists():
            link.symlink_to(f)
    print(f"done: {DEST}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
