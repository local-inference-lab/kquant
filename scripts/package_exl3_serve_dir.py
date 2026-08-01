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

import argparse
import json
from pathlib import Path

from safetensors import safe_open

DEFAULT_ARTIFACT = Path("/models/Kimi-K3-EXL3-3p19")
DEFAULT_NONEXPERT = Path("/models/Kimi-K3-mxfp8-nonexpert")
_TP_LOCAL_TRELLIS_KEYS = (
    "compatible_tp_sizes",
    "tp_local_quantization",
    "tp_local_intermediate_size",
    "intermediate_hadamard_blocks",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, nargs="?", default=DEFAULT_ARTIFACT)
    parser.add_argument("destination", type=Path, nargs="?")
    parser.add_argument("--nonexpert", type=Path, default=DEFAULT_NONEXPERT)
    return parser.parse_args()


def artifact_is_shared_su(art: Path, alloc: dict) -> bool:
    """True if the artifact stores identical H-side su rows across experts
    (kquant shared-su pack); verified against real tensors, not metadata."""
    import torch

    layer = min(alloc, key=int)
    dem = alloc[layer]["exl3"]
    base = f"language_model.model.layers.{layer}.block_sparse_moe.experts"
    with safe_open(
        str(art / f"exl3-layer-{int(layer):05d}.safetensors"), framework="pt"
    ) as sf:
        a = sf.get_tensor(f"{base}.{dem[0]}.w1.exl3_suh")
        b = sf.get_tensor(f"{base}.{dem[1]}.w1.exl3_suh")
    return torch.equal(a, b)


def trellis_config(manifest: dict, *, shared_su: bool) -> dict:
    config = {
        "bits": 3,
        "codebook": "mcg",
        "mcg_mult": manifest["mcg_mult"],
        "shared_su": bool(shared_su),
    }
    for key in _TP_LOCAL_TRELLIS_KEYS:
        if key in manifest:
            config[key] = manifest[key]
    return config


def require_complete_tp_build(artifact: Path, manifest: dict, alloc: dict) -> None:
    if not manifest.get("tp_local_quantization"):
        return
    if manifest.get("build_complete") is not True:
        raise RuntimeError("refusing to package an incomplete TP-local EXL3 artifact")
    compatible = manifest.get("compatible_tp_sizes")
    if not isinstance(compatible, list) or len(compatible) != 1:
        raise RuntimeError("TP-local artifact must declare exactly one TP size")
    build_id = manifest.get("build_id")
    if not isinstance(build_id, str) or not build_id:
        raise RuntimeError("TP-local artifact has no build_id")

    try:
        from scripts.pack_exl3_12gpu import layer_is_complete
    except ModuleNotFoundError:
        # Direct ``python /path/to/scripts/package_exl3_serve_dir.py`` puts the
        # scripts directory, rather than the repository root, on sys.path.
        from pack_exl3_12gpu import layer_is_complete

    missing = [
        int(layer)
        for layer in sorted(alloc, key=int)
        if not layer_is_complete(artifact, int(layer), build_id, int(compatible[0]))
    ]
    if missing:
        raise RuntimeError(
            f"refusing TP-local artifact with {len(missing)} incomplete layer(s); "
            f"sample={missing[:8]}"
        )


def main() -> None:
    from kquant.io.hf_cache import resolve

    args = parse_args()
    artifact = args.artifact
    destination = args.destination or Path(str(artifact) + "-serve")
    nonexpert = args.nonexpert
    cache = resolve()
    snap = Path(cache.snapshot_dir)
    alloc_doc = json.loads((artifact / "allocation-exl3.json").read_text())
    alloc = alloc_doc["layers"]
    manifest = json.loads((artifact / "kquant_exl3_manifest.json").read_text())
    require_complete_tp_build(artifact, manifest, alloc)
    destination.mkdir(parents=True, exist_ok=True)

    weight_map: dict[str, str] = {}
    total = 0
    groups = [
        sorted(artifact.glob("exl3-layer-*.safetensors")),
        sorted(artifact.glob("keep-mxfp4-*.safetensors")),
        sorted(nonexpert.glob("00-nonexpert-*.safetensors")),
    ]
    assert all(groups), "missing a shard group"
    for files in groups:
        for f in files:
            link = destination / f.name
            if not link.exists():
                link.symlink_to(f)
            with safe_open(str(f), framework="pt") as sf:
                for name in sf.keys():  # noqa: SIM118
                    weight_map[name] = f.name
            total += f.stat().st_size
    (destination / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map})
    )
    print(f"index: {len(weight_map)} tensors, {total / 2**30:.1f} GiB", flush=True)

    bit_map = {
        layer: [4 if e in set(v["keep"]) else 3 for e in range(896)]
        for layer, v in ((k, alloc[k]) for k in sorted(alloc, key=int))
    }
    cfg = json.loads((snap / "config.json").read_text())
    trellis = trellis_config(manifest, shared_su=artifact_is_shared_su(artifact, alloc))
    cfg["text_config"]["quantization_config"] = {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "group_size": 16,
        "hybrid_bit_map": bit_map,
        "kept_format": "mxfp4_e8m0k32",
        "demoted_format": "exl3_3",
        "trellis": trellis,
        # Non-expert linears are offline-baked MXFP8 (fp8 + e8m0 scales);
        # the listed modules stay BF16.
        "dense_format": "mxfp8",
        "ignored_layers": [
            "kv_b_proj",
            "g_proj",
            "f_a_proj",
            "f_b_proj",
            # The loader matches ignored entries as substrings.  Keep the
            # KDA projection ignored without accidentally matching MLA's
            # q_b_proj, which is present in the offline MXFP8 overlay.
            ".b_proj",
            "vision_tower",
            "mm_projector",
        ],
    }
    cfg.pop("quantization_config", None)
    (destination / "config.json").write_text(json.dumps(cfg, indent=1))

    for f in snap.iterdir():
        if f.suffix == ".safetensors" or f.name in (
            "config.json",
            "model.safetensors.index.json",
        ):
            continue
        link = destination / f.name
        if not link.exists():
            link.symlink_to(f)
    print(f"done: {destination}", flush=True)


if __name__ == "__main__":
    main()
