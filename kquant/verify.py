"""Round-trip verification of a packed artifact against the source checkpoint."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from safetensors import safe_open

from kquant import constants as C
from kquant.io import mxfp4
from kquant.io.hf_cache import CheckpointCache
from kquant.io.stream import load_tensor
from kquant.quantize import (
    CodebookFormat,
    decode_scale_bytes,
    quantize_blocks,
    unpack_3bit,
)


def _artifact_tensor(art_dir: Path, index: dict, name: str) -> torch.Tensor:
    shard = index["weight_map"][name]
    with safe_open(str(art_dir / shard), framework="pt") as f:
        return f.get_tensor(name)


def verify_artifact(
    cache: CheckpointCache,
    art_dir: str | Path,
    formats: dict[str, CodebookFormat],
    alloc_assignment,
    alloc_formats: list[str],
    samples: int = 64,
    seed: int = 0,
) -> dict:
    art_dir = Path(art_dir)
    index = json.loads((art_dir / "model-kquant.index.json").read_text())
    config = json.loads((art_dir / "kquant_config.json").read_text())
    layers = config["layers_packed"]
    rng = random.Random(seed)

    keep_ok = keep_bad = nf_ok = nf_bad = 0
    mse_by_fmt: dict[str, list[float]] = {}
    checked = []
    for _ in range(samples):
        layer = rng.choice(layers)
        expert = rng.randrange(C.NUM_EXPERTS)
        matrix = rng.choice(C.EXPERT_MATRICES)
        li = layer - C.MOE_LAYERS[0]
        fmt_name = alloc_formats[int(alloc_assignment[li, expert])]
        base = (
            f"{C.LM_PREFIX}layers.{layer}.block_sparse_moe."
            f"experts.{expert}.{matrix}"
        )
        src_packed = load_tensor(cache, f"{base}.weight_packed")
        src_scale = load_tensor(cache, f"{base}.weight_scale")
        if fmt_name == "keep_mxfp4":
            ap = _artifact_tensor(art_dir, index, f"{base}.weight_packed")
            asc = _artifact_tensor(art_dir, index, f"{base}.weight_scale")
            ok = torch.equal(ap, src_packed) and torch.equal(asc, src_scale)
            keep_ok += ok
            keep_bad += not ok
        else:
            fmt = formats[fmt_name]
            ap = _artifact_tensor(art_dir, index, f"{base}.{fmt.name}_packed")
            asc = _artifact_tensor(art_dir, index, f"{base}.{fmt.name}_scale")
            vals, s = mxfp4.dequant_blocked(src_packed.unsqueeze(0),
                                            src_scale.unsqueeze(0))
            w = vals * s.unsqueeze(-1)
            q = quantize_blocks(w, fmt)
            K = w.shape[-2] * C.MXFP4_BLOCK
            from kquant.quantize import encode_scale_bytes, pack_3bit
            exp_packed = pack_3bit(q.codes.reshape(1, w.shape[1], K))[0]
            exp_scale = encode_scale_bytes(q.scales[0])
            ok = torch.equal(ap, exp_packed) and torch.equal(asc, exp_scale)
            nf_ok += ok
            nf_bad += not ok
            # dequant artifact and report true MSE vs source
            codes = unpack_3bit(ap.unsqueeze(0), K)
            t = decode_scale_bytes(asc.unsqueeze(0))
            levels = fmt.levels_tensor(ap.device)
            recon = levels[codes.reshape(1, w.shape[1], -1, C.MXFP4_BLOCK)] \
                * t.unsqueeze(-1)
            mse = float(((w - recon) ** 2).mean())
            mse_by_fmt.setdefault(fmt_name, []).append(mse)
        checked.append((layer, expert, matrix, fmt_name))

    # byte accounting
    shard_bytes = sum(
        (art_dir / f).stat().st_size
        for f in set(index["weight_map"].values())
    )
    report = {
        "samples": samples,
        "keep_bitwise_ok": keep_ok,
        "keep_bitwise_bad": keep_bad,
        "demoted_deterministic_ok": nf_ok,
        "demoted_deterministic_bad": nf_bad,
        "mse_by_format": {
            k: {"mean": sum(v) / len(v), "max": max(v), "n": len(v)}
            for k, v in mse_by_fmt.items()
        },
        "artifact_tensor_bytes": index["metadata"]["total_size"],
        "artifact_file_bytes": shard_bytes,
        "config_total_bytes": config["total_bytes"],
    }
    report["ok"] = keep_bad == 0 and nf_bad == 0
    return report
