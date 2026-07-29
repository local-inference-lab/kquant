"""Extract the MXFP4 keep-tier experts into dedicated shards.

Reads allocation-exl3.json, copies the kept experts' weight_packed and
weight_scale tensors from the source checkpoint into keep-tier shards next
to the EXL3 layer shards, so the serve dir never references source files
(vLLM reads whole files; partial references leak tensors — Phase A lesson).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

from kquant.io.hf_cache import resolve
from kquant.io.stream import load_tensor

DEST = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/models/Kimi-K3-EXL3-3p19")
SHARD_BYTES = 8 << 30


def main() -> None:
    cache = resolve()
    alloc = json.loads((DEST / "allocation-exl3.json").read_text())["layers"]
    buf: dict[str, torch.Tensor] = {}
    buf_bytes = 0
    idx = 0
    total = 0

    def flush() -> None:
        nonlocal buf, buf_bytes, idx
        if not buf:
            return
        save_file(buf, str(DEST / f"keep-mxfp4-{idx:05d}.safetensors"))
        print(f"  keep-mxfp4-{idx:05d}: {buf_bytes/2**30:.2f} GiB", flush=True)
        idx += 1
        buf, buf_bytes = {}, 0

    for layer in sorted(alloc, key=int):
        for e in alloc[layer]["keep"]:
            for mat in ("w1", "w3", "w2"):
                base = (
                    f"language_model.model.layers.{layer}.block_sparse_moe."
                    f"experts.{e}.{mat}"
                )
                for part in (".weight_packed", ".weight_scale"):
                    t = load_tensor(cache, base + part)
                    buf[base + part] = t
                    buf_bytes += t.numel() * t.element_size()
                    total += t.numel() * t.element_size()
            if buf_bytes >= SHARD_BYTES:
                flush()
    flush()
    print(f"keep tier: {total/2**30:.1f} GiB total", flush=True)


if __name__ == "__main__":
    main()
