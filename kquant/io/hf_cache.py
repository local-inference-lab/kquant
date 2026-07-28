"""Resolve Kimi-K3 shards inside a (possibly partial) HuggingFace cache.

The downloader materializes finished files as content-addressed blobs and only
sometimes has snapshot symlinks in place, so we resolve shards in two passes:

1. snapshot symlinks (the normal, fully-downloaded case);
2. orphan blobs, identified by reading each blob's safetensors header and
   mapping its first tensor name back through the index's weight_map.

Blob identification results are cached in ``blob_map.json`` next to the blobs
(keyed by size+mtime) so repeated runs don't rescan a terabyte of headers.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path

from kquant import constants as C

_HEADER_LIMIT = 400_000_000  # sanity cap on safetensors header size


def read_safetensors_header(path: str | os.PathLike) -> dict:
    """Read only the JSON header of a safetensors file."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        if n > _HEADER_LIMIT:
            raise ValueError(f"implausible safetensors header size {n} in {path}")
        return json.loads(f.read(n))


def _default_repo_dir() -> Path:
    hub = Path(
        os.environ.get(
            "HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"
        )
    )
    return hub / ("models--" + C.MODEL_ID.replace("/", "--"))


@dataclass
class CheckpointCache:
    """A resolved view of the checkpoint: index + per-shard file paths."""

    snapshot_dir: Path
    index: dict
    shard_paths: dict[str, Path]  # shard filename -> readable file
    missing_shards: list[str] = field(default_factory=list)

    @property
    def weight_map(self) -> dict[str, str]:
        return self.index["weight_map"]

    def tensor_path(self, name: str) -> Path | None:
        shard = self.weight_map.get(name)
        if shard is None:
            raise KeyError(f"unknown tensor {name}")
        return self.shard_paths.get(shard)

    def has_tensor(self, name: str) -> bool:
        return self.tensor_path(name) is not None

    def complete_moe_layers(self) -> list[int]:
        """MoE layers for which every expert tensor is in a present shard."""
        out = []
        for layer in C.MOE_LAYERS:
            ok = all(
                self.weight_map.get(C.expert_tensor(layer, e, m, part))
                in self.shard_paths
                for e in range(C.NUM_EXPERTS)
                for m in C.EXPERT_MATRICES
                for part in ("weight_packed", "weight_scale")
            )
            if ok:
                out.append(layer)
        return out


def resolve(repo_dir: str | os.PathLike | None = None,
            revision: str = C.REVISION) -> CheckpointCache:
    repo = Path(repo_dir) if repo_dir else _default_repo_dir()
    snap = repo / "snapshots" / revision
    index_path = snap / C.INDEX_FILE
    if not index_path.exists():
        raise FileNotFoundError(
            f"{index_path} not found - download it first "
            f"(hf_hub_download('{C.MODEL_ID}', '{C.INDEX_FILE}'))"
        )
    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index["weight_map"]
    all_shards = sorted(set(weight_map.values()))

    shard_paths: dict[str, Path] = {}
    for shard in all_shards:
        p = snap / shard
        if p.exists():
            shard_paths[shard] = p

    unresolved = [s for s in all_shards if s not in shard_paths]
    if unresolved:
        shard_paths.update(_scan_blobs(repo / "blobs", weight_map, unresolved))

    missing = [s for s in all_shards if s not in shard_paths]
    return CheckpointCache(
        snapshot_dir=snap,
        index=index,
        shard_paths=shard_paths,
        missing_shards=missing,
    )


def _scan_blobs(blob_dir: Path, weight_map: dict[str, str],
                wanted: list[str]) -> dict[str, Path]:
    """Identify orphan blobs by their embedded first tensor name."""
    if not blob_dir.exists():
        return {}
    cache_file = blob_dir / "kquant_blob_map.json"
    cached: dict[str, dict] = {}
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
        except json.JSONDecodeError:
            cached = {}

    found: dict[str, Path] = {}
    dirty = False
    for blob in blob_dir.iterdir():
        if blob.name.endswith(".incomplete") or blob.name.endswith(".json"):
            continue
        st = blob.stat()
        key = blob.name
        entry = cached.get(key)
        if entry and entry["size"] == st.st_size and entry["mtime"] == st.st_mtime:
            shard = entry["shard"]
        else:
            if st.st_size < 1_000_000:
                continue
            try:
                header = read_safetensors_header(blob)
            except (ValueError, json.JSONDecodeError, struct.error):
                continue
            names = (k for k in header if k != "__metadata__")
            shard = next((weight_map.get(n) for n in names), None)
            cached[key] = {"size": st.st_size, "mtime": st.st_mtime, "shard": shard}
            dirty = True
        if shard in wanted:
            found[shard] = blob
    if dirty:
        try:
            cache_file.write_text(json.dumps(cached))
        except OSError:
            pass
    return found
