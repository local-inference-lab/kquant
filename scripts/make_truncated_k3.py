"""Build a deterministic first-N-layer Kimi-K3 checkpoint.

The source checkpoint can be a regular model directory or kquant's resolved
Hugging Face cache view. The latter is important for Kimi-K3 because its index
and weight shards may live in different cache snapshots/blobs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LAYER_RE = re.compile(r"language_model\.model\.layers\.(\d+)\.")
SHARD_BYTES = 12 << 30


@dataclass(frozen=True)
class Source:
    config_dir: Path
    index: dict
    shard_paths: dict[str, Path]


def wanted(name: str, n_layers: int) -> bool:
    match = LAYER_RE.search(name)
    return match is None or int(match.group(1)) < n_layers


def source_from_directory(path: Path) -> Source:
    index_path = path / "model.safetensors.index.json"
    config_path = path / "config.json"
    if not index_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            f"{path} must contain config.json and model.safetensors.index.json"
        )
    index = json.loads(index_path.read_text())
    files = sorted(set(index["weight_map"].values()))
    shard_paths = {name: path / name for name in files}
    missing = [name for name, shard in shard_paths.items() if not shard.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{path}: {len(missing)} indexed shards are missing: {missing[:4]}"
        )
    return Source(config_dir=path, index=index, shard_paths=shard_paths)


def source_from_cache(repo_dir: Path | None) -> Source:
    from kquant.io.hf_cache import resolve

    cache = resolve(repo_dir=repo_dir)
    if cache.missing_shards:
        raise FileNotFoundError(
            f"checkpoint cache is missing {len(cache.missing_shards)} shards"
        )
    return Source(
        config_dir=Path(cache.snapshot_dir),
        index=cache.index,
        shard_paths={name: Path(path) for name, path in cache.shard_paths.items()},
    )


def iter_source(source: Source):
    by_file: dict[str, list[str]] = {}
    for name, filename in source.index["weight_map"].items():
        by_file.setdefault(filename, []).append(name)
    for filename, names in sorted(by_file.items()):
        path = source.shard_paths[filename]
        with safe_open(str(path), framework="pt") as handle:
            available = set(handle.keys())
            missing = sorted(set(names) - available)
            if missing:
                raise KeyError(
                    f"{path}: index references {len(missing)} absent tensors"
                )
            for name in sorted(names):
                yield name, handle.get_tensor(name)


def truncate_config(
    config: dict,
    *,
    n_layers: int,
    quant_mode: str,
) -> dict:
    text_config = config.get("text_config", config)
    source_layers = int(text_config["num_hidden_layers"])
    if not 1 <= n_layers <= source_layers:
        raise ValueError(
            f"--layers must be in [1, {source_layers}], got {n_layers}"
        )
    text_config["num_hidden_layers"] = n_layers
    linear_attention = text_config["linear_attn_config"]
    linear_attention["kda_layers"] = [
        value
        for value in linear_attention["kda_layers"]
        if int(value) <= n_layers
    ]
    linear_attention["full_attn_layers"] = [
        value
        for value in linear_attention["full_attn_layers"]
        if int(value) <= n_layers
    ]

    moe_layers = [str(layer) for layer in range(1, n_layers)]
    if quant_mode == "hybrid-all-kept":
        text_config["quantization_config"] = {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "group_size": 16,
            "hybrid_bit_map": {
                layer: [4] * int(text_config["num_experts"])
                for layer in moe_layers
            },
            "kept_format": "mxfp4_e8m0k32",
        }
    elif quant_mode == "hybrid":
        quant = text_config.get("quantization_config")
        if not isinstance(quant, dict) or "hybrid_bit_map" not in quant:
            raise ValueError("hybrid mode requires a hybrid source config")
        quant["hybrid_bit_map"] = {
            layer: quant["hybrid_bit_map"][layer] for layer in moe_layers
        }
    elif quant_mode == "native":
        quant = text_config.get("quantization_config")
        if not isinstance(quant, dict) or quant.get("quant_method") != (
            "compressed-tensors"
        ):
            raise ValueError(
                "native mode requires a compressed-tensors source config"
            )
    else:
        raise ValueError(f"unsupported quantization mode {quant_mode!r}")

    if "text_config" in config:
        config.pop("quantization_config", None)
    return config


def copy_auxiliary_files(source_dir: Path, destination: Path) -> None:
    excluded = {"config.json", "model.safetensors.index.json"}
    for source in sorted(source_dir.iterdir()):
        if (
            not source.is_file()
            or source.suffix == ".safetensors"
            or source.name in excluded
        ):
            continue
        shutil.copy2(source.resolve(), destination / source.name)


def build(
    *,
    source: Source,
    destination: Path,
    n_layers: int,
    quant_mode: str,
    shard_bytes: int = SHARD_BYTES,
) -> dict:
    if destination.exists():
        raise FileExistsError(
            f"{destination} already exists; choose a fresh destination"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-{os.getpid()}-",
            dir=destination.parent,
        )
    )
    try:
        weight_map: dict[str, str] = {}
        buffer: dict[str, torch.Tensor] = {}
        buffer_bytes = 0
        shard_index = 0
        total_bytes = 0

        def flush() -> None:
            nonlocal buffer, buffer_bytes, shard_index
            if not buffer:
                return
            filename = f"model-{shard_index:05d}.safetensors"
            save_file(buffer, str(staging / filename))
            weight_map.update({name: filename for name in buffer})
            shard_index += 1
            buffer = {}
            buffer_bytes = 0

        for name, tensor in iter_source(source):
            if not wanted(name, n_layers):
                continue
            buffer[name] = tensor
            tensor_bytes = tensor.numel() * tensor.element_size()
            buffer_bytes += tensor_bytes
            total_bytes += tensor_bytes
            if buffer_bytes >= shard_bytes:
                flush()
        flush()
        if not weight_map:
            raise RuntimeError("truncation selected no tensors")

        index = {
            "metadata": {"total_size": total_bytes},
            "weight_map": dict(sorted(weight_map.items())),
        }
        (staging / "model.safetensors.index.json").write_text(
            json.dumps(index, sort_keys=True) + "\n"
        )
        config = json.loads((source.config_dir / "config.json").read_text())
        config = truncate_config(
            config,
            n_layers=n_layers,
            quant_mode=quant_mode,
        )
        (staging / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n"
        )
        copy_auxiliary_files(source.config_dir, staging)

        indexed_files = set(weight_map.values())
        missing = [
            name for name in indexed_files if not (staging / name).is_file()
        ]
        leaked_layers = [
            name
            for name in weight_map
            if (match := LAYER_RE.search(name))
            and int(match.group(1)) >= n_layers
        ]
        if missing or leaked_layers:
            raise RuntimeError(
                f"invalid output: missing={missing}, leaked={leaked_layers[:4]}"
            )
        staging.rename(destination)
        return {
            "destination": str(destination),
            "tensor_count": len(weight_map),
            "shard_count": shard_index,
            "total_bytes": total_bytes,
            "layers": n_layers,
            "quant_mode": quant_mode,
        }
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("stock", "ours"), required=True)
    parser.add_argument(
        "--src-dir",
        type=Path,
        help="explicit complete source model directory",
    )
    parser.add_argument(
        "--cache-repo-dir",
        type=Path,
        help="optional Hugging Face repo cache root for stock source",
    )
    parser.add_argument(
        "--quant-mode",
        choices=("native", "hybrid", "hybrid-all-kept"),
        help="default: native for stock, hybrid for ours",
    )
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()

    if args.src_dir is not None:
        source = source_from_directory(args.src_dir)
    elif args.source == "stock":
        source = source_from_cache(args.cache_repo_dir)
    else:
        source = source_from_directory(
            Path("/models/Kimi-K3-EXL3-3p09-serve")
        )
    quant_mode = args.quant_mode or (
        "native" if args.source == "stock" else "hybrid"
    )
    result = build(
        source=source,
        destination=args.dest,
        n_layers=args.layers,
        quant_mode=quant_mode,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
