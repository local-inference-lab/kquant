"""Offline MXFP8 conversion for Qwen3/Kimi-K3 DSpark checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from kquant.mxfp8 import MXFP8_BLOCK_SIZE, mxfp8_quantize_cpu

DEFAULT_SOURCE = "/models/Kimi-K3-DSpark"
DEFAULT_DESTINATION = "/models/Kimi-K3-DSpark-MXFP8"
DEFAULT_MAX_SHARD_SIZE = 1 << 30
DSPARK_VIRTUAL_TP_PROFILE = "dense-gqa"

_ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_PRESERVED_2D_PREFIXES = ("markov_head.", "confidence_head.")
_K3_DROPPED_PREFIXES = ("embed_tokens.", "confidence_head.", "lm_head.")


def _dropped_source_tensor(name: str, config: Mapping[str, Any]) -> bool:
    return "K3DSparkModel" in config.get("architectures", ()) and name.startswith(
        _K3_DROPPED_PREFIXES
    )


def expected_mxfp8_weights(config: Mapping[str, Any]) -> set[str]:
    """Return every dense projection that vLLM constructs with draft quant config."""

    num_layers = int(config["num_hidden_layers"])
    architectures = set(config.get("architectures", ()))
    if "K3DSparkModel" in architectures:
        names = {"context_proj.weight"}
        attention_projections = (
            "q_a_proj",
            "kv_a_proj_with_mqa",
            "q_b_proj",
            "kv_b_proj",
            "o_proj",
        )
        if config.get("mla_use_output_gate", False):
            attention_projections += ("g_proj",)
    else:
        names = {"fc.weight"}
        attention_projections = _ATTENTION_PROJECTIONS
    for layer_idx in range(num_layers):
        prefix = f"layers.{layer_idx}"
        names.update(
            f"{prefix}.self_attn.{projection}.weight"
            for projection in attention_projections
        )
        names.update(
            f"{prefix}.mlp.{projection}.weight" for projection in _MLP_PROJECTIONS
        )
    return names


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _read_config(source: Path) -> dict[str, Any]:
    config_path = source / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing DSpark config: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)

    required = {
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "vocab_size",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"DSpark config is missing required fields: {missing}")
    return config


def _source_weight_file(source: Path) -> Path:
    weight = source / "model.safetensors"
    if not weight.is_file():
        raise FileNotFoundError(
            "the DSpark converter currently expects a single source "
            f"model.safetensors file, missing: {weight}"
        )
    return weight


def _validate_source_tensors(
    names: set[str],
    expected: set[str],
    tensors: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
) -> None:
    missing = sorted(expected.difference(names))
    if missing:
        raise ValueError(f"source checkpoint is missing MXFP8 projections: {missing}")

    collisions = sorted(
        f"{name[:-7]}weight_scale"
        for name in expected
        if f"{name[:-7]}weight_scale" in names
    )
    if collisions:
        raise ValueError(
            f"source checkpoint already contains MXFP8 scales: {collisions}"
        )

    unsupported_2d = []
    for name in sorted(names.difference(expected)):
        if _dropped_source_tensor(name, config):
            continue
        tensor = tensors[name]
        if (
            name.endswith(".weight")
            and tensor.ndim == 2
            and not name.startswith(_PRESERVED_2D_PREFIXES)
        ):
            unsupported_2d.append(name)
    if unsupported_2d:
        raise ValueError(
            "refusing to silently preserve unknown dense weights; update the "
            f"DSpark conversion contract for: {unsupported_2d}"
        )


def _rewritten_config(source_config: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(source_config)
    is_k3_mla = "K3DSparkModel" in config.get("architectures", ())
    if not is_k3_mla:
        config["architectures"] = ["Qwen3DSparkModel"]
    config.pop("auto_map", None)
    config.setdefault("draft_vocab_size", int(config["vocab_size"]))
    if not is_k3_mla:
        config["vllm_virtual_tp_profile"] = DSPARK_VIRTUAL_TP_PROFILE
    config["quantization_config"] = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "dense_format": "mxfp8",
        # These heads intentionally remain BF16/FP32. vLLM does not pass the
        # draft quant config into them today; the ignores keep that invariant
        # explicit if their constructors change later.
        "ignored_layers": ["markov_head", "confidence_head", "lm_head"],
    }
    return config


def _pad_k3_mla_heads(
    name: str,
    tensor: torch.Tensor,
    config: Mapping[str, Any],
    tensor_parallel_size: int | None,
) -> torch.Tensor:
    if tensor_parallel_size is None or "K3DSparkModel" not in config.get(
        "architectures", ()
    ):
        return tensor
    if tensor_parallel_size <= 0:
        raise ValueError("tensor_parallel_size must be positive")

    num_heads = int(config["num_attention_heads"])
    padded_heads = math.ceil(num_heads / tensor_parallel_size) * tensor_parallel_size
    if padded_heads == num_heads:
        return tensor

    qk_dim = int(config["qk_nope_head_dim"]) + int(config["qk_rope_head_dim"])
    kv_dim = int(config["qk_nope_head_dim"]) + int(config["v_head_dim"])
    v_dim = int(config["v_head_dim"])
    if name.endswith(".self_attn.q_b_proj.weight"):
        expected = num_heads * qk_dim
        if tensor.shape[0] != expected:
            raise ValueError(f"{name} has {tensor.shape[0]} rows, expected {expected}")
        return torch.nn.functional.pad(
            tensor, (0, 0, 0, (padded_heads - num_heads) * qk_dim)
        )
    if name.endswith(".self_attn.kv_b_proj.weight"):
        expected = num_heads * kv_dim
        if tensor.shape[0] != expected:
            raise ValueError(f"{name} has {tensor.shape[0]} rows, expected {expected}")
        return torch.nn.functional.pad(
            tensor, (0, 0, 0, (padded_heads - num_heads) * kv_dim)
        )
    if name.endswith(".self_attn.o_proj.weight"):
        expected = num_heads * v_dim
        if tensor.shape[1] != expected:
            raise ValueError(
                f"{name} has {tensor.shape[1]} columns, expected {expected}"
            )
        return torch.nn.functional.pad(
            tensor, (0, (padded_heads - num_heads) * v_dim)
        )
    return tensor


def _pad_k3_mlp(
    name: str,
    tensor: torch.Tensor,
    config: Mapping[str, Any],
    tensor_parallel_size: int | None,
) -> torch.Tensor:
    if tensor_parallel_size is None or "K3DSparkModel" not in config.get(
        "architectures", ()
    ):
        return tensor
    if tensor_parallel_size <= 0:
        raise ValueError("tensor_parallel_size must be positive")

    intermediate_size = int(config["intermediate_size"])
    alignment = tensor_parallel_size * MXFP8_BLOCK_SIZE
    padded_size = math.ceil(intermediate_size / alignment) * alignment
    if padded_size == intermediate_size:
        return tensor

    if name.endswith((".mlp.gate_proj.weight", ".mlp.up_proj.weight")):
        if tensor.shape[0] != intermediate_size:
            raise ValueError(
                f"{name} has {tensor.shape[0]} rows, expected {intermediate_size}"
            )
        return torch.nn.functional.pad(
            tensor, (0, 0, 0, padded_size - intermediate_size)
        )
    if name.endswith(".mlp.down_proj.weight"):
        if tensor.shape[1] != intermediate_size:
            raise ValueError(
                f"{name} has {tensor.shape[1]} columns, expected "
                f"{intermediate_size}"
            )
        return torch.nn.functional.pad(
            tensor, (0, padded_size - intermediate_size)
        )
    return tensor


def _write_shards(
    tensors: Mapping[str, torch.Tensor],
    destination: Path,
    *,
    max_shard_size: int,
    metadata: dict[str, str] | None,
) -> tuple[list[str], dict[str, str], int]:
    if max_shard_size <= 0:
        raise ValueError(f"max_shard_size must be positive, got {max_shard_size}")

    groups: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    total_size = 0
    for name in sorted(tensors):
        size = _tensor_nbytes(tensors[name])
        total_size += size
        if current and current_size + size > max_shard_size:
            groups.append(current)
            current = []
            current_size = 0
        current.append(name)
        current_size += size
    if current:
        groups.append(current)

    if len(groups) < 2:
        raise ValueError(
            "requested a sharded artifact, but max_shard_size produced only one "
            "shard; lower --max-shard-size-bytes"
        )

    shard_names = [
        f"model-{index:05d}-of-{len(groups):05d}.safetensors"
        for index in range(1, len(groups) + 1)
    ]
    weight_map: dict[str, str] = {}
    for shard_name, names in zip(shard_names, groups, strict=True):
        shard_tensors = {name: tensors[name].contiguous() for name in names}
        save_file(shard_tensors, str(destination / shard_name), metadata=metadata)
        weight_map.update({name: shard_name for name in names})

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with (destination / "model.safetensors.index.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return shard_names, weight_map, total_size


def _copy_auxiliary_files(source: Path, destination: Path) -> None:
    for path in source.iterdir():
        if not path.is_file():
            continue
        if path.name == "config.json" or path.name.endswith(".safetensors"):
            continue
        if path.name == "model.safetensors.index.json":
            continue
        shutil.copy2(path, destination / path.name, follow_symlinks=True)


def _source_revision(source: Path) -> str | None:
    if source.parent.name == "snapshots":
        return source.name
    return None


def convert_dspark_checkpoint(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    max_shard_size: int = DEFAULT_MAX_SHARD_SIZE,
    tensor_parallel_size: int | None = None,
    progress: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    """Create an atomic, indexed MXFP8 copy of a DSpark checkpoint."""

    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    if not source_path.is_dir():
        raise FileNotFoundError(f"source model directory does not exist: {source_path}")
    if destination_path.exists():
        raise FileExistsError(f"destination already exists: {destination_path}")
    if source_path == destination_path:
        raise ValueError("source and destination must be different directories")

    source_config = _read_config(source_path)
    source_weight = _source_weight_file(source_path)
    expected = expected_mxfp8_weights(source_config)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_path.name}.incomplete-",
            dir=destination_path.parent,
        )
    )

    try:
        output: dict[str, torch.Tensor] = {}
        quantized: list[dict[str, Any]] = []
        with safe_open(source_weight, framework="pt", device="cpu") as source_file:
            names = set(source_file.keys())
            source_tensors = {name: source_file.get_tensor(name) for name in names}
            _validate_source_tensors(names, expected, source_tensors, source_config)
            metadata = source_file.metadata()

            for index, name in enumerate(sorted(names), start=1):
                if _dropped_source_tensor(name, source_config):
                    continue
                tensor = _pad_k3_mla_heads(
                    name,
                    source_tensors[name],
                    source_config,
                    tensor_parallel_size,
                )
                tensor = _pad_k3_mlp(
                    name,
                    tensor,
                    source_config,
                    tensor_parallel_size,
                )
                if name in expected:
                    if tensor.shape[-1] % MXFP8_BLOCK_SIZE:
                        raise ValueError(
                            f"{name} input dimension {tensor.shape[-1]} is not "
                            f"divisible by {MXFP8_BLOCK_SIZE}"
                        )
                    quantized_weight, scale = mxfp8_quantize_cpu(tensor)
                    output[name] = quantized_weight
                    scale_name = name.removesuffix(".weight") + ".weight_scale"
                    output[scale_name] = scale
                    quantized.append(
                        {
                            "name": name,
                            "shape": list(tensor.shape),
                            "source_dtype": str(tensor.dtype).removeprefix("torch."),
                            "scale_name": scale_name,
                            "scale_shape": list(scale.shape),
                        }
                    )
                    if progress is not None:
                        progress(
                            f"[{index}/{len(names)}] quantized {name} "
                            f"{tuple(tensor.shape)}"
                        )
                else:
                    output[name] = tensor

        shard_names, weight_map, total_size = _write_shards(
            output,
            staging,
            max_shard_size=max_shard_size,
            metadata=metadata,
        )
        _copy_auxiliary_files(source_path, staging)
        rewritten_config = _rewritten_config(source_config)
        if (
            tensor_parallel_size is not None
            and "K3DSparkModel" in rewritten_config.get("architectures", ())
        ):
            original_heads = int(rewritten_config["num_attention_heads"])
            padded_heads = (
                math.ceil(original_heads / tensor_parallel_size)
                * tensor_parallel_size
            )
            rewritten_config["original_num_attention_heads"] = original_heads
            rewritten_config["num_attention_heads"] = padded_heads
            rewritten_config["num_key_value_heads"] = padded_heads
            original_intermediate_size = int(rewritten_config["intermediate_size"])
            intermediate_alignment = tensor_parallel_size * MXFP8_BLOCK_SIZE
            padded_intermediate_size = (
                math.ceil(original_intermediate_size / intermediate_alignment)
                * intermediate_alignment
            )
            rewritten_config["original_intermediate_size"] = (
                original_intermediate_size
            )
            rewritten_config["intermediate_size"] = padded_intermediate_size
        architecture = rewritten_config["architectures"][0]
        with (staging / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(rewritten_config, handle, indent=2, sort_keys=True)
            handle.write("\n")

        manifest: dict[str, Any] = {
            "format": "kquant-dspark-mxfp8-v1",
            "source": str(source_path),
            "source_revision": _source_revision(source_path),
            "architecture": architecture,
            "quantized_weight_count": len(quantized),
            "preserved_tensor_count": len(output) - 2 * len(quantized),
            "output_tensor_count": len(output),
            "max_shard_size_bytes": max_shard_size,
            "tensor_data_bytes": total_size,
            "shards": shard_names,
            "quantized_weights": quantized,
            "runtime": {
                "backbone_tensor_parallel": True,
                "draft_load_config": {"load_format": "safetensors"},
                "draft_tensor_parallel_size": "must equal target tensor_parallel_size",
                "large_replicated_weights": ["fc.weight"],
                "virtual_tp_profile": rewritten_config.get(
                    "vllm_virtual_tp_profile"
                ),
                "original_num_attention_heads": rewritten_config.get(
                    "original_num_attention_heads"
                ),
                "padded_num_attention_heads": rewritten_config.get(
                    "num_attention_heads"
                ),
                "original_intermediate_size": rewritten_config.get(
                    "original_intermediate_size"
                ),
                "padded_intermediate_size": rewritten_config.get(
                    "intermediate_size"
                ),
            },
        }
        with (staging / "mxfp8_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")

        for path in staging.iterdir():
            if path.is_file():
                path.chmod(0o644)
        staging.chmod(0o755)

        if set(weight_map) != set(output):
            raise AssertionError("safetensors index does not cover every output tensor")
        if len(quantized) != len(expected):
            raise AssertionError(
                f"quantized {len(quantized)} weights, expected {len(expected)}"
            )
        os.replace(staging, destination_path)
        if progress is not None:
            progress(
                f"wrote {len(shard_names)} shards and {len(quantized)} MXFP8 "
                f"weights to {destination_path}"
            )
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a sharded MXFP8 copy of a Qwen3/Kimi-K3 DSpark model"
    )
    parser.add_argument("--src", default=DEFAULT_SOURCE)
    parser.add_argument("--dest", default=DEFAULT_DESTINATION)
    parser.add_argument(
        "--max-shard-size-bytes",
        type=int,
        default=DEFAULT_MAX_SHARD_SIZE,
        help="maximum tensor bytes per output safetensors shard (default: 1 GiB)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=12,
        help="pad K3 MLA heads to this TP size (default: 12)",
    )
    args = parser.parse_args()
    convert_dspark_checkpoint(
        args.src,
        args.dest,
        max_shard_size=args.max_shard_size_bytes,
        tensor_parallel_size=args.tensor_parallel_size,
    )


if __name__ == "__main__":
    main()
