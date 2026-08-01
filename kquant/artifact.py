"""Static integrity checks for Kimi-K3 mixed MXFP4/EXL3 artifacts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from safetensors import safe_open

_EXPERT_RE = re.compile(
    r"^language_model\.model\.layers\.(\d+)\.block_sparse_moe\.experts\."
    r"(\d+)\.(w[123])\.(.+)$"
)
_MATRICES = ("w1", "w2", "w3")
_KEPT_SUFFIXES = ("weight_packed", "weight_scale")
_EXL3_SUFFIXES = ("exl3_suh", "exl3_svh", "exl3_trellis")
_TP_LOCAL_BUILD_VERSION = 1


@dataclass
class IssueCollector:
    sample_limit: int = 40
    count: int = 0
    samples: list[str] = field(default_factory=list)

    def add(self, message: str) -> None:
        self.count += 1
        if len(self.samples) < self.sample_limit:
            self.samples.append(message)


def _text_config(config: dict) -> dict:
    value = config.get("text_config", config)
    if not isinstance(value, dict):
        raise ValueError("config text_config is not an object")
    return value


def _quant_config(config: dict) -> dict:
    text = _text_config(config)
    value = text.get("quantization_config") or config.get("quantization_config")
    if not isinstance(value, dict):
        raise ValueError("serve config has no quantization_config")
    return value


def _expected_spec(
    matrix: str,
    suffix: str,
    *,
    hidden_size: int,
    intermediate_size: int,
) -> tuple[tuple[int, ...], str]:
    if matrix in {"w1", "w3"}:
        if suffix == "weight_packed":
            return (intermediate_size, hidden_size // 2), "U8"
        if suffix == "weight_scale":
            return (intermediate_size, hidden_size // 32), "U8"
        if suffix == "exl3_trellis":
            return (hidden_size // 16, intermediate_size // 16, 48), "I16"
        if suffix == "exl3_suh":
            return (hidden_size,), "F16"
        if suffix == "exl3_svh":
            return (intermediate_size,), "F16"
    elif matrix == "w2":
        if suffix == "weight_packed":
            return (hidden_size, intermediate_size // 2), "U8"
        if suffix == "weight_scale":
            return (hidden_size, intermediate_size // 32), "U8"
        if suffix == "exl3_trellis":
            return (intermediate_size // 16, hidden_size // 16, 48), "I16"
        if suffix == "exl3_suh":
            return (intermediate_size,), "F16"
        if suffix == "exl3_svh":
            return (hidden_size,), "F16"
    raise ValueError(f"unsupported expert tensor {matrix}.{suffix}")


def _expert_names(
    layer: int,
    expert: int,
    *,
    kept: bool,
) -> Iterator[str]:
    suffixes = _KEPT_SUFFIXES if kept else _EXL3_SUFFIXES
    base = f"language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}"
    for matrix in _MATRICES:
        for suffix in suffixes:
            yield f"{base}.{matrix}.{suffix}"


def _load_allocation(
    path: Path,
    *,
    expected_layers: list[int],
    num_experts: int,
    issues: IssueCollector,
) -> tuple[dict[int, tuple[set[int], set[int]]], dict]:
    document = json.loads(path.read_text())
    raw_layers = document.get("layers")
    if not isinstance(raw_layers, dict):
        raise ValueError("allocation-exl3.json has no layers object")
    expected_keys = {str(layer) for layer in expected_layers}
    actual_keys = set(raw_layers)
    for key in sorted(expected_keys - actual_keys):
        issues.add(f"allocation is missing layer {key}")
    for key in sorted(actual_keys - expected_keys):
        issues.add(f"allocation has unexpected layer {key}")

    allocation: dict[int, tuple[set[int], set[int]]] = {}
    universe = set(range(num_experts))
    for layer in expected_layers:
        value = raw_layers.get(str(layer))
        if not isinstance(value, dict):
            continue
        kept_list = value.get("keep")
        exl3_list = value.get("exl3")
        if not isinstance(kept_list, list) or not isinstance(exl3_list, list):
            issues.add(f"layer {layer}: keep/exl3 entries must be lists")
            continue
        try:
            kept = {int(item) for item in kept_list}
            exl3 = {int(item) for item in exl3_list}
        except (TypeError, ValueError):
            issues.add(f"layer {layer}: non-integer expert ID")
            continue
        if len(kept) != len(kept_list):
            issues.add(f"layer {layer}: duplicate kept expert ID")
        if len(exl3) != len(exl3_list):
            issues.add(f"layer {layer}: duplicate EXL3 expert ID")
        overlap = kept & exl3
        if overlap:
            issues.add(f"layer {layer}: {len(overlap)} experts occur in both tiers")
        covered = kept | exl3
        if covered != universe:
            missing = sorted(universe - covered)
            extra = sorted(covered - universe)
            issues.add(
                f"layer {layer}: partition mismatch; "
                f"missing={missing[:8]}, extra={extra[:8]}"
            )
        allocation[layer] = (kept, exl3)
    return allocation, document


def validate_exl3_artifact(
    artifact_dir: Path,
    serve_dir: Path,
) -> dict:
    """Validate artifact headers, allocation, config, and serve index.

    This intentionally reads safetensors headers rather than materializing
    payloads. It is a fast structural gate, not a replacement for the
    numerical kernel closures.
    """
    artifact_dir = artifact_dir.resolve()
    serve_dir = serve_dir.resolve()
    issues = IssueCollector()
    config = json.loads((serve_dir / "config.json").read_text())
    text = _text_config(config)
    hidden_size = int(text["routed_expert_hidden_size"])
    intermediate_size = int(text["moe_intermediate_size"])
    num_experts = int(text["num_experts"])
    num_hidden_layers = int(text["num_hidden_layers"])
    expected_layers = list(range(1, num_hidden_layers))
    if hidden_size % 32 or intermediate_size % 32:
        issues.add("routed hidden and intermediate sizes must both be divisible by 32")

    allocation, allocation_document = _load_allocation(
        artifact_dir / "allocation-exl3.json",
        expected_layers=expected_layers,
        num_experts=num_experts,
        issues=issues,
    )
    manifest = json.loads((artifact_dir / "kquant_exl3_manifest.json").read_text())
    if manifest.get("kind") != "kquant_exl3_artifact":
        issues.add("manifest kind is not kquant_exl3_artifact")
    if int(manifest.get("bits", -1)) != 3:
        issues.add("manifest bits is not 3")
    if manifest.get("codebook") != "mcg":
        issues.add("manifest codebook is not mcg")
    if not isinstance(manifest.get("mcg_mult"), int):
        issues.add("manifest mcg_mult is not an integer")
    tp_local = bool(manifest.get("tp_local_quantization", False))
    if tp_local:
        compatible_tp = manifest.get("compatible_tp_sizes")
        if not isinstance(compatible_tp, list) or len(compatible_tp) != 1:
            issues.add("TP-local manifest must declare one compatible TP size")
            compatible_tp = []
        tp_size = int(compatible_tp[0]) if compatible_tp else -1
        expected_local = intermediate_size // tp_size if tp_size > 0 else -1
        if tp_size <= 0 or intermediate_size % tp_size:
            issues.add(f"TP-local manifest has invalid TP size {tp_size}")
        if manifest.get("tp_local_intermediate_size") != expected_local:
            issues.add(
                "TP-local manifest intermediate mismatch: "
                f"{manifest.get('tp_local_intermediate_size')} != {expected_local}"
            )
        expected_blocks = [128, 64] if expected_local % 128 else [128]
        if manifest.get("intermediate_hadamard_blocks") != expected_blocks:
            issues.add("TP-local manifest Hadamard blocks do not match its geometry")
        if manifest.get("build_complete") is not True:
            issues.add("TP-local manifest is not marked build_complete")
        if manifest.get("build_version") != _TP_LOCAL_BUILD_VERSION:
            issues.add(
                f"TP-local manifest build_version is not {_TP_LOCAL_BUILD_VERSION}"
            )
        if manifest.get("shared_su") is not True:
            issues.add("TP-local manifest must declare shared_su=true")
        build_id = manifest.get("build_id")
        if not isinstance(build_id, str) or not build_id:
            issues.add("TP-local manifest has no build_id")
        else:
            marked_layers: set[int] = set()
            for marker_path in artifact_dir.glob(".exl3-layer-*.complete.json"):
                try:
                    marker = json.loads(marker_path.read_text())
                except (OSError, json.JSONDecodeError):
                    issues.add(f"invalid completion marker: {marker_path.name}")
                    continue
                if marker.get("build_id") != build_id:
                    continue
                layer = int(marker.get("layer", -1))
                shard = artifact_dir / f"exl3-layer-{layer:05d}.safetensors"
                errs = artifact_dir / f"exl3-layer-{layer:05d}.errs.json"
                expected_marker = {
                    "version": _TP_LOCAL_BUILD_VERSION,
                    "build_id": build_id,
                    "layer": layer,
                    "tp_size": tp_size,
                    "local_intermediate_size": expected_local,
                    "intermediate_hadamard_blocks": expected_blocks,
                    "bits": 3,
                    "codebook": "mcg",
                    "shared_su": True,
                    "shard_size": shard.stat().st_size if shard.is_file() else -1,
                    "errs_size": errs.stat().st_size if errs.is_file() else -1,
                }
                if marker != expected_marker:
                    issues.add(
                        f"layer {layer}: completion marker does not match files/contract"
                    )
                    continue
                if layer in marked_layers:
                    issues.add(f"layer {layer}: duplicate completion marker")
                    continue
                marked_layers.add(layer)
            missing_markers = set(expected_layers) - marked_layers
            if missing_markers:
                issues.add(
                    f"TP-local build is missing {len(missing_markers)} layer "
                    f"markers; sample={sorted(missing_markers)[:8]}"
                )

    total_kept = sum(len(value[0]) for value in allocation.values())
    total_exl3 = sum(len(value[1]) for value in allocation.values())
    meta = allocation_document.get("meta", {})
    if meta.get("n_keep") != total_kept:
        issues.add(f"allocation meta n_keep={meta.get('n_keep')} != {total_kept}")
    if meta.get("n_exl3") != total_exl3:
        issues.add(f"allocation meta n_exl3={meta.get('n_exl3')} != {total_exl3}")

    index = json.loads((serve_dir / "model.safetensors.index.json").read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("serve model has an invalid weight map")

    quant = _quant_config(config)
    bit_map = quant.get("hybrid_bit_map")
    if not isinstance(bit_map, dict):
        issues.add("quantization config has no hybrid_bit_map")
        bit_map = {}
    if quant.get("kept_format") != "mxfp4_e8m0k32":
        issues.add("quantization config kept_format is not mxfp4_e8m0k32")
    if quant.get("demoted_format") != "exl3_3":
        issues.add("quantization config demoted_format is not exl3_3")
    trellis = quant.get("trellis")
    if not isinstance(trellis, dict):
        issues.add("quantization config has no trellis object")
        trellis = {}
    tp_keys = (
        "compatible_tp_sizes",
        "tp_local_quantization",
        "tp_local_intermediate_size",
        "intermediate_hadamard_blocks",
    )
    if tp_local:
        for key in tp_keys:
            if trellis.get(key) != manifest.get(key):
                issues.add(f"trellis {key} differs from the TP-local manifest")
    elif trellis.get("tp_local_quantization"):
        issues.add("serve config is TP-local but the artifact manifest is not")

    expected_count = total_kept * 6 + total_exl3 * 9
    expected_files: set[str] = set()
    expected_names: set[str] = set()
    for layer, tiers in allocation.items():
        kept, exl3 = tiers
        configured = bit_map.get(str(layer))
        expected_bits = [4 if expert in kept else 3 for expert in range(num_experts)]
        if configured != expected_bits:
            issues.add(f"layer {layer}: hybrid_bit_map differs from allocation")
        for expert in sorted(kept):
            for name in _expert_names(layer, expert, kept=True):
                expected_names.add(name)
                filename = weight_map.get(name)
                if not isinstance(filename, str):
                    issues.add(f"serve index is missing {name}")
                else:
                    expected_files.add(filename)
        for expert in sorted(exl3):
            for name in _expert_names(layer, expert, kept=False):
                expected_names.add(name)
                filename = weight_map.get(name)
                if not isinstance(filename, str):
                    issues.add(f"serve index is missing {name}")
                else:
                    expected_files.add(filename)

    artifact_files = sorted(
        {
            *artifact_dir.glob("keep-mxfp4-*.safetensors"),
            *artifact_dir.glob("exl3-layer-*.safetensors"),
        }
    )
    actual_names: set[str] = set()
    scanned_tensor_count = 0
    for path in artifact_files:
        with safe_open(str(path), framework="pt") as handle:
            for name in handle.keys():
                scanned_tensor_count += 1
                if name in actual_names:
                    issues.add(f"duplicate tensor across artifact files: {name}")
                    continue
                actual_names.add(name)
                match = _EXPERT_RE.fullmatch(name)
                if match is None:
                    issues.add(f"unexpected artifact tensor name: {name}")
                    continue
                layer = int(match.group(1))
                expert = int(match.group(2))
                matrix = match.group(3)
                suffix = match.group(4)
                tiers = allocation.get(layer)
                if tiers is None:
                    issues.add(f"{name}: layer has no valid allocation")
                    continue
                kept, exl3 = tiers
                expected_suffixes: tuple[str, ...]
                if expert in kept:
                    expected_suffixes = _KEPT_SUFFIXES
                elif expert in exl3:
                    expected_suffixes = _EXL3_SUFFIXES
                else:
                    issues.add(f"{name}: expert has no valid tier")
                    continue
                if suffix not in expected_suffixes:
                    issues.add(f"{name}: suffix does not match assigned tier")
                    continue
                expected_shape, expected_dtype = _expected_spec(
                    matrix,
                    suffix,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                )
                tensor_slice = handle.get_slice(name)
                shape = tuple(int(value) for value in tensor_slice.get_shape())
                dtype = tensor_slice.get_dtype()
                if shape != expected_shape:
                    issues.add(f"{name}: shape {shape} != expected {expected_shape}")
                if dtype != expected_dtype:
                    issues.add(f"{name}: dtype {dtype} != expected {expected_dtype}")
                mapped_file = weight_map.get(name)
                if mapped_file != path.name:
                    issues.add(
                        f"{name}: serve index maps to {mapped_file!r}, "
                        f"artifact file is {path.name!r}"
                    )

    missing_headers = expected_names - actual_names
    unexpected_headers = actual_names - expected_names
    for name in sorted(missing_headers):
        issues.add(f"artifact header is missing {name}")
    for name in sorted(unexpected_headers):
        issues.add(f"artifact header has unexpected {name}")
    for filename in sorted(expected_files):
        if not (serve_dir / filename).is_file():
            issues.add(f"serve model is missing indexed artifact file {filename}")

    return {
        "schema_version": 1,
        "artifact_dir": str(artifact_dir),
        "serve_dir": str(serve_dir),
        "validation_scope": "allocation, config, index, and safetensors headers",
        "geometry": {
            "moe_layers": len(expected_layers),
            "num_experts": num_experts,
            "routed_hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
        },
        "allocation": {
            "kept": total_kept,
            "exl3": total_exl3,
            "assignments": total_kept + total_exl3,
        },
        "expected_expert_tensors": expected_count,
        "scanned_expert_tensors": scanned_tensor_count,
        "artifact_files": len(artifact_files),
        "issue_count": issues.count,
        "issue_samples": issues.samples,
        "pass": issues.count == 0 and scanned_tensor_count == expected_count,
    }
