"""Package a complete fixed-slab TP12 artifact for local vLLM serving."""

from __future__ import annotations

import json
import os
from pathlib import Path

from safetensors import safe_open

from kquant import constants as C
from kquant.exl3_reference import (
    CODEBOOK_MCG,
    CODEBOOK_MUL1_E4M3,
    CODEBOOK_SQG_NORMAL_E4M3,
    EXL3_CODEBOOKS,
    MCG_MULT,
    MUL1_MULT,
)
from kquant.io.hf_cache import resolve
from kquant.mixed_exl3 import FORMAT_MXFP4, SCHEMA, TP_SIZE
from kquant.pack.mixed_allocation import (
    ALLOCATION_KIND,
    ALLOCATION_SCHEMA_VERSION,
)
from kquant.pack.mixed_materialize import (
    MATERIALIZATION_BUILD_FILENAME,
    MATERIALIZED_ALLOCATION_FILENAME,
    MATERIALIZED_MANIFEST_FILENAME,
    MATERIALIZED_MODE_VALIDATION_SUMMARY_FILENAME,
    MATERIALIZED_PERFORMANCE_SUMMARY_FILENAME,
    MATERIALIZED_TEACHER_PROXY_SUMMARY_FILENAME,
    layer_filename,
)
from kquant.pack.mixed_validate import validate_materialized_artifact


SERVE_PACKAGE_KIND = "kquant_tp12_mixed_exl3_serve_directory"
SERVE_PACKAGE_SCHEMA_VERSION = 1
SERVE_PACKAGE_MANIFEST = "mixed_exl3_tp12_serve.json"
DEFAULT_NONEXPERT = Path("/models/Kimi-K3-mxfp8-nonexpert")
IGNORED_DENSE_LAYERS = (
    "kv_b_proj",
    "g_proj",
    "f_a_proj",
    "f_b_proj",
    "b_proj",
    "vision_tower",
    "mm_projector",
)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, document: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x") as handle:
            json.dump(document, handle, indent=1, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def hybrid_bit_map_from_allocation(allocation: dict) -> dict[str, list[int]]:
    """Derive the only serving-time tier map from validated format codes."""

    if allocation.get("kind") != ALLOCATION_KIND:
        raise ValueError("mixed-EXL allocation kind mismatch")
    if allocation.get("schema_version") != ALLOCATION_SCHEMA_VERSION:
        raise ValueError("mixed-EXL allocation schema mismatch")
    layers = allocation.get("layers")
    if not isinstance(layers, dict) or set(layers) != {
        str(layer) for layer in C.MOE_LAYERS
    }:
        raise ValueError("mixed-EXL allocation must contain exactly 92 MoE layers")
    result: dict[str, list[int]] = {}
    for layer in C.MOE_LAYERS:
        entry = layers[str(layer)]
        if not isinstance(entry, dict):
            raise ValueError(f"mixed-EXL layer {layer} allocation is malformed")
        codes = entry.get("format_codes")
        if not isinstance(codes, list) or len(codes) != C.NUM_EXPERTS:
            raise ValueError(
                f"mixed-EXL layer {layer} must contain {C.NUM_EXPERTS} codes"
            )
        bits: list[int] = []
        for code in codes:
            if isinstance(code, bool) or not isinstance(code, int):
                raise ValueError(f"mixed-EXL layer {layer} has a non-integer code")
            if code == FORMAT_MXFP4:
                bits.append(4)
            elif 0 <= code <= 0xCC and code >> 4 <= 12 and code & 0x0F <= 12:
                bits.append(3)
            else:
                raise ValueError(
                    f"mixed-EXL layer {layer} has invalid code 0x{code:02x}"
                )
        keep = entry.get("keep")
        compressed = entry.get("compressed")
        expected_keep = [expert for expert, bit in enumerate(bits) if bit == 4]
        expected_compressed = [expert for expert, bit in enumerate(bits) if bit == 3]
        if keep != expected_keep or compressed != expected_compressed:
            raise ValueError(
                f"mixed-EXL layer {layer} tiers disagree with its format codes"
            )
        result[str(layer)] = bits
    return result


def mixed_tp12_quantization_config(
    allocation: dict, *, codebook: str = CODEBOOK_MCG
) -> dict:
    if codebook not in EXL3_CODEBOOKS:
        raise ValueError("mixed-EXL codebook is unsupported")
    trellis = {
        "bits": 3,
        "codebook": codebook,
        "shared_su": True,
        "pair_format": "tp12_p24_p33",
    }
    if codebook == CODEBOOK_MCG:
        trellis["mcg_mult"] = MCG_MULT
    elif codebook == CODEBOOK_MUL1_E4M3:
        trellis["mul1_mult"] = MUL1_MULT
        trellis["reconstruction_dtype"] = "e4m3"
    elif codebook == CODEBOOK_SQG_NORMAL_E4M3:
        trellis["labelling"] = "sqg-l16-normal-r44-v1"
        trellis["reconstruction_dtype"] = "e4m3"
        trellis["rate_dependent_reconstruction"] = True
    return {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "group_size": 16,
        "hybrid_bit_map": hybrid_bit_map_from_allocation(allocation),
        "kept_format": "mxfp4_e8m0k32",
        "demoted_format": "mixed_exl3_tp12",
        "mixed_exl3_tp12": {
            "schema": SCHEMA,
            "tp_size": TP_SIZE,
            "layer_file_pattern": "mixed-exl3-tp12-layer-{layer:05d}.bin",
        },
        "trellis": trellis,
        "dense_format": "mxfp8",
        "ignored_layers": list(IGNORED_DENSE_LAYERS),
    }


def _link(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.symlink_to(source.resolve(strict=True))


def _scan_nonexpert_weight_map(source: Path) -> tuple[dict[str, str], int, list[str]]:
    files = sorted(source.glob("00-nonexpert-*.safetensors"))
    if not files:
        raise ValueError(f"no MXFP8 non-expert shards found in {source}")
    weight_map: dict[str, str] = {}
    total = 0
    names: list[str] = []
    for path in files:
        names.append(path.name)
        total += path.stat().st_size
        with safe_open(path, framework="pt", device="cpu") as handle:
            for tensor_name in handle.keys():
                if ".block_sparse_moe.experts." in tensor_name:
                    raise ValueError(
                        f"non-expert overlay contains routed expert {tensor_name}"
                    )
                if tensor_name in weight_map:
                    raise ValueError(
                        f"non-expert tensor appears in multiple shards: {tensor_name}"
                    )
                weight_map[tensor_name] = path.name
    if not weight_map:
        raise ValueError("MXFP8 non-expert overlay contains no tensors")
    return weight_map, total, names


def _nonexpert_weight_map(
    source: Path, destination: Path
) -> tuple[dict[str, str], int, list[str]]:
    weight_map, total, names = _scan_nonexpert_weight_map(source)
    for name in names:
        _link(source / name, destination / name)
    return weight_map, total, names


def _auxiliary_sources(snapshot: Path) -> list[Path]:
    excluded = {
        "config.json",
        "model.safetensors.index.json",
        "hf_quant_config.json",
        "quant_config.json",
    }
    return [
        source
        for source in sorted(snapshot.iterdir())
        if source.name not in excluded and source.suffix != ".safetensors"
    ]


def _require_exact_link(link: Path, source: Path) -> None:
    if not link.is_symlink():
        raise ValueError(f"serve package entry is not a symlink: {link}")
    if link.resolve(strict=True) != source.resolve(strict=True):
        raise ValueError(f"serve package link {link.name} has the wrong target")


def validate_mixed_tp12_serve_directory(root: str | Path) -> dict:
    """Independently close a published directory against its named sources."""

    root = Path(root).resolve()
    package_path = root / SERVE_PACKAGE_MANIFEST
    if not package_path.is_file():
        raise ValueError(f"mixed-EXL serve package is missing {package_path}")
    package = _read_json(package_path)
    expected_scalars = {
        "kind": SERVE_PACKAGE_KIND,
        "schema_version": SERVE_PACKAGE_SCHEMA_VERSION,
        "tp_size": TP_SIZE,
        "layers": C.NUM_MOE_LAYERS,
        "hybrid_layers": C.NUM_MOE_LAYERS,
    }
    for name, expected in expected_scalars.items():
        if package.get(name) != expected:
            raise ValueError(
                f"mixed-EXL serve package {name} mismatch: "
                f"{package.get(name)!r} != {expected!r}"
            )

    artifact = Path(str(package.get("artifact", ""))).resolve()
    nonexpert_source = Path(str(package.get("nonexpert_source", ""))).resolve()
    official_snapshot = Path(str(package.get("official_snapshot", ""))).resolve()
    if not artifact.is_dir() or not official_snapshot.is_dir():
        raise ValueError("mixed-EXL serve package names an unavailable source")
    artifact_validation = package.get("artifact_validation")
    if (
        not isinstance(artifact_validation, dict)
        or Path(str(artifact_validation.get("artifact", ""))).resolve() != artifact
        or artifact_validation.get("complete") is not True
    ):
        raise ValueError("mixed-EXL serve package lacks a complete artifact verdict")
    current_artifact_validation = validate_materialized_artifact(
        artifact,
        validate_candidate_payload_headers=True,
        verify_payloads=False,
    )
    if artifact_validation != current_artifact_validation:
        raise ValueError(
            "mixed-EXL serve package artifact verdict is stale or inconsistent"
        )

    slab_names = [layer_filename(layer) for layer in C.MOE_LAYERS]
    if package.get("slabs") != slab_names:
        raise ValueError("mixed-EXL serve package slab inventory is not canonical")
    for name in slab_names:
        _require_exact_link(root / name, artifact / name)
    metadata_names = (
        MATERIALIZATION_BUILD_FILENAME,
        MATERIALIZED_ALLOCATION_FILENAME,
        MATERIALIZED_MANIFEST_FILENAME,
        MATERIALIZED_MODE_VALIDATION_SUMMARY_FILENAME,
        MATERIALIZED_TEACHER_PROXY_SUMMARY_FILENAME,
        MATERIALIZED_PERFORMANCE_SUMMARY_FILENAME,
    )
    for name in metadata_names:
        _require_exact_link(root / name, artifact / name)

    allocation = _read_json(root / MATERIALIZED_ALLOCATION_FILENAME)
    artifact_manifest = _read_json(root / MATERIALIZED_MANIFEST_FILENAME)
    expected_quantization = mixed_tp12_quantization_config(
        allocation,
        codebook=str(artifact_manifest.get("candidate_codebook", CODEBOOK_MCG)),
    )
    config = _read_json(root / "config.json")
    text = config.get("text_config")
    if (
        not isinstance(text, dict)
        or text.get("quantization_config") != expected_quantization
        or "quantization_config" in config
    ):
        raise ValueError("mixed-EXL serve quantization config does not close")

    expected_map, total_bytes, nonexpert_names = _scan_nonexpert_weight_map(
        nonexpert_source
    )
    if package.get("nonexpert_shards") != nonexpert_names:
        raise ValueError("mixed-EXL serve non-expert shard list does not close")
    if package.get("indexed_tensors") != len(expected_map):
        raise ValueError("mixed-EXL serve indexed tensor count does not close")
    if package.get("indexed_safetensors_bytes") != total_bytes:
        raise ValueError("mixed-EXL serve indexed byte count does not close")
    for name in nonexpert_names:
        _require_exact_link(root / name, nonexpert_source / name)
    index = _read_json(root / "model.safetensors.index.json")
    if index != {
        "metadata": {"total_size": total_bytes},
        "weight_map": dict(sorted(expected_map.items())),
    }:
        raise ValueError("mixed-EXL serve safetensors index does not close")

    auxiliary = _auxiliary_sources(official_snapshot)
    auxiliary_names = [source.name for source in auxiliary]
    if package.get("auxiliary_files") != auxiliary_names:
        raise ValueError("mixed-EXL serve auxiliary file list does not close")
    for source in auxiliary:
        _require_exact_link(root / source.name, source)

    expected_names = {
        *slab_names,
        *metadata_names,
        *nonexpert_names,
        *auxiliary_names,
        "config.json",
        "model.safetensors.index.json",
        SERVE_PACKAGE_MANIFEST,
    }
    actual_names = {path.name for path in root.iterdir()}
    if actual_names != expected_names:
        raise ValueError(
            "mixed-EXL serve file inventory does not close; "
            f"missing={sorted(expected_names - actual_names)[:3]}, "
            f"extra={sorted(actual_names - expected_names)[:3]}"
        )
    return {
        "kind": SERVE_PACKAGE_KIND,
        "schema_version": SERVE_PACKAGE_SCHEMA_VERSION,
        "serve_directory": str(root),
        "complete": True,
        "tp_size": TP_SIZE,
        "layers": len(slab_names),
        "indexed_tensors": len(expected_map),
        "indexed_safetensors_bytes": total_bytes,
    }


def package_mixed_tp12_serve_directory(
    artifact: str | Path,
    destination: str | Path,
    *,
    nonexpert_source: str | Path = DEFAULT_NONEXPERT,
    official_snapshot: str | Path | None = None,
) -> dict:
    """Atomically publish a fresh symlink-based TP12 serve directory."""

    artifact = Path(artifact).resolve()
    destination = Path(destination).resolve()
    nonexpert_source = Path(nonexpert_source).resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    artifact_validation = validate_materialized_artifact(
        artifact,
        validate_candidate_payload_headers=True,
        verify_payloads=False,
    )
    allocation = _read_json(artifact / MATERIALIZED_ALLOCATION_FILENAME)
    artifact_manifest = _read_json(artifact / MATERIALIZED_MANIFEST_FILENAME)
    bit_map = hybrid_bit_map_from_allocation(allocation)
    snapshot = (
        Path(official_snapshot).resolve()
        if official_snapshot is not None
        else Path(resolve().snapshot_dir).resolve()
    )
    config_path = snapshot / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(temporary)
    temporary.mkdir()
    try:
        slab_names = []
        for layer in C.MOE_LAYERS:
            name = layer_filename(layer)
            _link(artifact / name, temporary / name)
            slab_names.append(name)
        for name in (
            MATERIALIZATION_BUILD_FILENAME,
            MATERIALIZED_ALLOCATION_FILENAME,
            MATERIALIZED_MANIFEST_FILENAME,
            MATERIALIZED_MODE_VALIDATION_SUMMARY_FILENAME,
            MATERIALIZED_TEACHER_PROXY_SUMMARY_FILENAME,
            MATERIALIZED_PERFORMANCE_SUMMARY_FILENAME,
        ):
            _link(artifact / name, temporary / name)

        weight_map, safetensors_bytes, nonexpert_names = _nonexpert_weight_map(
            nonexpert_source, temporary
        )
        index = {
            "metadata": {"total_size": safetensors_bytes},
            "weight_map": dict(sorted(weight_map.items())),
        }
        _atomic_json(temporary / "model.safetensors.index.json", index)

        config = _read_json(config_path)
        text = config.get("text_config")
        if not isinstance(text, dict):
            raise ValueError("official K3 config has no text_config object")
        text["quantization_config"] = mixed_tp12_quantization_config(
            allocation,
            codebook=str(
                artifact_manifest.get("candidate_codebook", CODEBOOK_MCG)
            ),
        )
        config.pop("quantization_config", None)
        _atomic_json(temporary / "config.json", config)

        auxiliary = _auxiliary_sources(snapshot)
        for source in auxiliary:
            _link(source, temporary / source.name)

        package = {
            "kind": SERVE_PACKAGE_KIND,
            "schema_version": SERVE_PACKAGE_SCHEMA_VERSION,
            "artifact": str(artifact),
            "artifact_validation": artifact_validation,
            "tp_size": TP_SIZE,
            "layers": len(slab_names),
            "slabs": slab_names,
            "hybrid_layers": len(bit_map),
            "official_snapshot": str(snapshot),
            "auxiliary_files": [source.name for source in auxiliary],
            "nonexpert_source": str(nonexpert_source),
            "nonexpert_shards": nonexpert_names,
            "indexed_tensors": len(weight_map),
            "indexed_safetensors_bytes": safetensors_bytes,
        }
        _atomic_json(temporary / SERVE_PACKAGE_MANIFEST, package)
        validate_mixed_tp12_serve_directory(temporary)
        _fsync_directory(temporary)
        temporary.replace(destination)
        _fsync_directory(destination.parent)
    except BaseException:
        if temporary.is_dir():
            for path in temporary.iterdir():
                path.unlink()
            temporary.rmdir()
        raise
    return package


__all__ = [
    "DEFAULT_NONEXPERT",
    "IGNORED_DENSE_LAYERS",
    "SERVE_PACKAGE_KIND",
    "SERVE_PACKAGE_MANIFEST",
    "SERVE_PACKAGE_SCHEMA_VERSION",
    "hybrid_bit_map_from_allocation",
    "mixed_tp12_quantization_config",
    "package_mixed_tp12_serve_directory",
    "validate_mixed_tp12_serve_directory",
]
