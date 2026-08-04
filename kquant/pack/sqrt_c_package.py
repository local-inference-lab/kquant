"""Package a complete TP12 SQRT-C artifact for local vLLM serving."""

from __future__ import annotations

import os
from pathlib import Path

from kquant import constants as C
from kquant.exl3_reference import CODEBOOK_SQG_NORMAL_E4M3
from kquant.io.hf_cache import resolve
from kquant.mixed_exl3 import (
    FORMAT_MXFP4,
    KEEP_STORAGE_EXTERNAL_X4,
    PHASE1_MODE_IDS,
    SCHEMA,
    SQRT_C_LAYER_HEADER_VERSION,
    TP_SIZE,
    ExpertFormatSpec,
)
from kquant.pack.mixed_package import (
    DEFAULT_NONEXPERT,
    IGNORED_DENSE_LAYERS,
    _atomic_json,
    _auxiliary_sources,
    _fsync_directory,
    _link,
    _nonexpert_weight_map,
    _read_json,
    _require_exact_link,
    _scan_nonexpert_weight_map,
)
from kquant.pack.mixed_materialize import layer_filename
from kquant.pack.sqrt_c_allocation import (
    SQRT_C_ALLOCATION_KIND,
    SQRT_C_ALLOCATION_SCHEMA_VERSION,
)
from kquant.pack.sqrt_c_materialize import (
    SQRT_C_ALLOCATION_COPY_FILENAME,
    SQRT_C_BUILD_FILENAME,
    SQRT_C_CANDIDATE_COMPLETION_COPY_FILENAME,
    SQRT_C_CANDIDATE_MANIFEST_COPY_FILENAME,
    SQRT_C_MANIFEST_FILENAME,
    SQRT_C_X4_COMPLETION_COPY_FILENAME,
    SQRT_C_X4_MANIFEST_COPY_FILENAME,
    sqrt_c_layer_closure_filename,
)
from kquant.pack.sqrt_c_validate import validate_sqrt_c_artifact
from kquant.x4 import X4_VERSION, x4_layer_path


SQRT_C_SERVE_KIND = "kquant_kimi_k3_sqrt_c_tp12_serve_directory"
SQRT_C_SERVE_SCHEMA_VERSION = 1
SQRT_C_SERVE_MANIFEST = "sqrt_c_tp12_serve.json"

SQRT_C_METADATA_NAMES = (
    SQRT_C_BUILD_FILENAME,
    SQRT_C_ALLOCATION_COPY_FILENAME,
    SQRT_C_MANIFEST_FILENAME,
    SQRT_C_CANDIDATE_MANIFEST_COPY_FILENAME,
    SQRT_C_CANDIDATE_COMPLETION_COPY_FILENAME,
    SQRT_C_X4_MANIFEST_COPY_FILENAME,
    SQRT_C_X4_COMPLETION_COPY_FILENAME,
)


def sqrt_c_hybrid_bit_map(allocation: dict) -> dict[str, list[int]]:
    """Derive the serving tier map from the authenticated format codes."""

    if allocation.get("kind") != SQRT_C_ALLOCATION_KIND:
        raise ValueError("SQRT-C allocation kind mismatch")
    if allocation.get("schema_version") != SQRT_C_ALLOCATION_SCHEMA_VERSION:
        raise ValueError("SQRT-C allocation schema mismatch")
    meta = allocation.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("SQRT-C allocation has no meta object")
    expected_meta = {
        "codec": "SQRT-C",
        "tp_size": TP_SIZE,
        "high_tier_storage": KEEP_STORAGE_EXTERNAL_X4,
        "candidate_codebook": CODEBOOK_SQG_NORMAL_E4M3,
        "candidate_mode_ids": list(PHASE1_MODE_IDS),
    }
    for name, expected in expected_meta.items():
        if meta.get(name) != expected:
            raise ValueError(f"SQRT-C allocation {name} drifted")
    layers = allocation.get("layers")
    if not isinstance(layers, dict) or set(layers) != {
        str(layer) for layer in C.MOE_LAYERS
    }:
        raise ValueError("SQRT-C allocation must contain exactly 92 MoE layers")

    result: dict[str, list[int]] = {}
    for layer in C.MOE_LAYERS:
        entry = layers[str(layer)]
        if not isinstance(entry, dict):
            raise ValueError(f"SQRT-C layer {layer} allocation is malformed")
        codes = entry.get("format_codes")
        if not isinstance(codes, list) or len(codes) != C.NUM_EXPERTS:
            raise ValueError(
                f"SQRT-C layer {layer} must contain {C.NUM_EXPERTS} codes"
            )
        bits: list[int] = []
        for raw_code in codes:
            if isinstance(raw_code, bool) or not isinstance(raw_code, int):
                raise ValueError(f"SQRT-C layer {layer} has a non-integer code")
            if raw_code == FORMAT_MXFP4:
                bits.append(4)
                continue
            format_spec = ExpertFormatSpec.from_code(raw_code)
            if (
                format_spec.is_mxfp4
                or format_spec.r13 not in PHASE1_MODE_IDS
                or format_spec.r2 not in PHASE1_MODE_IDS
            ):
                raise ValueError(
                    f"SQRT-C layer {layer} has invalid code 0x{raw_code:02x}"
                )
            bits.append(3)
        expected_x4 = [expert for expert, bit in enumerate(bits) if bit == 4]
        expected_compressed = [
            expert for expert, bit in enumerate(bits) if bit == 3
        ]
        if (
            entry.get("x4") != expected_x4
            or entry.get("keep") != expected_x4
            or entry.get("compressed") != expected_compressed
        ):
            raise ValueError(
                f"SQRT-C layer {layer} tiers disagree with its format codes"
            )
        result[str(layer)] = bits
    return result


def sqrt_c_tp12_quantization_config(allocation: dict) -> dict:
    """Emit the explicit V5/SQG/X4 runtime contract."""

    return {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "group_size": 16,
        "hybrid_bit_map": sqrt_c_hybrid_bit_map(allocation),
        "kept_format": "mxfp4_e8m0k32",
        "kept_storage": KEEP_STORAGE_EXTERNAL_X4,
        "demoted_format": "mixed_exl3_tp12",
        "mixed_exl3_tp12": {
            "schema": SCHEMA,
            "layer_header_version": SQRT_C_LAYER_HEADER_VERSION,
            "tp_size": TP_SIZE,
            "layer_file_pattern": "mixed-exl3-tp12-layer-{layer:05d}.bin",
            "x4_layer_file_pattern": "x4-layer-{layer:05d}.bin",
            "x4_version": X4_VERSION,
        },
        "trellis": {
            "bits": 3,
            "codebook": CODEBOOK_SQG_NORMAL_E4M3,
            "labelling": "sqg-l16-normal-r44-v1",
            "reconstruction_dtype": "e4m3",
            "rate_dependent_reconstruction": True,
            "shared_su": True,
            "pair_format": "tp12_p24_p33",
            "mode_ids": list(PHASE1_MODE_IDS),
            "separate_r13_r2": True,
        },
        "dense_format": "mxfp8",
        "ignored_layers": list(IGNORED_DENSE_LAYERS),
    }


def _artifact_file_names() -> tuple[list[str], list[str], list[str]]:
    slabs = [layer_filename(layer) for layer in C.MOE_LAYERS]
    sidecars = [x4_layer_path(Path("."), layer).name for layer in C.MOE_LAYERS]
    receipts = [sqrt_c_layer_closure_filename(layer) for layer in C.MOE_LAYERS]
    return slabs, sidecars, receipts


def validate_sqrt_c_tp12_serve_directory(root: str | Path) -> dict:
    """Independently close a serve directory against its named sources."""

    root = Path(root).resolve()
    package_path = root / SQRT_C_SERVE_MANIFEST
    if not package_path.is_file():
        raise ValueError(f"SQRT-C serve package is missing {package_path}")
    package = _read_json(package_path)
    expected_scalars = {
        "kind": SQRT_C_SERVE_KIND,
        "schema_version": SQRT_C_SERVE_SCHEMA_VERSION,
        "tp_size": TP_SIZE,
        "layers": C.NUM_MOE_LAYERS,
        "hybrid_layers": C.NUM_MOE_LAYERS,
    }
    for name, expected in expected_scalars.items():
        if package.get(name) != expected:
            raise ValueError(f"SQRT-C serve package {name} mismatch")

    artifact = Path(str(package.get("artifact", ""))).resolve()
    nonexpert_source = Path(str(package.get("nonexpert_source", ""))).resolve()
    official_snapshot = Path(str(package.get("official_snapshot", ""))).resolve()
    if not artifact.is_dir() or not official_snapshot.is_dir():
        raise ValueError("SQRT-C serve package names an unavailable source")
    artifact_validation = package.get("artifact_validation")
    if (
        not isinstance(artifact_validation, dict)
        or Path(str(artifact_validation.get("artifact", ""))).resolve()
        != artifact
        or artifact_validation.get("complete") is not True
    ):
        raise ValueError("SQRT-C serve package lacks a complete artifact verdict")
    current_validation = validate_sqrt_c_artifact(
        artifact,
        validate_candidate_payload_headers=True,
        verify_payloads=False,
    )
    if artifact_validation != current_validation:
        raise ValueError("SQRT-C serve artifact verdict is stale or inconsistent")

    slabs, sidecars, receipts = _artifact_file_names()
    inventories = {
        "slabs": slabs,
        "x4_sidecars": sidecars,
        "closure_receipts": receipts,
        "artifact_metadata": list(SQRT_C_METADATA_NAMES),
    }
    for field, names in inventories.items():
        if package.get(field) != names:
            raise ValueError(f"SQRT-C serve {field} inventory is not canonical")
        for name in names:
            _require_exact_link(root / name, artifact / name)

    allocation = _read_json(root / SQRT_C_ALLOCATION_COPY_FILENAME)
    expected_quantization = sqrt_c_tp12_quantization_config(allocation)
    config = _read_json(root / "config.json")
    text = config.get("text_config")
    if (
        not isinstance(text, dict)
        or text.get("quantization_config") != expected_quantization
        or "quantization_config" in config
    ):
        raise ValueError("SQRT-C serve quantization config does not close")

    expected_map, total_bytes, nonexpert_names = _scan_nonexpert_weight_map(
        nonexpert_source
    )
    if package.get("nonexpert_shards") != nonexpert_names:
        raise ValueError("SQRT-C serve non-expert shard list does not close")
    if package.get("indexed_tensors") != len(expected_map):
        raise ValueError("SQRT-C serve indexed tensor count does not close")
    if package.get("indexed_safetensors_bytes") != total_bytes:
        raise ValueError("SQRT-C serve indexed byte count does not close")
    for name in nonexpert_names:
        _require_exact_link(root / name, nonexpert_source / name)
    index = _read_json(root / "model.safetensors.index.json")
    if index != {
        "metadata": {"total_size": total_bytes},
        "weight_map": dict(sorted(expected_map.items())),
    }:
        raise ValueError("SQRT-C serve safetensors index does not close")

    auxiliary = _auxiliary_sources(official_snapshot)
    auxiliary_names = [source.name for source in auxiliary]
    if package.get("auxiliary_files") != auxiliary_names:
        raise ValueError("SQRT-C serve auxiliary file list does not close")
    for source in auxiliary:
        _require_exact_link(root / source.name, source)

    expected_names = {
        *slabs,
        *sidecars,
        *receipts,
        *SQRT_C_METADATA_NAMES,
        *nonexpert_names,
        *auxiliary_names,
        "config.json",
        "model.safetensors.index.json",
        SQRT_C_SERVE_MANIFEST,
    }
    actual_names = {path.name for path in root.iterdir()}
    if actual_names != expected_names:
        raise ValueError(
            "SQRT-C serve file inventory does not close; "
            f"missing={sorted(expected_names - actual_names)[:3]}, "
            f"extra={sorted(actual_names - expected_names)[:3]}"
        )
    return {
        "kind": SQRT_C_SERVE_KIND,
        "schema_version": SQRT_C_SERVE_SCHEMA_VERSION,
        "serve_directory": str(root),
        "complete": True,
        "tp_size": TP_SIZE,
        "layers": len(slabs),
        "x4_sidecars": len(sidecars),
        "indexed_tensors": len(expected_map),
        "indexed_safetensors_bytes": total_bytes,
    }


def package_sqrt_c_tp12_serve_directory(
    artifact: str | Path,
    destination: str | Path,
    *,
    nonexpert_source: str | Path = DEFAULT_NONEXPERT,
    official_snapshot: str | Path | None = None,
) -> dict:
    """Atomically publish a fresh symlink-based SQRT-C serve directory."""

    artifact = Path(artifact).resolve()
    destination = Path(destination).resolve()
    nonexpert_source = Path(nonexpert_source).resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    artifact_validation = validate_sqrt_c_artifact(
        artifact,
        validate_candidate_payload_headers=True,
        verify_payloads=False,
    )
    allocation = _read_json(artifact / SQRT_C_ALLOCATION_COPY_FILENAME)
    bit_map = sqrt_c_hybrid_bit_map(allocation)
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
        slabs, sidecars, receipts = _artifact_file_names()
        for name in (*slabs, *sidecars, *receipts, *SQRT_C_METADATA_NAMES):
            _link(artifact / name, temporary / name)

        weight_map, safetensors_bytes, nonexpert_names = _nonexpert_weight_map(
            nonexpert_source,
            temporary,
        )
        _atomic_json(
            temporary / "model.safetensors.index.json",
            {
                "metadata": {"total_size": safetensors_bytes},
                "weight_map": dict(sorted(weight_map.items())),
            },
        )

        config = _read_json(config_path)
        text = config.get("text_config")
        if not isinstance(text, dict):
            raise ValueError("official K3 config has no text_config object")
        text["quantization_config"] = sqrt_c_tp12_quantization_config(allocation)
        config.pop("quantization_config", None)
        _atomic_json(temporary / "config.json", config)

        auxiliary = _auxiliary_sources(snapshot)
        for source in auxiliary:
            _link(source, temporary / source.name)

        package = {
            "kind": SQRT_C_SERVE_KIND,
            "schema_version": SQRT_C_SERVE_SCHEMA_VERSION,
            "artifact": str(artifact),
            "artifact_validation": artifact_validation,
            "tp_size": TP_SIZE,
            "layers": len(slabs),
            "slabs": slabs,
            "x4_sidecars": sidecars,
            "closure_receipts": receipts,
            "artifact_metadata": list(SQRT_C_METADATA_NAMES),
            "hybrid_layers": len(bit_map),
            "official_snapshot": str(snapshot),
            "auxiliary_files": [source.name for source in auxiliary],
            "nonexpert_source": str(nonexpert_source),
            "nonexpert_shards": nonexpert_names,
            "indexed_tensors": len(weight_map),
            "indexed_safetensors_bytes": safetensors_bytes,
        }
        _atomic_json(temporary / SQRT_C_SERVE_MANIFEST, package)
        validate_sqrt_c_tp12_serve_directory(temporary)
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
    "SQRT_C_SERVE_KIND",
    "SQRT_C_SERVE_MANIFEST",
    "SQRT_C_SERVE_SCHEMA_VERSION",
    "package_sqrt_c_tp12_serve_directory",
    "sqrt_c_hybrid_bit_map",
    "sqrt_c_tp12_quantization_config",
    "validate_sqrt_c_tp12_serve_directory",
]
