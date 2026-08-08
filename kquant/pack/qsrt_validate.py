"""Independent structural and source validation for TP-free QSRT artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from kquant import constants as C
from kquant.exl3_reference import QSRT_CODEBOOKS
from kquant.qsrt import (
    EXPERT_TRELLIS_BYTES,
    INTERMEDIATE_CHANNELS,
    LATENT_CHANNELS,
    PHASE1_MODE_IDS,
    SCALE_BYTES,
)
from kquant.pack.qsrt_atoms import layer_filename
from kquant.pack.qsrt_pool import load_qsrt_candidate_pool
from kquant.pack.qsrt_materialize import (
    QSRT_ALLOCATION_COPY_FILENAME,
    QSRT_ARTIFACT_KIND,
    QSRT_ARTIFACT_SCHEMA_VERSION,
    QSRT_BUILD_FILENAME,
    QSRT_BUILD_KIND,
    QSRT_BUILD_SCHEMA_VERSION,
    QSRT_CANDIDATE_COMPLETION_COPY_FILENAME,
    QSRT_CANDIDATE_MANIFEST_COPY_FILENAME,
    QSRT_MANIFEST_FILENAME,
    QSRT_X4T_COMPLETION_COPY_FILENAME,
    QSRT_X4T_MANIFEST_COPY_FILENAME,
    load_qsrt_layer_closure_receipt,
    qsrt_layer_closure_filename,
    validate_qsrt_layer_payloads,
    validate_qsrt_materialization_allocation,
)
from kquant.pack.x4t_index import (
    X4T_COST_COMPLETION_FILENAME,
    X4T_COST_MANIFEST_FILENAME,
    load_x4t_cost_index,
)
from kquant.source_weights import OfficialMXFP4Store
from kquant.x4t import x4t_layer_path


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(document: dict) -> str:
    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_copy(copy: Path, source: Path, *, name: str) -> None:
    if not copy.is_file() or not source.is_file():
        raise ValueError(f"QSRT {name} metadata is unavailable")
    if copy.read_bytes() != source.read_bytes():
        raise ValueError(f"QSRT {name} metadata copy drifted")


def _expected_logical_source_bytes(spec) -> int:
    compressed_expert_bytes = EXPERT_TRELLIS_BYTES + 3 * (
        LATENT_CHANNELS + INTERMEDIATE_CHANNELS
    ) * SCALE_BYTES
    source_matrix_bytes = (
        INTERMEDIATE_CHANNELS * LATENT_CHANNELS // 2
        + INTERMEDIATE_CHANNELS * LATENT_CHANNELS // 32
    )
    return (
        len(spec.compressed) * compressed_expert_bytes
        + len(spec.x4t) * 3 * source_matrix_bytes
    )


def validate_qsrt_artifact(
    root: str | Path,
    *,
    validate_candidate_payload_headers: bool = True,
    verify_payloads: bool = False,
    official_repo_dir: str | Path | None = None,
) -> dict:
    """Validate provenance, all layer pairs, and optionally every source byte."""

    root = Path(root).resolve()
    build_path = root / QSRT_BUILD_FILENAME
    allocation_path = root / QSRT_ALLOCATION_COPY_FILENAME
    manifest_path = root / QSRT_MANIFEST_FILENAME
    copied_paths = (
        root / QSRT_CANDIDATE_MANIFEST_COPY_FILENAME,
        root / QSRT_CANDIDATE_COMPLETION_COPY_FILENAME,
        root / QSRT_X4T_MANIFEST_COPY_FILENAME,
        root / QSRT_X4T_COMPLETION_COPY_FILENAME,
    )
    for path in (build_path, allocation_path, manifest_path, *copied_paths):
        if not path.is_file():
            raise ValueError(f"QSRT artifact is incomplete; missing {path}")
    partials = sorted(path.name for path in root.glob(".*.partial"))
    if partials:
        raise ValueError(f"QSRT artifact contains partial files: {partials[:3]}")

    build = _read_json(build_path)
    expected_build = {
        "kind": QSRT_BUILD_KIND,
        "schema_version": QSRT_BUILD_SCHEMA_VERSION,
        "codec": "QSRT",
        "source_model": C.MODEL_ID,
        "source_revision": C.REVISION,
        "layer_count": C.NUM_MOE_LAYERS,
        "byte_order": "little",
    }
    for name, expected in expected_build.items():
        if build.get(name) != expected:
            raise ValueError(
                f"QSRT build {name} mismatch: {build.get(name)!r} != {expected!r}"
            )
    codec_contract = build.get("codec_contract")
    if not isinstance(codec_contract, dict):
        raise ValueError("QSRT build codec contract is missing")
    codebook = codec_contract.get("trellis_codebook")
    if codebook not in QSRT_CODEBOOKS:
        raise ValueError("QSRT build trellis codebook is unsupported")
    expected_contract = {
        "trellis_codebook": codebook,
        "trellis_modes": list(PHASE1_MODE_IDS),
        "separate_r13_r2": True,
        "high_tier": "X4T",
    }
    if build.get("codec_contract") != expected_contract:
        raise ValueError("QSRT build codec contract drifted")

    candidate_root = Path(str(build.get("candidate_pool", ""))).resolve()
    x4t_root = Path(str(build.get("x4t_cost_index", ""))).resolve()
    candidate_manifest = candidate_root / "qsrt-candidate-manifest.json"
    candidate_completion = candidate_root / "qsrt-candidate-completion.json"
    x4t_manifest = x4t_root / X4T_COST_MANIFEST_FILENAME
    x4t_completion = x4t_root / X4T_COST_COMPLETION_FILENAME
    source_metadata = (
        (
            root / QSRT_CANDIDATE_MANIFEST_COPY_FILENAME,
            candidate_manifest,
            "candidate manifest",
            "candidate_manifest_sha256",
        ),
        (
            root / QSRT_CANDIDATE_COMPLETION_COPY_FILENAME,
            candidate_completion,
            "candidate completion",
            "candidate_completion_sha256",
        ),
        (
            root / QSRT_X4T_MANIFEST_COPY_FILENAME,
            x4t_manifest,
            "X4T cost manifest",
            "x4t_manifest_sha256",
        ),
        (
            root / QSRT_X4T_COMPLETION_COPY_FILENAME,
            x4t_completion,
            "X4T cost completion",
            "x4t_completion_sha256",
        ),
    )
    for copy, source, description, hash_field in source_metadata:
        _require_copy(copy, source, name=description)
        if build.get(hash_field) != _sha256(source):
            raise ValueError(f"QSRT build {description} hash drifted")
    if build.get("allocation_sha256") != _sha256(allocation_path):
        raise ValueError("QSRT build allocation hash drifted")

    pool = load_qsrt_candidate_pool(
        candidate_root,
        validate_payload_headers=validate_candidate_payload_headers,
    )
    if pool.codebook != codebook:
        raise ValueError("QSRT build codebook disagrees with its candidate pool")
    x4t_index = load_x4t_cost_index(x4t_root)
    if build.get("candidate_pool_content_sha256") != pool.content_sha256:
        raise ValueError("QSRT candidate pool content digest drifted")
    if build.get("x4t_cost_index_content_sha256") != x4t_index.content_sha256:
        raise ValueError("QSRT X4T cost index content digest drifted")
    allocation = _read_json(allocation_path)
    plan = validate_qsrt_materialization_allocation(allocation, pool, x4t_index)
    allocation_meta = allocation["meta"]
    expected_damage_contract = {
        "allocation_damage_metric": allocation_meta.get("damage_metric"),
        "allocation_damage_weighting": allocation_meta.get("damage_weighting"),
        "allocation_damage_provenance": allocation_meta.get("damage_provenance"),
    }
    for name, expected in expected_damage_contract.items():
        if build.get(name) != expected:
            raise ValueError(f"QSRT build {name} drifted from its allocation")
    expected_plan = {
        "qsrt_atom_bytes": plan.qsrt_atom_bytes,
        "x4t_bytes": plan.x4t_bytes,
        "container_bytes": plan.total_container_bytes,
        "x4t_experts": plan.x4t_experts,
        "compressed_experts": plan.compressed_experts,
    }
    for name, expected in expected_plan.items():
        if build.get(name) != expected:
            raise ValueError(f"QSRT build {name} does not close")

    manifest = _read_json(manifest_path)
    expected_manifest = {
        "kind": QSRT_ARTIFACT_KIND,
        "schema_version": QSRT_ARTIFACT_SCHEMA_VERSION,
        "complete": True,
        "build": QSRT_BUILD_FILENAME,
        "build_sha256": _canonical_json_sha256(build),
        "allocation": QSRT_ALLOCATION_COPY_FILENAME,
        "codec": "QSRT",
        "trellis_codebook": codebook,
        "trellis_modes": list(PHASE1_MODE_IDS),
        **expected_plan,
        "layer_count": C.NUM_MOE_LAYERS,
    }
    for name, expected in expected_manifest.items():
        if manifest.get(name) != expected:
            raise ValueError(f"QSRT manifest {name} does not close")
    for name in (
        "source_model",
        "source_revision",
        "candidate_pool_content_sha256",
        "x4t_cost_index_content_sha256",
    ):
        if manifest.get(name) != build.get(name):
            raise ValueError(f"QSRT manifest {name} disagrees with its build")
    layer_entries = manifest.get("layers")
    if not isinstance(layer_entries, dict) or set(layer_entries) != {
        str(layer) for layer in C.MOE_LAYERS
    }:
        raise ValueError("QSRT manifest must contain exactly 92 layers")

    expected_files = {
        QSRT_BUILD_FILENAME,
        QSRT_ALLOCATION_COPY_FILENAME,
        QSRT_CANDIDATE_MANIFEST_COPY_FILENAME,
        QSRT_CANDIDATE_COMPLETION_COPY_FILENAME,
        QSRT_X4T_MANIFEST_COPY_FILENAME,
        QSRT_X4T_COMPLETION_COPY_FILENAME,
        QSRT_MANIFEST_FILENAME,
    }
    for layer in C.MOE_LAYERS:
        expected_files.update(
            {
                layer_filename(layer),
                x4t_layer_path(Path("."), layer).name,
                qsrt_layer_closure_filename(layer),
            }
        )
    actual_files = {path.name for path in root.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise ValueError(
            "QSRT artifact file inventory does not close; "
            f"missing={sorted(expected_files - actual_files)[:3]}, "
            f"extra={sorted(actual_files - expected_files)[:3]}"
        )

    store = None
    if verify_payloads:
        store = OfficialMXFP4Store(
            repo_dir=official_repo_dir,
            revision=str(build["source_revision"]),
        )
        if store.root != Path(str(build["official_checkpoint"])).resolve():
            raise ValueError("resolved official checkpoint disagrees with QSRT build")

    total_trellis = 0
    total_x4t = 0
    total_logical = 0
    for spec, expected_x4t_bytes in zip(
        plan.layers,
        plan.x4t_layer_bytes,
        strict=True,
    ):
        atom_file = root / layer_filename(spec.layer)
        x4t_file = x4t_layer_path(root, spec.layer)
        receipt = load_qsrt_layer_closure_receipt(
            atom_file,
            x4t_file,
            spec,
            expected_x4t_bytes=expected_x4t_bytes,
            build_document=build,
        )
        if verify_payloads:
            assert store is not None
            current = validate_qsrt_layer_payloads(
                atom_file,
                x4t_file,
                spec,
                pool.root,
                store,
                expected_x4t_bytes=expected_x4t_bytes,
            )
            if current != receipt:
                raise ValueError(
                    f"layer {spec.layer} source revalidation disagrees with receipt"
                )
        entry = layer_entries[str(spec.layer)]
        expected_entry = {
            **receipt,
            "qsrt_atoms": atom_file.name,
            "x4t": x4t_file.name,
            "layout": spec.layout.to_manifest(),
        }
        if entry != expected_entry:
            raise ValueError(f"layer {spec.layer} manifest entry does not close")
        total_trellis += atom_file.stat().st_size
        total_x4t += x4t_file.stat().st_size
        total_logical += _expected_logical_source_bytes(spec)
    if (
        total_trellis != plan.qsrt_atom_bytes
        or total_x4t != plan.x4t_bytes
        or total_trellis + total_x4t != plan.total_container_bytes
    ):
        raise AssertionError("QSRT filesystem byte ledger does not close")

    return {
        "kind": QSRT_ARTIFACT_KIND,
        "schema_version": QSRT_ARTIFACT_SCHEMA_VERSION,
        "artifact": str(root),
        "complete": True,
        "layers": len(plan.layers),
        "x4t_experts": plan.x4t_experts,
        "compressed_experts": plan.compressed_experts,
        "qsrt_atom_bytes": total_trellis,
        "x4t_bytes": total_x4t,
        "container_bytes": total_trellis + total_x4t,
        "candidate_codebook": pool.codebook,
        "candidate_mode_ids": list(pool.mode_ids),
        "candidate_pool_content_sha256": pool.content_sha256,
        "x4t_cost_index_content_sha256": x4t_index.content_sha256,
        "logical_source_bytes_covered": total_logical,
        "candidate_payload_headers_validated": validate_candidate_payload_headers,
        "source_payloads_reverified": verify_payloads,
        "serving_gates": "pending-separate-runtime-validation",
    }


__all__ = ["validate_qsrt_artifact"]
