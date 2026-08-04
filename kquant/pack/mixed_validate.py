"""Independent validation for completed TP12 mixed-EXL artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from kquant import constants as C
from kquant.mixed_exl3 import SCHEMA, TP_SIZE
from kquant.pack.mixed_allocation import load_mixed_candidate_pool
from kquant.pack.mixed_materialize import (
    MATERIALIZATION_BUILD_FILENAME,
    MATERIALIZATION_BUILD_KIND,
    MATERIALIZATION_BUILD_SCHEMA_VERSION,
    MATERIALIZED_ALLOCATION_FILENAME,
    MATERIALIZED_ARTIFACT_KIND,
    MATERIALIZED_ARTIFACT_SCHEMA_VERSION,
    MATERIALIZED_CLOSURE_SUFFIX,
    MATERIALIZED_LAYER_PREFIX,
    MATERIALIZED_MANIFEST_FILENAME,
    MATERIALIZED_MODE_VALIDATION_SUMMARY_FILENAME,
    MATERIALIZED_PERFORMANCE_SUMMARY_FILENAME,
    MATERIALIZED_TEACHER_PROXY_SUMMARY_FILENAME,
    MXFP4_MATRIX_COMPONENT_ORDER,
    LayerMaterializationSpec,
    expected_logical_source_bytes as _expected_logical_source_bytes,
    layer_closure_filename,
    layer_filename,
    load_layer_closure_receipt,
    validate_materialization_allocation,
    validate_materialization_mode_gate,
    validate_materialization_performance_gate,
    validate_materialization_teacher_proxy_gate,
    validate_materialized_layer,
    validate_materialized_layer_payloads,
)
from kquant.source_weights import OfficialMXFP4Store


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def expected_logical_source_bytes(spec: LayerMaterializationSpec) -> int:
    """Bytes compared by the source-to-slab closure for one layer."""

    return _expected_logical_source_bytes(spec)


def _validate_layer_manifest_entry(
    entry: object,
    spec: LayerMaterializationSpec,
    structural: dict[str, int | str],
) -> None:
    if not isinstance(entry, dict):
        raise ValueError(f"artifact layer {spec.layer} manifest entry is not an object")
    expected = {
        **structural,
        "payload_closure": "bit_exact",
        "compressed_experts_verified": len(spec.compressed),
        "kept_experts_verified": len(spec.kept),
        "logical_source_bytes_verified": expected_logical_source_bytes(spec),
        "layout": spec.layout.to_manifest(),
    }
    if entry != expected:
        differing = sorted(
            key
            for key in set(entry).union(expected)
            if entry.get(key) != expected.get(key)
        )
        raise ValueError(
            f"artifact layer {spec.layer} manifest does not close: {differing[:8]}"
        )


def validate_materialized_artifact(
    root: str | Path,
    *,
    validate_candidate_payload_headers: bool = True,
    verify_payloads: bool = False,
    official_repo_dir: str | Path | None = None,
) -> dict:
    """Validate provenance, exact bytes, every layer, and optionally all sources."""

    root = Path(root).resolve()
    build_path = root / MATERIALIZATION_BUILD_FILENAME
    allocation_path = root / MATERIALIZED_ALLOCATION_FILENAME
    manifest_path = root / MATERIALIZED_MANIFEST_FILENAME
    mode_summary_path = root / MATERIALIZED_MODE_VALIDATION_SUMMARY_FILENAME
    teacher_proxy_path = root / MATERIALIZED_TEACHER_PROXY_SUMMARY_FILENAME
    performance_path = root / MATERIALIZED_PERFORMANCE_SUMMARY_FILENAME
    for path in (
        build_path,
        allocation_path,
        mode_summary_path,
        teacher_proxy_path,
        performance_path,
        manifest_path,
    ):
        if not path.is_file():
            raise ValueError(f"materialized artifact is incomplete; missing {path}")
    partials = sorted(root.glob(".*.partial"))
    if partials:
        raise ValueError(
            f"materialized artifact contains partial layers: {partials[:3]}"
        )

    build = _read_json(build_path)
    expected_build_scalars = {
        "kind": MATERIALIZATION_BUILD_KIND,
        "schema_version": MATERIALIZATION_BUILD_SCHEMA_VERSION,
        "format_schema": SCHEMA,
        "tp_size": TP_SIZE,
        "source_model": C.MODEL_ID,
        "source_revision": C.REVISION,
        "layer_count": C.NUM_MOE_LAYERS,
        "mxfp4_matrix_component_order": list(MXFP4_MATRIX_COMPONENT_ORDER),
        "byte_order": "little",
    }
    for name, expected in expected_build_scalars.items():
        if build.get(name) != expected:
            raise ValueError(
                f"materialization build {name} mismatch: "
                f"{build.get(name)!r} != {expected!r}"
            )
    candidate_root = Path(str(build.get("candidate_pool", ""))).resolve()
    candidate_manifest = candidate_root / "mixed_exl3_candidate_manifest.json"
    if not candidate_manifest.is_file():
        raise ValueError("materialization candidate manifest is unavailable")
    if build.get("candidate_manifest_sha256") != _sha256(candidate_manifest):
        raise ValueError("candidate manifest hash does not match the immutable build")
    if build.get("allocation_sha256") != _sha256(allocation_path):
        raise ValueError("artifact allocation hash does not match the immutable build")
    if build.get("mode_validation_summary_sha256") != _sha256(mode_summary_path):
        raise ValueError(
            "artifact mode-validation summary hash does not match the build"
        )
    if build.get("teacher_proxy_summary_sha256") != _sha256(teacher_proxy_path):
        raise ValueError(
            "artifact teacher-proxy summary hash does not match the build"
        )
    if build.get("performance_summary_sha256") != _sha256(performance_path):
        raise ValueError(
            "artifact TP12 performance summary hash does not match the build"
        )

    pool = load_mixed_candidate_pool(
        candidate_root,
        validate_payload_headers=validate_candidate_payload_headers,
    )
    if build.get("candidate_pool_content_sha256") != pool.content_sha256:
        raise ValueError(
            "materialization build candidate pool content digest drifted"
        )
    if build.get("candidate_logical_trellis_schema") != pool.trellis_schema:
        raise ValueError(
            "materialization build candidate logical trellis schema drifted"
        )
    if build.get("candidate_codebook") != pool.codebook:
        raise ValueError("materialization build candidate codebook drifted")
    if build.get("candidate_mode_ids") != list(pool.mode_ids):
        raise ValueError("materialization build candidate mode table drifted")
    allocation = _read_json(allocation_path)
    allocation_meta = allocation.get("meta")
    if not isinstance(allocation_meta, dict):
        raise ValueError("artifact allocation is missing its meta object")
    allocation_build_fields = {
        "allocation_damage_metric": allocation_meta.get("damage_metric"),
        "allocation_damage_provenance": allocation_meta.get(
            "damage_provenance"
        ),
    }
    for name, expected in allocation_build_fields.items():
        if build.get(name) != expected:
            raise ValueError(
                f"materialization build {name} disagrees with its allocation"
            )
    plan = validate_materialization_allocation(allocation, pool)
    mode_validation_gate = validate_materialization_mode_gate(
        mode_summary_path,
        allocation,
        pool,
    )
    if build.get("mode_validation_gate") != mode_validation_gate:
        raise ValueError("materialization build mode-validation gate drifted")
    teacher_proxy_gate = validate_materialization_teacher_proxy_gate(
        teacher_proxy_path
    )
    if build.get("teacher_proxy_gate") != teacher_proxy_gate:
        raise ValueError("materialization build teacher-proxy gate drifted")
    performance_gate = validate_materialization_performance_gate(
        performance_path,
        pool,
    )
    if build.get("performance_gate") != performance_gate:
        raise ValueError("materialization build TP12 performance gate drifted")
    expected_plan = {
        "container_bytes": plan.total_container_bytes,
        "retained_experts": plan.retained_experts,
        "compressed_experts": plan.compressed_experts,
    }
    for name, expected in expected_plan.items():
        if build.get(name) != expected:
            raise ValueError(
                f"materialization build {name} does not close: "
                f"{build.get(name)!r} != {expected}"
            )

    manifest = _read_json(manifest_path)
    expected_manifest_scalars = {
        "kind": MATERIALIZED_ARTIFACT_KIND,
        "schema_version": MATERIALIZED_ARTIFACT_SCHEMA_VERSION,
        "complete": True,
        "allocation": MATERIALIZED_ALLOCATION_FILENAME,
        "mode_validation_summary": (
            MATERIALIZED_MODE_VALIDATION_SUMMARY_FILENAME
        ),
        "teacher_proxy_summary": MATERIALIZED_TEACHER_PROXY_SUMMARY_FILENAME,
        "performance_summary": MATERIALIZED_PERFORMANCE_SUMMARY_FILENAME,
    }
    for name, expected in expected_manifest_scalars.items():
        if manifest.get(name) != expected:
            raise ValueError(
                f"artifact manifest {name} mismatch: "
                f"{manifest.get(name)!r} != {expected!r}"
            )
    copied_build_fields = (
        "format_schema",
        "tp_size",
        "source_model",
        "source_revision",
        "official_checkpoint",
        "candidate_pool",
        "candidate_manifest_sha256",
        "candidate_pool_content_sha256",
        "candidate_logical_trellis_schema",
        "candidate_codebook",
        "candidate_mode_ids",
        "allocation_source",
        "allocation_sha256",
        "allocation_damage_metric",
        "allocation_damage_provenance",
        "mode_validation_summary_source",
        "mode_validation_summary_sha256",
        "mode_validation_gate",
        "teacher_proxy_summary_source",
        "teacher_proxy_summary_sha256",
        "teacher_proxy_gate",
        "performance_summary_source",
        "performance_summary_sha256",
        "performance_gate",
        "container_bytes",
        "retained_experts",
        "compressed_experts",
        "mxfp4_matrix_component_order",
        "byte_order",
    )
    for name in copied_build_fields:
        if manifest.get(name) != build.get(name):
            raise ValueError(f"artifact manifest {name} disagrees with its build")
    layer_entries = manifest.get("layers")
    if not isinstance(layer_entries, dict) or set(layer_entries) != {
        str(layer) for layer in C.MOE_LAYERS
    }:
        raise ValueError("artifact manifest must contain exactly 92 MoE layers")

    expected_files = {layer_filename(layer) for layer in C.MOE_LAYERS}
    actual_files = {
        path.name
        for path in root.glob(f"{MATERIALIZED_LAYER_PREFIX}*.bin")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError(
            "artifact layer-file inventory does not close; "
            f"missing={sorted(expected_files - actual_files)[:3]}, "
            f"extra={sorted(actual_files - expected_files)[:3]}"
        )
    expected_receipts = {
        layer_closure_filename(layer) for layer in C.MOE_LAYERS
    }
    actual_receipts = {
        path.name
        for path in root.glob(
            f"{MATERIALIZED_LAYER_PREFIX}*{MATERIALIZED_CLOSURE_SUFFIX}"
        )
        if path.is_file()
    }
    if actual_receipts != expected_receipts:
        raise ValueError(
            "artifact closure-receipt inventory does not close; "
            f"missing={sorted(expected_receipts - actual_receipts)[:3]}, "
            f"extra={sorted(actual_receipts - expected_receipts)[:3]}"
        )

    store = None
    if verify_payloads:
        store = OfficialMXFP4Store(
            repo_dir=official_repo_dir,
            revision=str(build["source_revision"]),
        )
        if store.root != Path(str(build["official_checkpoint"])).resolve():
            raise ValueError("resolved official checkpoint disagrees with the build")

    total_bytes = 0
    total_logical_verified = 0
    for spec in plan.layers:
        path = root / layer_filename(spec.layer)
        receipt = load_layer_closure_receipt(
            path,
            spec,
            build_document=build,
        )
        if verify_payloads:
            assert store is not None
            structural = validate_materialized_layer_payloads(
                path,
                spec,
                pool.root,
                store,
            )
            if structural != receipt:
                raise ValueError(
                    f"layer {spec.layer} payload revalidation disagrees with "
                    "its closure receipt"
                )
        else:
            structural = validate_materialized_layer(path, spec)
        _validate_layer_manifest_entry(
            layer_entries[str(spec.layer)],
            spec,
            structural,
        )
        total_bytes += path.stat().st_size
        total_logical_verified += expected_logical_source_bytes(spec)
    if total_bytes != plan.total_container_bytes:
        raise AssertionError("artifact filesystem bytes do not close to the allocation")

    return {
        "kind": MATERIALIZED_ARTIFACT_KIND,
        "schema_version": MATERIALIZED_ARTIFACT_SCHEMA_VERSION,
        "artifact": str(root),
        "complete": True,
        "tp_size": TP_SIZE,
        "layers": len(plan.layers),
        "retained_experts": plan.retained_experts,
        "compressed_experts": plan.compressed_experts,
        "container_bytes": total_bytes,
        "format_schema": build["format_schema"],
        "candidate_logical_trellis_schema": build[
            "candidate_logical_trellis_schema"
        ],
        "candidate_codebook": build["candidate_codebook"],
        "candidate_mode_ids": build["candidate_mode_ids"],
        "candidate_pool_content_sha256": build[
            "candidate_pool_content_sha256"
        ],
        "mode_validation_status": mode_validation_gate["status"],
        "teacher_proxy_status": teacher_proxy_gate["status"],
        "tp12_performance_status": performance_gate["status"],
        "matched_r0_assignments": mode_validation_gate[
            "audited_assignments"
        ],
        "logical_source_bytes_covered": total_logical_verified,
        "candidate_payload_headers_validated": validate_candidate_payload_headers,
        "source_payloads_reverified": verify_payloads,
    }
