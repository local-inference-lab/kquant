"""Independent structural and source validation for TP12 SQRT-C artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from kquant import constants as C
from kquant.exl3_reference import CODEBOOK_SQG_NORMAL_E4M3
from kquant.mixed_exl3 import PHASE1_MODE_IDS
from kquant.pack.mixed_allocation import load_mixed_candidate_pool
from kquant.pack.mixed_materialize import (
    expected_logical_source_bytes,
    layer_filename,
)
from kquant.pack.sqrt_c_materialize import (
    SQRT_C_ALLOCATION_COPY_FILENAME,
    SQRT_C_ARTIFACT_KIND,
    SQRT_C_ARTIFACT_SCHEMA_VERSION,
    SQRT_C_BUILD_FILENAME,
    SQRT_C_BUILD_KIND,
    SQRT_C_BUILD_SCHEMA_VERSION,
    SQRT_C_CANDIDATE_COMPLETION_COPY_FILENAME,
    SQRT_C_CANDIDATE_MANIFEST_COPY_FILENAME,
    SQRT_C_MANIFEST_FILENAME,
    SQRT_C_X4_COMPLETION_COPY_FILENAME,
    SQRT_C_X4_MANIFEST_COPY_FILENAME,
    load_sqrt_c_layer_closure_receipt,
    sqrt_c_layer_closure_filename,
    validate_sqrt_c_layer_payloads,
    validate_sqrt_c_materialization_allocation,
)
from kquant.pack.x4_index import (
    X4_COST_COMPLETION_FILENAME,
    X4_COST_MANIFEST_FILENAME,
    load_x4_cost_index,
)
from kquant.source_weights import OfficialMXFP4Store
from kquant.x4 import x4_layer_path


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
        raise ValueError(f"SQRT-C {name} metadata is unavailable")
    if copy.read_bytes() != source.read_bytes():
        raise ValueError(f"SQRT-C {name} metadata copy drifted")


def validate_sqrt_c_artifact(
    root: str | Path,
    *,
    validate_candidate_payload_headers: bool = True,
    verify_payloads: bool = False,
    official_repo_dir: str | Path | None = None,
) -> dict:
    """Validate provenance, all layer pairs, and optionally every source byte."""

    root = Path(root).resolve()
    build_path = root / SQRT_C_BUILD_FILENAME
    allocation_path = root / SQRT_C_ALLOCATION_COPY_FILENAME
    manifest_path = root / SQRT_C_MANIFEST_FILENAME
    copied_paths = (
        root / SQRT_C_CANDIDATE_MANIFEST_COPY_FILENAME,
        root / SQRT_C_CANDIDATE_COMPLETION_COPY_FILENAME,
        root / SQRT_C_X4_MANIFEST_COPY_FILENAME,
        root / SQRT_C_X4_COMPLETION_COPY_FILENAME,
    )
    for path in (build_path, allocation_path, manifest_path, *copied_paths):
        if not path.is_file():
            raise ValueError(f"SQRT-C artifact is incomplete; missing {path}")
    partials = sorted(path.name for path in root.glob(".*.partial"))
    if partials:
        raise ValueError(f"SQRT-C artifact contains partial files: {partials[:3]}")

    build = _read_json(build_path)
    expected_build = {
        "kind": SQRT_C_BUILD_KIND,
        "schema_version": SQRT_C_BUILD_SCHEMA_VERSION,
        "codec": "SQRT-C",
        "source_model": C.MODEL_ID,
        "source_revision": C.REVISION,
        "layer_count": C.NUM_MOE_LAYERS,
        "byte_order": "little",
    }
    for name, expected in expected_build.items():
        if build.get(name) != expected:
            raise ValueError(
                f"SQRT-C build {name} mismatch: {build.get(name)!r} != {expected!r}"
            )
    expected_contract = {
        "tp_size": 12,
        "trellis_codebook": CODEBOOK_SQG_NORMAL_E4M3,
        "trellis_modes": list(PHASE1_MODE_IDS),
        "separate_r13_r2": True,
        "high_tier": "X4-exact-MXFP4",
    }
    if build.get("codec_contract") != expected_contract:
        raise ValueError("SQRT-C build codec contract drifted")

    candidate_root = Path(str(build.get("candidate_pool", ""))).resolve()
    x4_root = Path(str(build.get("x4_cost_index", ""))).resolve()
    candidate_manifest = candidate_root / "mixed_exl3_candidate_manifest.json"
    candidate_completion = candidate_root / "mixed_exl3_candidate_complete.json"
    x4_manifest = x4_root / X4_COST_MANIFEST_FILENAME
    x4_completion = x4_root / X4_COST_COMPLETION_FILENAME
    source_metadata = (
        (
            root / SQRT_C_CANDIDATE_MANIFEST_COPY_FILENAME,
            candidate_manifest,
            "candidate manifest",
            "candidate_manifest_sha256",
        ),
        (
            root / SQRT_C_CANDIDATE_COMPLETION_COPY_FILENAME,
            candidate_completion,
            "candidate completion",
            "candidate_completion_sha256",
        ),
        (
            root / SQRT_C_X4_MANIFEST_COPY_FILENAME,
            x4_manifest,
            "X4 cost manifest",
            "x4_manifest_sha256",
        ),
        (
            root / SQRT_C_X4_COMPLETION_COPY_FILENAME,
            x4_completion,
            "X4 cost completion",
            "x4_completion_sha256",
        ),
    )
    for copy, source, description, hash_field in source_metadata:
        _require_copy(copy, source, name=description)
        if build.get(hash_field) != _sha256(source):
            raise ValueError(f"SQRT-C build {description} hash drifted")
    if build.get("allocation_sha256") != _sha256(allocation_path):
        raise ValueError("SQRT-C build allocation hash drifted")

    pool = load_mixed_candidate_pool(
        candidate_root,
        validate_payload_headers=validate_candidate_payload_headers,
    )
    x4_index = load_x4_cost_index(x4_root)
    if build.get("candidate_pool_content_sha256") != pool.content_sha256:
        raise ValueError("SQRT-C candidate pool content digest drifted")
    if build.get("x4_cost_index_content_sha256") != x4_index.content_sha256:
        raise ValueError("SQRT-C X4 cost index content digest drifted")
    allocation = _read_json(allocation_path)
    plan = validate_sqrt_c_materialization_allocation(allocation, pool, x4_index)
    allocation_meta = allocation["meta"]
    expected_damage_contract = {
        "allocation_damage_metric": allocation_meta["damage_metric"],
        "allocation_damage_weighting": allocation_meta["damage_weighting"],
        "allocation_damage_provenance": allocation_meta["damage_provenance"],
    }
    for name, expected in expected_damage_contract.items():
        if build.get(name) != expected:
            raise ValueError(f"SQRT-C build {name} drifted from its allocation")
    expected_plan = {
        "trellis_slab_bytes": plan.trellis_slab_bytes,
        "x4_sidecar_bytes": plan.x4_sidecar_bytes,
        "container_bytes": plan.total_container_bytes,
        "x4_experts": plan.x4_experts,
        "compressed_experts": plan.compressed_experts,
    }
    for name, expected in expected_plan.items():
        if build.get(name) != expected:
            raise ValueError(f"SQRT-C build {name} does not close")

    manifest = _read_json(manifest_path)
    expected_manifest = {
        "kind": SQRT_C_ARTIFACT_KIND,
        "schema_version": SQRT_C_ARTIFACT_SCHEMA_VERSION,
        "complete": True,
        "build": SQRT_C_BUILD_FILENAME,
        "build_sha256": _canonical_json_sha256(build),
        "allocation": SQRT_C_ALLOCATION_COPY_FILENAME,
        "tp_size": 12,
        "codec": "SQRT-C",
        "trellis_codebook": CODEBOOK_SQG_NORMAL_E4M3,
        "trellis_modes": list(PHASE1_MODE_IDS),
        **expected_plan,
        "layer_count": C.NUM_MOE_LAYERS,
    }
    for name, expected in expected_manifest.items():
        if manifest.get(name) != expected:
            raise ValueError(f"SQRT-C manifest {name} does not close")
    for name in (
        "source_model",
        "source_revision",
        "candidate_pool_content_sha256",
        "x4_cost_index_content_sha256",
    ):
        if manifest.get(name) != build.get(name):
            raise ValueError(f"SQRT-C manifest {name} disagrees with its build")
    layer_entries = manifest.get("layers")
    if not isinstance(layer_entries, dict) or set(layer_entries) != {
        str(layer) for layer in C.MOE_LAYERS
    }:
        raise ValueError("SQRT-C manifest must contain exactly 92 layers")

    expected_files = {
        SQRT_C_BUILD_FILENAME,
        SQRT_C_ALLOCATION_COPY_FILENAME,
        SQRT_C_CANDIDATE_MANIFEST_COPY_FILENAME,
        SQRT_C_CANDIDATE_COMPLETION_COPY_FILENAME,
        SQRT_C_X4_MANIFEST_COPY_FILENAME,
        SQRT_C_X4_COMPLETION_COPY_FILENAME,
        SQRT_C_MANIFEST_FILENAME,
    }
    for layer in C.MOE_LAYERS:
        expected_files.update(
            {
                layer_filename(layer),
                x4_layer_path(Path("."), layer).name,
                sqrt_c_layer_closure_filename(layer),
            }
        )
    actual_files = {path.name for path in root.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise ValueError(
            "SQRT-C artifact file inventory does not close; "
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
            raise ValueError("resolved official checkpoint disagrees with SQRT-C build")

    total_trellis = 0
    total_x4 = 0
    total_logical = 0
    for spec, expected_x4_bytes in zip(
        plan.layers,
        plan.x4_layer_bytes,
        strict=True,
    ):
        slab = root / layer_filename(spec.layer)
        sidecar = x4_layer_path(root, spec.layer)
        receipt = load_sqrt_c_layer_closure_receipt(
            slab,
            sidecar,
            spec,
            expected_x4_bytes=expected_x4_bytes,
            build_document=build,
        )
        if verify_payloads:
            assert store is not None
            current = validate_sqrt_c_layer_payloads(
                slab,
                sidecar,
                spec,
                pool.root,
                store,
                expected_x4_bytes=expected_x4_bytes,
            )
            if current != receipt:
                raise ValueError(
                    f"layer {spec.layer} source revalidation disagrees with receipt"
                )
        entry = layer_entries[str(spec.layer)]
        expected_entry = {
            **receipt,
            "trellis_slab": slab.name,
            "x4_sidecar": sidecar.name,
            "layout": spec.layout.to_manifest(),
        }
        if entry != expected_entry:
            raise ValueError(f"layer {spec.layer} manifest entry does not close")
        total_trellis += slab.stat().st_size
        total_x4 += sidecar.stat().st_size
        total_logical += expected_logical_source_bytes(spec)
    if (
        total_trellis != plan.trellis_slab_bytes
        or total_x4 != plan.x4_sidecar_bytes
        or total_trellis + total_x4 != plan.total_container_bytes
    ):
        raise AssertionError("SQRT-C filesystem byte ledger does not close")

    return {
        "kind": SQRT_C_ARTIFACT_KIND,
        "schema_version": SQRT_C_ARTIFACT_SCHEMA_VERSION,
        "artifact": str(root),
        "complete": True,
        "tp_size": 12,
        "layers": len(plan.layers),
        "x4_experts": plan.x4_experts,
        "compressed_experts": plan.compressed_experts,
        "trellis_slab_bytes": total_trellis,
        "x4_sidecar_bytes": total_x4,
        "container_bytes": total_trellis + total_x4,
        "candidate_codebook": pool.codebook,
        "candidate_mode_ids": list(pool.mode_ids),
        "candidate_pool_content_sha256": pool.content_sha256,
        "x4_cost_index_content_sha256": x4_index.content_sha256,
        "logical_source_bytes_covered": total_logical,
        "candidate_payload_headers_validated": validate_candidate_payload_headers,
        "source_payloads_reverified": verify_payloads,
        "serving_gates": "pending-separate-TP12-validation",
    }


__all__ = ["validate_sqrt_c_artifact"]
