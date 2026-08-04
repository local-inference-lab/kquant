#!/usr/bin/env python3
"""Materialize a TP12 QSRT allocation into trellis/X4T layer pairs.

The official checkpoint is streamed only as an offline source.  Every output
layer consists of a fixed SQG trellis slab and an exact X4T sidecar.  A resume
accepts only the same sealed candidate pool, X4T index, allocation, and source
revision.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from kquant import constants as C
from kquant.pack.qsrt_pool import load_qsrt_candidate_pool
from kquant.pack.qsrt_slab import (
    layer_filename,
    materialize_layer,
    validate_materialized_layer,
)
from kquant.pack.qsrt_materialize import (
    QSRT_MANIFEST_FILENAME,
    load_qsrt_layer_closure_receipt,
    prepare_qsrt_destination,
    qsrt_artifact_manifest,
    qsrt_layer_closure_filename,
    qsrt_materialization_build_document,
    validate_qsrt_layer_pair,
    validate_qsrt_layer_payloads,
    validate_qsrt_materialization_allocation,
    write_qsrt_artifact_manifest,
    write_qsrt_layer_closure_receipt,
)
from kquant.pack.x4t_index import load_x4t_cost_index
from kquant.source_weights import OfficialMXFP4Store
from kquant.x4t import X4TLayerReader, x4t_layer_path


def _parse_ints(value: str) -> tuple[int, ...]:
    result: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            begin, end = map(int, item.split("-", 1))
            result.extend(range(begin, end + 1))
        else:
            result.append(int(item))
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError(
            "expected unique comma-separated integers/ranges"
        )
    if any(layer not in C.MOE_LAYERS for layer in result):
        raise argparse.ArgumentTypeError("layers must lie in 1..92")
    return tuple(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--x4t-cost-index", type=Path, required=True)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--official-repo-dir", type=Path)
    parser.add_argument("--official-revision", default=C.REVISION)
    parser.add_argument(
        "--layers",
        type=_parse_ints,
        help="materialize a diagnostic subset; omit to complete all 92 layers",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume only an identical immutable QSRT build contract",
    )
    parser.add_argument(
        "--sparse",
        action="store_true",
        help="use ftruncate instead of reserving each trellis slab extent",
    )
    parser.add_argument(
        "--skip-payload-header-validation",
        action="store_true",
        help="skip the initial all-candidate safetensors header inventory",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _combine_fresh_closure(
    metadata: dict[str, int | str],
    structural: dict[str, int | str],
) -> dict[str, int | str]:
    for name in (
        "file",
        "disk_bytes",
        "compressed_experts",
        "kept_experts",
        "codebook",
    ):
        if metadata.get(name) != structural.get(name):
            raise AssertionError(f"fresh QSRT closure field {name} drifted")
    return {
        **metadata,
        **{
            name: structural[name]
            for name in (
                "x4t_file",
                "x4t_disk_bytes",
                "x4t_records",
                "layer_container_bytes",
            )
        },
    }


def _load_or_rebuild_closure(
    slab: Path,
    sidecar: Path,
    spec,
    *,
    expected_x4t_bytes: int,
    build: dict,
    pool,
    store: OfficialMXFP4Store,
) -> dict[str, int | str]:
    try:
        return load_qsrt_layer_closure_receipt(
            slab,
            sidecar,
            spec,
            expected_x4t_bytes=expected_x4t_bytes,
            build_document=build,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(
            f"layer {spec.layer}: QSRT receipt unavailable ({exc}); "
            "revalidating both source tiers",
            flush=True,
        )
    closure = validate_qsrt_layer_payloads(
        slab,
        sidecar,
        spec,
        pool.root,
        store,
        expected_x4t_bytes=expected_x4t_bytes,
    )
    write_qsrt_layer_closure_receipt(
        slab,
        sidecar,
        spec,
        closure,
        expected_x4t_bytes=expected_x4t_bytes,
        build_document=build,
    )
    return closure


def _repair_orphan_pair(
    slab: Path,
    sidecar: Path,
    spec,
    *,
    expected_x4t_bytes: int,
    resume: bool,
) -> None:
    if slab.exists() == sidecar.exists():
        return
    if not resume:
        raise ValueError(
            f"layer {spec.layer} has only one member of its QSRT file pair"
        )
    if slab.exists():
        validate_materialized_layer(slab, spec)
        removed = slab
    else:
        reader = X4TLayerReader(sidecar)
        if reader.layer != spec.layer or reader.file_bytes != expected_x4t_bytes:
            raise ValueError(
                f"layer {spec.layer} orphan X4T sidecar does not match the build"
            )
        removed = sidecar
    removed.unlink()
    slab.with_name(qsrt_layer_closure_filename(spec.layer)).unlink(
        missing_ok=True
    )
    print(
        f"layer {spec.layer}: removed validated orphan {removed.name}; "
        "rebuilding the atomic pair",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    pool = load_qsrt_candidate_pool(
        args.candidate_pool,
        validate_payload_headers=not args.skip_payload_header_validation,
    )
    x4t_index = load_x4t_cost_index(args.x4t_cost_index)
    allocation_path = args.allocation.resolve()
    allocation = _read_json(allocation_path)
    plan = validate_qsrt_materialization_allocation(
        allocation,
        pool,
        x4t_index,
    )
    store = OfficialMXFP4Store(
        repo_dir=args.official_repo_dir,
        revision=args.official_revision,
    )
    if store.revision != pool.manifest["source_revision"]:
        raise ValueError(
            f"official source revision {store.revision} does not match candidate "
            f"revision {pool.manifest['source_revision']}"
        )
    build = qsrt_materialization_build_document(
        pool=pool,
        x4t_index=x4t_index,
        allocation_path=allocation_path,
        official_root=store.root,
        official_revision=store.revision,
        plan=plan,
    )
    destination = prepare_qsrt_destination(
        args.dest,
        build_document=build,
        allocation_path=allocation_path,
        pool=pool,
        x4t_index=x4t_index,
        resume=args.resume,
    )

    selected = set(C.MOE_LAYERS if args.layers is None else args.layers)
    verified: dict[int, dict[str, int | str]] = {}
    started = time.time()
    for index, (spec, expected_x4t_bytes) in enumerate(
        zip(plan.layers, plan.x4t_layer_bytes, strict=True),
        start=1,
    ):
        slab = destination / layer_filename(spec.layer)
        sidecar = x4t_layer_path(destination, spec.layer)
        _repair_orphan_pair(
            slab,
            sidecar,
            spec,
            expected_x4t_bytes=expected_x4t_bytes,
            resume=args.resume,
        )
        x4t_partial = sidecar.with_name(f".{sidecar.name}.partial")
        if x4t_partial.exists():
            if not args.resume:
                raise FileExistsError(x4t_partial)
            x4t_partial.unlink()
            print(
                f"layer {spec.layer}: discarded stale X4T partial",
                flush=True,
            )
        if slab.exists():
            if not args.resume:
                raise FileExistsError(slab)
            verified[spec.layer] = _load_or_rebuild_closure(
                slab,
                sidecar,
                spec,
                expected_x4t_bytes=expected_x4t_bytes,
                build=build,
                pool=pool,
                store=store,
            )
            print(f"layer {spec.layer}: bit-exact pair complete, skip", flush=True)
            continue
        if spec.layer not in selected:
            continue
        metadata = materialize_layer(
            pool.root,
            store,
            slab,
            spec,
            discard_partial=args.resume,
            preallocate=not args.sparse,
            x4t_destination=sidecar,
        )
        structural = validate_qsrt_layer_pair(
            slab,
            sidecar,
            spec,
            expected_x4t_bytes=expected_x4t_bytes,
        )
        closure = _combine_fresh_closure(metadata, structural)
        write_qsrt_layer_closure_receipt(
            slab,
            sidecar,
            spec,
            closure,
            expected_x4t_bytes=expected_x4t_bytes,
            build_document=build,
        )
        verified[spec.layer] = closure
        elapsed = time.time() - started
        print(
            f"layer {spec.layer}: wrote {structural['layer_container_bytes']} "
            f"bytes ({len(spec.compressed)} SQG, {len(spec.kept)} X4T); "
            f"{index}/{len(plan.layers)} layers considered in {elapsed:.1f}s",
            flush=True,
        )

    metadata: list[dict[str, int | str]] = []
    for spec, expected_x4t_bytes in zip(
        plan.layers,
        plan.x4t_layer_bytes,
        strict=True,
    ):
        slab = destination / layer_filename(spec.layer)
        sidecar = x4t_layer_path(destination, spec.layer)
        if not slab.is_file() or not sidecar.is_file():
            print(
                f"partial artifact: layer {spec.layer} pair is incomplete; "
                f"no {QSRT_MANIFEST_FILENAME} written",
                flush=True,
            )
            return
        closure = verified.get(spec.layer)
        if closure is None:
            closure = _load_or_rebuild_closure(
                slab,
                sidecar,
                spec,
                expected_x4t_bytes=expected_x4t_bytes,
                build=build,
                pool=pool,
                store=store,
            )
        metadata.append(closure)
    manifest = qsrt_artifact_manifest(
        build_document=build,
        plan=plan,
        layers=metadata,
    )
    manifest_path = write_qsrt_artifact_manifest(destination, manifest)
    print(
        f"complete: {manifest_path} closes {plan.total_container_bytes} bytes, "
        f"{plan.x4t_experts} exact X4T and {plan.compressed_experts} SQG experts",
        flush=True,
    )


if __name__ == "__main__":
    main()
