#!/usr/bin/env python3
"""Build fixed TP12 layer containers from a mixed-EXL allocation.

The official checkpoint remains an offline packed-weight source.  The script
streams one candidate tensor or original MXFP4 matrix at a time and writes
each multi-gigabyte layer through explicit offsets into an atomic temporary
file.  ``--resume`` accepts only the exact same candidate pool, allocation,
and official source revision.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from kquant import constants as C
from kquant.pack.mixed_allocation import load_mixed_candidate_pool
from kquant.pack.mixed_materialize import (
    MATERIALIZED_MANIFEST_FILENAME,
    artifact_manifest,
    layer_filename,
    load_layer_closure_receipt,
    materialization_build_document,
    materialize_layer,
    prepare_destination,
    validate_materialization_allocation,
    validate_materialized_layer,
    validate_materialized_layer_payloads,
    write_layer_closure_receipt,
    write_artifact_manifest,
)
from kquant.source_weights import OfficialMXFP4Store


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
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument(
        "--mode-validation-summary",
        type=Path,
        required=True,
        help="reproducible matched-R0 gate summary; must have status=pass",
    )
    parser.add_argument(
        "--teacher-proxy-summary",
        type=Path,
        required=True,
        help="reproducible official/interim proxy gate; must have status=pass",
    )
    parser.add_argument(
        "--performance-summary",
        type=Path,
        required=True,
        help="combined isolated+routed TP12 performance gate; must pass",
    )
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
        help="resume only an identical immutable build contract",
    )
    parser.add_argument(
        "--sparse",
        action="store_true",
        help="use ftruncate instead of reserving each layer's exact disk extent",
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


def _load_or_rebuild_closure(
    path: Path,
    spec,
    *,
    build: dict,
    pool,
    store: OfficialMXFP4Store,
) -> dict[str, int | str]:
    try:
        return load_layer_closure_receipt(
            path,
            spec,
            build_document=build,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(
            f"layer {spec.layer}: closure receipt unavailable ({exc}); "
            "revalidating source payloads",
            flush=True,
        )
    closure = validate_materialized_layer_payloads(
        path,
        spec,
        pool.root,
        store,
    )
    write_layer_closure_receipt(
        path,
        spec,
        closure,
        build_document=build,
    )
    return closure


def main() -> None:
    args = parse_args()
    pool = load_mixed_candidate_pool(
        args.candidate_pool,
        validate_payload_headers=not args.skip_payload_header_validation,
    )
    allocation_path = args.allocation.resolve()
    allocation = _read_json(allocation_path)
    plan = validate_materialization_allocation(allocation, pool)
    store = OfficialMXFP4Store(
        repo_dir=args.official_repo_dir,
        revision=args.official_revision,
    )
    if store.revision != pool.manifest["source_revision"]:
        raise ValueError(
            f"official source revision {store.revision} does not match candidate "
            f"revision {pool.manifest['source_revision']}"
        )
    build = materialization_build_document(
        pool=pool,
        allocation_path=allocation_path,
        mode_validation_summary_path=args.mode_validation_summary.resolve(),
        teacher_proxy_summary_path=args.teacher_proxy_summary.resolve(),
        performance_summary_path=args.performance_summary.resolve(),
        official_root=store.root,
        official_revision=store.revision,
        plan=plan,
    )
    destination = prepare_destination(
        args.dest,
        build_document=build,
        allocation_path=allocation_path,
        mode_validation_summary_path=args.mode_validation_summary.resolve(),
        teacher_proxy_summary_path=args.teacher_proxy_summary.resolve(),
        performance_summary_path=args.performance_summary.resolve(),
        resume=args.resume,
    )

    selected = set(C.MOE_LAYERS if args.layers is None else args.layers)
    verified: dict[int, dict[str, int | str]] = {}
    started = time.time()
    for index, spec in enumerate(plan.layers, start=1):
        target = destination / layer_filename(spec.layer)
        if target.exists():
            if not args.resume:
                raise FileExistsError(target)
            validate_materialized_layer(target, spec)
            verified[spec.layer] = _load_or_rebuild_closure(
                target,
                spec,
                build=build,
                pool=pool,
                store=store,
            )
            print(f"layer {spec.layer}: bit-exact complete, skip", flush=True)
            continue
        if spec.layer not in selected:
            continue
        closure = materialize_layer(
            pool.root,
            store,
            target,
            spec,
            discard_partial=args.resume,
            preallocate=not args.sparse,
        )
        write_layer_closure_receipt(
            target,
            spec,
            closure,
            build_document=build,
        )
        verified[spec.layer] = closure
        elapsed = time.time() - started
        print(
            f"layer {spec.layer}: wrote {spec.layout.disk_bytes} bytes; "
            f"{index}/{len(plan.layers)} layers considered in {elapsed:.1f}s",
            flush=True,
        )

    metadata = []
    for spec in plan.layers:
        path = destination / layer_filename(spec.layer)
        if not path.is_file():
            print(
                f"partial artifact: {path.name} is not materialized; "
                f"no {MATERIALIZED_MANIFEST_FILENAME} written",
                flush=True,
            )
            return
        closure = verified.get(spec.layer)
        if closure is None:
            closure = _load_or_rebuild_closure(
                path,
                spec,
                build=build,
                pool=pool,
                store=store,
            )
        metadata.append(closure)
    manifest = artifact_manifest(
        build_document=build,
        plan=plan,
        layers=metadata,
    )
    manifest_path = write_artifact_manifest(destination, manifest)
    print(
        f"complete: {manifest_path} closes {plan.total_container_bytes} bytes, "
        f"{plan.retained_experts} MXFP4 and "
        f"{plan.compressed_experts} mixed-EXL experts",
        flush=True,
    )


if __name__ == "__main__":
    main()
