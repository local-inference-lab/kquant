#!/usr/bin/env python3
"""Encode authenticated Fruit experts into canonical-profile QSRT pair shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import save_file

from kquant.exl3_loader import load_qsrt_encoder
from kquant.fruit_calibration import FruitCalibrationStore
from kquant.fruit_qsrt import (
    FRUIT_QSRT_CODEBOOK,
    FRUIT_QSRT_PROFILE_ID,
    FRUIT_QSRT_SCHEMA,
    encode_fruit_expert,
)
from kquant.fruit_source import (
    FRUIT_ANNEALED_SPEC,
    FruitCheckpointStore,
    FruitSafetensorsStore,
)
from kquant.sqg_quantizer import install_sqg_quantizer
from scripts.build_fruit_qsrt_model import (
    BASE_MANIFEST_SHA256,
    current_encoder_provenance,
)

SAMPLED_ASSIGNMENTS: tuple[tuple[int, int], ...] = (
    (3, 0),
    (3, 1),
    (3, 255),
    (4, 17),
    (4, 128),
    (5, 31),
    (5, 224),
    (6, 63),
    (7, 95),
    (8, 127),
    (9, 159),
    (10, 191),
    (11, 223),
    (12, 0),
    (12, 64),
    (12, 192),
    (12, 255),
    (13, 0),
    (13, 255),
)


def _assignment(value: str) -> tuple[int, int]:
    try:
        layer_text, expert_text = value.split(":", 1)
        layer, expert = int(layer_text), int(expert_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("assignment must be LAYER:EXPERT") from exc
    if not 3 <= layer <= 13 or not 0 <= expert < 256:
        raise argparse.ArgumentTypeError(
            "assignment requires layer 3..13 and expert 0..255"
        )
    return layer, expert


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return (
        json.dumps(
            value,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _artifact_paths(root: Path, layer: int, expert: int) -> tuple[Path, Path]:
    directory = root / f"layer-{layer:03d}"
    return (
        directory / f"expert-{expert:03d}.safetensors",
        directory / f"expert-{expert:03d}.json",
    )


def _run_manifest_path(
    root: Path,
    *,
    selection: str,
    shard_count: int,
    shard_index: int,
    assignments: tuple[tuple[int, int], ...],
) -> Path:
    identity = {
        "selection": selection,
        "shard_count": shard_count,
        "shard_index": shard_index,
        "assignments": [list(value) for value in assignments],
    }
    digest = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()[:16]
    return (
        root
        / "run-manifests"
        / f"run-{shard_index:04d}-of-{shard_count:04d}-{digest}.json"
    )


def _resume_evidence(
    tensor_path: Path,
    manifest_path: Path,
    *,
    layer: int,
    expert: int,
    source_sha256: str,
    encoder_fingerprint: str,
) -> dict[str, object] | None:
    if not tensor_path.exists() and not manifest_path.exists():
        return None
    if not tensor_path.is_file() or not manifest_path.is_file():
        raise ValueError(
            f"incomplete Fruit QSRT shard for layer {layer} expert {expert}"
        )
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed Fruit QSRT manifest: {manifest_path}") from exc
    expected = {
        "schema": FRUIT_QSRT_SCHEMA,
        "version": 1,
        "profile_id": FRUIT_QSRT_PROFILE_ID,
        "codebook": FRUIT_QSRT_CODEBOOK,
        "layer": layer,
        "expert": expert,
        "source_sha256": source_sha256,
        "encoder_fingerprint": encoder_fingerprint,
    }
    if not isinstance(value, dict) or any(
        value.get(key) != item for key, item in expected.items()
    ):
        raise ValueError(
            f"Fruit QSRT resume manifest identity mismatch: {manifest_path}"
        )
    actual_sha256 = _sha256(tensor_path)
    if value.get("safetensors_sha256") != actual_sha256:
        raise ValueError(f"Fruit QSRT resume shard hash mismatch: {tensor_path}")
    if value.get("safetensors_bytes") != tensor_path.stat().st_size:
        raise ValueError(f"Fruit QSRT resume shard size mismatch: {tensor_path}")
    return value


def _write_encoding(
    root: Path,
    encoding,
    *,
    source_sha256: str,
    encoder_fingerprint: str,
    elapsed_seconds: float,
    peak_cuda_bytes: int,
) -> dict[str, object]:
    tensor_path, manifest_path = _artifact_paths(root, encoding.layer, encoding.expert)
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tensor_path.with_name(f".{tensor_path.name}.tmp-{os.getpid()}")
    save_file(
        encoding.artifact_tensors(),
        temporary,
        metadata={
            "schema": FRUIT_QSRT_SCHEMA,
            "version": "1",
            "profile_id": str(FRUIT_QSRT_PROFILE_ID),
            "codebook": FRUIT_QSRT_CODEBOOK,
            "layer": str(encoding.layer),
            "expert": str(encoding.expert),
            "source_sha256": source_sha256,
            "encoder_fingerprint": encoder_fingerprint,
        },
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, tensor_path)
    manifest = {
        **encoding.manifest(),
        "version": 1,
        "source_sha256": source_sha256,
        "encoder_fingerprint": encoder_fingerprint,
        "safetensors_file": tensor_path.name,
        "safetensors_bytes": tensor_path.stat().st_size,
        "safetensors_sha256": _sha256(tensor_path),
        "encode_elapsed_seconds": elapsed_seconds,
        "peak_cuda_bytes": peak_cuda_bytes,
    }
    _atomic_text(manifest_path, _canonical_json(manifest))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--assignment",
        action="append",
        type=_assignment,
        help="encode one LAYER:EXPERT assignment; repeat for several",
    )
    selection.add_argument(
        "--sample",
        action="store_true",
        help=f"encode the frozen {len(SAMPLED_ASSIGNMENTS)}-assignment sample",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="encode all 2,816 expert assignments",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--exllamav3-root", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.shard_count < 1:
        parser.error("--shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must lie in 0..shard-count-1")
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Fruit QSRT encoder requires a CUDA device")
    torch.cuda.set_device(device)
    torch.empty(0, device=device)
    if args.sample:
        assignments = SAMPLED_ASSIGNMENTS
    elif args.all:
        assignments = tuple(
            (layer, expert)
            for layer in range(3, 14)
            for expert in range(FRUIT_ANNEALED_SPEC.num_experts)
        )
    else:
        assignments = tuple(sorted(set(args.assignment)))
    assignments = assignments[args.shard_index :: args.shard_count]
    if not assignments:
        raise ValueError("no Fruit assignments selected")

    if args.checkpoint.is_dir():
        store = FruitSafetensorsStore(
            args.checkpoint,
            spec=FRUIT_ANNEALED_SPEC,
            expected_manifest_sha256=BASE_MANIFEST_SHA256,
        )
    else:
        store = FruitCheckpointStore(
            args.checkpoint,
            spec=FRUIT_ANNEALED_SPEC,
            expected_sha256=FRUIT_ANNEALED_SPEC.checkpoint_sha256,
        )
    calibration_store = FruitCalibrationStore(args.calibration)
    quantizer_module = load_qsrt_encoder(args.exllamav3_root)
    install_sqg_quantizer(quantizer_module)
    source_sha256 = store.evidence.get("source_sha256")
    if not isinstance(source_sha256, str):
        raise TypeError("Fruit source evidence has no authenticated source digest")
    encoder = current_encoder_provenance(
        exllamav3_root=args.exllamav3_root,
        calibration=calibration_store,
    )
    encoder_fingerprint = str(encoder["fingerprint"])
    args.output.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    started = time.perf_counter()
    calibration_layer = None
    calibration_layer_id = None
    for index, (layer, expert) in enumerate(assignments, start=1):
        tensor_path, manifest_path = _artifact_paths(args.output, layer, expert)
        previous = (
            _resume_evidence(
                tensor_path,
                manifest_path,
                layer=layer,
                expert=expert,
                source_sha256=source_sha256,
                encoder_fingerprint=encoder_fingerprint,
            )
            if args.resume
            else None
        )
        if previous is not None:
            results.append(previous)
            print(f"[{index}/{len(assignments)}] layer {layer} expert {expert}: resume")
            continue
        if tensor_path.exists() or manifest_path.exists():
            raise FileExistsError(
                f"Fruit QSRT shard already exists; use --resume: {tensor_path}"
            )

        torch.cuda.reset_peak_memory_stats(device)
        item_started = time.perf_counter()
        if calibration_layer_id != layer:
            calibration_layer = calibration_store.load_layer(layer)
            calibration_layer_id = layer
        assert calibration_layer is not None
        encoding = encode_fruit_expert(
            store,
            layer=layer,
            expert=expert,
            device=device,
            calibration=calibration_layer.expert_rows(expert),
            quantizer_module=quantizer_module,
        )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - item_started
        peak = torch.cuda.max_memory_allocated(device)
        evidence = _write_encoding(
            args.output,
            encoding,
            source_sha256=source_sha256,
            encoder_fingerprint=encoder_fingerprint,
            elapsed_seconds=elapsed,
            peak_cuda_bytes=peak,
        )
        results.append(evidence)
        print(
            f"[{index}/{len(assignments)}] layer {layer} expert {expert}: "
            f"R13={encoding.r13} R2={encoding.r2}, {elapsed:.3f}s, "
            f"peak={peak / (1 << 20):.1f} MiB"
        )

    by_layer: dict[int, int] = defaultdict(int)
    for layer, _ in assignments:
        by_layer[layer] += 1
    selection = (
        "frozen_19_assignment_sample"
        if args.sample
        else "all_assignments"
        if args.all
        else "explicit_assignments"
    )
    run_manifest = {
        "schema": "kquant_fruit_qsrt_encode_run_v1",
        "version": 1,
        "artifact_schema": FRUIT_QSRT_SCHEMA,
        "profile_id": FRUIT_QSRT_PROFILE_ID,
        "codebook": FRUIT_QSRT_CODEBOOK,
        "source": store.evidence,
        "encoder": encoder,
        "selection": selection,
        "shard": {
            "count": args.shard_count,
            "index": args.shard_index,
        },
        "assignments": [list(value) for value in assignments],
        "assignment_count": len(assignments),
        "assignments_by_layer": {str(key): value for key, value in by_layer.items()},
        "elapsed_seconds": time.perf_counter() - started,
        "results": results,
        "status": "pass",
    }
    run_manifest_path = _run_manifest_path(
        args.output,
        selection=selection,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
        assignments=assignments,
    )
    run_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_text(run_manifest_path, _canonical_json(run_manifest))


if __name__ == "__main__":
    main()
