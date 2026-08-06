#!/usr/bin/env python3
"""Score selected TP12 QSRT candidates on the untouched capture.

The candidate pool's fit and confirmation documents are never read.  Every
selected candidate is decoded from its persisted payload and compared with
the corresponding official MXFP4 source expert on naturally routed rows from
the document-disjoint validation capture.  The official model is never
instantiated; source matrices are streamed one at a time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from kquant import constants as C
from kquant.capture import index_layer_samples, load_capture
from kquant.exl3_reference import CODEBOOK_SQG_NORMAL_E4M3, QSRT_CODEBOOKS
from kquant.qsrt_candidates import index_expert_rows, request_documents
from kquant.pack.qsrt_pool import (
    load_qsrt_candidate_pool,
    validate_candidate_pool_completion,
)
from kquant.pack.qsrt_candidates import (
    CANDIDATE_POOL_SCHEMA_VERSION,
    OFFICIAL_SOURCE_DAMAGE_METRIC,
)
from kquant.pack.qsrt_validation import (
    VALIDATION_DAMAGE_METRIC,
    VALIDATION_SCORE_KIND,
    VALIDATION_SCORE_SCHEMA_VERSION,
    score_selected_expert,
)
from kquant.source_weights import OfficialMXFP4Store


NUM_GPUS = 12
MANIFEST_NAME = "qsrt-validation-manifest.json"
COMPLETION_NAME = "qsrt-validation-completion.json"


@dataclass
class _WorkerResources:
    documents: dict[int, str]
    sample_index: object
    store: OfficialMXFP4Store
    logical_trellis_schema: str
    codebook: str
    mode_ids: tuple[int, ...]


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


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        save_file(tensors, temporary)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _parse_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(part) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("integer list must be nonempty and unique")
    return result


def _validate_corpus_contract(
    candidate_manifest: dict,
    *,
    validation_capture: Path,
    validation_report_path: Path,
) -> tuple[dict, dict[int, str]]:
    training_report_path = Path(str(candidate_manifest["training_report"]))
    training_report = _read_json(training_report_path)
    validation_report = _read_json(validation_report_path)
    for name, report in (
        ("training", training_report),
        ("validation", validation_report),
    ):
        if report.get("kind") != "kquant_interim_calibration_corpus_run":
            raise ValueError(f"{name} corpus report has the wrong kind")
        if not bool(report.get("finalized")):
            raise ValueError(f"{name} corpus report is not finalized")
    if Path(str(training_report.get("capture_dir", ""))).resolve() != Path(
        str(candidate_manifest["capture"])
    ).resolve():
        raise ValueError("candidate training report and capture disagree")
    if Path(str(validation_report.get("capture_dir", ""))).resolve() != (
        validation_capture.resolve()
    ):
        raise ValueError("validation report and capture disagree")

    training_fold = training_report.get("fold", {})
    validation_fold = validation_report.get("fold", {})
    if not isinstance(training_fold, dict) or not isinstance(validation_fold, dict):
        raise TypeError("corpus fold metadata must be mappings")
    common_fold = ("modulus", "index", "unit")
    if training_fold.get("mode") != "exclude":
        raise ValueError("training report must exclude the validation fold")
    if validation_fold.get("mode") != "include":
        raise ValueError("validation report must contain only the validation fold")
    if training_fold.get("unit") != validation_fold.get("unit"):
        raise ValueError("training and validation reports use different fold units")
    training_modulus = int(training_fold.get("modulus", 0))
    validation_modulus = int(validation_fold.get("modulus", 0))
    training_index = int(training_fold.get("index", -1))
    validation_index = int(validation_fold.get("index", -1))
    # An included validation remainder at a finer modulus is a strict subset
    # of the coarser remainder excluded from training when these congruences
    # hold.  Keep the explicit document-hash overlap check below as a second,
    # independent evidence gate.
    nested_validation_fold = (
        training_modulus > 0
        and validation_modulus > 0
        and validation_modulus % training_modulus == 0
        and validation_index % training_modulus == training_index
    )
    if not nested_validation_fold:
        raise ValueError(
            "validation fold is not contained in the fold excluded from training"
        )

    training_documents = request_documents(training_report)
    validation_documents = request_documents(validation_report, deduplicate=True)
    validation_requests = len(validation_report["documents"])
    overlap = set(training_documents.values()) & set(validation_documents.values())
    if overlap:
        raise ValueError(f"training/validation document overlap: {sorted(overlap)[:8]}")
    teacher = Path(str(candidate_manifest["resident_teacher_checkpoint"])).resolve()
    if Path(str(validation_report.get("model_dir", ""))).resolve() != teacher:
        raise ValueError("validation capture used a different resident teacher")
    capture_manifest, geometry, _ranks = load_capture(
        validation_capture, load_stats=False
    )
    if geometry.num_layers != C.NUM_MOE_LAYERS:
        raise ValueError("validation capture does not contain all MoE layers")
    if Path(str(capture_manifest.get("teacher_checkpoint", ""))).resolve() != teacher:
        raise ValueError("validation capture manifest used a different teacher")
    provenance = {
        "training_report": str(training_report_path.resolve()),
        "validation_report": str(validation_report_path.resolve()),
        "training_documents": len(training_documents),
        "validation_documents": len(validation_documents),
        "validation_requests_captured": validation_requests,
        "validation_duplicate_requests_dropped": (
            validation_requests - len(validation_documents)
        ),
        "document_overlap": 0,
        "training_fold": {key: training_fold.get(key) for key in common_fold},
        "validation_fold": {key: validation_fold.get(key) for key in common_fold},
    }
    return provenance, validation_documents


def _manifest(args: argparse.Namespace) -> dict:
    candidate_manifest_path = args.candidate_pool / "qsrt-candidate-manifest.json"
    candidate_manifest = _read_json(candidate_manifest_path)
    if (
        candidate_manifest.get("source_model") != C.MODEL_ID
        or candidate_manifest.get("source_revision") != args.official_revision
        or candidate_manifest.get("schema_version")
        != CANDIDATE_POOL_SCHEMA_VERSION
    ):
        raise ValueError("candidate manifest source or schema does not match")
    provenance, documents = _validate_corpus_contract(
        candidate_manifest,
        validation_capture=args.validation_capture,
        validation_report_path=args.validation_report,
    )
    completion = validate_candidate_pool_completion(args.candidate_pool)
    codebook = candidate_manifest.get("codebook")
    if codebook not in QSRT_CODEBOOKS:
        raise ValueError("candidate manifest codebook is unsupported")
    mode_ids = tuple(int(mode) for mode in candidate_manifest.get("mode_ids", ()))
    return {
        "kind": VALIDATION_SCORE_KIND,
        "schema_version": VALIDATION_SCORE_SCHEMA_VERSION,
        "candidate_pool": str(args.candidate_pool.resolve()),
        "candidate_schema_version": CANDIDATE_POOL_SCHEMA_VERSION,
        "candidate_manifest_sha256": _sha256(candidate_manifest_path),
        "candidate_pool_content_sha256": completion["content_sha256"],
        "candidate_logical_trellis_schema": completion[
            "logical_trellis_schema"
        ],
        "candidate_codebook": codebook,
        "candidate_mode_ids": list(mode_ids),
        "source_model": C.MODEL_ID,
        "source_revision": args.official_revision,
        "resident_teacher_checkpoint": candidate_manifest[
            "resident_teacher_checkpoint"
        ],
        "validation_capture": str(args.validation_capture.resolve()),
        "validation_report": str(args.validation_report.resolve()),
        "validation_documents": len(documents),
        "provenance": provenance,
        "tp_size": 12,
        "metric": VALIDATION_DAMAGE_METRIC,
        "selection_data_used": False,
        "official_source_residency": "one expert matrix at a time",
    }


def _layer_paths(root: Path, layer: int) -> tuple[Path, Path]:
    stem = f"qsrt-layer-{layer:05d}.validation"
    return root / "scores" / f"{stem}.safetensors", root / "scores" / f"{stem}.json"


def _layer_complete(root: Path, layer: int) -> bool:
    metrics_path, ledger_path = _layer_paths(root, layer)
    if not metrics_path.is_file() or not ledger_path.is_file():
        return False
    ledger = _read_json(ledger_path)
    return bool(
        ledger.get("kind") == VALIDATION_SCORE_KIND
        and ledger.get("schema_version") == VALIDATION_SCORE_SCHEMA_VERSION
        and ledger.get("layer") == layer
        and ledger.get("complete")
        and ledger.get("metrics") == metrics_path.name
    )


def _candidate_paths(root: Path, layer: int) -> tuple[Path, Path]:
    stem = f"qsrt-layer-{layer:05d}"
    return (
        root / "candidates" / f"{stem}.safetensors",
        root / "candidates" / f"{stem}.metrics.safetensors",
    )


def _allocate_metrics(documents: int) -> dict[str, torch.Tensor]:
    return {
        "expert_ids": torch.arange(C.NUM_EXPERTS, dtype=torch.int16),
        "selected_r13": torch.zeros(C.NUM_EXPERTS, dtype=torch.uint8),
        "selected_r2": torch.zeros(C.NUM_EXPERTS, dtype=torch.uint8),
        "validation_sse": torch.zeros(
            (C.NUM_EXPERTS, documents), dtype=torch.float64
        ),
        "validation_reference_energy": torch.zeros(
            (C.NUM_EXPERTS, documents), dtype=torch.float64
        ),
        "validation_counts": torch.zeros(
            (C.NUM_EXPERTS, documents), dtype=torch.int32
        ),
        "validation_gate_square_sum": torch.zeros(
            C.NUM_EXPERTS, dtype=torch.float64
        ),
        VALIDATION_DAMAGE_METRIC: torch.zeros(C.NUM_EXPERTS, dtype=torch.float64),
        "selection_corpus_damage": torch.zeros(C.NUM_EXPERTS, dtype=torch.float64),
    }


def score_layer(
    args: argparse.Namespace,
    layer: int,
    *,
    resources: _WorkerResources,
) -> dict:
    output_metrics, output_ledger = _layer_paths(args.dest, layer)
    if _layer_complete(args.dest, layer):
        print(f"layer {layer}: validation complete, skip", flush=True)
        return _read_json(output_ledger)
    if output_metrics.exists() or output_ledger.exists():
        raise FileExistsError(
            f"layer {layer} has incomplete validation output; use a fresh destination"
        )

    candidate_payload, candidate_metrics_path = _candidate_paths(
        args.candidate_pool, layer
    )
    candidate_metrics = load_file(str(candidate_metrics_path), device="cpu")
    selected_r13 = candidate_metrics["selected_r13"]
    selected_r2 = candidate_metrics["selected_r2"]
    for name, selected in (("r13", selected_r13), ("r2", selected_r2)):
        if (
            selected.dtype != torch.uint8
            or tuple(selected.shape) != (C.NUM_EXPERTS,)
            or any(int(mode) not in resources.mode_ids for mode in selected.tolist())
        ):
            raise ValueError(f"layer {layer} has invalid selected {name} modes")
    selection_damage = candidate_metrics[OFFICIAL_SOURCE_DAMAGE_METRIC]
    if selection_damage.dtype != torch.float64 or tuple(selection_damage.shape) != (
        C.NUM_EXPERTS,
    ):
        raise ValueError(f"layer {layer} has invalid selection-corpus damage")
    samples = resources.sample_index.pop(layer - 1)
    routed_rows = index_expert_rows(samples, resources.documents)
    metrics = _allocate_metrics(len(resources.documents))
    metrics["selected_r13"].copy_(selected_r13)
    metrics["selected_r2"].copy_(selected_r2)
    metrics["selection_corpus_damage"].copy_(selection_damage)
    started = time.time()
    with resources.store.open_layer(layer) as source_store:
        with safe_open(candidate_payload, framework="pt", device="cpu") as reader:
            with torch.inference_mode():
                for expert in range(C.NUM_EXPERTS):
                    rows = routed_rows.select(expert)
                    sse, reference, counts = score_selected_expert(
                        source_store,
                        reader,
                        rows,
                        resources.documents,
                        layer=layer,
                        expert=expert,
                        r13_mode_id=int(selected_r13[expert]),
                        r2_mode_id=int(selected_r2[expert]),
                        device=torch.device("cuda", 0),
                        logical_trellis_schema=resources.logical_trellis_schema,
                        codebook=resources.codebook,
                    )
                    metrics["validation_sse"][expert].copy_(sse)
                    metrics["validation_reference_energy"][expert].copy_(reference)
                    metrics["validation_counts"][expert].copy_(counts.to(torch.int32))
                    metrics["validation_gate_square_sum"][expert] = float(
                        rows.gates.double().square().sum()
                    )
                    metrics[VALIDATION_DAMAGE_METRIC][expert] = sse.sum()
                    del rows, sse, reference, counts
                    elapsed = time.time() - started
                    print(
                        f"layer {layer} expert {expert}: {expert + 1}/{C.NUM_EXPERTS} "
                        f"({elapsed / (expert + 1):.1f}s/expert)",
                        flush=True,
                    )
    torch.cuda.empty_cache()

    _atomic_safetensors(output_metrics, metrics)
    histogram = {
        f"R{r13}/R{r2}": int(
            ((selected_r13 == r13) & (selected_r2 == r2)).sum()
        )
        for r13 in resources.mode_ids
        for r2 in resources.mode_ids
    }
    supported = int((metrics["validation_counts"].sum(dim=1) > 0).sum())
    ledger = {
        "kind": VALIDATION_SCORE_KIND,
        "schema_version": VALIDATION_SCORE_SCHEMA_VERSION,
        "complete": True,
        "layer": layer,
        "candidate_payload": str(candidate_payload.resolve()),
        "candidate_metrics": str(candidate_metrics_path.resolve()),
        "metrics": output_metrics.name,
        "experts": C.NUM_EXPERTS,
        "experts_with_validation_rows": supported,
        "validation_documents": len(resources.documents),
        "validation_rows": int(metrics["validation_counts"].sum()),
        "selected_format_histogram": histogram,
        "validation_damage": float(metrics[VALIDATION_DAMAGE_METRIC].sum()),
        "validation_reference_energy": float(
            metrics["validation_reference_energy"].sum()
        ),
        "seconds": time.time() - started,
        "external_validation_used_for_mode_selection": False,
    }
    _atomic_json(output_ledger, ledger)
    print(
        f"layer {layer}: wrote validation for {supported}/{C.NUM_EXPERTS} "
        f"supported experts in {ledger['seconds']:.0f}s",
        flush=True,
    )
    return ledger


def worker(args: argparse.Namespace) -> None:
    layers = (
        list(args.layers)
        if args.layers is not None
        else [
            layer
            for index, layer in enumerate(C.MOE_LAYERS)
            if index % args.worker_count == args.worker
        ]
    )
    pending = [layer for layer in layers if not _layer_complete(args.dest, layer)]
    if not pending:
        return
    manifest = _manifest(args)
    _provenance, documents = _validate_corpus_contract(
        _read_json(args.candidate_pool / "qsrt-candidate-manifest.json"),
        validation_capture=args.validation_capture,
        validation_report_path=args.validation_report,
    )
    resources = _WorkerResources(
        documents=documents,
        sample_index=index_layer_samples(
            args.validation_capture, [layer - 1 for layer in pending]
        ),
        store=OfficialMXFP4Store(
            repo_dir=args.official_repo_dir,
            revision=args.official_revision,
        ),
        logical_trellis_schema=str(
            manifest["candidate_logical_trellis_schema"]
        ),
        codebook=str(manifest["candidate_codebook"]),
        mode_ids=tuple(int(mode) for mode in manifest["candidate_mode_ids"]),
    )
    if _read_json(args.dest / MANIFEST_NAME) != manifest:
        raise ValueError("validation destination manifest drifted")
    for layer in pending:
        score_layer(args, layer, resources=resources)


def parent(args: argparse.Namespace) -> None:
    if args.layers is not None:
        raise ValueError("parent mode always validates all 92 MoE layers")
    # Close the complete candidate pool, including its independently rederived
    # selection decisions, before launching validation workers.
    load_qsrt_candidate_pool(args.candidate_pool)
    manifest = _manifest(args)
    manifest_path = args.dest / MANIFEST_NAME
    if args.dest.exists():
        if not args.resume:
            raise FileExistsError(
                f"destination {args.dest} exists; use --resume for an identical run"
            )
        if not manifest_path.is_file() or _read_json(manifest_path) != manifest:
            raise ValueError("resume destination manifest does not match")
    else:
        args.dest.mkdir(parents=True)
        (args.dest / "scores").mkdir()
        _atomic_json(manifest_path, manifest)

    processes = []
    worker_count = NUM_GPUS * args.workers_per_gpu
    for worker_id in range(worker_count):
        device = worker_id % NUM_GPUS
        # Each worker owns a GPU and only needs host threads for small tensor
        # bookkeeping.  Letting every process create a full 24-thread OpenMP
        # team oversubscribes this host by 12x and starves the CUDA submission
        # threads, turning a ~20 ms expert replay into several seconds.
        environment = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=str(device),
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
        )
        command = [
            sys.executable,
            __file__,
            "--candidate-pool",
            str(args.candidate_pool),
            "--validation-capture",
            str(args.validation_capture),
            "--validation-report",
            str(args.validation_report),
            "--dest",
            str(args.dest),
            "--official-revision",
            args.official_revision,
            "--worker",
            str(worker_id),
            "--worker-count",
            str(worker_count),
        ]
        if args.official_repo_dir is not None:
            command.extend(("--official-repo-dir", str(args.official_repo_dir)))
        if args.resume:
            command.append("--resume")
        processes.append(subprocess.Popen(command, env=environment))
    return_code = 0
    for process in processes:
        return_code |= process.wait()
    if return_code:
        raise SystemExit(return_code)
    missing = [
        layer for layer in C.MOE_LAYERS if not _layer_complete(args.dest, layer)
    ]
    if missing:
        raise ValueError(f"validation workers left incomplete layers: {missing}")
    _atomic_json(
        args.dest / COMPLETION_NAME,
        {
            "kind": VALIDATION_SCORE_KIND,
            "schema_version": VALIDATION_SCORE_SCHEMA_VERSION,
            "complete": True,
            "layers": list(C.MOE_LAYERS),
            "manifest_sha256": _sha256(manifest_path),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--validation-capture", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--official-repo-dir", type=Path)
    parser.add_argument("--official-revision", default=C.REVISION)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--worker-count", type=int, default=NUM_GPUS)
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--layers", type=_parse_ints)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.candidate_pool = args.candidate_pool.resolve()
    args.validation_capture = args.validation_capture.resolve()
    args.validation_report = args.validation_report.resolve()
    args.dest = args.dest.resolve()
    if args.worker_count < 1:
        parser.error("--worker-count must be positive")
    if args.workers_per_gpu < 1:
        parser.error("--workers-per-gpu must be positive")
    if args.worker is not None and not 0 <= args.worker < args.worker_count:
        parser.error("--worker must be in 0..worker-count-1")
    if args.worker is None and args.worker_count != NUM_GPUS:
        parser.error("parent mode derives --worker-count from --workers-per-gpu")
    if args.layers is not None and any(
        layer not in C.MOE_LAYERS for layer in args.layers
    ):
        parser.error("--layers must contain decoder layers 1..92")
    return args


def main() -> None:
    args = parse_args()
    if args.worker is None:
        parent(args)
    else:
        if not args.dest.is_dir() or not (args.dest / MANIFEST_NAME).is_file():
            raise ValueError("worker destination was not initialized by the parent")
        worker(args)


if __name__ == "__main__":
    main()
