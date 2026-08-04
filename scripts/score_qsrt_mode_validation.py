#!/usr/bin/env python3
"""Re-encode selected nonzero formats as R0/R0 on untouched validation data.

This is deliberately separate from selected-candidate validation.  The latter
is the keep-tier damage estimate.  This pass verifies the codec policy itself:
for every assignment whose fit/confirmation selector accepted rate shifting on
either axis, it deterministically reproduces the original R0/R0
counterfactual and scores that control and the persisted selected payload on
the same document-disjoint rows.

Official weights remain an offline source and are opened one expert matrix at
a time.  The official model is never instantiated.
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
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from kquant import constants as C
from kquant.capture import index_layer_samples, load_layer_hessians
from kquant.qsrt_candidates import (
    RequestPartition,
    functional_sse_by_request,
    layer_fallback_contexts,
    partition_requests,
    request_documents,
    select_expert_rows,
)
from kquant.pack.qsrt_pool import load_qsrt_candidate_pool
from kquant.pack.qsrt_candidates import (
    CANDIDATE_POOL_KIND,
    OFFICIAL_SOURCE_DAMAGE_METRIC,
    QSRTCandidateEncoding,
    candidate_tensor_name,
    encode_phase1_expert,
)
from kquant.pack.qsrt_mode_validation import (
    COMPLETION_FILENAME,
    MANIFEST_FILENAME,
    MODE_VALIDATION_KIND,
    MODE_VALIDATION_METRIC,
    MODE_VALIDATION_SCHEMA_VERSION,
    load_qsrt_mode_validation_scores,
)
from kquant.pack.qsrt_validation import (
    QSRTValidationScores,
    load_qsrt_validation_scores,
    official_expert_output,
)
from kquant.source_weights import OfficialMXFP4Store
from kquant.tp_simulator import situ


@dataclass
class _WorkerResources:
    partition: RequestPartition
    validation_documents: dict[int, str]
    training_index: object
    validation_index: object
    store: OfficialMXFP4Store
    quantizer_module: object


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


def _sha256_files(paths: tuple[Path, ...]) -> str:
    """Bind the validation run to every source file defining the codec."""

    digest = hashlib.sha256()
    digest.update(b"kquant-qsrt-encoder-sources-v2\0")
    for path in paths:
        encoded = path.as_posix().encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
        digest.update(bytes.fromhex(_sha256(path)))
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


def _parse_devices(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(part) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated GPU indices"
        ) from exc
    if (
        not result
        or len(set(result)) != len(result)
        or any(device < 0 for device in result)
    ):
        raise argparse.ArgumentTypeError("GPU indices must be nonnegative and unique")
    return result


def _layer_paths(root: Path, layer: int) -> tuple[Path, Path]:
    stem = f"qsrt-layer-{layer:05d}.mode-validation"
    return root / "scores" / f"{stem}.safetensors", root / "scores" / f"{stem}.json"


def _layer_complete(root: Path, layer: int) -> bool:
    metrics, ledger = _layer_paths(root, layer)
    if not metrics.is_file() or not ledger.is_file():
        return False
    value = _read_json(ledger)
    return bool(
        value.get("kind") == MODE_VALIDATION_KIND
        and value.get("schema_version") == MODE_VALIDATION_SCHEMA_VERSION
        and value.get("complete")
        and value.get("layer") == layer
        and value.get("metrics") == metrics.name
    )


def _candidate_paths(root: Path, layer: int) -> tuple[Path, Path]:
    stem = f"qsrt-layer-{layer:05d}"
    return (
        root / "candidates" / f"{stem}.safetensors",
        root / "candidates" / f"{stem}.metrics.safetensors",
    )


def _selected_validation_metrics_path(root: Path, layer: int) -> Path:
    return (
        root
        / "scores"
        / f"qsrt-layer-{layer:05d}.validation.safetensors"
    )


def _load_quantizer_module(root: Path):
    # Match the all-expert encoder without importing exllamav3's serving stack.
    from kquant.exl3_loader import load_qsrt_encoder

    return load_qsrt_encoder(root)


def _validate_report_contract(manifest: dict) -> tuple[dict, dict, RequestPartition]:
    training_report_path = Path(str(manifest["training_report"]))
    validation_report_path = Path(str(manifest["validation_report"]))
    training_report = _read_json(training_report_path)
    validation_report = _read_json(validation_report_path)
    for name, report in (
        ("training", training_report),
        ("validation", validation_report),
    ):
        if (
            report.get("kind") != "kquant_interim_calibration_corpus_run"
            or not report.get("finalized")
        ):
            raise ValueError(f"{name} corpus report is not a finalized calibration run")
    if Path(str(training_report.get("capture_dir", ""))).resolve() != Path(
        str(manifest["training_capture"])
    ).resolve():
        raise ValueError("training report and mode-validation capture disagree")
    if Path(str(validation_report.get("capture_dir", ""))).resolve() != Path(
        str(manifest["validation_capture"])
    ).resolve():
        raise ValueError("validation report and mode-validation capture disagree")
    if _sha256(training_report_path) != manifest["training_report_sha256"]:
        raise ValueError("training corpus report changed after mode-validation setup")
    if _sha256(validation_report_path) != manifest["validation_report_sha256"]:
        raise ValueError("validation corpus report changed after mode-validation setup")
    training_documents = request_documents(training_report)
    validation_documents = request_documents(validation_report, deduplicate=True)
    if set(training_documents.values()) & set(validation_documents.values()):
        raise ValueError("training and validation documents overlap")
    partition = partition_requests(training_documents)
    return training_report, validation_report, partition


def _manifest(
    args: argparse.Namespace,
    pool: object,
    validation: QSRTValidationScores,
) -> dict:
    candidate_manifest = pool.manifest
    training_report = Path(str(candidate_manifest["training_report"])).resolve()
    training_capture = Path(str(candidate_manifest["capture"])).resolve()
    hessians = Path(str(candidate_manifest["hessians"])).resolve()
    validation_report = Path(str(validation.manifest["validation_report"])).resolve()
    validation_capture = Path(str(validation.manifest["validation_capture"])).resolve()
    exllamav3_root = Path(str(candidate_manifest["exllamav3_root"])).resolve()
    source_store = OfficialMXFP4Store(
        repo_dir=args.official_repo_dir,
        revision=C.REVISION,
    )
    source_index = source_store.root / C.INDEX_FILE
    project = Path(__file__).resolve().parents[1]
    quantizer_sources = (
        project / "kquant/exl3_encoder_backend.py",
        project / "kquant/exl3_loader.py",
        project / "kquant/ldlq.py",
        project / "kquant/sqg_quantizer.py",
        project / "kquant/sqg_e4m3.py",
        project / "kquant/pack/qsrt_encoder.py",
        project / "kquant/csrc/sqg_quantize.cpp",
        project / "kquant/csrc/sqg_quantize.cu",
        project / "kquant/csrc/qsrt_quantize_tiles_kernel.cuh",
        project / "kquant/csrc/exl3_compat/util.h",
        project / "kquant/csrc/exl3_compat/util.cuh",
        project / "kquant/csrc/exl3_compat/quant/codebook.cuh",
        exllamav3_root / "exllamav3/util/hadamard.py",
        exllamav3_root / "exllamav3/util/tensor.py",
    )
    provenance_files = {
        "training_report_sha256": _sha256(training_report),
        "training_capture_manifest_sha256": _sha256(training_capture / "manifest.json"),
        "hessian_manifest_sha256": _sha256(hessians / "manifest.json"),
        "validation_report_sha256": _sha256(validation_report),
        "validation_capture_manifest_sha256": _sha256(
            validation_capture / "manifest.json"
        ),
        "official_index_sha256": _sha256(source_index),
        "quantizer_source_sha256": _sha256(quantizer_sources[0]),
        "quantizer_codec_sources_sha256": _sha256_files(quantizer_sources),
    }
    result = {
        "kind": MODE_VALIDATION_KIND,
        "schema_version": MODE_VALIDATION_SCHEMA_VERSION,
        "candidate_pool": str(pool.root),
        "candidate_pool_content_sha256": pool.content_sha256,
        "candidate_logical_trellis_schema": pool.trellis_schema,
        "candidate_codebook": pool.codebook,
        "candidate_mode_ids": list(pool.mode_ids),
        "selected_validation_scores": str(validation.root),
        "selected_validation_score_set_sha256": validation.content_sha256,
        "selected_validation_schema_version": validation.manifest["schema_version"],
        "source_model": C.MODEL_ID,
        "source_revision": C.REVISION,
        "source_checkpoint": str(source_store.root),
        "training_capture": str(training_capture),
        "training_report": str(training_report),
        "hessians": str(hessians),
        "resident_teacher_checkpoint": candidate_manifest[
            "resident_teacher_checkpoint"
        ],
        "validation_capture": str(validation_capture),
        "validation_report": str(validation_report),
        "validation_documents": validation.manifest["validation_documents"],
        "exllamav3_root": str(exllamav3_root),
        "mode_ids": candidate_manifest["mode_ids"],
        "selection_contract": {
            name: candidate_manifest[name]
            for name in (
                "min_fit_documents",
                "min_confirmation_documents",
                "minimum_improvement",
                "bootstrap_replicates",
            )
        },
        "tp_size": 12,
        "metric": MODE_VALIDATION_METRIC,
        "audit_population": (
            "every assignment with selected r13>0 or selected r2>0"
        ),
        "external_validation_used_for_mode_selection": False,
        "allowed_action": (
            "global codec-policy accept, repeat, or reject; "
            "no per-expert retuning"
        ),
        "official_source_residency": "one expert matrix at a time",
        "r0_reencode_contract": (
            "seed each layer with expert 0 under the original shared-scale scope; "
            "require exact training evidence and shared-hidden-scale closure"
        ),
        **provenance_files,
    }
    _validate_report_contract(result)
    hessian_manifest = _read_json(hessians / "manifest.json")
    if (
        Path(str(hessian_manifest.get("source_capture", ""))).resolve()
        != training_capture
    ):
        raise ValueError("Hessian bundle was built from another training capture")
    return result


def _allocate_metrics(selected: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    selected_r13 = selected["selected_r13"].clone()
    selected_r2 = selected["selected_r2"].clone()
    audited = (selected_r13 != 0) | (selected_r2 != 0)
    selected_sse = selected["validation_sse"].clone()
    reference = selected["validation_reference_energy"].clone()
    counts = selected["validation_counts"].clone()
    return {
        "expert_ids": torch.arange(C.NUM_EXPERTS, dtype=torch.int16),
        "selected_r13": selected_r13,
        "selected_r2": selected_r2,
        "audited": audited,
        "selected_validation_sse": selected_sse,
        "r0_validation_sse": selected_sse.clone(),
        "validation_reference_energy": reference,
        "validation_counts": counts,
        "selected_validation_damage": selected_sse.sum(dim=1),
        "r0_validation_damage": selected_sse.sum(dim=1).clone(),
    }


def _require_exact(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.dtype != expected.dtype or tuple(actual.shape) != tuple(expected.shape):
        raise ValueError(
            f"{name} shape/dtype drifted: {actual.dtype} {tuple(actual.shape)} != "
            f"{expected.dtype} {tuple(expected.shape)}"
        )
    if not torch.equal(actual.cpu(), expected.cpu()):
        difference = (actual.double().cpu() - expected.double().cpu()).abs()
        raise ValueError(
            f"{name} failed exact closure: mismatched="
            f"{int(torch.count_nonzero(actual.cpu() != expected.cpu()))}/"
            f"{actual.numel()}, max_abs={float(difference.max())}"
        )


def _verify_training_r0(
    candidate: QSRTCandidateEncoding,
    stored: dict[str, torch.Tensor],
    *,
    expert: int,
) -> None:
    mode_ids = stored["mode_ids"].tolist()
    if 0 not in mode_ids:
        raise ValueError("candidate metrics do not contain R0")
    r0 = mode_ids.index(0)
    if (
        candidate.evaluated_modes != (0,)
        or candidate.evaluated_formats != ((0, 0),)
        or candidate.selection.selected != (0, 0)
    ):
        raise ValueError("matched-R0 re-encode did not produce only R0/R0")
    pairs = (
        ("block contexts", candidate.block_contexts, stored["block_contexts"][expert]),
        ("block scores", candidate.block_scores, stored["block_scores"][expert]),
        (
            "fit R0/R0 SSE",
            candidate.fit_sse[(0, 0)],
            stored["fit_sse"][expert, r0, r0],
        ),
        (
            "confirmation R0/R0 SSE",
            candidate.confirmation_sse[(0, 0)],
            stored["confirmation_sse"][expert, r0, r0],
        ),
        (
            "fit reference energy",
            candidate.fit_reference_energy,
            stored["fit_reference_energy"][expert],
        ),
        (
            "confirmation reference energy",
            candidate.confirmation_reference_energy,
            stored["confirmation_reference_energy"][expert],
        ),
        (
            "fit counts",
            candidate.fit_counts.to(torch.int32),
            stored["fit_counts"][expert],
        ),
        (
            "confirmation counts",
            candidate.confirmation_counts.to(torch.int32),
            stored["confirmation_counts"][expert],
        ),
    )
    for name, actual, expected in pairs:
        _require_exact(f"expert {expert} {name}", actual, expected)


def _payload_digest(candidate: QSRTCandidateEncoding) -> str:
    digest = hashlib.sha256()
    digest.update(b"kquant-matched-r0-payload-v1\0")
    for matrix in C.EXPERT_MATRICES:
        encoding = candidate.expert.matrices[matrix]
        for part in ("trellis", "suh", "svh"):
            tensor = encoding.tensors[part].detach().cpu().contiguous()
            label = f"{matrix}.{part}".encode("utf-8")
            digest.update(len(label).to_bytes(4, "little"))
            digest.update(label)
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _verify_stored_payload_scales(
    reader: object,
    candidate: QSRTCandidateEncoding,
    *,
    layer: int,
    expert: int,
) -> None:
    hidden_parts = {"w1": "suh", "w3": "suh", "w2": "svh"}
    for matrix, part in hidden_parts.items():
        stored = reader.get_tensor(candidate_tensor_name(layer, expert, matrix, part))
        actual = candidate.expert.matrices[matrix].tensors[part]
        _require_exact(f"expert {expert} {matrix} shared hidden scale", actual, stored)


def _verify_full_stored_r0_payload(
    reader: object,
    candidate: QSRTCandidateEncoding,
    *,
    layer: int,
    expert: int,
) -> None:
    for matrix in C.EXPERT_MATRICES:
        for part in ("trellis", "suh", "svh"):
            stored = reader.get_tensor(
                candidate_tensor_name(layer, expert, matrix, part)
            )
            actual = candidate.expert.matrices[matrix].tensors[part]
            _require_exact(f"expert {expert} stored R0 {matrix}.{part}", actual, stored)


def _score_r0_candidate(
    candidate: QSRTCandidateEncoding,
    rows: object,
    requests: dict[int, str],
    store: OfficialMXFP4Store,
    *,
    layer: int,
    expert: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if rows.rows == 0:
        count = len(requests)
        return (
            torch.zeros(count, dtype=torch.float64),
            torch.zeros(count, dtype=torch.float64),
            torch.zeros(count, dtype=torch.int64),
        )
    inputs = rows.inputs.float().to(device=device)
    gates = rows.gates.float().to(device=device)
    reference = official_expert_output(
        store,
        inputs,
        layer=layer,
        expert=expert,
        device=device,
    )
    matrices = candidate.expert.matrices
    gate = F.linear(inputs, matrices["w1"].reconstruction)
    up = F.linear(inputs, matrices["w3"].reconstruction)
    output = F.linear(situ(gate, up), matrices["w2"].reconstruction)
    result = functional_sse_by_request(
        reference,
        output,
        gates,
        rows.request_steps,
        requests,
    )
    del inputs, gates, reference, gate, up, output
    return result


def _encode_r0(
    manifest: dict,
    resources: _WorkerResources,
    samples: object,
    *,
    layer: int,
    expert: int,
    global_h13: torch.Tensor,
    global_h2: torch.Tensor,
    fallback_contexts: torch.Tensor,
    fallback_scores: torch.Tensor,
    device: torch.device,
) -> QSRTCandidateEncoding:
    contract = manifest["selection_contract"]
    return encode_phase1_expert(
        resources.store,
        samples,
        layer=layer,
        expert=expert,
        partition=resources.partition,
        global_h13=global_h13,
        global_h2=global_h2,
        fallback_block_contexts=fallback_contexts,
        fallback_block_scores=fallback_scores,
        device=device,
        shared_scale_scope=(
            CANDIDATE_POOL_KIND,
            str(Path(str(manifest["candidate_pool"])).resolve()),
            layer,
        ),
        mode_ids=(0,),
        min_fit_documents=int(contract["min_fit_documents"]),
        min_confirmation_documents=int(contract["min_confirmation_documents"]),
        bootstrap_replicates=int(contract["bootstrap_replicates"]),
        minimum_improvement=float(contract["minimum_improvement"]),
        quantizer_module=resources.quantizer_module,
        logical_trellis_schema=str(manifest["candidate_logical_trellis_schema"]),
        codebook=str(manifest["candidate_codebook"]),
    )


def score_layer(
    args: argparse.Namespace,
    manifest: dict,
    layer: int,
    *,
    resources: _WorkerResources | None,
) -> dict:
    output_metrics, output_ledger = _layer_paths(args.dest, layer)
    if _layer_complete(args.dest, layer):
        print(f"layer {layer}: matched-R0 validation complete, skip", flush=True)
        return _read_json(output_ledger)
    if output_metrics.exists() or output_ledger.exists():
        raise FileExistsError(
            f"layer {layer} has incomplete mode-validation output; "
            "use a fresh destination"
        )

    candidate_payload_path, candidate_metrics_path = _candidate_paths(
        args.candidate_pool, layer
    )
    selected_metrics_path = _selected_validation_metrics_path(
        args.validation_scores, layer
    )
    candidate_metrics = load_file(str(candidate_metrics_path), device="cpu")
    selected_metrics = load_file(str(selected_metrics_path), device="cpu")
    metrics = _allocate_metrics(selected_metrics)
    selected_r13 = metrics["selected_r13"]
    selected_r2 = metrics["selected_r2"]
    audited_mask = metrics["audited"]
    audited_experts = torch.nonzero(audited_mask, as_tuple=False).flatten().tolist()
    started = time.time()
    evidence: dict[str, dict] = {}
    control_expert: int | None = None

    if audited_experts:
        if resources is None:
            raise ValueError("audited layer has no worker resources")
        training_samples = resources.training_index.pop(layer - 1)
        validation_samples = resources.validation_index.pop(layer - 1)
        global_h13, global_h2 = load_layer_hessians(
            Path(str(manifest["hessians"])), layer
        )
        fallback_contexts, fallback_scores = layer_fallback_contexts(
            training_samples, resources.partition.fit
        )
        device = torch.device("cuda", 0)
        uniform_experts = torch.nonzero(
            ~audited_mask, as_tuple=False
        ).flatten().tolist()
        control_expert = uniform_experts[0] if uniform_experts else None
        control = [control_expert] if control_expert is not None else []
        order = list(dict.fromkeys([0, *control, *audited_experts]))
        with safe_open(candidate_payload_path, framework="pt", device="cpu") as reader:
            for position, expert in enumerate(order):
                candidate = _encode_r0(
                    manifest,
                    resources,
                    training_samples,
                    layer=layer,
                    expert=expert,
                    global_h13=global_h13,
                    global_h2=global_h2,
                    fallback_contexts=fallback_contexts,
                    fallback_scores=fallback_scores,
                    device=device,
                )
                _verify_training_r0(candidate, candidate_metrics, expert=expert)
                _verify_stored_payload_scales(
                    reader, candidate, layer=layer, expert=expert
                )
                if expert == control_expert:
                    _verify_full_stored_r0_payload(
                        reader, candidate, layer=layer, expert=expert
                    )
                if expert in audited_experts:
                    rows = select_expert_rows(
                        validation_samples,
                        expert,
                        resources.validation_documents,
                    )
                    r0_sse, reference, counts = _score_r0_candidate(
                        candidate,
                        rows,
                        resources.validation_documents,
                        resources.store,
                        layer=layer,
                        expert=expert,
                        device=device,
                    )
                    _require_exact(
                        f"expert {expert} validation reference energy",
                        reference,
                        selected_metrics["validation_reference_energy"][expert],
                    )
                    _require_exact(
                        f"expert {expert} validation counts",
                        counts.to(torch.int32),
                        selected_metrics["validation_counts"][expert],
                    )
                    metrics["r0_validation_sse"][expert].copy_(r0_sse)
                    metrics["r0_validation_damage"][expert] = r0_sse.sum()
                    selected_total = float(
                        metrics["selected_validation_damage"][expert]
                    )
                    r0_total = float(metrics["r0_validation_damage"][expert])
                    evidence[str(expert)] = {
                        "selected_r13": int(selected_r13[expert]),
                        "selected_r2": int(selected_r2[expert]),
                        "training_r0_metrics_exact": True,
                        "shared_hidden_scales_exact": True,
                        "r0_payload_sha256": _payload_digest(candidate),
                        "validation_rows": int(counts.sum()),
                        "validation_documents": int((counts > 0).sum()),
                        "selected_validation_damage": selected_total,
                        "r0_validation_damage": r0_total,
                        "relative_improvement": (
                            1.0 - selected_total / r0_total
                            if r0_total > 0
                            else None
                        ),
                    }
                    del rows, r0_sse, reference, counts
                del candidate
                elapsed = time.time() - started
                print(
                    f"layer {layer} R0 re-encode {position + 1}/{len(order)} "
                    f"expert {expert} ({elapsed / (position + 1):.1f}s/encode)",
                    flush=True,
                )
        del training_samples, validation_samples, global_h13, global_h2
    torch.cuda.empty_cache()

    _atomic_safetensors(output_metrics, metrics)
    audited_mask = metrics["audited"]
    selected_audited = float(
        metrics["selected_validation_damage"][audited_mask].sum()
    )
    r0_audited = float(metrics["r0_validation_damage"][audited_mask].sum())
    ledger = {
        "kind": MODE_VALIDATION_KIND,
        "schema_version": MODE_VALIDATION_SCHEMA_VERSION,
        "complete": True,
        "layer": layer,
        "metrics": output_metrics.name,
        "experts": C.NUM_EXPERTS,
        "validation_documents": int(manifest["validation_documents"]),
        "validation_rows": int(metrics["validation_counts"].sum()),
        "audited_experts": audited_experts,
        "audited_expert_count": len(audited_experts),
        "r0_payload_control_expert": control_expert,
        "r0_payload_control_exact": control_expert is not None or not audited_experts,
        "selected_validation_damage": float(
            metrics["selected_validation_damage"].sum()
        ),
        "r0_validation_damage": float(metrics["r0_validation_damage"].sum()),
        "audited_selected_validation_damage": selected_audited,
        "audited_r0_validation_damage": r0_audited,
        "audited_relative_improvement": (
            1.0 - selected_audited / r0_audited if r0_audited > 0 else None
        ),
        "reencoded_r0_evidence": evidence,
        "external_validation_used_for_mode_selection": False,
        "seconds": time.time() - started,
    }
    _atomic_json(output_ledger, ledger)
    print(
        f"layer {layer}: wrote matched-R0 evidence for "
        f"{len(audited_experts)} selected rate-shifted experts in {ledger['seconds']:.0f}s",
        flush=True,
    )
    return ledger


def _audited_layers(candidate_pool: Path) -> list[int]:
    result = []
    for layer in C.MOE_LAYERS:
        _payload, metrics_path = _candidate_paths(candidate_pool, layer)
        metrics = load_file(str(metrics_path), device="cpu")
        if bool(
            torch.any(
                (metrics["selected_r13"] != 0)
                | (metrics["selected_r2"] != 0)
            )
        ):
            result.append(layer)
    return result


def worker(args: argparse.Namespace) -> None:
    manifest = _read_json(args.dest / MANIFEST_FILENAME)
    _validate_report_contract(manifest)
    layers = [
        layer
        for index, layer in enumerate(C.MOE_LAYERS)
        if index % args.worker_count == args.worker_index
    ]
    pending = [layer for layer in layers if not _layer_complete(args.dest, layer)]
    if not pending:
        return
    audited = set(_audited_layers(args.candidate_pool)) & set(pending)
    resources = None
    if audited:
        _training_report, validation_report, partition = _validate_report_contract(
            manifest
        )
        resources = _WorkerResources(
            partition=partition,
            validation_documents=request_documents(
                validation_report, deduplicate=True
            ),
            training_index=index_layer_samples(
                Path(str(manifest["training_capture"])),
                [layer - 1 for layer in sorted(audited)],
            ),
            validation_index=index_layer_samples(
                Path(str(manifest["validation_capture"])),
                [layer - 1 for layer in sorted(audited)],
            ),
            store=OfficialMXFP4Store(
                repo_dir=args.official_repo_dir,
                revision=C.REVISION,
            ),
            quantizer_module=_load_quantizer_module(
                Path(str(manifest["exllamav3_root"]))
            ),
        )
    for layer in pending:
        score_layer(args, manifest, layer, resources=resources)


def parent(args: argparse.Namespace) -> None:
    if args.worker_index is not None:
        raise ValueError("parent mode cannot receive worker arguments")
    pool = load_qsrt_candidate_pool(args.candidate_pool)
    validation = load_qsrt_validation_scores(args.validation_scores, pool)
    manifest = _manifest(args, pool, validation)
    manifest_path = args.dest / MANIFEST_FILENAME
    if args.dest.exists():
        if not args.resume:
            raise FileExistsError(
                f"destination {args.dest} exists; use --resume for an identical run"
            )
        if not manifest_path.is_file() or _read_json(manifest_path) != manifest:
            raise ValueError("mode-validation resume manifest drifted")
    else:
        args.dest.mkdir(parents=True)
        (args.dest / "scores").mkdir()
        _atomic_json(manifest_path, manifest)

    completion_path = args.dest / COMPLETION_FILENAME
    if completion_path.exists():
        result = load_qsrt_mode_validation_scores(
            args.dest,
            pool,
            validation,
        )
        print(
            f"already sealed {args.dest}: {int(result.audited.sum())} "
            f"matched-R0 audits, digest {result.content_sha256}",
            flush=True,
        )
        return

    processes = []
    worker_count = len(args.devices) * args.workers_per_device
    for worker_index in range(worker_count):
        device = args.devices[worker_index % len(args.devices)]
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device))
        command = [
            sys.executable,
            __file__,
            "--candidate-pool",
            str(args.candidate_pool),
            "--validation-scores",
            str(args.validation_scores),
            "--dest",
            str(args.dest),
            "--worker-index",
            str(worker_index),
            "--worker-count",
            str(worker_count),
        ]
        if args.official_repo_dir is not None:
            command.extend(("--official-repo-dir", str(args.official_repo_dir)))
        processes.append(subprocess.Popen(command, env=environment))
    return_code = 0
    for process in processes:
        return_code |= process.wait()
    if return_code:
        raise SystemExit(return_code)
    missing = [layer for layer in C.MOE_LAYERS if not _layer_complete(args.dest, layer)]
    if missing:
        raise ValueError(f"mode-validation workers left incomplete layers: {missing}")
    _atomic_json(
        completion_path,
        {
            "kind": MODE_VALIDATION_KIND,
            "schema_version": MODE_VALIDATION_SCHEMA_VERSION,
            "complete": True,
            "layers": list(C.MOE_LAYERS),
            "manifest_sha256": _sha256(manifest_path),
        },
    )
    result = load_qsrt_mode_validation_scores(
        args.dest,
        pool,
        validation,
    )
    print(
        f"sealed {args.dest}: {int(result.audited.sum())} matched-R0 audits, "
        f"digest {result.content_sha256}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--validation-scores", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--official-repo-dir", type=Path)
    parser.add_argument(
        "--devices",
        type=_parse_devices,
        default=_parse_devices(",".join(str(index) for index in range(12))),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers-per-device", type=int, default=1)
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--worker-count", type=int)
    args = parser.parse_args()
    args.candidate_pool = args.candidate_pool.resolve()
    args.validation_scores = args.validation_scores.resolve()
    args.dest = args.dest.resolve()
    if (args.worker_index is None) != (args.worker_count is None):
        parser.error("worker index and count must be supplied together")
    if args.workers_per_device < 1:
        parser.error("--workers-per-device must be positive")
    if args.worker_index is not None and not 0 <= args.worker_index < args.worker_count:
        parser.error("worker index must lie in 0..worker-count-1")
    return args


def main() -> None:
    args = parse_args()
    if args.worker_index is None:
        parent(args)
    else:
        if not args.dest.is_dir() or not (args.dest / MANIFEST_FILENAME).is_file():
            raise ValueError("worker destination was not initialized by the parent")
        worker(args)


if __name__ == "__main__":
    main()
