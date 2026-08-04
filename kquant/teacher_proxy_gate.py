"""Document-clustered gate for the Kimi-K3 interim calibration teacher."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from kquant.constants import NUM_EXPERTS
from kquant.teacher_proxy import (
    POST_SITU_STAGE,
    ROUTE_IDS_STAGE,
    ROUTE_WEIGHTS_STAGE,
    compare_teacher_proxy_traces,
)
from kquant.teacher_proxy_suite import (
    LoadedTeacherProxySuite,
    file_sha256,
    load_teacher_proxy_suite,
)
from kquant.trace_compare import TraceFormatError, TraceKey, TraceRun, load_trace


TEACHER_PROXY_GATE_SCHEMA_VERSION = 1
TEACHER_PROXY_GATE_KIND = "kquant_teacher_proxy_suite_gate"
_DOCUMENT_STAGES = frozenset(
    {ROUTE_IDS_STAGE, ROUTE_WEIGHTS_STAGE, POST_SITU_STAGE}
)


@dataclass(frozen=True)
class TeacherProxyThresholds:
    route_assignment_retention_fraction: float = 0.85
    sparse_applied_gate_cosine: float = 0.90
    record_rank_spearman: float = 0.75
    r3_donor_record_overlap_fraction: float = 1 / 3
    r3_recipient_record_overlap_fraction: float = 1 / 3
    expert_case_support_fraction: float = 0.75
    expert_record_rank_spearman_median: float = 0.65
    expert_record_rank_spearman_p10: float = 0.20
    expert_r3_donor_record_overlap_fraction: float = 0.40
    expert_r3_recipient_record_overlap_fraction: float = 0.40


_DOCUMENT_BOOTSTRAP_METRICS = frozenset(
    {
        "route_assignment_retention_fraction",
        "sparse_applied_gate_cosine",
        "record_rank_spearman",
        "r3_donor_record_overlap_fraction",
        "r3_recipient_record_overlap_fraction",
    }
)


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty metric")
    return sum(values) / len(values)


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take a quantile of an empty metric")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _read_run(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise TraceFormatError(f"{path}: cannot read run manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise TraceFormatError(f"{path}: run manifest must be an object")
    return payload


def _validate_run(
    *,
    label: str,
    root: Path,
    run: dict[str, Any],
    suite: LoadedTeacherProxySuite,
    expected_interim: Path | None,
) -> None:
    if run.get("schema_version") != 1 or run.get("status") != "complete":
        raise TraceFormatError(f"{root}: {label} streamed run is incomplete")
    expected_shape = list(suite.input_ids.shape)
    if run.get("input_ids_shape") != expected_shape:
        raise TraceFormatError(f"{root}: {label} input batch shape differs")
    if run.get("input_ids") != suite.input_ids.flatten().tolist():
        raise TraceFormatError(f"{root}: {label} token IDs differ from suite")
    input_suite = run.get("input_suite")
    expected_hashes = [
        document["document_hash"] for document in suite.manifest["documents"]
    ]
    expected_suite_fields = {
        "file_sha256": suite.file_sha256,
        "kind": suite.manifest["kind"],
        "schema_version": suite.manifest["schema_version"],
        "suite_token_hash_sha256": suite.manifest["suite_token_hash_sha256"],
        "batch_shape": suite.manifest["batch_shape"],
        "document_hashes": expected_hashes,
    }
    if not isinstance(input_suite, dict) or any(
        input_suite.get(field) != value
        for field, value in expected_suite_fields.items()
    ):
        raise TraceFormatError(f"{root}: {label} suite provenance differs")
    if run.get("trace_layers") != suite.manifest["trace_layers"]:
        raise TraceFormatError(f"{root}: {label} trace-layer coverage differs")
    if run.get("end_layer") != suite.manifest["end_layer"]:
        raise TraceFormatError(f"{root}: {label} end layer differs")
    if not run.get("capture_routed_post_situ"):
        raise TraceFormatError(f"{root}: {label} omitted routed post-SiTU capture")
    checkpoint = Path(str(run.get("checkpoint", ""))).resolve()
    expert = Path(str(run.get("expert_checkpoint", ""))).resolve()
    nonexpert = Path(str(run.get("nonexpert_checkpoint", ""))).resolve()
    if label == "official":
        if expert != checkpoint or nonexpert != checkpoint:
            raise TraceFormatError(
                f"{root}: official run did not use the official checkpoint "
                "for both expert and non-expert weights"
            )
    else:
        if expected_interim is None:
            raise AssertionError("expected interim path is required")
        if expert != expected_interim or nonexpert != expected_interim:
            raise TraceFormatError(
                f"{root}: interim run sources {expert}, {nonexpert} differ "
                f"from {expected_interim}"
            )
        if run.get("exl3_manifest") is None:
            raise TraceFormatError(f"{root}: interim run has no EXL3 manifest")


def _validate_trace(
    *,
    label: str,
    trace: TraceRun,
    run: dict[str, Any],
    suite: LoadedTeacherProxySuite,
) -> None:
    if trace.world_size != 1 or trace.captured_ranks != (0,):
        raise TraceFormatError(f"{trace.root}: {label} must be a TP1 streamed trace")
    if trace.manifest.get("layers") != suite.manifest["trace_layers"]:
        raise TraceFormatError(f"{trace.root}: {label} trace manifest layers differ")
    if trace.manifest.get("max_tokens") != int(suite.input_ids.numel()):
        raise TraceFormatError(f"{trace.root}: {label} trace token count differs")
    if Path(str(trace.manifest.get("checkpoint", ""))).resolve() != Path(
        str(run["expert_checkpoint"])
    ).resolve():
        raise TraceFormatError(f"{trace.root}: {label} trace checkpoint differs")
    expected_layers = set(suite.manifest["trace_layers"])
    for stage in _DOCUMENT_STAGES:
        layers = {
            key.layer
            for key in trace.tensors
            if key.call == 0 and key.stage == stage
        }
        if layers != expected_layers:
            raise TraceFormatError(
                f"{trace.root}: {label} {stage} layers {sorted(layers)} "
                f"!= {sorted(expected_layers)}"
            )


def _document_trace(
    run: TraceRun,
    *,
    document_index: int,
    document_count: int,
    tokens_per_document: int,
) -> TraceRun:
    start = document_index * tokens_per_document
    stop = start + tokens_per_document
    expected_rows = document_count * tokens_per_document
    tensors: dict[TraceKey, dict[int, torch.Tensor]] = {}
    for key, rank_tensors in run.tensors.items():
        if key.call != 0 or key.stage not in _DOCUMENT_STAGES:
            continue
        sliced: dict[int, torch.Tensor] = {}
        for rank, value in rank_tensors.items():
            if value.ndim < 1 or value.shape[0] != expected_rows:
                raise TraceFormatError(
                    f"{run.root}: {key} has leading dimension "
                    f"{value.shape[0] if value.ndim else None}, expected "
                    f"{expected_rows}"
                )
            sliced[rank] = value[start:stop]
        tensors[key] = sliced
    manifest = dict(run.manifest)
    manifest["max_tokens"] = tokens_per_document
    manifest["suite_document_index"] = document_index
    return TraceRun(
        root=run.root / f"suite-document-{document_index:03d}",
        world_size=run.world_size,
        captured_ranks=run.captured_ranks,
        manifest=manifest,
        tensors=tensors,
    )


def _document_summary(report: dict[str, Any]) -> dict[str, float]:
    post_situ = report["post_situ_summary"]
    if not post_situ.get("available"):
        raise TraceFormatError("document comparison has no post-SiTU metrics")
    layers = post_situ.get("layers")
    if not isinstance(layers, list) or not layers:
        raise TraceFormatError("document comparison has no post-SiTU layers")
    r3 = [layer["rate_ladder_record_overlap"]["R3"] for layer in layers]
    return {
        "route_assignment_retention_fraction": float(
            report["route_summary"]["route_assignment_retention_fraction"]
        ),
        "route_row_jaccard_mean": float(
            report["route_summary"]["row_jaccard_mean"]
        ),
        "sparse_applied_gate_cosine": float(
            report["route_summary"]["sparse_applied_gate"]["cosine"]
        ),
        "record_rank_spearman": _mean(
            [float(layer["record_rank_spearman"]) for layer in layers]
        ),
        "r3_donor_record_overlap_fraction": _mean(
            [float(value["donor_record_overlap_fraction"]) for value in r3]
        ),
        "r3_recipient_record_overlap_fraction": _mean(
            [float(value["recipient_record_overlap_fraction"]) for value in r3]
        ),
    }


def _cluster_bootstrap(
    documents: list[dict[str, float]],
    *,
    samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    if len(documents) < 3:
        raise ValueError("document bootstrap requires at least three documents")
    if samples < 1000:
        raise ValueError("document bootstrap requires at least 1,000 samples")
    metrics = tuple(documents[0])
    if any(tuple(document) != metrics for document in documents):
        raise ValueError("document metric coverage differs")
    rng = random.Random(seed)
    distributions = {metric: [] for metric in metrics}
    for unused_sample in range(samples):
        selected = [rng.randrange(len(documents)) for _ in documents]
        for metric in metrics:
            distributions[metric].append(
                _mean([documents[index][metric] for index in selected])
            )
    return {
        metric: {
            "estimate": _mean([document[metric] for document in documents]),
            "bootstrap_mean": _mean(distribution),
            "lower_95": _quantile(distribution, 0.025),
            "upper_95": _quantile(distribution, 0.975),
        }
        for metric, distribution in distributions.items()
    }


def compare_teacher_proxy_suite(
    *,
    suite_path: Path,
    official_root: Path,
    interim_root: Path,
    expected_interim: Path,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260801,
    min_expert_post_situ_routes: int = 4,
    num_experts: int = NUM_EXPERTS,
    thresholds: TeacherProxyThresholds = TeacherProxyThresholds(),
) -> dict[str, Any]:
    suite = load_teacher_proxy_suite(suite_path)
    document_count, tokens_per_document = suite.input_ids.shape
    if document_count < 8:
        raise ValueError("teacher-proxy gate requires at least eight documents")
    if len(suite.manifest["trace_layers"]) < 3:
        raise ValueError("teacher-proxy gate requires at least three traced layers")
    source_count = len(
        {str(document["source"]) for document in suite.manifest["documents"]}
    )
    if source_count < 3:
        raise ValueError("teacher-proxy gate requires at least three sources")

    official_root = official_root.resolve()
    interim_root = interim_root.resolve()
    expected_interim = expected_interim.resolve()
    official_run_path = official_root / "run.json"
    interim_run_path = interim_root / "run.json"
    official_run = _read_run(official_run_path)
    interim_run = _read_run(interim_run_path)
    _validate_run(
        label="official",
        root=official_root,
        run=official_run,
        suite=suite,
        expected_interim=None,
    )
    _validate_run(
        label="interim",
        root=interim_root,
        run=interim_run,
        suite=suite,
        expected_interim=expected_interim,
    )
    if Path(str(official_run["checkpoint"])).resolve() != Path(
        str(interim_run["checkpoint"])
    ).resolve():
        raise TraceFormatError("official/interim model-code checkpoints differ")

    official_trace = load_trace(official_root / "trace")
    interim_trace = load_trace(interim_root / "trace")
    _validate_trace(
        label="official", trace=official_trace, run=official_run, suite=suite
    )
    _validate_trace(
        label="interim", trace=interim_trace, run=interim_run, suite=suite
    )
    pooled = compare_teacher_proxy_traces(
        official_trace,
        interim_trace,
        num_experts=num_experts,
        min_expert_post_situ_routes=min_expert_post_situ_routes,
    )
    pooled_experts = pooled["post_situ_summary"].get("experts")
    if not isinstance(pooled_experts, list) or not pooled_experts:
        raise TraceFormatError("pooled comparison has no supported expert rankings")
    expert_spearman = [
        float(expert["record_rank_spearman"]) for expert in pooled_experts
    ]
    expert_r3 = [
        expert["rate_ladder_record_overlap"]["R3"]
        for expert in pooled_experts
    ]
    expected_expert_cases = len(suite.manifest["trace_layers"]) * num_experts
    expert_metrics = {
        "expert_case_support_fraction": (
            len(pooled_experts) / expected_expert_cases
        ),
        "expert_record_rank_spearman_median": _quantile(
            expert_spearman, 0.50
        ),
        "expert_record_rank_spearman_p10": _quantile(expert_spearman, 0.10),
        "expert_r3_donor_record_overlap_fraction": _mean(
            [float(value["donor_record_overlap_fraction"]) for value in expert_r3]
        ),
        "expert_r3_recipient_record_overlap_fraction": _mean(
            [
                float(value["recipient_record_overlap_fraction"])
                for value in expert_r3
            ]
        ),
    }

    document_reports: list[dict[str, Any]] = []
    document_metrics: list[dict[str, float]] = []
    for index, descriptor in enumerate(suite.manifest["documents"]):
        official_document = _document_trace(
            official_trace,
            document_index=index,
            document_count=document_count,
            tokens_per_document=tokens_per_document,
        )
        interim_document = _document_trace(
            interim_trace,
            document_index=index,
            document_count=document_count,
            tokens_per_document=tokens_per_document,
        )
        comparison = compare_teacher_proxy_traces(
            official_document,
            interim_document,
            num_experts=num_experts,
            float_stages=(),
            min_expert_post_situ_routes=min_expert_post_situ_routes,
        )
        metrics = _document_summary(comparison)
        document_metrics.append(metrics)
        document_reports.append(
            {
                "suite_index": index,
                "document_hash": descriptor["document_hash"],
                "source": descriptor["source"],
                "metrics": metrics,
                "post_situ_expert_summary": comparison["post_situ_summary"][
                    "expert_record_rank_spearman"
                ],
            }
        )

    uncertainty = _cluster_bootstrap(
        document_metrics,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    threshold_values = asdict(thresholds)
    decisions: dict[str, dict[str, Any]] = {}
    point_failures: list[str] = []
    uncertainty_failures: list[str] = []
    for metric, threshold in threshold_values.items():
        if metric in _DOCUMENT_BOOTSTRAP_METRICS:
            interval = uncertainty[metric]
            estimate = interval["estimate"]
            lower_95: float | None = interval["lower_95"]
            lower_pass = lower_95 >= threshold
        else:
            estimate = expert_metrics[metric]
            lower_95 = None
            lower_pass = True
        point_pass = estimate >= threshold
        decisions[metric] = {
            "threshold": threshold,
            "estimate": estimate,
            "lower_95": lower_95,
            "point_pass": point_pass,
            "lower_bound_pass": lower_pass if point_pass else False,
        }
        if not point_pass:
            point_failures.append(metric)
        elif not lower_pass:
            uncertainty_failures.append(metric)
    if point_failures:
        status = "fail"
        next_action = (
            "progressive recapture or official-teacher calibration study required"
        )
    elif uncertainty_failures:
        status = "repeat_required"
        next_action = "increase document support before accepting the interim proxy"
    else:
        status = "pass"
        next_action = "interim teacher accepted for the scoped TP12 codec study"

    return {
        "kind": TEACHER_PROXY_GATE_KIND,
        "schema_version": TEACHER_PROXY_GATE_SCHEMA_VERSION,
        "status": status,
        "next_action": next_action,
        "scope": {
            "document_count": document_count,
            "tokens_per_document": tokens_per_document,
            "total_tokens": int(suite.input_ids.numel()),
            "source_count": source_count,
            "trace_layers": suite.manifest["trace_layers"],
            "teacher_constraint": (
                "official checkpoint streamed one layer at a time; interim 3p09 "
                "remains the only resident calibration teacher"
            ),
        },
        "provenance": {
            "suite": {
                "path": str(suite.path),
                "sha256": suite.file_sha256,
                "suite_token_hash_sha256": suite.manifest[
                    "suite_token_hash_sha256"
                ],
            },
            "official": {
                "root": str(official_root),
                "run_json_sha256": file_sha256(official_run_path),
                "checkpoint": official_run["checkpoint"],
                "trace_manifest_sha256": file_sha256(
                    official_root / "trace/tp-rank-000/manifest.json"
                ),
            },
            "interim": {
                "root": str(interim_root),
                "run_json_sha256": file_sha256(interim_run_path),
                "checkpoint": interim_run["expert_checkpoint"],
                "trace_manifest_sha256": file_sha256(
                    interim_root / "trace/tp-rank-000/manifest.json"
                ),
            },
        },
        "bootstrap": {
            "unit": "whole source document",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "confidence": 0.95,
            "metrics": uncertainty,
        },
        "decision": {
            "rule": (
                "pass only when every document-bootstrap 95% lower bound and "
                "every pooled expert-level point metric meets its "
                "preregistered threshold"
            ),
            "thresholds": threshold_values,
            "metrics": decisions,
            "point_failures": point_failures,
            "uncertainty_failures": uncertainty_failures,
        },
        "expert_ranking_gate": {
            "minimum_common_routes": min_expert_post_situ_routes,
            "supported_cases": len(pooled_experts),
            "possible_layer_expert_cases": expected_expert_cases,
            "metrics": expert_metrics,
        },
        "pooled": pooled,
        "documents": document_reports,
    }


def validate_teacher_proxy_gate_report(
    report_path: str | Path,
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260801,
    min_expert_post_situ_routes: int = 4,
    num_experts: int = NUM_EXPERTS,
    thresholds: TeacherProxyThresholds = TeacherProxyThresholds(),
    require_pass: bool = True,
) -> dict[str, Any]:
    """Recompute a teacher-proxy verdict from its trace and suite provenance."""

    path = Path(report_path).resolve()
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read teacher-proxy report {path}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"teacher-proxy report {path} must be an object")
    if (
        document.get("kind") != TEACHER_PROXY_GATE_KIND
        or document.get("schema_version") != TEACHER_PROXY_GATE_SCHEMA_VERSION
    ):
        raise ValueError(f"{path} is not a supported teacher-proxy report")
    bootstrap = document.get("bootstrap")
    decision = document.get("decision")
    expert_gate = document.get("expert_ranking_gate")
    provenance = document.get("provenance")
    if not all(
        isinstance(value, dict)
        for value in (bootstrap, decision, expert_gate, provenance)
    ):
        raise ValueError(f"{path} teacher-proxy evidence is malformed")
    assert isinstance(bootstrap, dict)
    assert isinstance(decision, dict)
    assert isinstance(expert_gate, dict)
    assert isinstance(provenance, dict)
    if (
        bootstrap.get("samples") != bootstrap_samples
        or bootstrap.get("seed") != bootstrap_seed
        or decision.get("thresholds") != asdict(thresholds)
        or expert_gate.get("minimum_common_routes")
        != min_expert_post_situ_routes
    ):
        raise ValueError(f"{path} does not use the frozen proxy-gate contract")
    suite = provenance.get("suite")
    official = provenance.get("official")
    interim = provenance.get("interim")
    if not all(isinstance(value, dict) for value in (suite, official, interim)):
        raise ValueError(f"{path} teacher-proxy provenance is malformed")
    assert isinstance(suite, dict)
    assert isinstance(official, dict)
    assert isinstance(interim, dict)
    recomputed = compare_teacher_proxy_suite(
        suite_path=Path(str(suite.get("path", ""))),
        official_root=Path(str(official.get("root", ""))),
        interim_root=Path(str(interim.get("root", ""))),
        expected_interim=Path(str(interim.get("checkpoint", ""))),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        min_expert_post_situ_routes=min_expert_post_situ_routes,
        num_experts=num_experts,
        thresholds=thresholds,
    )
    if recomputed != document:
        raise ValueError(f"{path} is stale or disagrees with its trace evidence")
    if require_pass and document.get("status") != "pass":
        raise ValueError(f"{path} did not pass the teacher-proxy gate")
    return {
        "kind": TEACHER_PROXY_GATE_KIND,
        "schema_version": TEACHER_PROXY_GATE_SCHEMA_VERSION,
        "status": str(document["status"]),
        "summary_sha256": file_sha256(path),
        "suite_sha256": suite.get("sha256"),
        "official_run_json_sha256": official.get("run_json_sha256"),
        "interim_run_json_sha256": interim.get("run_json_sha256"),
        "document_count": document["scope"]["document_count"],
        "trace_layers": document["scope"]["trace_layers"],
    }
