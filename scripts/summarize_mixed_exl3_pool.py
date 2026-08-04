#!/usr/bin/env python3
"""Summarize complete layers in a running TP12 mixed-EXL candidate pool.

The all-expert writer publishes payload, metric, and selection files atomically
per layer.  This tool deliberately ignores active ``.partial`` payloads and
summarizes only layers whose complete triplet is present.  It does not use the
external validation capture and it does not read multi-gigabyte payload data.

Fit and confirmation results remain training-time selection evidence, not an
external quality estimate.  Their purpose here is to catch population-scale
selector failures as soon as the first atomic layers land.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file

from kquant import constants as C
from kquant.mixed_exl3 import LEGACY_LOGICAL_SCHEMA, LOGICAL_CANDIDATE_SCHEMAS
from kquant.pack.mixed_allocation import (
    CANDIDATE_POOL_COMPLETION_FILENAME,
    _validate_payload_header,
    validate_candidate_pool_completion,
    validate_layer_metrics,
    validate_selection_ledger_evidence,
)
from kquant.pack.mixed_candidates import (
    CANDIDATE_POOL_KIND,
    CANDIDATE_POOL_SCHEMA_VERSION,
    OFFICIAL_SOURCE_DAMAGE_METRIC,
)


SUMMARY_KIND = "kquant_tp12_mixed_exl3_candidate_pool_summary"
SUMMARY_SCHEMA_VERSION = 1


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _quantiles(values: np.ndarray) -> dict[str, float] | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    probabilities = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
    result = np.quantile(values, probabilities)
    return {
        f"p{int(probability * 100):02d}": float(value)
        for probability, value in zip(probabilities, result, strict=True)
    }


def _relative_improvement(baseline: float, candidate: float) -> float | None:
    if not math.isfinite(baseline) or not math.isfinite(candidate) or baseline <= 0:
        return None
    return 1.0 - candidate / baseline


def _metric_summary(
    baseline: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
) -> dict[str, object]:
    baseline_total = float(np.asarray(baseline, dtype=np.float64).sum())
    candidate_total = float(np.asarray(candidate, dtype=np.float64).sum())
    reference_total = float(np.asarray(reference, dtype=np.float64).sum())
    return {
        "baseline_r0_sse": baseline_total,
        "selected_sse": candidate_total,
        "relative_improvement": _relative_improvement(
            baseline_total, candidate_total
        ),
        "reference_energy": reference_total,
        "baseline_r0_nmse": (
            baseline_total / reference_total if reference_total > 0 else None
        ),
        "selected_nmse": (
            candidate_total / reference_total if reference_total > 0 else None
        ),
    }


def _selected_values(
    table: torch.Tensor,
    selected_r13: torch.Tensor,
    selected_r2: torch.Tensor,
    mode_ids: Sequence[int],
) -> np.ndarray:
    columns = {int(mode): column for column, mode in enumerate(mode_ids)}
    try:
        r13_columns = torch.tensor(
            [columns[int(mode)] for mode in selected_r13.tolist()],
            dtype=torch.long,
        )
        r2_columns = torch.tensor(
            [columns[int(mode)] for mode in selected_r2.tolist()],
            dtype=torch.long,
        )
    except KeyError as exc:
        raise ValueError(f"selected rate R{exc.args[0]} is absent") from exc
    rows = torch.arange(selected_r13.numel())
    return table[rows, r13_columns, r2_columns].sum(dim=1).double().numpy()


def _extract_layer(
    metrics: Mapping[str, torch.Tensor],
    *,
    mode_ids: tuple[int, ...],
    expert_ids: tuple[int, ...],
    fit_documents: int,
    confirmation_documents: int,
    min_fit_documents: int,
    min_confirmation_documents: int,
    minimum_improvement: float,
) -> dict[str, np.ndarray]:
    validated = validate_layer_metrics(
        metrics,
        mode_ids=mode_ids,
        fit_documents=fit_documents,
        confirmation_documents=confirmation_documents,
        expert_ids=expert_ids,
        min_fit_documents=min_fit_documents,
        min_confirmation_documents=min_confirmation_documents,
        minimum_improvement=minimum_improvement,
    )
    selected_r13 = validated["selected_r13"]
    selected_r2 = validated["selected_r2"]
    proposed_r13 = validated["proposed_r13"]
    proposed_r2 = validated["proposed_r2"]
    fit_sse = metrics["fit_sse"]
    confirmation_sse = metrics["confirmation_sse"]
    mode_zero = mode_ids.index(0)
    ci95 = metrics.get("confirmation_ci95")
    expected_ci_shape = (len(expert_ids), 2)
    if (
        ci95 is None
        or ci95.dtype != torch.float64
        or tuple(ci95.shape) != expected_ci_shape
    ):
        raise ValueError(
            "confirmation_ci95 must be a float64 tensor with shape "
            f"{expected_ci_shape}"
        )
    return {
        "selected_r13": selected_r13.numpy(),
        "selected_r2": selected_r2.numpy(),
        "proposed_r13": proposed_r13.numpy(),
        "proposed_r2": proposed_r2.numpy(),
        "evaluated_count": validated["evaluated_format_count"].numpy(),
        "fit_r0": fit_sse[:, mode_zero, mode_zero]
        .sum(dim=1)
        .double()
        .numpy(),
        "fit_selected": _selected_values(
            fit_sse, selected_r13, selected_r2, mode_ids
        ),
        "confirmation_r0": confirmation_sse[:, mode_zero, mode_zero]
        .sum(dim=1)
        .double()
        .numpy(),
        "confirmation_selected": _selected_values(
            confirmation_sse, selected_r13, selected_r2, mode_ids
        ),
        "fit_reference": metrics["fit_reference_energy"]
        .sum(dim=1)
        .double()
        .numpy(),
        "confirmation_reference": metrics["confirmation_reference_energy"]
        .sum(dim=1)
        .double()
        .numpy(),
        "fit_support": (metrics["fit_counts"] > 0).sum(dim=1).numpy(),
        "confirmation_support": (metrics["confirmation_counts"] > 0)
        .sum(dim=1)
        .numpy(),
        "fit_support_sufficient": (
            (metrics["fit_counts"] > 0).sum(dim=1) >= min_fit_documents
        ).numpy(),
        "confirmation_support_sufficient": (
            (metrics["confirmation_counts"] > 0).sum(dim=1)
            >= min_confirmation_documents
        ).numpy(),
        "confirmation_ci_low": ci95[:, 0].numpy(),
        "confirmation_ci_high": ci95[:, 1].numpy(),
        "damage": validated["damage"].numpy(),
    }


def summarize_arrays(
    layers: Sequence[dict[str, np.ndarray]], mode_ids: Sequence[int]
) -> dict[str, object]:
    """Aggregate already validated per-layer arrays into diagnostic results."""

    if not layers:
        raise ValueError("at least one completed layer is required")
    names = tuple(layers[0])
    arrays = {
        name: np.concatenate([np.asarray(layer[name]) for layer in layers])
        for name in names
    }
    selected_r13 = arrays["selected_r13"].astype(np.int64, copy=False)
    selected_r2 = arrays["selected_r2"].astype(np.int64, copy=False)
    proposed_r13 = arrays["proposed_r13"].astype(np.int64, copy=False)
    proposed_r2 = arrays["proposed_r2"].astype(np.int64, copy=False)
    nonzero_proposal = (proposed_r13 != 0) | (proposed_r2 != 0)
    accepted = (selected_r13 != 0) | (selected_r2 != 0)
    fit_support_sufficient = arrays["fit_support_sufficient"].astype(
        np.bool_, copy=False
    )
    confirmation_support_sufficient = arrays[
        "confirmation_support_sufficient"
    ].astype(np.bool_, copy=False)
    full_mode_eligible = fit_support_sufficient & confirmation_support_sufficient
    selected_differs = (selected_r13 != proposed_r13) | (
        selected_r2 != proposed_r2
    )
    if np.any(accepted & selected_differs):
        raise ValueError(
            "accepted selected formats must equal their confirmation proposals"
        )
    evaluated_count = arrays["evaluated_count"]
    format_count = len(mode_ids) ** 2
    if np.any((evaluated_count == format_count) != full_mode_eligible) or np.any(
        (evaluated_count == 1) != ~full_mode_eligible
    ):
        raise ValueError("evaluated formats disagree with the document-support gate")

    selected_histogram = {
        f"R{r13}/R{r2}": int(
            np.count_nonzero((selected_r13 == r13) & (selected_r2 == r2))
        )
        for r13 in mode_ids
        for r2 in mode_ids
    }
    proposed_histogram = {
        f"R{r13}/R{r2}": int(
            np.count_nonzero((proposed_r13 == r13) & (proposed_r2 == r2))
        )
        for r13 in mode_ids
        for r2 in mode_ids
    }
    per_format: dict[str, dict[str, object]] = {}
    for r13 in mode_ids:
        for r2 in mode_ids:
            mask = (selected_r13 == r13) & (selected_r2 == r2)
            per_format[f"R{r13}/R{r2}"] = {
                "selected_assignments": int(mask.sum()),
                "fit": _metric_summary(
                    arrays["fit_r0"][mask],
                    arrays["fit_selected"][mask],
                    arrays["fit_reference"][mask],
                ),
                "confirmation": _metric_summary(
                    arrays["confirmation_r0"][mask],
                    arrays["confirmation_selected"][mask],
                    arrays["confirmation_reference"][mask],
                ),
            }

    confirmation_denominator = arrays["confirmation_r0"]
    relative = np.full(confirmation_denominator.shape, np.nan, dtype=np.float64)
    valid = confirmation_denominator > 0
    relative[valid] = (
        1.0
        - arrays["confirmation_selected"][valid]
        / confirmation_denominator[valid]
    )
    accepted_relative = relative[accepted]
    confirmation_regressions = int(
        np.count_nonzero(
            accepted
            & (
                arrays["confirmation_selected"]
                > arrays["confirmation_r0"]
            )
        )
    )
    fit = _metric_summary(
        arrays["fit_r0"], arrays["fit_selected"], arrays["fit_reference"]
    )
    confirmation = _metric_summary(
        arrays["confirmation_r0"],
        arrays["confirmation_selected"],
        arrays["confirmation_reference"],
    )
    combined = _metric_summary(
        arrays["fit_r0"] + arrays["confirmation_r0"],
        arrays["fit_selected"] + arrays["confirmation_selected"],
        arrays["fit_reference"] + arrays["confirmation_reference"],
    )
    return {
        "assignments": int(selected_r13.size),
        "selected_format_histogram": selected_histogram,
        "proposed_format_histogram": proposed_histogram,
        "selected_r13_histogram": {
            f"R{mode}": int(np.count_nonzero(selected_r13 == mode))
            for mode in mode_ids
        },
        "selected_r2_histogram": {
            f"R{mode}": int(np.count_nonzero(selected_r2 == mode))
            for mode in mode_ids
        },
        "selection": {
            "all_formats_evaluated": int(
                np.count_nonzero(arrays["evaluated_count"] == format_count)
            ),
            "r0_r0_only_evaluated": int(
                np.count_nonzero(arrays["evaluated_count"] == 1)
            ),
            "nonzero_confirmation_proposals": int(nonzero_proposal.sum()),
            "accepted_nonzero_proposals": int(accepted.sum()),
            "rejected_nonzero_proposals": int(
                np.count_nonzero(nonzero_proposal & ~accepted)
            ),
            "acceptance_rate_given_nonzero_proposal": (
                float(accepted.sum() / nonzero_proposal.sum())
                if nonzero_proposal.any()
                else None
            ),
            "accepted_confirmation_regressions": confirmation_regressions,
            "accepted_confirmation_relative_improvement_quantiles": (
                _quantiles(accepted_relative)
            ),
            "confirmation_ci95_lower_quantiles": _quantiles(
                arrays["confirmation_ci_low"][nonzero_proposal]
            ),
        },
        "support": {
            "fit_document_quantiles": _quantiles(arrays["fit_support"]),
            "confirmation_document_quantiles": _quantiles(
                arrays["confirmation_support"]
            ),
            "mode_evaluation_gate": {
                "sufficient_fit_and_confirmation": int(
                    np.count_nonzero(full_mode_eligible)
                ),
                "insufficient_fit_only": int(
                    np.count_nonzero(
                        ~fit_support_sufficient
                        & confirmation_support_sufficient
                    )
                ),
                "insufficient_confirmation_only": int(
                    np.count_nonzero(
                        fit_support_sufficient
                        & ~confirmation_support_sufficient
                    )
                ),
                "insufficient_both": int(
                    np.count_nonzero(
                        ~fit_support_sufficient
                        & ~confirmation_support_sufficient
                    )
                ),
            },
        },
        "fit_selection_evidence": fit,
        "confirmation_selection_evidence": confirmation,
        "combined_selection_corpus": combined,
        "official_source_excess_sse_quantiles": _quantiles(arrays["damage"]),
        "per_selected_format": per_format,
    }


def summarize_candidate_pool(
    root: str | Path,
    *,
    validate_payload_headers: bool = False,
) -> dict[str, object]:
    """Summarize the complete atomic layers currently present in ``root``."""

    root = Path(root).resolve()
    manifest = _read_json(root / "mixed_exl3_candidate_manifest.json")
    if manifest.get("kind") != CANDIDATE_POOL_KIND:
        raise ValueError("candidate pool manifest has the wrong kind")
    if manifest.get("schema_version") != CANDIDATE_POOL_SCHEMA_VERSION:
        raise ValueError("candidate pool manifest has an unsupported schema")
    mode_ids = tuple(int(mode) for mode in manifest.get("mode_ids", ()))
    if not mode_ids or mode_ids[0] != 0 or len(set(mode_ids)) != len(mode_ids):
        raise ValueError("candidate pool has an invalid mode table")

    completed: list[int] = []
    in_progress: list[int] = []
    pending: list[int] = []
    layer_arrays: list[dict[str, np.ndarray]] = []
    per_layer: dict[str, dict[str, object]] = {}
    declared_trellis_schema = manifest.get("logical_trellis_schema")
    if (
        declared_trellis_schema is not None
        and declared_trellis_schema not in LOGICAL_CANDIDATE_SCHEMAS
    ):
        raise ValueError("candidate manifest logical trellis schema is unsupported")
    # Candidate-pool schema v3 originally predated this manifest field and
    # emitted proto-v2 logical descriptors.  A field-less v3 manifest is
    # therefore not unknown: it has the same unambiguous historical meaning
    # used by the packer's resume binder.  Report it even before the first
    # atomic layer ledger arrives, then still cross-check every completed
    # ledger below.
    trellis_schema = (
        str(declared_trellis_schema)
        if declared_trellis_schema is not None
        else LEGACY_LOGICAL_SCHEMA
    )
    candidate_dir = root / "candidates"
    for layer in C.MOE_LAYERS:
        stem = f"mixed-exl3-layer-{layer:05d}"
        payload = candidate_dir / f"{stem}.safetensors"
        metrics_path = candidate_dir / f"{stem}.metrics.safetensors"
        ledger_path = candidate_dir / f"{stem}.selection.json"
        partial = candidate_dir / f".{stem}.safetensors.partial"
        if ledger_path.exists():
            if not payload.is_file() or not metrics_path.is_file():
                raise ValueError(
                    f"layer {layer} selection ledger exists without its "
                    "complete triplet"
                )
            ledger = _read_json(ledger_path)
            if (
                ledger.get("kind") != CANDIDATE_POOL_KIND
                or ledger.get("schema_version") != CANDIDATE_POOL_SCHEMA_VERSION
                or ledger.get("layer") != layer
                or not ledger.get("complete")
            ):
                raise ValueError(f"layer {layer} has an invalid completion ledger")
            contract = ledger.get("selection_contract")
            if not isinstance(contract, dict):
                raise ValueError(f"layer {layer} has no selection contract")
            if tuple(contract.get("mode_ids", ())) != mode_ids:
                raise ValueError(f"layer {layer} mode table drifted")
            fit_documents = int(contract.get("fit_documents", -1))
            confirmation_documents = int(
                contract.get("confirmation_documents", -1)
            )
            min_fit_documents = int(contract.get("min_fit_documents", -1))
            min_confirmation_documents = int(
                contract.get("min_confirmation_documents", -1)
            )
            minimum_improvement = float(
                contract.get("minimum_improvement", float("nan"))
            )
            expert_ids = tuple(int(expert) for expert in ledger.get("experts", ()))
            metrics = load_file(str(metrics_path), device="cpu")
            layer_trellis_schema = validate_selection_ledger_evidence(
                ledger,
                metrics,
                mode_ids=mode_ids,
                expert_ids=expert_ids,
                logical_trellis_schema=trellis_schema,
            )
            if layer_trellis_schema != trellis_schema:
                raise ValueError("completed layers mix logical trellis schemas")
            extracted = _extract_layer(
                metrics,
                mode_ids=mode_ids,
                expert_ids=expert_ids,
                fit_documents=fit_documents,
                confirmation_documents=confirmation_documents,
                min_fit_documents=min_fit_documents,
                min_confirmation_documents=min_confirmation_documents,
                minimum_improvement=minimum_improvement,
            )
            layer_summary = summarize_arrays([extracted], mode_ids)
            if ledger.get("selected_format_histogram") != layer_summary.get(
                "selected_format_histogram"
            ):
                raise ValueError(
                    f"layer {layer} selected-format histogram does not close"
                )
            if validate_payload_headers:
                _validate_payload_header(payload, layer)
            completed.append(layer)
            layer_arrays.append(extracted)
            per_layer[str(layer)] = layer_summary
        elif payload.exists() or metrics_path.exists():
            raise ValueError(
                f"layer {layer} has published data without its completion ledger"
            )
        elif partial.exists():
            in_progress.append(layer)
        else:
            pending.append(layer)

    aggregate = summarize_arrays(layer_arrays, mode_ids) if layer_arrays else None
    completion_path = root / CANDIDATE_POOL_COMPLETION_FILENAME
    completion = (
        validate_candidate_pool_completion(root)
        if completion_path.is_file()
        else None
    )
    return {
        "kind": SUMMARY_KIND,
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "candidate_pool": str(root),
        "candidate_schema_version": manifest["schema_version"],
        "mode_ids": list(mode_ids),
        "logical_trellis_schema": trellis_schema,
        "external_validation_used": False,
        "sealed": completion is not None,
        "content_sha256": (
            completion["content_sha256"] if completion is not None else None
        ),
        "interpretation": (
            "fit and confirmation are training-time selector evidence; only the "
            "separate validation capture can establish generalization"
        ),
        "progress": {
            "planned_layers": len(C.MOE_LAYERS),
            "completed_layers": completed,
            "in_progress_layers": in_progress,
            "pending_layers": pending,
        },
        "aggregate": aggregate,
        "per_layer": per_layer,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_pool", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--validate-payload-headers",
        action="store_true",
        help="also parse every completed multi-gigabyte payload header",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_candidate_pool(
        args.candidate_pool,
        validate_payload_headers=args.validate_payload_headers,
    )
    if args.output is None:
        print(json.dumps(summary, indent=1, sort_keys=True))
    else:
        _atomic_json(args.output, summary)
        progress = summary["progress"]
        aggregate = summary["aggregate"]
        message = (
            f"wrote {args.output}: {len(progress['completed_layers'])}/"
            f"{progress['planned_layers']} layers complete"
        )
        if aggregate is not None:
            message += f", formats {aggregate['selected_format_histogram']}"
        print(message)


if __name__ == "__main__":
    main()
