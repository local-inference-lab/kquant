"""Matched-R0 external validation for selected mixed-rate experts.

The ordinary validation sidecar measures the damage of the payload that the
training/confirmation selector already chose.  That is the correct input to
the MXFP4 keep allocator, but it does not answer a different question: whether
an accepted nonzero ``(r13, r2)`` format still beats a matched uniform
``(R0, R0)`` encode on the fully untouched corpus.

This module defines and authenticates that second sidecar.  The GPU producer
re-encodes only assignments whose selected format has either rate-transfer
axis nonzero.  Uniform assignments copy their already authenticated
selected-candidate score, because their selected payload is itself the
``(R0, R0)`` control.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from safetensors.torch import load_file

from kquant import constants as C
from kquant.pack.mixed_candidates import CANDIDATE_POOL_SCHEMA_VERSION
from kquant.pack.mixed_validation import (
    MixedValidationScores,
    VALIDATION_SCORE_SCHEMA_VERSION,
)


MODE_VALIDATION_KIND = "kquant_tp12_mixed_exl3_mode_validation"
MODE_VALIDATION_SCHEMA_VERSION = 2
MODE_VALIDATION_METRIC = "external_selected_vs_matched_r0_excess_sse"
MANIFEST_FILENAME = "mixed_exl3_mode_validation_manifest.json"
COMPLETION_FILENAME = "mixed_exl3_mode_validation_complete.json"
MODE_VALIDATION_SUMMARY_KIND = (
    "kquant_tp12_mixed_exl3_mode_validation_summary"
)
MODE_VALIDATION_SUMMARY_SCHEMA_VERSION = 2
MODE_VALIDATION_DECISION_SCOPE = (
    "global codec-policy accept/repeat/reject only; external validation "
    "does not retune individual expert modes"
)


@dataclass(frozen=True)
class MixedModeValidationScores:
    """Complete matched-R0 evidence for every selected codec assignment."""

    root: Path
    manifest: dict
    selected_r13: np.ndarray
    selected_r2: np.ndarray
    audited: np.ndarray
    selected_damage: np.ndarray
    r0_damage: np.ndarray
    per_document_selected: np.ndarray
    per_document_r0: np.ndarray
    routed_rows: np.ndarray
    supported_documents: np.ndarray
    content_sha256: str


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


def _update_content_digest(digest: Any, root: Path, path: Path) -> None:
    relative = path.relative_to(root).as_posix().encode("utf-8")
    digest.update(len(relative).to_bytes(4, "little"))
    digest.update(relative)
    digest.update(bytes.fromhex(_sha256(path)))


def _require_tensor(
    tensors: Mapping[str, torch.Tensor],
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    try:
        value = tensors[name]
    except KeyError as exc:
        raise ValueError(f"mode-validation metrics are missing {name}") from exc
    if value.dtype != dtype or tuple(value.shape) != shape:
        raise ValueError(
            f"mode-validation metric {name} has {value.dtype} "
            f"{tuple(value.shape)}, expected {dtype} {shape}"
        )
    return value


def validate_mode_validation_layer_metrics(
    tensors: Mapping[str, torch.Tensor],
    *,
    selected_r13: np.ndarray,
    selected_r2: np.ndarray,
    selected_validation_sse: np.ndarray,
    validation_reference_energy: np.ndarray,
    validation_counts: np.ndarray,
    documents: int,
    expert_ids: tuple[int, ...] = tuple(range(C.NUM_EXPERTS)),
) -> dict[str, np.ndarray]:
    """Close one layer against its authenticated selected-score sidecar."""

    experts = len(expert_ids)
    vector = (experts,)
    matrix = (experts, documents)
    ids = _require_tensor(tensors, "expert_ids", shape=vector, dtype=torch.int16)
    if ids.tolist() != list(expert_ids):
        raise ValueError("mode-validation expert IDs are not canonical")
    stored_r13 = _require_tensor(
        tensors, "selected_r13", shape=vector, dtype=torch.uint8
    )
    stored_r2 = _require_tensor(
        tensors, "selected_r2", shape=vector, dtype=torch.uint8
    )
    expected_r13 = np.asarray(selected_r13, dtype=np.uint8)
    expected_r2 = np.asarray(selected_r2, dtype=np.uint8)
    if expected_r13.shape != vector or not np.array_equal(
        stored_r13.numpy(), expected_r13
    ):
        raise ValueError("mode-validation selected r13 modes drifted")
    if expected_r2.shape != vector or not np.array_equal(
        stored_r2.numpy(), expected_r2
    ):
        raise ValueError("mode-validation selected r2 modes drifted")
    audited = _require_tensor(tensors, "audited", shape=vector, dtype=torch.bool)
    expected_audited = (expected_r13 != 0) | (expected_r2 != 0)
    if not np.array_equal(audited.numpy(), expected_audited):
        raise ValueError(
            "mode-validation audit mask must equal selected nonzero format"
        )

    selected = _require_tensor(
        tensors, "selected_validation_sse", shape=matrix, dtype=torch.float64
    )
    r0 = _require_tensor(
        tensors, "r0_validation_sse", shape=matrix, dtype=torch.float64
    )
    reference = _require_tensor(
        tensors,
        "validation_reference_energy",
        shape=matrix,
        dtype=torch.float64,
    )
    counts = _require_tensor(
        tensors, "validation_counts", shape=matrix, dtype=torch.int32
    )
    selected_damage = _require_tensor(
        tensors, "selected_validation_damage", shape=vector, dtype=torch.float64
    )
    r0_damage = _require_tensor(
        tensors, "r0_validation_damage", shape=vector, dtype=torch.float64
    )

    expected_selected = np.asarray(selected_validation_sse, dtype=np.float64)
    expected_reference = np.asarray(validation_reference_energy, dtype=np.float64)
    expected_counts = np.asarray(validation_counts, dtype=np.int32)
    if expected_selected.shape != matrix or not np.array_equal(
        selected.numpy(), expected_selected
    ):
        raise ValueError("mode-validation selected SSE drifted")
    if expected_reference.shape != matrix or not np.array_equal(
        reference.numpy(), expected_reference
    ):
        raise ValueError("mode-validation reference energy drifted")
    if expected_counts.shape != matrix or not np.array_equal(
        counts.numpy(), expected_counts
    ):
        raise ValueError("mode-validation row counts drifted")

    for name, value in (
        ("selected_validation_sse", selected),
        ("r0_validation_sse", r0),
        ("validation_reference_energy", reference),
        ("selected_validation_damage", selected_damage),
        ("r0_validation_damage", r0_damage),
    ):
        if not bool(torch.all(torch.isfinite(value))) or bool(torch.any(value < 0)):
            raise ValueError(f"mode-validation {name} must be finite and non-negative")
    if bool(torch.any(counts < 0)):
        raise ValueError("mode-validation counts must be non-negative")
    if not torch.equal(selected.sum(dim=1), selected_damage):
        raise ValueError("selected mode-validation damage does not close")
    if not torch.equal(r0.sum(dim=1), r0_damage):
        raise ValueError("R0 mode-validation damage does not close")
    uniform = ~audited
    if not torch.equal(r0[uniform], selected[uniform]):
        raise ValueError("uniform assignments must reuse their selected R0 score")
    rows = counts.sum(dim=1)
    supported = (counts > 0).sum(dim=1)
    if bool(
        torch.any(
            (rows == 0)
            & (
                (selected_damage != 0)
                | (r0_damage != 0)
                | (reference.sum(dim=1) != 0)
            )
        )
    ):
        raise ValueError("unrouted mode-validation experts have nonzero metrics")
    return {
        "selected_r13": stored_r13.numpy(),
        "selected_r2": stored_r2.numpy(),
        "audited": audited.numpy(),
        "selected_damage": selected_damage.numpy(),
        "r0_damage": r0_damage.numpy(),
        "per_document_selected": selected.numpy(),
        "per_document_r0": r0.numpy(),
        "routed_rows": rows.numpy(),
        "supported_documents": supported.numpy(),
    }


def _selected_layer_metrics_path(validation: MixedValidationScores, layer: int) -> Path:
    stem = f"mixed-exl3-layer-{layer:05d}.validation.safetensors"
    return validation.root / "scores" / stem


def load_mixed_mode_validation_scores(
    root: str | Path,
    candidate_pool: object,
    selected_validation: MixedValidationScores,
) -> MixedModeValidationScores:
    """Load and authenticate a complete 92-layer matched-R0 sidecar."""

    try:
        candidate_root = Path(candidate_pool.root).resolve()
        candidate_content_sha256 = str(candidate_pool.content_sha256)
        candidate_r13 = np.asarray(candidate_pool.selected_r13)
        candidate_r2 = np.asarray(candidate_pool.selected_r2)
        candidate_schema = int(candidate_pool.manifest["schema_version"])
        candidate_trellis_schema = str(candidate_pool.trellis_schema)
        candidate_codebook = str(candidate_pool.codebook)
        candidate_mode_ids = list(candidate_pool.mode_ids)
    except (AttributeError, KeyError, TypeError) as exc:
        raise TypeError("candidate_pool is not a complete MixedCandidatePool") from exc
    if candidate_schema != CANDIDATE_POOL_SCHEMA_VERSION:
        raise ValueError("mode-validation candidate schema drifted")
    root = Path(root).resolve()
    manifest_path = root / MANIFEST_FILENAME
    manifest = _read_json(manifest_path)
    expected = {
        "kind": MODE_VALIDATION_KIND,
        "schema_version": MODE_VALIDATION_SCHEMA_VERSION,
        "candidate_pool": str(candidate_root),
        "candidate_pool_content_sha256": candidate_content_sha256,
        "candidate_logical_trellis_schema": candidate_trellis_schema,
        "candidate_codebook": candidate_codebook,
        "candidate_mode_ids": candidate_mode_ids,
        "selected_validation_scores": str(selected_validation.root.resolve()),
        "selected_validation_score_set_sha256": selected_validation.content_sha256,
        "selected_validation_schema_version": VALIDATION_SCORE_SCHEMA_VERSION,
        "source_model": C.MODEL_ID,
        "source_revision": C.REVISION,
        "tp_size": 12,
        "metric": MODE_VALIDATION_METRIC,
    }
    for name, value in expected.items():
        actual = manifest.get(name)
        if name == "candidate_codebook" and actual is None:
            # Historical score sets predate an explicit codebook field and
            # were unconditionally MCG.  Never grant that fallback to a new
            # reconstruction family.
            actual = "mcg"
        if name == "candidate_mode_ids" and actual is None:
            actual = manifest.get("mode_ids")
        if actual != value:
            raise ValueError(
                f"mode-validation manifest {name} mismatch: "
                f"{actual!r} != {value!r}"
            )
    documents = int(manifest.get("validation_documents", -1))
    if documents < 1:
        raise ValueError("mode-validation sidecar has no documents")
    completion_path = root / COMPLETION_FILENAME
    completion = _read_json(completion_path)
    if (
        completion.get("kind") != MODE_VALIDATION_KIND
        or completion.get("schema_version") != MODE_VALIDATION_SCHEMA_VERSION
        or not completion.get("complete")
        or completion.get("layers") != list(C.MOE_LAYERS)
        or completion.get("manifest_sha256") != _sha256(manifest_path)
    ):
        raise ValueError("mode-validation completion marker does not close")

    content_digest = hashlib.sha256()
    content_digest.update(b"kquant-mixed-mode-validation-v2\0")
    _update_content_digest(content_digest, root, manifest_path)
    _update_content_digest(content_digest, root, completion_path)

    shape = (C.NUM_MOE_LAYERS, C.NUM_EXPERTS)
    selected_r13 = np.empty(shape, dtype=np.uint8)
    selected_r2 = np.empty(shape, dtype=np.uint8)
    audited = np.empty(shape, dtype=np.bool_)
    selected_damage = np.empty(shape, dtype=np.float64)
    r0_damage = np.empty(shape, dtype=np.float64)
    per_document_selected = np.empty((*shape, documents), dtype=np.float64)
    per_document_r0 = np.empty_like(per_document_selected)
    routed_rows = np.empty(shape, dtype=np.int64)
    supported_documents = np.empty(shape, dtype=np.int32)

    for row, layer in enumerate(C.MOE_LAYERS):
        stem = f"mixed-exl3-layer-{layer:05d}.mode-validation"
        metrics_path = root / "scores" / f"{stem}.safetensors"
        ledger_path = root / "scores" / f"{stem}.json"
        if not metrics_path.is_file() or not ledger_path.is_file():
            raise ValueError(f"mode-validation sidecar is missing layer {layer}")
        _update_content_digest(content_digest, root, metrics_path)
        _update_content_digest(content_digest, root, ledger_path)
        ledger = _read_json(ledger_path)
        expected_ledger = {
            "kind": MODE_VALIDATION_KIND,
            "schema_version": MODE_VALIDATION_SCHEMA_VERSION,
            "complete": True,
            "layer": layer,
            "metrics": metrics_path.name,
            "experts": C.NUM_EXPERTS,
            "validation_documents": documents,
        }
        for name, value in expected_ledger.items():
            if ledger.get(name) != value:
                raise ValueError(
                    f"layer {layer} mode-validation ledger {name} drifted"
                )
        selected_metrics = load_file(
            str(_selected_layer_metrics_path(selected_validation, layer)),
            device="cpu",
        )
        tensors = load_file(str(metrics_path), device="cpu")
        validated = validate_mode_validation_layer_metrics(
            tensors,
            selected_r13=candidate_r13[row],
            selected_r2=candidate_r2[row],
            selected_validation_sse=selected_metrics["validation_sse"].numpy(),
            validation_reference_energy=selected_metrics[
                "validation_reference_energy"
            ].numpy(),
            validation_counts=selected_metrics["validation_counts"].numpy(),
            documents=documents,
        )
        audited_experts = np.flatnonzero(validated["audited"]).tolist()
        evidence = ledger.get("reencoded_r0_evidence")
        if not isinstance(evidence, dict) or set(evidence) != {
            str(expert) for expert in audited_experts
        }:
            raise ValueError(
                f"layer {layer} R0 evidence does not cover exactly selected "
                "nonzero formats"
            )
        for expert in audited_experts:
            item = evidence[str(expert)]
            payload_sha256 = (
                item.get("r0_payload_sha256") if isinstance(item, dict) else None
            )
            try:
                payload_digest = bytes.fromhex(payload_sha256 or "")
            except ValueError:
                payload_digest = b""
            if (
                not isinstance(item, dict)
                or item.get("selected_r13") != int(candidate_r13[row, expert])
                or item.get("selected_r2") != int(candidate_r2[row, expert])
                or item.get("training_r0_metrics_exact") is not True
                or item.get("shared_hidden_scales_exact") is not True
                or not isinstance(payload_sha256, str)
                or len(payload_digest) != 32
            ):
                raise ValueError(
                    f"layer {layer} expert {expert} has invalid R0 closure evidence"
                )
        if ledger.get("audited_experts") != audited_experts:
            raise ValueError(f"layer {layer} audited expert inventory drifted")
        if int(validated["routed_rows"].sum()) != int(
            ledger.get("validation_rows", -1)
        ):
            raise ValueError(f"layer {layer} mode-validation rows drifted")
        if not np.isclose(
            float(validated["selected_damage"].sum()),
            float(ledger.get("selected_validation_damage", float("nan"))),
        ):
            raise ValueError(f"layer {layer} selected damage total drifted")
        if not np.isclose(
            float(validated["r0_damage"].sum()),
            float(ledger.get("r0_validation_damage", float("nan"))),
        ):
            raise ValueError(f"layer {layer} R0 damage total drifted")

        selected_r13[row] = validated["selected_r13"]
        selected_r2[row] = validated["selected_r2"]
        audited[row] = validated["audited"]
        selected_damage[row] = validated["selected_damage"]
        r0_damage[row] = validated["r0_damage"]
        per_document_selected[row] = validated["per_document_selected"]
        per_document_r0[row] = validated["per_document_r0"]
        routed_rows[row] = validated["routed_rows"]
        supported_documents[row] = validated["supported_documents"]

    return MixedModeValidationScores(
        root=root,
        manifest=manifest,
        selected_r13=selected_r13,
        selected_r2=selected_r2,
        audited=audited,
        selected_damage=selected_damage,
        r0_damage=r0_damage,
        per_document_selected=per_document_selected,
        per_document_r0=per_document_r0,
        routed_rows=routed_rows,
        supported_documents=supported_documents,
        content_sha256=content_digest.hexdigest(),
    )


def _quantiles(values: np.ndarray) -> dict[str, float] | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    probabilities = (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)
    quantiles = np.quantile(values, probabilities)
    return {
        f"p{int(probability * 100):02d}": float(value)
        for probability, value in zip(probabilities, quantiles, strict=True)
    }


def paired_document_bootstrap(
    baseline_by_document: np.ndarray,
    candidate_by_document: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    """Bootstrap a paired relative SSE improvement over whole documents."""

    baseline = np.asarray(baseline_by_document, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate_by_document, dtype=np.float64).reshape(-1)
    if baseline.shape != candidate.shape or baseline.size < 2:
        raise ValueError(
            "paired bootstrap needs equal vectors with at least two documents"
        )
    if (
        not np.isfinite(baseline).all()
        or not np.isfinite(candidate).all()
        or np.any(baseline < 0)
        or np.any(candidate < 0)
    ):
        raise ValueError("paired bootstrap SSE must be finite and non-negative")
    if isinstance(replicates, bool) or replicates < 100:
        raise ValueError("paired bootstrap requires at least 100 replicates")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("paired bootstrap seed must be an integer")
    baseline_total = float(baseline.sum())
    candidate_total = float(candidate.sum())
    observed = (
        1.0 - candidate_total / baseline_total if baseline_total > 0 else None
    )
    generator = np.random.default_rng(seed)
    values: list[np.ndarray] = []
    remaining = replicates
    while remaining:
        count = min(remaining, 512)
        indices = generator.integers(
            0,
            baseline.size,
            size=(count, baseline.size),
        )
        sampled_baseline = baseline[indices].sum(axis=1)
        sampled_candidate = candidate[indices].sum(axis=1)
        valid = sampled_baseline > 0
        if np.any(valid):
            values.append(
                1.0 - sampled_candidate[valid] / sampled_baseline[valid]
            )
        remaining -= count
    samples = np.concatenate(values) if values else np.empty(0, dtype=np.float64)
    interval = (
        [float(value) for value in np.quantile(samples, (0.025, 0.975))]
        if samples.size
        else [None, None]
    )
    return {
        "documents": int(baseline.size),
        "replicates_requested": int(replicates),
        "replicates_valid": int(samples.size),
        "baseline_damage": baseline_total,
        "candidate_damage": candidate_total,
        "relative_improvement": observed,
        "ci95": interval,
    }


def summarize_mode_validation_arrays(
    selected_r13: np.ndarray,
    selected_r2: np.ndarray,
    audited: np.ndarray,
    selected_by_document: np.ndarray,
    r0_by_document: np.ndarray,
    supported_documents: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    """Summarize authenticated matched-R0 arrays and decide the policy gate."""

    selected_r13 = np.asarray(selected_r13, dtype=np.uint8)
    selected_r2 = np.asarray(selected_r2, dtype=np.uint8)
    audited = np.asarray(audited, dtype=np.bool_)
    selected = np.asarray(selected_by_document, dtype=np.float64)
    r0 = np.asarray(r0_by_document, dtype=np.float64)
    support = np.asarray(supported_documents, dtype=np.int32)
    if (
        selected_r13.shape != audited.shape
        or selected_r2.shape != audited.shape
        or support.shape != audited.shape
    ):
        raise ValueError("mode-validation vectors have inconsistent shapes")
    if selected.shape != r0.shape or selected.shape[:-1] != audited.shape:
        raise ValueError("mode-validation document tensors have inconsistent shapes")
    if not np.array_equal(audited, (selected_r13 != 0) | (selected_r2 != 0)):
        raise ValueError("audit mask must equal selected nonzero format")
    if (
        not np.isfinite(selected).all()
        or not np.isfinite(r0).all()
        or np.any(selected < 0)
        or np.any(r0 < 0)
        or np.any(support < 0)
    ):
        raise ValueError("mode-validation arrays contain invalid values")

    audited_count = int(audited.sum())
    if audited_count == 0:
        return {
            "status": "pass",
            "pass": True,
            "reason": "the selector accepted no mixed-rate assignments",
            "audited_assignments": 0,
            "aggregate": None,
            "per_axis_mode": {"r13": {}, "r2": {}},
            "per_format": {},
            "per_expert": None,
        }

    aggregate = paired_document_bootstrap(
        r0[audited].sum(axis=0),
        selected[audited].sum(axis=0),
        replicates=replicates,
        seed=seed,
    )
    per_axis_mode: dict[str, dict[str, dict[str, object]]] = {
        "r13": {},
        "r2": {},
    }
    per_format: dict[str, dict[str, object]] = {}
    gate_items = [("aggregate", aggregate)]
    for axis_index, (axis, values) in enumerate(
        (("r13", selected_r13), ("r2", selected_r2)),
        start=1,
    ):
        for mode in sorted(int(value) for value in np.unique(values[audited])):
            if mode == 0:
                continue
            mask = audited & (values == mode)
            item = paired_document_bootstrap(
                r0[mask].sum(axis=0),
                selected[mask].sum(axis=0),
                replicates=replicates,
                seed=seed + axis_index * 10_000_019 + mode * 1_000_003,
            )
            item["assignments"] = int(mask.sum())
            item["assignments_with_validation_support"] = int(
                np.count_nonzero(mask & (support > 0))
            )
            per_axis_mode[axis][f"R{mode}"] = item
            gate_items.append((f"{axis}=R{mode}", item))

    selected_pairs = sorted(
        {
            (int(r13), int(r2))
            for r13, r2 in zip(
                selected_r13[audited], selected_r2[audited], strict=True
            )
        }
    )
    for r13, r2 in selected_pairs:
        mask = audited & (selected_r13 == r13) & (selected_r2 == r2)
        item = paired_document_bootstrap(
            r0[mask].sum(axis=0),
            selected[mask].sum(axis=0),
            replicates=replicates,
            seed=seed + 100_000_007 + r13 * 10_007 + r2 * 1_009,
        )
        item["assignments"] = int(mask.sum())
        item["assignments_with_validation_support"] = int(
            np.count_nonzero(mask & (support > 0))
        )
        item["policy_gate"] = False
        per_format[f"R{r13}/R{r2}"] = item

    failures: list[str] = []
    repeats: list[str] = []
    for name, item in gate_items:
        point = item["relative_improvement"]
        low = item["ci95"][0]
        if point is None or low is None:
            repeats.append(f"{name} has no positive-baseline bootstrap support")
        elif point <= 0:
            failures.append(
                f"{name} selected-mode point estimate does not beat matched R0"
            )
        elif low <= 0:
            repeats.append(
                f"{name} improvement is positive but its 95% lower bound is not"
            )

    expert_r0 = r0.sum(axis=-1)[audited]
    expert_selected = selected.sum(axis=-1)[audited]
    supported = expert_r0 > 0
    relative = np.full(expert_r0.shape, np.nan, dtype=np.float64)
    relative[supported] = 1.0 - expert_selected[supported] / expert_r0[supported]
    per_expert = {
        "assignments": audited_count,
        "assignments_with_positive_r0_damage": int(supported.sum()),
        "assignments_with_validation_documents": int(
            np.count_nonzero(support[audited] > 0)
        ),
        "improved_assignments": int(np.count_nonzero(relative > 0)),
        "regressed_assignments": int(np.count_nonzero(relative < 0)),
        "relative_improvement_quantiles": _quantiles(relative),
        "supported_document_quantiles": _quantiles(support[audited]),
    }
    if failures:
        status = "fail"
        reasons = failures + repeats
    elif repeats:
        status = "repeat_required"
        reasons = repeats
    else:
        status = "pass"
        reasons = []
    return {
        "status": status,
        "pass": status == "pass",
        "reason": "; ".join(reasons) if reasons else "all matched-R0 gates passed",
        "audited_assignments": audited_count,
        "aggregate": aggregate,
        "gate_basis": "aggregate and each nonzero rate-transfer axis/mode",
        "per_axis_mode": per_axis_mode,
        "per_format": per_format,
        "per_expert": per_expert,
    }


def mode_validation_summary_document(
    candidate_pool: object,
    selected_validation: MixedValidationScores,
    mode_validation: MixedModeValidationScores,
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    """Build the canonical, reproducible matched-R0 gate document."""

    decision = summarize_mode_validation_arrays(
        mode_validation.selected_r13,
        mode_validation.selected_r2,
        mode_validation.audited,
        mode_validation.per_document_selected,
        mode_validation.per_document_r0,
        mode_validation.supported_documents,
        replicates=replicates,
        seed=seed,
    )
    return {
        "kind": MODE_VALIDATION_SUMMARY_KIND,
        "schema_version": MODE_VALIDATION_SUMMARY_SCHEMA_VERSION,
        "status": decision["status"],
        "pass": decision["pass"],
        "candidate_pool": str(Path(candidate_pool.root).resolve()),
        "candidate_pool_content_sha256": candidate_pool.content_sha256,
        "selected_validation_scores": str(selected_validation.root.resolve()),
        "selected_validation_score_set_sha256": selected_validation.content_sha256,
        "mode_validation_scores": str(mode_validation.root.resolve()),
        "mode_validation_score_set_sha256": mode_validation.content_sha256,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "decision_scope": MODE_VALIDATION_DECISION_SCOPE,
        "decision": decision,
    }


def load_mixed_mode_validation_summary(
    path: str | Path,
    candidate_pool: object,
    selected_validation: MixedValidationScores,
    *,
    require_pass: bool = True,
) -> tuple[dict, MixedModeValidationScores]:
    """Recompute and authenticate a matched-R0 summary before consumption."""

    path = Path(path).resolve()
    document = _read_json(path)
    replicates = document.get("bootstrap_replicates")
    seed = document.get("bootstrap_seed")
    if isinstance(replicates, bool) or not isinstance(replicates, int):
        raise ValueError("mode-validation summary bootstrap count is invalid")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("mode-validation summary bootstrap seed is invalid")
    score_root = document.get("mode_validation_scores")
    if not isinstance(score_root, str) or not score_root:
        raise ValueError("mode-validation summary has no score-set path")
    scores = load_mixed_mode_validation_scores(
        score_root,
        candidate_pool,
        selected_validation,
    )
    expected = mode_validation_summary_document(
        candidate_pool,
        selected_validation,
        scores,
        replicates=replicates,
        seed=seed,
    )
    if document != expected:
        differing = sorted(
            name
            for name in set(document).union(expected)
            if document.get(name) != expected.get(name)
        )
        raise ValueError(
            f"mode-validation summary does not reproduce: {differing[:8]}"
        )
    if require_pass and document.get("pass") is not True:
        raise ValueError(
            f"mode-validation gate did not pass: {document.get('status')!r}"
        )
    return document, scores
