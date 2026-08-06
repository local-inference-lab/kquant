#!/usr/bin/env python3
"""Select conservative QSRT transform draws from confirmation-fold metrics."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

from kquant import constants as C
from kquant.qsrt_rotations import (
    QSRTRotationPlan,
    QSRTLayerRotationPlan,
    load_qsrt_rotation_plan,
)


@dataclass(frozen=True)
class _DrawMetrics:
    draw: int
    expert_ids: tuple[int, ...]
    confirmation_sse: torch.Tensor
    rotation_draws: dict[str, object]


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def _draw_path(value: str) -> tuple[int, Path]:
    try:
        draw_text, path_text = value.split("=", 1)
        draw = int(draw_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("expected DRAW=PATH") from exc
    if draw < 0 or not path_text:
        raise argparse.ArgumentTypeError("expected nonnegative DRAW=PATH")
    return draw, Path(path_text)


def _candidate_files(path: Path) -> tuple[Path, Path]:
    if path.is_file() and path.name.endswith(".metrics.safetensors"):
        metrics = path
        ledger = path.with_name(
            path.name.removesuffix(".metrics.safetensors") + ".selection.json"
        )
    else:
        candidate_dir = path / "candidates" if (path / "candidates").is_dir() else path
        metrics_matches = sorted(candidate_dir.glob("*.metrics.safetensors"))
        ledger_matches = sorted(candidate_dir.glob("*.selection.json"))
        if len(metrics_matches) != 1 or len(ledger_matches) != 1:
            raise ValueError(
                f"{path} must resolve to exactly one metrics/selection pair"
            )
        metrics, ledger = metrics_matches[0], ledger_matches[0]
    if not metrics.is_file() or not ledger.is_file():
        raise FileNotFoundError(f"candidate metrics/ledger pair is incomplete: {path}")
    return metrics, ledger


def _load_draw(draw: int, path: Path, *, layer: int) -> _DrawMetrics:
    metrics_path, ledger_path = _candidate_files(path)
    tensors = load_file(str(metrics_path), device="cpu")
    ledger = json.loads(ledger_path.read_text())
    if ledger.get("layer") != layer:
        raise ValueError(f"draw {draw} metrics belong to layer {ledger.get('layer')}")
    mode_ids = tensors["mode_ids"].to(torch.int64).tolist()
    if 0 not in mode_ids:
        raise ValueError(f"draw {draw} did not evaluate R0")
    r0 = mode_ids.index(0)
    confirmation = tensors["confirmation_sse"][:, r0, r0, :].to(torch.float64)
    return _DrawMetrics(
        draw=draw,
        expert_ids=tuple(int(value) for value in tensors["expert_ids"].tolist()),
        confirmation_sse=confirmation,
        rotation_draws=dict(ledger.get("rotation_draws", {})),
    )


def _load_family(
    values: list[tuple[int, Path]], *, layer: int
) -> dict[int, _DrawMetrics]:
    if len(values) != len({draw for draw, _path in values}):
        raise ValueError("rotation candidates contain duplicate draw IDs")
    result = {draw: _load_draw(draw, path, layer=layer) for draw, path in values}
    if 0 not in result:
        raise ValueError("rotation candidates must include draw zero")
    baseline_experts = result[0].expert_ids
    baseline_shape = result[0].confirmation_sse.shape
    for draw, candidate in result.items():
        if candidate.expert_ids != baseline_experts:
            raise ValueError(f"draw {draw} changes the expert panel")
        if candidate.confirmation_sse.shape != baseline_shape:
            raise ValueError(f"draw {draw} changes the confirmation fold")
    return result


def _paired_lower_bound(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    *,
    replicates: int,
    lower_quantile: float,
    seed: int,
) -> float:
    delta = (baseline - candidate).to(torch.float64)
    if delta.ndim != 1 or delta.numel() == 0:
        raise ValueError("paired confirmation vectors must be nonempty and one-dimensional")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(
        0,
        delta.numel(),
        (replicates, delta.numel()),
        generator=generator,
    )
    draws = delta[indices].sum(dim=1)
    return float(torch.quantile(draws, lower_quantile))


def _select_one(
    candidates: dict[int, torch.Tensor],
    *,
    replicates: int,
    minimum_relative_improvement: float,
    seed: int,
) -> tuple[int, dict[str, object]]:
    totals = {draw: float(values.sum()) for draw, values in candidates.items()}
    proposed = min(totals, key=lambda draw: (totals[draw], draw))
    report: dict[str, object] = {
        "confirmation_sse": {str(draw): value for draw, value in totals.items()},
        "proposed_draw": proposed,
        "selected_draw": 0,
        "reason": "draw_zero_is_argmin",
        "familywise_lower_bound": None,
    }
    if proposed == 0:
        return 0, report
    comparisons = max(1, len(candidates) - 1)
    lower = _paired_lower_bound(
        candidates[0],
        candidates[proposed],
        replicates=replicates,
        lower_quantile=0.05 / comparisons,
        seed=seed,
    )
    threshold = minimum_relative_improvement * totals[0]
    report["familywise_lower_bound"] = lower
    report["required_improvement"] = threshold
    if lower > threshold:
        report["selected_draw"] = proposed
        report["reason"] = "familywise_paired_lower_bound_passed"
        return proposed, report
    report["reason"] = "familywise_paired_lower_bound_did_not_pass"
    return 0, report


def _ledger_residual_draw(candidate: _DrawMetrics) -> int | None:
    value = candidate.rotation_draws.get("residual_layer_shared")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument(
        "--residual-candidate", action="append", type=_draw_path, required=True
    )
    parser.add_argument(
        "--intermediate-candidate", action="append", type=_draw_path, default=[]
    )
    parser.add_argument("--base-plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--minimum-relative-improvement", type=float, default=0.0)
    args = parser.parse_args()
    if args.layer not in C.MOE_LAYERS:
        parser.error("--layer must lie in 1..92")
    if args.bootstrap_replicates < 100:
        parser.error("--bootstrap-replicates must be at least 100")
    if args.minimum_relative_improvement < 0:
        parser.error("--minimum-relative-improvement must be nonnegative")

    residual = _load_family(args.residual_candidate, layer=args.layer)
    residual_vectors = {
        draw: candidate.confirmation_sse.sum(dim=0)
        for draw, candidate in residual.items()
    }
    residual_draw, residual_report = _select_one(
        residual_vectors,
        replicates=args.bootstrap_replicates,
        minimum_relative_improvement=args.minimum_relative_improvement,
        seed=args.layer * 1_000_003 + 17,
    )

    overrides: list[tuple[int, int]] = []
    intermediate_report: dict[str, object] = {}
    if args.intermediate_candidate:
        intermediate = _load_family(args.intermediate_candidate, layer=args.layer)
        for draw, candidate in intermediate.items():
            encoded_residual = _ledger_residual_draw(candidate)
            if encoded_residual != residual_draw:
                raise ValueError(
                    f"intermediate draw {draw} was encoded with residual draw "
                    f"{encoded_residual}, expected selected draw {residual_draw}"
                )
        expert_ids = intermediate[0].expert_ids
        for row, expert in enumerate(expert_ids):
            candidate_vectors = {
                draw: candidate.confirmation_sse[row]
                for draw, candidate in intermediate.items()
            }
            selected, evidence = _select_one(
                candidate_vectors,
                replicates=args.bootstrap_replicates,
                minimum_relative_improvement=args.minimum_relative_improvement,
                seed=args.layer * 1_000_003 + expert * 1_009 + 29,
            )
            if selected:
                overrides.append((expert, selected))
            intermediate_report[str(expert)] = evidence

    layer_plan = QSRTLayerRotationPlan(
        residual_draw=residual_draw,
        intermediate_overrides=tuple(overrides),
    )
    base = (
        QSRTRotationPlan()
        if args.base_plan is None
        else load_qsrt_rotation_plan(args.base_plan)
    )
    layers = dict(base.layers)
    layers[args.layer] = layer_plan
    plan = QSRTRotationPlan(tuple(sorted(layers.items())))
    _atomic_json(args.output, plan.to_json())
    _atomic_json(
        args.report,
        {
            "kind": "kquant_qsrt_rotation_selection",
            "schema_version": 1,
            "layer": args.layer,
            "selection_fold": "confirmation_documents",
            "familywise_error_rate": 0.05,
            "bootstrap_replicates": args.bootstrap_replicates,
            "minimum_relative_improvement": args.minimum_relative_improvement,
            "residual": residual_report,
            "intermediate": intermediate_report,
            "selected_plan": layer_plan.to_json(),
        },
    )
    print(json.dumps(layer_plan.to_json(), sort_keys=True))


if __name__ == "__main__":
    main()
