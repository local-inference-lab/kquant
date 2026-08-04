#!/usr/bin/env python3
"""Weight TP12 P24/P33 kernel scenarios by naturally routed capture rows.

The candidate encoder chooses one shared ``R_r`` mode per expert.  Schema v3
then rotates its ``r`` P24 pairs across physical TP12 ranks.  Static mode
frequency alone therefore does not say how often a routed request reaches the
P24 arm, or how much P24 work lands on the slowest rank.  This tool combines
sealed candidate-layer decisions with an independent natural-routing capture
and reports those runtime-facing quantities.  It is diagnostic only and does
not participate in mode selection or allocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file

from kquant import constants as C
from kquant.capture import index_layer_samples, load_capture
from kquant.mixed_exl3 import TP_SIZE, tp12_logical_pair_index
from scripts.summarize_mixed_exl3_pool import summarize_candidate_pool


KIND = "kquant_mixed_exl3_tp12_runtime_mix"
SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _parse_layers(value: str) -> tuple[int, ...]:
    try:
        layers = tuple(int(part) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated layers") from exc
    if not layers or len(set(layers)) != len(layers):
        raise argparse.ArgumentTypeError("layers must be nonempty and unique")
    invalid = [layer for layer in layers if layer not in C.MOE_LAYERS]
    if invalid:
        raise argparse.ArgumentTypeError(f"layers must be in 1..92: {invalid}")
    return layers


def _histogram(values: torch.Tensor) -> dict[str, int]:
    values = values.to(torch.int64).reshape(-1)
    if not values.numel():
        return {}
    counts = torch.bincount(values)
    return {
        str(index): int(count)
        for index, count in enumerate(counts.tolist())
        if count
    }


def _p24_table(layer: int, modes: torch.Tensor) -> torch.Tensor:
    if modes.dtype != torch.uint8 or tuple(modes.shape) != (C.NUM_EXPERTS,):
        raise ValueError(f"layer {layer} selected modes have the wrong contract")
    table = torch.empty((TP_SIZE, C.NUM_EXPERTS), dtype=torch.bool)
    for rank in range(TP_SIZE):
        logical = torch.tensor(
            [
                tp12_logical_pair_index(layer, expert, rank)
                for expert in range(C.NUM_EXPERTS)
            ],
            dtype=torch.uint8,
        )
        table[rank] = modes > logical
    if int(table.sum()) != int(modes.to(torch.int64).sum()):
        raise AssertionError(f"layer {layer} P24 ownership does not close")
    return table


def _layer_runtime_summary(
    *,
    layer: int,
    modes_r13: torch.Tensor,
    modes_r2: torch.Tensor,
    routes: torch.Tensor,
) -> tuple[dict[str, object], torch.Tensor, torch.Tensor, torch.Tensor]:
    if routes.ndim != 2 or routes.shape[1] != C.TOP_K:
        raise ValueError(f"layer {layer} routes have the wrong shape")
    routes = routes.to(torch.int64)
    if routes.numel() and (
        int(routes.min()) < 0 or int(routes.max()) >= C.NUM_EXPERTS
    ):
        raise ValueError(f"layer {layer} routes contain an invalid expert ID")

    fc1_p24 = _p24_table(layer, modes_r13)
    fc2_p24 = _p24_table(layer, modes_r2)
    # [rank, captured token row, routed expert slot].  FC1 and FC2 have
    # independent rate-shift decisions, so both projections contribute to the
    # work seen by a rank.
    routed_fc1_p24 = fc1_p24[:, routes]
    routed_fc2_p24 = fc2_p24[:, routes]
    per_rank_token_counts = routed_fc1_p24.sum(
        dim=2, dtype=torch.int64
    ) + routed_fc2_p24.sum(dim=2, dtype=torch.int64)
    per_rank_fc1_routes = routed_fc1_p24.sum(dim=(1, 2), dtype=torch.int64)
    per_rank_fc2_routes = routed_fc2_p24.sum(dim=(1, 2), dtype=torch.int64)
    per_rank_route_counts = per_rank_fc1_routes + per_rank_fc2_routes
    per_rank_token_hits = (per_rank_token_counts > 0).sum(dim=1, dtype=torch.int64)
    static_fc1_counts = fc1_p24.sum(dim=1, dtype=torch.int64)
    static_fc2_counts = fc2_p24.sum(dim=1, dtype=torch.int64)
    static_rank_counts = static_fc1_counts + static_fc2_counts
    mixed_experts = (modes_r13 > 0) | (modes_r2 > 0)
    mixed_routes = mixed_experts[routes]
    max_rank_routes = per_rank_token_counts.max(dim=0).values

    tokens = int(routes.shape[0])
    route_slots = int(routes.numel())
    p24_rank_routes = int(per_rank_route_counts.sum())
    summary: dict[str, object] = {
        "layer": layer,
        "captured_token_rows": tokens,
        "routed_expert_slots": route_slots,
        "selected_mixed_experts": int(mixed_experts.sum()),
        "selected_fc1_mixed_experts": int((modes_r13 > 0).sum()),
        "selected_fc2_mixed_experts": int((modes_r2 > 0).sum()),
        "selected_p24_expert_rank_assignments": int(static_rank_counts.sum()),
        "selected_fc1_p24_expert_rank_assignments": int(
            static_fc1_counts.sum()
        ),
        "selected_fc2_p24_expert_rank_assignments": int(
            static_fc2_counts.sum()
        ),
        "selected_p24_expert_rank_fraction": (
            float(static_rank_counts.sum())
            / (2 * TP_SIZE * C.NUM_EXPERTS)
        ),
        "mixed_routed_expert_slots": int(mixed_routes.sum()),
        "mixed_routed_expert_fraction": (
            float(mixed_routes.sum()) / route_slots if route_slots else 0.0
        ),
        "tokens_routing_any_mixed_expert": int(mixed_routes.any(dim=1).sum()),
        "token_any_mixed_expert_fraction": (
            float(mixed_routes.any(dim=1).sum()) / tokens if tokens else 0.0
        ),
        "p24_rank_route_executions": p24_rank_routes,
        "p24_rank_route_fraction": (
            p24_rank_routes / (2 * TP_SIZE * route_slots)
            if route_slots
            else 0.0
        ),
        "per_rank": [
            {
                "rank": rank,
                "static_p24_experts": int(static_rank_counts[rank]),
                "static_fc1_p24_experts": int(static_fc1_counts[rank]),
                "static_fc2_p24_experts": int(static_fc2_counts[rank]),
                "static_p24_expert_fraction": (
                    float(static_rank_counts[rank]) / (2 * C.NUM_EXPERTS)
                ),
                "p24_route_executions": int(per_rank_route_counts[rank]),
                "fc1_p24_route_executions": int(per_rank_fc1_routes[rank]),
                "fc2_p24_route_executions": int(per_rank_fc2_routes[rank]),
                "p24_route_fraction": (
                    float(per_rank_route_counts[rank]) / (2 * route_slots)
                    if route_slots
                    else 0.0
                ),
                "token_hits": int(per_rank_token_hits[rank]),
                "token_hit_fraction": (
                    float(per_rank_token_hits[rank]) / tokens if tokens else 0.0
                ),
            }
            for rank in range(TP_SIZE)
        ],
        "slowest_rank_p24_routes_per_token_histogram": _histogram(max_rank_routes),
        "slowest_rank_mean_p24_routes_per_token": (
            float(max_rank_routes.to(torch.float64).mean()) if tokens else 0.0
        ),
        "slowest_rank_max_p24_routes_per_token": (
            int(max_rank_routes.max()) if tokens else 0
        ),
    }
    return summary, per_rank_route_counts, per_rank_token_hits, max_rank_routes


def summarize_runtime_mix(
    candidate_pool: str | Path,
    capture: str | Path,
    *,
    layers: tuple[int, ...] | None = None,
) -> dict[str, object]:
    candidate_pool = Path(candidate_pool).resolve()
    capture = Path(capture).resolve()
    pool_summary = summarize_candidate_pool(candidate_pool)
    completed = tuple(int(layer) for layer in pool_summary["progress"]["completed_layers"])
    selected = completed if layers is None else tuple(layers)
    missing = sorted(set(selected) - set(completed))
    if missing:
        raise ValueError(f"candidate layers are not atomically complete: {missing}")
    if not selected:
        raise ValueError("candidate pool has no completed layers to summarize")

    capture_manifest, geometry, _ = load_capture(capture, load_stats=False)
    if (
        geometry.num_layers != C.NUM_MOE_LAYERS
        or geometry.num_experts != C.NUM_EXPERTS
        or geometry.top_k != C.TOP_K
    ):
        raise ValueError("capture does not have the production K3 route geometry")
    sample_index = index_layer_samples(capture, (layer - 1 for layer in selected))

    total_static_fc1 = torch.zeros(TP_SIZE, dtype=torch.int64)
    total_static_fc2 = torch.zeros(TP_SIZE, dtype=torch.int64)
    total_routes = torch.zeros(TP_SIZE, dtype=torch.int64)
    total_hits = torch.zeros(TP_SIZE, dtype=torch.int64)
    all_max_rank_routes: list[torch.Tensor] = []
    per_layer: dict[str, object] = {}
    token_rows = 0
    route_slots = 0
    mixed_route_slots = 0
    mixed_token_rows = 0
    for layer in selected:
        metrics_path = (
            candidate_pool
            / "candidates"
            / f"mixed-exl3-layer-{layer:05d}.metrics.safetensors"
        )
        metrics = load_file(str(metrics_path), device="cpu")
        modes_r13 = metrics["selected_r13"]
        modes_r2 = metrics["selected_r2"]
        samples = sample_index.pop(layer - 1)
        layer_summary, rank_routes, rank_hits, max_rank_routes = (
            _layer_runtime_summary(
                layer=layer,
                modes_r13=modes_r13,
                modes_r2=modes_r2,
                routes=samples.input_experts,
            )
        )
        per_layer[str(layer)] = layer_summary
        total_static_fc1 += _p24_table(layer, modes_r13).sum(
            dim=1, dtype=torch.int64
        )
        total_static_fc2 += _p24_table(layer, modes_r2).sum(
            dim=1, dtype=torch.int64
        )
        total_routes += rank_routes
        total_hits += rank_hits
        all_max_rank_routes.append(max_rank_routes)
        token_rows += int(samples.input_experts.shape[0])
        route_slots += int(samples.input_experts.numel())
        routed = samples.input_experts.to(torch.int64)
        routed_modes = (modes_r13[routed] > 0) | (modes_r2[routed] > 0)
        mixed_route_slots += int(routed_modes.sum())
        mixed_token_rows += int(routed_modes.any(dim=1).sum())

    max_rank_routes = torch.cat(all_max_rank_routes)
    total_p24_rank_routes = int(total_routes.sum())
    static_denominator = len(selected) * C.NUM_EXPERTS
    total_static = total_static_fc1 + total_static_fc2
    result: dict[str, object] = {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "candidate_pool": str(candidate_pool),
        "candidate_pool_sealed": bool(pool_summary["sealed"]),
        "candidate_logical_trellis_schema": pool_summary[
            "logical_trellis_schema"
        ],
        "candidate_manifest_sha256": _sha256(
            candidate_pool / "mixed_exl3_candidate_manifest.json"
        ),
        "capture": str(capture),
        "capture_manifest_sha256": _sha256(capture / "manifest.json"),
        "capture_source": capture_manifest.get("source"),
        "mode_selection_or_allocation_performed": False,
        "capture_used_for_performance_weighting_only": True,
        "interpretation": (
            "diagnostic natural-route weighting for the TP12 kernel gate; it "
            "does not select codec modes or MXFP4 keeps"
        ),
        "tp_size": TP_SIZE,
        "layers": list(selected),
        "totals": {
            "layer_token_rows": token_rows,
            "routed_expert_slots": route_slots,
            "mixed_routed_expert_slots": mixed_route_slots,
            "mixed_routed_expert_fraction": (
                mixed_route_slots / route_slots if route_slots else 0.0
            ),
            "layer_token_rows_routing_any_mixed_expert": mixed_token_rows,
            "layer_token_any_mixed_expert_fraction": (
                mixed_token_rows / token_rows if token_rows else 0.0
            ),
            "p24_expert_rank_assignments": int(total_static.sum()),
            "fc1_p24_expert_rank_assignments": int(total_static_fc1.sum()),
            "fc2_p24_expert_rank_assignments": int(total_static_fc2.sum()),
            "p24_expert_rank_assignment_fraction": (
                float(total_static.sum())
                / (2 * TP_SIZE * static_denominator)
            ),
            "p24_rank_route_executions": total_p24_rank_routes,
            "p24_rank_route_fraction": (
                total_p24_rank_routes / (2 * TP_SIZE * route_slots)
                if route_slots
                else 0.0
            ),
            "slowest_rank_p24_routes_per_token_histogram": _histogram(
                max_rank_routes
            ),
            "slowest_rank_mean_p24_routes_per_token": float(
                max_rank_routes.to(torch.float64).mean()
            ),
            "slowest_rank_max_p24_routes_per_token": int(max_rank_routes.max()),
        },
        "per_rank": [
            {
                "rank": rank,
                "static_p24_experts": int(total_static[rank]),
                "static_fc1_p24_experts": int(total_static_fc1[rank]),
                "static_fc2_p24_experts": int(total_static_fc2[rank]),
                "static_p24_expert_fraction": (
                    float(total_static[rank]) / (2 * static_denominator)
                ),
                "p24_route_executions": int(total_routes[rank]),
                "p24_route_fraction": (
                    float(total_routes[rank]) / (2 * route_slots)
                    if route_slots
                    else 0.0
                ),
                "layer_token_hits": int(total_hits[rank]),
                "layer_token_hit_fraction": (
                    float(total_hits[rank]) / token_rows if token_rows else 0.0
                ),
            }
            for rank in range(TP_SIZE)
        ],
        "per_layer": per_layer,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_pool", type=Path)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--layers", type=_parse_layers)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize_runtime_mix(
        args.candidate_pool,
        args.capture,
        layers=args.layers,
    )
    if args.output is None:
        print(json.dumps(result, indent=1, sort_keys=True))
    else:
        _atomic_json(args.output, result)
        totals = result["totals"]
        assert isinstance(totals, dict)
        print(
            f"wrote {args.output}: {len(result['layers'])} layers, "
            f"P24 rank-route fraction "
            f"{float(totals['p24_rank_route_fraction']):.6%}, "
            f"any-mixed token fraction "
            f"{float(totals['layer_token_any_mixed_expert_fraction']):.6%}"
        )


if __name__ == "__main__":
    main()
