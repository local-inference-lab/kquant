"""Exact-byte Lagrangian allocation between SQG trellis and X4 experts."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kquant import constants as C
from kquant.mixed_exl3 import (
    FORMAT_MXFP4,
    KEEP_STORAGE_EXTERNAL_X4,
    TP12LayerLayout,
    ExpertFormatSpec,
)
from kquant.pack.mixed_allocation import MixedCandidatePool
from kquant.pack.x4_index import X4CostIndex
from kquant.x4 import X4_DATA_OFFSET


SQRT_C_ALLOCATION_KIND = "kquant_kimi_k3_sqrt_c_allocation"
SQRT_C_ALLOCATION_SCHEMA_VERSION = 1
SQRT_C_ALLOCATION_FILENAME = "allocation-sqrt-c-tp12.json"


@dataclass(frozen=True)
class SqrtCAllocation:
    x4_mask: np.ndarray
    target_container_bytes: int | None
    container_bytes: int
    lagrange_lambda: float
    x4_experts: int
    compressed_experts: int
    retained_damage: float
    remaining_damage: float
    budget_slack_bytes: int | None


def _validate_inputs(
    damage: np.ndarray,
    x4_expert_bytes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    shape = (C.NUM_MOE_LAYERS, C.NUM_EXPERTS)
    damage = np.asarray(damage, dtype=np.float64)
    x4_expert_bytes = np.asarray(x4_expert_bytes, dtype=np.int64)
    if damage.shape != shape or x4_expert_bytes.shape != shape:
        raise ValueError(f"SQRT-C inputs must have shape {shape}")
    if not np.isfinite(damage).all() or np.any(damage < 0):
        raise ValueError("SQRT-C damage must be finite and non-negative")
    if np.any(x4_expert_bytes <= 0):
        raise ValueError("SQRT-C X4 costs must be positive")
    return damage, x4_expert_bytes


def sqrt_c_trellis_layer_bytes(x4_count: int) -> int:
    if isinstance(x4_count, bool) or not 0 <= x4_count <= C.NUM_EXPERTS:
        raise ValueError(f"X4 count must lie in 0..{C.NUM_EXPERTS}")
    return TP12LayerLayout(
        compressed_experts=C.NUM_EXPERTS - x4_count,
        kept_experts=x4_count,
        keep_storage=KEEP_STORAGE_EXTERNAL_X4,
    ).disk_bytes


def sqrt_c_total_container_bytes(
    x4_mask: np.ndarray,
    x4_expert_bytes: np.ndarray,
) -> int:
    _, costs = _validate_inputs(
        np.zeros((C.NUM_MOE_LAYERS, C.NUM_EXPERTS)),
        x4_expert_bytes,
    )
    if x4_mask.shape != costs.shape or x4_mask.dtype != np.bool_:
        raise ValueError("SQRT-C X4 mask has the wrong shape or dtype")
    total = 0
    for row in range(C.NUM_MOE_LAYERS):
        count = int(x4_mask[row].sum())
        total += sqrt_c_trellis_layer_bytes(count)
        total += X4_DATA_OFFSET + int(costs[row, x4_mask[row]].sum())
    return total


def _choose_layer_lagrangian(
    damage: np.ndarray,
    x4_bytes: np.ndarray,
    lagrange_lambda: float,
) -> np.ndarray:
    """Return the exact one-layer minimizer for a fixed Lagrange multiplier."""

    score = damage - lagrange_lambda * x4_bytes
    order = np.argsort(-score, kind="stable")
    ordered_damage = damage[order]
    ordered_bytes = x4_bytes[order]
    avoided = np.concatenate(([0.0], np.cumsum(ordered_damage, dtype=np.float64)))
    x4_prefix = np.concatenate(([0], np.cumsum(ordered_bytes, dtype=np.int64)))
    total_damage = float(damage.sum())
    best: tuple[float, int, float, int] | None = None
    best_count = 0
    for count in range(C.NUM_EXPERTS + 1):
        stored_bytes = (
            sqrt_c_trellis_layer_bytes(count)
            + X4_DATA_OFFSET
            + int(x4_prefix[count])
        )
        remaining = total_damage - float(avoided[count])
        objective = remaining + lagrange_lambda * stored_bytes
        # Deterministic ties prefer fewer bytes, then less damage, then fewer
        # X4 experts.  This prevents zero-damage promotions at lambda zero.
        candidate = (objective, stored_bytes, remaining, count)
        if best is None or candidate < best:
            best = candidate
            best_count = count
    mask = np.zeros(C.NUM_EXPERTS, dtype=np.bool_)
    mask[order[:best_count]] = True
    return mask


def choose_sqrt_c_lagrangian(
    damage: np.ndarray,
    x4_expert_bytes: np.ndarray,
    *,
    lagrange_lambda: float,
    target_container_bytes: int | None = None,
) -> SqrtCAllocation:
    damage, costs = _validate_inputs(damage, x4_expert_bytes)
    if not math.isfinite(lagrange_lambda) or lagrange_lambda < 0:
        raise ValueError("SQRT-C Lagrange multiplier must be finite and non-negative")
    mask = np.stack(
        [
            _choose_layer_lagrangian(damage[row], costs[row], lagrange_lambda)
            for row in range(C.NUM_MOE_LAYERS)
        ]
    )
    container_bytes = sqrt_c_total_container_bytes(mask, costs)
    retained = float(damage[mask].sum())
    total_damage = float(damage.sum())
    slack = (
        None
        if target_container_bytes is None
        else int(target_container_bytes - container_bytes)
    )
    return SqrtCAllocation(
        x4_mask=mask,
        target_container_bytes=target_container_bytes,
        container_bytes=container_bytes,
        lagrange_lambda=lagrange_lambda,
        x4_experts=int(mask.sum()),
        compressed_experts=int(mask.size - mask.sum()),
        retained_damage=retained,
        remaining_damage=total_damage - retained,
        budget_slack_bytes=slack,
    )


def choose_sqrt_c_target(
    damage: np.ndarray,
    x4_expert_bytes: np.ndarray,
    *,
    target_container_bytes: int,
    iterations: int = 72,
) -> SqrtCAllocation:
    """Choose the closest supported Lagrangian point at or below a byte cap.

    This is exact for the reported multiplier.  A hard byte cap can lie
    between two supported rate-distortion points, so the returned allocation
    may leave less than one frontier jump of slack rather than claiming an
    exact 82,432-item knapsack proof.
    """

    damage, costs = _validate_inputs(damage, x4_expert_bytes)
    if isinstance(target_container_bytes, bool) or target_container_bytes <= 0:
        raise ValueError("SQRT-C target bytes must be a positive integer")
    minimum = sqrt_c_total_container_bytes(
        np.zeros_like(damage, dtype=np.bool_), costs
    )
    maximum = sqrt_c_total_container_bytes(
        np.ones_like(damage, dtype=np.bool_), costs
    )
    if target_container_bytes < minimum:
        raise ValueError(
            f"SQRT-C target {target_container_bytes} is below all-compressed "
            f"storage {minimum}"
        )
    if target_container_bytes >= maximum:
        return choose_sqrt_c_lagrangian(
            damage,
            costs,
            lagrange_lambda=0.0,
            target_container_bytes=target_container_bytes,
        )

    low = 0.0
    high = max(float(np.max(damage / costs)), np.finfo(np.float64).tiny)
    above = choose_sqrt_c_lagrangian(
        damage,
        costs,
        lagrange_lambda=low,
        target_container_bytes=target_container_bytes,
    )
    below = choose_sqrt_c_lagrangian(
        damage,
        costs,
        lagrange_lambda=high,
        target_container_bytes=target_container_bytes,
    )
    while below.container_bytes > target_container_bytes:
        high *= 2.0
        below = choose_sqrt_c_lagrangian(
            damage,
            costs,
            lagrange_lambda=high,
            target_container_bytes=target_container_bytes,
        )
    for _ in range(iterations):
        middle = (low + high) / 2.0
        candidate = choose_sqrt_c_lagrangian(
            damage,
            costs,
            lagrange_lambda=middle,
            target_container_bytes=target_container_bytes,
        )
        if candidate.container_bytes > target_container_bytes:
            low = middle
            above = candidate
        else:
            high = middle
            below = candidate
    if below.container_bytes > target_container_bytes:
        raise AssertionError("SQRT-C target search failed to find a feasible point")
    return below


def sqrt_c_allocation_document(
    pool: MixedCandidatePool,
    x4_index: X4CostIndex,
    allocation: SqrtCAllocation,
) -> dict:
    mask = allocation.x4_mask
    if (
        pool.damage.shape != mask.shape
        or x4_index.expert_storage_bytes.shape != mask.shape
    ):
        raise ValueError("SQRT-C pool, X4 index, and allocation shapes disagree")
    exact_bytes = sqrt_c_total_container_bytes(mask, x4_index.expert_storage_bytes)
    if exact_bytes != allocation.container_bytes:
        raise ValueError("SQRT-C allocation byte ledger does not close")
    layers: dict[str, dict] = {}
    for row, layer in enumerate(C.MOE_LAYERS):
        x4 = np.flatnonzero(mask[row]).tolist()
        compressed = np.flatnonzero(~mask[row]).tolist()
        format_codes = [
            FORMAT_MXFP4
            if mask[row, expert]
            else ExpertFormatSpec.compressed(
                int(pool.selected_r13[row, expert]),
                int(pool.selected_r2[row, expert]),
            ).code
            for expert in range(C.NUM_EXPERTS)
        ]
        layers[str(layer)] = {
            "x4": x4,
            # Compatibility alias for readers whose semantic distinction is
            # supplied by meta.high_tier_storage.
            "keep": x4,
            "compressed": compressed,
            "format_codes": format_codes,
            "trellis_slab_bytes": sqrt_c_trellis_layer_bytes(len(x4)),
            "x4_sidecar_bytes": X4_DATA_OFFSET
            + int(x4_index.expert_storage_bytes[row, mask[row]].sum()),
        }
    parameters = C.NUM_MOE_LAYERS * C.NUM_EXPERTS * 3 * 3072 * 3584
    return {
        "kind": SQRT_C_ALLOCATION_KIND,
        "schema_version": SQRT_C_ALLOCATION_SCHEMA_VERSION,
        "meta": {
            "codec": "SQRT-C",
            "tp_size": 12,
            "high_tier_storage": KEEP_STORAGE_EXTERNAL_X4,
            "candidate_pool": str(pool.root),
            "candidate_pool_content_sha256": pool.content_sha256,
            "candidate_codebook": pool.codebook,
            "candidate_mode_ids": list(pool.mode_ids),
            "x4_cost_index": str(x4_index.root),
            "x4_cost_index_content_sha256": x4_index.content_sha256,
            "source_revision": pool.manifest["source_revision"],
            "damage_metric": pool.damage_metric,
            "damage_weighting": pool.damage_weighting,
            "damage_provenance": pool.damage_provenance,
            "x4_experts": allocation.x4_experts,
            "compressed_experts": allocation.compressed_experts,
            "realized_x4_fraction": allocation.x4_experts / mask.size,
            "target_container_bytes": allocation.target_container_bytes,
            "container_bytes": allocation.container_bytes,
            "budget_slack_bytes": allocation.budget_slack_bytes,
            "effective_bits_per_original_expert_weight": (
                allocation.container_bytes * 8 / parameters
            ),
            "lagrange_lambda_damage_per_byte": allocation.lagrange_lambda,
            "allocation_optimality": "exact-for-reported-lagrange-multiplier",
            "all_compressed_damage": float(pool.damage.sum()),
            "x4_damage_avoided": allocation.retained_damage,
            "remaining_compressed_damage": allocation.remaining_damage,
            "format_histogram_before_x4": pool.format_histogram,
        },
        "layers": layers,
    }


def write_sqrt_c_allocation(path: str | Path, document: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(document, indent=1, sort_keys=True))
    temporary.replace(path)
    return path
