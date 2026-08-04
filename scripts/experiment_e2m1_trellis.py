#!/usr/bin/env python3
"""Evaluate a minimal E2M1 trellis directly on one official K3 MoE layer.

The study preserves the source E8M0/K32 scales, applies no Hadamard or sign
transform, and never materializes an FP16/FP32 source matrix.  It compares
uniform K2/K3/K4 paths and diagnostic equal-rate K2/K4 allocations on sampled
16x16 source tiles.  K4 is an exact native-E2M1 control.

The transition labelling is trained on separate experts.  Both native nibble
order and an opposite-zero seed are retained as controls; production metrics
use a rate-specific table that preserves both numerical-zero labels but learns
their transition placement along with the other fourteen labels.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Iterable

import torch

from kquant import constants as C
from kquant.e2m1_trellis import (
    E2M1_VALUES,
    encode_tailbiting,
    identity_transition_codes,
    opposite_zero_transition_codes,
    tailbiting_cost,
)
from kquant.exl3_reference import tensor_core_permutation
from kquant.io.mxfp4 import unpack_codes
from kquant.io.stream import iter_expert_batches
from kquant.source_weights import OfficialMXFP4Store, PackedMXFP4Matrix


TILE_SIDE = 16
TILE_VALUES = TILE_SIDE * TILE_SIDE
ZERO_POSITIONS = (0, 15)
ZERO_CODES = (0, 8)


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    temporary.replace(path)


def _parse_experts(value: str) -> tuple[int, ...]:
    value = value.strip().lower()
    if value == "all":
        return tuple(range(C.NUM_EXPERTS))
    selected: list[int] = []
    try:
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                selected.append(int(item))
                continue
            pieces = [int(piece) if piece else None for piece in item.split(":")]
            if len(pieces) not in (2, 3):
                raise ValueError
            start = 0 if pieces[0] is None else pieces[0]
            stop = C.NUM_EXPERTS if pieces[1] is None else pieces[1]
            step = 1 if len(pieces) == 2 or pieces[2] is None else pieces[2]
            selected.extend(range(start, stop, step))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "experts must be 'all', comma-separated IDs, or Python-style ranges"
        ) from exc
    if not selected or len(selected) != len(set(selected)):
        raise argparse.ArgumentTypeError("experts must be nonempty and unique")
    if any(expert < 0 or expert >= C.NUM_EXPERTS for expert in selected):
        raise argparse.ArgumentTypeError("expert IDs must be in 0..895")
    return tuple(selected)


def _spread_experts(count: int, *, offset: int = 0, excluded: set[int] | None = None) -> tuple[int, ...]:
    if count <= 0:
        return ()
    excluded = set() if excluded is None else excluded
    candidates = [
        (offset + round(index * (C.NUM_EXPERTS - 1) / max(count - 1, 1)))
        % C.NUM_EXPERTS
        for index in range(count)
    ]
    result: list[int] = []
    for candidate in candidates:
        while candidate in excluded or candidate in result:
            candidate = (candidate + 1) % C.NUM_EXPERTS
        result.append(candidate)
    return tuple(result)


def _sample_tile_indices(total: int, count: int, *, seed: int) -> torch.Tensor:
    if count <= 0:
        raise ValueError("tile count must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if count >= total:
        return torch.arange(total, dtype=torch.long)
    return torch.randperm(total, generator=generator)[:count]


def sample_mxfp4_tiles(
    raw: PackedMXFP4Matrix,
    *,
    count: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return native codes, scale-square weights, and source energy per tile.

    Only uint8 codes are expanded for the complete matrix.  Floating-point
    scales are expanded for the sampled tiles alone.
    """

    codes = unpack_codes(raw.packed)
    rows, columns = codes.shape
    if rows % TILE_SIDE or columns % TILE_SIDE:
        raise ValueError(f"matrix shape {(rows, columns)} is not 16x16 tiled")
    row_tiles = rows // TILE_SIDE
    column_tiles = columns // TILE_SIDE
    tile_ids = _sample_tile_indices(row_tiles * column_tiles, count, seed=seed)
    tile_rows = torch.div(tile_ids, column_tiles, rounding_mode="floor")
    tile_columns = tile_ids.remainder(column_tiles)
    lane = torch.arange(TILE_SIDE, dtype=torch.long)
    row_indices = tile_rows[:, None] * TILE_SIDE + lane[None, :]
    column_indices = tile_columns[:, None] * TILE_SIDE + lane[None, :]
    # Official tensors are [out, in], whereas the GEMM/trellis tile is [K, N]
    # or [in, out].  Transpose each logical block before applying EXL's native
    # tensor-core lane order.
    tiles = codes[
        row_indices[:, :, None],
        column_indices[:, None, :],
    ].transpose(1, 2).reshape(-1, TILE_VALUES)

    # Every 16-column tile lies wholly within one original K32 scale block.
    # Its sixteen source scale bytes vary only over output rows.
    scale_columns = torch.div(tile_columns, 2, rounding_mode="floor")
    scale_codes = raw.scale[row_indices, scale_columns[:, None]]
    scale_factors = torch.exp2(scale_codes.float() - C.E8M0_BIAS)
    scale_weights = (
        scale_factors.square()
        .unsqueeze(-1)
        .expand(-1, TILE_SIDE, TILE_SIDE)
        .transpose(1, 2)
        .reshape(-1, TILE_VALUES)
    )

    permutation = tensor_core_permutation()
    tiles = tiles.index_select(1, permutation).to(device=device, non_blocking=True)
    scale_weights = scale_weights.index_select(1, permutation).to(
        device=device, non_blocking=True
    )
    values = E2M1_VALUES.to(device=device)[tiles.long()]
    source_energy = (values.square() * scale_weights).sum(dim=1)
    return tiles, scale_weights, source_energy


def _mapping_score(
    codes: torch.Tensor,
    weights: torch.Tensor,
    bits: int,
    mapping: tuple[int, ...],
    *,
    batch_size: int,
) -> float:
    total = 0.0
    with torch.inference_mode():
        for start in range(0, codes.shape[0], batch_size):
            stop = min(start + batch_size, codes.shape[0])
            cost = tailbiting_cost(
                codes[start:stop],
                bits,
                mapping,
                weights=weights[start:stop],
            )
            total += float(cost.double().sum().item())
    return total


def _random_opposite_zero_mapping(generator: random.Random) -> tuple[int, ...]:
    labels = [code for code in range(16) if code not in ZERO_CODES]
    generator.shuffle(labels)
    mapping = [0, *labels, 8]
    assert len(mapping) == 16
    return tuple(mapping)


def _random_mapping(generator: random.Random) -> tuple[int, ...]:
    mapping = list(range(16))
    generator.shuffle(mapping)
    return tuple(mapping)


def _hill_climb_mapping(
    codes: torch.Tensor,
    weights: torch.Tensor,
    bits: int,
    initial: tuple[int, ...],
    initial_score: float,
    *,
    steps: int,
    movable: list[int],
    generator: random.Random,
    batch_size: int,
) -> tuple[tuple[int, ...], float, int, int]:
    best = initial
    best_score = initial_score
    accepted = 0
    evaluated = 0
    for _ in range(steps):
        left, right = generator.sample(movable, 2)
        candidate_list = list(best)
        candidate_list[left], candidate_list[right] = (
            candidate_list[right],
            candidate_list[left],
        )
        candidate = tuple(candidate_list)
        score = _mapping_score(codes, weights, bits, candidate, batch_size=batch_size)
        evaluated += 1
        if score < best_score:
            best, best_score = candidate, score
            accepted += 1
    return best, best_score, accepted, evaluated


def optimize_mapping(
    codes: torch.Tensor,
    weights: torch.Tensor,
    bits: int,
    *,
    random_trials: int,
    hill_steps: int,
    seed: int,
    batch_size: int,
) -> tuple[tuple[int, ...], dict]:
    """Search a global table, including rather than assuming zero placement."""

    generator = random.Random(seed + bits * 1_000_003)
    identity = identity_transition_codes()
    opposite = opposite_zero_transition_codes()
    control_scores = {
        "native_nibble_order": _mapping_score(
            codes, weights, bits, identity, batch_size=batch_size
        ),
        "opposite_zero_seed": _mapping_score(
            codes, weights, bits, opposite, batch_size=batch_size
        ),
    }
    best_opposite = opposite
    best_opposite_score = control_scores["opposite_zero_seed"]
    best_free = (
        identity
        if control_scores["native_nibble_order"] < control_scores["opposite_zero_seed"]
        else opposite
    )
    best_free_score = min(control_scores.values())
    evaluated = 2
    for _ in range(random_trials):
        opposite_candidate = _random_opposite_zero_mapping(generator)
        opposite_score = _mapping_score(
            codes, weights, bits, opposite_candidate, batch_size=batch_size
        )
        free_candidate = _random_mapping(generator)
        free_score = _mapping_score(
            codes, weights, bits, free_candidate, batch_size=batch_size
        )
        evaluated += 2
        if opposite_score < best_opposite_score:
            best_opposite, best_opposite_score = opposite_candidate, opposite_score
        if free_score < best_free_score:
            best_free, best_free_score = free_candidate, free_score

    best_opposite, best_opposite_score, opposite_accepted, opposite_evaluated = (
        _hill_climb_mapping(
            codes,
            weights,
            bits,
            best_opposite,
            best_opposite_score,
            steps=hill_steps,
            movable=list(range(1, 15)),
            generator=generator,
            batch_size=batch_size,
        )
    )
    best_free, best_free_score, free_accepted, free_evaluated = _hill_climb_mapping(
        codes,
        weights,
        bits,
        best_free,
        best_free_score,
        steps=hill_steps,
        movable=list(range(16)),
        generator=generator,
        batch_size=batch_size,
    )
    evaluated += opposite_evaluated + free_evaluated
    best, best_score = (
        (best_free, best_free_score)
        if best_free_score <= best_opposite_score
        else (best_opposite, best_opposite_score)
    )

    def describe(mapping: tuple[int, ...], score: float) -> dict:
        return {
            "weighted_sse": score,
            "transition_codes": list(mapping),
            "transition_values": [float(C.E2M1_LUT[code]) for code in mapping],
            "zero_transition_indices": [
                index for index, code in enumerate(mapping) if code in ZERO_CODES
            ],
        }

    return best, {
        "bits": bits,
        "evaluated_tables": evaluated,
        "accepted_hill_swaps": {
            "opposite_zero": opposite_accepted,
            "unconstrained": free_accepted,
        },
        "training_weighted_sse": best_score,
        "control_weighted_sse": control_scores,
        "transition_codes": list(best),
        "transition_values": [float(C.E2M1_LUT[code]) for code in best],
        "zero_transition_indices": [
            index for index, code in enumerate(best) if code in ZERO_CODES
        ],
        "selected_family": (
            "unconstrained" if best_free_score <= best_opposite_score else "opposite_zero"
        ),
        "best_opposite_zero": describe(best_opposite, best_opposite_score),
        "best_unconstrained": describe(best_free, best_free_score),
    }


def _collect_mapping_tiles(
    store: OfficialMXFP4Store,
    experts: Iterable[int],
    *,
    layer: int,
    tiles_per_matrix: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    all_codes = []
    all_weights = []
    all_energy = []
    for expert in experts:
        for matrix_index, matrix in enumerate(C.EXPERT_MATRICES):
            raw = store.load_packed_matrix(layer, expert, matrix)
            codes, weights, energy = sample_mxfp4_tiles(
                raw,
                count=tiles_per_matrix,
                seed=seed + expert * 101 + matrix_index * 10_007,
                device=device,
            )
            all_codes.append(codes)
            all_weights.append(weights)
            all_energy.append(energy)
    return (
        torch.cat(all_codes, dim=0),
        torch.cat(all_weights, dim=0),
        torch.cat(all_energy, dim=0),
    )


def _validate_mappings(
    codes: torch.Tensor,
    weights: torch.Tensor,
    energy: torch.Tensor,
    mappings: dict[int, tuple[int, ...]],
    mapping_search: dict[str, dict],
    *,
    batch_size: int,
) -> dict:
    total_energy = float(energy.double().sum().item())
    result = {
        "tiles": codes.shape[0],
        "coefficients": codes.numel(),
        "source_energy": total_energy,
        "by_rate": {},
    }
    for bits in (2, 3):
        entries = {}
        candidates = (
            ("native_nibble_order", identity_transition_codes()),
            ("opposite_zero_seed", opposite_zero_transition_codes()),
            (
                "optimized_opposite_zero",
                tuple(mapping_search[str(bits)]["best_opposite_zero"]["transition_codes"]),
            ),
        )
        # The selected table can put zeros anywhere the training objective earns.
        candidates = candidates + (("selected_optimized", mappings[bits]),)
        for name, mapping in candidates:
            error = _mapping_score(
                codes, weights, bits, mapping, batch_size=batch_size
            )
            entries[name] = {
                "weighted_sse": error,
                "nmse": error / total_energy if total_energy else math.nan,
            }
        result["by_rate"][str(bits)] = entries
    return result


def _tile_metrics(
    codes: torch.Tensor,
    weights: torch.Tensor,
    energy: torch.Tensor,
    mappings: dict[int, tuple[int, ...]],
) -> dict:
    with torch.inference_mode():
        k2 = tailbiting_cost(codes, 2, mappings[2], weights=weights).double()
        k3 = tailbiting_cost(codes, 3, mappings[3], weights=weights).double()
    tile_count = codes.shape[0]
    paired = tile_count - tile_count % 2
    pair_cost = k2[:paired].reshape(-1, 2)
    fixed_mixed = float(pair_cost[:, 0].sum().item())
    pair_oracle = float(pair_cost.amin(dim=1).sum().item())
    global_oracle = float(torch.sort(k2).values[: tile_count // 2].sum().item())
    k2_error = float(k2.sum().item())
    k3_error = float(k3.sum().item())
    source_energy = float(energy.double().sum().item())
    source_histogram = torch.bincount(codes.reshape(-1).long(), minlength=16)
    return {
        "tiles": tile_count,
        "coefficients": codes.numel(),
        "source_energy": source_energy,
        "source_code_histogram": [int(value) for value in source_histogram.tolist()],
        "numeric_zero_fraction": float(
            (source_histogram[0] + source_histogram[8]).double().item()
            / codes.numel()
        ),
        "uniform": {
            "K2": {"weighted_sse": k2_error, "nmse": k2_error / source_energy},
            "K3": {"weighted_sse": k3_error, "nmse": k3_error / source_energy},
            "K4": {"weighted_sse": 0.0, "nmse": 0.0},
        },
        "equal_rate_K2_K4": {
            "fixed_alternating_weighted_sse": fixed_mixed,
            "fixed_alternating_to_K3": fixed_mixed / k3_error if k3_error else math.nan,
            "within_pair_oracle_weighted_sse": pair_oracle,
            "within_pair_oracle_to_K3": pair_oracle / k3_error if k3_error else math.nan,
            "global_half_oracle_weighted_sse": global_oracle,
            "global_half_oracle_to_K3": global_oracle / k3_error if k3_error else math.nan,
            "warning": (
                "tile-oracle assignments are diagnostic upper bounds and would require "
                "a rate map; they are not the proposed fixed-record TSH mode"
            ),
        },
    }


def _aggregate(results: dict[str, dict], training_experts: set[int]) -> dict:
    groups = {
        "all": list(results.values()),
        "mapping_holdout": [
            value for value in results.values() if value["expert"] not in training_experts
        ],
    }
    summary = {}
    for group_name, entries in groups.items():
        if not entries:
            continue
        source_energy = sum(entry["metrics"]["source_energy"] for entry in entries)
        k2 = sum(entry["metrics"]["uniform"]["K2"]["weighted_sse"] for entry in entries)
        k3 = sum(entry["metrics"]["uniform"]["K3"]["weighted_sse"] for entry in entries)
        fixed = sum(
            entry["metrics"]["equal_rate_K2_K4"]["fixed_alternating_weighted_sse"]
            for entry in entries
        )
        paired = sum(
            entry["metrics"]["equal_rate_K2_K4"]["within_pair_oracle_weighted_sse"]
            for entry in entries
        )
        global_oracle = sum(
            entry["metrics"]["equal_rate_K2_K4"]["global_half_oracle_weighted_sse"]
            for entry in entries
        )
        coefficients = sum(entry["metrics"]["coefficients"] for entry in entries)
        zero_count = sum(
            entry["metrics"]["source_code_histogram"][0]
            + entry["metrics"]["source_code_histogram"][8]
            for entry in entries
        )
        summary[group_name] = {
            "matrices": len(entries),
            "experts": len({entry["expert"] for entry in entries}),
            "coefficients": coefficients,
            "numeric_zero_fraction": zero_count / coefficients,
            "source_energy": source_energy,
            "uniform_K2_nmse": k2 / source_energy,
            "uniform_K3_nmse": k3 / source_energy,
            "fixed_K2_K4_to_K3": fixed / k3 if k3 else math.nan,
            "pair_oracle_K2_K4_to_K3": paired / k3 if k3 else math.nan,
            "global_oracle_K2_K4_to_K3": global_oracle / k3 if k3 else math.nan,
        }
    return summary


def _iter_selected_batches(
    store: OfficialMXFP4Store,
    *,
    layer: int,
    experts: tuple[int, ...],
    batch_size: int,
) -> Iterable[tuple[str, list[tuple[int, PackedMXFP4Matrix]]]]:
    """Avoid a full layer scan when a small explicit expert subset is requested."""

    if len(experts) < C.NUM_EXPERTS // 2:
        for matrix in C.EXPERT_MATRICES:
            for start in range(0, len(experts), batch_size):
                chunk = experts[start : start + batch_size]
                yield matrix, [
                    (expert, store.load_packed_matrix(layer, expert, matrix))
                    for expert in chunk
                ]
        return

    wanted = set(experts)
    for batch in iter_expert_batches(
        store.cache,
        batch_size=batch_size,
        layers={layer},
        matrices=C.EXPERT_MATRICES,
    ):
        selected = []
        for offset, expert_tensor in enumerate(batch.experts):
            expert = int(expert_tensor.item())
            if expert in wanted:
                selected.append(
                    (
                        expert,
                        PackedMXFP4Matrix(batch.packed[offset], batch.scale[offset]),
                    )
                )
        if selected:
            yield batch.matrix, selected


def _signature(args: argparse.Namespace, experts: tuple[int, ...], train: tuple[int, ...], validation: tuple[int, ...], store: OfficialMXFP4Store) -> dict:
    return {
        "source_checkpoint": str(store.root),
        "source_revision": store.revision,
        "layer": args.layer,
        "experts": list(experts),
        "mapping_train_experts": list(train),
        "mapping_validation_experts": list(validation),
        "mapping_tiles_per_matrix": args.mapping_tiles_per_matrix,
        "eval_tiles_per_matrix": args.eval_tiles_per_matrix,
        "mapping_random_trials": args.mapping_random_trials,
        "mapping_hill_steps": args.mapping_hill_steps,
        "seed": args.seed,
        "tile_order": (
            "official [out,in] blocks transposed to GEMM [K,N], then EXL "
            "tensor-core order; no value transform"
        ),
        "scale_contract": "original E8M0 scale per K32 block, unchanged",
        "transform": "none",
        "trellis": "4-K carried bits; tail-biting; no 16-bit MCG window",
    }


def run(args: argparse.Namespace) -> dict:
    if args.layer not in C.MOE_LAYERS:
        raise ValueError(f"layer must be one of {C.MOE_LAYERS[0]}..{C.MOE_LAYERS[-1]}")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    store = OfficialMXFP4Store(
        repo_dir=args.official_repo_dir,
        revision=args.official_revision,
    )
    experts = args.experts
    training_experts = (
        args.mapping_train_experts
        if args.mapping_train_experts is not None
        else _spread_experts(args.mapping_train_expert_count)
    )
    validation_experts = (
        args.mapping_validation_experts
        if args.mapping_validation_experts is not None
        else _spread_experts(
            args.mapping_validation_expert_count,
            offset=53,
            excluded=set(training_experts),
        )
    )
    if set(training_experts) & set(validation_experts):
        raise ValueError("mapping training and validation experts must be disjoint")
    signature = _signature(
        args, experts, training_experts, validation_experts, store
    )

    if args.resume:
        if not args.output.is_file():
            raise FileNotFoundError(args.output)
        payload = json.loads(args.output.read_text())
        if payload.get("signature") != signature:
            raise ValueError("resume output does not match the requested experiment")
        mappings = {
            int(bits): tuple(section["transition_codes"])
            for bits, section in payload["mapping_search"].items()
        }
        if payload.get("complete"):
            return payload
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        started = time.time()
        train_codes, train_weights, train_energy = _collect_mapping_tiles(
            store,
            training_experts,
            layer=args.layer,
            tiles_per_matrix=args.mapping_tiles_per_matrix,
            seed=args.seed,
            device=device,
        )
        mappings = {}
        mapping_search = {}
        for bits in (2, 3):
            mapping, report = optimize_mapping(
                train_codes,
                train_weights,
                bits,
                random_trials=args.mapping_random_trials,
                hill_steps=args.mapping_hill_steps,
                seed=args.seed,
                batch_size=args.viterbi_batch_size,
            )
            mappings[bits] = mapping
            mapping_search[str(bits)] = report
        validation_codes, validation_weights, validation_energy = _collect_mapping_tiles(
            store,
            validation_experts,
            layer=args.layer,
            tiles_per_matrix=args.mapping_tiles_per_matrix,
            seed=args.seed + 8_000_003,
            device=device,
        )
        payload = {
            "kind": "kquant_native_e2m1_small_state_trellis_study",
            "schema_version": 1,
            "signature": signature,
            "alphabet": {
                "wire_symbols": 16,
                "distinct_numeric_values": 15,
                "values": [float(value) for value in C.E2M1_LUT],
                "duplicate_zero_codes": list(ZERO_CODES),
                "duplicate_zero_policy": (
                    "retain both wire labels; compare opposite corners with learned placement"
                ),
            },
            "geometry": {
                "tile": "16x16 / 256 coefficients",
                "K2": {"carried_bits": 2, "states": 4, "payload_bytes_per_tile": 64},
                "K3": {"carried_bits": 1, "states": 2, "payload_bytes_per_tile": 96},
                "K4": {"carried_bits": 0, "states": 1, "payload_bytes_per_tile": 128},
                "unchanged_scale_bytes_attributed_per_tile": 8,
                "K2_K4_pair_bytes": 208,
                "K3_K3_pair_bytes": 208,
                "effective_bpw_including_original_scales": {
                    "K2": 2.25,
                    "K3": 3.25,
                    "K4": 4.25,
                    "equal_K2_K4": 3.25,
                },
            },
            "mapping_training": {
                "tiles": train_codes.shape[0],
                "coefficients": train_codes.numel(),
                "source_energy": float(train_energy.double().sum().item()),
            },
            "mapping_search": mapping_search,
            "mapping_validation": _validate_mappings(
                validation_codes,
                validation_weights,
                validation_energy,
                mappings,
                mapping_search,
                batch_size=args.viterbi_batch_size,
            ),
            "results": {},
            "summary": {},
            "started_unix": started,
            "complete": False,
        }
        _atomic_write(args.output, payload)

    completed = set(payload["results"])
    processed_batches = 0
    for matrix, selected_batch in _iter_selected_batches(
        store,
        layer=args.layer,
        experts=experts,
        batch_size=args.expert_batch_size,
    ):
        sampled_codes = []
        sampled_weights = []
        sampled_energy = []
        identities = []
        for expert, raw in selected_batch:
            key = f"{expert}:{matrix}"
            if key in completed:
                continue
            matrix_index = C.EXPERT_MATRICES.index(matrix)
            codes, weights, energy = sample_mxfp4_tiles(
                raw,
                count=args.eval_tiles_per_matrix,
                seed=args.seed + 16_000_057 + expert * 101 + matrix_index * 10_007,
                device=device,
            )
            sampled_codes.append(codes)
            sampled_weights.append(weights)
            sampled_energy.append(energy)
            identities.append((key, expert, codes.shape[0]))
        if not identities:
            continue

        codes = torch.cat(sampled_codes, dim=0)
        weights = torch.cat(sampled_weights, dim=0)
        energy = torch.cat(sampled_energy, dim=0)
        cursor = 0
        for key, expert, tile_count in identities:
            stop = cursor + tile_count
            metrics = _tile_metrics(
                codes[cursor:stop],
                weights[cursor:stop],
                energy[cursor:stop],
                mappings,
            )
            payload["results"][key] = {
                "layer": args.layer,
                "expert": expert,
                "matrix": matrix,
                "mapping_training_expert": expert in set(training_experts),
                "metrics": metrics,
            }
            completed.add(key)
            cursor = stop
        processed_batches += 1
        if processed_batches % args.write_every_batches == 0:
            payload["summary"] = _aggregate(
                payload["results"], set(training_experts)
            )
            payload["updated_unix"] = time.time()
            _atomic_write(args.output, payload)

    expected = {
        f"{expert}:{matrix}" for expert in experts for matrix in C.EXPERT_MATRICES
    }
    missing = sorted(expected - set(payload["results"]))
    payload["summary"] = _aggregate(payload["results"], set(training_experts))
    payload["missing"] = missing
    payload["updated_unix"] = time.time()
    payload["elapsed_seconds"] = payload["updated_unix"] - payload["started_unix"]
    payload["complete"] = not missing
    _atomic_write(args.output, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--experts", type=_parse_experts, default=_parse_experts("all"))
    parser.add_argument("--mapping-train-experts", type=_parse_experts)
    parser.add_argument("--mapping-validation-experts", type=_parse_experts)
    parser.add_argument("--mapping-train-expert-count", type=int, default=8)
    parser.add_argument("--mapping-validation-expert-count", type=int, default=8)
    parser.add_argument("--mapping-tiles-per-matrix", type=int, default=64)
    parser.add_argument("--eval-tiles-per-matrix", type=int, default=64)
    parser.add_argument("--mapping-random-trials", type=int, default=32)
    parser.add_argument("--mapping-hill-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--expert-batch-size", type=int, default=8)
    parser.add_argument("--viterbi-batch-size", type=int, default=4096)
    parser.add_argument("--write-every-batches", type=int, default=1)
    parser.add_argument("--official-repo-dir", type=Path)
    parser.add_argument("--official-revision", default=C.REVISION)
    parser.add_argument(
        "--output", type=Path, default=Path("out/e2m1-trellis-layer24.json")
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cpu_threads <= 0 or args.expert_batch_size <= 0 or args.viterbi_batch_size <= 0:
        raise SystemExit("thread and batch sizes must be positive")
    if args.mapping_train_expert_count <= 0 or args.mapping_validation_expert_count <= 0:
        raise SystemExit("mapping expert counts must be positive")
    if args.mapping_tiles_per_matrix <= 0 or args.eval_tiles_per_matrix <= 1:
        raise SystemExit("tile counts must be positive and eval must contain a pair")
    result = run(args)
    print(json.dumps({"complete": result["complete"], "summary": result["summary"]}, indent=1))


if __name__ == "__main__":
    main()
