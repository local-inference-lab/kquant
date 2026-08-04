"""Select a deterministic, keep-tier-blind TP12 ``w2`` source-weight sample.

Experts are ranked by natural-routing applied-gate-square mass, divided into
equal-population traffic strata, and sampled by a seeded hash.  The current
retained tier is recorded only after selection and cannot affect membership.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from kquant.capture import LayerSamples, load_layer_samples
from kquant.interim_weights import InterimKeepStore
from scripts.experiment_retained_w2_rate_transfer import (
    _read_json,
    _request_documents,
    _validate_corpus_reports,
)


def _request_mask(observations: torch.Tensor, request_documents: dict[int, str]):
    steps = torch.bitwise_right_shift(observations, 32)
    allowed = torch.tensor(sorted(request_documents), dtype=steps.dtype)
    return torch.isin(steps, allowed), steps


def _support(
    samples: LayerSamples, request_documents: dict[int, str]
) -> tuple[torch.Tensor, dict[int, set[str]]]:
    mask, steps = _request_mask(samples.mid_observations, request_documents)
    counts = torch.bincount(samples.mid_experts[mask].long(), minlength=896)
    documents: dict[int, set[str]] = {}
    for expert, step in zip(
        samples.mid_experts[mask].tolist(), steps[mask].tolist(), strict=True
    ):
        documents.setdefault(int(expert), set()).add(request_documents[int(step)])
    return counts, documents


def _traffic(
    samples: LayerSamples, request_documents: dict[int, str]
) -> torch.Tensor:
    mask, _steps = _request_mask(samples.input_observations, request_documents)
    experts = samples.input_experts[mask].long().reshape(-1)
    weights = samples.input_gates[mask].double().square().reshape(-1)
    traffic = torch.zeros(896, dtype=torch.float64)
    traffic.scatter_add_(0, experts, weights)
    return traffic


def _hash_order(seed: int, layer: int, expert: int) -> int:
    digest = hashlib.blake2b(
        f"{seed}:{layer}:{expert}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little")


def _stratified_sample(
    eligible: list[int],
    traffic: torch.Tensor,
    *,
    layer: int,
    sample_size: int,
    strata: int,
    seed: int,
) -> tuple[list[int], dict[int, int]]:
    if sample_size > len(eligible):
        raise ValueError("sample size exceeds eligible experts")
    if not 1 <= strata <= sample_size:
        raise ValueError("strata must be in 1..sample_size")
    ordered = sorted(eligible, key=lambda expert: (float(traffic[expert]), expert))
    buckets = [[] for _ in range(strata)]
    for rank, expert in enumerate(ordered):
        bucket = min(strata - 1, rank * strata // len(ordered))
        buckets[bucket].append(expert)
    base, remainder = divmod(sample_size, strata)
    selected = []
    membership = {}
    for stratum, bucket in enumerate(buckets):
        quota = base + (stratum < remainder)
        if len(bucket) < quota:
            raise ValueError(f"traffic stratum {stratum} cannot fill quota {quota}")
        chosen = sorted(
            bucket,
            key=lambda expert: (_hash_order(seed, layer, expert), expert),
        )[:quota]
        selected.extend(chosen)
        membership.update({expert: stratum for expert in chosen})
    return sorted(selected), membership


def run(args: argparse.Namespace) -> dict:
    provenance = _validate_corpus_reports(
        args.training_report,
        args.validation_report,
        training_capture=args.capture,
        validation_capture=args.validation_capture,
    )
    training_documents = _request_documents(_read_json(args.training_report))
    validation_documents = _request_documents(_read_json(args.validation_report))
    keep_store = InterimKeepStore(args.checkpoint)
    layer_results = {}
    for layer in args.layers:
        training = load_layer_samples(args.capture, layer - 1)
        validation = load_layer_samples(args.validation_capture, layer - 1)
        train_counts, train_docs = _support(training, training_documents)
        validation_counts, validation_docs = _support(
            validation, validation_documents
        )
        traffic = _traffic(training, training_documents)
        eligible = [
            expert
            for expert in range(896)
            if int(train_counts[expert]) >= args.min_training_rows
            and int(validation_counts[expert]) >= args.min_validation_rows
        ]
        selected, strata = _stratified_sample(
            eligible,
            traffic,
            layer=layer,
            sample_size=args.experts_per_layer,
            strata=args.traffic_strata,
            seed=args.seed,
        )
        retained = set(keep_store.kept_experts(layer))
        layer_results[str(layer)] = {
            "eligible_experts": len(eligible),
            "selected_experts": selected,
            "selected_retained": sum(expert in retained for expert in selected),
            "selected_compressed_in_interim": sum(
                expert not in retained for expert in selected
            ),
            "experts": {
                str(expert): {
                    "traffic_stratum_low_to_high": strata[expert],
                    "training_gate_square_mass": float(traffic[expert]),
                    "training_mid_rows": int(train_counts[expert]),
                    "training_mid_documents": len(train_docs.get(expert, set())),
                    "validation_mid_rows": int(validation_counts[expert]),
                    "validation_mid_documents": len(
                        validation_docs.get(expert, set())
                    ),
                    "interim_tier": (
                        "retained_mxfp4" if expert in retained else "exl3"
                    ),
                }
                for expert in selected
            },
        }
    result = {
        "kind": "kquant_keep_tier_blind_w2_source_sample",
        "schema_version": 1,
        "selection_uses_keep_tier": False,
        "checkpoint": str(args.checkpoint.resolve()),
        "capture_provenance": provenance,
        "criteria": {
            "layers": list(args.layers),
            "experts_per_layer": args.experts_per_layer,
            "traffic_strata": args.traffic_strata,
            "min_training_mid_rows": args.min_training_rows,
            "min_validation_mid_rows": args.min_validation_rows,
            "seed": args.seed,
            "ranking": "natural-routing applied-gate-square mass",
            "within_stratum_selection": "seeded blake2b(layer,expert)",
        },
        "layers": layer_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=1, sort_keys=True))
    temporary.replace(args.output)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/models/Kimi-K3-EXL3-3p09-serve"),
    )
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--validation-capture", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument(
        "--layers",
        type=lambda value: tuple(int(part) for part in value.split(",")),
        default=(1, 24, 40),
    )
    parser.add_argument("--experts-per-layer", type=int, default=48)
    parser.add_argument("--traffic-strata", type=int, default=8)
    parser.add_argument("--min-training-rows", type=int, default=4)
    parser.add_argument("--min-validation-rows", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.layers or any(not 1 <= layer <= 92 for layer in args.layers):
        parser.error("--layers must contain decoder MoE layers in 1..92")
    if args.experts_per_layer < 1 or args.traffic_strata < 1:
        parser.error("sample size and traffic strata must be positive")
    if min(args.min_training_rows, args.min_validation_rows) < 1:
        parser.error("support thresholds must be positive")
    return args


def main() -> None:
    args = parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                layer: {
                    key: value[key]
                    for key in (
                        "eligible_experts",
                        "selected_experts",
                        "selected_retained",
                        "selected_compressed_in_interim",
                    )
                }
                for layer, value in result["layers"].items()
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
