"""Compare lossy K3/K4 TSH with an exact compressed-MXFP4 tier.

The deployable selectors use only confirmation-fold routed SSE.  External
validation is used for reporting and for a clearly labelled oracle-regret
diagnostic.  Exact MXFP4 reconstruction has zero source-relative distortion;
only its measured, per-expert serialized byte count enters the allocation.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from scripts.summarize_uniform_k_mxfp4_endpoint import (
    _aggregate,
    _atomic_write,
    _candidate_record,
    _read_json,
)


Identity = tuple[int, int]


def _parse_targets(value: str) -> tuple[float, ...]:
    try:
        targets = tuple(float(part) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated bpw targets") from exc
    if not targets or len(set(targets)) != len(targets):
        raise argparse.ArgumentTypeError("targets must be nonempty and unique")
    if any(target <= 0.0 for target in targets):
        raise argparse.ArgumentTypeError("targets must be positive")
    return tuple(sorted(targets))


def _load_endpoint_records(
    path: Path, *, codebook: str
) -> tuple[dict[tuple[int, int, int], dict[str, object]], dict[str, object]]:
    payload = _read_json(path)
    if payload.get("kind") != "kquant_uniform_k_mxfp4_endpoint_study":
        raise ValueError("endpoint input has the wrong kind")
    if not bool(payload.get("complete")):
        raise ValueError("endpoint input is incomplete")
    signature = payload.get("signature")
    results = payload.get("results")
    if not isinstance(signature, Mapping) or not isinstance(results, Mapping):
        raise TypeError("endpoint input lacks its signature or results")
    layer = int(signature["layer"])
    records: dict[tuple[int, int, int], dict[str, object]] = {}
    for raw_expert, result in results.items():
        if not isinstance(result, Mapping):
            raise TypeError("endpoint expert result must be a mapping")
        candidates = result.get("candidates")
        if not isinstance(candidates, Mapping):
            raise TypeError("endpoint expert result lacks candidates")
        expert = int(raw_expert)
        for bits in (3, 4):
            candidate = candidates.get(f"{codebook}:K{bits}")
            if not isinstance(candidate, Mapping):
                raise ValueError(f"expert {layer}:{expert} lacks {codebook}:K{bits}")
            records[(layer, expert, bits)] = _candidate_record(
                candidate, layer=layer, expert=expert
            )
    return records, payload


def _load_exact_bytes(path: Path) -> tuple[dict[Identity, int], dict[str, object]]:
    payload = _read_json(path)
    if payload.get("kind") != "kquant_exact_mxfp4_scale_codec_experiment":
        raise ValueError("exact-tier input has the wrong kind")
    if not bool(payload.get("complete")):
        raise ValueError("exact-tier input is incomplete")
    results = payload.get("results")
    if not isinstance(results, Mapping):
        raise TypeError("exact-tier input lacks results")
    exact_bytes = {}
    for raw_layer, layer_results in results.items():
        if not isinstance(layer_results, Mapping):
            raise TypeError("exact-tier layer result must be a mapping")
        for raw_expert, result in layer_results.items():
            if not isinstance(result, Mapping) or not isinstance(
                result.get("aggregate"), Mapping
            ):
                raise TypeError("exact-tier expert result lacks an aggregate")
            exact_bytes[(int(raw_layer), int(raw_expert))] = int(
                result["aggregate"]["coded_total_bytes"]
            )
    return exact_bytes, payload


def _exact_record(k3: Mapping[str, object], *, bytes_stored: int) -> dict[str, object]:
    record = dict(k3)
    record.update(
        {
            "bits": 4,
            "codebook": "exact-mxfp4-row-palette",
            "bytes": bytes_stored,
            "raw_sse": 0.0,
            "dense_sse": 0.0,
            "confirmation_sse": 0.0,
            "validation_sse": 0.0,
            "validation_mixture_sse": 0.0,
        }
    )
    return record


def _greedy_binary_selection(
    low: Mapping[Identity, Mapping[str, object]],
    high: Mapping[Identity, Mapping[str, object]],
    *,
    target_bpw: float,
    selection_metric: str,
) -> tuple[dict[Identity, Mapping[str, object]], list[Identity]]:
    identities = sorted(low)
    if set(identities) != set(high):
        raise ValueError("low- and high-rate identities disagree")
    total_weights = sum(int(low[identity]["weights"]) for identity in identities)
    budget = round(target_bpw * total_weights / 8.0)
    current_bytes = sum(int(low[identity]["bytes"]) for identity in identities)
    if budget < current_bytes:
        raise ValueError("target lies below the low-rate endpoint")

    upgrades = []
    for identity in identities:
        extra_bytes = int(high[identity]["bytes"]) - int(low[identity]["bytes"])
        if extra_bytes <= 0:
            raise ValueError("high-rate candidate must consume more bytes")
        gain = float(low[identity][selection_metric]) - float(
            high[identity][selection_metric]
        )
        upgrades.append((gain / extra_bytes, gain, identity, extra_bytes))
    upgrades.sort(key=lambda value: (value[0], value[1], value[2]), reverse=True)

    selected: list[Identity] = []
    for _, _, identity, extra_bytes in upgrades:
        if current_bytes + extra_bytes <= budget:
            selected.append(identity)
            current_bytes += extra_bytes
    selected_set = set(selected)
    records = {
        identity: high[identity] if identity in selected_set else low[identity]
        for identity in identities
    }
    return records, sorted(selected)


def _binary_frontier_point(
    low: Mapping[Identity, Mapping[str, object]],
    high: Mapping[Identity, Mapping[str, object]],
    *,
    target_bpw: float,
    high_mode: str,
) -> dict[str, object]:
    selected_records, selected = _greedy_binary_selection(
        low,
        high,
        target_bpw=target_bpw,
        selection_metric="confirmation_sse",
    )
    oracle_records, oracle = _greedy_binary_selection(
        low,
        high,
        target_bpw=target_bpw,
        selection_metric="validation_sse",
    )
    evidence = _aggregate(list(selected_records.values()))
    oracle_evidence = _aggregate(list(oracle_records.values()))
    selected_set = set(selected)
    oracle_set = set(oracle)
    evidence.update(
        {
            "target_bpw": target_bpw,
            "high_mode": high_mode,
            "high_experts": len(selected),
            "high_fraction": len(selected) / len(low),
            "selected_high_experts": [list(identity) for identity in selected],
            "validation_oracle_high_experts": [list(identity) for identity in oracle],
            "selector_oracle_agreement": len(selected_set & oracle_set),
            "validation_oracle_routed_expert_nmse": oracle_evidence[
                "validation_routed_expert_nmse"
            ],
            "validation_selector_regret_sse": float(
                evidence["validation_routed_sse"]
            )
            - float(oracle_evidence["validation_routed_sse"]),
        }
    )
    return evidence


def _k4_on_per_expert_convex_hull(
    k3: Mapping[str, object],
    k4: Mapping[str, object],
    exact: Mapping[str, object],
    *,
    metric: str,
) -> bool:
    b3, b4, be = (float(value["bytes"]) for value in (k3, k4, exact))
    d3, d4, de = (float(value[metric]) for value in (k3, k4, exact))
    if not b3 < b4 < be:
        raise ValueError("expected K3 < K4 < exact byte ordering")
    first_slope = (d3 - d4) / (b4 - b3)
    second_slope = (d4 - de) / (be - b4)
    return first_slope >= second_slope


def summarize(
    endpoint_path: Path,
    exact_path: Path,
    *,
    codebook: str,
    targets: Sequence[float],
) -> dict[str, object]:
    records, endpoint_payload = _load_endpoint_records(endpoint_path, codebook=codebook)
    exact_bytes, exact_payload = _load_exact_bytes(exact_path)
    identities = sorted({(layer, expert) for layer, expert, _ in records})
    missing_exact = [identity for identity in identities if identity not in exact_bytes]
    if missing_exact:
        raise ValueError(f"exact tier lacks {len(missing_exact)} endpoint experts")

    k3 = {identity: records[(*identity, 3)] for identity in identities}
    k4 = {identity: records[(*identity, 4)] for identity in identities}
    exact = {
        identity: _exact_record(k3[identity], bytes_stored=exact_bytes[identity])
        for identity in identities
    }
    endpoints = {
        "K3_TSH": _aggregate(list(k3.values())),
        "K4_TSH": _aggregate(list(k4.values())),
        "N4_exact_MXFP4": _aggregate(list(exact.values())),
    }
    minimum = float(endpoints["K3_TSH"]["bpw_before_layer_deduplication"])
    k4_endpoint = float(endpoints["K4_TSH"]["bpw_before_layer_deduplication"])
    maximum = float(endpoints["N4_exact_MXFP4"]["bpw_before_layer_deduplication"])
    usable_targets = sorted(
        set(
            [
                minimum,
                k4_endpoint,
                maximum,
                *[target for target in targets if minimum <= target <= maximum],
            ]
        )
    )
    frontiers = {
        "K3_K4_TSH": [
            _binary_frontier_point(
                k3, k4, target_bpw=target, high_mode="K4_TSH"
            )
            for target in usable_targets
            if target <= k4_endpoint
        ],
        "K3_N4_exact": [
            _binary_frontier_point(
                k3, exact, target_bpw=target, high_mode="N4_exact_MXFP4"
            )
            for target in usable_targets
        ],
    }
    hull_counts = {}
    for metric in ("confirmation_sse", "validation_sse"):
        on_hull = [
            identity
            for identity in identities
            if _k4_on_per_expert_convex_hull(
                k3[identity], k4[identity], exact[identity], metric=metric
            )
        ]
        hull_counts[metric] = {
            "k4_on_convex_hull": len(on_hull),
            "experts": len(identities),
            "identities": [list(identity) for identity in on_hull],
        }
    return {
        "kind": "kquant_mxfp4_tsh_exact_tier_frontier",
        "schema_version": 1,
        "codebook": codebook,
        "selection_metric": "confirmation routed SSE gain per incremental byte",
        "evaluation_metric": "external-validation routed SSE",
        "endpoint_source": str(endpoint_path.resolve()),
        "exact_tier_source": str(exact_path.resolve()),
        "experts": len(identities),
        "endpoint_skipped": endpoint_payload.get("skipped", {}),
        "exact_codec_aggregate": exact_payload.get("aggregate"),
        "endpoints": endpoints,
        "k4_per_expert_convex_hull": hull_counts,
        "frontiers": frontiers,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("endpoint", type=Path)
    parser.add_argument("exact_tier", type=Path)
    parser.add_argument("--codebook", default="mul1-e4m3")
    parser.add_argument(
        "--targets",
        type=_parse_targets,
        default=(3.25, 3.5, 3.75, 4.0, 4.01),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = summarize(
        args.endpoint,
        args.exact_tier,
        codebook=args.codebook,
        targets=args.targets,
    )
    _atomic_write(args.output, payload)
    print(
        json.dumps(
            {
                "experts": payload["experts"],
                "output": str(args.output.resolve()),
                "endpoints": payload["endpoints"],
            },
            indent=1,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
