"""Summarize uniform-K MXFP4 endpoint studies into a rate frontier.

Rate choices are made from confirmation-fold routed SSE and evaluated on the
external validation fold.  A validation-ranked oracle is reported only to
measure selector regret; it is never used for the deployable selection.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} does not contain a JSON object")
    return value


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True))
    temporary.replace(path)


def _parse_fractions(value: str) -> tuple[float, ...]:
    try:
        fractions = tuple(float(part) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated fractions") from exc
    if not fractions or len(set(fractions)) != len(fractions):
        raise argparse.ArgumentTypeError("fractions must be nonempty and unique")
    if any(not 0.0 <= fraction <= 1.0 for fraction in fractions):
        raise argparse.ArgumentTypeError("fractions must lie in 0..1")
    return tuple(sorted(fractions))


def _numerical_contract(signature: Mapping[str, object]) -> dict[str, object]:
    ignored = {"compact", "experts", "layer"}
    return {key: value for key, value in signature.items() if key not in ignored}


def _candidate_record(
    candidate: Mapping[str, object], *, layer: int, expert: int
) -> dict[str, object]:
    matrix_evidence = candidate.get("matrix_evidence")
    routed = candidate.get("routed_function")
    if not isinstance(matrix_evidence, list) or not isinstance(routed, Mapping):
        raise TypeError("candidate lacks matrix or routed-function evidence")
    confirmation = routed.get("confirmation")
    validation = routed.get("validation")
    if not isinstance(confirmation, Mapping) or not isinstance(validation, Mapping):
        raise TypeError("candidate lacks confirmation or validation evidence")
    weights = 0
    bytes_stored = 0
    raw_sse = raw_energy = dense_sse = dense_energy = 0.0
    for matrix in matrix_evidence:
        if not isinstance(matrix, Mapping):
            raise TypeError("matrix evidence must be a mapping")
        shapes = matrix.get("tensor_shapes")
        if not isinstance(shapes, Mapping):
            raise TypeError("matrix evidence lacks tensor shapes")
        suh = shapes.get("suh")
        svh = shapes.get("svh")
        if not isinstance(suh, list) or not isinstance(svh, list):
            raise TypeError("matrix scale shapes must be lists")
        matrix_weights = 1
        for extent in (*suh, *svh):
            matrix_weights *= int(extent)
        weights += matrix_weights
        bytes_stored += int(matrix["bytes_before_layer_deduplication"])
        raw_sse += float(matrix["weight_squared_error"])
        raw_energy += float(matrix["weight_reference_energy"])
        dense_sse += float(matrix["captured_dense_h_numerator"])
        dense_energy += float(matrix["captured_dense_h_denominator"])
    return {
        "layer": layer,
        "expert": expert,
        "bits": int(candidate["bits"]),
        "codebook": str(candidate["codebook"]),
        "weights": weights,
        "bytes": bytes_stored,
        "raw_sse": raw_sse,
        "raw_energy": raw_energy,
        "dense_sse": dense_sse,
        "dense_energy": dense_energy,
        "confirmation_sse": float(confirmation["route_weighted_squared_error"]),
        "validation_sse": float(validation["route_weighted_squared_error"]),
        "validation_expert_energy": float(
            validation["route_weighted_reference_energy"]
        ),
        "validation_mixture_sse": float(
            validation["teacher_mixture_replacement_squared_error"]
        ),
        "validation_mixture_energy": float(validation["teacher_mixture_energy"]),
    }


def _aggregate(records: Sequence[Mapping[str, object]]) -> dict[str, float | int]:
    if not records:
        raise ValueError("cannot aggregate an empty record set")

    def total(key: str) -> float:
        return sum(float(record[key]) for record in records)

    weights = sum(int(record["weights"]) for record in records)
    bytes_stored = sum(int(record["bytes"]) for record in records)
    raw_energy = total("raw_energy")
    dense_energy = total("dense_energy")
    expert_energy = total("validation_expert_energy")
    mixture_energy = total("validation_mixture_energy")
    return {
        "experts": len(records),
        "weights": weights,
        "bytes_before_layer_deduplication": bytes_stored,
        "bpw_before_layer_deduplication": bytes_stored * 8 / weights,
        "weight_nmse": total("raw_sse") / raw_energy,
        "captured_dense_h_nmse": total("dense_sse") / dense_energy,
        "validation_routed_expert_nmse": total("validation_sse") / expert_energy,
        "validation_isolated_replacement_mixture_nmse": (
            total("validation_mixture_sse") / mixture_energy
        ),
        "validation_routed_sse": total("validation_sse"),
        "validation_routed_reference_energy": expert_energy,
        "validation_teacher_mixture_energy": mixture_energy,
    }


def _select_frontier(
    records: Mapping[tuple[int, int, int], Mapping[str, object]],
    *,
    fraction: float,
    selection_scope: str,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    identities = sorted({(layer, expert) for layer, expert, _ in records})
    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for identity in identities:
        group = identity[0] if selection_scope == "layer" else 0
        groups[group].append(identity)
    selected: list[tuple[int, int]] = []
    oracle: list[tuple[int, int]] = []
    for identities_in_group in groups.values():
        count = round(fraction * len(identities_in_group))

        def gain(identity: tuple[int, int], metric: str) -> float:
            layer, expert = identity
            return float(records[(layer, expert, 3)][metric]) - float(
                records[(layer, expert, 4)][metric]
            )

        selected.extend(
            sorted(
                identities_in_group,
                key=lambda identity: (gain(identity, "confirmation_sse"), identity),
                reverse=True,
            )[:count]
        )
        oracle.extend(
            sorted(
                identities_in_group,
                key=lambda identity: (gain(identity, "validation_sse"), identity),
                reverse=True,
            )[:count]
        )
    return sorted(selected), sorted(oracle)


def _materialize_selection(
    records: Mapping[tuple[int, int, int], Mapping[str, object]],
    selected: Sequence[tuple[int, int]],
) -> list[Mapping[str, object]]:
    selected_set = set(selected)
    identities = sorted({(layer, expert) for layer, expert, _ in records})
    return [
        records[(layer, expert, 4 if (layer, expert) in selected_set else 3)]
        for layer, expert in identities
    ]


def summarize(
    inputs: Sequence[Path],
    *,
    codebook: str,
    fractions: Sequence[float],
    selection_scope: str,
    mxfp4_bpw: float,
) -> dict:
    records: dict[tuple[int, int, int], dict[str, object]] = {}
    common_contract: dict[str, object] | None = None
    sources = []
    skipped: dict[str, object] = {}
    for path in inputs:
        payload = _read_json(path)
        if payload.get("kind") != "kquant_uniform_k_mxfp4_endpoint_study":
            raise ValueError(f"{path} is not a uniform-K endpoint study")
        if not bool(payload.get("complete")):
            raise ValueError(f"{path} is incomplete")
        signature = payload.get("signature")
        if not isinstance(signature, Mapping):
            raise TypeError(f"{path} has no signature")
        contract = _numerical_contract(signature)
        if common_contract is None:
            common_contract = contract
        elif contract != common_contract:
            raise ValueError(f"{path} uses a different numerical contract")
        layer = int(signature["layer"])
        sources.append(str(path.resolve()))
        raw_skipped = payload.get("skipped", {})
        if not isinstance(raw_skipped, Mapping):
            raise TypeError("skipped evidence must be a mapping")
        for expert, evidence in raw_skipped.items():
            skipped[f"{layer}:{expert}"] = evidence
        results = payload.get("results")
        if not isinstance(results, Mapping):
            raise TypeError("study results must be a mapping")
        for raw_expert, result in results.items():
            expert = int(raw_expert)
            if not isinstance(result, Mapping):
                raise TypeError("expert result must be a mapping")
            candidates = result.get("candidates")
            if not isinstance(candidates, Mapping):
                raise TypeError("expert result lacks candidates")
            for bits in (3, 4):
                key = f"{codebook}:K{bits}"
                candidate = candidates.get(key)
                if not isinstance(candidate, Mapping):
                    raise ValueError(f"expert {layer}:{expert} lacks {key}")
                identity = (layer, expert, bits)
                if identity in records:
                    raise ValueError(f"duplicate expert candidate {identity}")
                records[identity] = _candidate_record(
                    candidate, layer=layer, expert=expert
                )
    if common_contract is None or not records:
        raise ValueError("no endpoint records were loaded")
    identities = sorted({(layer, expert) for layer, expert, _ in records})
    endpoints = {
        f"K{bits}": _aggregate(
            [records[(layer, expert, bits)] for layer, expert in identities]
        )
        for bits in (3, 4)
    }
    k3_sse = float(endpoints["K3"]["validation_routed_sse"])
    k4_sse = float(endpoints["K4"]["validation_routed_sse"])
    frontier = []
    for fraction in fractions:
        selected, oracle = _select_frontier(
            records, fraction=fraction, selection_scope=selection_scope
        )
        evidence = _aggregate(_materialize_selection(records, selected))
        oracle_evidence = _aggregate(_materialize_selection(records, oracle))
        sse = float(evidence["validation_routed_sse"])
        oracle_sse = float(oracle_evidence["validation_routed_sse"])
        selected_set = set(selected)
        oracle_set = set(oracle)
        evidence.update(
            {
                "requested_k4_fraction": fraction,
                "realized_k4_fraction": len(selected) / len(identities),
                "selected_k4_experts": [list(value) for value in selected],
                "validation_oracle_k4_experts": [list(value) for value in oracle],
                "selector_oracle_agreement": len(selected_set & oracle_set),
                "validation_oracle_routed_expert_nmse": oracle_evidence[
                    "validation_routed_expert_nmse"
                ],
                "validation_selector_regret_sse": sse - oracle_sse,
                "validation_sse_ratio_vs_uniform_k3": sse / k3_sse,
                "fraction_of_k3_to_k4_gap_remaining": (
                    (sse - k4_sse) / (k3_sse - k4_sse)
                ),
                "nominal_savings_vs_mxfp4": 1.0
                - float(evidence["bpw_before_layer_deduplication"]) / mxfp4_bpw,
            }
        )
        frontier.append(evidence)
    return {
        "kind": "kquant_uniform_k_mxfp4_rate_frontier_summary",
        "schema_version": 1,
        "codebook": codebook,
        "selection_metric": "confirmation absolute route-weighted SSE reduction",
        "evaluation_metric": "external-validation route-weighted SSE",
        "selection_scope": selection_scope,
        "mxfp4_bpw_reference": mxfp4_bpw,
        "sources": sources,
        "numerical_contract": common_contract,
        "experts": len(identities),
        "skipped": skipped,
        "endpoints": endpoints,
        "frontier": frontier,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--codebook", default="mul1-e4m3")
    parser.add_argument("--fractions", type=_parse_fractions, default=(0.0, 0.5, 0.75, 1.0))
    parser.add_argument("--selection-scope", choices=("layer", "global"), default="layer")
    parser.add_argument("--mxfp4-bpw", type=float, default=4.25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mxfp4_bpw <= 0:
        parser.error("--mxfp4-bpw must be positive")
    return args


def main() -> None:
    args = parse_args()
    payload = summarize(
        args.inputs,
        codebook=args.codebook,
        fractions=args.fractions,
        selection_scope=args.selection_scope,
        mxfp4_bpw=args.mxfp4_bpw,
    )
    _atomic_write(args.output, payload)
    print(json.dumps({
        "experts": payload["experts"],
        "skipped": len(payload["skipped"]),
        "output": str(args.output.resolve()),
    }, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
