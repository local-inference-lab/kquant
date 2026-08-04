"""Summarize a complete TP12 ``w2`` ``R_r`` rate ladder.

Candidate reconstruction is performed elsewhere.  This script checks that
every input is an equal-byte, reference-closed study, chooses modes using only
the training corpus, and evaluates the resulting static expert decisions on
the document-disjoint validation corpus.  Arithmetic sums of routed,
gate-square-weighted output SSE are the primary metric.

The split-confirmed selector is deliberately conservative: it selects a mode
on one deterministic subset of training documents and admits it only when an
independent training-document subset has a positive paired-bootstrap lower
bound relative to R0.  The untouched validation corpus is never used to make
a mode decision.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from scripts.summarize_retained_w2_rate_transfer import (
    RecordKey,
    _bootstrap_improvement,
    _candidate,
    _document_reference,
    _document_sse,
    _metric,
    _quantile,
    _read_json,
    _seed_for,
)


FULL_LADDER = tuple(range(13))


def _parse_alphabets(value: str) -> tuple[tuple[int, ...], ...]:
    alphabets = []
    for group in value.split(";"):
        try:
            alphabet = tuple(sorted({int(part) for part in group.split(",")}))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "mode alphabets must be semicolon-separated comma lists"
            ) from exc
        if not alphabet or 0 not in alphabet or any(r not in FULL_LADDER for r in alphabet):
            raise argparse.ArgumentTypeError(
                "every mode alphabet must include R0 and contain only R0..R12"
            )
        alphabets.append(alphabet)
    if len(set(alphabets)) != len(alphabets):
        raise argparse.ArgumentTypeError("mode alphabets must be unique")
    return tuple(alphabets)


def _alphabet_name(alphabet: Sequence[int]) -> str:
    return "_".join(f"r{r}" for r in alphabet)


def _sum_for_documents(values: dict[str, float], documents: Iterable[str]) -> float:
    return sum(values.get(document, 0.0) for document in documents)


def _supported_documents(metric: dict | None, documents: Iterable[str]) -> list[str]:
    if metric is None:
        return []
    available = set(_document_sse(metric))
    return [document for document in documents if document in available]


def _argmin_for_result(
    result: dict,
    allowed_r: Sequence[int],
    *,
    split: str,
    documents: Sequence[str] | None = None,
) -> int:
    scores = []
    for r in allowed_r:
        metric = _metric(_candidate(result, r), split)
        if metric is None:
            continue
        score = (
            float(metric["weighted_squared_error"])
            if documents is None
            else _sum_for_documents(_document_sse(metric), documents)
        )
        scores.append((score, r))
    if not scores:
        return 0
    # Prefer less rate transfer on an exact tie.
    return min(scores)[1]


def _argmin_map(
    records: dict[RecordKey, dict],
    allowed_r: Sequence[int],
    *,
    split: str = "train",
    documents: Sequence[str] | None = None,
) -> dict[RecordKey, int]:
    return {
        key: _argmin_for_result(
            result,
            allowed_r,
            split=split,
            documents=documents,
        )
        for key, result in records.items()
    }


def _best_training_alphabets(
    records: dict[RecordKey, dict], *, maximum_size: int
) -> dict[int, tuple[tuple[int, ...], dict[RecordKey, int], float]]:
    """Return the minimum-training-SSE alphabet at every requested size.

    R0 is mandatory.  The search is only 2^12 subsets for the complete ladder,
    so exhaustive enumeration is both simpler and more trustworthy than a
    greedy mode-table heuristic.  The untouched validation fold is not read.
    """

    if not 1 <= maximum_size <= len(FULL_LADDER):
        raise ValueError("maximum alphabet size must be in 1..13")
    errors: dict[RecordKey, dict[int, float]] = {}
    for key, result in records.items():
        errors[key] = {
            r: float(_metric(_candidate(result, r), "train")["weighted_squared_error"])
            for r in FULL_LADDER
        }
    winners: dict[int, tuple[tuple[int, ...], dict[RecordKey, int], float]] = {}
    for size in range(1, maximum_size + 1):
        best: tuple[float, tuple[int, ...], dict[RecordKey, int]] | None = None
        for additions in itertools.combinations(FULL_LADDER[1:], size - 1):
            alphabet = (0, *additions)
            choices = {
                key: min((per_r[r], r) for r in alphabet)[1]
                for key, per_r in errors.items()
            }
            total = sum(errors[key][r] for key, r in choices.items())
            proposal = (total, alphabet, choices)
            if best is None or proposal[:2] < best[:2]:
                best = proposal
        assert best is not None
        winners[size] = (best[1], best[2], best[0])
    return winners


def _document_partition(
    documents: Sequence[str], *, modulus: int, confirmation_index: int
) -> tuple[list[str], list[str]]:
    fit = []
    confirmation = []
    for document in documents:
        digest = hashlib.blake2b(document.encode(), digest_size=8).digest()
        bucket = int.from_bytes(digest, "little") % modulus
        (confirmation if bucket == confirmation_index else fit).append(document)
    if not fit or not confirmation:
        raise ValueError("training-document partition produced an empty side")
    return fit, confirmation


def _bootstrap_screened_map(
    records: dict[RecordKey, dict],
    training_documents: Sequence[str],
    allowed_r: Sequence[int],
    *,
    replicates: int,
    seed: int,
    min_documents: int,
    minimum_improvement: float,
) -> tuple[dict[RecordKey, int], dict[str, dict]]:
    proposed = _argmin_map(records, allowed_r)
    selected: dict[RecordKey, int] = {}
    evidence: dict[str, dict] = {}
    for (layer, expert), result in records.items():
        proposed_r = proposed[layer, expert]
        r0 = _metric(_candidate(result, 0), "train")
        candidate = _metric(_candidate(result, proposed_r), "train")
        support = _supported_documents(r0, training_documents)
        if proposed_r == 0 or r0 is None or candidate is None:
            selected[layer, expert] = 0
            evidence[f"{layer}/{expert}"] = {
                "proposed_r": proposed_r,
                "selected_r": 0,
                "training_documents": len(support),
                "reason": "R0 is the training argmin or metrics are unavailable",
            }
            continue
        estimate = _bootstrap_improvement(
            _document_sse(r0),
            _document_sse(candidate),
            support,
            replicates=replicates,
            seed=_seed_for(seed, layer, expert),
        )
        lower = estimate["relative_improvement_ci95"][0]
        accepted = (
            len(support) >= min_documents
            and lower is not None
            and lower > minimum_improvement
        )
        selected[layer, expert] = proposed_r if accepted else 0
        evidence[f"{layer}/{expert}"] = {
            "proposed_r": proposed_r,
            "selected_r": selected[layer, expert],
            "training_documents": len(support),
            **estimate,
        }
    return selected, evidence


def _split_confirmed_map(
    records: dict[RecordKey, dict],
    fit_documents: Sequence[str],
    confirmation_documents: Sequence[str],
    allowed_r: Sequence[int],
    *,
    replicates: int,
    seed: int,
    min_fit_documents: int,
    min_confirmation_documents: int,
    minimum_improvement: float,
) -> tuple[dict[RecordKey, int], dict[str, dict]]:
    selected: dict[RecordKey, int] = {}
    evidence: dict[str, dict] = {}
    for (layer, expert), result in records.items():
        r0 = _metric(_candidate(result, 0), "train")
        fit_support = _supported_documents(r0, fit_documents)
        confirmation_support = _supported_documents(r0, confirmation_documents)
        if len(fit_support) < min_fit_documents:
            selected[layer, expert] = 0
            evidence[f"{layer}/{expert}"] = {
                "proposed_r": 0,
                "selected_r": 0,
                "fit_documents": len(fit_support),
                "confirmation_documents": len(confirmation_support),
                "reason": "insufficient fit-document support",
            }
            continue
        proposed_r = _argmin_for_result(
            result,
            allowed_r,
            split="train",
            documents=fit_support,
        )
        if proposed_r == 0 or r0 is None:
            selected[layer, expert] = 0
            evidence[f"{layer}/{expert}"] = {
                "proposed_r": proposed_r,
                "selected_r": 0,
                "fit_documents": len(fit_support),
                "confirmation_documents": len(confirmation_support),
                "reason": "R0 is the fit-document argmin",
            }
            continue
        candidate = _metric(_candidate(result, proposed_r), "train")
        assert candidate is not None
        estimate = _bootstrap_improvement(
            _document_sse(r0),
            _document_sse(candidate),
            confirmation_support,
            replicates=replicates,
            seed=_seed_for(seed, layer, expert),
        ) if confirmation_support else {
            "baseline_weighted_squared_error": 0.0,
            "candidate_weighted_squared_error": 0.0,
            "candidate_to_baseline_sse_ratio": None,
            "relative_improvement": None,
            "bootstrap_replicates_requested": replicates,
            "bootstrap_replicates_valid": 0,
            "relative_improvement_ci95": [None, None],
        }
        lower = estimate["relative_improvement_ci95"][0]
        accepted = (
            len(confirmation_support) >= min_confirmation_documents
            and lower is not None
            and lower > minimum_improvement
        )
        selected[layer, expert] = proposed_r if accepted else 0
        evidence[f"{layer}/{expert}"] = {
            "proposed_r": proposed_r,
            "selected_r": selected[layer, expert],
            "fit_documents": len(fit_support),
            "confirmation_documents": len(confirmation_support),
            **estimate,
        }
    return selected, evidence


def _aggregate_strategy(
    records: dict[RecordKey, dict],
    documents: list[str],
    selector: Callable[[RecordKey], int],
    *,
    replicates: int,
    seed: int,
) -> dict:
    baseline_sse = {document: 0.0 for document in documents}
    candidate_sse = {document: 0.0 for document in documents}
    reference = {document: 0.0 for document in documents}
    selected = Counter()
    supported = 0
    for key, result in records.items():
        selected_r = selector(key)
        selected[selected_r] += 1
        r0_metric = _metric(_candidate(result, 0), "validation")
        selected_metric = _metric(_candidate(result, selected_r), "validation")
        if r0_metric is None or selected_metric is None:
            continue
        supported += 1
        r0_sse = _document_sse(r0_metric)
        chosen_sse = _document_sse(selected_metric)
        r0_reference = _document_reference(r0_metric)
        chosen_reference = _document_reference(selected_metric)
        for document in documents:
            if not math.isclose(
                r0_reference.get(document, 0.0),
                chosen_reference.get(document, 0.0),
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise ValueError(f"candidate reference energies differ for {key}")
            baseline_sse[document] += r0_sse.get(document, 0.0)
            candidate_sse[document] += chosen_sse.get(document, 0.0)
            reference[document] += r0_reference.get(document, 0.0)
    estimate = _bootstrap_improvement(
        baseline_sse,
        candidate_sse,
        documents,
        replicates=replicates,
        seed=seed,
    )
    reference_total = sum(reference.values())
    estimate.update(
        {
            "experts": len(records),
            "experts_with_validation_rows": supported,
            "selected_mode_histogram": {
                f"R{r}": selected.get(r, 0) for r in FULL_LADDER
            },
            "weighted_reference_energy": reference_total,
            "baseline_weighted_output_nmse": (
                estimate["baseline_weighted_squared_error"] / reference_total
                if reference_total > 0
                else None
            ),
            "candidate_weighted_output_nmse": (
                estimate["candidate_weighted_squared_error"] / reference_total
                if reference_total > 0
                else None
            ),
        }
    )
    return estimate


def _support_summary(records: dict[RecordKey, dict]) -> dict:
    train_rows = []
    train_documents = []
    validation_rows = []
    validation_documents = []
    validation_argmin = Counter()
    ratios: dict[int, list[float]] = {r: [] for r in FULL_LADDER}
    for result in records.values():
        train = _metric(_candidate(result, 0), "train")
        validation = _metric(_candidate(result, 0), "validation")
        if train is not None:
            train_rows.append(int(train["rows"]))
            train_documents.append(int(train["documents"]))
        if validation is None:
            continue
        validation_rows.append(int(validation["rows"]))
        validation_documents.append(int(validation["documents"]))
        baseline = float(validation["weighted_squared_error"])
        validation_argmin[
            _argmin_for_result(result, FULL_LADDER, split="validation")
        ] += 1
        if baseline > 0:
            for r in FULL_LADDER:
                metric = _metric(_candidate(result, r), "validation")
                assert metric is not None
                ratios[r].append(float(metric["weighted_squared_error"]) / baseline)

    def distribution(values: Sequence[float]) -> dict:
        return {
            "count": len(values),
            "minimum": min(values) if values else None,
            "median": _quantile(values, 0.5),
            "p90": _quantile(values, 0.9),
            "p99": _quantile(values, 0.99),
            "maximum": max(values) if values else None,
        }

    return {
        "training_rows_per_expert": distribution(train_rows),
        "training_documents_per_expert": distribution(train_documents),
        "validation_rows_per_supported_expert": distribution(validation_rows),
        "validation_documents_per_supported_expert": distribution(validation_documents),
        "validation_per_expert_argmin_histogram_oracle_only": {
            f"R{r}": validation_argmin.get(r, 0) for r in FULL_LADDER
        },
        "fixed_r_to_r0_validation_sse_ratio_per_expert": {
            f"R{r}": distribution(ratios[r]) for r in FULL_LADDER
        },
    }


def _validate_and_collect(
    paths: Sequence[Path],
) -> tuple[dict[RecordKey, dict], Path, Path, list[int], int, int, dict]:
    records: dict[RecordKey, dict] = {}
    training_report_path: Path | None = None
    validation_report_path: Path | None = None
    layers = set()
    skipped = 0
    closures = 0
    common_contract = None
    study_contract = None
    for path in paths:
        payload = _read_json(path)
        if payload.get("kind") != "kquant_retained_w2_rate_transfer_study":
            raise ValueError(f"{path} has the wrong study kind")
        if payload.get("schema_version") != 2 or not payload.get("complete"):
            raise ValueError(f"{path} is not a complete schema-v2 study")
        signature = payload["signature"]
        if signature.get("metrics_schema") != 2:
            raise ValueError(f"{path} has the wrong metric contract")
        if tuple(signature.get("r_values", ())) != FULL_LADDER:
            raise ValueError(f"{path} does not contain the complete R0..R12 ladder")
        if signature.get("tp_size") != 12:
            raise ValueError(f"{path} is not a TP12 study")
        provenance = signature["provenance"]
        candidate_training_report = Path(provenance["training_report"])
        candidate_validation_report = Path(provenance["validation_report"])
        contract = (
            signature["checkpoint"],
            signature.get("weight_source", "interim-retained"),
            signature.get("source_checkpoint", signature["checkpoint"]),
            signature["capture"],
            signature["validation_capture"],
            signature["hessians"],
            signature["layout"],
            candidate_training_report,
            candidate_validation_report,
        )
        if common_contract not in (None, contract):
            raise ValueError("input studies use different encoder or corpus contracts")
        common_contract = contract
        study_contract = {
            "resident_teacher_checkpoint": signature["checkpoint"],
            "weight_source": signature.get("weight_source", "interim-retained"),
            "source_checkpoint": signature.get(
                "source_checkpoint", signature["checkpoint"]
            ),
            "capture": signature["capture"],
            "validation_capture": signature["validation_capture"],
            "hessians": signature["hessians"],
            "layout": signature["layout"],
            "tp_size": signature["tp_size"],
        }
        training_report_path = candidate_training_report
        validation_report_path = candidate_validation_report
        layer = int(signature["layer"])
        layers.add(layer)
        skipped += len(payload.get("skipped", {}))
        for expert_text, result in payload["results"].items():
            key = (layer, int(expert_text))
            if key in records:
                raise ValueError(f"duplicate layer/expert result {key}")
            byte_contract = None
            for r in FULL_LADDER:
                candidate = _candidate(result, r)
                if not candidate["reference_edge_roundtrip"] or not candidate[
                    "reference_state_roundtrip"
                ]:
                    raise ValueError(f"failed reference closure for {key} R{r}")
                candidate_contract = (
                    int(candidate["index_bits"]),
                    int(candidate["scale_bits"]),
                    int(candidate["provisional_expert_format_bits"]),
                )
                if byte_contract not in (None, candidate_contract):
                    raise ValueError(f"R modes do not have equal bytes for {key}")
                byte_contract = candidate_contract
                closures += 1
            records[key] = result
    if (
        training_report_path is None
        or validation_report_path is None
        or study_contract is None
    ):
        raise ValueError("no study inputs")
    return (
        records,
        training_report_path,
        validation_report_path,
        sorted(layers),
        skipped,
        closures,
        study_contract,
    )


def _validate_population_manifest(
    path: Path,
    records: dict[RecordKey, dict],
    *,
    training_report_path: Path,
    validation_report_path: Path,
) -> dict:
    payload = _read_json(path)
    if payload.get("kind") != "kquant_keep_tier_blind_w2_source_sample":
        raise ValueError("population manifest has the wrong kind")
    if payload.get("schema_version") != 1:
        raise ValueError("population manifest has the wrong schema")
    if payload.get("selection_uses_keep_tier") is not False:
        raise ValueError("population manifest must explicitly be keep-tier-blind")

    provenance = payload.get("capture_provenance", {})
    report_contract = {
        "training": (
            Path(str(provenance.get("training_report", ""))).resolve(),
            training_report_path.resolve(),
        ),
        "validation": (
            Path(str(provenance.get("validation_report", ""))).resolve(),
            validation_report_path.resolve(),
        ),
    }
    for name, (actual, expected) in report_contract.items():
        if actual != expected:
            raise ValueError(
                f"population manifest {name} report {actual} != {expected}"
            )

    selected: set[RecordKey] = set()
    retained = 0
    compressed = 0
    for layer_text, layer_payload in payload.get("layers", {}).items():
        layer = int(layer_text)
        experts = [int(expert) for expert in layer_payload["selected_experts"]]
        if len(experts) != len(set(experts)):
            raise ValueError(f"population manifest layer {layer} repeats experts")
        selected.update((layer, expert) for expert in experts)
        retained += int(layer_payload.get("selected_retained", 0))
        compressed += int(layer_payload.get("selected_compressed_in_interim", 0))
    actual = set(records)
    if selected != actual:
        missing = sorted(selected - actual)[:8]
        unexpected = sorted(actual - selected)[:8]
        raise ValueError(
            "population manifest does not match study results: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if retained + compressed != len(selected):
        raise ValueError("population manifest tier audit does not cover its sample")
    return {
        "manifest": str(path.resolve()),
        "kind": payload["kind"],
        "selection_uses_keep_tier": False,
        "criteria": payload.get("criteria"),
        "selected_experts": len(selected),
        "selected_retained_in_interim": retained,
        "selected_compressed_in_interim": compressed,
    }


def run(args: argparse.Namespace) -> dict:
    (
        records,
        training_report_path,
        validation_report_path,
        layers,
        skipped,
        closures,
        study_contract,
    ) = _validate_and_collect(args.input)
    training_documents = [
        str(document["document_hash"])
        for document in _read_json(training_report_path)["documents"]
    ]
    validation_documents = [
        str(document["document_hash"])
        for document in _read_json(validation_report_path)["documents"]
    ]
    fit_documents, confirmation_documents = _document_partition(
        training_documents,
        modulus=args.confirmation_modulus,
        confirmation_index=args.confirmation_index,
    )
    if args.population_manifest is not None:
        population = _validate_population_manifest(
            args.population_manifest,
            records,
            training_report_path=training_report_path,
            validation_report_path=validation_report_path,
        )
    else:
        population = {
            "manifest": None,
            "kind": "unspecified",
            "selection_uses_keep_tier": (
                True
                if study_contract["weight_source"] == "interim-retained"
                else None
            ),
            "selected_experts": len(records),
        }

    full_training_argmin = _argmin_map(records, FULL_LADDER)
    screened, screened_evidence = _bootstrap_screened_map(
        records,
        training_documents,
        FULL_LADDER,
        replicates=args.selection_bootstrap_replicates,
        seed=args.seed,
        min_documents=args.min_training_documents,
        minimum_improvement=args.minimum_training_improvement,
    )
    split_confirmed, split_evidence = _split_confirmed_map(
        records,
        fit_documents,
        confirmation_documents,
        FULL_LADDER,
        replicates=args.selection_bootstrap_replicates,
        seed=args.seed + 1_000_000,
        min_fit_documents=args.min_fit_documents,
        min_confirmation_documents=args.min_confirmation_documents,
        minimum_improvement=args.minimum_training_improvement,
    )

    def summarize(selected_records: dict[RecordKey, dict], seed_offset: int) -> dict:
        fixed = {
            f"R{r}": _aggregate_strategy(
                selected_records,
                validation_documents,
                lambda _key, r=r: r,
                replicates=args.evaluation_bootstrap_replicates,
                seed=args.seed + seed_offset + r,
            )
            for r in FULL_LADDER
        }
        alphabets = {}
        for index, alphabet in enumerate(args.mode_alphabets):
            choices = _argmin_map(selected_records, alphabet)
            alphabets[_alphabet_name(alphabet)] = {
                "allowed_r": list(alphabet),
                "training_argmin": _aggregate_strategy(
                    selected_records,
                    validation_documents,
                    lambda key, choices=choices: choices[key],
                    replicates=args.evaluation_bootstrap_replicates,
                    seed=args.seed + seed_offset + 100 + index,
                ),
            }
        searched_alphabets = {}
        for size, (alphabet, choices, training_sse) in _best_training_alphabets(
            selected_records,
            maximum_size=args.maximum_searched_alphabet_size,
        ).items():
            searched_alphabets[str(size)] = {
                "allowed_r": list(alphabet),
                "training_weighted_squared_error": training_sse,
                "training_argmin": _aggregate_strategy(
                    selected_records,
                    validation_documents,
                    lambda key, choices=choices: choices[key],
                    replicates=args.evaluation_bootstrap_replicates,
                    seed=args.seed + seed_offset + 300 + size,
                ),
            }
        return {
            "fixed_r": fixed,
            "mode_alphabets": alphabets,
            "training_searched_alphabets_by_size": searched_alphabets,
            "full_ladder_training_argmin": _aggregate_strategy(
                selected_records,
                validation_documents,
                lambda key: full_training_argmin[key],
                replicates=args.evaluation_bootstrap_replicates,
                seed=args.seed + seed_offset + 200,
            ),
            "full_ladder_training_bootstrap_screened": _aggregate_strategy(
                selected_records,
                validation_documents,
                lambda key: screened[key],
                replicates=args.evaluation_bootstrap_replicates,
                seed=args.seed + seed_offset + 201,
            ),
            "full_ladder_split_confirmed": _aggregate_strategy(
                selected_records,
                validation_documents,
                lambda key: split_confirmed[key],
                replicates=args.evaluation_bootstrap_replicates,
                seed=args.seed + seed_offset + 202,
            ),
        }

    per_layer = {}
    for offset, layer in enumerate(layers, start=1):
        layer_records = {
            key: result for key, result in records.items() if key[0] == layer
        }
        per_layer[str(layer)] = {
            "support": _support_summary(layer_records),
            "strategies": summarize(layer_records, 1000 * offset),
        }

    limitations = [
        "mid-route sampling leaves sparse per-expert validation support",
        "isolated gate-square-weighted expert self-terms omit routed-mixture cross terms",
        "the split-confirmed selector has low power for sparsely routed experts",
        "this intermediate capture is not the planned production-scale calibration",
    ]
    if population["selection_uses_keep_tier"] is True:
        limitations.insert(0, "retained experts are a keep-tier-biased subset")
    elif population["selection_uses_keep_tier"] is False:
        limitations.insert(
            0,
            "the keep-tier-blind stratified sample covers only representative layers",
        )
    else:
        limitations.insert(0, "population selection provenance was not supplied")

    result = {
        "kind": "kquant_w2_rate_ladder_summary",
        "schema_version": 2,
        "inputs": [str(path.resolve()) for path in args.input],
        "study_contract": study_contract,
        "population": population,
        "layers": layers,
        "experts": len(records),
        "experts_skipped_without_training_rows": skipped,
        "candidate_closures_checked": closures,
        "equal_byte_contract_checked_per_expert": True,
        "training_documents": len(training_documents),
        "validation_documents": len(validation_documents),
        "selection_contract": {
            "validation_used_for_selection": False,
            "full_training_argmin": "exploratory 13-way training argmin",
            "bootstrap_screened": (
                "same-training-data screen; useful but optimistic after 13-way selection"
            ),
            "split_confirmed": {
                "fit_documents": len(fit_documents),
                "confirmation_documents": len(confirmation_documents),
                "confirmation_modulus": args.confirmation_modulus,
                "confirmation_index": args.confirmation_index,
                "min_fit_documents": args.min_fit_documents,
                "min_confirmation_documents": args.min_confirmation_documents,
                "minimum_improvement": args.minimum_training_improvement,
            },
            "selection_bootstrap_replicates": args.selection_bootstrap_replicates,
            "alphabet_search": (
                "exhaustive training-SSE minimization with mandatory R0; "
                "validation is evaluation-only"
            ),
        },
        "limitations": limitations,
        "support": _support_summary(records),
        "strategies": summarize(records, 10_000),
        "per_layer": per_layer,
        "training_argmin_mode_map": {
            f"{layer}/{expert}": r
            for (layer, expert), r in sorted(full_training_argmin.items())
        },
        "training_bootstrap_screened_evidence": screened_evidence,
        "split_confirmed_evidence": split_evidence,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=1, sort_keys=True))
    temporary.replace(args.output)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--population-manifest",
        type=Path,
        help=(
            "optional keep-tier-blind sample manifest; its exact layer/expert "
            "set and corpus provenance are validated against the studies"
        ),
    )
    parser.add_argument(
        "--mode-alphabets",
        type=_parse_alphabets,
        default=_parse_alphabets(
            "0,6;0,3,6;0,1,3,6;0,1,2,3,4,5,6,7,8,9,10,11,12"
        ),
    )
    parser.add_argument("--min-training-documents", type=int, default=8)
    parser.add_argument("--min-fit-documents", type=int, default=6)
    parser.add_argument("--min-confirmation-documents", type=int, default=4)
    parser.add_argument("--maximum-searched-alphabet-size", type=int, default=6)
    parser.add_argument("--minimum-training-improvement", type=float, default=0.0)
    parser.add_argument("--confirmation-modulus", type=int, default=4)
    parser.add_argument("--confirmation-index", type=int, default=0)
    parser.add_argument("--selection-bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--evaluation-bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()
    if min(
        args.min_training_documents,
        args.min_fit_documents,
        args.min_confirmation_documents,
    ) < 1:
        parser.error("document-support minima must be positive")
    if args.confirmation_modulus < 2:
        parser.error("--confirmation-modulus must be at least two")
    if not 0 <= args.confirmation_index < args.confirmation_modulus:
        parser.error("--confirmation-index must be a valid bucket")
    if args.selection_bootstrap_replicates < 100:
        parser.error("--selection-bootstrap-replicates must be at least 100")
    if args.evaluation_bootstrap_replicates < 100:
        parser.error("--evaluation-bootstrap-replicates must be at least 100")
    if not 1 <= args.maximum_searched_alphabet_size <= len(FULL_LADDER):
        parser.error("--maximum-searched-alphabet-size must be in 1..13")
    return args


def main() -> None:
    args = parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "experts": result["experts"],
                "fixed_r": {
                    r: {
                        "relative_improvement": value["relative_improvement"],
                        "relative_improvement_ci95": value[
                            "relative_improvement_ci95"
                        ],
                    }
                    for r, value in result["strategies"]["fixed_r"].items()
                },
                "full_ladder_training_argmin": result["strategies"][
                    "full_ladder_training_argmin"
                ],
                "full_ladder_split_confirmed": result["strategies"][
                    "full_ladder_split_confirmed"
                ],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
