import json

import pytest

from scripts.summarize_retained_w2_rate_ladder import (
    _aggregate_strategy,
    _argmin_map,
    _bootstrap_screened_map,
    _best_training_alphabets,
    _document_partition,
    _split_confirmed_map,
    _validate_population_manifest,
)


def _metric(errors):
    contributions = [
        {
            "document_hash": document,
            "request_step": index,
            "rows": 1,
            "weight_sum": 1.0,
            "weighted_squared_error": error,
            "weighted_reference_energy": 100.0,
        }
        for index, (document, error) in enumerate(errors, start=1)
    ]
    return {
        "rows": len(errors),
        "documents": len(errors),
        "excluded_unmapped_rows": 0,
        "weight_sum": float(len(errors)),
        "weighted_squared_error": sum(error for _, error in errors),
        "weighted_reference_energy": 100.0 * len(errors),
        "weighted_output_nmse": sum(error for _, error in errors) / (100 * len(errors)),
        "request_contributions": contributions,
    }


def _result(documents):
    candidates = []
    for r in range(13):
        # R2 is stably best; all other transferred modes are worse than R0.
        error = 5.0 if r == 2 else (10.0 if r == 0 else 15.0 + r)
        metric = _metric([(document, error) for document in documents])
        candidates.append(
            {
                "r": r,
                "captured_activation": {"train": metric, "validation": metric},
            }
        )
    return {"candidates": candidates}


def test_ladder_argmin_screen_and_aggregate_choose_intermediate_mode():
    documents = [f"doc-{index}" for index in range(12)]
    records = {(40, 7): _result(documents)}
    argmin = _argmin_map(records, tuple(range(13)))
    assert argmin[(40, 7)] == 2

    screened, evidence = _bootstrap_screened_map(
        records,
        documents,
        tuple(range(13)),
        replicates=200,
        seed=123,
        min_documents=8,
        minimum_improvement=0.0,
    )
    assert screened[(40, 7)] == 2
    assert evidence["40/7"]["relative_improvement_ci95"] == [0.5, 0.5]

    summary = _aggregate_strategy(
        records,
        documents,
        lambda key: screened[key],
        replicates=200,
        seed=456,
    )
    assert summary["relative_improvement"] == 0.5
    assert summary["selected_mode_histogram"]["R2"] == 1


def test_split_confirmation_uses_independent_document_bucket():
    documents = [f"doc-{index}" for index in range(80)]
    fit, confirmation = _document_partition(
        documents, modulus=4, confirmation_index=0
    )
    assert set(fit).isdisjoint(confirmation)
    assert sorted(fit + confirmation) == sorted(documents)

    records = {(1, 3): _result(documents)}
    selected, evidence = _split_confirmed_map(
        records,
        fit,
        confirmation,
        tuple(range(13)),
        replicates=200,
        seed=789,
        min_fit_documents=6,
        min_confirmation_documents=4,
        minimum_improvement=0.0,
    )
    assert selected[(1, 3)] == 2
    assert evidence["1/3"]["fit_documents"] == len(fit)
    assert evidence["1/3"]["confirmation_documents"] == len(confirmation)


def test_training_only_alphabet_search_recovers_intermediate_mode():
    documents = [f"doc-{index}" for index in range(12)]
    records = {(1, 1): _result(documents), (24, 2): _result(documents)}
    searched = _best_training_alphabets(records, maximum_size=3)
    assert searched[1][0] == (0,)
    assert searched[2][0] == (0, 2)
    # Extra modes tie once R2 is available, so the lexicographically smallest
    # equal-objective table is retained.
    assert searched[3][0] == (0, 1, 2)


def test_keep_tier_blind_population_manifest_must_match_exact_sample(tmp_path):
    training_report = tmp_path / "training.json"
    validation_report = tmp_path / "validation.json"
    training_report.write_text("{}")
    validation_report.write_text("{}")
    manifest = tmp_path / "sample.json"
    manifest.write_text(
        json.dumps(
            {
                "kind": "kquant_keep_tier_blind_w2_source_sample",
                "schema_version": 1,
                "selection_uses_keep_tier": False,
                "criteria": {"ranking": "natural traffic"},
                "capture_provenance": {
                    "training_report": str(training_report),
                    "validation_report": str(validation_report),
                },
                "layers": {
                    "1": {
                        "selected_experts": [3, 8],
                        "selected_retained": 1,
                        "selected_compressed_in_interim": 1,
                    }
                },
            }
        )
    )
    records = {(1, 3): {}, (1, 8): {}}
    result = _validate_population_manifest(
        manifest,
        records,
        training_report_path=training_report,
        validation_report_path=validation_report,
    )
    assert result["selection_uses_keep_tier"] is False
    assert result["selected_retained_in_interim"] == 1

    records[(1, 9)] = {}
    with pytest.raises(ValueError, match="does not match"):
        _validate_population_manifest(
            manifest,
            records,
            training_report_path=training_report,
            validation_report_path=validation_report,
        )
