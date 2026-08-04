from scripts.summarize_retained_w2_rate_transfer import (
    _aggregate_strategy,
    _mode_maps,
)


def _metric(values):
    contributions = [
        {
            "document_hash": document,
            "request_step": index,
            "rows": 1,
            "weight_sum": 1.0,
            "weighted_squared_error": error,
            "weighted_reference_energy": reference,
        }
        for index, (document, error, reference) in enumerate(values, start=1)
    ]
    return {
        "rows": len(values),
        "documents": len(values),
        "excluded_unmapped_rows": 0,
        "weight_sum": float(len(values)),
        "weighted_squared_error": sum(value[1] for value in values),
        "weighted_reference_energy": sum(value[2] for value in values),
        "weighted_output_nmse": (
            sum(value[1] for value in values) / sum(value[2] for value in values)
        ),
        "request_contributions": contributions,
    }


def _result():
    return {
        "candidates": [
            {
                "r": 0,
                "captured_activation": {
                    "train": _metric((("a", 10.0, 100.0), ("b", 20.0, 100.0))),
                    "validation": _metric(
                        (("x", 12.0, 80.0), ("y", 18.0, 120.0))
                    ),
                },
            },
            {
                "r": 6,
                "captured_activation": {
                    "train": _metric((("a", 5.0, 100.0), ("b", 10.0, 100.0))),
                    "validation": _metric(
                        (("x", 6.0, 80.0), ("y", 9.0, 120.0))
                    ),
                },
            },
        ]
    }


def test_training_bootstrap_selects_stable_r6_and_validation_closes():
    records = {(1, 7): _result()}
    naive, conservative, confidence = _mode_maps(
        records,
        ["a", "b"],
        replicates=200,
        seed=123,
        min_training_documents=2,
        minimum_improvement=0.0,
    )
    assert naive[(1, 7)] == 6
    assert conservative[(1, 7)] == 6
    assert confidence["1/7"]["relative_improvement_ci95"] == [0.5, 0.5]

    summary = _aggregate_strategy(
        records,
        ["x", "y"],
        lambda key: conservative[key],
        replicates=200,
        seed=456,
    )
    assert summary["experts_selected_r6"] == 1
    assert summary["baseline_weighted_squared_error"] == 30.0
    assert summary["candidate_weighted_squared_error"] == 15.0
    assert summary["relative_improvement"] == 0.5
    assert summary["relative_improvement_ci95"] == [0.5, 0.5]
    assert summary["baseline_weighted_output_nmse"] == 0.15
    assert summary["candidate_weighted_output_nmse"] == 0.075


def test_selection_support_floor_falls_back_to_r0():
    records = {(24, 9): _result()}
    _naive, conservative, _confidence = _mode_maps(
        records,
        ["a", "b"],
        replicates=100,
        seed=123,
        min_training_documents=3,
        minimum_improvement=0.0,
    )
    assert conservative[(24, 9)] == 0
