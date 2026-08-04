import pytest

from scripts.summarize_coupled_exl3 import (
    _aggregate,
    _argmin_map,
    _compare_strategies,
)


def _metric(documents, error):
    contributions = [
        {
            "document_hash": document,
            "request_step": index,
            "rows": 1,
            "gate_square_sum": 1.0,
            "route_weighted_squared_error": error,
            "route_weighted_reference_energy": 100.0,
            "teacher_mixture_energy": 200.0,
        }
        for index, document in enumerate(documents, start=1)
    ]
    return {
        "rows": len(documents),
        "documents": len(documents),
        "gate_square_sum": float(len(documents)),
        "route_weighted_squared_error": error * len(documents),
        "route_weighted_reference_energy": 100.0 * len(documents),
        "route_weighted_expert_nmse": error / 100.0,
        "teacher_mixture_energy": 200.0 * len(documents),
        "teacher_mixture_replacement_squared_error": error * len(documents),
        "teacher_mixture_replacement_nmse": error / 200.0,
        "request_contributions": contributions,
    }


def _record(documents):
    results = []
    for r13, r2, error in ((0, 0, 10.0), (1, 1, 8.0), (1, 2, 5.0)):
        metric = _metric(documents, error)
        results.append(
            {
                "r13": r13,
                "r2": r2,
                "coupled_activation": {"train": metric, "validation": metric},
            }
        )
    return {"results": results}


def test_separate_mode_argmin_and_document_aggregate():
    documents = ["a", "b", "c"]
    records = {(1, 2): _record(documents)}
    modes = [(0, 0), (1, 1), (1, 2)]
    selected = _argmin_map(records, modes)
    assert selected[(1, 2)] == (1, 2)
    summary = _aggregate(
        records,
        documents,
        lambda key: selected[key],
        replicates=200,
        seed=123,
    )
    assert summary["relative_improvement"] == pytest.approx(0.5)
    assert summary["relative_improvement_ci95"] == pytest.approx([0.5, 0.5])
    assert summary["selected_mode_histogram"] == {"R1/R2": 1}
    assert summary["baseline_route_weighted_expert_nmse"] == pytest.approx(0.1)
    assert summary["candidate_teacher_mixture_replacement_nmse"] == pytest.approx(
        0.025
    )

    comparison = _compare_strategies(
        records,
        documents,
        lambda _key: (1, 1),
        lambda _key: (1, 2),
        replicates=200,
        seed=456,
    )
    assert comparison["relative_improvement"] == pytest.approx(0.375)
