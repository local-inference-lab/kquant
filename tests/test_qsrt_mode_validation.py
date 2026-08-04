from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import kquant.pack.qsrt_mode_validation as qsrt_mode_validation
from kquant.pack.qsrt_mode_validation import (
    load_qsrt_mode_validation_summary,
    mode_validation_summary_document,
    validate_mode_validation_layer_metrics,
)
from scripts.summarize_qsrt_mode_validation import (
    paired_document_bootstrap,
    summarize_arrays,
)


def _metrics() -> dict[str, torch.Tensor]:
    selected = torch.tensor([[4.0, 8.0, 12.0], [1.0, 2.0, 3.0]], dtype=torch.float64)
    r0 = torch.tensor([[5.0, 10.0, 15.0], [1.0, 2.0, 3.0]], dtype=torch.float64)
    return {
        "expert_ids": torch.tensor([0, 1], dtype=torch.int16),
        "selected_r13": torch.tensor([1, 0], dtype=torch.uint8),
        "selected_r2": torch.tensor([2, 0], dtype=torch.uint8),
        "audited": torch.tensor([True, False]),
        "selected_validation_sse": selected,
        "r0_validation_sse": r0,
        "validation_reference_energy": torch.tensor(
            [[20.0, 30.0, 40.0], [2.0, 3.0, 4.0]], dtype=torch.float64
        ),
        "validation_counts": torch.tensor([[2, 3, 4], [1, 1, 1]], dtype=torch.int32),
        "selected_validation_damage": selected.sum(dim=1),
        "r0_validation_damage": r0.sum(dim=1),
    }


def test_mode_validation_metrics_close_against_selected_sidecar() -> None:
    metrics = _metrics()
    result = validate_mode_validation_layer_metrics(
        metrics,
        selected_r13=np.array([1, 0], dtype=np.uint8),
        selected_r2=np.array([2, 0], dtype=np.uint8),
        selected_validation_sse=metrics["selected_validation_sse"].numpy(),
        validation_reference_energy=metrics[
            "validation_reference_energy"
        ].numpy(),
        validation_counts=metrics["validation_counts"].numpy(),
        documents=3,
        expert_ids=(0, 1),
    )

    assert result["audited"].tolist() == [True, False]
    assert result["selected_damage"].tolist() == [24.0, 6.0]
    assert result["r0_damage"].tolist() == [30.0, 6.0]
    assert result["routed_rows"].tolist() == [9, 3]


def test_mode_validation_metrics_reject_uniform_r0_drift() -> None:
    metrics = _metrics()
    metrics["r0_validation_sse"][1, 0] += 1
    metrics["r0_validation_damage"] = metrics["r0_validation_sse"].sum(dim=1)
    with pytest.raises(ValueError, match="uniform assignments"):
        validate_mode_validation_layer_metrics(
            metrics,
            selected_r13=np.array([1, 0], dtype=np.uint8),
            selected_r2=np.array([2, 0], dtype=np.uint8),
            selected_validation_sse=metrics["selected_validation_sse"].numpy(),
            validation_reference_energy=metrics[
                "validation_reference_energy"
            ].numpy(),
            validation_counts=metrics["validation_counts"].numpy(),
            documents=3,
            expert_ids=(0, 1),
        )


def test_paired_document_bootstrap_preserves_document_pairing() -> None:
    result = paired_document_bootstrap(
        np.array([10.0, 20.0, 30.0]),
        np.array([8.0, 16.0, 24.0]),
        replicates=1_000,
        seed=9,
    )

    assert result["relative_improvement"] == pytest.approx(0.2)
    assert result["ci95"][0] == pytest.approx(0.2)
    assert result["ci95"][1] == pytest.approx(0.2)


def test_mode_summary_passes_consistent_external_gain() -> None:
    r13 = np.array([[1, 0]], dtype=np.uint8)
    r2 = np.array([[2, 0]], dtype=np.uint8)
    audited = (r13 != 0) | (r2 != 0)
    r0 = np.array([[[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]]])
    selected = np.array([[[8.0, 16.0, 24.0], [1.0, 2.0, 3.0]]])
    support = np.array([[3, 3]], dtype=np.int32)

    result = summarize_arrays(
        r13,
        r2,
        audited,
        selected,
        r0,
        support,
        replicates=1_000,
        seed=11,
    )

    assert result["status"] == "pass"
    assert result["aggregate"]["relative_improvement"] == pytest.approx(0.2)
    assert result["per_axis_mode"]["r13"]["R1"][
        "relative_improvement"
    ] == pytest.approx(0.2)
    assert result["per_axis_mode"]["r2"]["R2"][
        "relative_improvement"
    ] == pytest.approx(0.2)
    assert result["per_format"]["R1/R2"]["relative_improvement"] == pytest.approx(
        0.2
    )


def test_mode_summary_fails_external_regression() -> None:
    r13 = np.array([[1]], dtype=np.uint8)
    r2 = np.array([[0]], dtype=np.uint8)
    audited = (r13 != 0) | (r2 != 0)
    r0 = np.array([[[10.0, 20.0, 30.0]]])
    selected = np.array([[[11.0, 22.0, 33.0]]])
    support = np.array([[3]], dtype=np.int32)

    result = summarize_arrays(
        r13,
        r2,
        audited,
        selected,
        r0,
        support,
        replicates=1_000,
        seed=13,
    )

    assert result["status"] == "fail"
    assert not result["pass"]


def test_mode_summary_loader_recomputes_and_requires_pass(
    tmp_path,
    monkeypatch,
) -> None:
    pool_root = (tmp_path / "pool").resolve()
    validation_root = (tmp_path / "selected").resolve()
    score_root = (tmp_path / "matched-r0").resolve()
    pool = SimpleNamespace(root=pool_root, content_sha256="11" * 32)
    validation = SimpleNamespace(
        root=validation_root,
        content_sha256="22" * 32,
    )
    scores = SimpleNamespace(
        root=score_root,
        content_sha256="33" * 32,
        selected_r13=np.array([[1, 0]], dtype=np.uint8),
        selected_r2=np.array([[2, 0]], dtype=np.uint8),
        audited=np.array([[True, False]], dtype=np.bool_),
        per_document_selected=np.array(
            [[[8.0, 16.0, 24.0], [1.0, 2.0, 3.0]]]
        ),
        per_document_r0=np.array(
            [[[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]]]
        ),
        supported_documents=np.array([[3, 3]], dtype=np.int32),
    )
    monkeypatch.setattr(
        qsrt_mode_validation,
        "load_qsrt_mode_validation_scores",
        lambda root, candidate_pool, selected: scores,
    )
    document = mode_validation_summary_document(
        pool,
        validation,
        scores,
        replicates=1_000,
        seed=17,
    )
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(document))

    loaded, loaded_scores = load_qsrt_mode_validation_summary(
        path,
        pool,
        validation,
    )

    assert loaded == document
    assert loaded_scores is scores
    document["pass"] = False
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="does not reproduce"):
        load_qsrt_mode_validation_summary(path, pool, validation)


def test_mode_summary_loader_rejects_reproducible_failure(
    tmp_path,
    monkeypatch,
) -> None:
    pool = SimpleNamespace(root=(tmp_path / "pool").resolve(), content_sha256="a")
    validation = SimpleNamespace(
        root=(tmp_path / "selected").resolve(),
        content_sha256="b",
    )
    scores = SimpleNamespace(
        root=(tmp_path / "matched-r0").resolve(),
        content_sha256="c",
        selected_r13=np.array([[1]], dtype=np.uint8),
        selected_r2=np.array([[0]], dtype=np.uint8),
        audited=np.array([[True]], dtype=np.bool_),
        per_document_selected=np.array([[[11.0, 22.0, 33.0]]]),
        per_document_r0=np.array([[[10.0, 20.0, 30.0]]]),
        supported_documents=np.array([[3]], dtype=np.int32),
    )
    monkeypatch.setattr(
        qsrt_mode_validation,
        "load_qsrt_mode_validation_scores",
        lambda root, candidate_pool, selected: scores,
    )
    document = mode_validation_summary_document(
        pool,
        validation,
        scores,
        replicates=1_000,
        seed=19,
    )
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="did not pass"):
        load_qsrt_mode_validation_summary(path, pool, validation)
