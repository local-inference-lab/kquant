import json

import pytest
import torch

from kquant.capture import LayerSamples
from scripts.experiment_context_palettes import _activation_metrics
from scripts.experiment_retained_w2_rate_transfer import (
    _request_documents,
    _validate_corpus_reports,
)


def _report(capture, mode, hashes):
    return {
        "kind": "kquant_interim_calibration_corpus_run",
        "finalized": True,
        "capture_dir": str(capture),
        "fold": {
            "modulus": 8,
            "index": 0,
            "mode": mode,
            "unit": "whole JSONL record by content hash",
        },
        "documents": [{"document_hash": value} for value in hashes],
    }


def test_corpus_report_validation_proves_document_disjointness(tmp_path):
    training_capture = tmp_path / "train.kqcapture"
    validation_capture = tmp_path / "validation.kqcapture"
    training_capture.mkdir()
    validation_capture.mkdir()
    training_report = tmp_path / "train.json"
    validation_report = tmp_path / "validation.json"
    training_report.write_text(
        json.dumps(_report(training_capture, "exclude", ["a", "b"]))
    )
    validation_report.write_text(
        json.dumps(_report(validation_capture, "include", ["c"]))
    )

    result = _validate_corpus_reports(
        training_report,
        validation_report,
        training_capture=training_capture,
        validation_capture=validation_capture,
    )

    assert result["training_documents"] == 2
    assert result["validation_documents"] == 1
    assert result["document_overlap"] == 0


def test_corpus_report_validation_rejects_document_overlap(tmp_path):
    training_capture = tmp_path / "train.kqcapture"
    validation_capture = tmp_path / "validation.kqcapture"
    training_capture.mkdir()
    validation_capture.mkdir()
    training_report = tmp_path / "train.json"
    validation_report = tmp_path / "validation.json"
    training_report.write_text(
        json.dumps(_report(training_capture, "exclude", ["same"]))
    )
    validation_report.write_text(
        json.dumps(_report(validation_capture, "include", ["same"]))
    )

    with pytest.raises(ValueError, match="document overlap"):
        _validate_corpus_reports(
            training_report,
            validation_report,
            training_capture=training_capture,
            validation_capture=validation_capture,
        )


def test_request_documents_maps_sequential_corpus_epochs():
    report = _report("capture", "exclude", ["first", "second"])
    report["planned_requests"] = 2
    report["completed_requests"] = 2
    assert _request_documents(report) == {1: "first", 2: "second"}


def test_activation_metrics_exclude_startup_and_preserve_document_totals():
    mid_values = torch.tensor(
        [[100.0, 0.0], [1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]
    )
    request_steps = (0, 1, 1, 2)
    observations = torch.tensor(
        [(step << 32) | index for index, step in enumerate(request_steps)],
        dtype=torch.int64,
    )
    samples = LayerSamples(
        input_values=torch.empty(0, 2),
        input_weights=torch.empty(0),
        input_observations=torch.empty(0, dtype=torch.int64),
        input_experts=torch.empty(0, 1, dtype=torch.int32),
        input_gates=torch.empty(0, 1),
        input_split=torch.empty(0, dtype=torch.int8),
        routed_latent=torch.empty(0, 2),
        mid_values=mid_values,
        mid_weights=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        mid_observations=observations,
        mid_experts=torch.zeros(4, dtype=torch.int32),
        mid_split=torch.zeros(4, dtype=torch.int8),
    )
    metrics = _activation_metrics(
        torch.eye(2),
        torch.zeros(2, 2),
        samples=samples,
        expert=0,
        device=torch.device("cpu"),
        training_request_documents={1: "doc-a", 2: "doc-b"},
    )["train"]

    assert metrics["rows"] == 3
    assert metrics["documents"] == 2
    assert metrics["excluded_unmapped_rows"] == 1
    assert metrics["weight_sum"] == 9.0
    assert metrics["weighted_squared_error"] == 22.0
    assert metrics["weighted_reference_energy"] == 22.0
    assert metrics["weighted_output_nmse"] == 1.0
    assert metrics["request_contributions"] == [
        {
            "request_step": 1,
            "document_hash": "doc-a",
            "rows": 2,
            "weight_sum": 5.0,
            "weighted_squared_error": 14.0,
            "weighted_reference_energy": 14.0,
        },
        {
            "request_step": 2,
            "document_hash": "doc-b",
            "rows": 1,
            "weight_sum": 4.0,
            "weighted_squared_error": 8.0,
            "weighted_reference_energy": 8.0,
        },
    ]
