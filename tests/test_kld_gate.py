from __future__ import annotations

import hashlib

import pytest

from kquant.kld_gate import (
    ComparisonWindow,
    summarize_paired_kld,
    validate_comparison_report,
    validate_logits_manifest,
    validate_suite_manifest,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _suite() -> dict:
    token_hashes = [_hash("tokens-0"), _hash("tokens-1")]
    suite_hash = hashlib.sha256("\n".join(token_hashes).encode()).hexdigest()
    return {
        "context_length": 4,
        "window_count": 2,
        "scored_positions_per_window": 3,
        "total_scored_positions": 6,
        "suite_token_hash_sha256": suite_hash,
        "domain_counts": {"code": 1, "prose": 1},
        "windows": [
            {
                "index": 0,
                "domain": "prose",
                "num_tokens": 4,
                "token_file": "tokens/window-000.json",
                "token_ids_json_sha256": token_hashes[0],
            },
            {
                "index": 1,
                "domain": "code",
                "num_tokens": 4,
                "token_file": "tokens/window-001.json",
                "token_ids_json_sha256": token_hashes[1],
            },
        ],
    }


def _logits_manifest(suite: dict) -> dict:
    records = []
    for window in suite["windows"]:
        records.append(
            {
                "window_index": window["index"],
                "domain": window["domain"],
                "file": f"logits_{window['index']:03d}.safetensors",
                "key": "logits",
                "shape": [3, 7],
                "dtype": "F32",
                "size_bytes": 100,
                "sha256": _hash(f"file-{window['index']}"),
                "token_ids_json_sha256": window["token_ids_json_sha256"],
            }
        )
    return {
        "suite_token_hash_sha256": suite["suite_token_hash_sha256"],
        "vocab_size": 7,
        "window_count": 2,
        "window_range": [0, 2],
        "total_scored_positions": 6,
        "total_size_bytes": 200,
        "windows": records,
    }


def _comparison(suite: dict, kl: list[float]) -> dict:
    def stats(value: float) -> dict[str, float]:
        return {
            "mean": value,
            "median": value * 0.8,
            "p95": value * 1.2,
            "p99": value * 1.4,
            "max": value * 2.0,
        }

    per_window = []
    for window, value in zip(suite["windows"], kl, strict=True):
        per_window.append(
            {
                "window_index": window["index"],
                "domain": window["domain"],
                "positions": 3,
                "kl_reference_to_candidate": stats(value),
                "js": stats(value / 4),
                "top1_agreement": 1.0 - value,
            }
        )
    return {
        "direction": "KL(reference || candidate)",
        "windows": 2,
        "window_range": [0, 2],
        "positions": 6,
        "kl_reference_to_candidate": stats(sum(kl) / 2),
        "kl_window_cluster_bootstrap": {"cluster_unit": "window"},
        "js": stats(sum(kl) / 8),
        "top1_agreement": 1.0 - sum(kl) / 2,
        "per_window": per_window,
    }


def test_suite_and_logit_manifests_close_exactly() -> None:
    suite = _suite()
    suite_hash, windows = validate_suite_manifest(
        suite, expected_suite_hash=suite["suite_token_hash_sha256"]
    )

    records = validate_logits_manifest(
        _logits_manifest(suite),
        label="candidate",
        suite_hash=suite_hash,
        expected_windows=windows,
    )

    assert len(records) == 2
    assert records[1]["domain"] == "code"


def test_logit_manifest_can_cover_more_windows_than_quick_gate() -> None:
    suite = _suite()
    suite_hash, windows = validate_suite_manifest(
        suite, expected_suite_hash=suite["suite_token_hash_sha256"]
    )

    records = validate_logits_manifest(
        _logits_manifest(suite),
        label="reference",
        suite_hash=suite_hash,
        expected_windows=windows[:1],
    )

    assert len(records) == 1
    assert records[0]["window_index"] == 0


def test_full_reference_manifest_can_infer_legacy_missing_window_range() -> None:
    suite = _suite()
    suite_hash, windows = validate_suite_manifest(
        suite, expected_suite_hash=suite["suite_token_hash_sha256"]
    )
    manifest = _logits_manifest(suite)
    manifest.pop("window_range")

    records = validate_logits_manifest(
        manifest,
        label="reference",
        suite_hash=suite_hash,
        expected_windows=windows,
    )

    assert [record["window_index"] for record in records] == [0, 1]


def test_comparison_rejects_a_window_domain_mismatch() -> None:
    suite = _suite()
    _, windows = validate_suite_manifest(
        suite, expected_suite_hash=suite["suite_token_hash_sha256"]
    )
    report = _comparison(suite, [0.1, 0.2])
    report["per_window"][1]["domain"] = "prose"

    with pytest.raises(ValueError, match="domain differs"):
        validate_comparison_report(
            report, label="candidate", expected_windows=windows
        )


def test_paired_gate_passes_only_a_supported_above_noise_improvement() -> None:
    baseline = tuple(
        ComparisonWindow(index, "all", 3, value, value / 4, 0.8)
        for index, value in enumerate([0.20, 0.22, 0.18, 0.21])
    )
    candidate = tuple(
        ComparisonWindow(index, "all", 3, value, value / 4, 0.9)
        for index, value in enumerate([0.10, 0.11, 0.09, 0.10])
    )

    result = summarize_paired_kld(
        baseline,
        candidate,
        baseline_label="3p09",
        candidate_label="new",
        bootstrap_samples=1000,
        documented_runtime_noise_kl=0.004,
    )

    assert result["decision"]["verdict"] == "pass"
    assert result["decision"]["statistical_status"] == "improvement_supported"
    assert result["change"]["paired_kl_improvement"]["ci95_low"] > 0


def test_paired_gate_requires_repeat_inside_runtime_noise_floor() -> None:
    baseline = tuple(
        ComparisonWindow(index, "all", 3, 0.10, 0.02, 0.9)
        for index in range(4)
    )
    candidate = tuple(
        ComparisonWindow(index, "all", 3, 0.098, 0.019, 0.9)
        for index in range(4)
    )

    result = summarize_paired_kld(
        baseline,
        candidate,
        baseline_label="3p09",
        candidate_label="new",
        bootstrap_samples=1000,
        documented_runtime_noise_kl=0.004,
    )

    assert result["decision"]["statistical_status"] == "improvement_supported"
    assert result["decision"]["point_delta_exceeds_runtime_noise"] is False
    assert result["decision"]["verdict"] == "repeat_required"


def test_paired_gate_fails_a_supported_domain_regression() -> None:
    baseline = (
        ComparisonWindow(0, "code", 3, 0.10, 0.02, 0.9),
        ComparisonWindow(1, "code", 3, 0.10, 0.02, 0.9),
        ComparisonWindow(2, "prose", 3, 0.30, 0.06, 0.8),
        ComparisonWindow(3, "prose", 3, 0.30, 0.06, 0.8),
    )
    candidate = (
        ComparisonWindow(0, "code", 3, 0.20, 0.04, 0.8),
        ComparisonWindow(1, "code", 3, 0.20, 0.04, 0.8),
        ComparisonWindow(2, "prose", 3, 0.05, 0.01, 0.95),
        ComparisonWindow(3, "prose", 3, 0.05, 0.01, 0.95),
    )

    result = summarize_paired_kld(
        baseline,
        candidate,
        baseline_label="3p09",
        candidate_label="new",
        bootstrap_samples=1000,
    )

    assert result["decision"]["supported_domain_regressions"] == ["code"]
    assert result["decision"]["verdict"] == "fail"
