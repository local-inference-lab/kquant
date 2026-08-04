from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics

import pytest
from safetensors.torch import save_file
import torch

from kquant.tp12_performance_gate import (
    BENCHMARK_KIND,
    BENCHMARK_SCHEMA_VERSION,
    FIXTURE_KIND,
    FIXTURE_SCHEMA_VERSION,
    ISOLATED_BENCHMARK_KIND,
    ISOLATED_BENCHMARK_SCHEMA_VERSION,
    TP12PerformanceThresholds,
    select_tp12_performance_layers,
    summarize_tp12_isolated_performance,
    summarize_tp12_performance,
    summarize_tp12_routed_performance,
    validate_tp12_performance_report,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(
    path: Path,
    *,
    layer: int,
    rank: int,
    sealed: bool,
    candidate_pool: Path | None = None,
    capture: Path | None = None,
) -> tuple[dict[str, object], float]:
    rows = 5
    routes = torch.arange(16, dtype=torch.int32).repeat(rows, 1)
    modes = torch.zeros(896, dtype=torch.uint8)
    modes[rank] = 1
    routed = modes[routes.long()]
    fraction = float(routed.to(torch.float64).mean())
    tensors = {
        "fc1_pair_modes": modes,
        "fc2_pair_modes": modes.clone(),
        "topk_ids": routes,
        "topk_weights": torch.full((rows, 16), 1.0 / 16, dtype=torch.float32),
        "observation_ids": torch.arange(rows, dtype=torch.int64),
        "source_row_indices": torch.arange(rows, dtype=torch.int64),
    }
    candidate_pool_value = (
        str(candidate_pool.resolve()) if candidate_pool is not None else "/candidate/pool"
    )
    capture_value = (
        str(capture.resolve()) if capture is not None else "/validation/capture"
    )
    candidate_digest = (
        _sha256(candidate_pool / "mixed_exl3_candidate_manifest.json")
        if candidate_pool is not None
        else "a" * 64
    )
    capture_digest = (
        _sha256(capture / "manifest.json")
        if capture is not None
        else "b" * 64
    )
    manifest: dict[str, object] = {
        "kind": FIXTURE_KIND,
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "candidate_pool": candidate_pool_value,
        "candidate_pool_sealed": sealed,
        "candidate_logical_trellis_schema": "kquant_mixed_exl3_tp12_proto_v2",
        "candidate_manifest_sha256": candidate_digest,
        "capture": capture_value,
        "capture_manifest_sha256": capture_digest,
        "tp_size": 12,
        "num_experts": 896,
        "top_k": 16,
        "hidden_size": 3584,
        "local_intermediate_size": 256,
        "layer": layer,
        "rank": rank,
        "fixture_rows": rows,
        "p24_experts": 1,
        "p24_route_executions": int(routed.sum()),
    }
    save_file(tensors, str(path), metadata={"manifest": json.dumps(manifest)})
    return manifest, fraction


def _case(
    *,
    tokens: int,
    ratio: float,
    p24_fraction: float,
    correctness: bool = True,
    replays: int = 4,
) -> dict[str, object]:
    base = [100.0, 101.0, 99.0, 100.5]
    p33 = [base[index % len(base)] for index in range(replays)]
    sparse = [value * ratio for value in p33]
    return {
        "tokens": tokens,
        "mode_summary": {
            "P33": {
                "routed_fc1_p24_fraction": 0.0,
                "routed_fc2_p24_fraction": 0.0,
            },
            "sparse": {
                "routed_fc1_p24_fraction": p24_fraction,
                "routed_fc2_p24_fraction": p24_fraction,
            },
        },
        "kernel_summary": {
            scenario: {
                "resource_introspection_available": True,
                "local_memory_bytes_per_thread": 0,
            }
            for scenario in ("P33", "sparse")
        },
        "correctness": {
            scenario: {
                "passed": correctness,
                "finite": True,
                "eager_graph_bit_exact": True,
            }
            for scenario in ("P33", "sparse")
        },
        "raw_hot_us": {"P33": p33, "sparse": sparse},
        "hot_median_over_baseline": {
            "P33": 1.0,
            "sparse": statistics.median(sparse) / statistics.median(p33),
        },
        "graph_replay_allocation": {
            "allocated_stable": True,
            "reserved_stable": True,
        },
    }


def _write_benchmark_matrix(
    root: Path,
    *,
    ratio: float = 1.005,
    sealed: bool = True,
    bad_correctness_coordinate: tuple[int, int, int] | None = None,
    candidate_pool: Path | None = None,
    capture: Path | None = None,
    replays: int = 4,
) -> list[Path]:
    paths: list[Path] = []
    for layer in (6,):
        for rank in range(12):
            fixture_path = root / f"fixture-l{layer}-r{rank}.safetensors"
            manifest, p24_fraction = _write_fixture(
                fixture_path,
                layer=layer,
                rank=rank,
                sealed=sealed,
                candidate_pool=candidate_pool,
                capture=capture,
            )
            claimed_manifest = {
                **manifest,
                "loaded_fc1_p24_route_fraction": p24_fraction,
                "loaded_fc2_p24_route_fraction": p24_fraction,
            }
            cases = [
                _case(
                    tokens=tokens,
                    ratio=ratio,
                    p24_fraction=p24_fraction,
                    correctness=(
                        bad_correctness_coordinate != (layer, rank, tokens)
                    ),
                    replays=replays,
                )
                for tokens in (1, 4)
            ]
            document = {
                "kind": BENCHMARK_KIND,
                "schema_version": BENCHMARK_SCHEMA_VERSION,
                "provenance": {
                    "source_sha256": "c" * 64,
                    "route_fixture": {
                        "path": str(fixture_path),
                        "sha256": _sha256(fixture_path),
                        "manifest": claimed_manifest,
                    },
                },
                "contract": {
                    "tp": 12,
                    "hidden_size": 3584,
                    "local_intermediate_size": 256,
                    "route_experts": 896,
                    "topk": 16,
                    "materialized_experts": 896,
                    "route_source": "captured_fixture",
                    "timed_route_window_rotation": True,
                    "route_copy_in_timed_region": False,
                    "complete_routed_pipeline": True,
                    "outer_hadamard_included": True,
                    "cuda_graph_replay": True,
                    "cuda_graph_replay_allocation_stable_required": True,
                    "scenarios": ["P33", "sparse"],
                    "tokens": [1, 4],
                },
                "device": {
                    "name": "test-sm120",
                    "uuid": "GPU-test",
                    "capability": [12, 0],
                },
                "cases": cases,
            }
            benchmark_path = root / f"benchmark-l{layer}-r{rank}.json"
            benchmark_path.write_text(json.dumps(document))
            paths.append(benchmark_path)
    return paths


def _thresholds() -> TP12PerformanceThresholds:
    return TP12PerformanceThresholds(
        min_hot_replays=4,
        bootstrap_replicates=200,
        bootstrap_seed=7,
    )


def test_performance_layer_selection_balances_depth_and_hotspots() -> None:
    scores = {layer: 0.0001 * layer for layer in range(1, 93)}
    scores.update({6: 0.9, 24: 0.8, 48: 0.7, 72: 0.6, 92: 0.5})
    runtime_mix = {
        "kind": "kquant_mixed_exl3_tp12_runtime_mix",
        "schema_version": 1,
        "candidate_pool_sealed": True,
        "layers": list(range(1, 93)),
        "per_layer": {
            str(layer): {
                "layer": layer,
                "p24_rank_route_fraction": scores[layer],
            }
            for layer in range(1, 93)
        },
    }

    selected = select_tp12_performance_layers(runtime_mix, count=5)

    assert selected == (6, 24, 48, 72, 92)


def test_performance_layer_selection_requires_sealed_complete_p24_mix() -> None:
    runtime_mix = {
        "kind": "kquant_mixed_exl3_tp12_runtime_mix",
        "schema_version": 1,
        "candidate_pool_sealed": False,
        "layers": [1, 24, 48, 72],
        "per_layer": {},
    }
    with pytest.raises(ValueError, match="sealed"):
        select_tp12_performance_layers(runtime_mix, count=4)


def _write_isolated_benchmark(
    path: Path,
    *,
    ratio: float = 1.005,
    correctness: bool = True,
    omit_coordinate: tuple[int, str] | None = None,
    replays: int = 4,
) -> Path:
    cases: list[dict[str, object]] = []
    for rows in (1, 2):
        for axis in ("n", "k"):
            if (rows, axis) == omit_coordinate:
                continue
            base = [100.0, 101.0, 99.0, 100.5]
            p33 = [base[index % len(base)] for index in range(replays)]
            p24 = [sample * ratio for sample in p33]
            cases.append(
                {
                    "rows": rows,
                    "rate_axis": axis,
                    "raw_hot_us": {"P24": p24, "P33": p33, "K3": p33},
                    "hot_p24_over_p33": (
                        statistics.median(p24) / statistics.median(p33)
                    ),
                    "correctness": {
                        codec: {
                            "passed": correctness,
                            "finite": True,
                            "eager_graph_bit_exact": True,
                        }
                        for codec in ("P24", "P33", "K3")
                    },
                    "graph_replay_allocation": {
                        "allocated_stable": True,
                        "reserved_stable": True,
                    },
                }
            )
    document = {
        "kind": ISOLATED_BENCHMARK_KIND,
        "schema_version": ISOLATED_BENCHMARK_SCHEMA_VERSION,
        "provenance": {"source_sha256": "d" * 64},
        "contract": {
            "tp": 12,
            "hidden_size": 3584,
            "local_intermediate_size": 256,
            "rows": [1, 2],
            "outer_hadamard_included": True,
            "cuda_graph_replay": True,
            "cuda_graph_replay_allocation_stable_required": True,
        },
        "device": {
            "name": "test-sm120",
            "uuid": "GPU-test",
            "capability": [12, 0],
        },
        "cases": cases,
    }
    path.write_text(json.dumps(document))
    return path


def test_isolated_decoder_matrix_passes(tmp_path: Path) -> None:
    path = _write_isolated_benchmark(tmp_path / "isolated.json")

    report = summarize_tp12_isolated_performance(
        path,
        expected_rows=(1, 2),
        thresholds=_thresholds(),
    )

    assert report["status"] == "pass"
    assert report["scope"]["evaluated_cases"] == 4
    assert report["decision"]["maximum_p24_ci95_high_over_p33"] == pytest.approx(
        1.005
    )


def test_isolated_decoder_regression_and_correctness_fail(tmp_path: Path) -> None:
    path = _write_isolated_benchmark(
        tmp_path / "isolated.json", ratio=1.04, correctness=False
    )

    report = summarize_tp12_isolated_performance(
        path,
        expected_rows=(1, 2),
        thresholds=_thresholds(),
    )

    assert report["status"] == "fail"
    failures = "\n".join(report["decision"]["failures"])
    assert "CI95 high" in failures
    assert "P24: passed failed" in failures


def test_isolated_decoder_rejects_incomplete_matrix(tmp_path: Path) -> None:
    path = _write_isolated_benchmark(
        tmp_path / "isolated.json", omit_coordinate=(2, "k")
    )

    with pytest.raises(ValueError, match="does not cover"):
        summarize_tp12_isolated_performance(
            path,
            expected_rows=(1, 2),
            thresholds=_thresholds(),
        )


def test_isolated_decoder_rejects_reported_ratio_mismatch(tmp_path: Path) -> None:
    path = _write_isolated_benchmark(tmp_path / "isolated.json")
    document = json.loads(path.read_text())
    document["cases"][0]["hot_p24_over_p33"] = 0.5
    path.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="reported ratio disagrees"):
        summarize_tp12_isolated_performance(
            path,
            expected_rows=(1, 2),
            thresholds=_thresholds(),
        )


def test_combined_gate_requires_both_components(tmp_path: Path) -> None:
    routed = _write_benchmark_matrix(tmp_path)
    isolated = _write_isolated_benchmark(tmp_path / "isolated.json")

    report = summarize_tp12_performance(
        isolated,
        routed,
        expected_layers=(6,),
        expected_isolated_rows=(1, 2),
        expected_tokens=(1, 4),
        thresholds=_thresholds(),
    )

    assert report["status"] == "pass"
    assert report["decision"] == {
        "verdict": "pass",
        "failures": [],
        "isolated_verdict": "pass",
        "routed_verdict": "pass",
    }

    failing_isolated = _write_isolated_benchmark(
        tmp_path / "isolated-failing.json", ratio=1.04
    )
    failed = summarize_tp12_performance(
        failing_isolated,
        routed,
        expected_layers=(6,),
        expected_isolated_rows=(1, 2),
        expected_tokens=(1, 4),
        thresholds=_thresholds(),
    )
    assert failed["status"] == "fail"
    assert failed["decision"]["isolated_verdict"] == "fail"
    assert failed["decision"]["routed_verdict"] == "pass"


def test_combined_report_is_recomputed_from_bound_evidence(tmp_path: Path) -> None:
    candidate_pool = tmp_path / "candidate-pool"
    candidate_pool.mkdir()
    (candidate_pool / "mixed_exl3_candidate_manifest.json").write_text(
        '{"sealed": true}\n'
    )
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "manifest.json").write_text('{"complete": true}\n')
    routed = _write_benchmark_matrix(
        tmp_path,
        candidate_pool=candidate_pool,
        capture=capture,
        replays=200,
    )
    isolated = _write_isolated_benchmark(
        tmp_path / "isolated-production.json", replays=200
    )
    report = summarize_tp12_performance(
        isolated,
        routed,
        expected_layers=(6,),
        expected_isolated_rows=(1, 2),
        expected_tokens=(1, 4),
    )
    report_path = tmp_path / "performance-summary.json"
    report_path.write_text(json.dumps(report))

    receipt = validate_tp12_performance_report(
        report_path,
        expected_candidate_pool=candidate_pool,
    )

    assert receipt["status"] == "pass"
    assert receipt["candidate_pool"] == str(candidate_pool.resolve())
    assert receipt["routed_cases"] == 24

    benchmark = routed[0]
    changed = json.loads(benchmark.read_text())
    changed["cases"][0]["raw_hot_us"]["sparse"][0] *= 1.5
    benchmark.write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        validate_tp12_performance_report(
            report_path,
            expected_candidate_pool=candidate_pool,
        )


def test_complete_tp12_matrix_passes_and_is_authenticated(tmp_path: Path) -> None:
    paths = _write_benchmark_matrix(tmp_path)

    report = summarize_tp12_routed_performance(
        paths,
        expected_layers=(6,),
        expected_tokens=(1, 4),
        thresholds=_thresholds(),
    )

    assert report["status"] == "pass"
    assert report["scope"]["benchmark_files"] == 12
    assert report["scope"]["evaluated_cases"] == 24
    assert report["decision"]["maximum_sparse_median_over_p33"] == pytest.approx(
        1.005
    )
    assert report["decision"]["failures"] == []
    assert [item["rank"] for item in report["inputs"]] == list(range(12))


def test_performance_or_correctness_regression_fails_gate(tmp_path: Path) -> None:
    paths = _write_benchmark_matrix(
        tmp_path,
        ratio=1.03,
        bad_correctness_coordinate=(6, 4, 1),
    )

    report = summarize_tp12_routed_performance(
        paths,
        expected_layers=(6,),
        expected_tokens=(1, 4),
        thresholds=_thresholds(),
    )

    assert report["status"] == "fail"
    failures = "\n".join(report["decision"]["failures"])
    assert "median" in failures
    assert "CI95 high" in failures
    assert "layer=6/rank=4/tokens=1/P33: passed failed" in failures


def test_gate_rejects_incomplete_rank_matrix(tmp_path: Path) -> None:
    paths = _write_benchmark_matrix(tmp_path)

    with pytest.raises(ValueError, match="expected 12 benchmark files"):
        summarize_tp12_routed_performance(
            paths[:-1],
            expected_layers=(6,),
            expected_tokens=(1, 4),
            thresholds=_thresholds(),
        )


def test_gate_rejects_unsealed_fixture_by_default(tmp_path: Path) -> None:
    paths = _write_benchmark_matrix(tmp_path, sealed=False)

    with pytest.raises(ValueError, match="not exported from a sealed pool"):
        summarize_tp12_routed_performance(
            paths,
            expected_layers=(6,),
            expected_tokens=(1, 4),
            thresholds=_thresholds(),
        )


def test_gate_rejects_fixture_changed_after_benchmark(tmp_path: Path) -> None:
    paths = _write_benchmark_matrix(tmp_path)
    document = json.loads(paths[0].read_text())
    fixture = Path(document["provenance"]["route_fixture"]["path"])
    fixture.write_bytes(fixture.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="fixture SHA-256 disagrees"):
        summarize_tp12_routed_performance(
            paths,
            expected_layers=(6,),
            expected_tokens=(1, 4),
            thresholds=_thresholds(),
        )
