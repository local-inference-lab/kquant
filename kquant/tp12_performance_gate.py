"""Fail-closed TP12 routed P24/P33 performance acceptance.

The B12X benchmark owns CUDA timing.  This module independently authenticates
its natural-route fixtures, recomputes paired bootstrap intervals from the raw
timings, and requires complete logical-rank coverage for every requested
candidate layer.  It never selects codec modes or changes an allocation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from safetensors import safe_open
import torch


KIND = "kquant_mixed_exl3_tp12_performance_gate"
ISOLATED_KIND = "kquant_mixed_exl3_tp12_isolated_performance_gate"
ROUTED_KIND = "kquant_mixed_exl3_tp12_routed_performance_gate"
SCHEMA_VERSION = 1
BENCHMARK_KIND = "b12x_trellis_pair_moe_tp12_benchmark"
BENCHMARK_SCHEMA_VERSION = 1
ISOLATED_BENCHMARK_KIND = "b12x_trellis_pair_tp12_benchmark"
ISOLATED_BENCHMARK_SCHEMA_VERSION = 1
FIXTURE_KIND = "kquant_mixed_exl3_tp12_benchmark_fixture"
FIXTURE_SCHEMA_VERSION = 1
TP_SIZE = 12
NUM_EXPERTS = 896
TOP_K = 16
HIDDEN_SIZE = 3584
LOCAL_INTERMEDIATE_SIZE = 256
RUNTIME_MIX_KIND = "kquant_mixed_exl3_tp12_runtime_mix"
RUNTIME_MIX_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TP12PerformanceThresholds:
    """Frozen production-frequency routed-kernel acceptance thresholds."""

    max_isolated_p24_ci95_high_over_p33: float = 1.03
    max_sparse_median_over_p33: float = 1.01
    max_sparse_ci95_high_over_p33: float = 1.02
    min_hot_replays: int = 200
    bootstrap_replicates: int = 10_000
    bootstrap_seed: int = 20260801

    def validate(self) -> None:
        if self.max_isolated_p24_ci95_high_over_p33 <= 0:
            raise ValueError("isolated CI threshold must be positive")
        if self.max_sparse_median_over_p33 <= 0:
            raise ValueError("median-ratio threshold must be positive")
        if self.max_sparse_ci95_high_over_p33 <= 0:
            raise ValueError("CI threshold must be positive")
        if self.min_hot_replays < 2:
            raise ValueError("at least two hot replays are required")
        if self.bootstrap_replicates < 100:
            raise ValueError("at least 100 bootstrap replicates are required")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_value(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _list(value: object, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    return value


def _finite_float(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _positive_samples(value: object, context: str) -> list[float]:
    samples = [_finite_float(item, context) for item in _list(value, context)]
    if not samples or any(item <= 0 for item in samples):
        raise ValueError(f"{context} must contain positive timings")
    return samples


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read benchmark JSON {path}") from exc
    return _mapping(value, str(path))


def _read_fixture(path: Path) -> tuple[Mapping[str, Any], float, float, int]:
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            manifest_value = json.loads(metadata["manifest"])
            manifest = _mapping(manifest_value, f"{path} manifest")
            required = {"fc1_pair_modes", "fc2_pair_modes", "topk_ids"}
            missing = sorted(required - set(handle.keys()))
            if missing:
                raise ValueError(f"fixture {path} is missing tensors {missing}")
            fc1 = handle.get_tensor("fc1_pair_modes")
            fc2 = handle.get_tensor("fc2_pair_modes")
            ids = handle.get_tensor("topk_ids")
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read route fixture {path}") from exc

    if (
        manifest.get("kind") != FIXTURE_KIND
        or manifest.get("schema_version") != FIXTURE_SCHEMA_VERSION
    ):
        raise ValueError(f"fixture {path} has an unsupported schema")
    for name, modes in (("fc1", fc1), ("fc2", fc2)):
        if modes.dtype != torch.uint8 or tuple(modes.shape) != (NUM_EXPERTS,):
            raise ValueError(f"fixture {path} {name} modes have the wrong shape")
        if not bool(torch.all((modes == 0) | (modes == 1))):
            raise ValueError(f"fixture {path} {name} modes are not P33/P24 flags")
    if ids.dtype != torch.int32 or ids.ndim != 2 or ids.shape[1] != TOP_K:
        raise ValueError(f"fixture {path} routes have the wrong contract")
    if not ids.numel() or int(ids.min()) < 0 or int(ids.max()) >= NUM_EXPERTS:
        raise ValueError(f"fixture {path} routes contain invalid expert IDs")
    routed = ids.to(torch.int64)
    fc1_fraction = float(fc1[routed].to(torch.float64).mean())
    fc2_fraction = float(fc2[routed].to(torch.float64).mean())
    static_p24 = int(fc1.to(torch.int64).sum() + fc2.to(torch.int64).sum())
    return manifest, fc1_fraction, fc2_fraction, static_p24


def _paired_ratio_bootstrap(
    samples: Sequence[float],
    baseline: Sequence[float],
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    if len(samples) != len(baseline) or not samples:
        raise ValueError("paired samples must have the same nonzero length")
    sample_tensor = torch.tensor(samples, dtype=torch.float64)
    baseline_tensor = torch.tensor(baseline, dtype=torch.float64)
    if bool(torch.any(baseline_tensor <= 0)):
        raise ValueError("baseline timings must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(
        len(samples),
        (replicates, len(samples)),
        dtype=torch.int64,
        generator=generator,
    )
    sample_medians = torch.quantile(sample_tensor[indices], 0.5, dim=1)
    baseline_medians = torch.quantile(baseline_tensor[indices], 0.5, dim=1)
    ratios = sample_medians / baseline_medians
    bounds = torch.quantile(
        ratios,
        torch.tensor((0.025, 0.975), dtype=torch.float64),
    )
    return {
        "point_estimate": float(torch.quantile(sample_tensor, 0.5))
        / float(torch.quantile(baseline_tensor, 0.5)),
        "bootstrap_ci95_low": float(bounds[0]),
        "bootstrap_ci95_high": float(bounds[1]),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }


def _identity_tuple(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    candidate_pool = manifest.get("candidate_pool")
    logical_schema = manifest.get("candidate_logical_trellis_schema")
    capture = manifest.get("capture")
    for value, context in (
        (candidate_pool, "fixture candidate pool"),
        (logical_schema, "fixture logical trellis schema"),
        (capture, "fixture capture"),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{context} must be a nonempty string")
    return (
        candidate_pool,
        _sha256_value(
            manifest.get("candidate_manifest_sha256"),
            "fixture candidate manifest digest",
        ),
        logical_schema,
        capture,
        _sha256_value(
            manifest.get("capture_manifest_sha256"),
            "fixture capture manifest digest",
        ),
    )


def _check_true(
    value: object,
    *,
    reason: str,
    failures: list[str],
) -> None:
    if value is not True:
        failures.append(reason)


def select_tp12_performance_layers(
    runtime_mix: Mapping[str, Any],
    *,
    count: int = 5,
    require_all_layers: bool = True,
) -> tuple[int, ...]:
    """Choose natural-route hotspots while retaining depth coverage."""

    if (
        runtime_mix.get("kind") != RUNTIME_MIX_KIND
        or runtime_mix.get("schema_version") != RUNTIME_MIX_SCHEMA_VERSION
    ):
        raise ValueError("unsupported TP12 runtime-mix report")
    if runtime_mix.get("candidate_pool_sealed") is not True:
        raise ValueError("performance layers must be selected from a sealed pool")
    if isinstance(count, bool) or not isinstance(count, int) or not 4 <= count <= 92:
        raise ValueError("performance layer count must be in 4..92")
    raw_layers = _list(runtime_mix.get("layers"), "runtime-mix layers")
    layers = tuple(int(value) for value in raw_layers)
    if len(set(layers)) != len(layers) or any(not 1 <= layer <= 92 for layer in layers):
        raise ValueError("runtime-mix layers are invalid")
    if require_all_layers and set(layers) != set(range(1, 93)):
        raise ValueError("final performance selection requires all 92 layers")
    if len(layers) < count:
        raise ValueError("runtime mix has fewer layers than requested")
    per_layer = _mapping(runtime_mix.get("per_layer"), "runtime-mix per-layer data")
    if set(per_layer) != {str(layer) for layer in layers}:
        raise ValueError("runtime-mix per-layer coverage disagrees")
    scores: dict[int, float] = {}
    for layer in layers:
        entry = _mapping(per_layer[str(layer)], f"runtime-mix layer {layer}")
        if entry.get("layer") != layer:
            raise ValueError(f"runtime-mix layer {layer} identity disagrees")
        score = _finite_float(
            entry.get("p24_rank_route_fraction"),
            f"runtime-mix layer {layer} P24 route fraction",
        )
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"runtime-mix layer {layer} route fraction is invalid")
        scores[layer] = score
    if not any(value > 0 for value in scores.values()):
        raise ValueError("runtime mix never exercises a natural P24 route")

    selected: set[int] = set()
    # One winner from each quarter prevents a single hot depth region from
    # dominating the benchmark matrix. Ties prefer the lower layer.
    for quarter in range(4):
        candidates = [
            layer
            for layer in layers
            if min(3, (layer - 1) * 4 // 92) == quarter
        ]
        if not candidates:
            raise ValueError(f"runtime mix has no layer in depth quarter {quarter}")
        selected.add(max(candidates, key=lambda layer: (scores[layer], -layer)))
    for layer in sorted(layers, key=lambda value: (-scores[value], value)):
        if len(selected) >= count:
            break
        selected.add(layer)
    if len(selected) != count:
        raise AssertionError("TP12 performance layer selection did not close")
    return tuple(sorted(selected))


def summarize_tp12_isolated_performance(
    benchmark_path: str | Path,
    *,
    expected_rows: Sequence[int] = (1, 2, 4, 8),
    expected_codebook: str | None = None,
    thresholds: TP12PerformanceThresholds = TP12PerformanceThresholds(),
) -> dict[str, object]:
    """Gate the isolated N-axis and K-axis P24 decoder against P33."""

    thresholds.validate()
    path = Path(benchmark_path).resolve()
    rows_expected = tuple(int(value) for value in expected_rows)
    if not rows_expected or len(set(rows_expected)) != len(rows_expected):
        raise ValueError("expected isolated rows must be nonempty and unique")
    if any(rows <= 0 for rows in rows_expected):
        raise ValueError("expected isolated rows must be positive")

    document = _load_json(path)
    if (
        document.get("kind") != ISOLATED_BENCHMARK_KIND
        or document.get("schema_version") != ISOLATED_BENCHMARK_SCHEMA_VERSION
    ):
        raise ValueError(f"{path} is not a supported isolated benchmark")
    provenance = _mapping(document.get("provenance"), f"{path} provenance")
    source_sha256 = _sha256_value(
        provenance.get("source_sha256"), f"{path} B12X source digest"
    )
    contract = _mapping(document.get("contract"), f"{path} contract")
    expected_contract = {
        "tp": TP_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "local_intermediate_size": LOCAL_INTERMEDIATE_SIZE,
        "outer_hadamard_included": True,
        "cuda_graph_replay": True,
        "cuda_graph_replay_allocation_stable_required": True,
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            raise ValueError(
                f"{path} contract {key}={contract.get(key)!r}, expected {expected!r}"
            )
    codebook = contract.get("codebook")
    if expected_codebook is not None and codebook != expected_codebook:
        raise ValueError(
            f"{path} contract codebook={codebook!r}, "
            f"expected {expected_codebook!r}"
        )
    contract_rows = tuple(
        int(value) for value in _list(contract.get("rows"), f"{path} rows")
    )
    if set(contract_rows) != set(rows_expected):
        raise ValueError(f"{path} row matrix disagrees with the gate")
    device = _mapping(document.get("device"), f"{path} device")
    if device.get("capability") not in ([12, 0], [12, 1]):
        raise ValueError(f"{path} was not measured on SM120/SM121")

    raw_cases = _list(document.get("cases"), f"{path} cases")
    cases: dict[tuple[int, str], Mapping[str, Any]] = {}
    for value in raw_cases:
        case = _mapping(value, f"{path} case")
        coordinate = (int(case.get("rows", -1)), str(case.get("rate_axis", "")))
        if coordinate in cases:
            raise ValueError(f"{path} repeats isolated case {coordinate}")
        cases[coordinate] = case
    expected_coordinates = {
        (rows, axis) for rows in rows_expected for axis in ("n", "k")
    }
    if set(cases) != expected_coordinates:
        raise ValueError(f"{path} does not cover the isolated row/axis matrix")

    failures: list[str] = []
    case_reports: list[dict[str, object]] = []
    max_ci_high = -math.inf
    worst_case: dict[str, object] | None = None
    for rows, axis in sorted(expected_coordinates):
        case = cases[(rows, axis)]
        raw_hot = _mapping(case.get("raw_hot_us"), f"{path} raw hot timings")
        p24 = _positive_samples(raw_hot.get("P24"), f"{path} P24 timings")
        p33 = _positive_samples(raw_hot.get("P33"), f"{path} P33 timings")
        if len(p24) != len(p33):
            raise ValueError(f"{path} rows={rows}/axis={axis} timings are unpaired")
        if len(p24) < thresholds.min_hot_replays:
            raise ValueError(
                f"{path} rows={rows}/axis={axis} has only {len(p24)} hot replays"
            )
        seed = (
            thresholds.bootstrap_seed
            + rows * 100
            + (0 if axis == "n" else 1)
        )
        ratio = _paired_ratio_bootstrap(
            p24,
            p33,
            replicates=thresholds.bootstrap_replicates,
            seed=seed,
        )
        point = float(ratio["point_estimate"])
        ci_high = float(ratio["bootstrap_ci95_high"])
        reported = _finite_float(
            case.get("hot_p24_over_p33"),
            f"{path} reported rows={rows}/axis={axis} ratio",
        )
        if not math.isclose(reported, point, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                f"{path} rows={rows}/axis={axis} reported ratio disagrees "
                "with raw timings"
            )
        coordinate_label = f"rows={rows}/axis={axis}"
        case_failures: list[str] = []
        correctness = _mapping(
            case.get("correctness"), f"{path} {coordinate_label} correctness"
        )
        for codec in ("P24", "P33", "K3"):
            codec_correctness = _mapping(
                correctness.get(codec),
                f"{path} {coordinate_label}/{codec} correctness",
            )
            for key in ("passed", "finite", "eager_graph_bit_exact"):
                _check_true(
                    codec_correctness.get(key),
                    reason=f"{coordinate_label}/{codec}: {key} failed",
                    failures=case_failures,
                )
        allocation = _mapping(
            case.get("graph_replay_allocation"),
            f"{path} {coordinate_label} graph allocation",
        )
        for key in ("allocated_stable", "reserved_stable"):
            _check_true(
                allocation.get(key),
                reason=f"{coordinate_label}: graph {key} failed",
                failures=case_failures,
            )
        if ci_high > thresholds.max_isolated_p24_ci95_high_over_p33:
            case_failures.append(
                f"{coordinate_label}: P24/P33 CI95 high {ci_high:.6f} exceeds "
                f"{thresholds.max_isolated_p24_ci95_high_over_p33:.6f}"
            )
        failures.extend(case_failures)
        if ci_high > max_ci_high:
            max_ci_high = ci_high
            worst_case = {"rows": rows, "rate_axis": axis}
        case_reports.append(
            {
                "rows": rows,
                "rate_axis": axis,
                "hot_replays": len(p24),
                "p24_over_p33": ratio,
                "passed": not case_failures,
                "failures": case_failures,
            }
        )

    status = "pass" if not failures else "fail"
    return {
        "kind": ISOLATED_KIND,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "thresholds": asdict(thresholds),
        "input": {
            "benchmark": str(path),
            "benchmark_sha256": _sha256(path),
            "benchmark_source_sha256": source_sha256,
            "device_name": device.get("name"),
            "device_uuid": device.get("uuid"),
            "codebook": codebook,
        },
        "scope": {
            "rows": list(rows_expected),
            "rate_axes": ["n", "k"],
            "evaluated_cases": len(case_reports),
        },
        "decision": {
            "verdict": status,
            "failures": failures,
            "maximum_p24_ci95_high_over_p33": max_ci_high,
            "worst_case": worst_case,
        },
        "cases": case_reports,
    }


def summarize_tp12_routed_performance(
    benchmark_paths: Sequence[str | Path],
    *,
    expected_layers: Sequence[int],
    expected_tokens: Sequence[int] = (1, 4, 16),
    expected_codebook: str | None = None,
    thresholds: TP12PerformanceThresholds = TP12PerformanceThresholds(),
    require_sealed_pool: bool = True,
) -> dict[str, object]:
    """Authenticate and gate a complete logical-TP12 benchmark matrix."""

    thresholds.validate()
    layers = tuple(int(value) for value in expected_layers)
    tokens_expected = tuple(int(value) for value in expected_tokens)
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("expected layers must be nonempty and unique")
    if any(layer < 1 or layer > 92 for layer in layers):
        raise ValueError("expected layers must be in 1..92")
    if not tokens_expected or len(set(tokens_expected)) != len(tokens_expected):
        raise ValueError("expected tokens must be nonempty and unique")
    if any(tokens <= 0 for tokens in tokens_expected):
        raise ValueError("expected tokens must be positive")

    paths = tuple(Path(value).resolve() for value in benchmark_paths)
    required_count = len(layers) * TP_SIZE
    if len(paths) != required_count:
        raise ValueError(
            f"expected {required_count} benchmark files for complete TP12 coverage, "
            f"got {len(paths)}"
        )

    seen: dict[tuple[int, int], Path] = {}
    identity: tuple[object, ...] | None = None
    benchmark_source_sha256: str | None = None
    benchmark_codebook: object = None
    failures: list[str] = []
    case_reports: list[dict[str, object]] = []
    input_reports: list[dict[str, object]] = []
    total_static_p24 = 0
    total_p24_route_fraction = 0.0
    max_median = -math.inf
    max_ci_high = -math.inf
    worst_median_case: dict[str, int] | None = None
    worst_ci_case: dict[str, int] | None = None

    for path in paths:
        document = _load_json(path)
        if (
            document.get("kind") != BENCHMARK_KIND
            or document.get("schema_version") != BENCHMARK_SCHEMA_VERSION
        ):
            raise ValueError(f"{path} is not a supported routed benchmark")
        provenance = _mapping(document.get("provenance"), f"{path} provenance")
        fixture_info = _mapping(
            provenance.get("route_fixture"), f"{path} route fixture"
        )
        fixture_path = Path(str(fixture_info.get("path", ""))).resolve()
        if not fixture_path.is_file():
            raise ValueError(f"{path} route fixture does not exist: {fixture_path}")
        fixture_sha256 = _sha256(fixture_path)
        if fixture_info.get("sha256") != fixture_sha256:
            raise ValueError(f"{path} route fixture SHA-256 disagrees")
        fixture_manifest, fc1_fraction, fc2_fraction, static_p24 = _read_fixture(
            fixture_path
        )
        claimed_manifest = _mapping(
            fixture_info.get("manifest"), f"{path} claimed fixture manifest"
        )
        for key, value in fixture_manifest.items():
            if claimed_manifest.get(key) != value:
                raise ValueError(f"{path} fixture manifest disagrees at {key}")
        if require_sealed_pool and fixture_manifest.get("candidate_pool_sealed") is not True:
            raise ValueError(f"{path} fixture was not exported from a sealed pool")
        fixture_identity = _identity_tuple(fixture_manifest)
        if identity is None:
            identity = fixture_identity
        elif fixture_identity != identity:
            raise ValueError(f"{path} fixture provenance differs from the other ranks")

        source_sha256 = _sha256_value(
            provenance.get("source_sha256"), f"{path} B12X source digest"
        )
        if benchmark_source_sha256 is None:
            benchmark_source_sha256 = source_sha256
        elif source_sha256 != benchmark_source_sha256:
            raise ValueError(f"{path} used a different B12X benchmark source")

        layer = int(fixture_manifest.get("layer", -1))
        rank = int(fixture_manifest.get("rank", -1))
        coordinate = (layer, rank)
        if layer not in layers or not 0 <= rank < TP_SIZE:
            raise ValueError(f"{path} has unexpected layer/rank {coordinate}")
        if coordinate in seen:
            raise ValueError(
                f"duplicate layer/rank benchmark {coordinate}: {seen[coordinate]}, {path}"
            )
        seen[coordinate] = path
        total_static_p24 += static_p24
        total_p24_route_fraction += fc1_fraction + fc2_fraction

        contract = _mapping(document.get("contract"), f"{path} contract")
        expected_contract = {
            "tp": TP_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "local_intermediate_size": LOCAL_INTERMEDIATE_SIZE,
            "route_experts": NUM_EXPERTS,
            "topk": TOP_K,
            "materialized_experts": NUM_EXPERTS,
            "route_source": "captured_fixture",
            "timed_route_window_rotation": True,
            "route_copy_in_timed_region": False,
            "complete_routed_pipeline": True,
            "outer_hadamard_included": True,
            "cuda_graph_replay": True,
            "cuda_graph_replay_allocation_stable_required": True,
        }
        for key, expected in expected_contract.items():
            if contract.get(key) != expected:
                raise ValueError(
                    f"{path} contract {key}={contract.get(key)!r}, expected {expected!r}"
                )
        codebook = contract.get("codebook")
        if expected_codebook is not None and codebook != expected_codebook:
            raise ValueError(
                f"{path} contract codebook={codebook!r}, "
                f"expected {expected_codebook!r}"
            )
        if benchmark_codebook is None:
            benchmark_codebook = codebook
        elif codebook != benchmark_codebook:
            raise ValueError(f"{path} used a different trellis codebook")
        scenarios = set(_list(contract.get("scenarios"), f"{path} scenarios"))
        if not {"P33", "sparse"}.issubset(scenarios):
            raise ValueError(f"{path} did not benchmark P33 and sparse")
        contract_tokens = tuple(int(value) for value in _list(
            contract.get("tokens"), f"{path} contract tokens"
        ))
        if set(contract_tokens) != set(tokens_expected):
            raise ValueError(f"{path} token matrix disagrees with the gate")
        device = _mapping(document.get("device"), f"{path} device")
        if device.get("capability") not in ([12, 0], [12, 1]):
            raise ValueError(f"{path} was not measured on SM120/SM121")

        cases = _list(document.get("cases"), f"{path} cases")
        cases_by_tokens: dict[int, Mapping[str, Any]] = {}
        for raw_case in cases:
            case = _mapping(raw_case, f"{path} case")
            token_count = int(case.get("tokens", -1))
            if token_count in cases_by_tokens:
                raise ValueError(f"{path} repeats tokens={token_count}")
            cases_by_tokens[token_count] = case
        if set(cases_by_tokens) != set(tokens_expected):
            raise ValueError(f"{path} cases do not cover the requested token matrix")

        input_reports.append(
            {
                "benchmark": str(path),
                "benchmark_sha256": _sha256(path),
                "fixture": str(fixture_path),
                "fixture_sha256": fixture_sha256,
                "layer": layer,
                "rank": rank,
                "device_name": device.get("name"),
                "device_uuid": device.get("uuid"),
                "codebook": codebook,
                "fc1_p24_route_fraction": fc1_fraction,
                "fc2_p24_route_fraction": fc2_fraction,
            }
        )

        for token_count in tokens_expected:
            case = cases_by_tokens[token_count]
            raw_hot = _mapping(case.get("raw_hot_us"), f"{path} raw hot timings")
            p33 = _positive_samples(raw_hot.get("P33"), f"{path} P33 timings")
            sparse = _positive_samples(
                raw_hot.get("sparse"), f"{path} sparse timings"
            )
            if len(p33) != len(sparse):
                raise ValueError(f"{path} tokens={token_count} timings are unpaired")
            if len(p33) < thresholds.min_hot_replays:
                raise ValueError(
                    f"{path} tokens={token_count} has only {len(p33)} hot replays"
                )
            bootstrap_seed = (
                thresholds.bootstrap_seed
                + layer * 100_000
                + rank * 1_000
                + token_count
            )
            ratio = _paired_ratio_bootstrap(
                sparse,
                p33,
                replicates=thresholds.bootstrap_replicates,
                seed=bootstrap_seed,
            )
            point = float(ratio["point_estimate"])
            ci_high = float(ratio["bootstrap_ci95_high"])
            coordinate_label = f"layer={layer}/rank={rank}/tokens={token_count}"

            reported_ratios = _mapping(
                case.get("hot_median_over_baseline"),
                f"{path} reported hot ratios",
            )
            reported_sparse = _finite_float(
                reported_ratios.get("sparse"), f"{path} reported sparse ratio"
            )
            if not math.isclose(reported_sparse, point, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(
                    f"{path} tokens={token_count} reported median ratio disagrees "
                    "with raw timings"
                )

            mode_summary = _mapping(
                case.get("mode_summary"), f"{path} mode summary"
            )
            p33_modes = _mapping(mode_summary.get("P33"), f"{path} P33 modes")
            sparse_modes = _mapping(
                mode_summary.get("sparse"), f"{path} sparse modes"
            )
            for key in ("routed_fc1_p24_fraction", "routed_fc2_p24_fraction"):
                if _finite_float(p33_modes.get(key), f"{path} P33 {key}") != 0.0:
                    raise ValueError(f"{path} P33 control contains P24 routes")
            measured_fc1 = _finite_float(
                sparse_modes.get("routed_fc1_p24_fraction"),
                f"{path} sparse FC1 route fraction",
            )
            measured_fc2 = _finite_float(
                sparse_modes.get("routed_fc2_p24_fraction"),
                f"{path} sparse FC2 route fraction",
            )
            if not math.isclose(measured_fc1, fc1_fraction, abs_tol=1e-12):
                raise ValueError(f"{path} sparse FC1 route fraction disagrees")
            if not math.isclose(measured_fc2, fc2_fraction, abs_tol=1e-12):
                raise ValueError(f"{path} sparse FC2 route fraction disagrees")

            case_failures: list[str] = []
            correctness = _mapping(
                case.get("correctness"), f"{path} correctness"
            )
            kernels = _mapping(case.get("kernel_summary"), f"{path} kernels")
            for scenario in ("P33", "sparse"):
                scenario_correctness = _mapping(
                    correctness.get(scenario),
                    f"{path} {scenario} correctness",
                )
                for key in ("passed", "finite", "eager_graph_bit_exact"):
                    _check_true(
                        scenario_correctness.get(key),
                        reason=f"{coordinate_label}/{scenario}: {key} failed",
                        failures=case_failures,
                    )
                kernel = _mapping(kernels.get(scenario), f"{path} {scenario} kernel")
                _check_true(
                    kernel.get("resource_introspection_available"),
                    reason=f"{coordinate_label}/{scenario}: resources unavailable",
                    failures=case_failures,
                )
                if int(kernel.get("local_memory_bytes_per_thread", -1)) != 0:
                    case_failures.append(
                        f"{coordinate_label}/{scenario}: local-memory spill"
                    )
            allocation = _mapping(
                case.get("graph_replay_allocation"), f"{path} graph allocation"
            )
            for key in ("allocated_stable", "reserved_stable"):
                _check_true(
                    allocation.get(key),
                    reason=f"{coordinate_label}: graph {key} failed",
                    failures=case_failures,
                )
            if point > thresholds.max_sparse_median_over_p33:
                case_failures.append(
                    f"{coordinate_label}: sparse/P33 median {point:.6f} exceeds "
                    f"{thresholds.max_sparse_median_over_p33:.6f}"
                )
            if ci_high > thresholds.max_sparse_ci95_high_over_p33:
                case_failures.append(
                    f"{coordinate_label}: sparse/P33 CI95 high {ci_high:.6f} exceeds "
                    f"{thresholds.max_sparse_ci95_high_over_p33:.6f}"
                )
            failures.extend(case_failures)
            if point > max_median:
                max_median = point
                worst_median_case = {
                    "layer": layer,
                    "rank": rank,
                    "tokens": token_count,
                }
            if ci_high > max_ci_high:
                max_ci_high = ci_high
                worst_ci_case = {
                    "layer": layer,
                    "rank": rank,
                    "tokens": token_count,
                }
            case_reports.append(
                {
                    "layer": layer,
                    "rank": rank,
                    "tokens": token_count,
                    "hot_replays": len(p33),
                    "sparse_over_p33": ratio,
                    "passed": not case_failures,
                    "failures": case_failures,
                }
            )

    missing = sorted(
        (layer, rank)
        for layer in layers
        for rank in range(TP_SIZE)
        if (layer, rank) not in seen
    )
    if missing:
        raise ValueError(f"missing logical TP12 benchmark coordinates: {missing}")
    if total_static_p24 <= 0 or total_p24_route_fraction <= 0:
        raise ValueError("benchmark matrix never exercised a P24 expert or route")
    assert identity is not None
    assert benchmark_source_sha256 is not None
    status = "pass" if not failures else "fail"
    return {
        "kind": ROUTED_KIND,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "interpretation": (
            "production-frequency TP12 routed-kernel gate; route fixtures are "
            "read-only performance inputs and do not select modes or keep tiers"
        ),
        "thresholds": asdict(thresholds),
        "scope": {
            "layers": list(layers),
            "logical_ranks": list(range(TP_SIZE)),
            "tokens": list(tokens_expected),
            "benchmark_files": len(paths),
            "evaluated_cases": len(case_reports),
            "require_sealed_pool": require_sealed_pool,
        },
        "identity": {
            "candidate_pool": identity[0],
            "candidate_manifest_sha256": identity[1],
            "candidate_logical_trellis_schema": identity[2],
            "capture": identity[3],
            "capture_manifest_sha256": identity[4],
            "benchmark_source_sha256": benchmark_source_sha256,
            "codebook": benchmark_codebook,
        },
        "decision": {
            "verdict": status,
            "failures": failures,
            "maximum_sparse_median_over_p33": max_median,
            "maximum_sparse_ci95_high_over_p33": max_ci_high,
            "worst_median_case": worst_median_case,
            "worst_ci95_high_case": worst_ci_case,
        },
        "inputs": sorted(
            input_reports,
            key=lambda item: (int(item["layer"]), int(item["rank"])),
        ),
        "cases": sorted(
            case_reports,
            key=lambda item: (
                int(item["layer"]),
                int(item["rank"]),
                int(item["tokens"]),
            ),
        ),
    }


def summarize_tp12_performance(
    isolated_benchmark_path: str | Path,
    routed_benchmark_paths: Sequence[str | Path],
    *,
    expected_layers: Sequence[int],
    expected_isolated_rows: Sequence[int] = (1, 2, 4, 8),
    expected_tokens: Sequence[int] = (1, 4, 16),
    expected_codebook: str | None = None,
    thresholds: TP12PerformanceThresholds = TP12PerformanceThresholds(),
    require_sealed_pool: bool = True,
) -> dict[str, object]:
    """Build one production verdict from isolated and natural-route evidence."""

    isolated = summarize_tp12_isolated_performance(
        isolated_benchmark_path,
        expected_rows=expected_isolated_rows,
        expected_codebook=expected_codebook,
        thresholds=thresholds,
    )
    routed = summarize_tp12_routed_performance(
        routed_benchmark_paths,
        expected_layers=expected_layers,
        expected_tokens=expected_tokens,
        expected_codebook=expected_codebook,
        thresholds=thresholds,
        require_sealed_pool=require_sealed_pool,
    )
    failures: list[str] = []
    for label, report in (("isolated", isolated), ("routed", routed)):
        if report["status"] != "pass":
            decision = _mapping(report.get("decision"), f"{label} decision")
            component_failures = _list(
                decision.get("failures"), f"{label} decision failures"
            )
            failures.extend(f"{label}: {failure}" for failure in component_failures)
    status = "pass" if not failures else "fail"
    return {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "interpretation": (
            "TP12 production performance acceptance requires both the isolated "
            "P24 decoder matrix and the complete natural-route logical-rank "
            "matrix to pass their frozen thresholds"
        ),
        "thresholds": asdict(thresholds),
        "identity": routed["identity"],
        "scope": {
            "isolated": isolated["scope"],
            "routed": routed["scope"],
        },
        "decision": {
            "verdict": status,
            "failures": failures,
            "isolated_verdict": isolated["status"],
            "routed_verdict": routed["status"],
        },
        "isolated": isolated,
        "routed": routed,
    }


def validate_tp12_performance_report(
    report_path: str | Path,
    *,
    expected_candidate_pool: str | Path | None = None,
    require_pass: bool = True,
) -> dict[str, object]:
    """Recompute a frozen production report from all of its named evidence."""

    path = Path(report_path).resolve()
    document = _load_json(path)
    if (
        document.get("kind") != KIND
        or document.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError(f"{path} is not a supported TP12 performance report")
    thresholds = TP12PerformanceThresholds()
    if document.get("thresholds") != asdict(thresholds):
        raise ValueError(f"{path} does not use the frozen production thresholds")

    isolated = _mapping(document.get("isolated"), f"{path} isolated report")
    isolated_input = _mapping(
        isolated.get("input"), f"{path} isolated benchmark input"
    )
    isolated_scope = _mapping(
        isolated.get("scope"), f"{path} isolated benchmark scope"
    )
    routed = _mapping(document.get("routed"), f"{path} routed report")
    routed_scope = _mapping(routed.get("scope"), f"{path} routed scope")
    routed_inputs = _list(routed.get("inputs"), f"{path} routed inputs")
    identity = _mapping(document.get("identity"), f"{path} identity")
    expected_codebook = identity.get("codebook")
    if expected_codebook is not None and not isinstance(expected_codebook, str):
        raise ValueError(f"{path} identity codebook must be a string")
    benchmark_paths = [
        str(_mapping(value, f"{path} routed input").get("benchmark", ""))
        for value in routed_inputs
    ]
    recomputed = summarize_tp12_performance(
        str(isolated_input.get("benchmark", "")),
        benchmark_paths,
        expected_layers=[
            int(value)
            for value in _list(routed_scope.get("layers"), f"{path} layers")
        ],
        expected_isolated_rows=[
            int(value)
            for value in _list(
                isolated_scope.get("rows"), f"{path} isolated rows"
            )
        ],
        expected_tokens=[
            int(value)
            for value in _list(routed_scope.get("tokens"), f"{path} tokens")
        ],
        expected_codebook=expected_codebook,
        thresholds=thresholds,
        require_sealed_pool=True,
    )
    if recomputed != document:
        raise ValueError(f"{path} is stale or disagrees with its raw evidence")
    if require_pass and document.get("status") != "pass":
        raise ValueError(f"{path} did not pass the TP12 performance gate")

    candidate_pool = Path(str(identity.get("candidate_pool", ""))).resolve()
    if expected_candidate_pool is not None and candidate_pool != Path(
        expected_candidate_pool
    ).resolve():
        raise ValueError(
            f"{path} candidate pool {candidate_pool} does not match "
            f"{Path(expected_candidate_pool).resolve()}"
        )
    manifest_path = candidate_pool / "mixed_exl3_candidate_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"{path} candidate manifest is unavailable")
    manifest_sha256 = _sha256(manifest_path)
    if identity.get("candidate_manifest_sha256") != manifest_sha256:
        raise ValueError(f"{path} candidate manifest digest has drifted")
    return {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "status": str(document["status"]),
        "summary_sha256": _sha256(path),
        "candidate_pool": str(candidate_pool),
        "candidate_manifest_sha256": manifest_sha256,
        "candidate_logical_trellis_schema": identity.get(
            "candidate_logical_trellis_schema"
        ),
        "codebook": identity.get("codebook"),
        "isolated_cases": isolated_scope.get("evaluated_cases"),
        "routed_cases": routed_scope.get("evaluated_cases"),
        "routed_layers": routed_scope.get("layers"),
        "logical_ranks": routed_scope.get("logical_ranks"),
    }


__all__ = [
    "BENCHMARK_KIND",
    "BENCHMARK_SCHEMA_VERSION",
    "ISOLATED_BENCHMARK_KIND",
    "ISOLATED_BENCHMARK_SCHEMA_VERSION",
    "ISOLATED_KIND",
    "KIND",
    "ROUTED_KIND",
    "RUNTIME_MIX_KIND",
    "RUNTIME_MIX_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "TP12PerformanceThresholds",
    "summarize_tp12_isolated_performance",
    "summarize_tp12_performance",
    "summarize_tp12_routed_performance",
    "select_tp12_performance_layers",
    "validate_tp12_performance_report",
]
