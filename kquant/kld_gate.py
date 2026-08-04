"""Integrity checks and paired statistics for the final Kimi-K3 KLD gate.

The external KLD tools produce one comparison against the original MXFP4
reference at a time.  This module validates that two such reports describe the
same token windows and computes the paired question that matters for codec
promotion: does the candidate reduce reference KLD relative to a named
baseline on those exact windows?

Window resampling measures corpus uncertainty.  It does not measure serving
runtime nondeterminism, so the latter remains a separate decision field.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np


KLD_GATE_KIND = "kquant_kimi_k3_paired_kld_gate"
KLD_GATE_SCHEMA_VERSION = 1
KLD_DIRECTION = "KL(reference || candidate)"
CANONICAL_SUITE_TOKEN_HASH = (
    "a6856e1d0504fd00d13c67a5515c081f349088664d7ea0894dc4d15db2c7d209"
)
CANONICAL_REFERENCE_DATASET = (
    "festr2/kimi-k3-full-mxfp4-kld-reference-32x2048"
)
CANONICAL_REFERENCE_DATASET_REVISION = (
    "097b2775900c0940d31c6469c2e930be8d17b2f8"
)
CANONICAL_EVALUATION_TOOLS_COMMIT = (
    "ec33b89f3c3ef74557d01897587d9045d3317cc4"
)
CANONICAL_SUITE_MANIFEST_SHA256 = (
    "2112611fa037cb3266c115b26f3759d92f5d7b3c24892ca7f058fec05514acf0"
)
CANONICAL_REFERENCE_MANIFEST_SHA256 = (
    "64806a05af1d595d0194da01f38b314906df4b65470b374b5399bf30e7f32883"
)
CANONICAL_TOOL_SHA256 = {
    "capture-kimi-k3-kld-suite.py": (
        "33779ad4e3d531fea73286d3ec4921923f521ed1e7d08403dcd3484b97e86f62"
    ),
    "compare-kimi-k3-kld-suite.py": (
        "5e091c003e5c3b8955be1a0f626cc1cfca06ae870ea968725c7a518aa3ba08ec"
    ),
    "finalize-kimi-k3-kld-suite.py": (
        "63311b01d54378cbedc59dae48910142306e79c3a451be39f4f6e87bcbe75822"
    ),
    "prepare-kimi-k3-kld-suite.py": (
        "1c52176b3dc13e5d810ed26a0f2c04be29fac0dc9822864d2a34ce8b88c8bc43"
    ),
    "vllm-kld-capture.patch": (
        "13d7f7bba21f07472f9ea373b957587155f4f3431527d14597d4dc61a1048265"
    ),
}
DOCUMENTED_RUNTIME_NOISE_KL = 0.004


@dataclass(frozen=True)
class SuiteWindow:
    index: int
    domain: str
    positions: int
    token_ids_json_sha256: str
    token_file: str


@dataclass(frozen=True)
class ComparisonWindow:
    index: int
    domain: str
    positions: int
    kl_mean: float
    js_mean: float
    top1_agreement: float


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _finite_float(
    value: Any,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{label} must be <= {maximum}")
    return result


def _sha256(value: str, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA-256")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{label} is not hexadecimal") from exc
    return value.lower()


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be an array")
    return value


def _safe_relative_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must not escape the suite root")
    return value


def validate_suite_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_suite_hash: str = CANONICAL_SUITE_TOKEN_HASH,
) -> tuple[str, tuple[SuiteWindow, ...]]:
    """Validate the suite's internal contract and return ordered windows."""

    expected_suite_hash = _sha256(
        expected_suite_hash, label="expected suite token hash"
    )
    suite_hash = _sha256(
        manifest.get("suite_token_hash_sha256"), label="suite token hash"
    )
    if suite_hash != expected_suite_hash:
        raise ValueError(
            f"suite token hash {suite_hash} != expected {expected_suite_hash}"
        )
    context_length = _integer(
        manifest.get("context_length"), label="suite context length", minimum=2
    )
    raw_windows = _sequence(manifest.get("windows"), label="suite windows")
    declared_count = _integer(
        manifest.get("window_count"), label="suite window count", minimum=1
    )
    if len(raw_windows) != declared_count:
        raise ValueError("suite window_count does not match its window records")

    windows: list[SuiteWindow] = []
    token_hashes: list[str] = []
    domain_counts: dict[str, int] = {}
    for position, raw in enumerate(raw_windows):
        item = _mapping(raw, label=f"suite window {position}")
        index = _integer(item.get("index"), label="suite window index")
        if index != position:
            raise ValueError("suite windows must be ordered and zero-based")
        domain = item.get("domain")
        if not isinstance(domain, str) or not domain:
            raise ValueError(f"suite window {index} has an invalid domain")
        num_tokens = _integer(
            item.get("num_tokens"), label=f"suite window {index} token count"
        )
        if num_tokens != context_length:
            raise ValueError(
                f"suite window {index} has {num_tokens} tokens; "
                f"expected {context_length}"
            )
        token_hash = _sha256(
            item.get("token_ids_json_sha256"),
            label=f"suite window {index} token hash",
        )
        token_file = _safe_relative_path(
            item.get("token_file"), label=f"suite window {index} token file"
        )
        windows.append(
            SuiteWindow(
                index=index,
                domain=domain,
                positions=context_length - 1,
                token_ids_json_sha256=token_hash,
                token_file=token_file,
            )
        )
        token_hashes.append(token_hash)
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    derived_hash = hashlib.sha256("\n".join(token_hashes).encode("ascii")).hexdigest()
    if derived_hash != suite_hash:
        raise ValueError("suite token hash does not match its window hash sequence")
    if _integer(
        manifest.get("scored_positions_per_window"),
        label="suite scored positions per window",
    ) != context_length - 1:
        raise ValueError("suite scored_positions_per_window is inconsistent")
    if _integer(
        manifest.get("total_scored_positions"),
        label="suite total scored positions",
    ) != declared_count * (context_length - 1):
        raise ValueError("suite total_scored_positions is inconsistent")
    declared_domains = _mapping(
        manifest.get("domain_counts"), label="suite domain counts"
    )
    normalized_domains = {
        str(domain): _integer(count, label=f"domain count {domain}")
        for domain, count in declared_domains.items()
    }
    if normalized_domains != domain_counts:
        raise ValueError("suite domain_counts do not match its windows")
    return suite_hash, tuple(windows)


def validate_token_ids(
    token_ids: Any,
    *,
    window: SuiteWindow,
) -> None:
    values = _sequence(token_ids, label=f"window {window.index} token IDs")
    if len(values) != window.positions + 1:
        raise ValueError(f"window {window.index} token count is incorrect")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError(f"window {window.index} token IDs must be non-negative integers")
    compact = json.dumps(list(values), separators=(",", ":"))
    actual = hashlib.sha256(compact.encode("utf-8")).hexdigest()
    if actual != window.token_ids_json_sha256:
        raise ValueError(f"window {window.index} token payload hash differs")


def select_suite_windows(
    windows: Sequence[SuiteWindow],
    *,
    start_window: int,
    stop_window: int | None,
) -> tuple[SuiteWindow, ...]:
    start = _integer(start_window, label="start window")
    stop = len(windows) if stop_window is None else _integer(
        stop_window, label="stop window", minimum=1
    )
    if start >= stop or stop > len(windows):
        raise ValueError(
            f"invalid window range [{start}, {stop}) for {len(windows)} windows"
        )
    return tuple(windows[start:stop])


def validate_logits_manifest(
    manifest: Mapping[str, Any],
    *,
    label: str,
    suite_hash: str,
    expected_windows: Sequence[SuiteWindow],
) -> tuple[Mapping[str, Any], ...]:
    """Validate a finalized logit manifest against the selected suite slice."""

    actual_hash = _sha256(
        manifest.get("suite_token_hash_sha256"), label=f"{label} suite token hash"
    )
    if actual_hash != suite_hash:
        raise ValueError(f"{label} suite token hash differs")
    raw_records = _sequence(manifest.get("windows"), label=f"{label} windows")
    if _integer(
        manifest.get("window_count"), label=f"{label} window count"
    ) != len(raw_records):
        raise ValueError(f"{label} window_count is inconsistent")
    declared_range_value = manifest.get("window_range")
    if declared_range_value is None:
        # The pinned full-MXFP4 reference predates the current finalize tool's
        # explicit window_range field. Its ordered window_index records still
        # define the range without ambiguity. New captures retain the explicit
        # field, but accepting this one older shape is required to validate the
        # canonical manifest byte-for-byte rather than rewriting it.
        if not raw_records:
            raise ValueError(f"{label} window records are empty")
        first_record = _mapping(raw_records[0], label=f"{label} first window")
        range_start = _integer(
            first_record.get("window_index"), label=f"{label} range start"
        )
        range_stop = range_start + len(raw_records)
    else:
        declared_range = list(
            _sequence(declared_range_value, label=f"{label} window range")
        )
        if len(declared_range) != 2:
            raise ValueError(f"{label} window_range is inconsistent")
        range_start = _integer(declared_range[0], label=f"{label} range start")
        range_stop = _integer(
            declared_range[1], label=f"{label} range stop", minimum=1
        )
    if range_stop - range_start != len(raw_records):
        raise ValueError(f"{label} window_range is inconsistent")
    expected_indices = {window.index for window in expected_windows}
    if not expected_indices.issubset(set(range(range_start, range_stop))):
        raise ValueError(f"{label} window records do not cover the requested range")

    expected_by_index = {window.index: window for window in expected_windows}
    records_by_index: dict[int, Mapping[str, Any]] = {}
    total_size = 0
    positions_per_window = expected_windows[0].positions
    for offset, raw in enumerate(raw_records):
        expected_index = range_start + offset
        item = _mapping(raw, label=f"{label} window {expected_index}")
        index = _integer(item.get("window_index"), label=f"{label} window index")
        if index != expected_index:
            raise ValueError(f"{label} window records are not contiguous and ordered")
        domain = item.get("domain")
        if not isinstance(domain, str) or not domain:
            raise ValueError(f"{label} window {index} has an invalid domain")
        token_hash = _sha256(
            item.get("token_ids_json_sha256"),
            label=f"{label} window {index} token hash",
        )
        shape = list(
            _sequence(item.get("shape"), label=f"{label} window {index} shape")
        )
        vocab_size = _integer(
            manifest.get("vocab_size"), label=f"{label} vocabulary size", minimum=1
        )
        if shape != [positions_per_window, vocab_size]:
            raise ValueError(f"{label} window {index} shape differs")
        if item.get("dtype") != "F32":
            raise ValueError(f"{label} window {index} must be F32")
        if item.get("key") != "logits":
            raise ValueError(f"{label} window {index} tensor key differs")
        _sha256(item.get("sha256"), label=f"{label} window {index} file hash")
        total_size += _integer(
            item.get("size_bytes"), label=f"{label} window {index} size", minimum=1
        )
        expected = expected_by_index.get(index)
        if expected is not None:
            if domain != expected.domain:
                raise ValueError(f"{label} window {index} domain differs")
            if token_hash != expected.token_ids_json_sha256:
                raise ValueError(f"{label} window {index} token hash differs")
            records_by_index[index] = item
    if _integer(
        manifest.get("total_scored_positions"),
        label=f"{label} total scored positions",
    ) != len(raw_records) * positions_per_window:
        raise ValueError(f"{label} total_scored_positions is inconsistent")
    if _integer(
        manifest.get("total_size_bytes"), label=f"{label} total size", minimum=1
    ) != total_size:
        raise ValueError(f"{label} total_size_bytes is inconsistent")
    if set(records_by_index) != expected_indices:
        raise ValueError(f"{label} window records are incomplete")
    return tuple(records_by_index[window.index] for window in expected_windows)


_DISTRIBUTION_STATISTICS = ("mean", "median", "p95", "p99", "max")


def metric_statistics(
    item: Mapping[str, Any], *, key: str, label: str
) -> dict[str, float]:
    """Validate and normalize one upstream distribution-statistics object."""

    metric = _mapping(item.get(key), label=f"{label} {key}")
    result = {
        name: _finite_float(
            metric.get(name), label=f"{label} {key} {name}", minimum=0.0
        )
        for name in _DISTRIBUTION_STATISTICS
    }
    if not (
        result["median"]
        <= result["p95"]
        <= result["p99"]
        <= result["max"]
    ):
        raise ValueError(f"{label} {key} quantiles are not ordered")
    if result["mean"] > result["max"]:
        raise ValueError(f"{label} {key} mean exceeds its maximum")
    return result


def _metric_mean(item: Mapping[str, Any], *, key: str, label: str) -> float:
    return metric_statistics(item, key=key, label=label)["mean"]


def validate_comparison_report(
    report: Mapping[str, Any],
    *,
    label: str,
    expected_windows: Sequence[SuiteWindow],
) -> tuple[ComparisonWindow, ...]:
    """Validate an upstream comparator report and return aligned windows."""

    if report.get("direction") != KLD_DIRECTION:
        raise ValueError(f"{label} has the wrong KLD direction")
    if _integer(report.get("windows"), label=f"{label} window count") != len(
        expected_windows
    ):
        raise ValueError(f"{label} window count differs")
    expected_range = [expected_windows[0].index, expected_windows[-1].index + 1]
    if list(_sequence(report.get("window_range"), label=f"{label} window range")) != expected_range:
        raise ValueError(f"{label} window range differs")
    raw_windows = _sequence(report.get("per_window"), label=f"{label} per-window data")
    if len(raw_windows) != len(expected_windows):
        raise ValueError(f"{label} per-window data are incomplete")

    windows: list[ComparisonWindow] = []
    for expected, raw in zip(expected_windows, raw_windows, strict=True):
        item = _mapping(raw, label=f"{label} window {expected.index}")
        if _integer(item.get("window_index"), label=f"{label} window index") != expected.index:
            raise ValueError(f"{label} window order differs")
        if item.get("domain") != expected.domain:
            raise ValueError(f"{label} window {expected.index} domain differs")
        positions = _integer(
            item.get("positions"), label=f"{label} window {expected.index} positions"
        )
        if positions != expected.positions:
            raise ValueError(f"{label} window {expected.index} position count differs")
        windows.append(
            ComparisonWindow(
                index=expected.index,
                domain=expected.domain,
                positions=positions,
                kl_mean=_metric_mean(
                    item,
                    key="kl_reference_to_candidate",
                    label=f"{label} window {expected.index}",
                ),
                js_mean=_metric_mean(
                    item, key="js", label=f"{label} window {expected.index}"
                ),
                top1_agreement=_finite_float(
                    item.get("top1_agreement"),
                    label=f"{label} window {expected.index} top-1 agreement",
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
        )

    positions = sum(window.positions for window in windows)
    if _integer(report.get("positions"), label=f"{label} total positions") != positions:
        raise ValueError(f"{label} total positions are inconsistent")
    cluster = _mapping(
        report.get("kl_window_cluster_bootstrap"),
        label=f"{label} clustered bootstrap",
    )
    if cluster.get("cluster_unit") != "window":
        raise ValueError(f"{label} bootstrap unit must be the window")

    weights = np.asarray([window.positions for window in windows], dtype=np.float64)
    for report_key, field in (
        ("kl_reference_to_candidate", "kl_mean"),
        ("js", "js_mean"),
    ):
        declared = _metric_mean(report, key=report_key, label=label)
        derived = float(
            np.average(
                np.asarray([getattr(window, field) for window in windows]),
                weights=weights,
            )
        )
        if not math.isclose(declared, derived, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError(f"{label} aggregate {report_key} mean is inconsistent")
    declared_top1 = _finite_float(
        report.get("top1_agreement"),
        label=f"{label} aggregate top-1 agreement",
        minimum=0.0,
        maximum=1.0,
    )
    derived_top1 = float(
        np.average(
            np.asarray([window.top1_agreement for window in windows]),
            weights=weights,
        )
    )
    if not math.isclose(declared_top1, derived_top1, rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError(f"{label} aggregate top-1 agreement is inconsistent")
    return tuple(windows)


def _bootstrap_mean(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be a finite nonempty vector")
    if isinstance(samples, bool) or samples < 100:
        raise ValueError("bootstrap samples must be at least 100")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("bootstrap seed must be an integer")
    generator = np.random.default_rng(seed)
    choices = generator.integers(0, values.size, size=(samples, values.size))
    means = values[choices].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "standard_error": float(means.std(ddof=1)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "cluster_unit": "window",
    }


def summarize_paired_kld(
    baseline: Sequence[ComparisonWindow],
    candidate: Sequence[ComparisonWindow],
    *,
    baseline_label: str,
    candidate_label: str,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 1,
    minimum_mean_improvement: float = 0.0,
    documented_runtime_noise_kl: float = DOCUMENTED_RUNTIME_NOISE_KL,
) -> dict[str, Any]:
    """Summarize paired baseline-minus-candidate KLD by independent window."""

    if not baseline_label or not candidate_label:
        raise ValueError("baseline and candidate labels must be nonempty")
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("baseline and candidate windows must be equal and nonempty")
    minimum = _finite_float(
        minimum_mean_improvement,
        label="minimum mean improvement",
        minimum=0.0,
    )
    noise = _finite_float(
        documented_runtime_noise_kl,
        label="documented runtime noise KLD",
        minimum=0.0,
    )
    for left, right in zip(baseline, candidate, strict=True):
        if (left.index, left.domain, left.positions) != (
            right.index,
            right.domain,
            right.positions,
        ):
            raise ValueError("baseline and candidate windows are not aligned")

    baseline_kl = np.asarray([window.kl_mean for window in baseline], dtype=np.float64)
    candidate_kl = np.asarray([window.kl_mean for window in candidate], dtype=np.float64)
    improvements = baseline_kl - candidate_kl
    paired = _bootstrap_mean(
        improvements, samples=bootstrap_samples, seed=bootstrap_seed
    )

    domains: dict[str, Any] = {}
    for domain in sorted({window.domain for window in baseline}):
        indices = np.asarray(
            [index for index, window in enumerate(baseline) if window.domain == domain],
            dtype=np.int64,
        )
        domain_summary = _bootstrap_mean(
            improvements[indices],
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
        domains[domain] = {
            "windows": int(indices.size),
            "baseline_kl_mean": float(baseline_kl[indices].mean()),
            "candidate_kl_mean": float(candidate_kl[indices].mean()),
            "paired_improvement": domain_summary,
        }

    if paired["ci95_low"] > minimum:
        statistical_status = "improvement_supported"
    elif paired["ci95_high"] < -minimum:
        statistical_status = "regression_supported"
    else:
        statistical_status = "inconclusive"
    supported_domain_regressions = sorted(
        domain
        for domain, value in domains.items()
        if value["paired_improvement"]["ci95_high"] < -minimum
    )
    point_domain_regressions = sorted(
        domain
        for domain, value in domains.items()
        if value["paired_improvement"]["estimate"] < 0
    )
    exceeds_noise = abs(paired["estimate"]) > noise
    if statistical_status == "regression_supported" or supported_domain_regressions:
        verdict = "fail"
    elif (
        statistical_status == "improvement_supported"
        and exceeds_noise
        and not point_domain_regressions
    ):
        verdict = "pass"
    else:
        verdict = "repeat_required"

    baseline_mean = float(baseline_kl.mean())
    baseline_js = float(np.mean([window.js_mean for window in baseline]))
    candidate_js = float(np.mean([window.js_mean for window in candidate]))
    baseline_top1 = float(np.mean([window.top1_agreement for window in baseline]))
    candidate_top1 = float(np.mean([window.top1_agreement for window in candidate]))
    return {
        "kind": KLD_GATE_KIND,
        "schema_version": KLD_GATE_SCHEMA_VERSION,
        "direction": KLD_DIRECTION,
        "improvement_definition": (
            "baseline mean KLD minus candidate mean KLD; positive is better"
        ),
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "windows": len(baseline),
        "positions": sum(window.positions for window in baseline),
        "baseline": {
            "kl_mean": baseline_mean,
            "js_mean": baseline_js,
            "top1_agreement": baseline_top1,
        },
        "candidate": {
            "kl_mean": float(candidate_kl.mean()),
            "js_mean": candidate_js,
            "top1_agreement": candidate_top1,
        },
        "change": {
            "paired_kl_improvement": paired,
            "relative_kl_improvement": (
                paired["estimate"] / baseline_mean if baseline_mean > 0 else None
            ),
            "js_mean_reduction": baseline_js - candidate_js,
            "top1_agreement_increase": candidate_top1 - baseline_top1,
            "windows_improved": int(np.count_nonzero(improvements > 0)),
            "windows_tied": int(np.count_nonzero(improvements == 0)),
            "windows_regressed": int(np.count_nonzero(improvements < 0)),
        },
        "domains": domains,
        "decision": {
            "verdict": verdict,
            "statistical_status": statistical_status,
            "minimum_mean_improvement": minimum,
            "documented_runtime_noise_kl": noise,
            "point_delta_exceeds_runtime_noise": exceeds_noise,
            "point_domain_regressions": point_domain_regressions,
            "supported_domain_regressions": supported_domain_regressions,
            "runtime_noise_scope": (
                "The noise floor comes from repeated same-model captures and is "
                "separate from the window-cluster bootstrap."
            ),
        },
        "per_window": [
            {
                "window_index": left.index,
                "domain": left.domain,
                "positions": left.positions,
                "baseline_kl_mean": left.kl_mean,
                "candidate_kl_mean": right.kl_mean,
                "kl_improvement": left.kl_mean - right.kl_mean,
                "baseline_js_mean": left.js_mean,
                "candidate_js_mean": right.js_mean,
                "baseline_top1_agreement": left.top1_agreement,
                "candidate_top1_agreement": right.top1_agreement,
            }
            for left, right in zip(baseline, candidate, strict=True)
        ],
    }
