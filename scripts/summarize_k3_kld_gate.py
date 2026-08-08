#!/usr/bin/env python3
"""Build a fail-closed paired KLD gate for two Kimi-K3 serving captures."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from kquant.correctness import sha256_file
from kquant.kld_gate import (
    CANONICAL_EVALUATION_TOOLS_COMMIT,
    CANONICAL_REFERENCE_DATASET,
    CANONICAL_REFERENCE_DATASET_REVISION,
    CANONICAL_REFERENCE_MANIFEST_SHA256,
    CANONICAL_SUITE_MANIFEST_SHA256,
    CANONICAL_SUITE_TOKEN_HASH,
    CANONICAL_TOOL_SHA256,
    DOCUMENTED_RUNTIME_NOISE_KL,
    metric_statistics,
    select_suite_windows,
    summarize_paired_kld,
    validate_comparison_report,
    validate_logits_manifest,
    validate_suite_manifest,
    validate_token_ids,
)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _require_report_path(
    report: dict[str, Any], *, key: str, expected: Path, label: str
) -> None:
    value = report.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{label} report is missing {key}")
    if Path(value).resolve() != expected.resolve():
        raise ValueError(
            f"{label} report {key} resolves to {Path(value).resolve()}, "
            f"not {expected.resolve()}"
        )


def _validate_payload_records(
    directory: Path,
    records: tuple[dict[str, Any], ...] | tuple[Any, ...],
    *,
    label: str,
    rehash: bool,
) -> None:
    for record in records:
        filename = record.get("file")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"{label} manifest contains an unsafe logit filename")
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(record["size_bytes"]):
            raise ValueError(f"{label} logit size differs for {filename}")
        if rehash and sha256_file(path) != record["sha256"]:
            raise ValueError(f"{label} logit SHA-256 differs for {filename}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--baseline-comparison", type=Path, required=True)
    parser.add_argument("--candidate-comparison", type=Path, required=True)
    parser.add_argument("--baseline-label", default="3p09")
    parser.add_argument("--candidate-label", default="qsrt")
    parser.add_argument("--start-window", type=int, default=0)
    parser.add_argument("--stop-window", type=int)
    parser.add_argument(
        "--expected-suite-hash", default=CANONICAL_SUITE_TOKEN_HASH
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=1)
    parser.add_argument("--minimum-mean-improvement", type=float, default=0.0)
    parser.add_argument(
        "--documented-runtime-noise-kl",
        type=float,
        default=DOCUMENTED_RUNTIME_NOISE_KL,
    )
    parser.add_argument(
        "--allow-missing-runtime",
        action="store_true",
        help="Allow smoke manifests that omit serving/model provenance.",
    )
    parser.add_argument(
        "--skip-logit-rehash",
        action="store_true",
        help="Skip full payload hashes for a smoke; promotion runs must not use this.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_root = args.reference_root.resolve()
    reference_dir = reference_root / "ref"
    baseline_dir = args.baseline_dir.resolve()
    candidate_dir = args.candidate_dir.resolve()
    suite_path = reference_root / "suite-manifest.json"
    reference_manifest_path = reference_dir / "manifest.json"
    baseline_manifest_path = baseline_dir / "manifest.json"
    candidate_manifest_path = candidate_dir / "manifest.json"

    if sha256_file(suite_path) != CANONICAL_SUITE_MANIFEST_SHA256:
        raise ValueError("reference suite manifest is not the pinned canonical file")
    if sha256_file(reference_manifest_path) != CANONICAL_REFERENCE_MANIFEST_SHA256:
        raise ValueError("reference logit manifest is not the pinned canonical file")
    for filename, expected_hash in CANONICAL_TOOL_SHA256.items():
        tool_path = reference_root / "tools" / filename
        if sha256_file(tool_path) != expected_hash:
            raise ValueError(f"reference tool is not canonical: {filename}")

    suite = _load_object(suite_path)
    suite_hash, all_windows = validate_suite_manifest(
        suite, expected_suite_hash=args.expected_suite_hash
    )
    windows = select_suite_windows(
        all_windows,
        start_window=args.start_window,
        stop_window=args.stop_window,
    )
    for window in windows:
        token_path = reference_root / window.token_file
        validate_token_ids(
            json.loads(token_path.read_text(encoding="utf-8")), window=window
        )

    manifests = {
        "reference": (
            reference_manifest_path,
            _load_object(reference_manifest_path),
            reference_dir,
        ),
        "baseline": (
            baseline_manifest_path,
            _load_object(baseline_manifest_path),
            baseline_dir,
        ),
        "candidate": (
            candidate_manifest_path,
            _load_object(candidate_manifest_path),
            candidate_dir,
        ),
    }
    manifest_records: dict[str, tuple[Any, ...]] = {}
    for label, (_, manifest, directory) in manifests.items():
        records = validate_logits_manifest(
            manifest,
            label=label,
            suite_hash=suite_hash,
            expected_windows=windows,
        )
        if not args.allow_missing_runtime and not isinstance(manifest.get("runtime"), dict):
            raise ValueError(
                f"{label} manifest lacks a runtime identity; finalize with "
                "--runtime-manifest or use --allow-missing-runtime only for a smoke"
            )
        _validate_payload_records(
            directory,
            records,
            label=label,
            rehash=not args.skip_logit_rehash,
        )
        manifest_records[label] = records

    baseline_report = _load_object(args.baseline_comparison)
    candidate_report = _load_object(args.candidate_comparison)
    _require_report_path(
        baseline_report,
        key="reference_dir",
        expected=reference_dir,
        label="baseline",
    )
    _require_report_path(
        baseline_report,
        key="candidate_dir",
        expected=baseline_dir,
        label="baseline",
    )
    _require_report_path(
        candidate_report,
        key="reference_dir",
        expected=reference_dir,
        label="candidate",
    )
    _require_report_path(
        candidate_report,
        key="candidate_dir",
        expected=candidate_dir,
        label="candidate",
    )
    baseline_windows = validate_comparison_report(
        baseline_report, label="baseline", expected_windows=windows
    )
    candidate_windows = validate_comparison_report(
        candidate_report, label="candidate", expected_windows=windows
    )
    result = summarize_paired_kld(
        baseline_windows,
        candidate_windows,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        minimum_mean_improvement=args.minimum_mean_improvement,
        documented_runtime_noise_kl=args.documented_runtime_noise_kl,
    )
    for label, report in (
        ("baseline", baseline_report),
        ("candidate", candidate_report),
    ):
        result[label]["all_position_statistics"] = {
            "kl_reference_to_candidate": metric_statistics(
                report, key="kl_reference_to_candidate", label=label
            ),
            "js": metric_statistics(report, key="js", label=label),
        }
    result["suite"] = {
        "reference_dataset": CANONICAL_REFERENCE_DATASET,
        "reference_dataset_revision": CANONICAL_REFERENCE_DATASET_REVISION,
        "evaluation_tools_commit": CANONICAL_EVALUATION_TOOLS_COMMIT,
        "token_hash_sha256": suite_hash,
        "window_range": [windows[0].index, windows[-1].index + 1],
        "reference_root": str(reference_root),
    }
    result["provenance"] = {
        "suite_manifest": {
            "path": str(suite_path),
            "sha256": sha256_file(suite_path),
        },
        "reference_manifest": {
            "path": str(reference_manifest_path),
            "sha256": sha256_file(reference_manifest_path),
            "runtime": manifests["reference"][1].get("runtime"),
        },
        "baseline_manifest": {
            "path": str(baseline_manifest_path),
            "sha256": sha256_file(baseline_manifest_path),
            "runtime": manifests["baseline"][1].get("runtime"),
        },
        "candidate_manifest": {
            "path": str(candidate_manifest_path),
            "sha256": sha256_file(candidate_manifest_path),
            "runtime": manifests["candidate"][1].get("runtime"),
        },
        "baseline_comparison": {
            "path": str(args.baseline_comparison.resolve()),
            "sha256": sha256_file(args.baseline_comparison),
        },
        "candidate_comparison": {
            "path": str(args.candidate_comparison.resolve()),
            "sha256": sha256_file(args.candidate_comparison),
        },
        "payload_sha256_rechecked": not args.skip_logit_rehash,
    }
    _atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "verdict": result["decision"]["verdict"],
                "statistical_status": result["decision"]["statistical_status"],
                "paired_kl_improvement": result["change"][
                    "paired_kl_improvement"
                ],
                "point_domain_regressions": result["decision"][
                    "point_domain_regressions"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
