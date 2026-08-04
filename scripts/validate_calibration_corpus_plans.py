"""Validate deterministic calibration corpus plans and split isolation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

CORPUS_KIND = "kquant_interim_calibration_corpus_run"


def _is_digest(value: object, *, bytes_: int = 16) -> bool:
    if not isinstance(value, str) or len(value) != bytes_ * 2:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


def _quantile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def _validate_report(path: Path) -> tuple[dict, list[str], dict[str, set[str]]]:
    report = json.loads(path.read_text())
    issues: list[str] = []
    if report.get("kind") != CORPUS_KIND:
        issues.append("unsupported report kind")
    documents = report.get("documents")
    if not isinstance(documents, list) or not documents:
        issues.append("document inventory is missing or empty")
        documents = []

    document_hashes: list[str] = []
    prompt_hashes: list[str] = []
    source_rows: dict[str, list[int]] = defaultdict(list)
    for index, document in enumerate(documents):
        if not isinstance(document, dict):
            issues.append(f"document {index} is not an object")
            continue
        document_hash = document.get("document_hash")
        prompt_hash = document.get("prompt_hash")
        tokens = document.get("tokens")
        source = document.get("source")
        if not _is_digest(document_hash):
            issues.append(f"document {index} has an invalid document hash")
        else:
            document_hashes.append(document_hash)
        if not _is_digest(prompt_hash):
            issues.append(f"document {index} has an invalid prompt hash")
        else:
            prompt_hashes.append(prompt_hash)
        if not isinstance(tokens, int) or tokens <= 0:
            issues.append(f"document {index} has an invalid token count")
        elif not isinstance(source, str):
            issues.append(f"document {index} has an invalid source")
        else:
            source_rows[source].append(tokens)

    if len(document_hashes) != len(set(document_hashes)):
        issues.append("duplicate document hashes within report")
    if len(prompt_hashes) != len(set(prompt_hashes)):
        issues.append("duplicate prompt hashes within report")
    actual_tokens = sum(sum(values) for values in source_rows.values())
    if report.get("planned_requests") != len(documents):
        issues.append("planned request count disagrees with document inventory")
    if report.get("planned_tokens") != actual_tokens:
        issues.append("planned token count disagrees with document inventory")
    if report.get("target_tokens") != actual_tokens:
        issues.append("target token count disagrees with document inventory")

    source_contract = {
        str(source.get("path")): source
        for source in report.get("sources", [])
        if isinstance(source, dict) and isinstance(source.get("path"), str)
    }
    for source, lengths in source_rows.items():
        contract = source_contract.get(source)
        if contract is None:
            issues.append(f"document source is absent from source contract: {source}")
            continue
        cap = contract.get("max_prompt_tokens")
        if not isinstance(cap, int) or cap <= 0:
            issues.append(f"source has an invalid prompt cap: {source}")
        elif max(lengths) > cap:
            issues.append(f"source exceeds its prompt cap: {source}")

    fold = report.get("fold")
    if isinstance(fold, dict):
        modulus = fold.get("modulus")
        index = fold.get("index")
        mode = fold.get("mode")
        if (
            isinstance(modulus, int)
            and modulus > 0
            and isinstance(index, int)
            and 0 <= index < modulus
            and mode in {"include", "exclude"}
        ):
            for value in document_hashes:
                selected = int(value, 16) % modulus == index
                if selected != (mode == "include"):
                    issues.append("document violates the configured content-hash fold")
                    break
        else:
            issues.append("invalid fold contract")
    else:
        issues.append("missing fold contract")

    excluded_documents = set(report.get("excluded_document_hashes", []))
    excluded_prompts = set(report.get("excluded_prompt_hashes", []))
    if set(document_hashes) & excluded_documents:
        issues.append("report includes an explicitly excluded document")
    if set(prompt_hashes) & excluded_prompts:
        issues.append("report includes an explicitly excluded prompt")

    sources = {}
    for source, lengths in sorted(source_rows.items()):
        sources[source] = {
            "documents": len(lengths),
            "tokens": sum(lengths),
            "p50_tokens": _quantile(lengths, 0.50),
            "p90_tokens": _quantile(lengths, 0.90),
            "max_tokens": max(lengths),
        }
    summary = {
        "path": str(path.resolve()),
        "documents": len(documents),
        "tokens": actual_tokens,
        "sources": sources,
    }
    hashes = {
        "documents": set(document_hashes),
        "prompts": set(prompt_hashes),
    }
    return summary, issues, hashes


def validate_reports(paths: list[Path]) -> dict:
    if not paths:
        raise ValueError("at least one corpus report is required")
    summaries = []
    all_issues: list[dict[str, str]] = []
    hash_sets = []
    for path in paths:
        summary, issues, hashes = _validate_report(path)
        summaries.append(summary)
        hash_sets.append(hashes)
        all_issues.extend({"path": str(path), "issue": issue} for issue in issues)

    overlaps = []
    for left in range(len(paths)):
        for right in range(left + 1, len(paths)):
            document_overlap = hash_sets[left]["documents"] & hash_sets[right]["documents"]
            prompt_overlap = hash_sets[left]["prompts"] & hash_sets[right]["prompts"]
            overlap = {
                "left": str(paths[left].resolve()),
                "right": str(paths[right].resolve()),
                "document_hashes": len(document_overlap),
                "prompt_hashes": len(prompt_overlap),
            }
            overlaps.append(overlap)
            if document_overlap:
                all_issues.append(
                    {
                        "path": f"{paths[left]} :: {paths[right]}",
                        "issue": "corpus reports share document hashes",
                    }
                )
            if prompt_overlap:
                all_issues.append(
                    {
                        "path": f"{paths[left]} :: {paths[right]}",
                        "issue": "corpus reports share tokenized prompt hashes",
                    }
                )
    return {
        "kind": "kquant_calibration_corpus_plan_validation",
        "schema_version": 1,
        "status": "pass" if not all_issues else "fail",
        "reports": summaries,
        "pairwise_overlaps": overlaps,
        "issues": all_issues,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_reports(args.reports)
    payload = json.dumps(result, indent=1, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(payload)
        temporary.replace(args.output)
    print(payload, end="")
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
