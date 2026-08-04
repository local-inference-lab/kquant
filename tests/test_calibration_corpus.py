from __future__ import annotations

import json
from pathlib import Path

from scripts.run_interim_calibration_corpus import (
    SourceSpec,
    _allocate_quotas,
    _content_hash,
    _fold_selected,
    _load_excluded_documents,
    _resume_report,
    build_plan,
)


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [ord(character) for character in text]

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        assert tokenize
        assert not add_generation_prompt
        text = "|".join(
            f"{message['role']}:{message['content']}" for message in messages
        )
        return [ord(character) for character in text]


def _write_source(path: Path, count: int) -> None:
    rows = []
    for index in range(count):
        rows.append(
            json.dumps(
                {"messages": [{"role": "user", "content": f"document-{index}-" * 8}]}
            )
        )
    path.write_text("\n".join(rows) + "\n")


def test_content_fold_is_document_disjoint() -> None:
    hashes = [_content_hash(f"record-{index}") for index in range(100)]
    included = {
        value
        for value in hashes
        if _fold_selected(value, modulus=8, index=0, mode="include")
    }
    excluded = {
        value
        for value in hashes
        if _fold_selected(value, modulus=8, index=0, mode="exclude")
    }

    assert included
    assert excluded
    assert included.isdisjoint(excluded)
    assert included | excluded == set(hashes)


def test_weighted_quotas_are_exact() -> None:
    sources = [
        SourceSpec(Path("a"), 2),
        SourceSpec(Path("b"), 1),
        SourceSpec(Path("c"), 1),
    ]

    assert _allocate_quotas(101, sources) == [51, 25, 25]


def test_plan_is_deterministic_and_exact(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_source(first, 100)
    _write_source(second, 100)
    sources = [SourceSpec(first, 3), SourceSpec(second, 1)]
    kwargs = {
        "target_tokens": 1_000,
        "max_prompt_tokens": 100,
        "min_prompt_tokens": 20,
        "fold_modulus": 8,
        "fold_index": 0,
        "fold_mode": "exclude",
        "seed": 7,
    }

    plan = build_plan(sources, _Tokenizer(), **kwargs)
    repeated = build_plan(sources, _Tokenizer(), **kwargs)

    assert plan == repeated
    assert sum(request.tokens for request in plan) == 1_000
    assert all(request.tokens <= 100 for request in plan)


def test_plan_deduplicates_repeated_source_records(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    unique = [
        json.dumps({"prompt": f"document-{index}-" * 8})
        for index in range(12)
    ]
    source.write_text("\n".join(unique + unique[:4]) + "\n")

    plan = build_plan(
        [SourceSpec(source, 1)],
        _Tokenizer(),
        target_tokens=400,
        max_prompt_tokens=100,
        min_prompt_tokens=20,
        fold_modulus=1,
        fold_index=0,
        fold_mode="include",
        seed=7,
    )

    hashes = [request.document_hash for request in plan]
    assert len(hashes) == len(set(hashes))


def test_plan_excludes_prior_corpus_documents(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_source(source, 100)
    baseline = build_plan(
        [SourceSpec(source, 1)],
        _Tokenizer(),
        target_tokens=500,
        max_prompt_tokens=100,
        min_prompt_tokens=20,
        fold_modulus=1,
        fold_index=0,
        fold_mode="include",
        seed=7,
    )
    excluded = {request.document_hash for request in baseline[:2]}
    replacement = build_plan(
        [SourceSpec(source, 1)],
        _Tokenizer(),
        target_tokens=500,
        max_prompt_tokens=100,
        min_prompt_tokens=20,
        fold_modulus=1,
        fold_index=0,
        fold_mode="include",
        seed=7,
        excluded_document_hashes=excluded,
    )

    assert excluded.isdisjoint(request.document_hash for request in replacement)
    assert sum(request.tokens for request in replacement) == 500


def test_excluded_report_inventory_is_authenticated(tmp_path: Path) -> None:
    report = tmp_path / "prior.json"
    report.write_text(
        json.dumps(
            {
                "kind": "kquant_interim_calibration_corpus_run",
                "documents": [
                    {"document_hash": "11" * 16},
                    {"document_hash": "22" * 16},
                ],
            }
        )
    )

    hashes, identities = _load_excluded_documents([report])

    assert hashes == {"11" * 16, "22" * 16}
    assert identities[0]["path"] == str(report.resolve())
    assert identities[0]["documents"] == 2
    assert len(identities[0]["sha256"]) == 64


def test_resume_report_requires_identical_document_plan(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    planned = {
        "kind": "kquant_interim_calibration_corpus_run",
        "schema_version": 1,
        "constraint": "interim only",
        "model_dir": "/models/interim",
        "capture_dir": "/data/capture",
        "sources": [],
        "fold": {},
        "seed": 7,
        "target_tokens": 10,
        "planned_tokens": 10,
        "planned_requests": 1,
        "documents": [{"prompt_hash": "abc", "tokens": 10}],
        "completed_requests": 1,
    }
    report_path.write_text(json.dumps(planned))

    assert _resume_report(report_path, planned)["completed_requests"] == 1

    changed = planned | {"documents": [{"prompt_hash": "def", "tokens": 10}]}
    try:
        _resume_report(report_path, changed)
    except ValueError as error:
        assert "documents" in str(error)
    else:
        raise AssertionError("changed plan should not be resumable")
