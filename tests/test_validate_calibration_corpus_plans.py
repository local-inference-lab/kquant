from __future__ import annotations

import json
from pathlib import Path

from scripts.validate_calibration_corpus_plans import validate_reports


def _report(path: Path, *, document_hash: str, prompt_hash: str) -> None:
    source = str((path.parent / "source.jsonl").resolve())
    path.write_text(
        json.dumps(
            {
                "kind": "kquant_interim_calibration_corpus_run",
                "planned_requests": 1,
                "planned_tokens": 64,
                "target_tokens": 64,
                "sources": [
                    {"path": source, "weight": 1.0, "max_prompt_tokens": 64}
                ],
                "fold": {"modulus": 1, "index": 0, "mode": "include"},
                "excluded_document_hashes": [],
                "excluded_prompt_hashes": [],
                "documents": [
                    {
                        "source": source,
                        "document_hash": document_hash,
                        "prompt_hash": prompt_hash,
                        "tokens": 64,
                    }
                ],
            }
        )
    )


def test_validator_accepts_disjoint_reports(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _report(first, document_hash="11" * 16, prompt_hash="22" * 16)
    _report(second, document_hash="33" * 16, prompt_hash="44" * 16)

    result = validate_reports([first, second])

    assert result["status"] == "pass"
    assert result["pairwise_overlaps"][0]["document_hashes"] == 0
    assert result["pairwise_overlaps"][0]["prompt_hashes"] == 0


def test_validator_rejects_shared_tokenized_prompt(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _report(first, document_hash="11" * 16, prompt_hash="22" * 16)
    _report(second, document_hash="33" * 16, prompt_hash="22" * 16)

    result = validate_reports([first, second])

    assert result["status"] == "fail"
    assert result["pairwise_overlaps"][0]["document_hashes"] == 0
    assert result["pairwise_overlaps"][0]["prompt_hashes"] == 1
