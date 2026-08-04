from __future__ import annotations

from scripts.assemble_hf_calibration_text import (
    _eligible,
    _output_record,
    _parse_minimum,
)


def test_minimum_filter_requires_numeric_threshold() -> None:
    assert _parse_minimum("language_score=0.9") == ("language_score", 0.9)
    assert _eligible({"language_score": 0.95}, [("language_score", 0.9)])
    assert not _eligible({"language_score": 0.8}, [("language_score", 0.9)])


def test_output_record_is_bounded_and_provenance_pinned() -> None:
    built = _output_record(
        {"text": "  abcdef  ", "id": "row-1", "nested": {"ignored": True}},
        text_field="text",
        max_chars=4,
        dataset="owner/data",
        revision="abc123",
        config="default",
        split="train",
        provenance_fields=["id", "nested"],
    )

    assert built is not None
    record, content_hash = built
    assert record["prompt"] == "abcd"
    assert record["provenance"] == {
        "dataset": "owner/data",
        "revision": "abc123",
        "config": "default",
        "split": "train",
        "content_hash": content_hash,
        "id": "row-1",
    }
