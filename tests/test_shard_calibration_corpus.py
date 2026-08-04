from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.shard_calibration_corpus import shard_corpus


def test_shard_corpus_preserves_selected_raw_records(tmp_path: Path) -> None:
    source = tmp_path / "mixed.jsonl"
    rows = [
        b'{"source":"chat","prompt":"one"}\n',
        b'{"source":"tool","prompt":"two"}\n',
        b'{"source":"chat","prompt":"three"}\n',
        b'{"source":"other","prompt":"four"}\n',
    ]
    source.write_bytes(b"".join(rows))

    output = tmp_path / "shards"
    manifest = shard_corpus(
        source, output, field="source", labels=["chat", "tool"]
    )

    chat = rows[0] + rows[2]
    assert (output / "chat.jsonl").read_bytes() == chat
    assert (output / "tool.jsonl").read_bytes() == rows[1]
    assert manifest["unselected_records"] == 1
    assert manifest["shards"]["chat"]["sha256"] == hashlib.sha256(chat).hexdigest()
    assert json.loads((output / "manifest.json").read_text()) == manifest


def test_shard_corpus_refuses_to_overwrite_manifest(tmp_path: Path) -> None:
    source = tmp_path / "mixed.jsonl"
    source.write_text('{"source":"chat","prompt":"one"}\n')
    output = tmp_path / "shards"
    shard_corpus(source, output, field="source", labels=["chat"])

    with pytest.raises(FileExistsError):
        shard_corpus(source, output, field="source", labels=["chat"])
