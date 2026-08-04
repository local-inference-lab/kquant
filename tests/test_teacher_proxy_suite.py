from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from kquant.kimi_stream import write_trace_manifest, write_trace_tensor
from kquant.teacher_proxy import DEFAULT_FLOAT_STAGES
from kquant.teacher_proxy_gate import (
    compare_teacher_proxy_suite,
    validate_teacher_proxy_gate_report,
)
from kquant.teacher_proxy_suite import (
    TEACHER_PROXY_SUITE_KIND,
    build_teacher_proxy_suite,
    file_sha256,
    load_teacher_proxy_suite,
    suite_token_hash,
    token_ids_sha256,
)


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [int(value) for value in text.split()]

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        assert tokenize is True
        assert add_generation_prompt is False
        return self.encode(messages[0]["content"])


def _blake_tokens(values: list[int]) -> str:
    payload = b"".join(value.to_bytes(4, "little") for value in values)
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def _write_corpus(tmp_path: Path) -> tuple[Path, Path]:
    tokenizer_root = tmp_path / "tokenizer"
    tokenizer_root.mkdir()
    (tokenizer_root / "tokenizer.json").write_text("{}")
    source_specs = []
    documents = []
    for source_index, weight in enumerate((2.0, 1.0, 1.0)):
        source = tmp_path / f"source-{source_index}.jsonl"
        rows = []
        for line in range(1, 5):
            values = [source_index * 100 + line * 10 + offset for offset in range(6)]
            raw = json.dumps({"prompt": " ".join(map(str, values))})
            rows.append(raw)
            documents.append(
                {
                    "source": str(source.resolve()),
                    "line": line,
                    "document_hash": hashlib.blake2b(
                        raw.encode(), digest_size=16
                    ).hexdigest(),
                    "prompt_hash": _blake_tokens(values),
                    "tokens": len(values),
                }
            )
        source.write_text("\n".join(rows) + "\n")
        source_specs.append({"path": str(source.resolve()), "weight": weight})
    report = {
        "kind": "kquant_interim_calibration_corpus_run",
        "schema_version": 1,
        "sources": source_specs,
        "documents": documents,
        "planned_requests": len(documents),
        "completed_requests": len(documents),
        "planned_tokens": len(documents) * 6,
        "reported_prompt_tokens": len(documents) * 6,
        "finalized": True,
        "fold": {"modulus": 8, "index": 0, "mode": "include"},
        "seed": 7,
    }
    report_path = tmp_path / "corpus.json"
    report_path.write_text(json.dumps(report))
    return report_path, tokenizer_root


def test_build_and_load_teacher_proxy_suite(tmp_path: Path) -> None:
    report, tokenizer_root = _write_corpus(tmp_path)
    payload = build_teacher_proxy_suite(
        corpus_report_path=report,
        tokenizer_root=tokenizer_root,
        tokenizer=_Tokenizer(),
        document_count=8,
        tokens_per_document=4,
        trace_layers=(1, 24, 40),
    )
    assert payload["batch_shape"] == [8, 4]
    assert payload["selection"]["source_document_counts"] == {
        str((tmp_path / "source-0.jsonl").resolve()): 4,
        str((tmp_path / "source-1.jsonl").resolve()): 2,
        str((tmp_path / "source-2.jsonl").resolve()): 2,
    }
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(payload))
    loaded = load_teacher_proxy_suite(path)
    assert loaded.input_ids.shape == (8, 4)
    assert loaded.file_sha256 == file_sha256(path)

    payload["documents"][0]["input_ids"][0] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="token hash differs"):
        load_teacher_proxy_suite(path)


def _synthetic_suite(path: Path) -> dict:
    corpus = path.with_name("corpus-source.json")
    corpus.write_text("{}")
    tokenizer_root = path.with_name("tokenizer-source")
    tokenizer_root.mkdir()
    tokenizer_file = tokenizer_root / "tokenizer.json"
    tokenizer_file.write_text("{}")
    documents = []
    token_hashes = []
    for index in range(8):
        values = [index * 2 + 1, index * 2 + 2]
        digest = token_ids_sha256(values)
        token_hashes.append(digest)
        documents.append(
            {
                "suite_index": index,
                "source": f"/source/{index % 3}",
                "line": index + 1,
                "document_hash": f"doc-{index}",
                "input_tokens": 2,
                "input_ids_sha256": digest,
                "input_ids": values,
            }
        )
    payload = {
        "kind": TEACHER_PROXY_SUITE_KIND,
        "schema_version": 1,
        "corpus_report": {
            "path": str(corpus),
            "sha256": file_sha256(corpus),
        },
        "tokenizer": {
            "root": str(tokenizer_root),
            "files_sha256": {"tokenizer.json": file_sha256(tokenizer_file)},
        },
        "batch_shape": [8, 2],
        "trace_layers": [1, 2, 3],
        "end_layer": 4,
        "suite_token_hash_sha256": suite_token_hash(token_hashes),
        "documents": documents,
    }
    path.write_text(json.dumps(payload))
    return payload


def _write_run(
    root: Path,
    *,
    suite_path: Path,
    suite: dict,
    checkpoint: Path,
    expert_checkpoint: Path,
    interim: bool,
) -> None:
    root.mkdir()
    loaded = load_teacher_proxy_suite(suite_path)
    run = {
        "schema_version": 1,
        "status": "complete",
        "checkpoint": str(checkpoint),
        "expert_checkpoint": str(expert_checkpoint),
        "nonexpert_checkpoint": str(expert_checkpoint),
        "exl3_manifest": (
            str(expert_checkpoint / "kquant_exl3_manifest.json")
            if interim
            else None
        ),
        "input_ids": loaded.input_ids.flatten().tolist(),
        "input_ids_shape": list(loaded.input_ids.shape),
        "input_suite": {
            "path": str(suite_path),
            "file_sha256": loaded.file_sha256,
            "kind": suite["kind"],
            "schema_version": suite["schema_version"],
            "suite_token_hash_sha256": suite["suite_token_hash_sha256"],
            "batch_shape": suite["batch_shape"],
            "document_hashes": [
                document["document_hash"] for document in suite["documents"]
            ],
        },
        "trace_layers": suite["trace_layers"],
        "end_layer": suite["end_layer"],
        "capture_routed_post_situ": True,
    }
    (root / "run.json").write_text(json.dumps(run))
    trace = root / "trace"
    write_trace_manifest(
        trace,
        layers=suite["trace_layers"],
        max_tokens=16,
        checkpoint=str(expert_checkpoint),
    )
    ids = torch.tensor([[row % 4, (row + 1) % 4] for row in range(16)])
    weights = torch.tensor([[0.6, 0.4]]).repeat(16, 1)
    post_situ = torch.arange(768, dtype=torch.float32).view(1, 1, -1)
    post_situ = post_situ.repeat(16, 2, 1)
    for layer in suite["trace_layers"]:
        write_trace_tensor(trace, layer=layer, stage="canonical_topk_ids", tensor=ids)
        write_trace_tensor(
            trace, layer=layer, stage="canonical_topk_weights", tensor=weights
        )
        write_trace_tensor(
            trace, layer=layer, stage="routed_post_situ", tensor=post_situ
        )
        for offset, stage in enumerate(DEFAULT_FLOAT_STAGES):
            value = torch.tensor([[float(layer), float(offset + 1)]]).repeat(16, 1)
            write_trace_tensor(trace, layer=layer, stage=stage, tensor=value)


def test_teacher_proxy_suite_gate_passes_identical_document_batch(
    tmp_path: Path,
) -> None:
    suite_path = tmp_path / "suite.json"
    suite = _synthetic_suite(suite_path)
    checkpoint = tmp_path / "official-checkpoint"
    interim_checkpoint = tmp_path / "interim-checkpoint"
    official = tmp_path / "official"
    interim = tmp_path / "interim"
    _write_run(
        official,
        suite_path=suite_path,
        suite=suite,
        checkpoint=checkpoint,
        expert_checkpoint=checkpoint,
        interim=False,
    )
    _write_run(
        interim,
        suite_path=suite_path,
        suite=suite,
        checkpoint=checkpoint,
        expert_checkpoint=interim_checkpoint,
        interim=True,
    )

    report = compare_teacher_proxy_suite(
        suite_path=suite_path,
        official_root=official,
        interim_root=interim,
        expected_interim=interim_checkpoint,
        bootstrap_samples=1000,
        min_expert_post_situ_routes=1,
        num_experts=4,
    )

    assert report["status"] == "pass"
    assert report["scope"]["total_tokens"] == 16
    assert not report["decision"]["point_failures"]
    assert all(
        metric["lower_bound_pass"]
        for metric in report["decision"]["metrics"].values()
    )

    report_path = tmp_path / "teacher-proxy-report.json"
    report_path.write_text(json.dumps(report))
    receipt = validate_teacher_proxy_gate_report(
        report_path,
        bootstrap_samples=1000,
        min_expert_post_situ_routes=1,
        num_experts=4,
    )
    assert receipt["status"] == "pass"
    assert receipt["document_count"] == 8

    report["status"] = "fail"
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="stale or disagrees"):
        validate_teacher_proxy_gate_report(
            report_path,
            bootstrap_samples=1000,
            min_expert_post_situ_routes=1,
            num_experts=4,
        )
