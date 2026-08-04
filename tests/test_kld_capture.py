from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import kquant.kld_capture as capture
from kquant.correctness import sha256_file
from scripts.run_k3_kld_capture import build_server_environment


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(",", ":")), encoding="utf-8")


def _fake_reference_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "reference"
    token_ids = [3, 7]
    token_payload = json.dumps(token_ids, separators=(",", ":"))
    token_hash = hashlib.sha256(token_payload.encode()).hexdigest()
    suite_hash = hashlib.sha256(token_hash.encode("ascii")).hexdigest()
    token_file = "tokens/window-000-prose.json"
    _write_json(root / token_file, token_ids)
    suite = {
        "context_length": 2,
        "window_count": 1,
        "scored_positions_per_window": 1,
        "total_scored_positions": 1,
        "suite_token_hash_sha256": suite_hash,
        "domain_counts": {"prose": 1},
        "windows": [
            {
                "index": 0,
                "domain": "prose",
                "num_tokens": 2,
                "token_ids_json_sha256": token_hash,
                "token_file": token_file,
            }
        ],
    }
    suite_path = root / "suite-manifest.json"
    _write_json(suite_path, suite)

    payload = root / "ref/logits_000.safetensors"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"fake-logits")
    manifest = {
        "suite_token_hash_sha256": suite_hash,
        "window_count": 1,
        "window_range": [0, 1],
        "vocab_size": 3,
        "total_scored_positions": 1,
        "total_size_bytes": payload.stat().st_size,
        "windows": [
            {
                "window_index": 0,
                "domain": "prose",
                "file": payload.name,
                "key": "logits",
                "shape": [1, 3],
                "dtype": "F32",
                "size_bytes": payload.stat().st_size,
                "sha256": sha256_file(payload),
                "token_ids_json_sha256": token_hash,
            }
        ],
    }
    manifest_path = root / "ref/manifest.json"
    _write_json(manifest_path, manifest)
    tool = root / "tools/tool.py"
    tool.parent.mkdir(parents=True)
    tool.write_text("pass\n", encoding="utf-8")

    monkeypatch.setattr(capture, "CANONICAL_SUITE_TOKEN_HASH", suite_hash)
    monkeypatch.setattr(
        capture, "CANONICAL_SUITE_MANIFEST_SHA256", sha256_file(suite_path)
    )
    monkeypatch.setattr(
        capture, "CANONICAL_REFERENCE_MANIFEST_SHA256", sha256_file(manifest_path)
    )
    monkeypatch.setattr(capture, "CANONICAL_TOOL_SHA256", {"tool.py": sha256_file(tool)})
    return root


def test_validate_reference_root_rehashes_every_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fake_reference_root(tmp_path, monkeypatch)

    result = capture.validate_reference_root(root, rehash_payloads=True)

    assert result["windows"] == 1
    assert result["total_size_bytes"] == len(b"fake-logits")
    assert result["payloads_rehashed"] is True

    (root / "ref/logits_000.safetensors").write_bytes(b"bad-content")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        capture.validate_reference_root(root, rehash_payloads=True)


def test_prepare_suite_workspace_links_read_only_inputs(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    (reference / "tokens").mkdir(parents=True)
    (reference / "suite-manifest.json").write_text("{}", encoding="utf-8")

    workspace = capture.prepare_suite_workspace(reference, tmp_path / "run/suite")

    assert (workspace / "suite-manifest.json").resolve() == (
        reference / "suite-manifest.json"
    ).resolve()
    assert (workspace / "tokens").resolve() == (reference / "tokens").resolve()


def test_atomic_write_json_replaces_complete_document(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    capture.atomic_write_json(path, {"status": "starting"})
    capture.atomic_write_json(path, {"status": "passed", "count": 32})

    assert json.loads(path.read_text()) == {"status": "passed", "count": 32}
    assert not list(tmp_path.glob(".*.tmp"))


def test_kld_server_environment_is_tp12_capture_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("K3_KQUANT_CAPTURE_DIR", "/bad/calibration")
    monkeypatch.setenv("VLLM_KQUANT_CAPTURE_DIR", "/bad/calibration")
    gpus = [f"GPU-{index}" for index in range(12)]

    env = build_server_environment(
        model=tmp_path / "model",
        served_model_name="Kimi-K3",
        capture_dir=tmp_path / "chunks",
        host="127.0.0.1",
        port=8123,
        selected_gpus=gpus,
    )

    assert env["CUDA_VISIBLE_DEVICES"] == ",".join(gpus)
    assert env["K3_KLD_CAPTURE_DIR"] == str(tmp_path / "chunks")
    assert env["K3_MAX_NUM_BATCHED_TOKENS"] == "256"
    assert env["K3_DSPARK"] == "0"
    assert "K3_KQUANT_CAPTURE_DIR" not in env
    assert "VLLM_KQUANT_CAPTURE_DIR" not in env
