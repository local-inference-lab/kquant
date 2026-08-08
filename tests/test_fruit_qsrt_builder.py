from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.build_fruit_qsrt_model as builder


def test_current_encoder_provenance_hashes_local_source_trees(tmp_path: Path) -> None:
    exllamav3_root = tmp_path / "exllamav3-root"
    package = exllamav3_root / "exllamav3"
    package.mkdir(parents=True)
    source = package / "codec.py"
    source.write_text("CODEBOOK = 1\n", encoding="utf-8")
    calibration = SimpleNamespace(
        fingerprint="calibration-fingerprint",
        capture_id="capture-id",
        manifest_sha256="manifest-sha256",
    )

    before = builder.current_encoder_provenance(
        exllamav3_root=exllamav3_root,
        calibration=calibration,
    )
    source.write_text("CODEBOOK = 2\n", encoding="utf-8")
    after = builder.current_encoder_provenance(
        exllamav3_root=exllamav3_root,
        calibration=calibration,
    )

    assert before["exllamav3_source_sha256"] != after["exllamav3_source_sha256"]
    assert before["fingerprint"] != after["fingerprint"]


def test_encoder_fingerprint_is_content_addressed_not_revision_addressed(
    monkeypatch, tmp_path: Path
) -> None:
    package = tmp_path / "exllamav3" / "exllamav3"
    package.mkdir(parents=True)
    (package / "codec.py").write_text("CODEBOOK = 1\n", encoding="utf-8")
    calibration = SimpleNamespace(
        fingerprint="calibration-fingerprint",
        capture_id="capture-id",
        manifest_sha256="manifest-sha256",
    )
    revisions = iter(("first-revision", "second-revision"))
    monkeypatch.setattr(builder, "_git_revision", lambda _root: next(revisions))

    first = builder.current_encoder_provenance(
        exllamav3_root=package.parent,
        calibration=calibration,
    )
    second = builder.current_encoder_provenance(
        exllamav3_root=package.parent,
        calibration=calibration,
    )

    assert first["kquant_revision"] != second["kquant_revision"]
    assert first["fingerprint"] == second["fingerprint"]


def _rate_sweep(
    *,
    source: dict[str, object] | None = None,
    calibration: dict[str, str] | None = None,
    encoder: dict[str, object] | None = None,
) -> dict[str, object]:
    def endpoint(bits: int) -> dict[str, object]:
        error = float(5 - bits)
        matrices = [
            {
                "matrix": matrix,
                "weight_squared_error": error,
                "weight_reference_energy": 10.0,
                "captured_dense_h_numerator": error * 2,
                "captured_dense_h_denominator": 20.0,
            }
            for matrix in ("w1", "w3", "w2")
        ]
        return {
            "bits": bits,
            "aggregate": {
                "bytes_before_layer_deduplication": bits * 100,
                "bpw_before_layer_deduplication": bits + 0.046875,
            },
            "matrices": matrices,
            "routed_function": {
                "validation": {
                    "routed_sse": error * 3,
                    "reference_energy": 30.0,
                }
            },
        }

    return {
        "schema": builder._RATE_SWEEP_SCHEMA,
        "complete": True,
        "signature": {
            "schema": builder._RATE_SWEEP_SCHEMA,
            "source": source or {},
            "calibration": calibration or {},
            "encoder": encoder or {},
            "rates": [2, 3, 4],
            "assignments": [[3, 0]],
        },
        "results": {
            "3:0": {
                "status": "measured",
                "rates": {
                    "K2": endpoint(2),
                    "K3": endpoint(3),
                    "K4": endpoint(4),
                },
            }
        },
    }


def test_model_card_uses_sealed_calibration_and_layer_evidence(monkeypatch) -> None:
    monkeypatch.setattr(builder, "LAYERS", (3,))
    monkeypatch.setattr(builder, "EXPERTS", 2)
    calibration = SimpleNamespace(
        capture_id="capture-id",
        manifest_sha256="calibration-manifest",
        manifest={"documents": [{"tokens": 7}, {"tokens": 11}]},
    )
    layers = {
        "3": {
            "format_counts": {"R13=0,R2=0": 1, "R13=1,R2=0": 1},
            "experts": [
                {"encode_elapsed_seconds": 1.25, "peak_cuda_bytes": 1 << 30},
                {"encode_elapsed_seconds": 2.75, "peak_cuda_bytes": 2 << 30},
            ],
        }
    }
    producer = {
        "encoder": {"kquant_revision": "kquant-revision"},
        "runtime": {
            "b12x_revision": "b12x-revision",
            "vllm_revision": "vllm-revision",
        },
    }

    card = builder._render_model_card(
        source_evidence={
            "source_kind": "safetensors_manifest",
            "source_sha256": "source-digest",
        },
        calibration=calibration,
        producer=producer,
        rate_sweep=_rate_sweep(),
        layers=layers,
    )

    assert "| `R13=0,R2=0` | 1 |" in card
    assert "| `R13=1,R2=0` | 1 |" in card
    assert "capture-id" in card
    assert "calibration-manifest" in card
    assert "2 documents /" in card and "18 tokens" in card
    assert "2 experts, 4.00 GPU-seconds" in card
    assert "2.000 GiB peak CUDA allocation" in card
    assert "Adjacent-rate evidence" in card
    assert "1 of 1 predeclared assignments" in card
    assert "__" not in card


def test_rate_sweep_validation_binds_build_provenance(tmp_path: Path) -> None:
    source = {"source_kind": "manifest", "source_sha256": "a" * 64}
    calibration_identity = {
        "capture_id": "capture-id",
        "fingerprint": "calibration-fingerprint",
        "manifest_sha256": "b" * 64,
    }
    calibration = SimpleNamespace(
        capture_id=calibration_identity["capture_id"],
        fingerprint=calibration_identity["fingerprint"],
        manifest_sha256=calibration_identity["manifest_sha256"],
    )
    encoder = {
        "fingerprint_schema": builder._ENCODER_FINGERPRINT_SCHEMA,
        "fingerprint": "c" * 64,
    }
    producer = {"encoder": encoder}
    payload = _rate_sweep(
        source=source,
        calibration=calibration_identity,
        encoder=encoder,
    )
    path = tmp_path / "rates.json"
    path.write_text(builder._canonical_json(payload), encoding="utf-8")

    assert (
        builder._validate_rate_sweep(
            path,
            source_evidence=source,
            calibration=calibration,
            producer=producer,
        )
        == payload
    )

    producer["encoder"] = {
        "fingerprint_schema": builder._ENCODER_FINGERPRINT_SCHEMA,
        "fingerprint": "d" * 64,
    }
    with pytest.raises(ValueError, match="provenance"):
        builder._validate_rate_sweep(
            path,
            source_evidence=source,
            calibration=calibration,
            producer=producer,
        )


def test_package_files_include_evaluation_but_exclude_resume_cache(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    (evaluation / "report.json").write_text("{}", encoding="utf-8")
    parts = tmp_path / ".qsrt-parts"
    parts.mkdir()
    (parts / "resume.bin").write_bytes(b"resume")

    files = builder._package_files(tmp_path)

    assert set(files) == {"config.json", "evaluation/report.json"}


def test_remove_part_cache_preserves_package_files(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    parts = tmp_path / ".qsrt-parts" / "layer-003"
    parts.mkdir(parents=True)
    (parts / "expert-000.safetensors").write_bytes(b"resume")

    builder._remove_part_cache(tmp_path)

    assert not (tmp_path / ".qsrt-parts").exists()
    assert config.read_text(encoding="utf-8") == "{}"


def test_seed_parts_validate_source_read_only_before_copy(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(builder, "LAYERS", (3,))
    monkeypatch.setattr(builder, "EXPERTS", 1)
    seed = tmp_path / "seed"
    seed_layer = seed / "layer-003"
    seed_layer.mkdir(parents=True)
    source_tensor = seed_layer / "expert-000.safetensors"
    source_manifest = seed_layer / "expert-000.json"
    source_tensor.write_bytes(b"authenticated tensor")
    source_manifest.write_text('{"authenticated": true}', encoding="utf-8")
    calls = []

    def validate(tensor_path, manifest_path, **kwargs):
        calls.append((tensor_path, manifest_path, kwargs))
        return {"status": "valid"}

    monkeypatch.setattr(builder, "_validate_part", validate)
    output = tmp_path / "output"

    assert builder._seed_parts(output, seed, "a" * 64, "b" * 64) == 1
    assert calls == [
        (
            source_tensor,
            source_manifest,
            {
                "layer": 3,
                "expert": 0,
                "source_sha256": "a" * 64,
                "encoder_fingerprint": "b" * 64,
                "repair": False,
            },
        )
    ]
    assert source_tensor.read_bytes() == b"authenticated tensor"
    assert source_manifest.read_text(encoding="utf-8") == '{"authenticated": true}'
    assert (
        output / ".qsrt-parts/layer-003/expert-000.safetensors"
    ).read_bytes() == b"authenticated tensor"
