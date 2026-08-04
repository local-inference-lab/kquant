from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from kquant import constants as C
from kquant.exl3_reference import CODEBOOK_SQG_NORMAL_E4M3
from kquant.mixed_exl3 import (
    FORMAT_MXFP4,
    KEEP_STORAGE_EXTERNAL_X4,
    PHASE1_MODE_IDS,
    SQRT_C_LAYER_HEADER_VERSION,
    ExpertFormatSpec,
)
from kquant.pack.mixed_materialize import layer_filename
from kquant.pack.sqrt_c_allocation import (
    SQRT_C_ALLOCATION_KIND,
    SQRT_C_ALLOCATION_SCHEMA_VERSION,
)
from kquant.pack.sqrt_c_materialize import (
    SQRT_C_ALLOCATION_COPY_FILENAME,
    sqrt_c_layer_closure_filename,
)
from kquant.pack import sqrt_c_package
from kquant.pack.sqrt_c_package import (
    SQRT_C_METADATA_NAMES,
    SQRT_C_SERVE_MANIFEST,
    package_sqrt_c_tp12_serve_directory,
    sqrt_c_hybrid_bit_map,
    sqrt_c_tp12_quantization_config,
    validate_sqrt_c_tp12_serve_directory,
)
from kquant.x4 import x4_layer_path


def _allocation() -> dict:
    layers = {}
    for layer in C.MOE_LAYERS:
        codes = [ExpertFormatSpec.compressed(0, 0).code] * C.NUM_EXPERTS
        codes[7] = FORMAT_MXFP4
        layers[str(layer)] = {
            "x4": [7],
            "keep": [7],
            "compressed": [
                expert for expert in range(C.NUM_EXPERTS) if expert != 7
            ],
            "format_codes": codes,
        }
    return {
        "kind": SQRT_C_ALLOCATION_KIND,
        "schema_version": SQRT_C_ALLOCATION_SCHEMA_VERSION,
        "meta": {
            "codec": "SQRT-C",
            "tp_size": 12,
            "high_tier_storage": KEEP_STORAGE_EXTERNAL_X4,
            "candidate_codebook": CODEBOOK_SQG_NORMAL_E4M3,
            "candidate_mode_ids": list(PHASE1_MODE_IDS),
        },
        "layers": layers,
    }


def test_sqrt_c_config_pins_v5_sqg_and_external_x4() -> None:
    allocation = _allocation()
    bit_map = sqrt_c_hybrid_bit_map(allocation)
    config = sqrt_c_tp12_quantization_config(allocation)

    assert bit_map["1"] == [3] * 7 + [4] + [3] * (C.NUM_EXPERTS - 8)
    assert config["kept_storage"] == KEEP_STORAGE_EXTERNAL_X4
    assert (
        config["mixed_exl3_tp12"]["layer_header_version"]
        == SQRT_C_LAYER_HEADER_VERSION
    )
    assert config["trellis"]["codebook"] == CODEBOOK_SQG_NORMAL_E4M3
    assert config["trellis"]["mode_ids"] == list(PHASE1_MODE_IDS)
    assert config["trellis"]["separate_r13_r2"] is True


def test_sqrt_c_bit_map_rejects_tier_drift() -> None:
    allocation = _allocation()
    allocation["layers"]["1"]["x4"] = []

    with pytest.raises(ValueError, match="disagree"):
        sqrt_c_hybrid_bit_map(allocation)


def test_fresh_sqrt_c_package_closes_all_links(tmp_path, monkeypatch) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    for name in SQRT_C_METADATA_NAMES:
        (artifact / name).write_text("{}\n")
    (artifact / SQRT_C_ALLOCATION_COPY_FILENAME).write_text(
        json.dumps(_allocation())
    )
    for layer in C.MOE_LAYERS:
        (artifact / layer_filename(layer)).write_bytes(b"slab")
        x4_layer_path(artifact, layer).write_bytes(b"x4")
        (artifact / sqrt_c_layer_closure_filename(layer)).write_text("{}\n")

    nonexpert = tmp_path / "nonexpert"
    nonexpert.mkdir()
    save_file(
        {"language_model.model.embed_tokens.weight": torch.zeros(2)},
        str(nonexpert / "00-nonexpert-00000.safetensors"),
    )
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(json.dumps({"text_config": {}}))
    (snapshot / "tokenizer_config.json").write_text("{}")

    def fake_validation(root, **kwargs):
        return {"artifact": str(root), "complete": True}

    monkeypatch.setattr(
        sqrt_c_package,
        "validate_sqrt_c_artifact",
        fake_validation,
    )
    destination = tmp_path / "serve"
    package_sqrt_c_tp12_serve_directory(
        artifact,
        destination,
        nonexpert_source=nonexpert,
        official_snapshot=snapshot,
    )
    report = validate_sqrt_c_tp12_serve_directory(destination)

    assert report["complete"] is True
    assert report["x4_sidecars"] == C.NUM_MOE_LAYERS
    assert x4_layer_path(destination, 1).is_symlink()
    assert not list(tmp_path.glob(".serve.*.partial"))

    config_path = destination / "config.json"
    config = json.loads(config_path.read_text())
    config["text_config"]["quantization_config"]["kept_storage"] = "inline"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="config does not close"):
        validate_sqrt_c_tp12_serve_directory(destination)

    package_path = destination / SQRT_C_SERVE_MANIFEST
    assert package_path.is_file()
