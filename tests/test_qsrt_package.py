from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from kquant import constants as C
from kquant.exl3_reference import CODEBOOK_SQG_NORMAL_E4M3
from kquant.qsrt import (
    FORMAT_MXFP4,
    KEEP_STORAGE_EXTERNAL_X4T,
    PHASE1_MODE_IDS,
    QSRT_LAYER_HEADER_VERSION,
    ExpertFormatSpec,
)
from kquant.pack.qsrt_slab import layer_filename
from kquant.pack.qsrt_allocation import (
    QSRT_ALLOCATION_KIND,
    QSRT_ALLOCATION_SCHEMA_VERSION,
)
from kquant.pack.qsrt_materialize import (
    QSRT_ALLOCATION_COPY_FILENAME,
    qsrt_layer_closure_filename,
)
from kquant.pack import qsrt_package
from kquant.pack.qsrt_package import (
    QSRT_METADATA_NAMES,
    QSRT_SERVE_MANIFEST,
    package_qsrt_tp12_serve_directory,
    qsrt_hybrid_bit_map,
    qsrt_tp12_quantization_config,
    validate_qsrt_tp12_serve_directory,
)
from kquant.pack.x4t_tp12 import (
    write_x4t_tp12_manifest,
    x4t_tp12_rank_filename,
)
from kquant.x4t import x4t_layer_path


def _allocation() -> dict:
    layers = {}
    for layer in C.MOE_LAYERS:
        codes = [ExpertFormatSpec.compressed(0, 0).code] * C.NUM_EXPERTS
        codes[7] = FORMAT_MXFP4
        layers[str(layer)] = {
            "x4t": [7],
            "compressed": [
                expert for expert in range(C.NUM_EXPERTS) if expert != 7
            ],
            "format_codes": codes,
        }
    return {
        "kind": QSRT_ALLOCATION_KIND,
        "schema_version": QSRT_ALLOCATION_SCHEMA_VERSION,
        "meta": {
            "codec": "QSRT",
            "tp_size": 12,
            "high_tier_storage": KEEP_STORAGE_EXTERNAL_X4T,
            "candidate_codebook": CODEBOOK_SQG_NORMAL_E4M3,
            "candidate_mode_ids": list(PHASE1_MODE_IDS),
        },
        "layers": layers,
    }


def test_qsrt_config_pins_v5_sqg_and_external_x4t() -> None:
    allocation = _allocation()
    bit_map = qsrt_hybrid_bit_map(allocation)
    config = qsrt_tp12_quantization_config(allocation)

    assert bit_map["1"] == [3] * 7 + [4] + [3] * (C.NUM_EXPERTS - 8)
    assert config["kept_storage"] == KEEP_STORAGE_EXTERNAL_X4T
    assert (
        config["qsrt_tp12"]["layer_header_version"]
        == QSRT_LAYER_HEADER_VERSION
    )
    assert config["trellis"]["codebook"] == CODEBOOK_SQG_NORMAL_E4M3
    assert config["trellis"]["mode_ids"] == list(PHASE1_MODE_IDS)
    assert config["trellis"]["separate_r13_r2"] is True
    assert config["qsrt_tp12"]["x4t_tp12_version"] == 1
    assert "x4t_layer_file_pattern" not in config["qsrt_tp12"]


def test_qsrt_bit_map_rejects_tier_drift() -> None:
    allocation = _allocation()
    allocation["layers"]["1"]["x4t"] = []

    with pytest.raises(ValueError, match="disagree"):
        qsrt_hybrid_bit_map(allocation)


def test_fresh_qsrt_package_closes_all_links(tmp_path, monkeypatch) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    for name in QSRT_METADATA_NAMES:
        (artifact / name).write_text("{}\n")
    (artifact / QSRT_ALLOCATION_COPY_FILENAME).write_text(
        json.dumps(_allocation())
    )
    for layer in C.MOE_LAYERS:
        (artifact / layer_filename(layer)).write_bytes(b"slab")
        x4t_layer_path(artifact, layer).write_bytes(b"x4t")
        (artifact / qsrt_layer_closure_filename(layer)).write_text("{}\n")

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
        qsrt_package,
        "validate_qsrt_artifact",
        fake_validation,
    )
    x4t_tp12 = tmp_path / "x4t-tp12"
    x4t_tp12.mkdir()
    x4t_layers = {}
    allocation = _allocation()
    for layer in C.MOE_LAYERS:
        for rank in range(12):
            (x4t_tp12 / x4t_tp12_rank_filename(layer, rank)).write_bytes(b"x")
        x4t_layers[str(layer)] = {
            "layer": layer,
            "expert_ids": allocation["layers"][str(layer)]["x4t"],
        }
    write_x4t_tp12_manifest(
        x4t_tp12,
        artifact=artifact,
        layers=x4t_layers,
    )
    destination = tmp_path / "serve"
    package_qsrt_tp12_serve_directory(
        artifact,
        destination,
        x4t_tp12_source=x4t_tp12,
        nonexpert_source=nonexpert,
        official_snapshot=snapshot,
    )
    report = validate_qsrt_tp12_serve_directory(destination)

    assert report["complete"] is True
    assert report["x4t_tp12_shards"] == C.NUM_MOE_LAYERS * 12
    assert not x4t_layer_path(destination, 1).exists()
    assert (destination / x4t_tp12_rank_filename(1, 0)).is_symlink()
    assert not list(tmp_path.glob(".serve.*.partial"))

    config_path = destination / "config.json"
    config = json.loads(config_path.read_text())
    config["text_config"]["quantization_config"]["kept_storage"] = "inline"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="config does not close"):
        validate_qsrt_tp12_serve_directory(destination)

    package_path = destination / QSRT_SERVE_MANIFEST
    assert package_path.is_file()
