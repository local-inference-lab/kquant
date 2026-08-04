from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from kquant import constants as C
from kquant.mixed_exl3 import ExpertFormatSpec, SCHEMA
from kquant.pack.mixed_allocation import (
    ALLOCATION_KIND,
    ALLOCATION_SCHEMA_VERSION,
)
from kquant.pack.mixed_package import (
    hybrid_bit_map_from_allocation,
    mixed_tp12_quantization_config,
    package_mixed_tp12_serve_directory,
    validate_mixed_tp12_serve_directory,
)
from kquant.pack import mixed_package
from kquant.pack.mixed_materialize import (
    MATERIALIZATION_BUILD_FILENAME,
    MATERIALIZED_ALLOCATION_FILENAME,
    MATERIALIZED_MANIFEST_FILENAME,
    MATERIALIZED_MODE_VALIDATION_SUMMARY_FILENAME,
    MATERIALIZED_PERFORMANCE_SUMMARY_FILENAME,
    MATERIALIZED_TEACHER_PROXY_SUMMARY_FILENAME,
    layer_filename,
)


def _allocation() -> dict:
    layers = {}
    for layer in C.MOE_LAYERS:
        codes = [ExpertFormatSpec.compressed(0).code] * C.NUM_EXPERTS
        codes[7] = ExpertFormatSpec.mxfp4().code
        layers[str(layer)] = {
            "keep": [7],
            "compressed": [expert for expert in range(C.NUM_EXPERTS) if expert != 7],
            "format_codes": codes,
        }
    return {
        "kind": ALLOCATION_KIND,
        "schema_version": ALLOCATION_SCHEMA_VERSION,
        "layers": layers,
    }


def test_hybrid_bit_map_is_derived_from_exact_format_table() -> None:
    bit_map = hybrid_bit_map_from_allocation(_allocation())

    assert set(bit_map) == {str(layer) for layer in C.MOE_LAYERS}
    assert bit_map["1"] == [3] * 7 + [4] + [3] * (C.NUM_EXPERTS - 8)


def test_mixed_tp12_quantization_config_pins_serving_contract() -> None:
    config = mixed_tp12_quantization_config(_allocation())

    assert config["demoted_format"] == "mixed_exl3_tp12"
    assert config["kept_format"] == "mxfp4_e8m0k32"
    assert config["mixed_exl3_tp12"] == {
        "schema": SCHEMA,
        "tp_size": 12,
        "layer_file_pattern": "mixed-exl3-tp12-layer-{layer:05d}.bin",
    }
    assert config["trellis"]["pair_format"] == "tp12_p24_p33"
    assert config["dense_format"] == "mxfp8"


def test_hybrid_bit_map_rejects_tier_drift() -> None:
    allocation = _allocation()
    allocation["layers"]["1"]["keep"] = []

    with pytest.raises(ValueError, match="disagree"):
        hybrid_bit_map_from_allocation(allocation)


def test_hybrid_bit_map_rejects_invalid_rate_nibble() -> None:
    allocation = _allocation()
    allocation["layers"]["1"]["format_codes"][0] = 0xD0

    with pytest.raises(ValueError, match="invalid code"):
        hybrid_bit_map_from_allocation(allocation)


def test_fresh_serve_package_closes_every_link_and_index(tmp_path, monkeypatch) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    allocation = _allocation()
    (artifact / MATERIALIZED_ALLOCATION_FILENAME).write_text(
        json.dumps(allocation)
    )
    (artifact / MATERIALIZATION_BUILD_FILENAME).write_text("{}")
    (artifact / MATERIALIZED_MANIFEST_FILENAME).write_text("{}")
    (artifact / MATERIALIZED_MODE_VALIDATION_SUMMARY_FILENAME).write_text("{}")
    (artifact / MATERIALIZED_TEACHER_PROXY_SUMMARY_FILENAME).write_text("{}")
    (artifact / MATERIALIZED_PERFORMANCE_SUMMARY_FILENAME).write_text("{}")
    for layer in C.MOE_LAYERS:
        (artifact / layer_filename(layer)).write_bytes(b"slab")

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
    monkeypatch.setattr(
        mixed_package,
        "validate_materialized_artifact",
        lambda root, **kwargs: {
            "artifact": str(root),
            "complete": True,
        },
    )

    destination = tmp_path / "serve"
    package_mixed_tp12_serve_directory(
        artifact,
        destination,
        nonexpert_source=nonexpert,
        official_snapshot=snapshot,
    )
    report = validate_mixed_tp12_serve_directory(destination)

    assert report["complete"] is True
    assert report["layers"] == C.NUM_MOE_LAYERS
    assert (destination / layer_filename(1)).is_symlink()
    assert not list(tmp_path.glob(".serve.*.partial"))

    package_path = destination / "mixed_exl3_tp12_serve.json"
    package = json.loads(package_path.read_text())
    package["artifact_validation"]["stale"] = True
    package_path.write_text(json.dumps(package))
    with pytest.raises(ValueError, match="verdict is stale"):
        validate_mixed_tp12_serve_directory(destination)
    package["artifact_validation"].pop("stale")
    package_path.write_text(json.dumps(package))

    index_path = destination / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"].clear()
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="index does not close"):
        validate_mixed_tp12_serve_directory(destination)
