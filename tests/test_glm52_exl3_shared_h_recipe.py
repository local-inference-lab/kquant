from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


RECIPE = Path(__file__).parents[1] / "recipes" / "glm52_exl3_shared_h"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recipe_patches_reproduce_pinned_encoder(tmp_path):
    published = Path(
        "/root/.cache/huggingface/hub/"
        "models--brandonmusic--GLM-5.2-EXL3-TR3-3.0bpw/"
        "snapshots/9297b9f1d53af5c67cffa01e30cc071a1ff7144b/"
        "calibration_encoder"
    )
    if not published.is_dir():
        pytest.skip("published calibration bundle is not available")
    prepare = _load_module(
        "prepare_shared_h_encoder", RECIPE / "prepare_shared_h_encoder.py"
    )

    prepare.prepare(published, tmp_path / "encoder")

    prepare.verify_files(tmp_path / "encoder", prepare.OUTPUT_SHA256, "test output")


def test_shared_h_algebra_seed_and_artifact_contract(tmp_path):
    published = Path(
        "/root/.cache/huggingface/hub/"
        "models--brandonmusic--GLM-5.2-EXL3-TR3-3.0bpw/"
        "snapshots/9297b9f1d53af5c67cffa01e30cc071a1ff7144b/"
        "calibration_encoder"
    )
    if not published.is_dir():
        pytest.skip("published calibration bundle is not available")
    prepare = _load_module(
        "prepare_shared_h_encoder_2", RECIPE / "prepare_shared_h_encoder.py"
    )
    destination = tmp_path / "encoder"
    prepare.prepare(published, destination)
    encoder = _load_module("shared_h_encoder", destination / "encode_tr3_v31.py")
    encoder._lazy_torch = lambda: (torch, None)

    weight = torch.randn(8, 8)
    suh = torch.rand(8, 1) + 0.25
    svh = torch.rand(1, 8) + 0.25
    scale = 1.37
    standard = encoder.regularize_post(
        weight.clone(), suh.clone(), svh.clone(), scale, False
    )
    shared = encoder.regularize_post(
        weight.clone(), suh.clone(), svh.clone(), scale, True
    )
    torch.testing.assert_close(
        standard[0] * standard[1] * standard[2],
        shared[0] * shared[1] * shared[2],
        rtol=2e-6,
        atol=2e-7,
    )

    _, gate_a = encoder._shared_h_encode_kwargs(40, 7, "gate_proj", 0, 2)
    _, gate_b = encoder._shared_h_encode_kwargs(40, 99, "gate_proj", 0, 2)
    assert gate_a["su_seed"] == gate_b["su_seed"]
    assert gate_a["sv_seed"] != gate_b["sv_seed"]
    assert gate_a["g_scale_into_svh"]
    _, down_a = encoder._shared_h_encode_kwargs(40, 7, "down_proj", 2, 2)
    _, down_b = encoder._shared_h_encode_kwargs(40, 99, "down_proj", 2, 2)
    assert down_a["sv_seed"] == down_b["sv_seed"]

    adapter = _load_module("shared_h_adapter", destination / "encode_b300.py")
    adapter.BASE = SimpleNamespace(
        NUM_EXPERTS=256,
        PROJS=("gate_proj", "up_proj", "down_proj"),
        TP=4,
        SLICE=512,
        HIDDEN=6144,
    )
    entries = adapter._expected_layer_entries(3)
    assert len(entries) == adapter.EXPECTED_LAYER_TENSORS == 9228
    assert "model.layers.3.mlp.experts.shared_h.gate_proj.rank0.suh" in entries
    assert "model.layers.3.mlp.experts.0.gate_proj.rank0.suh" not in entries
    assert "model.layers.3.mlp.experts.0.gate_proj.rank0.svh" in entries
    assert "model.layers.3.mlp.experts.shared_h.down_proj.rank3.svh" in entries
