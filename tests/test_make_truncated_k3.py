import importlib.util
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _script_module():
    script = Path(__file__).parents[1] / "scripts" / "make_truncated_k3.py"
    spec = importlib.util.spec_from_file_location("make_truncated_k3", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    tensors = {
        "language_model.model.embed_tokens.weight": torch.ones(3, 4),
        "language_model.model.layers.0.weight": torch.full((2, 2), 1.0),
        "language_model.model.layers.1.weight": torch.full((2, 2), 2.0),
        "language_model.model.layers.2.weight": torch.full((2, 2), 3.0),
        "language_model.lm_head.weight": torch.ones(3, 4),
    }
    save_file(tensors, str(source / "model.safetensors"))
    weight_map = {name: "model.safetensors" for name in tensors}
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )
    config = {
        "text_config": {
            "num_hidden_layers": 3,
            "num_experts": 4,
            "linear_attn_config": {
                "kda_layers": [1, 2, 3],
                "full_attn_layers": [3],
            },
            "quantization_config": {
                "quant_method": "compressed-tensors",
            },
        }
    }
    (source / "config.json").write_text(json.dumps(config))
    (source / "tokenizer.json").write_text("{}")
    return source


def test_build_is_complete_deterministic_and_drops_later_layers(
    tmp_path: Path,
) -> None:
    module = _script_module()
    source_dir = _source(tmp_path)
    source = module.source_from_directory(source_dir)
    destination = tmp_path / "truncated"

    report = module.build(
        source=source,
        destination=destination,
        n_layers=2,
        quant_mode="native",
        shard_bytes=1,
    )

    index = json.loads(
        (destination / "model.safetensors.index.json").read_text()
    )
    assert report["layers"] == 2
    assert not any(".layers.2." in name for name in index["weight_map"])
    assert (destination / "tokenizer.json").read_text() == "{}"
    config = json.loads((destination / "config.json").read_text())
    text_config = config["text_config"]
    assert text_config["num_hidden_layers"] == 2
    assert text_config["linear_attn_config"]["kda_layers"] == [1, 2]
    for filename in set(index["weight_map"].values()):
        with safe_open(str(destination / filename), framework="pt") as handle:
            assert list(handle.keys())


def test_hybrid_all_kept_map_uses_model_expert_count() -> None:
    module = _script_module()
    config = {
        "num_hidden_layers": 4,
        "num_experts": 5,
        "linear_attn_config": {
            "kda_layers": [1, 2, 3, 4],
            "full_attn_layers": [],
        },
    }

    result = module.truncate_config(
        config,
        n_layers=3,
        quant_mode="hybrid-all-kept",
    )

    bit_map = result["quantization_config"]["hybrid_bit_map"]
    assert bit_map == {"1": [4] * 5, "2": [4] * 5}
