from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from kquant.dspark_mxfp8 import (
    convert_dspark_checkpoint,
    expected_mxfp8_weights,
)
from kquant.mxfp8 import mxfp8_dequantize_cpu, mxfp8_quantize_cpu


def _fake_source(path: Path) -> tuple[dict, dict[str, torch.Tensor]]:
    path.mkdir()
    config = {
        "architectures": ["DSparkDraftModel"],
        "auto_map": {"AutoModel": "dspark.DSparkDraftModel"},
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "vocab_size": 128,
        "dflash_config": {"target_layer_ids": [1, 2]},
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "README.md").write_text("source card\n", encoding="utf-8")

    generator = torch.Generator().manual_seed(7)
    tensors: dict[str, torch.Tensor] = {}
    for name in expected_mxfp8_weights(config):
        if name == "fc.weight":
            shape = (32, 64)
        elif ".mlp.gate_proj." in name or ".mlp.up_proj." in name:
            shape = (64, 32)
        elif ".mlp.down_proj." in name:
            shape = (32, 64)
        else:
            shape = (32, 32)
        tensors[name] = torch.randn(shape, generator=generator).to(torch.bfloat16)

    tensors.update(
        {
            "layers.0.input_layernorm.weight": torch.randn(32, generator=generator).to(
                torch.bfloat16
            ),
            "markov_head.markov_w1.weight": torch.randn(
                128, 32, generator=generator
            ).to(torch.bfloat16),
            "markov_head.markov_w2.weight": torch.randn(
                128, 32, generator=generator
            ).to(torch.bfloat16),
            "confidence_head.proj.weight": torch.randn(1, 64, generator=generator).to(
                torch.float32
            ),
            "confidence_head.proj.bias": torch.randn(1, generator=generator).to(
                torch.float32
            ),
        }
    )
    save_file(tensors, str(path / "model.safetensors"), metadata={"format": "pt"})
    return config, tensors


def _load_indexed_tensors(path: Path) -> dict[str, torch.Tensor]:
    index = json.loads((path / "model.safetensors.index.json").read_text())
    result: dict[str, torch.Tensor] = {}
    for shard in sorted(set(index["weight_map"].values())):
        with safe_open(path / shard, framework="pt", device="cpu") as handle:
            result.update(
                {
                    name: handle.get_tensor(name)
                    for name in handle.keys()  # noqa: SIM118 - not iterable
                }
            )
    return result


def test_k3_mla_expected_weights_and_config_are_preserved(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    config = {
        "architectures": ["K3DSparkModel"],
        "model_type": "k3_dspark",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "vocab_size": 128,
    }
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
    generator = torch.Generator().manual_seed(11)
    tensors = {
        name: torch.randn((32, 32), generator=generator).to(torch.bfloat16)
        for name in expected_mxfp8_weights(config)
    }
    tensors["layers.0.input_layernorm.weight"] = torch.ones(32)
    save_file(tensors, str(source / "model.safetensors"))

    destination = tmp_path / "output"
    manifest = convert_dspark_checkpoint(
        source,
        destination,
        max_shard_size=8_000,
        progress=None,
    )

    expected = {
        "context_proj.weight",
        "layers.0.self_attn.q_a_proj.weight",
        "layers.0.self_attn.kv_a_proj_with_mqa.weight",
        "layers.0.self_attn.q_b_proj.weight",
        "layers.0.self_attn.kv_b_proj.weight",
        "layers.0.self_attn.o_proj.weight",
        "layers.0.mlp.gate_proj.weight",
        "layers.0.mlp.up_proj.weight",
        "layers.0.mlp.down_proj.weight",
    }
    assert expected_mxfp8_weights(config) == expected
    rewritten = json.loads((destination / "config.json").read_text())
    assert rewritten["architectures"] == ["K3DSparkModel"]
    assert rewritten["model_type"] == "k3_dspark"
    assert "vllm_virtual_tp_profile" not in rewritten
    assert manifest["architecture"] == "K3DSparkModel"
    assert manifest["runtime"]["virtual_tp_profile"] is None


def test_k3_mla_heads_are_zero_padded_for_tensor_parallelism(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    config = {
        "architectures": ["K3DSparkModel"],
        "model_type": "k3_dspark",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 3,
        "num_key_value_heads": 3,
        "qk_nope_head_dim": 4,
        "qk_rope_head_dim": 4,
        "v_head_dim": 8,
        "vocab_size": 128,
    }
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {
        name: torch.ones((32, 32), dtype=torch.bfloat16)
        for name in expected_mxfp8_weights(config)
    }
    tensors["layers.0.self_attn.q_b_proj.weight"] = torch.ones(
        (24, 32), dtype=torch.bfloat16
    )
    tensors["layers.0.self_attn.kv_b_proj.weight"] = torch.ones(
        (36, 32), dtype=torch.bfloat16
    )
    tensors["layers.0.self_attn.o_proj.weight"] = torch.ones(
        (32, 24), dtype=torch.bfloat16
    )
    tensors["layers.0.input_layernorm.weight"] = torch.ones(32)
    tensors["embed_tokens.weight"] = torch.ones((128, 32))
    tensors["confidence_head.proj.weight"] = torch.ones((1, 64))
    save_file(tensors, str(source / "model.safetensors"))

    destination = tmp_path / "output"
    manifest = convert_dspark_checkpoint(
        source,
        destination,
        max_shard_size=8_000,
        tensor_parallel_size=2,
        progress=None,
    )

    rewritten = json.loads((destination / "config.json").read_text())
    assert rewritten["original_num_attention_heads"] == 3
    assert rewritten["num_attention_heads"] == 4
    assert rewritten["num_key_value_heads"] == 4
    assert manifest["runtime"]["padded_num_attention_heads"] == 4
    output = _load_indexed_tensors(destination)
    assert "embed_tokens.weight" not in output
    assert "confidence_head.proj.weight" not in output
    assert output["layers.0.self_attn.q_b_proj.weight"].shape == (32, 32)
    assert output["layers.0.self_attn.kv_b_proj.weight"].shape == (48, 32)
    assert output["layers.0.self_attn.o_proj.weight"].shape == (32, 32)


def test_k3_mlp_is_zero_padded_for_tensor_parallelism(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    config = {
        "architectures": ["K3DSparkModel"],
        "model_type": "k3_dspark",
        "hidden_size": 32,
        "intermediate_size": 80,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "qk_nope_head_dim": 4,
        "qk_rope_head_dim": 4,
        "v_head_dim": 8,
        "vocab_size": 128,
    }
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {
        name: torch.ones((32, 32), dtype=torch.bfloat16)
        for name in expected_mxfp8_weights(config)
    }
    tensors["layers.0.mlp.gate_proj.weight"] = torch.ones((80, 32))
    tensors["layers.0.mlp.up_proj.weight"] = torch.ones((80, 32))
    tensors["layers.0.mlp.down_proj.weight"] = torch.ones((32, 80))
    tensors["layers.0.self_attn.kv_b_proj.weight"] = torch.ones((48, 32))
    tensors["layers.0.input_layernorm.weight"] = torch.ones(32)
    save_file(tensors, str(source / "model.safetensors"))

    destination = tmp_path / "output"
    manifest = convert_dspark_checkpoint(
        source,
        destination,
        max_shard_size=8_000,
        tensor_parallel_size=8,
        progress=None,
    )

    rewritten = json.loads((destination / "config.json").read_text())
    assert rewritten["original_intermediate_size"] == 80
    assert rewritten["intermediate_size"] == 256
    assert manifest["runtime"]["padded_intermediate_size"] == 256
    output = _load_indexed_tensors(destination)
    assert output["layers.0.mlp.gate_proj.weight"].shape == (256, 32)
    assert output["layers.0.mlp.up_proj.weight"].shape == (256, 32)
    assert output["layers.0.mlp.down_proj.weight"].shape == (32, 256)


def test_mxfp8_quantize_layout_and_roundtrip() -> None:
    weight = torch.linspace(-3.0, 3.0, 128, dtype=torch.float32).view(2, 64)
    quantized, scale = mxfp8_quantize_cpu(weight)

    assert quantized.dtype == torch.float8_e4m3fn
    assert quantized.shape == weight.shape
    assert scale.dtype == torch.uint8
    assert scale.shape == (2, 2)
    restored = mxfp8_dequantize_cpu(quantized, scale)
    assert torch.mean(torch.abs(restored - weight)).item() < 0.04


def test_convert_dspark_checkpoint_is_sharded_and_complete(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "output"
    config, source_tensors = _fake_source(source)

    manifest = convert_dspark_checkpoint(
        source,
        destination,
        max_shard_size=12_000,
        progress=None,
    )

    rewritten = json.loads((destination / "config.json").read_text())
    assert rewritten["architectures"] == ["Qwen3DSparkModel"]
    assert "auto_map" not in rewritten
    assert rewritten["draft_vocab_size"] == config["vocab_size"]
    assert rewritten["vllm_virtual_tp_profile"] == "dense-gqa"
    assert rewritten["quantization_config"] == {
        "activation_scheme": "dynamic",
        "dense_format": "mxfp8",
        "ignored_layers": ["markov_head", "confidence_head", "lm_head"],
        "quant_method": "fp8",
    }

    assert len(manifest["shards"]) > 1
    assert manifest["quantized_weight_count"] == 15
    assert manifest["runtime"]["draft_load_config"] == {"load_format": "safetensors"}
    assert (destination / "README.md").read_text() == "source card\n"

    output = _load_indexed_tensors(destination)
    targets = expected_mxfp8_weights(config)
    assert set(output) == set(source_tensors) | {
        name.removesuffix(".weight") + ".weight_scale" for name in targets
    }
    for name in targets:
        assert output[name].dtype == torch.float8_e4m3fn
        scale_name = name.removesuffix(".weight") + ".weight_scale"
        assert output[scale_name].dtype == torch.uint8
        assert output[scale_name].shape == (
            output[name].shape[0],
            output[name].shape[1] // 32,
        )
    for name in set(source_tensors).difference(targets):
        assert torch.equal(output[name], source_tensors[name])


def test_convert_refuses_existing_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "output"
    _fake_source(source)
    destination.mkdir()

    with pytest.raises(FileExistsError, match="destination already exists"):
        convert_dspark_checkpoint(source, destination, progress=None)
