import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from kquant.artifact import validate_exl3_artifact


def _name(layer: int, expert: int, matrix: str, suffix: str) -> str:
    return (
        f"language_model.model.layers.{layer}.block_sparse_moe."
        f"experts.{expert}.{matrix}.{suffix}"
    )


def _mini_artifact(tmp_path: Path) -> tuple[Path, Path]:
    artifact = tmp_path / "artifact"
    serve = tmp_path / "serve"
    artifact.mkdir()
    serve.mkdir()
    hidden = intermediate = 32
    kept_tensors = {}
    exl3_tensors = {}
    for matrix in ("w1", "w2", "w3"):
        kept_tensors[_name(1, 0, matrix, "weight_packed")] = torch.zeros(
            intermediate if matrix != "w2" else hidden,
            (hidden if matrix != "w2" else intermediate) // 2,
            dtype=torch.uint8,
        )
        kept_tensors[_name(1, 0, matrix, "weight_scale")] = torch.zeros(
            intermediate if matrix != "w2" else hidden,
            (hidden if matrix != "w2" else intermediate) // 32,
            dtype=torch.uint8,
        )
        exl3_tensors[_name(1, 1, matrix, "exl3_trellis")] = torch.zeros(
            (
                hidden // 16,
                intermediate // 16,
                48,
            )
            if matrix != "w2"
            else (
                intermediate // 16,
                hidden // 16,
                48,
            ),
            dtype=torch.int16,
        )
        exl3_tensors[_name(1, 1, matrix, "exl3_suh")] = torch.zeros(
            hidden if matrix != "w2" else intermediate,
            dtype=torch.float16,
        )
        exl3_tensors[_name(1, 1, matrix, "exl3_svh")] = torch.zeros(
            intermediate if matrix != "w2" else hidden,
            dtype=torch.float16,
        )
    kept_file = "keep-mxfp4-00000.safetensors"
    exl3_file = "exl3-layer-00001.safetensors"
    save_file(kept_tensors, str(artifact / kept_file))
    save_file(exl3_tensors, str(artifact / exl3_file))
    (serve / kept_file).symlink_to(artifact / kept_file)
    (serve / exl3_file).symlink_to(artifact / exl3_file)
    (artifact / "allocation-exl3.json").write_text(
        json.dumps(
            {
                "meta": {"n_keep": 1, "n_exl3": 1},
                "layers": {"1": {"keep": [0], "exl3": [1]}},
            }
        )
    )
    (artifact / "kquant_exl3_manifest.json").write_text(
        json.dumps(
            {
                "kind": "kquant_exl3_artifact",
                "bits": 3,
                "codebook": "mcg",
                "mcg_mult": 7,
            }
        )
    )
    config = {
        "text_config": {
            "num_hidden_layers": 2,
            "num_experts": 2,
            "routed_expert_hidden_size": hidden,
            "moe_intermediate_size": intermediate,
            "quantization_config": {
                "kept_format": "mxfp4_e8m0k32",
                "demoted_format": "exl3_3",
                "hybrid_bit_map": {"1": [4, 3]},
            },
        }
    }
    (serve / "config.json").write_text(json.dumps(config))
    weight_map = {
        **{name: kept_file for name in kept_tensors},
        **{name: exl3_file for name in exl3_tensors},
    }
    (serve / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )
    return artifact, serve


def test_validate_exl3_artifact_accepts_complete_contract(
    tmp_path: Path,
) -> None:
    artifact, serve = _mini_artifact(tmp_path)

    report = validate_exl3_artifact(artifact, serve)

    assert report["pass"] is True
    assert report["allocation"]["assignments"] == 2
    assert report["scanned_expert_tensors"] == 15


def test_validate_exl3_artifact_rejects_config_allocation_drift(
    tmp_path: Path,
) -> None:
    artifact, serve = _mini_artifact(tmp_path)
    config_path = serve / "config.json"
    config = json.loads(config_path.read_text())
    config["text_config"]["quantization_config"]["hybrid_bit_map"]["1"] = [3, 4]
    config_path.write_text(json.dumps(config))

    report = validate_exl3_artifact(artifact, serve)

    assert report["pass"] is False
    assert any("hybrid_bit_map" in issue for issue in report["issue_samples"])
