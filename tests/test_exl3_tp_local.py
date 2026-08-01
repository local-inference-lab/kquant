from __future__ import annotations

import json

import pytest
import torch

from kquant.pack.exl3 import merge_tp_local_results, split_tp_local_weight
from scripts.pack_exl3_12gpu import (
    completion_marker,
    layer_is_complete,
    marker_contract,
)
from scripts.package_exl3_serve_dir import trellis_config


def _encoded(*, matrix: str, rank: int) -> tuple[float, dict[str, torch.Tensor]]:
    if matrix == "w2":
        trellis = torch.full((12, 2, 48), rank, dtype=torch.int16)
        suh = torch.full((192,), rank + 1, dtype=torch.float16)
        svh = torch.arange(32, dtype=torch.float16)
    else:
        trellis = torch.full((2, 12, 48), rank, dtype=torch.int16)
        suh = torch.arange(32, dtype=torch.float16)
        svh = torch.full((192,), rank + 1, dtype=torch.float16)
    return float(rank + 1), {
        "trellis": trellis,
        "suh": suh,
        "svh": svh,
        "mcg": torch.tensor(7, dtype=torch.int32),
    }


@pytest.mark.parametrize("matrix", ["w1", "w3"])
def test_split_tp_local_column_parallel_uses_exact_192_shards(matrix: str) -> None:
    weight = torch.arange(32 * 384, dtype=torch.float32).view(32, 384)
    shards = split_tp_local_weight(weight, matrix, 2)
    assert [tuple(shard.shape) for shard in shards] == [(32, 192), (32, 192)]
    assert torch.equal(torch.cat(shards, dim=1), weight)


def test_split_tp_local_row_parallel_uses_exact_192_shards() -> None:
    weight = torch.arange(384 * 32, dtype=torch.float32).view(384, 32)
    shards = split_tp_local_weight(weight, "w2", 2)
    assert [tuple(shard.shape) for shard in shards] == [(192, 32), (192, 32)]
    assert torch.equal(torch.cat(shards, dim=0), weight)


@pytest.mark.parametrize("matrix", ["w1", "w3", "w2"])
def test_merge_tp_local_results_preserves_old_tensor_shapes(matrix: str) -> None:
    error, tensors = merge_tp_local_results(
        matrix, [_encoded(matrix=matrix, rank=0), _encoded(matrix=matrix, rank=1)]
    )
    assert error == 1.5
    if matrix == "w2":
        assert tuple(tensors["trellis"].shape) == (24, 2, 48)
        assert tuple(tensors["suh"].shape) == (384,)
        assert tuple(tensors["svh"].shape) == (32,)
    else:
        assert tuple(tensors["trellis"].shape) == (2, 24, 48)
        assert tuple(tensors["suh"].shape) == (32,)
        assert tuple(tensors["svh"].shape) == (384,)


def test_merge_tp_local_rejects_inconsistent_replicated_scale() -> None:
    results = [_encoded(matrix="w2", rank=0), _encoded(matrix="w2", rank=1)]
    results[1][1]["svh"][0] = -1
    with pytest.raises(ValueError, match="replicated svh"):
        merge_tp_local_results("w2", results)


def test_split_tp_local_rejects_non_h64_tail() -> None:
    with pytest.raises(ValueError, match="H128"):
        split_tp_local_weight(torch.empty(16, 160), "w1", 1)


def test_packager_preserves_tp_local_transform_contract() -> None:
    config = trellis_config(
        {
            "mcg_mult": 0xCBAC1FED,
            "compatible_tp_sizes": [16],
            "tp_local_quantization": True,
            "tp_local_intermediate_size": 192,
            "intermediate_hadamard_blocks": [128, 64],
        },
        shared_su=True,
    )

    assert config["compatible_tp_sizes"] == [16]
    assert config["tp_local_quantization"] is True
    assert config["tp_local_intermediate_size"] == 192
    assert config["intermediate_hadamard_blocks"] == [128, 64]


def test_completion_marker_binds_tp_contract_and_file_sizes(tmp_path) -> None:
    layer = 7
    build_id = "tp16-h128h64-test"
    shard = tmp_path / f"exl3-layer-{layer:05d}.safetensors"
    errs = tmp_path / f"exl3-layer-{layer:05d}.errs.json"
    shard.write_bytes(b"complete-shard")
    errs.write_text("{}")
    marker = completion_marker(tmp_path, layer, build_id)
    marker.write_text(
        json.dumps(
            marker_contract(
                layer=layer,
                build_id=build_id,
                tp_size=16,
                shard_size=shard.stat().st_size,
                errs_size=errs.stat().st_size,
            ),
            sort_keys=True,
        )
    )

    assert layer_is_complete(tmp_path, layer, build_id, 16)
    assert not layer_is_complete(tmp_path, layer, build_id, 12)

    shard.write_bytes(b"truncated")
    assert not layer_is_complete(tmp_path, layer, build_id, 16)


def test_completion_marker_rejects_invalid_json(tmp_path) -> None:
    layer = 3
    build_id = "broken"
    (tmp_path / f"exl3-layer-{layer:05d}.safetensors").write_bytes(b"x")
    (tmp_path / f"exl3-layer-{layer:05d}.errs.json").write_text("{}")
    completion_marker(tmp_path, layer, build_id).write_text("{")

    assert not layer_is_complete(tmp_path, layer, build_id, 16)
