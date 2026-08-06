import torch
import pytest
from safetensors.torch import save_file

from kquant import constants as C
from kquant.io.hf_cache import CheckpointCache
from kquant.source_weights import OfficialMXFP4Store, weight_name


def test_official_store_streams_one_packed_matrix(tmp_path):
    packed_name = weight_name(1, 7, "w2", "weight_packed")
    scale_name = weight_name(1, 7, "w2", "weight_scale")
    out_features, in_features = C.EXPERT_SHAPES["w2"]
    packed = torch.zeros((out_features, in_features // 2), dtype=torch.uint8)
    packed[:, 0] = 0x10
    scales = torch.full(
        (out_features, in_features // C.MXFP4_BLOCK), 127, dtype=torch.uint8
    )
    shard = tmp_path / "model.safetensors"
    save_file({packed_name: packed, scale_name: scales}, shard)
    cache = CheckpointCache(
        snapshot_dir=tmp_path,
        index={"weight_map": {packed_name: shard.name, scale_name: shard.name}},
        shard_paths={shard.name: shard},
    )

    store = OfficialMXFP4Store(cache)
    raw = store.load_packed_matrix(1, 7, "w2")
    with store.open_layer(1, experts=(7,), matrices=("w2",)) as layer_store:
        layer_raw = layer_store.load_packed_matrix(1, 7, "w2")
        layer_matrix = layer_store.load_matrix(1, 7, "w2")
    scale_only = store.load_scale_plane(1, 7, "w2")
    streamed_scales = list(
        store.iter_layer_scale_planes(1, experts=(7,), matrices=("w2",))
    )
    matrix = store.load_matrix(1, 7, "w2")

    assert torch.equal(raw.packed, packed)
    assert torch.equal(raw.scale, scales)
    assert torch.equal(layer_raw.packed, raw.packed)
    assert torch.equal(layer_raw.scale, raw.scale)
    assert torch.equal(layer_matrix, matrix)
    assert torch.equal(scale_only, scales)
    assert len(streamed_scales) == 1
    assert streamed_scales[0][:2] == (7, "w2")
    assert torch.equal(streamed_scales[0][2], scales)
    assert matrix.shape == C.EXPERT_SHAPES["w2"]
    assert torch.equal(matrix[:, 0], torch.zeros(out_features))
    assert torch.equal(matrix[:, 1], torch.full((out_features,), 0.5))
    assert len(store.available_experts(1)) == 896


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_official_store_dequantizes_on_requested_cuda_device(tmp_path):
    packed_name = weight_name(1, 7, "w2", "weight_packed")
    scale_name = weight_name(1, 7, "w2", "weight_scale")
    out_features, in_features = C.EXPERT_SHAPES["w2"]
    generator = torch.Generator().manual_seed(17)
    packed = torch.randint(
        0,
        256,
        (out_features, in_features // 2),
        dtype=torch.uint8,
        generator=generator,
    )
    scales = torch.randint(
        120,
        135,
        (out_features, in_features // C.MXFP4_BLOCK),
        dtype=torch.uint8,
        generator=generator,
    )
    shard = tmp_path / "model.safetensors"
    save_file({packed_name: packed, scale_name: scales}, shard)
    cache = CheckpointCache(
        snapshot_dir=tmp_path,
        index={"weight_map": {packed_name: shard.name, scale_name: shard.name}},
        shard_paths={shard.name: shard},
    )

    store = OfficialMXFP4Store(cache)
    cpu = store.load_matrix(1, 7, "w2")
    cuda = store.load_matrix(1, 7, "w2", device="cuda")

    assert cuda.device.type == "cuda"
    assert cuda.dtype == torch.float32
    assert cuda.is_contiguous()
    assert torch.equal(cuda.cpu(), cpu)
