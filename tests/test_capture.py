from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

import kquant.capture as capture_module
from kquant.capture import build_hessians, index_layer_samples, load_layer_samples


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value))


def _make_capture(
    root: Path,
    *,
    mismatch_mid: bool = False,
    mismatch_mid_expert: bool = False,
    num_layers: int = 1,
    encoded_request_steps: bool = False,
) -> Path:
    root.mkdir()
    _write_json(
        root / "manifest.json",
        {
            "schema_version": 2,
            "run_id": "synthetic",
            "tp_world_size": 2,
            "complete": True,
            "geometry": {
                "num_layers": num_layers,
                "num_experts": 2,
                "input_size": 2,
                "intermediate_size": 4,
                "top_k": 2,
            },
        },
    )
    input_rows = torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16)
    input_weight = torch.tensor([0.5, 1.25], dtype=torch.float32)
    input_obs = torch.tensor([10, 20], dtype=torch.int64)
    input_experts = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    input_gates = torch.tensor([[0.5, 0.5], [1.0, 0.5]], dtype=torch.float32)
    input_split = torch.tensor([0, 1], dtype=torch.int8)
    routed_latent = torch.tensor([[5, 6], [7, 8]], dtype=torch.bfloat16)
    mid_rows = torch.tensor([[1, 2, 3, 4], [2, 0, 1, 3]], dtype=torch.bfloat16)
    mid_weight = torch.tensor([0.25, 2.0], dtype=torch.float32)
    mid_obs = torch.tensor([11, 22], dtype=torch.int64)
    if encoded_request_steps:
        input_obs = input_obs | torch.tensor([0, 1 << 32], dtype=torch.int64)
        mid_obs = mid_obs | torch.tensor([0, 1 << 32], dtype=torch.int64)
    sample_layers = torch.arange(num_layers, dtype=torch.int16).repeat_interleave(2)
    input_rows = input_rows.repeat((num_layers, 1))
    input_weight = input_weight.repeat(num_layers)
    input_obs = torch.cat([input_obs + 100 * layer for layer in range(num_layers)])
    input_experts = input_experts.repeat((num_layers, 1))
    input_gates = input_gates.repeat((num_layers, 1))
    input_split = input_split.repeat(num_layers)
    routed_latent = routed_latent.repeat((num_layers, 1))
    mid_rows = mid_rows.repeat((num_layers, 1))
    mid_weight = mid_weight.repeat(num_layers)
    mid_obs = torch.cat([mid_obs + 100 * layer for layer in range(num_layers)])
    mid_split = torch.tensor([0, 1], dtype=torch.int8).repeat(num_layers)

    for rank in range(2):
        rank_dir = root / f"rank-{rank:05d}"
        samples = rank_dir / "samples"
        samples.mkdir(parents=True)
        _write_json(
            rank_dir / "manifest.json",
            {
                "schema_version": 2,
                "rank": rank,
                "tp_world_size": 2,
                "run_id": "synthetic",
                "complete": True,
                "input_expert_range": [rank, rank + 1],
                "intermediate_channel_range": [2 * rank, 2 * rank + 2],
            },
        )
        save_file({"dummy": torch.tensor([rank])}, rank_dir / "stats.safetensors")
        rank_obs = mid_obs.clone()
        if mismatch_mid and rank == 1:
            rank_obs[1] += 1
        mid_expert = torch.tensor([0, 1], dtype=torch.int32).repeat(num_layers)
        if mismatch_mid_expert and rank == 1:
            mid_expert[0] = 1
        part = {
            "mid.values": mid_rows[:, 2 * rank : 2 * rank + 2].contiguous(),
            "mid.weight": mid_weight,
            "mid.observation": rank_obs,
            "mid.expert": mid_expert,
            "mid.split": mid_split,
            "mid.layer": sample_layers,
        }
        if rank == 0:
            part.update(
                {
                    "input.values": input_rows,
                    "input.weight": input_weight,
                    "input.observation": input_obs,
                    "input.experts": input_experts,
                    "input.gates": input_gates,
                    "input.split": input_split,
                    "input.routed_latent": routed_latent,
                    "input.layer": sample_layers.clone(),
                }
            )
        save_file(part, samples / "part-00000001.safetensors")
    return root


def test_build_hessians_joins_tp_channel_shards(tmp_path: Path) -> None:
    capture = _make_capture(tmp_path / "tiny.kqcapture")
    output = build_hessians(
        capture,
        tmp_path / "tiny",
        layers=[0],
        min_rows=2,
        sample_split="all",
    )
    tensors = load_file(output / "layer-00001.safetensors")

    x13 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    w13 = torch.tensor([0.5, 1.25], dtype=torch.float32)
    expected13 = (x13.T @ (x13 * w13[:, None])) / w13.sum()
    x2 = torch.tensor([[1, 2, 3, 4], [2, 0, 1, 3]], dtype=torch.float32)
    w2 = torch.tensor([0.25, 2.0], dtype=torch.float32)
    expected2 = (x2.T @ (x2 * w2[:, None])) / w2.sum()
    torch.testing.assert_close(tensors["w13"], expected13)
    torch.testing.assert_close(tensors["w2"], expected2)

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["layers"]["1"]["w13_rows"] == 2
    assert manifest["layers"]["1"]["w2_rows"] == 2
    assert manifest["sample_split"] == "all"


def test_build_hessians_filters_capture_request_epochs(tmp_path: Path) -> None:
    capture = _make_capture(
        tmp_path / "request-filter.kqcapture", encoded_request_steps=True
    )
    output = build_hessians(
        capture,
        tmp_path / "request-filter",
        layers=[0],
        min_rows=1,
        sample_split="all",
        request_step_min=1,
        request_step_max=1,
    )
    tensors = load_file(output / "layer-00001.safetensors")

    torch.testing.assert_close(
        tensors["w13"],
        torch.tensor([[9, 12], [12, 16]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        tensors["w2"],
        torch.tensor(
            [[4, 0, 2, 6], [0, 0, 0, 0], [2, 0, 1, 3], [6, 0, 3, 9]],
            dtype=torch.float32,
        ),
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["layers"]["1"]["w13_rows"] == 1
    assert manifest["layers"]["1"]["w2_rows"] == 1
    assert manifest["request_step_range"]["minimum_inclusive"] == 1
    assert manifest["request_step_range"]["maximum_inclusive"] == 1


def test_build_hessians_filters_an_exact_request_epoch_set(tmp_path: Path) -> None:
    capture = _make_capture(
        tmp_path / "request-set-filter.kqcapture", encoded_request_steps=True
    )
    output = build_hessians(
        capture,
        tmp_path / "request-set-filter",
        layers=[0],
        min_rows=1,
        sample_split="all",
        request_steps=[0],
    )
    tensors = load_file(output / "layer-00001.safetensors")

    torch.testing.assert_close(
        tensors["w13"],
        torch.tensor([[1, 2], [2, 4]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        tensors["w2"],
        torch.tensor(
            [[1, 2, 3, 4], [2, 4, 6, 8], [3, 6, 9, 12], [4, 8, 12, 16]],
            dtype=torch.float32,
        ),
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["request_step_filter"] == [0]


def test_build_hessians_rejects_unaligned_tp_samples(tmp_path: Path) -> None:
    capture = _make_capture(tmp_path / "bad.kqcapture", mismatch_mid=True)
    with pytest.raises(ValueError, match="observation IDs do not align"):
        build_hessians(capture, tmp_path / "bad", layers=[0])


def test_build_hessians_rejects_mismatched_tp_experts(tmp_path: Path) -> None:
    capture = _make_capture(
        tmp_path / "bad-expert.kqcapture", mismatch_mid_expert=True
    )
    with pytest.raises(ValueError, match="mid expert IDs differ"):
        build_hessians(capture, tmp_path / "bad-expert", layers=[0])


def test_build_hessians_reads_each_sample_part_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = _make_capture(tmp_path / "multi.kqcapture", num_layers=2)
    real_load = capture_module.load_torch_file
    sample_loads: list[str] = []

    def counted_load(path: str, *args, **kwargs):
        if "/samples/" in path:
            sample_loads.append(path)
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(capture_module, "load_torch_file", counted_load)
    output = build_hessians(
        capture,
        tmp_path / "multi",
        layers=[0, 1],
        min_rows=2,
        sample_split="all",
    )

    assert len(sample_loads) == 2
    assert (output / "layer-00001.safetensors").is_file()
    assert (output / "layer-00002.safetensors").is_file()


def test_build_hessians_keeps_validation_rows_held_out(tmp_path: Path) -> None:
    capture = _make_capture(tmp_path / "split.kqcapture")
    output = build_hessians(capture, tmp_path / "split", layers=[0])
    tensors = load_file(output / "layer-00001.safetensors")

    x13 = torch.tensor([[1, 2]], dtype=torch.float32)
    torch.testing.assert_close(tensors["w13"], x13.T @ x13)
    x2 = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
    torch.testing.assert_close(tensors["w2"], x2.T @ x2)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["sample_split"] == "train"
    assert manifest["layers"]["1"]["w13_total_rows"] == 2
    assert manifest["layers"]["1"]["w13_validation_rows"] == 1


def test_load_layer_samples_preserves_route_and_expert_identity(tmp_path: Path) -> None:
    capture = _make_capture(tmp_path / "samples.kqcapture")

    samples = load_layer_samples(capture, 0)

    assert samples.input_experts.tolist() == [[0, 1], [1, 0]]
    torch.testing.assert_close(
        samples.input_gates,
        torch.tensor([[0.5, 0.5], [1.0, 0.5]], dtype=torch.float32),
    )
    assert samples.input_split.tolist() == [0, 1]
    assert samples.routed_latent.tolist() == [[5, 6], [7, 8]]
    assert samples.mid_experts.tolist() == [0, 1]
    assert samples.mid_split.tolist() == [0, 1]


def test_reusable_layer_index_reads_capture_once_and_skips_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = _make_capture(tmp_path / "indexed.kqcapture", num_layers=2)
    real_load = capture_module.load_torch_file
    sample_loads: list[str] = []
    stats_loads: list[str] = []

    def counted_load(path: str, *args, **kwargs):
        if "/samples/" in path:
            sample_loads.append(path)
        elif path.endswith("stats.safetensors"):
            stats_loads.append(path)
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(capture_module, "load_torch_file", counted_load)
    index = index_layer_samples(capture, [0, 1])

    assert len(sample_loads) == 2
    assert stats_loads == []
    first = index.pop(0)
    second = index.pop(1)
    assert first.input_observations.tolist() == [10, 20]
    assert second.input_observations.tolist() == [110, 120]
    assert index.remaining == set()
    with pytest.raises(ValueError, match="absent or already consumed"):
        index.pop(0)


def test_grouped_layer_samples_do_not_retain_complete_part_storage(
    tmp_path: Path,
) -> None:
    rows = 100
    layers = torch.arange(rows, dtype=torch.int16) % 2
    route_gates = torch.full((rows, 2), 0.5, dtype=torch.float32)
    values = {
        "input.values": torch.arange(rows * 2, dtype=torch.bfloat16).reshape(rows, 2),
        "input.weight": route_gates.square().sum(dim=1),
        "input.observation": torch.arange(rows, dtype=torch.int64),
        "input.experts": torch.zeros((rows, 2), dtype=torch.int32),
        "input.gates": route_gates,
        "input.split": torch.zeros(rows, dtype=torch.int8),
        "input.routed_latent": torch.ones((rows, 2), dtype=torch.bfloat16),
        "input.layer": layers,
    }
    geometry = capture_module.CaptureGeometry(
        num_layers=2,
        num_experts=2,
        input_size=2,
        intermediate_size=4,
        top_k=2,
    )

    grouped = capture_module._group_sample_part(
        values,
        "input",
        {0},
        path=tmp_path / "part.safetensors",
        geometry=geometry,
    )

    selected = grouped[0]
    assert selected.values.shape == (rows // 2, 2)
    assert selected.values.untyped_storage().nbytes() == selected.values.numel() * 2
    assert (
        selected.routed_latent.untyped_storage().nbytes()
        == selected.routed_latent.numel() * 2
    )
