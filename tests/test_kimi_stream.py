from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from kquant.kimi_stream import (
    IndexedSafetensors,
    StreamState,
    assign_parameter,
    fit_checkpoint_parameter,
    load_stream_state,
    save_stream_state,
    write_trace_manifest,
    write_trace_tensor,
)
from kquant.trace_compare import load_trace


def _indexed_tensor(tmp_path: Path) -> tuple[IndexedSafetensors, torch.Tensor]:
    tensor = torch.arange(80, dtype=torch.float32).view(10, 8)
    shard = "model-00001.safetensors"
    save_file({"rows": tensor}, str(tmp_path / shard))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"rows": shard}})
    )
    return IndexedSafetensors(tmp_path), tensor


def test_indexed_safetensors_reads_arbitrary_rows(tmp_path: Path) -> None:
    store, tensor = _indexed_tensor(tmp_path)

    loaded = store.get_rows("rows", [7, 2, 3, 7, 9])

    assert torch.equal(loaded, tensor[[7, 2, 3, 7, 9]])
    with pytest.raises(IndexError):
        store.get_rows("rows", [10])


def test_indexed_safetensors_bulk_loads_and_eagerly_clones(
    tmp_path: Path,
) -> None:
    first = torch.arange(12, dtype=torch.float32).view(3, 4)
    second = torch.arange(5, dtype=torch.int32)
    shard = "model-00001.safetensors"
    save_file({"first": first, "second": second}, str(tmp_path / shard))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {"first": shard, "second": shard}}
        )
    )
    store = IndexedSafetensors(tmp_path)

    loaded = store.get_many(["second", "first", "second"], clone=True)

    assert list(loaded) == ["second", "first"]
    assert torch.equal(loaded["first"], first)
    assert torch.equal(loaded["second"], second)
    with pytest.raises(KeyError):
        store.get_many(["missing"])


def test_assign_parameter_replaces_meta_parameter() -> None:
    module = torch.nn.Sequential(
        torch.nn.Linear(4, 3, bias=False, device="meta")
    )
    value = torch.arange(12, dtype=torch.float32).view(3, 4)

    assign_parameter(module, "0.weight", value)

    assert module[0].weight.device.type == "cpu"
    assert torch.equal(module[0].weight, value)
    assert module[0].weight.requires_grad is False


def test_fit_checkpoint_parameter_only_narrows_zero_padded_a_log() -> None:
    value = torch.cat((torch.arange(3), torch.zeros(2))).float()

    fitted, note = fit_checkpoint_parameter(
        "model.layers.0.self_attn.A_log", value, (3,)
    )

    assert torch.equal(fitted, torch.arange(3).float())
    assert note is not None
    with pytest.raises(ValueError, match="nonzero"):
        fit_checkpoint_parameter(
            "model.layers.0.self_attn.A_log",
            torch.arange(5).float(),
            (3,),
        )
    with pytest.raises(ValueError, match="checkpoint shape"):
        fit_checkpoint_parameter("model.layers.0.other", value, (3,))


def test_stream_state_round_trip(tmp_path: Path) -> None:
    state = StreamState(
        next_layer=4,
        input_ids=torch.tensor([[1, 2, 3]]),
        hidden_states=torch.randn(1, 3, 8, dtype=torch.bfloat16),
        block_residual=torch.randn(3, 1, 8, dtype=torch.bfloat16),
    )
    path = tmp_path / "state.pt"

    save_stream_state(
        path,
        state,
        checkpoint="/checkpoint",
        prompt="prompt",
    )
    loaded, metadata = load_stream_state(path)

    assert loaded.next_layer == 4
    assert torch.equal(loaded.input_ids, state.input_ids)
    assert torch.equal(loaded.hidden_states, state.hidden_states)
    assert torch.equal(loaded.block_residual, state.block_residual)
    assert metadata == {"checkpoint": "/checkpoint", "prompt": "prompt"}


def test_official_trace_is_vllm_trace_schema_compatible(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "trace"
    value = torch.randn(3, 8, dtype=torch.bfloat16)
    write_trace_manifest(
        trace,
        layers=[0],
        max_tokens=3,
        checkpoint="/checkpoint",
    )
    write_trace_tensor(
        trace,
        layer=0,
        stage="layer_boundary",
        tensor=value,
    )

    loaded = load_trace(trace)

    assert loaded.world_size == 1
    tensor = next(iter(loaded.tensors.values()))[0]
    assert torch.equal(tensor, value)
