from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from kquant.fruit_source import (
    FRUIT_ANNEALED_SPEC,
    FruitCheckpointStore,
    FruitModelSpec,
    preflight_fruit_checkpoint,
)

_BASE_SPEC = FruitModelSpec(
    model_id="synthetic-fruit",
    checkpoint_sha256="0" * 64,
    layers=(3, 4),
    mtp_layer=13,
    num_experts=3,
    hidden_size=3,
    intermediate_size=4,
    trained_rope_theta=500_000.0,
)
_STACKED = {"w1": "w_gate", "w3": "w_up", "w2": "w_down"}
_PROJECTION = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}


def _prefix(spec: FruitModelSpec, layer: int) -> str:
    return "mtp_block.mlp" if layer == spec.mtp_layer else f"layers.{layer}.mlp"


def _matrix(
    spec: FruitModelSpec,
    matrix: str,
    *,
    offset: int,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    shape = (
        (spec.intermediate_size, spec.hidden_size)
        if matrix != "w2"
        else (spec.hidden_size, spec.intermediate_size)
    )
    values = torch.arange(shape[0] * shape[1], dtype=torch.float32)
    return (values.reshape(shape) + offset).to(dtype)


def _state(
    spec: FruitModelSpec,
    representation: str,
    *,
    native_markers: bool = False,
) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {
        "embed_tokens.weight": torch.ones(2, 2, dtype=torch.bfloat16)
    }
    for layer_index, layer in enumerate((*spec.layers, spec.mtp_layer)):
        prefix = _prefix(spec, layer)
        if representation == "stacked":
            for matrix_index, matrix in enumerate(("w1", "w3", "w2")):
                state[f"{prefix}.{_STACKED[matrix]}"] = torch.stack(
                    [
                        _matrix(
                            spec,
                            matrix,
                            offset=1000 * layer_index + 100 * matrix_index + expert,
                        )
                        for expert in range(spec.num_experts)
                    ]
                )
        elif representation == "per_expert":
            for expert in range(spec.num_experts):
                for matrix_index, matrix in enumerate(("w1", "w3", "w2")):
                    state[f"{prefix}.experts.{expert}.{_PROJECTION[matrix]}.weight"] = (
                        _matrix(
                            spec,
                            matrix,
                            offset=1000 * layer_index + 100 * matrix_index + expert,
                        )
                    )
        else:
            raise AssertionError(representation)
    if native_markers:
        state["serve_conv_v"] = torch.tensor([2], dtype=torch.int32)
        state["rope_theta_trained"] = torch.tensor(
            [spec.trained_rope_theta], dtype=torch.float64
        )
    return state


def _save(
    path: Path,
    payload: object,
    spec: FruitModelSpec = _BASE_SPEC,
) -> FruitModelSpec:
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return replace(spec, checkpoint_sha256=digest)


def test_real_annealed_spec_freezes_source_identity_and_geometry() -> None:
    assert FRUIT_ANNEALED_SPEC == FruitModelSpec(
        model_id="malaiwah/GLM-5.2-SIQ-Fruit",
        checkpoint_sha256=(
            "98ac7cb4f7799194424782b505d622069fecf4dbca5f5acb2658f2a66c3631f6"
        ),
        layers=tuple(range(3, 13)),
        mtp_layer=13,
        num_experts=256,
        hidden_size=1024,
        intermediate_size=512,
        trained_rope_theta=500_000.0,
        serve_conv_v=None,
    )


def test_stacked_store_maps_w1_and_preserves_output_contract(tmp_path: Path) -> None:
    path = tmp_path / "stacked.pt"
    state = _state(_BASE_SPEC, "stacked")
    spec = _save(path, state)

    store = FruitCheckpointStore(path, spec=spec)
    result = store.load_matrix(3, 2, "w1", device=torch.device("cpu"))

    assert store.representation == "stacked"
    assert result.shape == (_BASE_SPEC.intermediate_size, _BASE_SPEC.hidden_size)
    assert result.dtype == torch.float32
    assert result.device == torch.device("cpu")
    assert result.is_contiguous()
    torch.testing.assert_close(
        result, state["layers.3.mlp.w_gate"][2].float(), rtol=0, atol=0
    )


def test_stacked_mtp_maps_w2_to_mtp_down_without_transpose(tmp_path: Path) -> None:
    path = tmp_path / "mtp.pt"
    state = _state(_BASE_SPEC, "stacked")
    spec = _save(path, state)

    result = FruitCheckpointStore(path, spec=spec).load_matrix(13, 1, "w2")

    assert result.shape == (_BASE_SPEC.hidden_size, _BASE_SPEC.intermediate_size)
    torch.testing.assert_close(
        result, state["mtp_block.mlp.w_down"][1].float(), rtol=0, atol=0
    )


def test_per_expert_wrapper_maps_w3_to_up_proj(tmp_path: Path) -> None:
    path = tmp_path / "per-expert.pt"
    state = _state(_BASE_SPEC, "per_expert")
    spec = _save(path, {"model": state})

    store = FruitCheckpointStore(path, spec=spec)
    result = store.load_matrix(4, 0, "w3")

    assert store.representation == "per_expert"
    assert store.evidence["source_container"] == "model"
    torch.testing.assert_close(
        result,
        state["layers.4.mlp.experts.0.up_proj.weight"].float(),
        rtol=0,
        atol=0,
    )


def test_preflight_is_json_serializable_and_pins_legacy_theta(tmp_path: Path) -> None:
    path = tmp_path / "preflight.pt"
    spec = _save(path, _state(_BASE_SPEC, "stacked"))

    evidence = preflight_fruit_checkpoint(path, spec, spec.checkpoint_sha256)

    json.dumps(evidence, allow_nan=False, sort_keys=True)
    assert evidence["status"] == "pass"
    assert evidence["checkpoint_sha256"] == spec.checkpoint_sha256
    assert evidence["expert_inventory"]["matrices"] == ["w1", "w3", "w2"]
    assert evidence["conventions"] == {
        "kind": "legacy",
        "trained_rope_theta": 500_000.0,
        "trained_rope_theta_provenance": "model_spec",
        "serve_conv_v": None,
    }


def test_native_marker_pair_is_validated_and_reported(tmp_path: Path) -> None:
    native = replace(_BASE_SPEC, serve_conv_v=2)
    path = tmp_path / "native.pt"
    spec = _save(path, _state(native, "stacked", native_markers=True), native)

    evidence = preflight_fruit_checkpoint(path, spec, spec.checkpoint_sha256)

    assert evidence["conventions"] == {
        "kind": "native",
        "trained_rope_theta": 500_000.0,
        "trained_rope_theta_provenance": (
            "checkpoint_marker_validated_against_model_spec"
        ),
        "serve_conv_v": 2,
    }


def test_hash_and_source_identity_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "bad-hash.pt"
    spec = _save(path, _state(_BASE_SPEC, "stacked"))
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        FruitCheckpointStore(path, spec=spec)

    intact = tmp_path / "identity.pt"
    intact_spec = _save(intact, _state(_BASE_SPEC, "stacked"))
    with pytest.raises(ValueError, match="does not identify"):
        FruitCheckpointStore(intact, spec=intact_spec, expected_sha256="f" * 64)


def test_store_rejects_checkpoint_changed_after_authentication(tmp_path: Path) -> None:
    path = tmp_path / "mutable.pt"
    spec = _save(path, _state(_BASE_SPEC, "stacked"))
    store = FruitCheckpointStore(path, spec=spec)

    with path.open("r+b") as handle:
        handle.seek(-1, 2)
        original = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([original[0] ^ 1]))
    changed = path.stat()
    os.utime(
        path,
        ns=(changed.st_atime_ns, changed.st_mtime_ns + 1_000_000_000),
    )

    with pytest.raises(ValueError, match="changed after authentication"):
        store.load_matrix(3, 0, "w1")


def test_shape_dtype_and_exact_inventory_fail_closed(tmp_path: Path) -> None:
    shape_path = tmp_path / "shape.pt"
    shape_state = _state(_BASE_SPEC, "stacked")
    shape_state["layers.3.mlp.w_gate"] = shape_state["layers.3.mlp.w_gate"][:, :-1]
    shape_spec = _save(shape_path, shape_state)
    with pytest.raises(ValueError, match="shape"):
        FruitCheckpointStore(shape_path, spec=shape_spec)

    dtype_path = tmp_path / "dtype.pt"
    dtype_state = _state(_BASE_SPEC, "per_expert")
    key = "mtp_block.mlp.experts.2.down_proj.weight"
    dtype_state[key] = dtype_state[key].to(torch.int32)
    dtype_spec = _save(dtype_path, dtype_state)
    with pytest.raises(ValueError, match="non-floating dtype"):
        FruitCheckpointStore(dtype_path, spec=dtype_spec)

    missing_path = tmp_path / "missing.pt"
    missing_state = _state(_BASE_SPEC, "per_expert")
    del missing_state["layers.4.mlp.experts.1.up_proj.weight"]
    missing_spec = _save(missing_path, missing_state)
    with pytest.raises(ValueError, match="inventory is not exact"):
        FruitCheckpointStore(missing_path, spec=missing_spec)


def test_mixed_layout_and_missing_marker_fail_closed(tmp_path: Path) -> None:
    mixed_path = tmp_path / "mixed.pt"
    mixed_state = _state(_BASE_SPEC, "stacked")
    mixed_state["layers.3.mlp.experts.0.gate_proj.weight"] = _matrix(
        _BASE_SPEC, "w1", offset=0
    )
    mixed_spec = _save(mixed_path, mixed_state)
    with pytest.raises(ValueError, match="mixes stacked and per-expert"):
        FruitCheckpointStore(mixed_path, spec=mixed_spec)

    native = replace(_BASE_SPEC, serve_conv_v=2)
    marker_path = tmp_path / "missing-marker.pt"
    marker_state = _state(native, "stacked")
    marker_state["serve_conv_v"] = torch.tensor([2], dtype=torch.int32)
    marker_spec = _save(marker_path, marker_state, native)
    with pytest.raises(ValueError, match="present or absent together"):
        FruitCheckpointStore(marker_path, spec=marker_spec)


def test_unsupported_layer_expert_and_native_matrix_names_are_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "arguments.pt"
    spec = _save(path, _state(_BASE_SPEC, "stacked"))
    store = FruitCheckpointStore(path, spec=spec)

    with pytest.raises(ValueError, match="layer must be one of"):
        store.load_matrix(12, 0, "w1")
    with pytest.raises(ValueError, match="expert must be"):
        store.load_matrix(3, spec.num_experts, "w1")
    with pytest.raises(ValueError, match="matrix must be"):
        store.load_matrix(3, 0, "w_gate")


def test_model_wrapper_is_unambiguous(tmp_path: Path) -> None:
    path = tmp_path / "wrapper.pt"
    payload = {"model": _state(_BASE_SPEC, "stacked"), "step": torch.tensor(1)}
    spec = _save(path, payload)

    with pytest.raises(ValueError, match="wrapper must be exactly"):
        FruitCheckpointStore(path, spec=spec)
