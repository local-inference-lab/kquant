from __future__ import annotations

import pytest
import torch
from safetensors import safe_open

from kquant.io.safetensors_stream import AtomicSafetensorsWriter, TensorSpec


def test_streamed_safetensors_round_trip_without_combining_tensors(tmp_path) -> None:
    path = tmp_path / "streamed.safetensors"
    specs = (
        TensorSpec("a", torch.int16, (3, 4)),
        TensorSpec("b", torch.float16, (5,)),
        TensorSpec("c", torch.uint8, (2, 3)),
    )
    expected = {
        "a": torch.arange(12, dtype=torch.int16).reshape(3, 4),
        "b": torch.linspace(-1, 1, 5, dtype=torch.float16),
        "c": torch.arange(6, dtype=torch.uint8).reshape(2, 3),
    }

    with AtomicSafetensorsWriter(path, specs, metadata={"source": "unit"}) as writer:
        writer.write("b", expected["b"])
        writer.write("a", expected["a"])
        writer.write("c", expected["c"])

    with safe_open(path, framework="pt", device="cpu") as handle:
        assert set(handle.keys()) == set(expected)
        assert handle.metadata() == {"source": "unit"}
        for name, value in expected.items():
            assert torch.equal(handle.get_tensor(name), value)


def test_streamed_safetensors_rejects_missing_duplicate_and_wrong_shape(tmp_path) -> None:
    missing_path = tmp_path / "missing.safetensors"
    with pytest.raises(ValueError, match="missing 1 tensors"):
        with AtomicSafetensorsWriter(
            missing_path, (TensorSpec("a", torch.float16, (2,)),)
        ):
            pass
    assert not missing_path.exists()
    assert not (tmp_path / ".missing.safetensors.partial").exists()

    duplicate_path = tmp_path / "duplicate.safetensors"
    with pytest.raises(ValueError, match="written twice"):
        with AtomicSafetensorsWriter(
            duplicate_path, (TensorSpec("a", torch.float16, (2,)),)
        ) as writer:
            writer.write("a", torch.zeros(2, dtype=torch.float16))
            writer.write("a", torch.zeros(2, dtype=torch.float16))
    assert not duplicate_path.exists()

    shape_path = tmp_path / "shape.safetensors"
    with pytest.raises(ValueError, match="expected"):
        with AtomicSafetensorsWriter(
            shape_path, (TensorSpec("a", torch.int16, (2,)),)
        ) as writer:
            writer.write("a", torch.zeros(3, dtype=torch.int16))
    assert not shape_path.exists()


def test_streamed_safetensors_only_discards_an_unlocked_partial(tmp_path) -> None:
    path = tmp_path / "recover.safetensors"
    writer = AtomicSafetensorsWriter(
        path, (TensorSpec("a", torch.float16, (2,)),)
    )
    writer.__enter__()
    try:
        with pytest.raises(RuntimeError, match="live writer"):
            AtomicSafetensorsWriter.discard_stale_partial(path)
    finally:
        writer.abort()

    partial = AtomicSafetensorsWriter.partial_path(path)
    partial.write_bytes(b"abandoned")
    assert AtomicSafetensorsWriter.discard_stale_partial(path)
    assert not partial.exists()
    assert not AtomicSafetensorsWriter.discard_stale_partial(path)
