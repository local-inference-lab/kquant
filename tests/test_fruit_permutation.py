from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from kquant.fruit_source import FruitCheckpointStore, FruitModelSpec
from kquant.qsrt import coupled_intermediate_permutation

_ORDINARY_LAYER = 3
_MTP_LAYER = 13
_EXPERTS = 3
_HIDDEN = 5
_INTERMEDIATE = 12
_LAYERS = (_ORDINARY_LAYER, _MTP_LAYER)
_MATRICES = ("w1", "w3", "w2")
_SOURCE_NAMES = {
    "w1": ("w_gate", "gate_proj"),
    "w3": ("w_up", "up_proj"),
    "w2": ("w_down", "down_proj"),
}


def _prefix(layer: int) -> str:
    return "mtp_block.mlp" if layer == _MTP_LAYER else f"layers.{layer}.mlp"


def _synthetic_weights() -> dict[tuple[int, int, str], torch.Tensor]:
    generator = torch.Generator().manual_seed(20260807)
    weights: dict[tuple[int, int, str], torch.Tensor] = {}
    for layer in _LAYERS:
        for expert in range(_EXPERTS):
            weights[layer, expert, "w1"] = torch.randn(
                _INTERMEDIATE, _HIDDEN, generator=generator, dtype=torch.float32
            )
            weights[layer, expert, "w3"] = torch.randn(
                _INTERMEDIATE, _HIDDEN, generator=generator, dtype=torch.float32
            )
            weights[layer, expert, "w2"] = torch.randn(
                _HIDDEN, _INTERMEDIATE, generator=generator, dtype=torch.float32
            )
    return weights


def _checkpoint_state(
    layout: str, weights: dict[tuple[int, int, str], torch.Tensor]
) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for layer in _LAYERS:
        prefix = _prefix(layer)
        for matrix in _MATRICES:
            stacked_name, projection_name = _SOURCE_NAMES[matrix]
            if layout == "stacked":
                state[f"{prefix}.{stacked_name}"] = torch.stack(
                    [weights[layer, expert, matrix] for expert in range(_EXPERTS)]
                )
            elif layout == "per_expert":
                for expert in range(_EXPERTS):
                    state[f"{prefix}.experts.{expert}.{projection_name}.weight"] = (
                        weights[layer, expert, matrix]
                    )
            else:  # pragma: no cover - test helper misuse
                raise AssertionError(f"unknown test layout: {layout}")
    return state


def _synthetic_store(
    tmp_path: Path, layout: str
) -> tuple[FruitCheckpointStore, dict[tuple[int, int, str], torch.Tensor]]:
    weights = _synthetic_weights()
    path = tmp_path / f"fruit-{layout}.pt"
    torch.save(_checkpoint_state(layout, weights), path)
    with path.open("rb") as checkpoint:
        digest = hashlib.file_digest(checkpoint, "sha256").hexdigest()
    spec = FruitModelSpec(
        model_id=f"synthetic-fruit-{layout}",
        checkpoint_sha256=digest,
        layers=(_ORDINARY_LAYER,),
        mtp_layer=_MTP_LAYER,
        num_experts=_EXPERTS,
        hidden_size=_HIDDEN,
        intermediate_size=_INTERMEDIATE,
        trained_rope_theta=500_000.0,
        serve_conv_v=None,
    )
    return FruitCheckpointStore(path, spec=spec), weights


def _expert_output(
    inputs: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor, w2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    middle = F.silu(F.linear(inputs, w1)) * F.linear(inputs, w3)
    return middle, F.linear(middle, w2)


@pytest.mark.parametrize("layout", ("stacked", "per_expert"))
@pytest.mark.parametrize(
    "layer",
    (
        pytest.param(_ORDINARY_LAYER, id="ordinary-layer"),
        pytest.param(_MTP_LAYER, id="logical-mtp-layer"),
    ),
)
def test_fruit_expert_local_coupled_permutations_preserve_silu_output(
    tmp_path: Path, layout: str, layer: int
) -> None:
    store, source_weights = _synthetic_store(tmp_path, layout)
    input_generator = torch.Generator().manual_seed(1100 + layer)
    inputs = torch.randn(7, _HIDDEN, generator=input_generator, dtype=torch.float32)
    permutations: list[torch.Tensor] = []

    for expert in range(_EXPERTS):
        loaded = {
            matrix: store.load_matrix(layer, expert, matrix, device="cpu")
            for matrix in _MATRICES
        }
        for matrix, tensor in loaded.items():
            assert tensor.dtype == torch.float32
            assert tensor.is_contiguous()
            assert torch.equal(tensor, source_weights[layer, expert, matrix])

        # Each expert independently owns its new-channel -> old-channel mapping.
        permutation = torch.roll(torch.arange(_INTERMEDIATE), shifts=expert + 1)
        permutations.append(permutation)
        reference_middle, reference = _expert_output(
            inputs, loaded["w1"], loaded["w3"], loaded["w2"]
        )
        permuted_w1, permuted_w3, permuted_w2 = coupled_intermediate_permutation(
            loaded["w1"], loaded["w3"], loaded["w2"], permutation
        )
        actual_middle, actual = _expert_output(
            inputs, permuted_w1, permuted_w3, permuted_w2
        )

        # BLAS may accumulate a row in a different output tile after reordering.
        torch.testing.assert_close(
            actual_middle,
            reference_middle[:, permutation],
            rtol=1.3e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(actual, reference, rtol=1.3e-6, atol=1e-5)

    assert all(
        not torch.equal(left, right)
        for index, left in enumerate(permutations)
        for right in permutations[index + 1 :]
    )


@pytest.mark.parametrize("layout", ("stacked", "per_expert"))
@pytest.mark.parametrize("layer", (_ORDINARY_LAYER, _MTP_LAYER))
def test_fruit_mismatched_cross_matrix_permutations_change_silu_output(
    tmp_path: Path, layout: str, layer: int
) -> None:
    store, _ = _synthetic_store(tmp_path, layout)
    inputs = torch.randn(
        9,
        _HIDDEN,
        generator=torch.Generator().manual_seed(2100 + layer),
        dtype=torch.float32,
    )

    for expert in range(_EXPERTS):
        w1 = store.load_matrix(layer, expert, "w1", device="cpu")
        w3 = store.load_matrix(layer, expert, "w3", device="cpu")
        w2 = store.load_matrix(layer, expert, "w2", device="cpu")
        gate_permutation = torch.roll(torch.arange(_INTERMEDIATE), shifts=expert + 1)
        up_permutation = torch.roll(torch.arange(_INTERMEDIATE), shifts=expert + 2)
        permuted_w1, _, permuted_w2 = coupled_intermediate_permutation(
            w1, w3, w2, gate_permutation
        )
        _, mismatched_w3, _ = coupled_intermediate_permutation(
            w1, w3, w2, up_permutation
        )
        _, reference = _expert_output(inputs, w1, w3, w2)
        _, mismatched = _expert_output(inputs, permuted_w1, mismatched_w3, permuted_w2)

        assert not torch.allclose(mismatched, reference, rtol=1.3e-6, atol=1e-5)


@pytest.mark.parametrize("layout", ("stacked", "per_expert"))
def test_fruit_permutation_has_no_router_or_expert_axis(
    tmp_path: Path, layout: str
) -> None:
    store, source_weights = _synthetic_store(tmp_path, layout)
    router_logits = torch.tensor(
        [[3.0, -1.0, 0.5], [-2.0, 4.0, 1.0]], dtype=torch.float32
    )
    routed_experts = router_logits.argmax(dim=-1)
    router_logits_before = router_logits.clone()
    routed_experts_before = routed_experts.clone()
    permutation = torch.arange(_INTERMEDIATE - 1, -1, -1)

    for expert in range(_EXPERTS):
        w1 = store.load_matrix(_ORDINARY_LAYER, expert, "w1", device="cpu")
        w3 = store.load_matrix(_ORDINARY_LAYER, expert, "w3", device="cpu")
        w2 = store.load_matrix(_ORDINARY_LAYER, expert, "w2", device="cpu")
        permuted_w1, permuted_w3, permuted_w2 = coupled_intermediate_permutation(
            w1, w3, w2, permutation
        )

        assert torch.equal(
            permuted_w1, source_weights[_ORDINARY_LAYER, expert, "w1"][permutation]
        )
        assert torch.equal(
            permuted_w3, source_weights[_ORDINARY_LAYER, expert, "w3"][permutation]
        )
        assert torch.equal(
            permuted_w2,
            source_weights[_ORDINARY_LAYER, expert, "w2"][:, permutation],
        )

    assert torch.equal(router_logits, router_logits_before)
    assert torch.equal(routed_experts, routed_experts_before)

    stacked_w1 = torch.stack(
        [source_weights[_ORDINARY_LAYER, expert, "w1"] for expert in range(_EXPERTS)]
    )
    stacked_w3 = torch.stack(
        [source_weights[_ORDINARY_LAYER, expert, "w3"] for expert in range(_EXPERTS)]
    )
    stacked_w2 = torch.stack(
        [source_weights[_ORDINARY_LAYER, expert, "w2"] for expert in range(_EXPERTS)]
    )
    with pytest.raises(ValueError, match="two-dimensional"):
        coupled_intermediate_permutation(
            stacked_w1, stacked_w3, stacked_w2, permutation
        )
