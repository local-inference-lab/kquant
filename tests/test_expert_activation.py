from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from kquant.expert_activation import EXPERT_ACTIVATIONS, expert_middle
from kquant.tp_simulator import situ


def test_expert_middle_supports_fruit_silu_and_kimi_situ_equations() -> None:
    gate = torch.tensor([[-3.0, 0.0, 2.5]], dtype=torch.float32)
    up = torch.tensor([[4.0, -2.0, 0.5]], dtype=torch.float32)

    torch.testing.assert_close(expert_middle(gate, up, "silu"), F.silu(gate) * up)
    torch.testing.assert_close(expert_middle(gate, up, "situ"), situ(gate, up))
    assert EXPERT_ACTIVATIONS == ("silu", "situ")


def test_expert_middle_rejects_unknown_activation() -> None:
    values = torch.ones((1, 2))

    with pytest.raises(ValueError, match="unsupported expert activation"):
        expert_middle(values, values, "gelu")  # type: ignore[arg-type]
