"""Shared expert-middle activation equations for offline QSRT tooling."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

from kquant.tp_simulator import situ

ExpertActivation = Literal["silu", "situ"]
EXPERT_ACTIVATIONS: tuple[ExpertActivation, ...] = ("silu", "situ")


def expert_middle(
    gate: torch.Tensor,
    up: torch.Tensor,
    activation: ExpertActivation,
) -> torch.Tensor:
    """Apply the selected coordinatewise gate/up expert-middle equation."""

    if activation == "silu":
        return F.silu(gate) * up
    if activation == "situ":
        return situ(gate, up)
    raise ValueError(f"unsupported expert activation: {activation!r}")
