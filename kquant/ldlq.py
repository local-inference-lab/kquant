"""Shared dense-Hessian state for the offline EXL3 BlockLDLQ encoder."""

from __future__ import annotations

import torch


SIGMA_REG = 0.025


def make_shared_h(
    in_features: int,
    device: torch.device,
    hessian: torch.Tensor | None = None,
) -> dict:
    """Build the mutable EXL3 Hessian wrapper while retaining an exact copy."""

    if hessian is None:
        hessian = torch.eye(in_features, dtype=torch.float32)
    if tuple(hessian.shape) != (in_features, in_features):
        raise ValueError(
            f"Hessian shape {tuple(hessian.shape)} does not match {in_features}"
        )
    canonical = hessian.to(
        device=device, dtype=torch.float32, copy=True
    ).contiguous()
    return {
        "H": canonical.clone(),
        "error_H": canonical,
        "first_key": f"h{in_features}",
        "count": 1,
        "finalized": False,
        "num_total": 1,
        "inf_nan": torch.zeros(2, dtype=torch.long, device=device),
        "device": device,
    }
