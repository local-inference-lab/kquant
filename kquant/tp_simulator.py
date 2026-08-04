"""Sequential simulation of tensor-parallel linear algebra.

These helpers model the logical sharding and padding contract without
requiring one CUDA process per logical TP rank.
"""

from __future__ import annotations

import torch


def padded_size(size: int, tp_size: int) -> int:
    if tp_size <= 0:
        raise ValueError("tp_size must be positive")
    return -(-size // tp_size) * tp_size


def column_parallel(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    tp_size: int,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Concatenate logical column-parallel shards, zero-padding output rows."""
    logical_rows = weight.shape[0]
    padded_rows = padded_size(logical_rows, tp_size)
    if padded_rows != logical_rows:
        padding = weight.new_zeros(
            (padded_rows - logical_rows, weight.shape[1])
        )
        weight = torch.cat((weight, padding), dim=0)
    local_rows = padded_rows // tp_size
    outputs = []
    for rank in range(tp_size):
        shard = weight.narrow(0, rank * local_rows, local_rows)
        if out_dtype is not None and inputs.is_cuda:
            output = torch.mm(inputs, shard.T, out_dtype=out_dtype)
        else:
            output = torch.nn.functional.linear(inputs, shard)
            if out_dtype is not None:
                output = output.to(out_dtype)
        outputs.append(output)
    return torch.cat(outputs, dim=-1)[..., :logical_rows].contiguous()


def row_parallel_partials(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    tp_size: int,
) -> list[torch.Tensor]:
    """Return logical row-parallel rank partials with zero-padded input."""
    logical_columns = weight.shape[1]
    padded_columns = padded_size(logical_columns, tp_size)
    if padded_columns != logical_columns:
        inputs = torch.nn.functional.pad(
            inputs, (0, padded_columns - logical_columns)
        )
        weight = torch.nn.functional.pad(
            weight, (0, padded_columns - logical_columns)
        )
    local_columns = padded_columns // tp_size
    return [
        torch.nn.functional.linear(
            inputs.narrow(-1, rank * local_columns, local_columns),
            weight.narrow(1, rank * local_columns, local_columns),
        )
        for rank in range(tp_size)
    ]


def reduce_partials(
    partials: list[torch.Tensor],
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Deterministically reduce rank partials and round to collective dtype."""
    if not partials:
        raise ValueError("at least one partial is required")
    return torch.stack([value.float() for value in partials]).sum(0).to(dtype)


def situ(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    beta: float = 4.0,
    linear_beta: float = 25.0,
) -> torch.Tensor:
    gate_value = beta * torch.tanh(gate.float() / beta) * torch.sigmoid(
        gate.float()
    )
    up_value = linear_beta * torch.tanh(up.float() / linear_beta)
    return (gate_value * up_value).to(gate.dtype)


def rms_norm(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    variance = inputs.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = inputs.float() * torch.rsqrt(variance + epsilon)
    return (normalized * weight.float()).to(inputs.dtype)


def kimi_topk(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    *,
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """K3's one-group sigmoid router with biased selection."""
    scores = router_logits.float().sigmoid()
    selected = torch.topk(
        scores + correction_bias.float().unsqueeze(0),
        k=topk,
        dim=-1,
        sorted=True,
    ).indices
    weights = scores.gather(1, selected)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights * routed_scaling_factor
    return weights.float(), selected.to(torch.int32)


def comparison_metrics(
    reference: torch.Tensor,
    actual: torch.Tensor,
) -> dict[str, float]:
    reference = reference.float()
    actual = actual.float()
    delta = actual - reference
    denominator = reference.norm().clamp_min(torch.finfo(torch.float32).tiny)
    cosine = torch.nn.functional.cosine_similarity(
        reference.reshape(1, -1), actual.reshape(1, -1)
    )
    # Transfer the three scalar diagnostics together.  Calling ``float`` on
    # each CUDA tensor separately introduces three device synchronizations in
    # hot offline-validation loops while producing the same float32 values.
    values = torch.stack(
        (delta.norm() / denominator, delta.abs().max(), cosine.reshape(()))
    ).cpu()
    return {
        "relative_l2": float(values[0]),
        "max_abs": float(values[1]),
        "cosine": float(values[2]),
    }
