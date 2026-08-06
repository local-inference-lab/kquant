"""Candidate-specific covariance helpers for offline expert encoding studies."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence

import torch


def partition_documents(
    documents: Sequence[str], *, modulus: int, confirmation_index: int
) -> tuple[list[str], list[str]]:
    """Deterministically split document hashes into fit and confirmation sets."""

    if modulus < 2:
        raise ValueError("document partition modulus must be at least 2")
    if not 0 <= confirmation_index < modulus:
        raise ValueError("confirmation index must be in the partition modulus")
    if not documents or len(set(documents)) != len(documents):
        raise ValueError("documents must be a nonempty unique sequence")
    fit: list[str] = []
    confirmation: list[str] = []
    for document in documents:
        digest = hashlib.blake2b(document.encode(), digest_size=8).digest()
        bucket = int.from_bytes(digest, "little") % modulus
        (confirmation if bucket == confirmation_index else fit).append(document)
    if not fit or not confirmation:
        raise ValueError("document partition produced an empty side")
    return fit, confirmation


def subset_request_documents(
    request_documents: Mapping[int, str], documents: Sequence[str]
) -> dict[int, str]:
    """Return request/document entries belonging to ``documents`` exactly."""

    selected = set(documents)
    if not selected:
        raise ValueError("document subset must be nonempty")
    result = {
        int(request): document
        for request, document in request_documents.items()
        if document in selected
    }
    covered = set(result.values())
    if covered != selected:
        missing = sorted(selected - covered)
        raise ValueError(f"documents lack request epochs: {missing[:8]}")
    return result


def request_mask(
    observations: torch.Tensor, request_documents: Mapping[int, str]
) -> torch.Tensor:
    """Select capture rows whose encoded request epoch is in a corpus subset."""

    if observations.ndim != 1:
        raise ValueError("observations must be one-dimensional")
    if not request_documents:
        raise ValueError("request document mapping must be nonempty")
    steps = torch.bitwise_right_shift(observations, 32)
    allowed = torch.tensor(
        sorted(request_documents), dtype=steps.dtype, device=steps.device
    )
    return torch.isin(steps, allowed)


def weighted_covariance(
    rows: torch.Tensor,
    weights: torch.Tensor,
    *,
    device: torch.device,
    chunk_rows: int = 256,
    return_cpu: bool = True,
) -> tuple[torch.Tensor, float]:
    """Return ``sum(w * x xT) / sum(w)`` as symmetric FP32."""

    if rows.ndim != 2 or weights.ndim != 1 or rows.shape[0] != weights.shape[0]:
        raise ValueError("invalid covariance sample shapes")
    if not rows.shape[0] or not rows.shape[1]:
        raise ValueError("covariance samples must be nonempty")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    if not torch.all(torch.isfinite(rows)) or not torch.all(torch.isfinite(weights)):
        raise ValueError("covariance samples must be finite")
    if torch.any(weights < 0):
        raise ValueError("covariance weights must be nonnegative")

    dimension = int(rows.shape[1])
    accumulator = torch.zeros(
        (dimension, dimension), dtype=torch.float32, device=device
    )
    denominator = float(weights.double().sum())
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("covariance weights must have a positive finite sum")
    for begin in range(0, int(rows.shape[0]), chunk_rows):
        values = rows[begin : begin + chunk_rows].to(
            device=device, dtype=torch.float32
        )
        local_weights = weights[begin : begin + chunk_rows].to(
            device=device, dtype=torch.float32
        )
        accumulator.addmm_(values.T, values * local_weights[:, None])
    covariance = accumulator.div_(denominator)
    covariance = ((covariance + covariance.T) * 0.5).contiguous()
    if return_cpu:
        covariance = covariance.cpu()
    if not torch.all(torch.isfinite(covariance)):
        raise ValueError("weighted covariance is non-finite")
    return covariance, denominator


def shrink_covariance(
    global_covariance: torch.Tensor,
    local_covariance: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Convexly shrink a low-support local covariance toward a global one."""

    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and in [0, 1]")
    if (
        global_covariance.ndim != 2
        or global_covariance.shape[0] != global_covariance.shape[1]
        or local_covariance.shape != global_covariance.shape
    ):
        raise ValueError("global and local covariances must be equal-size squares")
    if not torch.all(torch.isfinite(global_covariance)) or not torch.all(
        torch.isfinite(local_covariance)
    ):
        raise ValueError("covariances must be finite")
    blended = torch.lerp(
        global_covariance.float(), local_covariance.float(), alpha
    )
    return ((blended + blended.T) * 0.5).contiguous()


def weighted_effective_sample_size(weights: torch.Tensor) -> float:
    """Return Kish effective sample size for nonnegative row weights."""

    if weights.ndim != 1 or not weights.numel():
        raise ValueError("weights must be a nonempty vector")
    if not torch.all(torch.isfinite(weights)) or torch.any(weights < 0):
        raise ValueError("weights must be finite and nonnegative")
    weights64 = weights.double()
    total = weights64.sum()
    squared_total = weights64.square().sum()
    if total <= 0 or squared_total <= 0:
        raise ValueError("weights must have positive effective support")
    return float(total.square() / squared_total)


def adaptive_identity_shrinkage(
    local_covariance: torch.Tensor,
    weights: torch.Tensor,
    *,
    max_local_alpha: float = 0.75,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Shrink a weighted local covariance toward its scaled identity.

    The Oracle Approximating Shrinkage expression is evaluated with Kish
    effective sample size in place of an unweighted row count.  The target is
    always ``trace(local) / dimension * I``.  This is important for expert
    ``H2`` matrices: intermediate coordinates have no common meaning across
    experts, so a pooled post-SiTU covariance is not a valid prior.
    ``max_local_alpha`` preserves a nonzero identity contribution even for
    very well-supported experts.
    """

    max_local_alpha = float(max_local_alpha)
    if not math.isfinite(max_local_alpha) or not 0.0 <= max_local_alpha <= 1.0:
        raise ValueError("max_local_alpha must be finite and in [0, 1]")
    if local_covariance.ndim != 2 or (
        local_covariance.shape[0] != local_covariance.shape[1]
    ):
        raise ValueError("local covariance must be square")
    if not torch.all(torch.isfinite(local_covariance)):
        raise ValueError("local covariance must be finite")

    effective_rows = weighted_effective_sample_size(weights)
    dimension = int(local_covariance.shape[0])
    local64 = local_covariance.double()
    local_trace = torch.trace(local64)
    if local_trace <= 0:
        raise ValueError("local covariance trace must be positive")

    mu = local_trace / dimension
    mean_square = local64.square().mean()
    denominator = (effective_rows + 1.0) * (
        mean_square - mu.square() / dimension
    )
    if denominator <= 0:
        oas_shrinkage = 1.0
    else:
        oas_shrinkage = min(
            1.0,
            max(0.0, float((mean_square + mu.square()) / denominator)),
        )
    local_alpha = min(max_local_alpha, 1.0 - oas_shrinkage)
    identity_scale = float(mu)
    scaled_identity = torch.eye(
        dimension,
        dtype=torch.float32,
        device=local_covariance.device,
    ).mul_(identity_scale)
    blended = shrink_covariance(scaled_identity, local_covariance, local_alpha)
    return blended, {
        "effective_sample_size": effective_rows,
        "oas_shrinkage": oas_shrinkage,
        "local_alpha": local_alpha,
        "max_local_alpha": max_local_alpha,
        "identity_scale": identity_scale,
    }


def covariance_comparison(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, float]:
    """Compact scale/shape diagnostics for two covariance matrices."""

    if reference.ndim != 2 or reference.shape[0] != reference.shape[1]:
        raise ValueError("reference covariance must be square")
    if candidate.shape != reference.shape:
        raise ValueError("candidate covariance shape differs")
    reference = reference.float()
    candidate = candidate.float()
    if not torch.all(torch.isfinite(reference)) or not torch.all(torch.isfinite(candidate)):
        raise ValueError("covariances must be finite")
    reference_norm = torch.linalg.vector_norm(reference.double())
    reference_diagonal = torch.diagonal(reference).double()
    candidate_diagonal = torch.diagonal(candidate).double()
    diagonal_denominator = torch.linalg.vector_norm(
        reference_diagonal
    ) * torch.linalg.vector_norm(candidate_diagonal)
    trace = torch.trace(reference.double())
    return {
        "relative_frobenius_difference": float(
            torch.linalg.vector_norm((candidate - reference).double())
            / reference_norm
        )
        if reference_norm > 0
        else math.inf,
        "trace_ratio": float(torch.trace(candidate.double()) / trace)
        if trace != 0
        else math.inf,
        "diagonal_cosine_similarity": float(
            torch.dot(reference_diagonal, candidate_diagonal)
            / diagonal_denominator
        )
        if diagonal_denominator > 0
        else 0.0,
        "candidate_symmetry_max_abs": float(
            (candidate - candidate.T).abs().max()
        ),
    }
