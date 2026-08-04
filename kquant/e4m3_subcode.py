"""Small learned E4M3 enhancement codes over procedural predictors."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from kquant.trellis_window import fp8_constrained_codebook


E4M3_CODES = 256


@dataclass(frozen=True)
class ConditionalBinaryE4M3Fit:
    """One learned bit per transition plus a tiny predictor-conditioned table."""

    codebook: torch.Tensor
    transition_bits: torch.Tensor
    pair_codes: torch.Tensor
    pair_values: torch.Tensor
    objective: float
    iterations: int
    fixed_primary: bool


def e4m3_codes(values: torch.Tensor) -> torch.Tensor:
    """Encode finite floating values to their exact E4M3 byte representation."""

    if not values.is_floating_point() or not bool(torch.isfinite(values).all()):
        raise ValueError("E4M3 source values must be finite floating values")
    maximum = torch.finfo(torch.float8_e4m3fn).max
    return (
        values.float()
        .clamp(-maximum, maximum)
        .to(torch.float8_e4m3fn)
        .contiguous()
        .view(torch.uint8)
    )


def e4m3_values(codes: torch.Tensor) -> torch.Tensor:
    """Decode E4M3 bytes to float32 while rejecting its two NaN codes."""

    if codes.dtype != torch.uint8:
        raise TypeError("E4M3 codes must be uint8")
    values = codes.contiguous().view(torch.float8_e4m3fn).float()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("E4M3 codes contain NaN")
    return values


def expand_conditional_binary_e4m3(
    predictor: torch.Tensor,
    transition_bits: torch.Tensor,
    pair_codes: torch.Tensor,
) -> torch.Tensor:
    """Expand the procedural predictor, bit plane, and 256x2 byte table."""

    if not predictor.is_floating_point() or predictor.ndim != 1:
        raise ValueError("predictor must be a floating vector")
    if (
        transition_bits.shape != predictor.shape
        or transition_bits.dtype == torch.bool
        or transition_bits.is_floating_point()
    ):
        raise ValueError("transition_bits must be an integer vector matching predictor")
    if pair_codes.shape != (E4M3_CODES, 2) or pair_codes.dtype != torch.uint8:
        raise ValueError("pair_codes must be a uint8 [256, 2] table")
    if predictor.device != transition_bits.device or predictor.device != pair_codes.device:
        raise ValueError("binary-subcode tensors must share a device")
    bits = transition_bits.long()
    if int(bits.min().item()) < 0 or int(bits.max().item()) > 1:
        raise ValueError("transition_bits must contain only zero or one")
    anchors = e4m3_codes(predictor).long()
    selected = pair_codes[anchors, bits]
    return e4m3_values(selected).contiguous()


def logical_conditional_binary_e4m3_bytes(
    transitions: int,
    *,
    fixed_primary: bool,
) -> int:
    """Return packed bit-plane plus global reconstruction-table bytes."""

    if transitions <= 0:
        raise ValueError("transitions must be positive")
    bitplane = (transitions + 7) // 8
    # If the primary output is the predictor itself, only its alternative is
    # stored. Otherwise both reconstruction bytes are explicit.
    pair_table = E4M3_CODES if fixed_primary else E4M3_CODES * 2
    return bitplane + pair_table


def pack_transition_bits(transition_bits: torch.Tensor) -> torch.Tensor:
    """Pack one transition bit per edge, least-significant bit first."""

    if (
        transition_bits.ndim != 1
        or transition_bits.dtype == torch.bool
        or transition_bits.is_floating_point()
    ):
        raise ValueError("transition_bits must be an integer vector")
    bits = transition_bits.to(torch.uint8)
    if int(bits.min().item()) < 0 or int(bits.max().item()) > 1:
        raise ValueError("transition_bits must contain only zero or one")
    padding = (-bits.numel()) % 8
    if padding:
        bits = torch.cat(
            (
                bits,
                torch.zeros(padding, dtype=torch.uint8, device=bits.device),
            )
        )
    lanes = bits.reshape(-1, 8)
    shifts = torch.arange(8, dtype=torch.uint8, device=bits.device)
    return torch.sum(lanes << shifts, dim=1, dtype=torch.int16).to(torch.uint8)


def unpack_transition_bits(packed: torch.Tensor, count: int) -> torch.Tensor:
    """Unpack a little-endian transition bit plane."""

    if packed.ndim != 1 or packed.dtype != torch.uint8:
        raise ValueError("packed transition bits must be a uint8 vector")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    if packed.numel() != (count + 7) // 8:
        raise ValueError("packed transition bit length does not match count")
    shifts = torch.arange(8, dtype=torch.uint8, device=packed.device)
    bits = ((packed.unsqueeze(1) >> shifts) & 1).reshape(-1)
    return bits[:count].contiguous()


def decode_predictor_anchor_plus_alternative(
    predictor: torch.Tensor,
    packed_transition_bits: torch.Tensor,
    alternative_codes: torch.Tensor,
) -> torch.Tensor:
    """Reference decode for the 8-KiB bitplane plus 256-byte table format."""

    if alternative_codes.shape != (E4M3_CODES,) or alternative_codes.dtype != torch.uint8:
        raise ValueError("alternative_codes must be a uint8 [256] table")
    if predictor.device != packed_transition_bits.device or predictor.device != alternative_codes.device:
        raise ValueError("binary-subcode tensors must share a device")
    bits = unpack_transition_bits(packed_transition_bits, predictor.numel()).long()
    anchors = e4m3_codes(predictor)
    selected = torch.where(
        bits.bool(),
        alternative_codes.index_select(0, anchors.long()),
        anchors,
    )
    return e4m3_values(selected).contiguous()


# Backward-compatible name retained for existing experiment artifacts/tests.
decode_mcg_anchor_plus_alternative = decode_predictor_anchor_plus_alternative


def _project_scalar(value: torch.Tensor) -> torch.Tensor:
    return fp8_constrained_codebook(
        value.reshape(1), dtype=torch.float8_e4m3fn
    )[0]


def fit_conditional_binary_e4m3(
    transition_targets: torch.Tensor,
    transition_weights: torch.Tensor,
    predictor: torch.Tensor,
    *,
    iterations: int = 32,
    fixed_primary: bool = False,
    initial_pair_codes: torch.Tensor | None = None,
    initial_transition_bits: torch.Tensor | None = None,
) -> ConditionalBinaryE4M3Fit:
    """Fit two E4M3 outputs per projected procedural-predictor value.

    The learned transition bit is stored once for each graph edge.  The pair
    table is conditioned only on the E4M3 byte obtained by projecting the
    procedural predictor value, so it remains a fixed 256-row decoder table.
    """

    if (
        transition_targets.ndim != 1
        or not transition_targets.is_floating_point()
        or predictor.shape != transition_targets.shape
        or not predictor.is_floating_point()
    ):
        raise ValueError("targets and predictor must be matching floating vectors")
    if (
        transition_weights.shape != transition_targets.shape
        or not transition_weights.is_floating_point()
    ):
        raise ValueError("weights must be floating and match targets")
    if len({transition_targets.device, transition_weights.device, predictor.device}) != 1:
        raise ValueError("binary-subcode inputs must share a device")
    if not bool(torch.isfinite(transition_targets).all()) or not bool(
        torch.isfinite(predictor).all()
    ):
        raise ValueError("targets and predictor must be finite")
    if (
        not bool(torch.isfinite(transition_weights).all())
        or bool(torch.any(transition_weights < 0.0))
        or not bool(torch.any(transition_weights > 0.0))
    ):
        raise ValueError("weights must be finite, nonnegative, and nonzero")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be positive")

    targets = transition_targets.float()
    weights = transition_weights.float()
    anchor_codes = e4m3_codes(predictor)
    anchor_values = e4m3_values(anchor_codes)
    all_codes = torch.arange(E4M3_CODES, dtype=torch.uint8, device=targets.device)
    all_values_raw = all_codes.view(torch.float8_e4m3fn).float()
    finite_code = torch.isfinite(all_values_raw)
    safe_values = torch.where(finite_code, all_values_raw, torch.zeros_like(all_values_raw))

    if initial_pair_codes is not None:
        if initial_pair_codes.shape != (E4M3_CODES, 2) or initial_pair_codes.dtype != torch.uint8:
            raise ValueError("initial_pair_codes must be uint8 [256, 2]")
        if initial_pair_codes.device != targets.device:
            raise ValueError("initial_pair_codes must share the input device")
        pair_codes = initial_pair_codes.clone()
        pair_values = e4m3_values(pair_codes)
    else:
        pair_codes = torch.empty(
            (E4M3_CODES, 2), dtype=torch.uint8, device=targets.device
        )
        pair_codes[:, 0] = all_codes
        pair_codes[:, 1] = all_codes
        # Replace the two NaN anchors with zero. They cannot be selected from
        # a finite predictor, but keeping the complete table valid simplifies
        # serialization and exhaustive decoder tests.
        pair_codes[~finite_code] = e4m3_codes(
            torch.zeros(int(torch.count_nonzero(~finite_code).item()), device=targets.device)
        ).unsqueeze(1)
        pair_values = e4m3_values(pair_codes)
        for code in torch.unique(anchor_codes).tolist():
            mask = anchor_codes == code
            local_target = targets[mask]
            local_weight = weights[mask]
            anchor = anchor_values[mask][0]
            farthest = torch.argmax(local_weight * (local_target - anchor).square())
            alternative = _project_scalar(local_target[farthest])
            pair_values[code, 0] = anchor
            pair_values[code, 1] = alternative
        pair_codes = e4m3_codes(pair_values)

    if initial_transition_bits is not None:
        if initial_transition_bits.shape != targets.shape:
            raise ValueError("initial_transition_bits have the wrong shape")
        transition_bits = initial_transition_bits.long().clone()
    else:
        candidates = pair_values.index_select(0, anchor_codes.long())
        transition_bits = (
            weights[:, None] * (targets[:, None] - candidates).square()
        ).argmin(dim=1)

    completed = 0
    for completed in range(1, iterations + 1):
        previous_codes = pair_codes
        previous_bits = transition_bits
        updated_values = pair_values.clone()
        for code in torch.unique(anchor_codes).tolist():
            mask = anchor_codes == code
            local_target = targets[mask]
            local_weight = weights[mask]
            local_bits = transition_bits[mask]
            start = 1 if fixed_primary else 0
            if fixed_primary:
                updated_values[code, 0] = safe_values[code]
            for branch in range(start, 2):
                selected = local_bits == branch
                if bool(torch.any(selected)):
                    centroid = (
                        local_weight[selected] * local_target[selected]
                    ).sum() / local_weight[selected].sum()
                    updated_values[code, branch] = _project_scalar(centroid)
        pair_codes = e4m3_codes(updated_values)
        pair_values = e4m3_values(pair_codes)
        candidates = pair_values.index_select(0, anchor_codes.long())
        transition_bits = (
            weights[:, None] * (targets[:, None] - candidates).square()
        ).argmin(dim=1)
        if torch.equal(pair_codes, previous_codes) and torch.equal(
            transition_bits, previous_bits
        ):
            break

    codebook = expand_conditional_binary_e4m3(
        predictor, transition_bits, pair_codes
    )
    objective = (
        weights * (codebook - targets).square()
    ).sum()
    return ConditionalBinaryE4M3Fit(
        codebook=codebook,
        transition_bits=transition_bits.to(torch.uint8),
        pair_codes=pair_codes,
        pair_values=pair_values,
        objective=float(objective.item()),
        iterations=completed,
        fixed_primary=fixed_primary,
    )
