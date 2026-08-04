"""Small-state trellis coding over the native MXFP4 E2M1 alphabet.

This is deliberately not the EXL3 MCG construction.  An E2M1 trellis uses
exactly sixteen labelled transitions per coefficient.  At rate ``K`` the
payload supplies ``K`` branch bits and the remaining ``4 - K`` bits are the
carried state::

    transition_index = current_state * (1 << K) + branch
    next_state = branch & ((1 << (4 - K)) - 1)

The transition label is a permutation of the sixteen native E2M1 wire codes.
Codes 0 and 8 are distinct labels but both decode to numerical zero.  Keeping
both labels lets the graph use them on different transitions without changing
the reconstructed weight.

Tiles are tail-bitten.  Since the next state is the low carried-state bits of
the branch, the decoder recovers the initial state from the final branch and
no per-tile state metadata is required.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from kquant import constants as C


E2M1_VALUES = torch.tensor(C.E2M1_LUT, dtype=torch.float32)
TRANSITIONS = 16


def _validate_bits(bits: int) -> None:
    if isinstance(bits, bool) or not isinstance(bits, int) or bits not in (2, 3, 4):
        raise ValueError("E2M1 trellis rate must be K=2, K=3, or K=4")


def _as_transition_codes(
    transition_codes: torch.Tensor | tuple[int, ...] | list[int],
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    codes = torch.as_tensor(transition_codes, dtype=torch.long, device=device)
    if codes.ndim != 1 or codes.numel() != TRANSITIONS:
        raise ValueError("transition_codes must contain exactly 16 entries")
    if not torch.equal(torch.sort(codes).values.cpu(), torch.arange(TRANSITIONS)):
        raise ValueError("transition_codes must be a permutation of E2M1 codes 0..15")
    return codes.contiguous()


def identity_transition_codes() -> tuple[int, ...]:
    """Return the native nibble ordering (useful only as a control)."""

    return tuple(range(TRANSITIONS))


def opposite_zero_transition_codes() -> tuple[int, ...]:
    """Return a deterministic seed with the two zero labels at 0000 and 1111.

    The other labels retain their native order.  This is a topology seed, not
    a claim that native ordering is the optimal labelling for K2 or K3.
    """

    codes = list(range(TRANSITIONS))
    codes[8], codes[15] = codes[15], codes[8]
    assert codes[0] == 0 and codes[15] == 8
    return tuple(codes)


@dataclass(frozen=True)
class TrellisGeometry:
    bits: int
    carried_bits: int
    states: int
    branches: int

    @property
    def state_mask(self) -> int:
        return self.states - 1


def geometry(bits: int) -> TrellisGeometry:
    _validate_bits(bits)
    carried_bits = 4 - bits
    return TrellisGeometry(
        bits=bits,
        carried_bits=carried_bits,
        states=1 << carried_bits,
        branches=1 << bits,
    )


@dataclass(frozen=True)
class TrellisEncodeResult:
    payload: torch.Tensor
    reconstructed_codes: torch.Tensor
    weighted_squared_error: torch.Tensor


def _validate_source_codes(source_codes: torch.Tensor) -> None:
    if source_codes.ndim != 2 or source_codes.shape[1] <= 0:
        raise ValueError("source_codes must have shape [sequences, length]")
    if source_codes.dtype == torch.bool or source_codes.is_floating_point():
        raise TypeError("source_codes must use an integer dtype")
    if source_codes.numel() and (
        int(source_codes.min().item()) < 0 or int(source_codes.max().item()) >= TRANSITIONS
    ):
        raise ValueError("source E2M1 codes must be in 0..15")


def _validate_weights(source_codes: torch.Tensor, weights: torch.Tensor | None) -> None:
    if weights is None:
        return
    if weights.shape != source_codes.shape:
        raise ValueError("weights must have the same shape as source_codes")
    if not weights.is_floating_point():
        raise TypeError("weights must use a floating-point dtype")
    if weights.numel() and (not bool(torch.isfinite(weights).all()) or bool((weights < 0).any())):
        raise ValueError("weights must be finite and nonnegative")


def _edge_topology(
    bits: int, device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    spec = geometry(bits)
    edges = torch.arange(TRANSITIONS, dtype=torch.long, device=device)
    previous = torch.div(edges, spec.branches, rounding_mode="floor")
    branch = edges.remainder(spec.branches)
    following = branch.bitwise_and(spec.state_mask)
    # Every next state has the same number of incoming edges.  Grouping them
    # here avoids scatter operations in the inner Viterbi loop.
    by_next = torch.stack(
        [edges[following == state] for state in range(spec.states)], dim=0
    )
    return previous, branch, by_next


def _transition_distances(
    source_codes: torch.Tensor,
    transition_codes: torch.Tensor,
) -> torch.Tensor:
    values = E2M1_VALUES.to(device=source_codes.device)
    source_values = values[source_codes.long()]
    emitted_values = values[transition_codes]
    return (source_values.unsqueeze(-1) - emitted_values).square()


def _tailbiting_terminal_costs(
    source_codes: torch.Tensor,
    bits: int,
    labels: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return closed-path costs for each possible initial state."""

    spec = geometry(bits)
    previous, _, by_next = _edge_topology(bits, source_codes.device)
    batch = source_codes.shape[0]
    inf = torch.tensor(float("inf"), dtype=torch.float32, device=source_codes.device)
    # Carry all possible starting states together.  costs[:, start, current].
    costs = torch.full(
        (batch, spec.states, spec.states),
        inf,
        dtype=torch.float32,
        device=source_codes.device,
    )
    diagonal = torch.arange(spec.states, device=source_codes.device)
    costs[:, diagonal, diagonal] = 0.0

    for position in range(source_codes.shape[1]):
        distance = _transition_distances(source_codes[:, position], labels)
        if weights is not None:
            distance = distance * weights[:, position, None].float()
        candidates = costs.index_select(-1, previous) + distance[:, None, :]
        grouped = candidates[:, :, by_next]
        costs = grouped.amin(dim=-1)

    return costs[:, diagonal, diagonal]


def tailbiting_cost(
    source_codes: torch.Tensor,
    bits: int,
    transition_codes: torch.Tensor | tuple[int, ...] | list[int],
    *,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the minimum weighted SSE for every source sequence.

    ``weights`` multiplies squared error elementwise.  Supplying the square of
    the unchanged MX scale therefore measures exact weight-space SSE without
    materializing a floating-point source matrix.
    """

    _validate_bits(bits)
    _validate_source_codes(source_codes)
    _validate_weights(source_codes, weights)
    labels = _as_transition_codes(transition_codes, device=source_codes.device)

    if bits == 4:
        # A permutation still contains every native wire code, including both
        # zeros, so K4 has exact numerical and raw-symbol closure.
        return torch.zeros(
            source_codes.shape[0], dtype=torch.float32, device=source_codes.device
        )

    return _tailbiting_terminal_costs(
        source_codes, bits, labels, weights=weights
    ).amin(dim=-1)


def _forward_choices(
    source_codes: torch.Tensor,
    bits: int,
    labels: torch.Tensor,
    start_state: int,
    weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one fixed-start Viterbi pass and retain traceback edges."""

    spec = geometry(bits)
    previous, _, by_next = _edge_topology(bits, source_codes.device)
    batch, length = source_codes.shape
    costs = torch.full(
        (batch, spec.states),
        float("inf"),
        dtype=torch.float32,
        device=source_codes.device,
    )
    costs[:, start_state] = 0.0
    choices = torch.empty(
        (length, batch, spec.states), dtype=torch.uint8, device=source_codes.device
    )

    for position in range(length):
        distance = _transition_distances(source_codes[:, position], labels)
        if weights is not None:
            distance = distance * weights[:, position, None].float()
        candidates = costs.index_select(-1, previous) + distance
        grouped = candidates[:, by_next]
        costs, selected = grouped.min(dim=-1)
        selected_edges = (
            by_next.unsqueeze(0)
            .expand(batch, -1, -1)
            .gather(2, selected.unsqueeze(-1))
            .squeeze(-1)
        )
        choices[position] = selected_edges.to(torch.uint8)
    return costs[:, start_state], choices


def decode_payload(
    payload: torch.Tensor,
    bits: int,
    transition_codes: torch.Tensor | tuple[int, ...] | list[int],
) -> torch.Tensor:
    """Decode unpacked K-bit branches to native E2M1 wire codes."""

    _validate_bits(bits)
    if payload.ndim != 2 or payload.shape[1] <= 0:
        raise ValueError("payload must have shape [sequences, length]")
    if payload.dtype == torch.bool or payload.is_floating_point():
        raise TypeError("payload must use an integer dtype")
    spec = geometry(bits)
    if payload.numel() and (
        int(payload.min().item()) < 0 or int(payload.max().item()) >= spec.branches
    ):
        raise ValueError(f"K{bits} payload symbols must be in 0..{spec.branches - 1}")
    labels = _as_transition_codes(transition_codes, device=payload.device)
    branches = payload.long()
    state = branches[:, -1].bitwise_and(spec.state_mask)
    initial_state = state.clone()
    decoded = torch.empty_like(payload, dtype=torch.uint8)
    for position in range(payload.shape[1]):
        branch = branches[:, position]
        transition = state * spec.branches + branch
        decoded[:, position] = labels[transition].to(torch.uint8)
        state = branch.bitwise_and(spec.state_mask)
    if not torch.equal(state, initial_state):  # pragma: no cover - construction invariant
        raise RuntimeError("tail-biting state did not close")
    return decoded


def encode_tailbiting(
    source_codes: torch.Tensor,
    bits: int,
    transition_codes: torch.Tensor | tuple[int, ...] | list[int],
    *,
    weights: torch.Tensor | None = None,
) -> TrellisEncodeResult:
    """Encode and reconstruct a batch of tail-bitten E2M1 sequences."""

    _validate_bits(bits)
    _validate_source_codes(source_codes)
    _validate_weights(source_codes, weights)
    labels = _as_transition_codes(transition_codes, device=source_codes.device)
    spec = geometry(bits)

    if bits == 4:
        inverse = torch.empty(TRANSITIONS, dtype=torch.long, device=source_codes.device)
        inverse[labels] = torch.arange(TRANSITIONS, device=source_codes.device)
        payload = inverse[source_codes.long()].to(torch.uint8)
        reconstructed = decode_payload(payload, bits, labels)
        error = torch.zeros(
            source_codes.shape[0], dtype=torch.float32, device=source_codes.device
        )
        return TrellisEncodeResult(payload, reconstructed, error)

    # First find the best closed starting state without retaining traceback.
    terminal_costs = _tailbiting_terminal_costs(
        source_codes, bits, labels, weights=weights
    )
    best_cost = terminal_costs.amin(dim=1)
    best_start = terminal_costs.argmin(dim=1)

    payload = torch.empty_like(source_codes, dtype=torch.uint8)
    for start_state in range(spec.states):
        selected = torch.nonzero(best_start == start_state, as_tuple=False).flatten()
        if not selected.numel():
            continue
        selected_weights = None if weights is None else weights.index_select(0, selected)
        _, choices = _forward_choices(
            source_codes.index_select(0, selected),
            bits,
            labels,
            start_state,
            selected_weights,
        )
        current = torch.full(
            (selected.numel(),), start_state, dtype=torch.long, device=source_codes.device
        )
        local_batch = torch.arange(selected.numel(), device=source_codes.device)
        for position in range(source_codes.shape[1] - 1, -1, -1):
            edge = choices[position, local_batch, current].long()
            payload[selected, position] = edge.remainder(spec.branches).to(torch.uint8)
            current = torch.div(edge, spec.branches, rounding_mode="floor")
        if not bool((current == start_state).all()):  # pragma: no cover - traceback invariant
            raise RuntimeError("Viterbi traceback did not return to its start state")

    reconstructed = decode_payload(payload, bits, labels)
    # Guard the separate cost-only and traceback paths against drifting apart.
    reconstructed_distance = _transition_distances(
        source_codes.reshape(-1),
        torch.arange(TRANSITIONS, device=source_codes.device),
    )
    # Select the distance to each reconstructed native code.  This expression
    # treats +0 and -0 as equal, as weight-space reconstruction should.
    flat_reconstruction = reconstructed.long().reshape(-1, 1)
    flat_error = reconstructed_distance.gather(1, flat_reconstruction).reshape_as(
        source_codes
    )
    if weights is not None:
        flat_error = flat_error * weights.float()
    traced_cost = flat_error.sum(dim=1)
    if not torch.allclose(traced_cost, best_cost, rtol=2e-5, atol=1e-6):
        raise RuntimeError("Viterbi traceback cost does not match the optimum")
    return TrellisEncodeResult(payload, reconstructed, best_cost)


def pack_payload(payload: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack unpacked K-bit branch symbols into a little-endian byte stream."""

    _validate_bits(bits)
    if payload.ndim < 1:
        raise ValueError("payload must have at least one dimension")
    if payload.dtype == torch.bool or payload.is_floating_point():
        raise TypeError("payload must use an integer dtype")
    branches = 1 << bits
    if payload.numel() and (
        int(payload.min().item()) < 0 or int(payload.max().item()) >= branches
    ):
        raise ValueError(f"K{bits} payload symbols must be in 0..{branches - 1}")
    bit_count = payload.shape[-1] * bits
    if bit_count % 8:
        raise ValueError("payload length times K must be byte aligned")
    shifts = torch.arange(bits, dtype=torch.long, device=payload.device)
    expanded = payload.long().unsqueeze(-1).bitwise_right_shift(shifts).bitwise_and(1)
    expanded = expanded.reshape(*payload.shape[:-1], bit_count)
    byte_bits = expanded.reshape(*payload.shape[:-1], bit_count // 8, 8)
    byte_shifts = torch.arange(8, dtype=torch.long, device=payload.device)
    return (byte_bits.bitwise_left_shift(byte_shifts).sum(dim=-1)).to(torch.uint8)


def unpack_payload(packed: torch.Tensor, bits: int, *, symbols: int) -> torch.Tensor:
    """Inverse of :func:`pack_payload` for a known number of symbols."""

    _validate_bits(bits)
    if packed.ndim < 1 or packed.dtype != torch.uint8:
        raise TypeError("packed payload must be a uint8 tensor with at least one dimension")
    if symbols <= 0 or symbols * bits % 8:
        raise ValueError("symbols must be positive and byte aligned at the requested K")
    expected_bytes = symbols * bits // 8
    if packed.shape[-1] != expected_bytes:
        raise ValueError(
            f"packed payload has {packed.shape[-1]} bytes, expected {expected_bytes}"
        )
    shifts = torch.arange(8, dtype=torch.long, device=packed.device)
    bits_flat = packed.long().unsqueeze(-1).bitwise_right_shift(shifts).bitwise_and(1)
    bits_flat = bits_flat.reshape(*packed.shape[:-1], symbols * bits)
    symbol_bits = bits_flat.reshape(*packed.shape[:-1], symbols, bits)
    symbol_shifts = torch.arange(bits, dtype=torch.long, device=packed.device)
    return (symbol_bits.bitwise_left_shift(symbol_shifts).sum(dim=-1)).to(torch.uint8)
