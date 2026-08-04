"""Reference trellis quantization with a configurable rolling-state window.

EXL3's production encoder fixes the rolling code index to 16 bits.  This
module deliberately favors clarity over kernel speed so experiments can vary
that inherited choice without changing the production extension.  Its
two-pass tail-biting approximation matches the control flow used by EXL3:
an unconstrained pass rotated by half a tile proposes a boundary state, then
a second pass starts and ends at that state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from kquant.exl3_reference import decode_mcg_states


@dataclass(frozen=True)
class TrellisGeometry:
    bits: int
    window_bits: int
    branches: int
    states: int
    transitions: int
    predecessor_shift: int


@dataclass(frozen=True)
class TrellisEncoding:
    reconstructed: torch.Tensor
    squared_error: torch.Tensor
    transition_indices: torch.Tensor | None


def _edge_alias_summary(values: torch.Tensor) -> tuple[dict[str, float | int], torch.Tensor]:
    """Summarize equal-valued edges in ``[state, branch]`` order."""

    equality = values.unsqueeze(2) == values.unsqueeze(1)
    multiplicity = equality.sum(dim=2)
    aliased = multiplicity > 1
    unique_per_state = (~equality.tril(diagonal=-1).any(dim=2)).sum(dim=1)
    return (
        {
            "states": int(values.shape[0]),
            "branches_per_state": int(values.shape[1]),
            "states_with_alias_fraction": float(aliased.any(dim=1).float().mean().item()),
            "edges_with_alias_fraction": float(aliased.float().mean().item()),
            "mean_distinct_labels_per_state": float(unique_per_state.float().mean().item()),
            "min_distinct_labels_per_state": int(unique_per_state.min().item()),
            "max_alias_multiplicity": int(multiplicity.max().item()),
            # If branches are equiprobable, this is H(edge | state, emitted
            # value): the number of current-output-preserving branch bits.
            "conditional_alias_bits_uniform": float(
                multiplicity.float().log2().mean().item()
            ),
        },
        multiplicity,
    )


def trellis_alias_metrics(
    codebook: torch.Tensor,
    bits: int,
    window_bits: int,
    *,
    transition_indices: torch.Tensor | None = None,
) -> dict[str, object]:
    """Measure duplicate-label steering in a rolling-window trellis.

    A transition index is ``(origin_state << K) | edge_symbol``.  Equal
    reconstruction values on two outgoing edges preserve the current numeric
    output while choosing different successor states.  The outgoing metrics
    therefore quantify the state-steering freedom created by duplicate
    numerical labels.  Incoming aliases are reported separately because they
    quantify equal-cost predecessor merges in the Viterbi recurrence.

    ``transition_indices`` may contain paths chosen on a corpus.  When
    supplied, the result also reports how often those selected transitions
    had an equal-valued alternative available from the same origin state.
    """

    spec = geometry(bits, window_bits)
    if codebook.ndim != 1 or codebook.numel() != spec.transitions:
        raise ValueError(f"codebook must contain exactly {spec.transitions} values")
    if not codebook.is_floating_point():
        raise TypeError("codebook must be floating point")
    if not bool(torch.isfinite(codebook).all()):
        raise ValueError("codebook must be finite")

    labels, inverse, counts = torch.unique(
        codebook,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )
    probabilities = counts.double() / counts.sum()
    label_entropy = -(probabilities * probabilities.log2()).sum()

    # In causal order, each origin state owns a contiguous run of 2**K
    # transition labels.  Every edge appends a different K-bit symbol and
    # therefore reaches a different successor state.
    outgoing_values = codebook.reshape(spec.states, spec.branches)
    outgoing, outgoing_multiplicity = _edge_alias_summary(outgoing_values)

    # The forward recurrence stores the same transitions in
    # [predecessor-high-bits, destination-state] order.
    incoming_values = codebook.reshape(spec.branches, spec.states).T.contiguous()
    incoming, _ = _edge_alias_summary(incoming_values)

    # Refine state classes by their ordered future labelled behavior.  At
    # depth d, two states share a class only when the same next d branch
    # symbols emit the same numeric sequence.  This distinguishes useful
    # aliases from duplicate edges whose successor states are observationally
    # equivalent over the remaining rolling-memory horizon.
    outgoing_label_ids = inverse.reshape(spec.states, spec.branches)
    origins = torch.arange(spec.states, device=codebook.device).unsqueeze(1)
    edges = torch.arange(spec.branches, device=codebook.device).unsqueeze(0)
    successors = ((origins << bits) | edges) & (spec.states - 1)
    label_equality = outgoing_label_ids.unsqueeze(2) == outgoing_label_ids.unsqueeze(1)
    aliased_edges = outgoing_multiplicity > 1
    right_context: list[dict[str, float | int]] = []
    classes = torch.zeros(spec.states, dtype=torch.long, device=codebook.device)
    context_depth = math.ceil((window_bits - bits) / bits)
    for depth in range(1, context_depth + 1):
        signatures = torch.stack(
            (outgoing_label_ids, classes.index_select(0, successors.flatten()).reshape_as(successors)),
            dim=2,
        ).reshape(spec.states, -1)
        _, classes = torch.unique(signatures, dim=0, return_inverse=True)
        successor_classes = classes.index_select(0, successors.flatten()).reshape_as(successors)
        same_label_and_context = label_equality & (
            successor_classes.unsqueeze(2) == successor_classes.unsqueeze(1)
        )
        first_context_in_label = ~same_label_and_context.tril(diagonal=-1).any(dim=2)
        distinct_contexts = (
            label_equality & first_context_in_label.unsqueeze(1)
        ).sum(dim=2)
        distinguishable = distinct_contexts > 1
        right_context.append(
            {
                "future_steps": depth,
                "right_context_classes": int(classes.max().item()) + 1,
                "edges_with_distinguishable_alias_fraction": float(
                    distinguishable.float().mean().item()
                ),
                "distinguishable_fraction_among_aliased_edges": float(
                    distinguishable[aliased_edges].float().mean().item()
                    if bool(torch.any(aliased_edges))
                    else 0.0
                ),
                "conditional_distinguishable_alias_bits_uniform": float(
                    distinct_contexts.float().log2().mean().item()
                ),
            }
        )

    result: dict[str, object] = {
        "bits": bits,
        "window_bits": window_bits,
        "states": spec.states,
        "transitions": spec.transitions,
        "numeric_labels": int(labels.numel()),
        "uniform_numeric_label_entropy_bits": float(label_entropy.item()),
        "outgoing": outgoing,
        "incoming": incoming,
        "right_context_distinguishability": right_context,
    }

    if transition_indices is not None:
        if transition_indices.dtype == torch.bool or transition_indices.is_floating_point():
            raise TypeError("transition_indices must use an integer dtype")
        selected = transition_indices.to(device=codebook.device, dtype=torch.long).flatten()
        if selected.numel() == 0:
            raise ValueError("transition_indices must be nonempty")
        if int(selected.min().item()) < 0 or int(selected.max().item()) >= spec.transitions:
            raise ValueError("transition_indices contain an out-of-range transition")
        selected_multiplicity = outgoing_multiplicity.flatten().index_select(0, selected)
        selected_labels = inverse.index_select(0, selected)
        selected_counts = torch.bincount(selected_labels, minlength=labels.numel())
        selected_probabilities = selected_counts[selected_counts > 0].double()
        selected_probabilities /= selected_probabilities.sum()
        selected_entropy = -(
            selected_probabilities * selected_probabilities.log2()
        ).sum()
        result["selected"] = {
            "transitions": int(selected.numel()),
            "numeric_labels": int(torch.count_nonzero(selected_counts).item()),
            "numeric_label_entropy_bits": float(selected_entropy.item()),
            "edges_with_alias_available_fraction": float(
                (selected_multiplicity > 1).float().mean().item()
            ),
            # This is the available branch freedom on selected edges, not an
            # empirical entropy estimate of the path-generating process.
            "mean_available_alias_bits": float(
                selected_multiplicity.float().log2().mean().item()
            ),
            "max_available_alias_multiplicity": int(
                selected_multiplicity.max().item()
            ),
        }
    return result


def geometry(bits: int, window_bits: int) -> TrellisGeometry:
    if isinstance(bits, bool) or not isinstance(bits, int) or bits not in (2, 3, 4, 5):
        raise ValueError("bits must be K=2, K=3, K=4, or K=5")
    if isinstance(window_bits, bool) or not isinstance(window_bits, int):
        raise TypeError("window_bits must be an integer")
    if window_bits < 2 * bits:
        raise ValueError("window_bits must be at least 2*K")
    if window_bits > 16:
        raise ValueError("the MCG reference accepts at most a 16-bit state")
    return TrellisGeometry(
        bits=bits,
        window_bits=window_bits,
        branches=1 << bits,
        states=1 << (window_bits - bits),
        transitions=1 << window_bits,
        predecessor_shift=window_bits - 2 * bits,
    )


def mcg_codebook(
    window_bits: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Decode the low ``window_bits`` MCG indices, zero-extended to 16 bits."""

    # Validate through a legal K.  L >= 8 in every planned experiment, so K=2
    # is sufficient and avoids giving this public helper a second validator.
    geometry(2, window_bits)
    states = torch.arange(1 << window_bits, dtype=torch.int64, device=device)
    return decode_mcg_states(states).to(dtype=dtype).contiguous()


def fp8_constrained_codebook(
    codebook: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float8_e4m3fn,
    alphabet_scale: float = 1.0,
) -> torch.Tensor:
    """Project a codebook onto a scaled FP8 reconstruction alphabet.

    The returned tensor is float32 for the reference dynamic program, but
    every value divided by ``alphabet_scale`` is exactly representable in the
    requested FP8 format.  This projection defines the alphabet; callers must
    rerun Viterbi to obtain an FP8-aware path.
    """

    if dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError("dtype must be float8_e4m3fn or float8_e5m2")
    if not math.isfinite(alphabet_scale) or alphabet_scale <= 0.0:
        raise ValueError("alphabet_scale must be positive and finite")
    maximum = torch.finfo(dtype).max
    normalized = (codebook.float() / alphabet_scale).clamp(-maximum, maximum)
    return (
        normalized.to(dtype=dtype).float() * alphabet_scale
    ).contiguous()


def refit_transition_codebook(
    source: torch.Tensor,
    transition_indices: torch.Tensor,
    initial_codebook: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    ridge_pseudocount: float = 0.0,
    reconstruction_dtype: torch.dtype | None = None,
    alphabet_scale: float = 1.0,
) -> torch.Tensor:
    """Fit fixed-path reconstruction centroids for a transition table.

    This is the diagonal-metric Lloyd update.  ``source`` is already in the
    codebook's numeric domain.  Optional nonnegative ``weights`` allow callers
    to undo per-sample gain weighting or supply a diagonal distortion metric.
    Unvisited transitions retain their initial labels.  A ridge pseudocount
    shrinks visited labels toward ``initial_codebook`` and is useful when a
    65,536-entry free table is fitted from a modest calibration corpus.

    ``reconstruction_dtype=None`` returns unrestricted float32 centroids.
    FP16 and FP8 constraints round the fitted centroids into the corresponding
    scaled reconstruction alphabet; paths must be reassigned afterward.
    """

    if source.shape != transition_indices.shape or source.ndim == 0:
        raise ValueError("source and transition_indices must have the same non-scalar shape")
    if not source.is_floating_point():
        raise TypeError("source must be floating point")
    if transition_indices.dtype == torch.bool or transition_indices.is_floating_point():
        raise TypeError("transition_indices must use an integer dtype")
    if initial_codebook.ndim != 1 or not initial_codebook.is_floating_point():
        raise ValueError("initial_codebook must be a one-dimensional floating tensor")
    if source.device != transition_indices.device or source.device != initial_codebook.device:
        raise ValueError("source, transition_indices, and initial_codebook must share a device")
    if not math.isfinite(ridge_pseudocount) or ridge_pseudocount < 0.0:
        raise ValueError("ridge_pseudocount must be finite and nonnegative")
    if not math.isfinite(alphabet_scale) or alphabet_scale <= 0.0:
        raise ValueError("alphabet_scale must be positive and finite")
    if reconstruction_dtype not in (
        None,
        torch.float16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        raise ValueError("reconstruction_dtype must be None, FP16, E4M3, or E5M2")
    if not bool(torch.isfinite(source).all()) or not bool(torch.isfinite(initial_codebook).all()):
        raise ValueError("source and initial_codebook must be finite")

    indices = transition_indices.to(dtype=torch.long).flatten()
    if indices.numel() == 0:
        raise ValueError("source must be nonempty")
    if int(indices.min().item()) < 0 or int(indices.max().item()) >= initial_codebook.numel():
        raise ValueError("transition_indices contain an out-of-range transition")
    targets = source.float().flatten()
    if weights is None:
        sample_weights = torch.ones_like(targets)
    else:
        if weights.shape != source.shape or not weights.is_floating_point():
            raise ValueError("weights must be a floating tensor with the source shape")
        if weights.device != source.device:
            raise ValueError("weights must share the source device")
        if not bool(torch.isfinite(weights).all()) or bool(torch.any(weights < 0.0)):
            raise ValueError("weights must be finite and nonnegative")
        sample_weights = weights.float().flatten()

    total_weight = torch.zeros_like(initial_codebook, dtype=torch.float32)
    weighted_sum = torch.zeros_like(initial_codebook, dtype=torch.float32)
    total_weight.scatter_add_(0, indices, sample_weights)
    weighted_sum.scatter_add_(0, indices, sample_weights * targets)
    initial = initial_codebook.float()
    denominator = total_weight + ridge_pseudocount
    numerator = weighted_sum + ridge_pseudocount * initial
    fitted = torch.where(denominator > 0.0, numerator / denominator, initial)

    if reconstruction_dtype is None:
        return fitted.contiguous()
    if reconstruction_dtype == torch.float16:
        normalized = fitted / alphabet_scale
        return (normalized.half().float() * alphabet_scale).contiguous()
    return fp8_constrained_codebook(
        fitted,
        dtype=reconstruction_dtype,
        alphabet_scale=alphabet_scale,
    )


def tie_codebook_values(
    codebook: torch.Tensor,
    levels: int,
    *,
    weights: torch.Tensor | None = None,
    initial_centroids: torch.Tensor | None = None,
    iterations: int = 32,
    reconstruction_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tie a transition table to a learned one-dimensional centroid alphabet.

    This control preserves duplicate-label state steering while removing the
    E4M3 grid constraint.  It is a weighted scalar Lloyd fit over transition
    labels, not over source coefficients directly; callers should use path
    occupancy (plus any ridge mass) as ``weights``.
    """

    if codebook.ndim != 1 or not codebook.is_floating_point():
        raise ValueError("codebook must be a one-dimensional floating tensor")
    if isinstance(levels, bool) or not isinstance(levels, int) or levels <= 0:
        raise ValueError("levels must be a positive integer")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    if reconstruction_dtype not in (
        torch.float16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        raise ValueError("reconstruction_dtype must be FP16, E4M3, or E5M2")
    if not bool(torch.isfinite(codebook).all()):
        raise ValueError("codebook must be finite")
    values = codebook.float()
    if weights is None:
        masses = torch.ones_like(values)
    else:
        if weights.shape != codebook.shape or not weights.is_floating_point():
            raise ValueError("weights must be a floating tensor with the codebook shape")
        if weights.device != codebook.device:
            raise ValueError("weights must share the codebook device")
        if not bool(torch.isfinite(weights).all()) or bool(torch.any(weights < 0.0)):
            raise ValueError("weights must be finite and nonnegative")
        masses = weights.float()
    positive = masses > 0.0
    if not bool(torch.any(positive)):
        raise ValueError("at least one transition weight must be positive")

    if initial_centroids is not None:
        if initial_centroids.ndim != 1 or not initial_centroids.is_floating_point():
            raise ValueError("initial_centroids must be a one-dimensional floating tensor")
        if initial_centroids.device != codebook.device:
            raise ValueError("initial_centroids must share the codebook device")
        if initial_centroids.numel() > levels or initial_centroids.numel() == 0:
            raise ValueError("initial_centroids must contain between one and levels values")
        if not bool(torch.isfinite(initial_centroids).all()):
            raise ValueError("initial_centroids must be finite")
        centroids = torch.unique(initial_centroids.float(), sorted=True)
    else:
        active_values = values[positive]
        active_masses = masses[positive]
        order = active_values.argsort()
        sorted_values = active_values.index_select(0, order)
        cumulative = active_masses.index_select(0, order).cumsum(0)
        targets = (
            (torch.arange(levels, device=values.device, dtype=torch.float32) + 0.5)
            * (cumulative[-1] / levels)
        )
        indices = torch.searchsorted(cumulative, targets).clamp_max_(sorted_values.numel() - 1)
        centroids = torch.unique(sorted_values.index_select(0, indices), sorted=True)

    for _ in range(iterations):
        assignment = (values.unsqueeze(1) - centroids.unsqueeze(0)).abs().argmin(dim=1)
        total_mass = torch.zeros_like(centroids)
        weighted_sum = torch.zeros_like(centroids)
        total_mass.scatter_add_(0, assignment, masses)
        weighted_sum.scatter_add_(0, assignment, masses * values)
        updated = torch.where(total_mass > 0.0, weighted_sum / total_mass, centroids)
        if torch.equal(updated, centroids):
            break
        centroids = updated

    if reconstruction_dtype == torch.float16:
        centroids = centroids.half().float()
    else:
        centroids = fp8_constrained_codebook(
            centroids,
            dtype=reconstruction_dtype,
        )
    centroids = torch.unique(centroids, sorted=True)
    assignment = (values.unsqueeze(1) - centroids.unsqueeze(0)).abs().argmin(dim=1)
    return centroids.index_select(0, assignment).contiguous(), centroids.contiguous()


def reconstruct_window_states(edge_symbols: torch.Tensor, bits: int, window_bits: int) -> torch.Tensor:
    """Recover cyclic rolling states from K-bit edge symbols."""

    spec = geometry(bits, window_bits)
    if edge_symbols.ndim < 1:
        raise ValueError("edge_symbols must have at least one dimension")
    if edge_symbols.dtype == torch.bool or edge_symbols.is_floating_point():
        raise TypeError("edge_symbols must use an integer dtype")
    edges = edge_symbols.to(torch.int64) & (spec.branches - 1)
    mask = spec.transitions - 1
    state = torch.zeros(edges.shape[:-1], dtype=torch.int64, device=edges.device)
    warm_symbols = math.ceil(window_bits / bits)
    if edges.shape[-1] < warm_symbols:
        repeats = math.ceil(warm_symbols / edges.shape[-1])
        warm = edges.repeat(*(1 for _ in edges.shape[:-1]), repeats)[..., -warm_symbols:]
    else:
        warm = edges[..., -warm_symbols:]
    for edge in warm.unbind(dim=-1):
        state = ((state << bits) | edge) & mask
    states = []
    for edge in edges.unbind(dim=-1):
        state = ((state << bits) | edge) & mask
        states.append(state)
    return torch.stack(states, dim=-1)


def _traceback(
    decisions: torch.Tensor,
    *,
    spec: TrellisGeometry,
    roll: int,
    final_state: torch.Tensor,
    stop_at_rotated_zero: bool,
    codebook: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    batch, length, states = decisions.shape
    if states != spec.states:
        raise ValueError("decision tensor has the wrong state count")
    batch_indices = torch.arange(batch, device=decisions.device)
    state = final_state.to(dtype=torch.long)
    reconstructed = None
    transitions = None
    if codebook is not None:
        reconstructed = torch.empty(
            (batch, length), dtype=codebook.dtype, device=decisions.device
        )
        transitions = torch.empty(
            (batch, length), dtype=torch.int32, device=decisions.device
        )
    for index in range(length - 1, -1, -1):
        rotated = (index + roll) % length
        branch = decisions[batch_indices, rotated, state].to(dtype=torch.long)
        previous = (branch << spec.predecessor_shift) | (state >> spec.bits)
        if reconstructed is not None and transitions is not None:
            transition = (previous << spec.bits) | state
            transitions[:, rotated] = transition.to(torch.int32)
            reconstructed[:, rotated] = codebook[transition]
        state = previous
        if stop_at_rotated_zero and rotated == 0:
            break
    return state, reconstructed, transitions


def _forward(
    source: torch.Tensor,
    codebook: torch.Tensor,
    spec: TrellisGeometry,
    *,
    reconstruction_scale: torch.Tensor | None,
    roll: int,
    initial_state: torch.Tensor | None,
    trace_start: int,
    cost_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, length = source.shape
    device = source.device
    output_states = torch.arange(spec.states, dtype=torch.long, device=device)
    predecessor_base = output_states >> spec.bits
    branches = torch.arange(spec.branches, dtype=torch.long, device=device)
    predecessors = predecessor_base.unsqueeze(0) | (
        branches.unsqueeze(1) << spec.predecessor_shift
    )
    values = codebook.reshape(spec.branches, spec.states).to(dtype=cost_dtype)
    decisions = torch.empty(
        (batch, length, spec.states), dtype=torch.uint8, device=device
    )

    costs: torch.Tensor | None
    if initial_state is None:
        costs = None
    else:
        costs = torch.full(
            (batch, spec.states),
            float("inf"),
            dtype=cost_dtype,
            device=device,
        )
        costs.scatter_(1, initial_state.to(torch.long).unsqueeze(1), 0.0)

    for index in range(length):
        rotated = (index + roll) % length
        target = source[:, rotated].to(dtype=cost_dtype).reshape(batch, 1, 1)
        candidate_values = values.unsqueeze(0)
        if reconstruction_scale is not None:
            scale = reconstruction_scale[:, rotated].to(dtype=cost_dtype)
            candidate_values = candidate_values * scale.reshape(batch, 1, 1)
        candidate = (target - candidate_values).square()
        if costs is not None:
            candidate = candidate + costs[:, predecessors]
        costs, choice = candidate.min(dim=1)
        if index >= trace_start:
            decisions[:, rotated] = choice.to(torch.uint8)
    assert costs is not None
    return costs, decisions


def _encode_chunk(
    source: torch.Tensor,
    codebook: torch.Tensor,
    spec: TrellisGeometry,
    *,
    reconstruction_scale: torch.Tensor | None,
    cost_dtype: torch.dtype,
    return_transitions: bool,
) -> TrellisEncoding:
    length = source.shape[1]
    roll = length // 2
    costs, decisions = _forward(
        source,
        codebook,
        spec,
        reconstruction_scale=reconstruction_scale,
        roll=roll,
        initial_state=None,
        trace_start=roll,
        cost_dtype=cost_dtype,
    )
    proposed_final = costs.argmin(dim=1)
    boundary, _, _ = _traceback(
        decisions,
        spec=spec,
        roll=roll,
        final_state=proposed_final,
        stop_at_rotated_zero=True,
        codebook=None,
    )
    del costs, decisions

    _, decisions = _forward(
        source,
        codebook,
        spec,
        reconstruction_scale=reconstruction_scale,
        roll=0,
        initial_state=boundary,
        trace_start=0,
        cost_dtype=cost_dtype,
    )
    recovered_boundary, reconstructed, transitions = _traceback(
        decisions,
        spec=spec,
        roll=0,
        final_state=boundary,
        stop_at_rotated_zero=False,
        codebook=codebook,
    )
    if not torch.equal(recovered_boundary, boundary):
        raise RuntimeError("tail-biting traceback did not close")
    assert reconstructed is not None and transitions is not None
    if reconstruction_scale is not None:
        reconstructed = reconstructed * reconstruction_scale.to(reconstructed.dtype)
    error = (source.float() - reconstructed.float()).square().sum(dim=1)
    return TrellisEncoding(
        reconstructed=reconstructed,
        squared_error=error,
        transition_indices=transitions if return_transitions else None,
    )


def encode_trellis(
    source: torch.Tensor,
    bits: int,
    window_bits: int,
    codebook: torch.Tensor,
    *,
    reconstruction_scale: torch.Tensor | None = None,
    cost_dtype: torch.dtype = torch.float32,
    max_trace_bytes: int = 256 << 20,
    max_batch_size: int | None = None,
    return_transitions: bool = True,
) -> TrellisEncoding:
    """Encode batches of scalar sequences with EXL's approximate tail-biting DP.

    ``reconstruction_scale`` optionally supplies a positive scale for every
    source coefficient.  The dynamic program then minimizes the exact error
    to ``scale * codebook[transition]``.  This is the reference primitive for
    fixed-rate paths with MX-style block reconstruction scales.
    """

    spec = geometry(bits, window_bits)
    if source.ndim != 2 or source.shape[1] <= 0 or source.shape[1] % 2:
        raise ValueError("source must have shape [batch, positive even sequence length]")
    if not source.is_floating_point():
        raise TypeError("source must be floating point")
    if source.device != codebook.device:
        raise ValueError("source and codebook must be on the same device")
    if codebook.ndim != 1 or codebook.numel() != spec.transitions:
        raise ValueError(f"codebook must contain exactly {spec.transitions} values")
    if cost_dtype not in (torch.float16, torch.float32):
        raise ValueError("cost_dtype must be float16 or float32")
    if max_trace_bytes <= 0:
        raise ValueError("max_trace_bytes must be positive")
    if not torch.isfinite(source).all() or not torch.isfinite(codebook).all():
        raise ValueError("source and codebook must be finite")
    if reconstruction_scale is not None:
        if reconstruction_scale.shape != source.shape:
            raise ValueError("reconstruction_scale must have the source shape")
        if not reconstruction_scale.is_floating_point():
            raise TypeError("reconstruction_scale must be floating point")
        if reconstruction_scale.device != source.device:
            raise ValueError("reconstruction_scale must share the source device")
        if not bool(torch.isfinite(reconstruction_scale).all()) or bool(
            torch.any(reconstruction_scale <= 0.0)
        ):
            raise ValueError("reconstruction_scale must be finite and positive")

    trace_bytes_per_item = source.shape[1] * spec.states
    chunk_size = max(1, max_trace_bytes // trace_bytes_per_item)
    if max_batch_size is not None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        chunk_size = min(chunk_size, max_batch_size)

    reconstructions = []
    errors = []
    all_transitions = []
    with torch.inference_mode():
        for start in range(0, source.shape[0], chunk_size):
            stop = min(start + chunk_size, source.shape[0])
            encoded = _encode_chunk(
                source[start:stop],
                codebook,
                spec,
                reconstruction_scale=(
                    None
                    if reconstruction_scale is None
                    else reconstruction_scale[start:stop]
                ),
                cost_dtype=cost_dtype,
                return_transitions=return_transitions,
            )
            reconstructions.append(encoded.reconstructed)
            errors.append(encoded.squared_error)
            if return_transitions:
                assert encoded.transition_indices is not None
                all_transitions.append(encoded.transition_indices)
    return TrellisEncoding(
        reconstructed=torch.cat(reconstructions, dim=0),
        squared_error=torch.cat(errors, dim=0),
        transition_indices=(
            torch.cat(all_transitions, dim=0) if return_transitions else None
        ),
    )
