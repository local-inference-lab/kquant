"""Hierarchical reconstruction tables for a fixed rolling trellis graph.

An L16/K3 transition table has 8,192 origin states and eight outgoing
branches.  A full byte-valued reconstruction table therefore occupies 64
KiB.  This module leaves the state-transition graph untouched and compresses
only the reconstruction function: every origin state selects one learned
eight-value palette row.

The intended decoder is::

    palette_id = state_to_palette[origin_state]
    packed_row = palettes[palette_id]       # eight E4M3 bytes / one u64
    value_code = byte(packed_row, branch)

The fitting objective is a weighted, missing-data-safe vector Lloyd update
over the eight outgoing labels.  It is deliberately independent of Viterbi;
the experiment driver alternates this update with fresh path assignment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from kquant.trellis_window import fp8_constrained_codebook, geometry


@dataclass(frozen=True)
class StatePaletteFit:
    codebook: torch.Tensor
    state_to_palette: torch.Tensor
    palettes: torch.Tensor
    objective: float
    iterations: int


@dataclass(frozen=True)
class StateSubpaletteFit:
    codebook: torch.Tensor
    state_to_palette: torch.Tensor
    palettes: torch.Tensor
    objective: float
    iterations: int
    branch_groups: int
    iterations_by_group: tuple[int, ...]


@dataclass(frozen=True)
class StateScaledPaletteFit:
    """A palette row plus a power-of-two scale selected by each state."""

    codebook: torch.Tensor
    state_to_palette: torch.Tensor
    state_to_exponent: torch.Tensor
    palettes: torch.Tensor
    exponent_values: torch.Tensor
    objective: float
    iterations: int


def expand_state_palette(
    state_to_palette: torch.Tensor,
    palettes: torch.Tensor,
    *,
    bits: int,
    window_bits: int,
) -> torch.Tensor:
    """Expand a state-paletted reconstruction function to transition order."""

    spec = geometry(bits, window_bits)
    if (
        state_to_palette.ndim != 1
        or state_to_palette.numel() != spec.states
        or state_to_palette.dtype == torch.bool
        or state_to_palette.is_floating_point()
    ):
        raise ValueError(f"state_to_palette must contain {spec.states} integers")
    if (
        palettes.ndim != 2
        or palettes.shape[1] != spec.branches
        or not palettes.is_floating_point()
        or palettes.shape[0] == 0
    ):
        raise ValueError(
            f"palettes must be a nonempty floating [classes, {spec.branches}] tensor"
        )
    if state_to_palette.device != palettes.device:
        raise ValueError("state_to_palette and palettes must share a device")
    classes = state_to_palette.long()
    if int(classes.min().item()) < 0 or int(classes.max().item()) >= palettes.shape[0]:
        raise ValueError("state_to_palette contains an out-of-range class")
    if not bool(torch.isfinite(palettes).all()):
        raise ValueError("palettes must be finite")
    return palettes.index_select(0, classes).reshape(-1).contiguous()


def expand_state_subpalettes(
    state_to_palette: torch.Tensor,
    palettes: torch.Tensor,
    *,
    bits: int,
    window_bits: int,
) -> torch.Tensor:
    """Expand independently clustered branch groups to transition order."""

    spec = geometry(bits, window_bits)
    if state_to_palette.ndim != 2 or state_to_palette.shape[1] != spec.states:
        raise ValueError("state_to_palette must have shape [branch_groups, states]")
    branch_groups = state_to_palette.shape[0]
    if branch_groups < 1 or spec.branches % branch_groups:
        raise ValueError("branch_groups must be a positive divisor of branches")
    branches_per_group = spec.branches // branch_groups
    if (
        palettes.ndim != 3
        or palettes.shape[0] != branch_groups
        or palettes.shape[2] != branches_per_group
        or palettes.shape[1] == 0
        or not palettes.is_floating_point()
    ):
        raise ValueError(
            "palettes must have shape [branch_groups, classes, branches_per_group]"
        )
    if state_to_palette.dtype == torch.bool or state_to_palette.is_floating_point():
        raise ValueError("state_to_palette must contain integers")
    if state_to_palette.device != palettes.device:
        raise ValueError("state_to_palette and palettes must share a device")
    assignments = state_to_palette.long()
    if int(assignments.min().item()) < 0 or int(assignments.max().item()) >= palettes.shape[1]:
        raise ValueError("state_to_palette contains an out-of-range class")
    if not bool(torch.isfinite(palettes).all()):
        raise ValueError("palettes must be finite")
    expanded = torch.empty(
        (spec.states, spec.branches), dtype=palettes.dtype, device=palettes.device
    )
    for group in range(branch_groups):
        start = group * branches_per_group
        stop = start + branches_per_group
        expanded[:, start:stop] = palettes[group].index_select(0, assignments[group])
    return expanded.flatten().contiguous()


def expand_state_scaled_palette(
    state_to_palette: torch.Tensor,
    state_to_exponent: torch.Tensor,
    palettes: torch.Tensor,
    exponent_values: torch.Tensor,
    *,
    bits: int,
    window_bits: int,
    reconstruction_dtype: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    """Expand state-selected palette rows and exact power-of-two scales."""

    spec = geometry(bits, window_bits)
    if (
        state_to_palette.ndim != 1
        or state_to_palette.numel() != spec.states
        or state_to_palette.dtype == torch.bool
        or state_to_palette.is_floating_point()
    ):
        raise ValueError(f"state_to_palette must contain {spec.states} integers")
    if (
        state_to_exponent.ndim != 1
        or state_to_exponent.numel() != spec.states
        or state_to_exponent.dtype == torch.bool
        or state_to_exponent.is_floating_point()
    ):
        raise ValueError(f"state_to_exponent must contain {spec.states} integers")
    if (
        palettes.ndim != 2
        or palettes.shape[1] != spec.branches
        or palettes.shape[0] == 0
        or not palettes.is_floating_point()
    ):
        raise ValueError(
            f"palettes must be a nonempty floating [classes, {spec.branches}] tensor"
        )
    if (
        exponent_values.ndim != 1
        or exponent_values.numel() == 0
        or not exponent_values.is_floating_point()
    ):
        raise ValueError("exponent_values must be a nonempty floating vector")
    devices = {
        state_to_palette.device,
        state_to_exponent.device,
        palettes.device,
        exponent_values.device,
    }
    if len(devices) != 1:
        raise ValueError("scaled-palette tensors must share a device")
    classes = state_to_palette.long()
    exponent_codes = state_to_exponent.long()
    if int(classes.min().item()) < 0 or int(classes.max().item()) >= palettes.shape[0]:
        raise ValueError("state_to_palette contains an out-of-range class")
    if (
        int(exponent_codes.min().item()) < 0
        or int(exponent_codes.max().item()) >= exponent_values.numel()
    ):
        raise ValueError("state_to_exponent contains an out-of-range code")
    if not bool(torch.isfinite(palettes).all()) or not bool(
        torch.isfinite(exponent_values).all()
    ):
        raise ValueError("scaled-palette values must be finite")
    rows = palettes.index_select(0, classes)
    scales = torch.exp2(exponent_values.index_select(0, exponent_codes)).unsqueeze(1)
    return _project(rows * scales, reconstruction_dtype).reshape(-1).contiguous()


def logical_state_palette_bytes(
    *, states: int, branches: int, classes: int, packed_classes: bool
) -> int:
    """Return logical bytes for a byte-valued palette representation.

    Packed maps use the information-minimal fixed-width class ID. Unpacked
    maps use the smallest ordinary 8-, 16-, or 32-bit integer container.
    """

    if min(states, branches, classes) <= 0:
        raise ValueError("state-palette dimensions must be positive")
    if packed_classes:
        class_bits = max(1, math.ceil(math.log2(classes)))
        map_bytes = (states * class_bits + 7) // 8
    else:
        class_bytes = 1 if classes <= 256 else 2 if classes <= 65536 else 4
        map_bytes = states * class_bytes
    return map_bytes + classes * branches


def logical_state_subpalette_bytes(
    *,
    states: int,
    branches: int,
    classes: int,
    branch_groups: int,
    packed_classes: bool,
) -> int:
    """Return bytes for independent state maps over branch sub-vectors."""

    if branch_groups <= 0 or branches % branch_groups:
        raise ValueError("branch_groups must be a positive divisor of branches")
    single_map_and_rows = logical_state_palette_bytes(
        states=states,
        branches=branches,
        classes=classes,
        packed_classes=packed_classes,
    )
    # Palette values remain ``classes * branches`` in total; only the state
    # map is replicated for each independent branch group.
    palette_bytes = classes * branches
    map_bytes = single_map_and_rows - palette_bytes
    return branch_groups * map_bytes + palette_bytes


def logical_state_scaled_palette_bytes(
    *,
    states: int,
    branches: int,
    classes: int,
    exponent_levels: int,
    packed_descriptor: bool,
) -> int:
    """Return bytes for a class ID and exponent code selected per state."""

    if min(states, branches, classes, exponent_levels) <= 0:
        raise ValueError("scaled-palette dimensions must be positive")
    descriptor_bits = max(1, math.ceil(math.log2(classes))) + max(
        1, math.ceil(math.log2(exponent_levels))
    )
    if packed_descriptor:
        map_bytes = (states * descriptor_bits + 7) // 8
    else:
        descriptor_bytes = 1 if descriptor_bits <= 8 else 2 if descriptor_bits <= 16 else 4
        map_bytes = states * descriptor_bytes
    return map_bytes + classes * branches


def _project(values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.float16:
        return values.half().float()
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return fp8_constrained_codebook(values, dtype=dtype)
    raise ValueError("reconstruction_dtype must be FP16, E4M3, or E5M2")


def _distances(
    targets: torch.Tensor, weights: torch.Tensor, palettes: torch.Tensor
) -> torch.Tensor:
    distances = torch.zeros(
        (targets.shape[0], palettes.shape[0]),
        dtype=torch.float32,
        device=targets.device,
    )
    for branch in range(targets.shape[1]):
        delta = targets[:, branch, None] - palettes[None, :, branch]
        distances.add_(weights[:, branch, None] * delta.square())
    return distances


def _farthest_palettes(
    targets: torch.Tensor,
    weights: torch.Tensor,
    classes: int,
    reconstruction_dtype: torch.dtype,
) -> torch.Tensor:
    """Deterministic weighted farthest-point initialization."""

    state_mass = weights.sum(dim=1)
    selected = torch.zeros(targets.shape[0], dtype=torch.bool, device=targets.device)
    first = int(state_mass.argmax().item())
    centers = [targets[first]]
    selected[first] = True
    closest = (
        weights * (targets - targets[first].unsqueeze(0)).square()
    ).sum(dim=1)
    for _ in range(1, classes):
        score = closest.masked_fill(selected, -1.0)
        index = int(score.argmax().item())
        if float(score[index].item()) <= 0.0:
            remaining = (~selected).nonzero().flatten()
            if remaining.numel() == 0:
                centers.append(centers[-1])
                continue
            index = int(remaining[state_mass[remaining].argmax()].item())
        center = targets[index]
        centers.append(center)
        selected[index] = True
        distance = (weights * (targets - center.unsqueeze(0)).square()).sum(dim=1)
        closest = torch.minimum(closest, distance)
    return _project(torch.stack(centers), reconstruction_dtype)


def fit_state_palette(
    transition_targets: torch.Tensor,
    transition_weights: torch.Tensor,
    *,
    bits: int,
    window_bits: int,
    classes: int,
    reconstruction_dtype: torch.dtype = torch.float8_e4m3fn,
    iterations: int = 32,
    initial_state_to_palette: torch.Tensor | None = None,
    initial_palettes: torch.Tensor | None = None,
) -> StatePaletteFit:
    """Fit shared eight-branch palettes to weighted transition targets.

    ``transition_targets`` and ``transition_weights`` are in ordinary
    transition-index order.  A zero weight removes that transition from the
    objective.  Callers can express shrinkage by adding ridge mass before
    invoking this function.
    """

    spec = geometry(bits, window_bits)
    if (
        transition_targets.ndim != 1
        or transition_targets.numel() != spec.transitions
        or not transition_targets.is_floating_point()
    ):
        raise ValueError(
            f"transition_targets must contain {spec.transitions} floating values"
        )
    if (
        transition_weights.shape != transition_targets.shape
        or not transition_weights.is_floating_point()
    ):
        raise ValueError("transition_weights must be floating and match targets")
    if transition_targets.device != transition_weights.device:
        raise ValueError("transition targets and weights must share a device")
    if not bool(torch.isfinite(transition_targets).all()):
        raise ValueError("transition targets must be finite")
    if (
        not bool(torch.isfinite(transition_weights).all())
        or bool(torch.any(transition_weights < 0.0))
        or not bool(torch.any(transition_weights > 0.0))
    ):
        raise ValueError("transition weights must be finite, nonnegative, and nonzero")
    if isinstance(classes, bool) or not isinstance(classes, int) or not 1 <= classes <= spec.states:
        raise ValueError(f"classes must be in 1..{spec.states}")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be positive")

    targets = transition_targets.float().reshape(spec.states, spec.branches)
    weights = transition_weights.float().reshape(spec.states, spec.branches)
    supplied = initial_state_to_palette is not None or initial_palettes is not None
    if supplied:
        if initial_state_to_palette is None or initial_palettes is None:
            raise ValueError("both initial state assignments and palettes are required")
        if initial_palettes.shape != (classes, spec.branches):
            raise ValueError("initial palettes have the wrong shape")
        palettes = _project(initial_palettes.float(), reconstruction_dtype)
        state_to_palette = initial_state_to_palette.long().clone()
        # Reuse the common validator and establish exact expanded order.
        expand_state_palette(
            state_to_palette,
            palettes,
            bits=bits,
            window_bits=window_bits,
        )
    else:
        palettes = _farthest_palettes(
            targets, weights, classes, reconstruction_dtype
        )
        state_to_palette = _distances(targets, weights, palettes).argmin(dim=1)

    completed = 0
    for completed in range(1, iterations + 1):
        previous_classes = state_to_palette
        previous_palettes = palettes
        total_weight = torch.zeros_like(palettes, dtype=torch.float32)
        weighted_sum = torch.zeros_like(palettes, dtype=torch.float32)
        for branch in range(spec.branches):
            total_weight[:, branch].scatter_add_(
                0, state_to_palette, weights[:, branch]
            )
            weighted_sum[:, branch].scatter_add_(
                0, state_to_palette, weights[:, branch] * targets[:, branch]
            )
        updated = torch.where(
            total_weight > 0.0,
            weighted_sum / total_weight.clamp_min(torch.finfo(torch.float32).tiny),
            palettes,
        )
        palettes = _project(updated, reconstruction_dtype)
        state_to_palette = _distances(targets, weights, palettes).argmin(dim=1)
        if torch.equal(state_to_palette, previous_classes) and torch.equal(
            palettes, previous_palettes
        ):
            break

    distances = _distances(targets, weights, palettes)
    objective = distances.gather(1, state_to_palette[:, None]).sum()
    codebook = expand_state_palette(
        state_to_palette,
        palettes,
        bits=bits,
        window_bits=window_bits,
    )
    return StatePaletteFit(
        codebook=codebook,
        state_to_palette=state_to_palette,
        palettes=palettes,
        objective=float(objective.item()),
        iterations=completed,
    )


def _scaled_palette_assignment(
    targets: torch.Tensor,
    weights: torch.Tensor,
    palettes: torch.Tensor,
    exponent_values: torch.Tensor,
    reconstruction_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Choose the best palette class and exponent without a giant 3-D tensor."""

    best_distance = torch.full(
        (targets.shape[0],),
        float("inf"),
        dtype=torch.float32,
        device=targets.device,
    )
    best_class = torch.zeros_like(best_distance, dtype=torch.long)
    best_exponent = torch.zeros_like(best_class)
    for exponent_code, exponent in enumerate(exponent_values):
        scaled = _project(
            palettes * torch.exp2(exponent), reconstruction_dtype
        )
        distances = _distances(targets, weights, scaled)
        local_distance, local_class = distances.min(dim=1)
        improved = local_distance < best_distance
        best_distance = torch.where(improved, local_distance, best_distance)
        best_class = torch.where(improved, local_class, best_class)
        best_exponent = torch.where(
            improved,
            torch.full_like(best_exponent, exponent_code),
            best_exponent,
        )
    return best_class, best_exponent, best_distance


def fit_state_scaled_palette(
    transition_targets: torch.Tensor,
    transition_weights: torch.Tensor,
    *,
    bits: int,
    window_bits: int,
    classes: int,
    exponent_values: tuple[int, ...],
    reconstruction_dtype: torch.dtype = torch.float8_e4m3fn,
    iterations: int = 32,
    initial_state_to_palette: torch.Tensor | None = None,
    initial_state_to_exponent: torch.Tensor | None = None,
    initial_palettes: torch.Tensor | None = None,
) -> StateScaledPaletteFit:
    """Fit E4M3 palette shapes with one power-of-two scale per trellis state.

    The state graph and branch rate are unchanged.  The scale is part of the
    global reconstruction function, not per-weight checkpoint metadata.
    """

    spec = geometry(bits, window_bits)
    if (
        transition_targets.ndim != 1
        or transition_targets.numel() != spec.transitions
        or not transition_targets.is_floating_point()
    ):
        raise ValueError(
            f"transition_targets must contain {spec.transitions} floating values"
        )
    if (
        transition_weights.shape != transition_targets.shape
        or not transition_weights.is_floating_point()
    ):
        raise ValueError("transition_weights must be floating and match targets")
    if transition_targets.device != transition_weights.device:
        raise ValueError("transition targets and weights must share a device")
    if not bool(torch.isfinite(transition_targets).all()):
        raise ValueError("transition targets must be finite")
    if (
        not bool(torch.isfinite(transition_weights).all())
        or bool(torch.any(transition_weights < 0.0))
        or not bool(torch.any(transition_weights > 0.0))
    ):
        raise ValueError("transition weights must be finite, nonnegative, and nonzero")
    if isinstance(classes, bool) or not isinstance(classes, int) or not 1 <= classes <= spec.states:
        raise ValueError(f"classes must be in 1..{spec.states}")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be positive")
    if (
        not exponent_values
        or len(set(exponent_values)) != len(exponent_values)
        or 0 not in exponent_values
        or any(isinstance(value, bool) or not isinstance(value, int) for value in exponent_values)
    ):
        raise ValueError("exponent_values must be unique integers including zero")

    targets = transition_targets.float().reshape(spec.states, spec.branches)
    weights = transition_weights.float().reshape(spec.states, spec.branches)
    exponents = torch.tensor(
        exponent_values, dtype=torch.float32, device=targets.device
    )
    supplied = any(
        value is not None
        for value in (
            initial_state_to_palette,
            initial_state_to_exponent,
            initial_palettes,
        )
    )
    if supplied:
        if initial_state_to_palette is None or initial_palettes is None:
            raise ValueError("initial state assignments and palettes are required")
        if initial_palettes.shape != (classes, spec.branches):
            raise ValueError("initial palettes have the wrong shape")
        palettes = _project(initial_palettes.float(), reconstruction_dtype)
        state_to_palette = initial_state_to_palette.long().clone()
        if initial_state_to_exponent is None:
            zero_code = exponent_values.index(0)
            state_to_exponent = torch.full_like(state_to_palette, zero_code)
        else:
            state_to_exponent = initial_state_to_exponent.long().clone()
        expand_state_scaled_palette(
            state_to_palette,
            state_to_exponent,
            palettes,
            exponents,
            bits=bits,
            window_bits=window_bits,
            reconstruction_dtype=reconstruction_dtype,
        )
    else:
        palettes = _farthest_palettes(
            targets, weights, classes, reconstruction_dtype
        )
        state_to_palette, state_to_exponent, _ = _scaled_palette_assignment(
            targets, weights, palettes, exponents, reconstruction_dtype
        )

    completed = 0
    for completed in range(1, iterations + 1):
        previous_classes = state_to_palette
        previous_exponents = state_to_exponent
        previous_palettes = palettes
        scales = torch.exp2(exponents.index_select(0, state_to_exponent))
        total_weight = torch.zeros_like(palettes, dtype=torch.float32)
        weighted_sum = torch.zeros_like(palettes, dtype=torch.float32)
        for branch in range(spec.branches):
            branch_weights = weights[:, branch]
            total_weight[:, branch].scatter_add_(
                0, state_to_palette, branch_weights * scales.square()
            )
            weighted_sum[:, branch].scatter_add_(
                0,
                state_to_palette,
                branch_weights * scales * targets[:, branch],
            )
        updated = torch.where(
            total_weight > 0.0,
            weighted_sum / total_weight.clamp_min(torch.finfo(torch.float32).tiny),
            palettes,
        )
        palettes = _project(updated, reconstruction_dtype)
        state_to_palette, state_to_exponent, best_distance = (
            _scaled_palette_assignment(
                targets, weights, palettes, exponents, reconstruction_dtype
            )
        )
        if (
            torch.equal(state_to_palette, previous_classes)
            and torch.equal(state_to_exponent, previous_exponents)
            and torch.equal(palettes, previous_palettes)
        ):
            break

    _, _, best_distance = _scaled_palette_assignment(
        targets, weights, palettes, exponents, reconstruction_dtype
    )
    codebook = expand_state_scaled_palette(
        state_to_palette,
        state_to_exponent,
        palettes,
        exponents,
        bits=bits,
        window_bits=window_bits,
        reconstruction_dtype=reconstruction_dtype,
    )
    return StateScaledPaletteFit(
        codebook=codebook,
        state_to_palette=state_to_palette,
        state_to_exponent=state_to_exponent,
        palettes=palettes,
        exponent_values=exponents,
        objective=float(best_distance.sum().item()),
        iterations=completed,
    )


def fit_state_subpalettes(
    transition_targets: torch.Tensor,
    transition_weights: torch.Tensor,
    *,
    bits: int,
    window_bits: int,
    classes: int,
    branch_groups: int,
    reconstruction_dtype: torch.dtype = torch.float8_e4m3fn,
    iterations: int = 32,
    initial_state_to_palette: torch.Tensor | None = None,
    initial_palettes: torch.Tensor | None = None,
) -> StateSubpaletteFit:
    """Fit independent palette assignments for contiguous branch groups.

    With K3 and ``branch_groups=2``, branches 0--3 and 4--7 obtain separate
    state-to-palette maps. Only the map selected by the current branch is read
    during decoding, so the lookup depth is unchanged while the number of
    composite eight-branch rows grows combinatorially.
    """

    spec = geometry(bits, window_bits)
    if (
        transition_targets.ndim != 1
        or transition_targets.numel() != spec.transitions
        or not transition_targets.is_floating_point()
    ):
        raise ValueError(
            f"transition_targets must contain {spec.transitions} floating values"
        )
    if (
        transition_weights.shape != transition_targets.shape
        or not transition_weights.is_floating_point()
    ):
        raise ValueError("transition_weights must be floating and match targets")
    if transition_targets.device != transition_weights.device:
        raise ValueError("transition targets and weights must share a device")
    if not bool(torch.isfinite(transition_targets).all()):
        raise ValueError("transition targets must be finite")
    if (
        not bool(torch.isfinite(transition_weights).all())
        or bool(torch.any(transition_weights < 0.0))
        or not bool(torch.any(transition_weights > 0.0))
    ):
        raise ValueError("transition weights must be finite, nonnegative, and nonzero")
    if isinstance(classes, bool) or not isinstance(classes, int) or not 1 <= classes <= spec.states:
        raise ValueError(f"classes must be in 1..{spec.states}")
    if (
        isinstance(branch_groups, bool)
        or not isinstance(branch_groups, int)
        or branch_groups < 2
        or spec.branches % branch_groups
    ):
        raise ValueError(
            f"branch_groups must be a divisor of {spec.branches} greater than one"
        )
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be positive")

    branches_per_group = spec.branches // branch_groups
    targets = transition_targets.float().reshape(spec.states, spec.branches)
    weights = transition_weights.float().reshape(spec.states, spec.branches)
    supplied = initial_state_to_palette is not None or initial_palettes is not None
    if supplied:
        if initial_state_to_palette is None or initial_palettes is None:
            raise ValueError("both initial state assignments and palettes are required")
        if initial_state_to_palette.shape != (branch_groups, spec.states):
            raise ValueError("initial state assignments have the wrong shape")
        if initial_palettes.shape != (branch_groups, classes, branches_per_group):
            raise ValueError("initial palettes have the wrong shape")
        expand_state_subpalettes(
            initial_state_to_palette,
            initial_palettes,
            bits=bits,
            window_bits=window_bits,
        )

    assignments_by_group = []
    palettes_by_group = []
    iterations_by_group = []
    objective = 0.0
    for group in range(branch_groups):
        start = group * branches_per_group
        stop = start + branches_per_group
        local_targets = targets[:, start:stop]
        local_weights = weights[:, start:stop]
        if supplied:
            assert initial_state_to_palette is not None and initial_palettes is not None
            assignments = initial_state_to_palette[group].long().clone()
            palettes = _project(initial_palettes[group].float(), reconstruction_dtype)
        else:
            palettes = _farthest_palettes(
                local_targets, local_weights, classes, reconstruction_dtype
            )
            assignments = _distances(local_targets, local_weights, palettes).argmin(dim=1)

        completed = 0
        for completed in range(1, iterations + 1):
            previous_assignments = assignments
            previous_palettes = palettes
            total_weight = torch.zeros_like(palettes, dtype=torch.float32)
            weighted_sum = torch.zeros_like(palettes, dtype=torch.float32)
            for branch in range(branches_per_group):
                total_weight[:, branch].scatter_add_(
                    0, assignments, local_weights[:, branch]
                )
                weighted_sum[:, branch].scatter_add_(
                    0,
                    assignments,
                    local_weights[:, branch] * local_targets[:, branch],
                )
            updated = torch.where(
                total_weight > 0.0,
                weighted_sum / total_weight.clamp_min(torch.finfo(torch.float32).tiny),
                palettes,
            )
            palettes = _project(updated, reconstruction_dtype)
            assignments = _distances(
                local_targets, local_weights, palettes
            ).argmin(dim=1)
            if torch.equal(assignments, previous_assignments) and torch.equal(
                palettes, previous_palettes
            ):
                break
        distances = _distances(local_targets, local_weights, palettes)
        objective += float(distances.gather(1, assignments[:, None]).sum().item())
        assignments_by_group.append(assignments)
        palettes_by_group.append(palettes)
        iterations_by_group.append(completed)

    state_to_palette = torch.stack(assignments_by_group)
    palettes = torch.stack(palettes_by_group)
    return StateSubpaletteFit(
        codebook=expand_state_subpalettes(
            state_to_palette,
            palettes,
            bits=bits,
            window_bits=window_bits,
        ),
        state_to_palette=state_to_palette,
        palettes=palettes,
        objective=objective,
        iterations=max(iterations_by_group),
        branch_groups=branch_groups,
        iterations_by_group=tuple(iterations_by_group),
    )
