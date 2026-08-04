"""Small codec-inspired primitives for offline K3 weight experiments.

These are intentionally format-agnostic research tools, not production
quantizers.  They let us reject or refine transfers from texture and video
coding before committing to a kernel or checkpoint format.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CodecResult:
    reconstruction: torch.Tensor
    bits_per_weight: float


@dataclass(frozen=True)
class ResidualVQModel:
    codebooks: tuple[torch.Tensor, ...]
    vector_dimension: int
    index_bits: int

    @property
    def bits_per_weight(self) -> float:
        return len(self.codebooks) * self.index_bits / self.vector_dimension


@dataclass(frozen=True)
class MatrixGains:
    row: torch.Tensor
    column: torch.Tensor
    stored_bits: int

    def apply(self, normalized: torch.Tensor) -> torch.Tensor:
        return normalized * self.row[:, None] * self.column[None, :]


def normalized_mse(reference: torch.Tensor, reconstruction: torch.Tensor) -> float:
    """Energy-normalized squared error, accumulated in FP64."""

    if reference.shape != reconstruction.shape:
        raise ValueError(
            f"shape mismatch: {tuple(reference.shape)} vs {tuple(reconstruction.shape)}"
        )
    ref = reference.double()
    err = ref - reconstruction.double()
    denominator = ref.square().sum()
    if denominator <= 0:
        return 0.0 if not torch.count_nonzero(err) else math.inf
    return float(err.square().sum() / denominator)


def packed_mxfp4_zero_statistics(packed: torch.Tensor) -> dict[str, float | int]:
    """Count exact E2M1 zero codes and all-zero logical input columns."""

    if packed.ndim != 2 or packed.dtype != torch.uint8:
        raise ValueError("packed MXFP4 weights must be a uint8 matrix")
    # Both +0 (code 0) and -0 (code 8) are zero. The low three bits of either
    # nibble are therefore sufficient and avoid unpacking the tensor.
    low_zero = (packed & 0x07) == 0
    high_zero = (packed & 0x70) == 0
    zero_columns = int(low_zero.all(dim=0).sum() + high_zero.all(dim=0).sum())
    zero_elements = int(low_zero.sum() + high_zero.sum())
    logical_elements = packed.numel() * 2
    return {
        "zero_elements": zero_elements,
        "zero_element_fraction": zero_elements / logical_elements,
        "zero_columns": zero_columns,
        "logical_columns": packed.shape[1] * 2,
    }


def normalize_matrix_gains(
    matrix: torch.Tensor,
    *,
    mode: str,
    vector_orientation: str,
    iterations: int = 6,
) -> tuple[torch.Tensor, MatrixGains]:
    """Factor FP16 channel gains from a matrix before dictionary coding.

    ``aligned`` stores the gain shared by every component of a contiguous
    vector (a column gain for column-oriented vectors and a row gain for
    row-oriented vectors). ``cross`` stores gains across vector components,
    and ``separable`` alternately equilibrates both axes.
    """

    if matrix.ndim != 2 or not torch.is_floating_point(matrix):
        raise ValueError("gain normalization expects a floating-point matrix")
    if mode not in {"none", "aligned", "cross", "separable"}:
        raise ValueError(f"unsupported gain mode: {mode}")
    if vector_orientation not in {"row", "column"}:
        raise ValueError(f"unsupported vector orientation: {vector_orientation}")
    if iterations <= 0:
        raise ValueError("gain normalization iterations must be positive")

    source = matrix.float()
    rows, columns = source.shape
    row = torch.ones(rows, dtype=torch.float32, device=source.device)
    column = torch.ones(columns, dtype=torch.float32, device=source.device)

    def safe_rms(values: torch.Tensor, dimension: int) -> torch.Tensor:
        rms = values.square().mean(dim=dimension).sqrt()
        return torch.where(rms > 0, rms, torch.ones_like(rms))

    aligned_axis = "row" if vector_orientation == "row" else "column"
    selected_axes = {
        "none": (),
        "aligned": (aligned_axis,),
        "cross": ({"row": "column", "column": "row"}[aligned_axis],),
        "separable": ("row", "column"),
    }[mode]
    normalized = source.clone()
    rounds = iterations if mode == "separable" else 1
    for _ in range(rounds):
        if "row" in selected_axes:
            update = safe_rms(normalized, 1)
            row *= update
            normalized /= update[:, None]
        if "column" in selected_axes:
            update = safe_rms(normalized, 0)
            column *= update
            normalized /= update[None, :]

    # These are decoder-side tensors, so include storage rounding in the
    # experiment rather than scoring ideal FP32 side information.
    row = row.half().float()
    column = column.half().float()
    normalized = source / row[:, None] / column[None, :]
    stored_bits = 16 * (
        (rows if "row" in selected_axes else 0)
        + (columns if "column" in selected_axes else 0)
    )
    return normalized, MatrixGains(row=row, column=column, stored_bits=stored_bits)


def sample_tile_coordinates(
    shape: tuple[int, int],
    *,
    tile_size: int,
    count: int,
    seed: int,
) -> torch.Tensor:
    """Choose complete non-overlapping tile coordinates without replacement."""

    if len(shape) != 2 or tile_size <= 0 or count <= 0:
        raise ValueError("shape must be 2-D and tile_size/count must be positive")
    tile_rows = shape[0] // tile_size
    tile_cols = shape[1] // tile_size
    total = tile_rows * tile_cols
    if total == 0:
        raise ValueError(f"shape {shape} has no complete {tile_size}x{tile_size} tile")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    flat = torch.randperm(total, generator=generator)[: min(count, total)]
    return torch.stack((flat // tile_cols, flat % tile_cols), dim=1)


def gather_tiles(
    matrix: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    tile_size: int,
) -> torch.Tensor:
    if matrix.ndim != 2 or coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("expected a matrix and [num_tiles, 2] coordinates")
    tiles = []
    for tile_row, tile_col in coordinates.tolist():
        row = int(tile_row) * tile_size
        col = int(tile_col) * tile_size
        tile = matrix[row : row + tile_size, col : col + tile_size]
        if tuple(tile.shape) != (tile_size, tile_size):
            raise ValueError(f"coordinate {(tile_row, tile_col)} is out of range")
        tiles.append(tile)
    return torch.stack(tiles)


def _quantize_signed(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    if not 2 <= bits <= 8:
        raise ValueError("signed factor quantization supports 2..8 bits")
    limit = (1 << (bits - 1)) - 1
    maximum = tensor.abs().amax()
    if maximum == 0:
        return torch.zeros_like(tensor)
    scale = maximum / limit
    return (tensor / scale).round().clamp(-limit, limit) * scale


def scalar_affine_3bit(tiles: torch.Tensor) -> CodecResult:
    """Per-tile min/max scalar quantization with eight reconstruction levels."""

    if tiles.ndim != 3:
        raise ValueError("tiles must have shape [batch, rows, columns]")
    minimum = tiles.amin(dim=(-2, -1), keepdim=True)
    maximum = tiles.amax(dim=(-2, -1), keepdim=True)
    step = (maximum - minimum) / 7.0
    safe_step = torch.where(step > 0, step, torch.ones_like(step))
    indexes = ((tiles - minimum) / safe_step).round().clamp(0, 7)
    reconstruction = minimum + indexes * step
    # Two FP16 endpoints plus three bits per scalar.
    values = tiles.shape[-2] * tiles.shape[-1]
    rate = 3.0 + 32.0 / values
    return CodecResult(reconstruction, rate)


def _endpoint_axis(vectors: torch.Tensor, levels: int = 8) -> torch.Tensor:
    """Fit one quantized endpoint line to [texels, channels]."""

    mean = vectors.mean(dim=0, keepdim=True)
    centered = vectors - mean
    if not torch.count_nonzero(centered):
        return vectors.clone()
    _, _, vh = torch.linalg.svd(centered.float(), full_matrices=False)
    direction = vh[0].to(vectors.dtype)
    coefficient = centered @ direction
    low, high = coefficient.amin(), coefficient.amax()
    if high == low:
        return mean.expand_as(vectors).clone()
    index = ((coefficient - low) * (levels - 1) / (high - low)).round()
    alpha = (index / (levels - 1)).unsqueeze(1)
    endpoints = torch.stack((mean[0] + low * direction, mean[0] + high * direction))
    endpoints = _quantize_signed(endpoints, 8)
    return endpoints[0] + alpha * (endpoints[1] - endpoints[0])


def _ternary_residual(residual: torch.Tensor, iterations: int = 8) -> torch.Tensor:
    """Lloyd refinement for a symmetric {-scale, 0, +scale} residual."""

    magnitude = residual.abs()
    scale = magnitude.mean()
    if scale == 0:
        return torch.zeros_like(residual)
    for _ in range(iterations):
        active = magnitude >= scale / 2
        if not torch.count_nonzero(active):
            return torch.zeros_like(residual)
        scale = magnitude[active].mean()
    return torch.where(magnitude >= scale / 2, residual.sign() * scale, 0.0)


def endpoint_ternary(tiles: torch.Tensor) -> CodecResult:
    """ASTC-like endpoint axis plus a radix-3 predictive residual.

    Each 16-vector endpoint component is int8 with one FP16 block scale; each
    texel has a 3-bit interpolation selector.  The remaining residual uses a
    ternary alphabet and one FP16 scale.  Row- and column-vector orientations
    are both tried and one mode bit selects the winner.
    """

    if tiles.ndim != 3 or tiles.shape[-1] != tiles.shape[-2]:
        raise ValueError("endpoint coding expects square [batch, tile, tile] data")
    tile_size = int(tiles.shape[-1])
    if tile_size != 16:
        raise ValueError("the current rate model is defined for 16x16 tiles")
    reconstructions = []
    for tile in tiles:
        candidates = []
        for transpose in (False, True):
            vectors = tile.T if transpose else tile
            prediction = _endpoint_axis(vectors)
            reconstruction = prediction + _ternary_residual(vectors - prediction)
            candidates.append(reconstruction.T if transpose else reconstruction)
        errors = torch.stack([(tile - value).square().sum() for value in candidates])
        reconstructions.append(candidates[int(errors.argmin())])
    reconstruction = torch.stack(reconstructions)
    values = tile_size * tile_size
    endpoint_bits = 2 * tile_size * 8
    rate = (
        endpoint_bits
        + 16  # endpoint scale
        + tile_size * 3  # interpolation selectors
        + values * math.log2(3)  # radix-3 residual digits
        + 16  # residual scale
        + 1  # orientation mode
    ) / values
    return CodecResult(reconstruction, rate)


def endpoint_two_subset(tiles: torch.Tensor, iterations: int = 6) -> CodecResult:
    """BC7-like two-subset endpoint coding with an explicit 16-bit mask."""

    if tiles.ndim != 3 or tuple(tiles.shape[-2:]) != (16, 16):
        raise ValueError("two-subset coding expects [batch, 16, 16] tiles")
    reconstructed_tiles = []
    for tile in tiles:
        orientation_candidates = []
        for transpose in (False, True):
            vectors = (tile.T if transpose else tile).float()
            _, _, vh = torch.linalg.svd(
                vectors - vectors.mean(dim=0, keepdim=True), full_matrices=False
            )
            projection = vectors @ vh[0]
            labels = projection >= projection.median()
            for _ in range(iterations):
                centers = []
                for label in (False, True):
                    members = vectors[labels == label]
                    centers.append(
                        members.mean(dim=0) if members.numel() else vectors.mean(dim=0)
                    )
                distance = torch.stack(
                    [(vectors - center).square().sum(dim=1) for center in centers],
                    dim=1,
                )
                labels = distance[:, 1] < distance[:, 0]
            prediction = torch.empty_like(vectors)
            for label in (False, True):
                selected = labels == label
                if torch.count_nonzero(selected) < 2:
                    prediction[selected] = vectors[selected]
                else:
                    prediction[selected] = _endpoint_axis(vectors[selected])
            orientation_candidates.append(prediction.T if transpose else prediction)
        errors = torch.stack(
            [(tile - value).square().sum() for value in orientation_candidates]
        )
        reconstructed_tiles.append(orientation_candidates[int(errors.argmin())])
    reconstruction = torch.stack(reconstructed_tiles).to(tiles.dtype)
    values = 16 * 16
    rate = (
        4 * 16 * 8  # four int8 endpoints
        + 16  # shared endpoint scale
        + 16 * 3  # selectors
        + 16  # arbitrary subset mask
        + 1  # orientation mode
    ) / values
    return CodecResult(reconstruction, rate)


def quantized_low_rank(
    tiles: torch.Tensor, *, rank: int, factor_bits: int
) -> CodecResult:
    """Tile-local SVD with independently scaled signed integer factors."""

    if tiles.ndim != 3 or tiles.shape[-1] != tiles.shape[-2]:
        raise ValueError("low-rank coding expects square tiles")
    tile_size = int(tiles.shape[-1])
    if not 1 <= rank <= tile_size:
        raise ValueError("rank is outside the tile extent")
    u, singular, vh = torch.linalg.svd(tiles.float(), full_matrices=False)
    left = u[:, :, :rank] * singular[:, None, :rank]
    right = vh[:, :rank, :]
    reconstructed = []
    for left_tile, right_tile in zip(left, right):
        left_q = _quantize_signed(left_tile, factor_bits)
        right_q = _quantize_signed(right_tile, factor_bits)
        reconstructed.append(left_q @ right_q)
    reconstruction = torch.stack(reconstructed).to(tiles.dtype)
    values = tile_size * tile_size
    rate = (2 * tile_size * rank * factor_bits + 32) / values
    return CodecResult(reconstruction, rate)


def affine_prediction(target: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Fit target = scale * reference + offset independently per leading row."""

    if target.shape != reference.shape or target.ndim < 2:
        raise ValueError("target and reference must have equal rank >= 2 and shape")
    flat_target = target.reshape(*target.shape[:-2], -1).float()
    flat_reference = reference.reshape(*reference.shape[:-2], -1).float()
    target_mean = flat_target.mean(dim=-1, keepdim=True)
    reference_mean = flat_reference.mean(dim=-1, keepdim=True)
    centered_reference = flat_reference - reference_mean
    denominator = centered_reference.square().sum(dim=-1, keepdim=True)
    numerator = ((flat_target - target_mean) * centered_reference).sum(
        dim=-1, keepdim=True
    )
    scale = torch.where(denominator > 0, numerator / denominator, 0.0)
    predicted = target_mean + scale * centered_reference
    return predicted.reshape_as(target).to(target.dtype)


def scaled_residual_ratio(target: torch.Tensor, reference: torch.Tensor) -> float:
    """Optimistic per-vector scaled-reference residual energy ratio."""

    if target.shape != reference.shape or target.ndim != 2:
        raise ValueError("target/reference must be equal [vectors, dimensions] tensors")
    target_f = target.float()
    reference_f = reference.float()
    denominator = reference_f.square().sum(dim=1, keepdim=True)
    scale = torch.where(
        denominator > 0,
        (target_f * reference_f).sum(dim=1, keepdim=True) / denominator,
        0.0,
    )
    return normalized_mse(target_f, reference_f * scale)


def nearest_neuron_prediction(
    target_gate: torch.Tensor,
    reference_gate: torch.Tensor,
    target_signature: torch.Tensor,
    reference_signature: torch.Tensor,
) -> dict[str, float | torch.Tensor]:
    """Match neurons by gate-row cosine and score an optimistic prediction.

    Matching is deliberately non-bijective.  It is a lower bound on residual
    error: if this cannot find redundancy, a practical permutation coder will
    not recover it either.
    """

    if target_gate.ndim != 2 or reference_gate.ndim != 2:
        raise ValueError("gate signatures must be matrices")
    if target_gate.shape[1] != reference_gate.shape[1]:
        raise ValueError("gate signature dimensions differ")
    if target_signature.shape[0] != target_gate.shape[0]:
        raise ValueError("target signature row count differs")
    if reference_signature.shape[0] != reference_gate.shape[0]:
        raise ValueError("reference signature row count differs")
    target_norm = torch.nn.functional.normalize(target_gate.float(), dim=1)
    reference_norm = torch.nn.functional.normalize(reference_gate.float(), dim=1)
    similarity = target_norm @ reference_norm.T
    best_similarity, best_index = similarity.max(dim=1)
    reverse_index = similarity.max(dim=0).indices
    target_index = torch.arange(best_index.numel(), device=best_index.device)
    mutual = reverse_index[best_index] == target_index
    matched = reference_signature[best_index]
    return {
        "best_index": best_index.cpu(),
        "cosine_mean": float(best_similarity.mean()),
        "cosine_p50": float(best_similarity.quantile(0.5)),
        "cosine_p95": float(best_similarity.quantile(0.95)),
        "mutual_fraction": float(mutual.float().mean()),
        "scaled_residual_ratio": scaled_residual_ratio(target_signature, matched),
    }


def pvq_codebook_size(dimension: int, pulses: int) -> int:
    """Number of signed integer vectors with L1 norm exactly ``pulses``."""

    if dimension <= 0 or pulses <= 0:
        raise ValueError("PVQ dimension and pulse count must be positive")
    return sum(
        math.comb(dimension, nonzero)
        * math.comb(pulses - 1, nonzero - 1)
        * (1 << nonzero)
        for nonzero in range(1, min(dimension, pulses) + 1)
    )


def pvq_gain_shape(
    vectors: torch.Tensor,
    *,
    pulses: int,
    power: float = 1.0,
    gain_bits: int = 8,
) -> CodecResult:
    """Radial/power-projected pyramid vector quantization.

    Pulse allocation uses largest remainders after projecting magnitudes onto
    the L1 sphere.  A matrix-wide logarithmic gain grid is used to model the
    cheap shared gain side information available for offline weights.
    """

    if vectors.ndim != 2 or vectors.shape[1] <= 1:
        raise ValueError("PVQ input must be [vectors, dimension]")
    if pulses <= 0 or power <= 0 or not 2 <= gain_bits <= 16:
        raise ValueError("invalid PVQ pulse, power, or gain-bit setting")
    source = vectors.float()
    magnitude = source.abs().pow(power)
    total = magnitude.sum(dim=1, keepdim=True)
    projected = torch.where(total > 0, magnitude * pulses / total, 0.0)
    allocation = projected.floor().to(torch.int64)
    missing = pulses - allocation.sum(dim=1)
    fraction = projected - allocation
    # K is small in the regimes of interest.  Selecting all coordinates once
    # avoids a Python loop over vectors while preserving exact L1 pulse count.
    order = fraction.argsort(dim=1, descending=True)
    ranks = torch.empty_like(order)
    rank_values = torch.arange(order.shape[1], device=order.device).expand_as(order)
    ranks.scatter_(1, order, rank_values)
    allocation += (ranks < missing[:, None]).to(allocation.dtype)
    pulse_vector = allocation.to(source.dtype) * source.sign()
    direction = torch.nn.functional.normalize(pulse_vector, dim=1)

    gain = source.norm(dim=1)
    positive = gain > 0
    quantized_gain = torch.zeros_like(gain)
    if torch.count_nonzero(positive):
        log_gain = gain[positive].log2()
        low, high = log_gain.amin(), log_gain.amax()
        if high == low:
            quantized_gain[positive] = gain[positive]
        else:
            levels = (1 << gain_bits) - 1
            index = ((log_gain - low) * levels / (high - low)).round()
            quantized_gain[positive] = torch.exp2(low + index * (high - low) / levels)
    reconstruction = direction * quantized_gain[:, None]
    shape_bits = math.ceil(math.log2(pvq_codebook_size(vectors.shape[1], pulses)))
    rate = (shape_bits + gain_bits) / vectors.shape[1]
    return CodecResult(reconstruction.to(vectors.dtype), rate)


def _assign_codebook(
    vectors: torch.Tensor,
    codebook: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    labels = []
    center_norm = codebook.square().sum(dim=1)
    for start in range(0, vectors.shape[0], batch_size):
        batch = vectors[start : start + batch_size]
        distance = (
            batch.square().sum(dim=1, keepdim=True)
            + center_norm[None]
            - 2 * batch @ codebook.T
        )
        labels.append(distance.argmin(dim=1))
    return torch.cat(labels)


def _train_codebook(
    vectors: torch.Tensor,
    *,
    size: int,
    iterations: int,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if vectors.shape[0] < size:
        raise ValueError("fewer training vectors than codebook entries")
    initial = torch.randperm(
        vectors.shape[0], generator=generator, device=vectors.device
    )[:size]
    codebook = vectors[initial].clone()
    labels = torch.zeros(vectors.shape[0], dtype=torch.long, device=vectors.device)
    for _ in range(iterations):
        labels = _assign_codebook(vectors, codebook, batch_size=batch_size)
        sums = torch.zeros_like(codebook)
        sums.index_add_(0, labels, vectors)
        counts = torch.bincount(labels, minlength=size)
        populated = counts > 0
        codebook[populated] = sums[populated] / counts[populated, None]
        if torch.any(~populated):
            replacements = torch.randperm(
                vectors.shape[0], generator=generator, device=vectors.device
            )[: int((~populated).sum())]
            codebook[~populated] = vectors[replacements]
    labels = _assign_codebook(vectors, codebook, batch_size=batch_size)
    return codebook, labels


def train_residual_vq(
    vectors: torch.Tensor,
    *,
    stages: int,
    codebook_size: int,
    iterations: int = 12,
    batch_size: int = 8192,
    seed: int = 0,
) -> ResidualVQModel:
    """Train a greedy layer-shared residual vector quantizer."""

    if vectors.ndim != 2 or not torch.is_floating_point(vectors):
        raise ValueError("RVQ training data must be a floating-point matrix")
    if stages <= 0 or iterations <= 0 or codebook_size <= 1:
        raise ValueError("invalid RVQ stage/codebook/iteration count")
    if codebook_size & (codebook_size - 1):
        raise ValueError("fixed-rate RVQ codebook size must be a power of two")
    generator = torch.Generator(device=vectors.device).manual_seed(seed)
    residual = vectors.float().clone()
    codebooks = []
    for _ in range(stages):
        codebook, labels = _train_codebook(
            residual,
            size=codebook_size,
            iterations=iterations,
            batch_size=batch_size,
            generator=generator,
        )
        # FP16 storage is part of the proposed decoder contract.  The model is
        # converted back to FP32 for assignment/reconstruction arithmetic.
        # Round before updating the residual so training and inference use
        # exactly the same reconstruction values.
        stored_codebook = codebook.half().contiguous()
        rounded_codebook = stored_codebook.float()
        labels = _assign_codebook(residual, rounded_codebook, batch_size=batch_size)
        residual -= rounded_codebook[labels]
        codebooks.append(stored_codebook)
    return ResidualVQModel(
        codebooks=tuple(codebooks),
        vector_dimension=int(vectors.shape[1]),
        index_bits=int(math.log2(codebook_size)),
    )


def apply_residual_vq(
    vectors: torch.Tensor,
    model: ResidualVQModel,
    *,
    batch_size: int = 8192,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Greedily encode and reconstruct vectors from an RVQ model."""

    if vectors.ndim != 2 or vectors.shape[1] != model.vector_dimension:
        raise ValueError("RVQ vector dimension does not match the model")
    residual = vectors.float().clone()
    reconstruction = torch.zeros_like(residual)
    all_labels = []
    for stored_codebook in model.codebooks:
        codebook = stored_codebook.to(vectors.device, dtype=torch.float32)
        labels = _assign_codebook(residual, codebook, batch_size=batch_size)
        contribution = codebook[labels]
        reconstruction += contribution
        residual -= contribution
        all_labels.append(labels)
    return reconstruction.to(vectors.dtype), tuple(all_labels)


def index_entropy(labels: torch.Tensor, codebook_size: int) -> float:
    """Empirical zero-order entropy in bits per index."""

    counts = torch.bincount(labels.long(), minlength=codebook_size).double()
    probability = counts[counts > 0] / counts.sum()
    return float(-(probability * probability.log2()).sum())


def rank_contexts(scores: torch.Tensor, contexts: int) -> torch.Tensor:
    """Partition scalar context scores into nearly equal rank populations.

    Equal populations make a list of per-context fixed index widths have an
    exactly predictable average rate. The context IDs are ordered from the
    lowest scores to the highest scores.
    """

    if scores.ndim != 1 or not torch.is_floating_point(scores):
        raise ValueError("context scores must be a floating-point vector")
    if contexts <= 0 or contexts > scores.numel():
        raise ValueError("invalid number of contexts")
    if not torch.all(torch.isfinite(scores)):
        raise ValueError("context scores must be finite")
    order = torch.argsort(scores, stable=True)
    ordered_contexts = torch.div(
        torch.arange(scores.numel(), device=scores.device) * contexts,
        scores.numel(),
        rounding_mode="floor",
    )
    result = torch.empty_like(order)
    result[order] = ordered_contexts
    return result


def contextual_index_rate(
    assignments: torch.Tensor,
    index_bits: tuple[int, ...] | list[int],
    *,
    vector_dimension: int,
) -> float:
    """Return fixed index bits per scalar for deterministic VQ contexts."""

    if assignments.ndim != 1 or assignments.dtype != torch.long:
        raise ValueError("context assignments must be an int64 vector")
    if vector_dimension <= 0 or not index_bits:
        raise ValueError("invalid vector dimension or empty index-bit list")
    if min(index_bits) <= 0:
        raise ValueError("context index widths must be positive")
    if assignments.numel() == 0:
        raise ValueError("context assignments must not be empty")
    if int(assignments.min()) < 0 or int(assignments.max()) >= len(index_bits):
        raise ValueError("context assignment is outside the index-bit list")
    widths = torch.tensor(index_bits, dtype=torch.double, device=assignments.device)
    return float(widths[assignments].mean() / vector_dimension)
