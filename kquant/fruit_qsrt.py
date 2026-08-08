"""Canonical-profile QSRT encoding and fixed-pair storage for Fruit experts.

This module is deliberately separate from Kimi-K3's frozen TP12 atom ABI.
Fruit has a 512-channel intermediate axis: four 128-channel records stored as
exactly two fixed-size P24/P33 pairs. The low-level SQG encoder is shared; the
model geometry, artifact schema, and pair-addressable runtime metadata are not.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Protocol

import torch
import torch.nn.functional as F

from kquant.candidate_hessian import weighted_effective_sample_size
from kquant.exl3_reference import (
    CODEBOOK_SQG_XOR_CHEB_T12,
    decode_qsrt_weight,
)
from kquant.fruit_calibration import (
    FruitCalibrationRows,
    FruitExpertCalibration,
)
from kquant.ldlq import SIGMA_REG, make_shared_h
from kquant.logical_qsrt import (
    FRUIT_QSRT_GEOMETRY,
    LogicalRateMode,
    MatrixName,
    fruit_rate_modes,
    matrix_shape,
    pair_kinds,
    paired_record_bits,
    paired_record_order,
    repack_record_axis,
)
from kquant.qsrt import pack_trellis_edges, unpack_trellis_states
from kquant.qsrt_candidates import (
    PHASE1_MIN_CONFIRMATION_DOCUMENTS,
    PHASE1_MIN_FIT_DOCUMENTS,
    build_expert_hessians,
    deterministic_expert_seed,
    functional_sse_by_request,
    select_phase1_rate_pair,
)
from kquant.sqg_e4m3 import sqg_xor_cheb_t12_bytes
from kquant.tp_simulator import comparison_metrics

FRUIT_QSRT_SCHEMA = "kquant_fruit_qsrt_pairs_v1"
FRUIT_QSRT_PROFILE_ID = 1
FRUIT_QSRT_CODEBOOK = CODEBOOK_SQG_XOR_CHEB_T12
FRUIT_QSRT_MATRICES: tuple[MatrixName, ...] = ("w1", "w3", "w2")
FRUIT_QSRT_DECODE_CLOSURE_RELATIVE_L2_TOLERANCE = 1e-6
FRUIT_CALIBRATION_MIN_FIT_EFFECTIVE_ROWS = 32.0
FRUIT_QSRT_RECORD_CHANNELS = FRUIT_QSRT_GEOMETRY.record_channels
FRUIT_QSRT_RECORD_TILES = FRUIT_QSRT_RECORD_CHANNELS // 16
FRUIT_QSRT_PAIR_COUNT = FRUIT_QSRT_GEOMETRY.record_count // 2
FRUIT_QSRT_MATRIX_BYTES = (
    FRUIT_QSRT_GEOMETRY.hidden_channels
    * FRUIT_QSRT_GEOMETRY.intermediate_channels
    * 3
    // 8
)
FRUIT_QSRT_PAIR_WORDS = FRUIT_QSRT_MATRIX_BYTES // FRUIT_QSRT_PAIR_COUNT // 2
FRUIT_QSRT_ATOM_SCHEMA = "kquant_fruit_qsrt_atoms_v1"
FRUIT_QSRT_ATOM_STORAGE = "qsrt_atoms_v1"
FRUIT_QSRT_ATOM_CHANNELS = 32
FRUIT_QSRT_ATOMS_PER_PAIR = FRUIT_QSRT_RECORD_CHANNELS // (
    FRUIT_QSRT_ATOM_CHANNELS // 2
)
FRUIT_QSRT_ATOMS_PER_EXPERT = FRUIT_QSRT_PAIR_COUNT * FRUIT_QSRT_ATOMS_PER_PAIR
FRUIT_QSRT_MATRIX_ATOM_TRELLIS_BYTES = (
    FRUIT_QSRT_ATOM_CHANNELS * FRUIT_QSRT_GEOMETRY.hidden_channels * 3 // 8
)
FRUIT_QSRT_MATRIX_ATOM_SCALE_BYTES = FRUIT_QSRT_ATOM_CHANNELS * torch.float16.itemsize
FRUIT_QSRT_ATOM_TRELLIS_BYTES = 3 * FRUIT_QSRT_MATRIX_ATOM_TRELLIS_BYTES
FRUIT_QSRT_ATOM_BUNDLE_BYTES = (
    FRUIT_QSRT_ATOM_TRELLIS_BYTES + 3 * FRUIT_QSRT_MATRIX_ATOM_SCALE_BYTES
)
FRUIT_QSRT_FORMAT_SECTION_BYTES = 4096
FRUIT_QSRT_STORAGE_ALIGNMENT = 4096
FRUIT_QSRT_FORMAT_TENSOR = "_qsrt_format_section"
FRUIT_QSRT_SHARED_SCALE_TENSOR = "_qsrt_shared_scale_section"
FRUIT_QSRT_ATOM_TENSOR = "qsrt_atoms"
FRUIT_QSRT_ATOM_TENSORS = frozenset(
    {
        FRUIT_QSRT_FORMAT_TENSOR,
        FRUIT_QSRT_SHARED_SCALE_TENSOR,
        FRUIT_QSRT_ATOM_TENSOR,
    }
)
_FRUIT_QSRT_MATRIX_ATOM_TRELLIS_OFFSETS = {
    "w1": 0,
    "w3": FRUIT_QSRT_MATRIX_ATOM_TRELLIS_BYTES,
    "w2": 2 * FRUIT_QSRT_MATRIX_ATOM_TRELLIS_BYTES,
}
_FRUIT_QSRT_MATRIX_ATOM_SCALE_OFFSETS = {
    "w1": FRUIT_QSRT_ATOM_TRELLIS_BYTES,
    "w3": FRUIT_QSRT_ATOM_TRELLIS_BYTES + FRUIT_QSRT_MATRIX_ATOM_SCALE_BYTES,
    "w2": FRUIT_QSRT_ATOM_TRELLIS_BYTES + 2 * FRUIT_QSRT_MATRIX_ATOM_SCALE_BYTES,
}
FRUIT_QSRT_ARTIFACT_TENSORS = (
    "expert_ids",
    "formats",
    "permutations",
    "w13_trellis",
    "w2_trellis",
    "fc1_pair_modes",
    "fc2_pair_modes",
    "gate_suh",
    "up_suh",
    "intermediate_rotations",
    "down_svh",
)


class FruitMatrixStore(Protocol):
    """Minimal authenticated source interface consumed by the encoder."""

    def load_matrix(
        self,
        layer: int,
        expert: int,
        matrix: str,
        device: torch.device | str | None = None,
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class FruitTransformSeeds:
    """Independent deterministic EXL transform streams for one matrix."""

    input_sign: int
    output_sign: int


@dataclass
class FruitMatrixCandidate:
    """One logical R candidate before fixed-pair finalization."""

    matrix: MatrixName
    mode: LogicalRateMode
    reconstruction: torch.Tensor
    encoded: torch.Tensor
    suh: torch.Tensor
    svh: torch.Tensor
    proxy: float
    seeds: FruitTransformSeeds


@dataclass
class FruitMatrixEncoding:
    """One selected matrix in physical pair order plus canonical reconstruction."""

    matrix: MatrixName
    mode: LogicalRateMode
    trellis: torch.Tensor
    suh: torch.Tensor
    svh: torch.Tensor
    reconstruction: torch.Tensor
    coding: dict[str, object]


@dataclass
class FruitExpertEncoding:
    """Three selected matrices sharing one coupled intermediate permutation."""

    layer: int
    expert: int
    r13: int
    r2: int
    encoder_permutation: torch.Tensor
    physical_permutation: torch.Tensor
    matrices: dict[MatrixName, FruitMatrixEncoding]
    selection: dict[str, object]

    def artifact_tensors(self) -> dict[str, torch.Tensor]:
        """Return one-expert tensors with the stable layer-artifact dimensions."""

        gate = self.matrices["w1"]
        up = self.matrices["w3"]
        down = self.matrices["w2"]
        return {
            "expert_ids": torch.tensor([self.expert], dtype=torch.int32),
            "formats": torch.tensor([[self.r13, self.r2]], dtype=torch.int8),
            "permutations": self.physical_permutation.to(
                device="cpu", dtype=torch.int16
            ).unsqueeze(0),
            "w13_trellis": torch.stack((gate.trellis, up.trellis)).unsqueeze(1),
            "w2_trellis": down.trellis.unsqueeze(0),
            "fc1_pair_modes": fruit_pair_modes(self.r13).unsqueeze(0),
            "fc2_pair_modes": fruit_pair_modes(self.r2).unsqueeze(0),
            "gate_suh": gate.suh.unsqueeze(0),
            "up_suh": up.suh.unsqueeze(0),
            "intermediate_rotations": torch.cat((gate.svh, up.svh, down.suh)).unsqueeze(
                0
            ),
            "down_svh": down.svh.unsqueeze(0),
        }

    def manifest(self) -> dict[str, object]:
        return {
            "schema": FRUIT_QSRT_SCHEMA,
            "profile_id": FRUIT_QSRT_PROFILE_ID,
            "codebook": FRUIT_QSRT_CODEBOOK,
            "layer": self.layer,
            "expert": self.expert,
            "format": {"r13": self.r13, "r2": self.r2},
            "pair_kinds": {
                "w13": list(pair_kinds(fruit_rate_modes()[self.r13])),
                "w2": list(pair_kinds(fruit_rate_modes()[self.r2])),
            },
            "selection": self.selection,
            "matrices": {
                matrix: encoding.coding for matrix, encoding in self.matrices.items()
            },
        }


@dataclass(frozen=True)
class FruitLayerArtifact:
    """Stacked tensors and evidence for one complete or sampled Fruit layer."""

    layer: int
    expert_ids: tuple[int, ...]
    tensors: dict[str, torch.Tensor]
    experts: tuple[dict[str, object], ...]

    def manifest(self) -> dict[str, object]:
        return {
            "schema": FRUIT_QSRT_SCHEMA,
            "version": 1,
            "profile_id": FRUIT_QSRT_PROFILE_ID,
            "codebook": FRUIT_QSRT_CODEBOOK,
            "layer": self.layer,
            "expert_ids": list(self.expert_ids),
            "expert_count": len(self.expert_ids),
            "geometry": {
                "hidden_size": FRUIT_QSRT_GEOMETRY.hidden_channels,
                "intermediate_size": FRUIT_QSRT_GEOMETRY.intermediate_channels,
                "record_channels": FRUIT_QSRT_GEOMETRY.record_channels,
                "record_count": FRUIT_QSRT_GEOMETRY.record_count,
                "pair_count": FRUIT_QSRT_PAIR_COUNT,
            },
            "matrix_bytes": FRUIT_QSRT_MATRIX_BYTES,
            "pair_words": FRUIT_QSRT_PAIR_WORDS,
            "tensor_shapes": {
                name: list(tensor.shape) for name, tensor in self.tensors.items()
            },
            "experts": list(self.experts),
        }


def fruit_transform_seeds(
    layer: int, expert: int, matrix: MatrixName
) -> FruitTransformSeeds:
    """Derive stable per-assignment transform streams without Kimi seed gates."""

    if isinstance(layer, bool) or not isinstance(layer, int) or not 3 <= layer <= 13:
        raise ValueError("Fruit expert layer must be in 3..13")
    if isinstance(expert, bool) or not isinstance(expert, int) or not 0 <= expert < 256:
        raise ValueError("Fruit expert must be in 0..255")
    if matrix not in FRUIT_QSRT_MATRICES:
        raise ValueError(f"unknown Fruit expert matrix: {matrix!r}")
    matrix_index = FRUIT_QSRT_MATRICES.index(matrix)
    base = 17_000_003 + layer * 1_000_003 + expert * 10_007 + matrix_index * 101
    return FruitTransformSeeds(base, base + 499_979)


def fruit_pair_modes(mode: LogicalRateMode | int) -> torch.Tensor:
    """Encode pair descriptors as 0=P33 and 1=P24 for the runtime."""

    resolved = fruit_rate_modes()[mode] if isinstance(mode, int) else mode
    if not isinstance(resolved, LogicalRateMode) or resolved.record_count != 4:
        raise ValueError("Fruit pair modes require one four-record rate mode")
    return torch.tensor(
        [0 if kind == "P33" else 1 for kind in pair_kinds(resolved)],
        dtype=torch.int32,
    )


def fruit_weight_permutations(
    w1: torch.Tensor,
    w3: torch.Tensor,
    w2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build deterministic low-to-high and pair-physical channel orders.

    Four-channel group importance is the sum of each matrix's normalized source
    energy. This is a weight-only diagnostic policy, not corpus calibration.
    """

    expected = {
        "w1": matrix_shape(FRUIT_QSRT_GEOMETRY, "w1"),
        "w3": matrix_shape(FRUIT_QSRT_GEOMETRY, "w3"),
        "w2": matrix_shape(FRUIT_QSRT_GEOMETRY, "w2"),
    }
    for name, value in (("w1", w1), ("w3", w3), ("w2", w2)):
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected[name]:
            raise ValueError(
                f"{name} must have Fruit source shape {expected[name]}, got "
                f"{getattr(value, 'shape', None)}"
            )
        if value.dtype != torch.float32:
            raise TypeError(f"{name} must use torch.float32")
        if not bool(torch.all(torch.isfinite(value))):
            raise ValueError(f"{name} contains non-finite values")
    if w1.device != w3.device or w1.device != w2.device:
        raise ValueError("Fruit source matrices must share one device")

    group_channels = 4
    group_count = FRUIT_QSRT_GEOMETRY.intermediate_channels // group_channels
    energies = (
        w1.square().mean(dim=1) / w1.square().mean().clamp_min(1e-30)
        + w3.square().mean(dim=1) / w3.square().mean().clamp_min(1e-30)
        + w2.square().mean(dim=0) / w2.square().mean().clamp_min(1e-30)
    )
    group_scores = energies.reshape(group_count, group_channels).mean(dim=1)
    group_order = torch.argsort(group_scores, stable=True)
    offsets = torch.arange(group_channels, device=w1.device)
    encoder = (group_order[:, None] * group_channels + offsets[None]).flatten()
    physical = repack_record_axis(
        encoder,
        paired_record_order(fruit_rate_modes()[0]),
        axis=0,
        record_channels=FRUIT_QSRT_RECORD_CHANNELS,
    )
    return encoder.contiguous(), physical.contiguous(), group_scores.contiguous()


def _rate_axis(matrix: MatrixName) -> str:
    return "k" if matrix == "w2" else "n"


def _encoder_weight(
    source: torch.Tensor, matrix: MatrixName, permutation: torch.Tensor
) -> torch.Tensor:
    if matrix == "w2":
        return source.index_select(1, permutation).T.contiguous()
    return source.index_select(0, permutation).T.contiguous()


def _canonical_reconstruction(
    encoder_weight: torch.Tensor,
    source: torch.Tensor,
    matrix: MatrixName,
    permutation: torch.Tensor,
) -> torch.Tensor:
    result = torch.empty_like(source)
    if matrix == "w2":
        result[:, permutation] = encoder_weight.T
    else:
        result[permutation] = encoder_weight.T
    return result


def _tile_bits(mode: LogicalRateMode, *, physical: bool) -> tuple[int, ...]:
    rates = paired_record_bits(mode) if physical else mode.record_bits
    return tuple(bits for bits in rates for _ in range(FRUIT_QSRT_RECORD_TILES))


def _quant_args(
    matrix: MatrixName,
    mode: LogicalRateMode,
    *,
    layer: int,
    expert: int,
    device: torch.device,
) -> dict[str, object]:
    seeds = fruit_transform_seeds(layer, expert, matrix)
    result: dict[str, object] = {
        "K": 3,
        "mixed_rate_axis": _rate_axis(matrix),
        "mixed_tile_bits": _tile_bits(mode, physical=False),
        "seed": seeds.input_sign,
        "sv_seed": seeds.output_sign,
        "sigma_reg": SIGMA_REG,
        "devices": [str(device)],
        "device_ratios": None,
        "apply_out_scales": False,
        "ldlq_tf32": False,
        "tailbite_context": 128,
        "sqg_e4m3_luts_by_bits": {
            bits: sqg_xor_cheb_t12_bytes(bits) for bits in (2, 3, 4)
        },
    }
    if matrix in ("w1", "w3"):
        result["g_scale_into_sv"] = True
    return result


def _validate_encoded(encoded: torch.Tensor, rate_axis: str) -> None:
    if encoded.ndim != 3 or encoded.shape[-1] != 256:
        raise ValueError("Fruit encoded states must have shape [K/16,N/16,256]")
    if encoded.dtype == torch.bool or encoded.is_floating_point():
        raise TypeError("Fruit encoded states must use an integer dtype")
    expected = FRUIT_QSRT_GEOMETRY.intermediate_channels // 16
    rate_tiles = encoded.shape[0] if rate_axis == "k" else encoded.shape[1]
    orthogonal_tiles = encoded.shape[1] if rate_axis == "k" else encoded.shape[0]
    if (
        rate_tiles != expected
        or orthogonal_tiles != FRUIT_QSRT_GEOMETRY.hidden_channels // 16
    ):
        raise ValueError("encoded state geometry does not match Fruit H=1024/I=512")


def _record_tiles(encoded: torch.Tensor, record: int, rate_axis: str) -> torch.Tensor:
    begin = record * FRUIT_QSRT_RECORD_TILES
    if rate_axis == "k":
        return encoded.narrow(0, begin, FRUIT_QSRT_RECORD_TILES).reshape(-1, 256)
    return encoded.narrow(1, begin, FRUIT_QSRT_RECORD_TILES).reshape(-1, 256)


def pack_fruit_trellis(
    encoded_physical: torch.Tensor,
    mode: LogicalRateMode,
    *,
    rate_axis: str,
    validate_states: bool = True,
) -> torch.Tensor:
    """Pack one physical-order matrix as ``[two_pairs, fixed_words]`` int16."""

    _validate_encoded(encoded_physical, rate_axis)
    if mode.record_count != FRUIT_QSRT_GEOMETRY.record_count:
        raise ValueError("Fruit trellis mode must contain four records")
    pieces: list[torch.Tensor] = []
    for record, bits in enumerate(paired_record_bits(mode)):
        states = _record_tiles(encoded_physical, record, rate_axis)
        packed = pack_trellis_edges(states, bits)
        if validate_states and not torch.equal(
            unpack_trellis_states(packed, bits), states.to(torch.int16)
        ):
            raise ValueError("Fruit encoder produced a non-cyclic trellis path")
        pieces.append(packed.reshape(-1))
    pairs = [
        torch.cat((pieces[index], pieces[index + 1])).contiguous()
        for index in range(0, len(pieces), 2)
    ]
    result = torch.stack(pairs)
    if tuple(result.shape) != (FRUIT_QSRT_PAIR_COUNT, FRUIT_QSRT_PAIR_WORDS):
        raise AssertionError("Fruit pair packing violated its fixed-size contract")
    return result


def unpack_fruit_trellis(
    payload: torch.Tensor,
    mode: LogicalRateMode,
    *,
    rate_axis: str,
) -> torch.Tensor:
    """Reconstruct physical-order 16-bit states from two fixed-size pairs."""

    if payload.dtype != torch.int16:
        raise TypeError("Fruit trellis payload must use torch.int16")
    if tuple(payload.shape) != (FRUIT_QSRT_PAIR_COUNT, FRUIT_QSRT_PAIR_WORDS):
        raise ValueError(
            "Fruit trellis payload must have shape "
            f"{(FRUIT_QSRT_PAIR_COUNT, FRUIT_QSRT_PAIR_WORDS)}"
        )
    rates = paired_record_bits(mode)
    k_tiles = (
        FRUIT_QSRT_GEOMETRY.intermediate_channels // 16
        if rate_axis == "k"
        else FRUIT_QSRT_GEOMETRY.hidden_channels // 16
    )
    n_tiles = (
        FRUIT_QSRT_GEOMETRY.hidden_channels // 16
        if rate_axis == "k"
        else FRUIT_QSRT_GEOMETRY.intermediate_channels // 16
    )
    output = torch.empty(
        (k_tiles, n_tiles, 256), dtype=torch.int16, device=payload.device
    )
    orthogonal_tiles = n_tiles if rate_axis == "k" else k_tiles
    tiles_per_record = orthogonal_tiles * FRUIT_QSRT_RECORD_TILES
    for pair in range(FRUIT_QSRT_PAIR_COUNT):
        cursor = 0
        pair_payload = payload[pair]
        for side in range(2):
            record = pair * 2 + side
            bits = rates[record]
            words = tiles_per_record * 16 * bits
            packed = pair_payload.narrow(0, cursor, words).reshape(
                tiles_per_record, 16 * bits
            )
            states = unpack_trellis_states(packed, bits)
            begin = record * FRUIT_QSRT_RECORD_TILES
            if rate_axis == "k":
                output.narrow(0, begin, FRUIT_QSRT_RECORD_TILES).copy_(
                    states.reshape(FRUIT_QSRT_RECORD_TILES, n_tiles, 256)
                )
            else:
                output.narrow(1, begin, FRUIT_QSRT_RECORD_TILES).copy_(
                    states.reshape(k_tiles, FRUIT_QSRT_RECORD_TILES, 256)
                )
            cursor += words
        if cursor != FRUIT_QSRT_PAIR_WORDS:
            raise AssertionError("Fruit pair decoder did not consume its fixed payload")
    return output


def _validate_fruit_pair_payload(
    payload: torch.Tensor, mode: LogicalRateMode, matrix: MatrixName
) -> None:
    if matrix not in FRUIT_QSRT_MATRICES:
        raise ValueError(f"unknown Fruit expert matrix: {matrix!r}")
    if not isinstance(mode, LogicalRateMode) or mode.record_count != 4:
        raise ValueError("Fruit atom packing requires one four-record rate mode")
    if payload.dtype != torch.int16 or tuple(payload.shape) != (
        FRUIT_QSRT_PAIR_COUNT,
        FRUIT_QSRT_PAIR_WORDS,
    ):
        raise ValueError(
            "Fruit atom packing requires contiguous int16 pair payload "
            f"{(FRUIT_QSRT_PAIR_COUNT, FRUIT_QSRT_PAIR_WORDS)}"
        )
    if not payload.is_contiguous():
        raise ValueError("Fruit atom packing requires a contiguous pair payload")


def pack_fruit_matrix_atoms(
    payload: torch.Tensor,
    mode: LogicalRateMode,
    *,
    matrix: MatrixName,
) -> torch.Tensor:
    """Transpose one fixed-pair matrix into 16 logical 32-channel atoms."""

    _validate_fruit_pair_payload(payload, mode, matrix)
    hidden_tiles = FRUIT_QSRT_GEOMETRY.hidden_channels // 16
    words_per_atom = FRUIT_QSRT_MATRIX_ATOM_TRELLIS_BYTES // torch.int16.itemsize
    output = torch.empty(
        (FRUIT_QSRT_ATOMS_PER_EXPERT, words_per_atom),
        dtype=torch.int16,
        device=payload.device,
    )
    rates = paired_record_bits(mode)
    fc1 = matrix in ("w1", "w3")
    for pair in range(FRUIT_QSRT_PAIR_COUNT):
        pair_payload = payload[pair]
        cursor = 0
        sides: list[torch.Tensor] = []
        for side in range(2):
            bits = rates[2 * pair + side]
            side_words = hidden_tiles * FRUIT_QSRT_ATOMS_PER_PAIR * 16 * bits
            side_payload = pair_payload.narrow(0, cursor, side_words)
            if fc1:
                side_payload = side_payload.reshape(
                    hidden_tiles, FRUIT_QSRT_ATOMS_PER_PAIR, 16 * bits
                ).permute(1, 0, 2)
            else:
                side_payload = side_payload.reshape(
                    FRUIT_QSRT_ATOMS_PER_PAIR, hidden_tiles, 16 * bits
                )
            sides.append(side_payload.reshape(FRUIT_QSRT_ATOMS_PER_PAIR, -1))
            cursor += side_words
        if cursor != FRUIT_QSRT_PAIR_WORDS:
            raise AssertionError("Fruit atom packer did not consume one pair")
        begin = pair * FRUIT_QSRT_ATOMS_PER_PAIR
        output.narrow(0, begin, FRUIT_QSRT_ATOMS_PER_PAIR).copy_(
            torch.cat(sides, dim=1)
        )
    return output.contiguous()


def unpack_fruit_matrix_atoms(
    atoms: torch.Tensor,
    mode: LogicalRateMode,
    *,
    matrix: MatrixName,
) -> torch.Tensor:
    """Invert :func:`pack_fruit_matrix_atoms` without decoding trellis states."""

    if matrix not in FRUIT_QSRT_MATRICES:
        raise ValueError(f"unknown Fruit expert matrix: {matrix!r}")
    words_per_atom = FRUIT_QSRT_MATRIX_ATOM_TRELLIS_BYTES // torch.int16.itemsize
    if (
        atoms.dtype != torch.int16
        or tuple(atoms.shape) != (FRUIT_QSRT_ATOMS_PER_EXPERT, words_per_atom)
        or not atoms.is_contiguous()
    ):
        raise ValueError(
            "Fruit matrix atoms must be contiguous int16 "
            f"{(FRUIT_QSRT_ATOMS_PER_EXPERT, words_per_atom)}"
        )
    if not isinstance(mode, LogicalRateMode) or mode.record_count != 4:
        raise ValueError("Fruit atom unpacking requires one four-record rate mode")
    hidden_tiles = FRUIT_QSRT_GEOMETRY.hidden_channels // 16
    rates = paired_record_bits(mode)
    fc1 = matrix in ("w1", "w3")
    output = torch.empty(
        (FRUIT_QSRT_PAIR_COUNT, FRUIT_QSRT_PAIR_WORDS),
        dtype=torch.int16,
        device=atoms.device,
    )
    for pair in range(FRUIT_QSRT_PAIR_COUNT):
        source = atoms.narrow(
            0, pair * FRUIT_QSRT_ATOMS_PER_PAIR, FRUIT_QSRT_ATOMS_PER_PAIR
        )
        source_cursor = 0
        pieces: list[torch.Tensor] = []
        for side in range(2):
            bits = rates[2 * pair + side]
            side_words_per_atom = hidden_tiles * 16 * bits
            side_payload = source.narrow(1, source_cursor, side_words_per_atom)
            if fc1:
                side_payload = side_payload.reshape(
                    FRUIT_QSRT_ATOMS_PER_PAIR, hidden_tiles, 16 * bits
                ).permute(1, 0, 2)
            pieces.append(side_payload.contiguous().reshape(-1))
            source_cursor += side_words_per_atom
        if source_cursor != words_per_atom:
            raise AssertionError("Fruit atom unpacker did not consume one atom")
        output[pair].copy_(torch.cat(pieces))
    return output.contiguous()


def pack_fruit_local_scale_atoms(scale: torch.Tensor) -> torch.Tensor:
    """Transpose one physical 512-value local scale into logical atoms."""

    expected_shape = (FRUIT_QSRT_GEOMETRY.intermediate_channels,)
    if (
        scale.dtype != torch.float16
        or tuple(scale.shape) != expected_shape
        or not scale.is_contiguous()
    ):
        raise ValueError(
            f"Fruit local scale must be contiguous float16 {expected_shape}"
        )
    side_channels = FRUIT_QSRT_ATOM_CHANNELS // 2
    output = torch.empty(
        (FRUIT_QSRT_ATOMS_PER_EXPERT, FRUIT_QSRT_ATOM_CHANNELS),
        dtype=torch.float16,
        device=scale.device,
    )
    for pair in range(FRUIT_QSRT_PAIR_COUNT):
        for stripe in range(FRUIT_QSRT_ATOMS_PER_PAIR):
            atom = pair * FRUIT_QSRT_ATOMS_PER_PAIR + stripe
            for side in range(2):
                record = 2 * pair + side
                source_begin = (
                    record * FRUIT_QSRT_RECORD_CHANNELS + stripe * side_channels
                )
                output[atom, side * side_channels : (side + 1) * side_channels].copy_(
                    scale.narrow(0, source_begin, side_channels)
                )
    return output.contiguous()


def unpack_fruit_local_scale_atoms(atoms: torch.Tensor) -> torch.Tensor:
    """Invert :func:`pack_fruit_local_scale_atoms`."""

    expected_shape = (
        FRUIT_QSRT_ATOMS_PER_EXPERT,
        FRUIT_QSRT_ATOM_CHANNELS,
    )
    if (
        atoms.dtype != torch.float16
        or tuple(atoms.shape) != expected_shape
        or not atoms.is_contiguous()
    ):
        raise ValueError(
            f"Fruit local-scale atoms must be contiguous float16 {expected_shape}"
        )
    side_channels = FRUIT_QSRT_ATOM_CHANNELS // 2
    output = torch.empty(
        FRUIT_QSRT_GEOMETRY.intermediate_channels,
        dtype=torch.float16,
        device=atoms.device,
    )
    for pair in range(FRUIT_QSRT_PAIR_COUNT):
        for stripe in range(FRUIT_QSRT_ATOMS_PER_PAIR):
            atom = pair * FRUIT_QSRT_ATOMS_PER_PAIR + stripe
            for side in range(2):
                record = 2 * pair + side
                target_begin = (
                    record * FRUIT_QSRT_RECORD_CHANNELS + stripe * side_channels
                )
                output.narrow(0, target_begin, side_channels).copy_(
                    atoms[atom, side * side_channels : (side + 1) * side_channels]
                )
    return output.contiguous()


def fruit_physical_atom_rotation(layer: int, expert: int) -> int:
    """Return the whole-pair storage rotation for one Fruit expert."""

    if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
        raise ValueError("Fruit atom layer must be a nonnegative integer")
    if isinstance(expert, bool) or not isinstance(expert, int) or expert < 0:
        raise ValueError("Fruit atom expert must be a nonnegative integer")
    pair_rotation = (5 * expert + layer) % FRUIT_QSRT_PAIR_COUNT
    return pair_rotation * FRUIT_QSRT_ATOMS_PER_PAIR


def rotate_fruit_expert_atoms(
    logical_atoms: torch.Tensor, layer: int, expert: int
) -> torch.Tensor:
    """Place logical Fruit atoms in global physical-slot order."""

    if logical_atoms.ndim < 1 or logical_atoms.shape[0] != FRUIT_QSRT_ATOMS_PER_EXPERT:
        raise ValueError(
            "logical Fruit atoms must have leading dimension "
            f"{FRUIT_QSRT_ATOMS_PER_EXPERT}"
        )
    return torch.roll(
        logical_atoms,
        shifts=fruit_physical_atom_rotation(layer, expert),
        dims=0,
    ).contiguous()


def unrotate_fruit_expert_atoms(
    physical_atoms: torch.Tensor, layer: int, expert: int
) -> torch.Tensor:
    """Return physical Fruit atoms to logical record-pair order."""

    if (
        physical_atoms.ndim < 1
        or physical_atoms.shape[0] != FRUIT_QSRT_ATOMS_PER_EXPERT
    ):
        raise ValueError(
            "physical Fruit atoms must have leading dimension "
            f"{FRUIT_QSRT_ATOMS_PER_EXPERT}"
        )
    return torch.roll(
        physical_atoms,
        shifts=-fruit_physical_atom_rotation(layer, expert),
        dims=0,
    ).contiguous()


def pack_fruit_expert_atoms(
    *,
    layer: int,
    expert: int,
    r13: int,
    r2: int,
    w13_trellis: torch.Tensor,
    w2_trellis: torch.Tensor,
    intermediate_rotations: torch.Tensor,
) -> torch.Tensor:
    """Pack one pair-shaped Fruit expert as physical byte atoms."""

    modes = fruit_rate_modes()
    if r13 not in range(len(modes)) or r2 not in range(len(modes)):
        raise ValueError("Fruit atom formats must encode R0/R1/R2")
    if (
        w13_trellis.dtype != torch.int16
        or tuple(w13_trellis.shape) != (2, FRUIT_QSRT_PAIR_COUNT, FRUIT_QSRT_PAIR_WORDS)
        or not w13_trellis.is_contiguous()
    ):
        raise ValueError("Fruit w13 atom source has the wrong pair contract")
    if (
        w2_trellis.dtype != torch.int16
        or tuple(w2_trellis.shape) != (FRUIT_QSRT_PAIR_COUNT, FRUIT_QSRT_PAIR_WORDS)
        or not w2_trellis.is_contiguous()
    ):
        raise ValueError("Fruit w2 atom source has the wrong pair contract")
    if (
        intermediate_rotations.dtype != torch.float16
        or tuple(intermediate_rotations.shape)
        != (3 * FRUIT_QSRT_GEOMETRY.intermediate_channels,)
        or not intermediate_rotations.is_contiguous()
    ):
        raise ValueError("Fruit local rotations have the wrong atom contract")

    logical = torch.empty(
        (FRUIT_QSRT_ATOMS_PER_EXPERT, FRUIT_QSRT_ATOM_BUNDLE_BYTES),
        dtype=torch.uint8,
        device=w13_trellis.device,
    )
    for matrix, payload, mode_id, local_index in (
        ("w1", w13_trellis[0].contiguous(), r13, 0),
        ("w3", w13_trellis[1].contiguous(), r13, 1),
        ("w2", w2_trellis, r2, 2),
    ):
        trellis = pack_fruit_matrix_atoms(payload, modes[mode_id], matrix=matrix).view(
            torch.uint8
        )
        trellis_begin = _FRUIT_QSRT_MATRIX_ATOM_TRELLIS_OFFSETS[matrix]
        logical[
            :, trellis_begin : trellis_begin + FRUIT_QSRT_MATRIX_ATOM_TRELLIS_BYTES
        ].copy_(trellis.reshape(FRUIT_QSRT_ATOMS_PER_EXPERT, -1))
        scale_begin = _FRUIT_QSRT_MATRIX_ATOM_SCALE_OFFSETS[matrix]
        local = intermediate_rotations.narrow(
            0,
            local_index * FRUIT_QSRT_GEOMETRY.intermediate_channels,
            FRUIT_QSRT_GEOMETRY.intermediate_channels,
        )
        scale_atoms = pack_fruit_local_scale_atoms(local.contiguous()).view(torch.uint8)
        logical[
            :, scale_begin : scale_begin + FRUIT_QSRT_MATRIX_ATOM_SCALE_BYTES
        ].copy_(scale_atoms.reshape(FRUIT_QSRT_ATOMS_PER_EXPERT, -1))
    return rotate_fruit_expert_atoms(logical, layer, expert)


def _align_fruit_atom_section(value: int) -> int:
    return (
        (value + FRUIT_QSRT_STORAGE_ALIGNMENT - 1)
        // FRUIT_QSRT_STORAGE_ALIGNMENT
        * FRUIT_QSRT_STORAGE_ALIGNMENT
    )


def pack_fruit_atom_layer(
    tensors: Mapping[str, torch.Tensor], *, layer: int
) -> dict[str, torch.Tensor]:
    """Convert a complete pair-shaped layer into the serving atom tensors."""

    required = {
        "expert_ids",
        "formats",
        "w13_trellis",
        "w2_trellis",
        "gate_suh",
        "up_suh",
        "intermediate_rotations",
        "down_svh",
    }
    if not required.issubset(tensors):
        raise ValueError("Fruit pair layer is missing atom source tensors")
    expert_ids = tensors["expert_ids"]
    formats = tensors["formats"]
    experts = int(expert_ids.numel())
    if (
        expert_ids.dtype != torch.int32
        or tuple(expert_ids.shape) != (experts,)
        or not torch.equal(expert_ids, torch.arange(experts, dtype=torch.int32))
    ):
        raise ValueError("Fruit atom layers require dense ordered expert IDs")
    if (
        formats.dtype != torch.int8
        or tuple(formats.shape) != (experts, 2)
        or bool(torch.any((formats < 0) | (formats > 2)))
    ):
        raise ValueError("Fruit atom layer formats must be [experts,2] R0/R1/R2")
    atoms = torch.empty(
        (FRUIT_QSRT_ATOMS_PER_EXPERT, experts, FRUIT_QSRT_ATOM_BUNDLE_BYTES),
        dtype=torch.uint8,
    )
    for expert in range(experts):
        atoms[:, expert].copy_(
            pack_fruit_expert_atoms(
                layer=layer,
                expert=expert,
                r13=int(formats[expert, 0]),
                r2=int(formats[expert, 1]),
                w13_trellis=tensors["w13_trellis"][:, expert].contiguous(),
                w2_trellis=tensors["w2_trellis"][expert].contiguous(),
                intermediate_rotations=tensors["intermediate_rotations"][
                    expert
                ].contiguous(),
            )
        )

    format_section = torch.zeros(FRUIT_QSRT_FORMAT_SECTION_BYTES, dtype=torch.uint8)
    format_section[:experts] = (formats[:, 0].to(torch.uint8) << 4) | formats[:, 1].to(
        torch.uint8
    )
    shared = torch.stack(
        (tensors["gate_suh"], tensors["up_suh"], tensors["down_svh"])
    ).contiguous()
    expected_shared = (3, experts, FRUIT_QSRT_GEOMETRY.hidden_channels)
    if shared.dtype != torch.float16 or tuple(shared.shape) != expected_shared:
        raise ValueError(f"Fruit shared rotations must have shape {expected_shared}")
    shared_raw = shared.view(torch.uint8).reshape(-1)
    shared_section = torch.zeros(
        _align_fruit_atom_section(shared_raw.numel()), dtype=torch.uint8
    )
    shared_section[: shared_raw.numel()].copy_(shared_raw)

    payload_bytes = experts * FRUIT_QSRT_ATOM_BUNDLE_BYTES
    stride_bytes = _align_fruit_atom_section(payload_bytes)
    atom_slab = torch.zeros(
        (FRUIT_QSRT_ATOMS_PER_EXPERT, stride_bytes), dtype=torch.uint8
    )
    atom_slab[:, :payload_bytes].copy_(
        atoms.reshape(FRUIT_QSRT_ATOMS_PER_EXPERT, payload_bytes)
    )
    return {
        FRUIT_QSRT_FORMAT_TENSOR: format_section,
        FRUIT_QSRT_SHARED_SCALE_TENSOR: shared_section,
        FRUIT_QSRT_ATOM_TENSOR: atom_slab,
    }


def decode_fruit_matrix(
    payload: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    *,
    matrix: MatrixName,
    mode: LogicalRateMode,
) -> torch.Tensor:
    """Decode one stored matrix to physical EXL ``[K,N]`` orientation."""

    rate_axis = _rate_axis(matrix)
    return decode_qsrt_weight(
        unpack_fruit_trellis(payload, mode, rate_axis=rate_axis),
        suh,
        svh,
        rate_axis=rate_axis,
        tile_bits=_tile_bits(mode, physical=True),
        codebook=FRUIT_QSRT_CODEBOOK,
    )


def _candidate(
    matrix: MatrixName,
    mode: LogicalRateMode,
    raw: Mapping[str, object],
    source: torch.Tensor,
    permutation: torch.Tensor,
    seeds: FruitTransformSeeds,
) -> FruitMatrixCandidate:
    encoder_weight = raw.get("weight_q")
    encoded = raw.get("encoded")
    suh = raw.get("suh")
    svh = raw.get("svh")
    proxy = raw.get("proxy")
    if not all(
        isinstance(value, torch.Tensor) for value in (encoder_weight, encoded, suh, svh)
    ) or not isinstance(proxy, (int, float)):
        raise ValueError("QSRT backend returned an incomplete Fruit candidate")
    assert isinstance(encoder_weight, torch.Tensor)
    assert isinstance(encoded, torch.Tensor)
    assert isinstance(suh, torch.Tensor)
    assert isinstance(svh, torch.Tensor)
    if not math.isfinite(float(proxy)):
        raise ValueError("QSRT backend returned a non-finite Fruit proxy")
    reconstruction = _canonical_reconstruction(
        encoder_weight, source, matrix, permutation
    )
    if not bool(torch.all(torch.isfinite(reconstruction))):
        raise ValueError("QSRT backend returned a non-finite Fruit reconstruction")
    return FruitMatrixCandidate(
        matrix=matrix,
        mode=mode,
        reconstruction=reconstruction,
        encoded=encoded,
        suh=suh,
        svh=svh,
        proxy=float(proxy),
        seeds=seeds,
    )


def _normalized_sse(source: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    return (
        source - candidate
    ).double().square().sum() / source.double().square().sum().clamp_min(1e-30)


def _finalize_candidate(
    candidate: FruitMatrixCandidate,
    source: torch.Tensor,
    encoder_permutation: torch.Tensor,
    physical_permutation: torch.Tensor,
) -> FruitMatrixEncoding:
    matrix = candidate.matrix
    rate_axis = _rate_axis(matrix)
    order = paired_record_order(candidate.mode)
    tile_axis = 0 if rate_axis == "k" else 1
    physical_states = repack_record_axis(
        candidate.encoded,
        order,
        axis=tile_axis,
        record_channels=FRUIT_QSRT_RECORD_TILES,
    )
    suh = candidate.suh
    svh = candidate.svh
    if rate_axis == "k":
        suh = repack_record_axis(
            suh,
            order,
            axis=0,
            record_channels=FRUIT_QSRT_RECORD_CHANNELS,
        )
    else:
        svh = repack_record_axis(
            svh,
            order,
            axis=0,
            record_channels=FRUIT_QSRT_RECORD_CHANNELS,
        )
    trellis = pack_fruit_trellis(physical_states, candidate.mode, rate_axis=rate_axis)
    decoded_physical = decode_fruit_matrix(
        trellis, suh, svh, matrix=matrix, mode=candidate.mode
    )
    encoder_weight = _encoder_weight(
        candidate.reconstruction, matrix, encoder_permutation
    )
    expected_physical = repack_record_axis(
        encoder_weight,
        order,
        axis=tile_axis,
        record_channels=FRUIT_QSRT_RECORD_CHANNELS,
    )
    stored_metrics = comparison_metrics(expected_physical, decoded_physical)
    if not all(math.isfinite(value) for value in stored_metrics.values()):
        raise ValueError("Fruit stored decode produced non-finite comparison metrics")
    if stored_metrics["relative_l2"] > FRUIT_QSRT_DECODE_CLOSURE_RELATIVE_L2_TOLERANCE:
        raise ValueError("Fruit stored decode exceeded the closure tolerance")

    reconstruction = _canonical_reconstruction(
        decoded_physical, source, matrix, physical_permutation
    )
    canonical_metrics = comparison_metrics(candidate.reconstruction, reconstruction)
    if not all(math.isfinite(value) for value in canonical_metrics.values()):
        raise ValueError("Fruit canonical reconstruction produced non-finite metrics")
    if (
        canonical_metrics["relative_l2"]
        > FRUIT_QSRT_DECODE_CLOSURE_RELATIVE_L2_TOLERANCE
    ):
        raise ValueError(
            "Fruit canonical reconstruction exceeded the closure tolerance"
        )
    coding = {
        "mode": candidate.mode.name,
        "mode_id": candidate.mode.transfer,
        "pair_kinds": list(pair_kinds(candidate.mode)),
        "record_bits_encoder": list(candidate.mode.record_bits),
        "record_bits_physical": list(paired_record_bits(candidate.mode)),
        "rate_axis": rate_axis,
        "proxy": candidate.proxy,
        "trellis_bytes": trellis.numel() * trellis.element_size(),
        "scale_bytes": (suh.numel() + svh.numel()) * torch.float16.itemsize,
        "stored_decode_vs_encoder": stored_metrics,
        "canonical_decode_closure": canonical_metrics,
        "transform_seeds": {
            "input_sign": candidate.seeds.input_sign,
            "output_sign": candidate.seeds.output_sign,
        },
    }
    return FruitMatrixEncoding(
        matrix=matrix,
        mode=candidate.mode,
        trellis=trellis.to(device="cpu", dtype=torch.int16).contiguous(),
        suh=suh.to(device="cpu", dtype=torch.float16).contiguous(),
        svh=svh.to(device="cpu", dtype=torch.float16).contiguous(),
        reconstruction=reconstruction,
        coding=coding,
    )


def _encode_matrix_family(
    sources: Sequence[torch.Tensor],
    matrices: Sequence[MatrixName],
    permutation: torch.Tensor,
    *,
    hessians: Sequence[torch.Tensor],
    layer: int,
    expert: int,
    device: torch.device,
    quantizer_module: ModuleType | object,
) -> dict[MatrixName, tuple[FruitMatrixCandidate, ...]]:
    modes = fruit_rate_modes()
    weights = [
        _encoder_weight(source, matrix, permutation)
        for source, matrix in zip(sources, matrices, strict=True)
    ]
    shared_h = [
        make_shared_h(weight.shape[0], device, hessian)
        for weight, hessian in zip(weights, hessians, strict=True)
    ]
    groups = [
        [
            _quant_args(matrix, mode, layer=layer, expert=expert, device=device)
            for mode in modes
        ]
        for matrix in matrices
    ]
    batch_api = getattr(quantizer_module, "quantize_qsrt_batch", None)
    if batch_api is None:
        raise ValueError("QSRT encoder backend lacks quantize_qsrt_batch")
    raw_groups = batch_api(weights, shared_h, groups, return_weight_q=True)
    if len(raw_groups) != len(matrices):
        raise ValueError("QSRT backend returned the wrong Fruit matrix count")
    result: dict[MatrixName, tuple[FruitMatrixCandidate, ...]] = {}
    for matrix, source, raw_group in zip(matrices, sources, raw_groups, strict=True):
        if len(raw_group) != len(modes):
            raise ValueError("QSRT backend returned the wrong Fruit mode count")
        seeds = fruit_transform_seeds(layer, expert, matrix)
        result[matrix] = tuple(
            _candidate(matrix, mode, raw, source, permutation, seeds)
            for mode, raw in zip(modes, raw_group, strict=True)
        )
    return result


def _finalize_expert(
    sources: Mapping[MatrixName, torch.Tensor],
    selected: Mapping[MatrixName, FruitMatrixCandidate],
    encoder_permutation: torch.Tensor,
    physical_permutation: torch.Tensor,
    *,
    layer: int,
    expert: int,
    r13: int,
    r2: int,
    selection: dict[str, object],
) -> FruitExpertEncoding:
    finalized = {
        matrix: _finalize_candidate(
            selected[matrix],
            sources[matrix],
            encoder_permutation,
            physical_permutation,
        )
        for matrix in FRUIT_QSRT_MATRICES
    }
    return FruitExpertEncoding(
        layer=layer,
        expert=expert,
        r13=r13,
        r2=r2,
        encoder_permutation=encoder_permutation.detach().cpu().contiguous(),
        physical_permutation=physical_permutation.detach().cpu().contiguous(),
        matrices=finalized,
        selection=selection,
    )


def _encode_identity_expert(
    sources: Mapping[MatrixName, torch.Tensor],
    *,
    layer: int,
    expert: int,
    device: torch.device,
    quantizer_module: ModuleType | object,
    global_h13: torch.Tensor | None = None,
    fallback_evidence: Mapping[str, object] | None = None,
    force_r0_reason: str | None = None,
    calibration_fingerprint: str | None = None,
) -> FruitExpertEncoding:
    encoder_permutation, physical_permutation, group_scores = fruit_weight_permutations(
        sources["w1"], sources["w3"], sources["w2"]
    )
    if global_h13 is None:
        h13 = torch.eye(
            FRUIT_QSRT_GEOMETRY.hidden_channels, dtype=torch.float32, device=device
        )
    else:
        expected_h13 = (FRUIT_QSRT_GEOMETRY.hidden_channels,) * 2
        if global_h13.dtype != torch.float32 or tuple(global_h13.shape) != expected_h13:
            raise ValueError("Fruit fallback global H13 has the wrong contract")
        h13 = global_h13.to(device=device, dtype=torch.float32).contiguous()
    identity_h2 = torch.eye(
        FRUIT_QSRT_GEOMETRY.intermediate_channels, dtype=torch.float32, device=device
    )
    upstream = _encode_matrix_family(
        (sources["w1"], sources["w3"]),
        ("w1", "w3"),
        encoder_permutation,
        hessians=(h13, h13),
        layer=layer,
        expert=expert,
        device=device,
        quantizer_module=quantizer_module,
    )
    downstream = _encode_matrix_family(
        (sources["w2"],),
        ("w2",),
        encoder_permutation,
        hessians=(identity_h2,),
        layer=layer,
        expert=expert,
        device=device,
        quantizer_module=quantizer_module,
    )
    candidates = {**upstream, **downstream}
    r13_scores = torch.stack(
        [
            _normalized_sse(sources["w1"], candidates["w1"][mode].reconstruction)
            + _normalized_sse(sources["w3"], candidates["w3"][mode].reconstruction)
            for mode in range(3)
        ]
    )
    r2_scores = torch.stack(
        [
            _normalized_sse(sources["w2"], candidates["w2"][mode].reconstruction)
            for mode in range(3)
        ]
    )
    if force_r0_reason is None:
        r13, r2 = (int(r13_scores.argmin().cpu()), int(r2_scores.argmin().cpu()))
        selection: dict[str, object] = {
            "policy": "weight_normalized_sse_v1",
            "calibrated_activations": False,
            "hessian_policy": "identity",
        }
    else:
        r13 = r2 = 0
        selection = {
            "policy": "activation_coupled_functional_sse_v1",
            "calibrated_activations": True,
            "hessian_policy": (
                "global_fit_h13_identity_h2_fallback"
                if global_h13 is not None
                else "identity_fallback"
            ),
            "accepted": False,
            "reason": force_r0_reason,
            "calibration_fingerprint": calibration_fingerprint,
        }
        if fallback_evidence is not None:
            selection["support"] = dict(fallback_evidence)
    selection.update(
        {
            "r13_scores": [float(value) for value in r13_scores.cpu().tolist()],
            "r2_scores": [float(value) for value in r2_scores.cpu().tolist()],
            "group_score_min": float(group_scores.min().cpu()),
            "group_score_max": float(group_scores.max().cpu()),
        }
    )
    return _finalize_expert(
        sources,
        {
            "w1": candidates["w1"][r13],
            "w3": candidates["w3"][r13],
            "w2": candidates["w2"][r2],
        },
        encoder_permutation,
        physical_permutation,
        layer=layer,
        expert=expert,
        r13=r13,
        r2=r2,
        selection=selection,
    )


def _middle_rows(
    rows: FruitCalibrationRows,
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    values = rows.inputs.to(device=device, dtype=torch.float32)
    return F.silu(values @ gate.T) * (values @ up.T)


def _activation_permutations(
    middle: torch.Tensor,
    gates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if middle.ndim != 2 or middle.shape[1] != FRUIT_QSRT_GEOMETRY.intermediate_channels:
        raise ValueError("Fruit calibration middle rows have the wrong shape")
    if gates.ndim != 1 or gates.numel() != middle.shape[0] or not gates.numel():
        raise ValueError("Fruit calibration permutation requires routed fit rows")
    weights = gates.to(device=middle.device, dtype=torch.float32).square()
    denominator = weights.sum()
    if not bool(torch.isfinite(denominator)) or float(denominator) <= 0:
        raise ValueError("Fruit calibration permutation has no positive route weight")
    channel_scores = (middle.square() * weights[:, None]).sum(dim=0) / denominator
    group_scores = channel_scores.reshape(-1, 4).mean(dim=1)
    group_order = torch.argsort(group_scores, stable=True)
    offsets = torch.arange(4, device=middle.device)
    encoder = (group_order[:, None] * 4 + offsets[None]).flatten().contiguous()
    physical = repack_record_axis(
        encoder,
        paired_record_order(fruit_rate_modes()[0]),
        axis=0,
        record_channels=FRUIT_QSRT_RECORD_CHANNELS,
    ).contiguous()
    return encoder, physical, group_scores.contiguous()


def _functional_metric(
    rows: FruitCalibrationRows,
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return functional_sse_by_request(
        reference,
        candidate,
        rows.gates,
        rows.document_ids,
        rows.requests,
    )


def _metric_summary(
    metric: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, object]:
    sse, energy, counts = metric
    sse_total = float(sse.double().sum())
    energy_total = float(energy.double().sum())
    return {
        "routed_sse": sse_total,
        "reference_energy": energy_total,
        "normalized_sse": sse_total / energy_total if energy_total > 0 else None,
        "documents": int(torch.count_nonzero(counts)),
        "rows": int(counts.sum()),
    }


def _calibration_support(
    calibration: FruitExpertCalibration,
) -> tuple[str | None, dict[str, object]]:
    fit_documents = int(torch.unique(calibration.fit.document_ids).numel())
    confirmation_documents = int(
        torch.unique(calibration.confirmation.document_ids).numel()
    )
    fit_effective_rows = (
        weighted_effective_sample_size(calibration.fit.gates.float().square())
        if calibration.fit.row_count
        else 0.0
    )
    evidence: dict[str, object] = {
        "fit_documents": fit_documents,
        "minimum_fit_documents": PHASE1_MIN_FIT_DOCUMENTS,
        "confirmation_documents": confirmation_documents,
        "minimum_confirmation_documents": PHASE1_MIN_CONFIRMATION_DOCUMENTS,
        "fit_effective_rows": fit_effective_rows,
        "minimum_fit_effective_rows": FRUIT_CALIBRATION_MIN_FIT_EFFECTIVE_ROWS,
    }
    if fit_documents == 0:
        reason = "no routed fit rows; deterministic R0/R0 fallback"
    elif fit_documents < PHASE1_MIN_FIT_DOCUMENTS:
        reason = "insufficient fit-document support; deterministic R0/R0 fallback"
    elif fit_effective_rows < FRUIT_CALIBRATION_MIN_FIT_EFFECTIVE_ROWS:
        reason = "insufficient effective fit rows; deterministic R0/R0 fallback"
    elif confirmation_documents < PHASE1_MIN_CONFIRMATION_DOCUMENTS:
        reason = (
            "insufficient confirmation-document support; deterministic R0/R0 fallback"
        )
    else:
        reason = None
    return reason, evidence


def _encode_calibrated_expert(
    sources: Mapping[MatrixName, torch.Tensor],
    calibration: FruitExpertCalibration,
    *,
    layer: int,
    expert: int,
    device: torch.device,
    quantizer_module: ModuleType | object,
) -> FruitExpertEncoding:
    if calibration.layer != layer or calibration.expert != expert:
        raise ValueError("Fruit calibration assignment does not match the expert")
    fallback_reason, support_evidence = _calibration_support(calibration)
    if fallback_reason is not None:
        return _encode_identity_expert(
            sources,
            layer=layer,
            expert=expert,
            device=device,
            quantizer_module=quantizer_module,
            global_h13=calibration.global_h13,
            fallback_evidence=support_evidence,
            force_r0_reason=fallback_reason,
            calibration_fingerprint=calibration.fingerprint,
        )

    global_h13 = calibration.global_h13.to(
        device=device, dtype=torch.float32
    ).contiguous()
    source_middle = {
        split: _middle_rows(rows, sources["w1"], sources["w3"], device=device)
        for split, rows in (
            ("fit", calibration.fit),
            ("confirmation", calibration.confirmation),
            ("validation", calibration.validation),
        )
    }
    encoder_permutation, physical_permutation, group_scores = _activation_permutations(
        source_middle["fit"], calibration.fit.gates
    )
    upstream = _encode_matrix_family(
        (sources["w1"], sources["w3"]),
        ("w1", "w3"),
        encoder_permutation,
        hessians=(global_h13, global_h13),
        layer=layer,
        expert=expert,
        device=device,
        quantizer_module=quantizer_module,
    )

    downstream_by_r13: dict[
        int, dict[MatrixName, tuple[FruitMatrixCandidate, ...]]
    ] = {}
    h2_evidence: dict[str, object] = {}
    identity_h2 = torch.eye(
        FRUIT_QSRT_GEOMETRY.intermediate_channels,
        dtype=torch.float32,
        device=device,
    )
    fit_middle_by_r13: dict[int, torch.Tensor] = {}
    for r13 in range(3):
        middle = _middle_rows(
            calibration.fit,
            upstream["w1"][r13].reconstruction,
            upstream["w3"][r13].reconstruction,
            device=device,
        )
        fit_middle_by_r13[r13] = middle
        _, h2, evidence = build_expert_hessians(
            calibration.fit.inputs,
            calibration.fit.gates,
            middle,
            global_h13=global_h13,
            global_h2=identity_h2,
            device=device,
        )
        h2 = h2.index_select(0, encoder_permutation).index_select(
            1, encoder_permutation
        )
        downstream_by_r13[r13] = _encode_matrix_family(
            (sources["w2"],),
            ("w2",),
            encoder_permutation,
            hessians=(h2,),
            layer=layer,
            expert=expert,
            device=device,
            quantizer_module=quantizer_module,
        )
        h2_evidence[f"R{r13}"] = evidence

    rows_by_split = {
        "fit": calibration.fit,
        "confirmation": calibration.confirmation,
        "validation": calibration.validation,
    }
    reference_outputs = {
        split: middle @ sources["w2"].T for split, middle in source_middle.items()
    }
    middle_by_mode_split: dict[tuple[int, str], torch.Tensor] = {}
    for r13 in range(3):
        middle_by_mode_split[(r13, "fit")] = fit_middle_by_r13[r13]
        for split in ("confirmation", "validation"):
            middle_by_mode_split[(r13, split)] = _middle_rows(
                rows_by_split[split],
                upstream["w1"][r13].reconstruction,
                upstream["w3"][r13].reconstruction,
                device=device,
            )

    modes = tuple((r13, r2) for r13 in range(3) for r2 in range(3))
    metrics: dict[
        str, dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ] = {split: {} for split in rows_by_split}
    for r13, r2 in modes:
        down = downstream_by_r13[r13]["w2"][r2].reconstruction
        for split, rows in rows_by_split.items():
            candidate_output = middle_by_mode_split[(r13, split)] @ down.T
            metrics[split][(r13, r2)] = _functional_metric(
                rows,
                reference_outputs[split],
                candidate_output,
            )

    fit_counts = metrics["fit"][(0, 0)][2]
    confirmation_counts = metrics["confirmation"][(0, 0)][2]
    for split, expected_counts in (
        ("fit", fit_counts),
        ("confirmation", confirmation_counts),
    ):
        if any(
            not torch.equal(metric[2], expected_counts)
            for metric in metrics[split].values()
        ):
            raise ValueError(f"Fruit {split} metric support changed across formats")
    decision = select_phase1_rate_pair(
        {mode: metrics["fit"][mode][0] for mode in modes},
        {mode: metrics["confirmation"][mode][0] for mode in modes},
        fit_counts=fit_counts,
        confirmation_counts=confirmation_counts,
        modes=modes,
        seed=deterministic_expert_seed(layer, expert),
    )
    r13, r2 = decision.selected
    selected_mode = (r13, r2)
    baseline_validation = metrics["validation"][(0, 0)][0].double().sum()
    selected_validation = metrics["validation"][selected_mode][0].double().sum()
    validation_improvement = (
        1.0 - float(selected_validation / baseline_validation)
        if float(baseline_validation) > 0
        else None
    )
    selection = {
        "policy": "activation_coupled_functional_sse_v1",
        "calibrated_activations": True,
        "hessian_policy": "global_fit_h13_candidate_conditional_expert_h2",
        "permutation_policy": "gate_square_post_situ_energy_h2_reverse",
        "calibration_fingerprint": calibration.fingerprint,
        "proposed": {
            "r13": decision.proposed_r13,
            "r2": decision.proposed_r2,
        },
        "selected": {
            "r13": decision.selected_r13,
            "r2": decision.selected_r2,
        },
        "accepted": decision.accepted,
        "reason": decision.reason,
        "fit_documents": decision.fit_documents,
        "confirmation_documents": decision.confirmation_documents,
        "confirmation_relative_improvement": (
            decision.confirmation_relative_improvement
        ),
        "confirmation_ci95": list(decision.confirmation_ci95),
        "bootstrap_replicates_valid": decision.bootstrap_replicates_valid,
        "group_score_min": float(group_scores.min().cpu()),
        "group_score_max": float(group_scores.max().cpu()),
        "candidate_conditional_h2": h2_evidence,
        "functional_metrics": {
            split: {
                f"R13={mode[0]},R2={mode[1]}": _metric_summary(metric)
                for mode, metric in sorted(values.items())
            }
            for split, values in metrics.items()
        },
        "validation_selected_vs_r0_relative_improvement": validation_improvement,
    }
    return _finalize_expert(
        sources,
        {
            "w1": upstream["w1"][r13],
            "w3": upstream["w3"][r13],
            "w2": downstream_by_r13[r13]["w2"][r2],
        },
        encoder_permutation,
        physical_permutation,
        layer=layer,
        expert=expert,
        r13=r13,
        r2=r2,
        selection=selection,
    )


def encode_fruit_expert(
    store: FruitMatrixStore,
    *,
    layer: int,
    expert: int,
    device: torch.device,
    hessians: Mapping[MatrixName, torch.Tensor] | None = None,
    calibration: FruitExpertCalibration | None = None,
    quantizer_module: ModuleType | object | None = None,
) -> FruitExpertEncoding:
    """Encode one Fruit expert with diagnostic or document-disjoint calibration."""

    if device.type != "cuda":
        raise ValueError("Fruit QSRT encoding requires a CUDA device")
    if hessians is not None and calibration is not None:
        raise ValueError("Fruit expert cannot mix direct Hessians and calibration")
    sources = {
        matrix: store.load_matrix(layer, expert, matrix, device=device)
        for matrix in FRUIT_QSRT_MATRICES
    }
    if quantizer_module is None:
        raise ValueError(
            "quantizer_module is required; load it with "
            "kquant.exl3_loader.load_qsrt_encoder and install the SQG quantizer"
        )
    if calibration is not None:
        return _encode_calibrated_expert(
            sources,
            calibration,
            layer=layer,
            expert=expert,
            device=device,
            quantizer_module=quantizer_module,
        )
    if hessians is not None:
        raise ValueError(
            "direct Fruit Hessians are unsupported; provide an authenticated "
            "document-disjoint calibration capture"
        )
    return _encode_identity_expert(
        sources,
        layer=layer,
        expert=expert,
        device=device,
        quantizer_module=quantizer_module,
    )


def stack_fruit_layer(
    layer: int, encodings: Sequence[FruitExpertEncoding]
) -> FruitLayerArtifact:
    """Stack a unique ordered set of expert encodings into one artifact shard."""

    values = tuple(encodings)
    if not values:
        raise ValueError("Fruit layer artifact requires at least one expert")
    if any(value.layer != layer for value in values):
        raise ValueError("Fruit layer artifact cannot mix layer assignments")
    expert_ids = tuple(value.expert for value in values)
    if len(set(expert_ids)) != len(expert_ids):
        raise ValueError("Fruit layer artifact expert IDs must be unique")
    if tuple(sorted(expert_ids)) != expert_ids:
        raise ValueError("Fruit layer artifact expert IDs must be sorted")

    per_expert = [value.artifact_tensors() for value in values]
    tensors: dict[str, torch.Tensor] = {}
    for name in FRUIT_QSRT_ARTIFACT_TENSORS:
        parts = [item[name] for item in per_expert]
        axis = 1 if name == "w13_trellis" else 0
        tensors[name] = torch.cat(parts, dim=axis).contiguous()
    return FruitLayerArtifact(
        layer=layer,
        expert_ids=expert_ids,
        tensors=tensors,
        experts=tuple(value.manifest() for value in values),
    )
