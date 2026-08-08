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

from kquant.exl3_reference import (
    CODEBOOK_SQG_XOR_CHEB_T12,
    decode_qsrt_weight,
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
from kquant.sqg_e4m3 import sqg_xor_cheb_t12_bytes
from kquant.tp_simulator import comparison_metrics

FRUIT_QSRT_SCHEMA = "kquant_fruit_qsrt_pairs_v1"
FRUIT_QSRT_PROFILE_ID = 1
FRUIT_QSRT_CODEBOOK = CODEBOOK_SQG_XOR_CHEB_T12
FRUIT_QSRT_MATRICES: tuple[MatrixName, ...] = ("w1", "w3", "w2")
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

    reconstruction = _canonical_reconstruction(
        decoded_physical, source, matrix, physical_permutation
    )
    canonical_metrics = comparison_metrics(candidate.reconstruction, reconstruction)
    if not all(math.isfinite(value) for value in canonical_metrics.values()):
        raise ValueError("Fruit canonical reconstruction did not close")
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


def encode_fruit_expert(
    store: FruitMatrixStore,
    *,
    layer: int,
    expert: int,
    device: torch.device,
    hessians: Mapping[MatrixName, torch.Tensor] | None = None,
    quantizer_module: ModuleType | object | None = None,
) -> FruitExpertEncoding:
    """Encode one real Fruit expert and select R13/R2 by normalized weight SSE.

    The selection policy is intentionally diagnostic and weight-only. A future
    corpus-calibrated caller may supply dense Hessians, but must not relabel this
    selector as activation-calibrated without changing the artifact evidence.
    """

    if device.type != "cuda":
        raise ValueError("Fruit QSRT encoding requires a CUDA device")
    sources = {
        matrix: store.load_matrix(layer, expert, matrix, device=device)
        for matrix in FRUIT_QSRT_MATRICES
    }
    encoder_permutation, physical_permutation, group_scores = fruit_weight_permutations(
        sources["w1"], sources["w3"], sources["w2"]
    )
    resolved_hessians: dict[MatrixName, torch.Tensor] = {}
    supplied = {} if hessians is None else dict(hessians)
    if set(supplied) - set(FRUIT_QSRT_MATRICES):
        raise ValueError("Fruit Hessian map contains an unknown matrix")
    for matrix in FRUIT_QSRT_MATRICES:
        dimension = (
            FRUIT_QSRT_GEOMETRY.intermediate_channels
            if matrix == "w2"
            else FRUIT_QSRT_GEOMETRY.hidden_channels
        )
        value = supplied.get(matrix)
        if value is None:
            value = torch.eye(dimension, dtype=torch.float32, device=device)
        else:
            value = value.to(device=device, dtype=torch.float32).contiguous()
        if tuple(value.shape) != (dimension, dimension):
            raise ValueError(
                f"{matrix} Hessian must have shape {(dimension, dimension)}"
            )
        if matrix == "w2":
            value = value.index_select(0, encoder_permutation).index_select(
                1, encoder_permutation
            )
        resolved_hessians[matrix] = value

    if quantizer_module is None:
        raise ValueError(
            "quantizer_module is required; load it with "
            "kquant.exl3_loader.load_qsrt_encoder and install the SQG quantizer"
        )

    upstream = _encode_matrix_family(
        (sources["w1"], sources["w3"]),
        ("w1", "w3"),
        encoder_permutation,
        hessians=(resolved_hessians["w1"], resolved_hessians["w3"]),
        layer=layer,
        expert=expert,
        device=device,
        quantizer_module=quantizer_module,
    )
    downstream = _encode_matrix_family(
        (sources["w2"],),
        ("w2",),
        encoder_permutation,
        hessians=(resolved_hessians["w2"],),
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
    r13, r2 = (int(r13_scores.argmin().cpu()), int(r2_scores.argmin().cpu()))
    selected = {
        "w1": candidates["w1"][r13],
        "w3": candidates["w3"][r13],
        "w2": candidates["w2"][r2],
    }
    finalized = {
        matrix: _finalize_candidate(
            selected[matrix],
            sources[matrix],
            encoder_permutation,
            physical_permutation,
        )
        for matrix in FRUIT_QSRT_MATRICES
    }
    selection = {
        "policy": "weight_normalized_sse_v1",
        "calibrated_activations": False,
        "hessian_policy": "identity" if hessians is None else "caller_supplied",
        "r13_scores": [float(value) for value in r13_scores.cpu().tolist()],
        "r2_scores": [float(value) for value in r2_scores.cpu().tolist()],
        "group_score_min": float(group_scores.min().cpu()),
        "group_score_max": float(group_scores.max().cpu()),
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
