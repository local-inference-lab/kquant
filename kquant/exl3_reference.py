"""Pure-PyTorch reference operations for EXL3 procedural codebooks."""

from __future__ import annotations

import math

import torch

from kquant.sqg_e4m3 import SQG_NORMAL_E4M3, sqg_e4m3_codebook


TILE_VALUES = 256
HADAMARD_BLOCK = 128
CODEBOOK_MCG = "mcg"
CODEBOOK_MUL1_E4M3 = "mul1-e4m3"
CODEBOOK_SQG_NORMAL_E4M3 = SQG_NORMAL_E4M3
EXL3_CODEBOOKS = (
    CODEBOOK_MCG,
    CODEBOOK_MUL1_E4M3,
    CODEBOOK_SQG_NORMAL_E4M3,
)
MCG_MULT = 0xCBAC1FED
MUL1_MULT = 0x83DCD12D
MUL1_ACC_BITS = 0x6400
MUL1_INV_BITS = 0x1EEE
MUL1_BIAS_BITS = 0xC931


def _validate_bits(bits: int) -> None:
    if isinstance(bits, bool) or not isinstance(bits, int) or bits not in (2, 3, 4):
        raise ValueError("the TP12 reference supports K=2, K=3, or K=4")


def reconstruct_trellis_states(
    edge_symbols: torch.Tensor, bits: int
) -> torch.Tensor:
    """Recover cyclic 16-bit trellis states from their stored K-bit edges.

    A state is the low 16 bits of ``(previous_state << K) | edge``.  EXL's
    256-symbol tile is cyclic, so state zero also incorporates the final edge
    symbols in the tile.
    """

    _validate_bits(bits)
    if edge_symbols.ndim < 1 or edge_symbols.shape[-1] != TILE_VALUES:
        raise ValueError("edge_symbols must have a final dimension of 256")
    if edge_symbols.dtype == torch.bool or edge_symbols.is_floating_point():
        raise TypeError("edge_symbols must use an integer dtype")
    edges = edge_symbols.to(dtype=torch.int64) & ((1 << bits) - 1)
    # Unroll the cyclic recurrence in state space.  At position i, edge i is
    # in the low K bits, edge i-1 is above it, and so on.  Only ceil(16 / K)
    # edges can survive the low-16-bit mask.  Expressing that closed form with
    # 4--8 rolls avoids launching one tensor operation for every one of the
    # 256 symbols when this reference check runs on CUDA.
    states = torch.zeros_like(edges)
    for lag in range(math.ceil(16 / bits)):
        states |= torch.roll(edges, shifts=lag, dims=-1) << (lag * bits)
    return (states & 0xFFFF).to(dtype=torch.int16).contiguous()


def decode_mcg_states(states: torch.Tensor) -> torch.Tensor:
    """Decode 16-bit EXL MCG states exactly to their FP16 codebook values."""

    if states.dtype == torch.bool or states.is_floating_point():
        raise TypeError("trellis states must use an integer dtype")
    values = states.to(dtype=torch.int64) & 0xFFFF
    product = (values * MCG_MULT) & 0xFFFFFFFF
    # PTX lop3(a, 0x8fff8fff, 0x3b603b60, 0x6a) has the Boolean
    # expression c XOR (a AND b), with a as the most-significant LUT input.
    transformed = 0x3B603B60 ^ (product & 0x8FFF8FFF)
    low = (transformed & 0xFFFF).to(torch.int16).view(torch.float16)
    high = ((transformed >> 16) & 0xFFFF).to(torch.int16).view(torch.float16)
    return (low + high).to(torch.float16).contiguous()


def mul1_byte_sums(
    states: torch.Tensor, *, multiplier: int = MUL1_MULT
) -> torch.Tensor:
    """Return MUL1's four-byte statistic for a configurable u32 multiplier."""

    if states.dtype == torch.bool or states.is_floating_point():
        raise TypeError("trellis states must use an integer dtype")
    if (
        isinstance(multiplier, bool)
        or not isinstance(multiplier, int)
        or multiplier < 0
        or multiplier > 0xFFFFFFFF
    ):
        raise ValueError("multiplier must be a uint32 integer")
    values = states.to(dtype=torch.int64) & 0xFFFF
    product = (values * multiplier) & 0xFFFFFFFF
    return (
        (product & 0xFF)
        + ((product >> 8) & 0xFF)
        + ((product >> 16) & 0xFF)
        + ((product >> 24) & 0xFF)
    ).contiguous()


def decode_mul1_states(
    states: torch.Tensor, *, multiplier: int = MUL1_MULT
) -> torch.Tensor:
    """Decode 16-bit EXL MUL1 states to the CUDA decoder's FP16 values.

    EXL multiplies the state by ``0x83DCD12D``, sums the four product bytes
    with ``dp4a``, reinterprets ``0x6400 + bytesum`` as FP16, and performs one
    FP16 fused multiply-add using literal half constants. Evaluating those
    exact half inputs in float64 and rounding once reproduces that half FMA.
    """

    byte_sum = mul1_byte_sums(states, multiplier=multiplier)
    accumulator = (byte_sum + MUL1_ACC_BITS).to(torch.int16).view(torch.float16)
    inv = torch.tensor(MUL1_INV_BITS, dtype=torch.int16, device=states.device).view(
        torch.float16
    )
    bias = torch.tensor(
        MUL1_BIAS_BITS - 0x10000, dtype=torch.int16, device=states.device
    ).view(torch.float16)
    return (accumulator.double() * inv.double() + bias.double()).to(
        torch.float16
    ).contiguous()


def decode_mul1_e4m3_states(states: torch.Tensor) -> torch.Tensor:
    """Decode MUL1 states and round each reconstruction to finite E4M3.

    This mirrors the production candidate encoder: the exact EXL MUL1 half
    result is converted with round-to-nearest-even into the E4M3 finite
    alphabet, then represented as FP16 for reference arithmetic.
    """

    return (
        decode_mul1_states(states)
        .to(dtype=torch.float8_e4m3fn)
        .to(dtype=torch.float16)
        .contiguous()
    )


def _decode_codebook_states(
    states: torch.Tensor,
    codebook: str,
    *,
    bits: int | None = None,
) -> torch.Tensor:
    if codebook == CODEBOOK_MCG:
        return decode_mcg_states(states)
    if codebook == CODEBOOK_MUL1_E4M3:
        return decode_mul1_e4m3_states(states)
    if codebook == CODEBOOK_SQG_NORMAL_E4M3:
        if bits is None:
            raise ValueError("SQG reconstruction requires an explicit trellis rate")
        _validate_bits(bits)
        indices = (states.to(dtype=torch.int64) & 0xFFFF).long()
        values = sqg_e4m3_codebook(
            bits, "normal", device=states.device, dtype=torch.float16
        )
        return values.index_select(0, indices.flatten()).reshape_as(states)
    raise ValueError(
        f"unsupported EXL3 codebook {codebook!r}; expected one of {EXL3_CODEBOOKS}"
    )


def tensor_core_permutation(device: torch.device | str = "cpu") -> torch.Tensor:
    """Return the native EXL 16x16 row-major -> tensor-core element order."""

    permutation = [0] * TILE_VALUES
    for thread in range(32):
        row0 = (thread % 4) * 2
        row1 = row0 + 1
        row2 = row0 + 8
        row3 = row0 + 9
        column0 = thread // 4
        column1 = column0 + 8
        permutation[thread * 8 + 0] = row0 * 16 + column0
        permutation[thread * 8 + 1] = row1 * 16 + column0
        permutation[thread * 8 + 2] = row2 * 16 + column0
        permutation[thread * 8 + 3] = row3 * 16 + column0
        permutation[thread * 8 + 4] = row0 * 16 + column1
        permutation[thread * 8 + 5] = row1 * 16 + column1
        permutation[thread * 8 + 6] = row2 * 16 + column1
        permutation[thread * 8 + 7] = row3 * 16 + column1
    return torch.tensor(permutation, dtype=torch.long, device=device)


def decode_regularized_weight(
    states: torch.Tensor,
    *,
    codebook: str = CODEBOOK_MCG,
    codebook_values: torch.Tensor | None = None,
    bits: int | None = None,
) -> torch.Tensor:
    """Decode ``[K/16, N/16, 256]`` states to transformed ``[K, N]``."""

    if states.ndim != 3 or states.shape[-1] != TILE_VALUES:
        raise ValueError("states must have shape [K/16, N/16, 256]")
    if codebook_values is None:
        decoded = _decode_codebook_states(states, codebook, bits=bits).float()
    else:
        if codebook_values.ndim != 1 or codebook_values.numel() != 1 << 16:
            raise ValueError("custom codebook must contain exactly 65,536 values")
        if not codebook_values.is_floating_point():
            raise TypeError("custom codebook must be floating point")
        if not bool(torch.isfinite(codebook_values).all()):
            raise ValueError("custom codebook must be finite")
        indices = (states.to(dtype=torch.int64) & 0xFFFF).long()
        decoded = codebook_values.to(device=states.device).index_select(
            0, indices.flatten()
        ).reshape_as(states).float()
    inverse = torch.argsort(tensor_core_permutation(states.device))
    decoded = decoded.index_select(-1, inverse)
    k_tiles, n_tiles, _ = decoded.shape
    return (
        decoded.reshape(k_tiles, n_tiles, 16, 16)
        .permute(0, 2, 1, 3)
        .reshape(k_tiles * 16, n_tiles * 16)
        .contiguous()
    )


def decode_mixed_regularized_weight(
    states: torch.Tensor,
    *,
    rate_axis: str,
    tile_bits: tuple[int, ...],
    codebook: str,
) -> torch.Tensor:
    """Decode a K2/K3/K4 tile map to transformed ``[K, N]`` weights.

    Procedural MCG and MUL1 use one reconstruction function at every rate.
    SQG deliberately uses a different state labelling for K2, K3, and K4, so
    its physical tile rate is part of the numerical decode contract.
    """

    if states.ndim != 3 or states.shape[-1] != TILE_VALUES:
        raise ValueError("states must have shape [K/16, N/16, 256]")
    if rate_axis not in ("k", "n"):
        raise ValueError("rate_axis must be 'k' or 'n'")
    axis = 0 if rate_axis == "k" else 1
    if len(tile_bits) != states.shape[axis]:
        raise ValueError("tile_bits length does not match the selected rate axis")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value not in (2, 3, 4)
        for value in tile_bits
    ):
        raise ValueError("tile_bits must contain only integer K2, K3, or K4")
    if codebook != CODEBOOK_SQG_NORMAL_E4M3:
        return decode_regularized_weight(states, codebook=codebook)

    decoded_tiles = torch.empty_like(states, dtype=torch.float32)
    bits_tensor = torch.tensor(tile_bits, dtype=torch.long, device=states.device)
    for bits in sorted(set(tile_bits)):
        positions = torch.nonzero(bits_tensor == bits).flatten()
        selected = states.index_select(axis, positions)
        values = _decode_codebook_states(selected, codebook, bits=bits).float()
        decoded_tiles.index_copy_(axis, positions, values)

    inverse = torch.argsort(tensor_core_permutation(states.device))
    decoded_tiles = decoded_tiles.index_select(-1, inverse)
    k_tiles, n_tiles, _ = decoded_tiles.shape
    return (
        decoded_tiles.reshape(k_tiles, n_tiles, 16, 16)
        .permute(0, 2, 1, 3)
        .reshape(k_tiles * 16, n_tiles * 16)
        .contiguous()
    )


def normalized_hadamard(
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    size: int = HADAMARD_BLOCK,
) -> torch.Tensor:
    if size <= 0 or size & (size - 1):
        raise ValueError("Hadamard size must be a positive power of two")
    matrix = torch.ones((1, 1), dtype=dtype, device=device)
    while matrix.shape[0] < size:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix * (1.0 / math.sqrt(size))


def _blockwise_hadamard_left(weight: torch.Tensor) -> torch.Tensor:
    if weight.shape[0] % HADAMARD_BLOCK:
        raise ValueError("EXL K dimension must be divisible by 128")
    hadamard = normalized_hadamard(device=weight.device, dtype=weight.dtype)
    blocks = weight.reshape(-1, HADAMARD_BLOCK, weight.shape[1])
    return torch.matmul(hadamard, blocks).reshape_as(weight).contiguous()


def _blockwise_hadamard_right(weight: torch.Tensor) -> torch.Tensor:
    if weight.shape[1] % HADAMARD_BLOCK:
        raise ValueError("EXL N dimension must be divisible by 128")
    hadamard = normalized_hadamard(device=weight.device, dtype=weight.dtype)
    blocks = weight.reshape(weight.shape[0], -1, HADAMARD_BLOCK)
    return torch.matmul(blocks, hadamard).reshape_as(weight).contiguous()


def decode_exl3_weight(
    states: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    *,
    codebook: str = CODEBOOK_MCG,
    codebook_values: torch.Tensor | None = None,
    bits: int | None = None,
) -> torch.Tensor:
    """Decode stored states/scales to an EXL-oriented ``[K, N]`` weight."""

    weight = decode_regularized_weight(
        states, codebook=codebook, codebook_values=codebook_values, bits=bits
    )
    if suh.ndim != 1 or suh.numel() != weight.shape[0]:
        raise ValueError("suh length does not match the EXL K dimension")
    if svh.ndim != 1 or svh.numel() != weight.shape[1]:
        raise ValueError("svh length does not match the EXL N dimension")
    weight = _blockwise_hadamard_left(weight)
    weight *= suh.to(device=weight.device, dtype=weight.dtype).unsqueeze(1)
    weight = _blockwise_hadamard_right(weight)
    weight *= svh.to(device=weight.device, dtype=weight.dtype).unsqueeze(0)
    return weight.contiguous()


def decode_mixed_exl3_weight(
    states: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    *,
    rate_axis: str,
    tile_bits: tuple[int, ...],
    codebook: str = CODEBOOK_MCG,
) -> torch.Tensor:
    """Decode stored mixed-rate states/scales to an EXL ``[K, N]`` weight."""

    weight = decode_mixed_regularized_weight(
        states,
        rate_axis=rate_axis,
        tile_bits=tile_bits,
        codebook=codebook,
    )
    if suh.ndim != 1 or suh.numel() != weight.shape[0]:
        raise ValueError("suh length does not match the EXL K dimension")
    if svh.ndim != 1 or svh.numel() != weight.shape[1]:
        raise ValueError("svh length does not match the EXL N dimension")
    weight = _blockwise_hadamard_left(weight)
    weight *= suh.to(device=weight.device, dtype=weight.dtype).unsqueeze(1)
    weight = _blockwise_hadamard_right(weight)
    weight *= svh.to(device=weight.device, dtype=weight.dtype).unsqueeze(0)
    return weight.contiguous()
