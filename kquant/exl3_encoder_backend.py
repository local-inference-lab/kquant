"""QSRT offline encoder backend derived from ExLlamaV3's EXL3 quantizer.

This copy is maintained by kquant so QSRT does not require a patched
ExLlamaV3 checkout. The upstream implementation is MIT-licensed; see
THIRD_PARTY_NOTICES.md for attribution.
"""

import torch
import torch.nn.functional as F
import math
from ....ext import exllamav3_ext as ext
from ....util.progress import ProgressBar
from ....util.memory import free_mem, list_gpu_tensors
from ....util.hadamard import get_hadamard_dt
from ....util import cuda_sync_active
from ....util.tensor import save_tensor_image
from functools import lru_cache
import threading

# Constant
had_k, had_n = 128, 128
codebook_scale = 1.24371088

codebook_mcg_mult = 0xCBAC1FED
codebook_mul1_mult = 0x83DCD12D

@lru_cache
def tensor_core_perm(device):
    """
    Return the 16x16 tile permutation expected by the EXL3 tensor-core quantization kernels.

    The cached index maps row-major tile elements into the lane/interleave order used by the CUDA encoder.
    """
    perm_a = [0] * 256
    for t in range(32):
        r0 = (t % 4) * 2
        r1 = r0 + 1
        r2 = r0 + 8
        r3 = r0 + 9
        c0 = t // 4
        c1 = c0 + 8
        perm_a[t * 8 + 0] = r0 * 16 + c0
        perm_a[t * 8 + 1] = r1 * 16 + c0
        perm_a[t * 8 + 2] = r2 * 16 + c0
        perm_a[t * 8 + 3] = r3 * 16 + c0
        perm_a[t * 8 + 4] = r0 * 16 + c1
        perm_a[t * 8 + 5] = r1 * 16 + c1
        perm_a[t * 8 + 6] = r2 * 16 + c1
        perm_a[t * 8 + 7] = r3 * 16 + c1
    return torch.tensor(perm_a, dtype = torch.int, device = device)


@lru_cache
def tensor_core_perm_i(device):
    return torch.argsort(tensor_core_perm(device))


@lru_cache
def get_temp_buffers(device, K: int):
    """Allocate workspaces for the unmodified upstream EXL tile encoder.

    QSRT has a different packed-traceback workspace owned by
    :mod:`kquant.sqg_quantizer`.  Keeping the two contracts separate is what
    lets this backend run against a clean ExLlamaV3 checkout.
    """

    max_batch_size = 256
    if K >= 4:
        mp_count = torch.cuda.get_device_properties(device).multi_processor_count
        max_batch_size = max(256, 2 * mp_count)
    edges = 65536 >> K
    temp_costs = torch.zeros(
        (max_batch_size, 2, edges), dtype=torch.half, device=device
    )
    temp_edges = torch.zeros(
        (max_batch_size, 256, edges), dtype=torch.short, device=device
    )
    return temp_costs, temp_edges


def quantize_tiles(tiles, quant_args: dict):
    """
    Quantize a batch of 16x16 tiles on the current device.

    tiles is shaped (num_tiles, 256) in the kernel's expected element order. The CUDA extension returns both the
    reconstructed float tile values and the short encoded indices used later for packing.
    """
    tiles = tiles.contiguous()
    assert tiles.shape[1] == 256
    assert tiles.dtype == torch.float

    K = quant_args["K"]
    mcg = "mcg" in quant_args
    mul1 = "mul1" in quant_args
    # quantize_tiles_kernel writes all 256 reconstructed values and indices
    # for every tile.  Clearing these output-only buffers before each of the
    # many LDLQ launches adds a pair of needless memset kernels.
    quantized_tiles = torch.empty_like(tiles)
    quantized_idx = torch.empty_like(tiles, dtype = torch.short)
    temp_costs, temp_edges = get_temp_buffers(tiles.device, K)
    ext.quantize_tiles(
        tiles,
        quantized_tiles,
        quantized_idx,
        temp_costs,
        temp_edges,
        K,
        mcg,
        mul1,
    )
    return quantized_tiles, quantized_idx


@lru_cache
def get_quant_stream(device):
    return torch.cuda.Stream(device = device)


pinned_tiles: torch.Tensor | None = None
pinned_q_tiles: torch.Tensor | None = None
pinned_q_idx: torch.Tensor | None = None
def get_pinned(num_tiles: int):
    global pinned_tiles, pinned_q_tiles, pinned_q_idx
    if pinned_tiles is None or pinned_tiles.shape[0] < num_tiles:
        pinned_tiles = torch.empty((num_tiles, 256), device = "cpu", dtype = torch.float, pin_memory = True)
        pinned_q_tiles = torch.empty((num_tiles, 256), device = "cpu", dtype = torch.float, pin_memory = True)
        pinned_q_idx = torch.empty((num_tiles, 256), device = "cpu", dtype = torch.int16, pin_memory = True)
    return pinned_tiles[:num_tiles, :], pinned_q_tiles[:num_tiles, :], pinned_q_idx[:num_tiles, :]


def quantize_tiles_multigpu(tiles, quant_args: dict):
    """
    Quantize tiles across multiple GPUs using pinned host memory for asynchronous fan-out/fan-in.

    The input starts on the first device, is copied once to pinned CPU memory, split by device_ratios or evenly, and
    each GPU quantizes its slice on a per-device stream. Results are copied back through pinned memory and gathered
    on the first device.
    """
    devices = quant_args["devices"]
    if len(devices) == 1:
        return quantize_tiles(tiles, quant_args)

    # Get pinned buffers
    pin_tiles, pin_q_tiles, pin_q_idx = get_pinned(tiles.shape[0])

    # Copy input tiles to pinned memory. Input is always on the first device in the split
    copy_input_event = torch.cuda.Event(blocking = False)
    main_stream = get_quant_stream(devices[0])
    with torch.cuda.stream(main_stream):
        tiles = tiles.contiguous()
        pin_tiles.copy_(tiles, non_blocking = True)
        copy_input_event.record(main_stream)

    # Create split slices for input tiles, output tiles and output indices
    ratios = quant_args.get("device_ratios")
    if ratios:
        s = sum(ratios)
        split_sizes = [tiles.shape[0] * r / s for r in ratios]
        split_sizes = [round(s / 16) * 16 for s in split_sizes]
        split_sizes[-1] += tiles.shape[0] - sum(split_sizes)
    else:
        split_sizes = [tiles.shape[0] // len(devices)] * len(devices)
        split_sizes[-1] += tiles.shape[0] - sum(split_sizes)

    # Account for negative splits (edge case if too many GPUs and/or tensor too small)
    for i in range(len(split_sizes) - 2, -1, -1):
        if split_sizes[i + 1] < 0:
            split_sizes[i] += split_sizes[i + 1]
            split_sizes[i + 1] = 0

    pin_split_tiles = torch.split(pin_tiles, split_sizes)
    pin_split_q_tiles = torch.split(pin_q_tiles, split_sizes)
    pin_split_q_idx = torch.split(pin_q_idx, split_sizes)

    slice_done_events = []
    for i, device in enumerate(devices):

        stream = get_quant_stream(device)
        with torch.cuda.stream(stream):

            # Wait for input in host memory
            if i > 0:
                stream.wait_event(copy_input_event)

            if split_sizes[i] > 0:

                # Asynchronously copy the slice from the pinned buffer to device memory
                dev_tiles = pin_split_tiles[i].to(device, non_blocking = True)

                # Preallocate output tensors on the device.
                dev_q_tiles = torch.empty_like(dev_tiles, device = device)
                dev_q_idx = torch.empty_like(dev_tiles, dtype = torch.short, device = device)

                # Work buffers
                K = quant_args["K"]
                mcg = "mcg" in quant_args
                mul1 = "mul1" in quant_args
                temp_costs, temp_edges = get_temp_buffers(device, K)
                ext.quantize_tiles(
                    dev_tiles,
                    dev_q_tiles,
                    dev_q_idx,
                    temp_costs,
                    temp_edges,
                    K,
                    mcg,
                    mul1,
                )

                # Async copy back to pinned memory
                pin_split_q_tiles[i].copy_(dev_q_tiles, non_blocking = True)
                pin_split_q_idx[i].copy_(dev_q_idx, non_blocking = True)

            # Finished slice
            evt = torch.cuda.Event(blocking = False)
            slice_done_events.append(evt)
            evt.record(stream)

    # Copy pinned buffers to original device
    with torch.cuda.stream(main_stream):
        for evt in slice_done_events:
            main_stream.wait_event(evt)
        q_tiles = torch.empty_like(tiles, device = devices[0])
        q_idx = torch.empty_like(tiles, dtype = torch.short, device = devices[0])
        q_tiles.copy_(pin_q_tiles, non_blocking = True)
        q_idx.copy_(pin_q_idx, non_blocking = True)

    return q_tiles, q_idx


def quantize_tiles_multigpu_sync(tiles, quant_args: dict):
    """
    Simpler synchronized multi-GPU tile quantization path.

    This variant explicitly copies tile chunks to each device, synchronizes around the work, then gathers results
    back to the first device. It is easier to reason about than the pinned asynchronous path but offers less overlap.
    """
    devices = quant_args["devices"]
    if len(devices) == 1:
        return quantize_tiles(tiles, quant_args)

    tiles = tiles.contiguous()

    split_sizes = [tiles.shape[0] // len(devices)] * len(devices)
    split_sizes[-1] += tiles.shape[0] - sum(split_sizes)
    split_tiles = torch.split(tiles, split_sizes)
    tiles_per_device = [chunk.to(device) for chunk, device in zip(split_tiles, devices)]
    torch.cuda.synchronize()

    q_tiles_per_device = []
    q_idx_per_device = []
    for dev_tiles, device in zip(tiles_per_device, devices):
        with torch.cuda.stream(get_quant_stream(device)):
            dev_q_tiles, dev_q_idx = quantize_tiles(dev_tiles, quant_args)
            q_tiles_per_device.append(dev_q_tiles)
            q_idx_per_device.append(dev_q_idx)

    for device in devices:
        torch.cuda.synchronize(device)

    q_tiles_per_device = [x.to(devices[0]) for x in q_tiles_per_device]
    q_idx_per_device = [x.to(devices[0]) for x in q_idx_per_device]
    quantized_tiles = torch.cat(q_tiles_per_device, dim = 0)
    quantized_idx = torch.cat(q_idx_per_device, dim = 0)
    return quantized_tiles, quantized_idx


def preapply_had_l(x: torch.Tensor, had_dim):
    k, n = x.shape
    x_dtype = x.dtype
    x = x.to(torch.float)
    had = get_hadamard_dt(had_dim, x.device, x.dtype, 1 / math.sqrt(had_dim))
    x = (had @ x.view(-1, had_dim, n)).view(k, n)
    x = x.to(x_dtype)
    return x


def preapply_had_r(x: torch.Tensor, had_dim):
    k, n = x.shape
    x_dtype = x.dtype
    x = x.to(torch.float)
    had = get_hadamard_dt(had_dim, x.device, x.dtype, 1 / math.sqrt(had_dim))
    x = (x.view(k, -1, had_dim) @ had).view(k, n)
    x = x.to(x_dtype)
    return x


def blockwise_preapply_had_l_(x: torch.Tensor, had_dim):
    k, n = x.shape
    assert k % had_dim == 0
    assert x.dtype == torch.float
    had = get_hadamard_dt(had_dim, x.device, x.dtype, 1 / math.sqrt(had_dim))
    num_blocks = k // had_dim
    for i in range(num_blocks):
        start = i * had_dim
        end = start + had_dim
        block = x[start:end, :]  # shape (had_dim, n)
        block_transformed = had @ block.view(had_dim, n)
        x[start:end, :] = block_transformed


def blockwise_preapply_had_r_(x: torch.Tensor, had_dim):
    k, n = x.shape
    assert n % had_dim == 0
    assert x.dtype == torch.float
    had = get_hadamard_dt(had_dim, x.device, x.dtype, 1 / math.sqrt(had_dim))
    num_blocks = n // had_dim
    for i in range(num_blocks):
        start = i * had_dim
        end = start + had_dim
        block = x[:, start:end]  # shape (k, had_dim)
        block_transformed = block @ had
        x[:, start:end] = block_transformed


def block_ldl(H: torch.Tensor, b: int, quant_args: dict, verbose: bool):

    n, _ = H.shape
    assert (n % b == 0)
    m = n // b

    # Cholesky factorization: H = L @ L.T
    # Try on GPU first
    num_cholesky_retries = 0
    retry_cpu = False
    while True:
        try:
            L = torch.linalg.cholesky(H)
            # H is not needed after this, move to CPU. Then overwrite H's GPU storage with L, since we can't otherwise
            # free up that VRAM as the tensor is referenced by the parent frame
            H_cpu = H.cpu()
            H.copy_(L)  # VRAM copy, tiny overhead
            L = H
            H = H_cpu
            break

        except torch._C._LinAlgError as e:
            num_cholesky_retries += 1
            if num_cholesky_retries > 10:
                print(" ## Cholesky decomp. failed, number of retries exceeded")
                raise e
            print(f" !! Cholesky decomp. failed, increasing diagonal damping, attempt {num_cholesky_retries}/10")
            H.diagonal().add_(2.0 * quant_args.get("sigma_reg", 0.025) * H.diagonal().mean())
            continue

        # Fall back on CPU factorization
        except Exception as e:
            if e.__class__.__name__ == "OutOfMemoryError" or "CUDA out of memory" in str(e) or "HIP out of memory" in str(e):
                retry_cpu = True
                break
            else:
                raise e

    if retry_cpu:
        print(f" !! Out of memory on {str(H.device)}, trying CPU fallback")
        free_mem()
        H_cpu = H.cpu()
        L_cpu = torch.linalg.cholesky(H_cpu)
        # This is ugly, but overwrite H in VRAM to avoid allocating a new tensor, then replace reference with CPU copy
        H.copy_(L_cpu)
        del L_cpu
        L = H
        H = H_cpu

    # Get blocks along diagonal of L: DL.shape = (m, b, b)
    DL = torch.diagonal(L.reshape(m, b, m, b), dim1 = 0, dim2 = 2).permute(2, 0, 1)

    # Compute D as D[i] = DL[i] @ DL[i].T for each diagonal block i (don't actually end up needing this)
    # D = DL @ DL.transpose(1, 2)

    # Invert each diagonal block
    DL = torch.linalg.inv(DL)

    # Multiply each block's column with its inverse
    L = L.view(n, m, b)
    for i in range(m):
        L[:, i, :] = L[:, i, :] @ DL[i, :, :]  # TODO: Could maybe be L[m * b:, i, :]?
    L = L.reshape(n, n).contiguous()

    # Insert block identity matrices along the diagonal.
    # TODO: Figure out if this is necessary. Diagonal blocks should already be identities after previous step
    L_block = L.view(m, b, m, b).permute(0, 2, 1,3)
    dr = torch.arange(m)
    L_block[dr, dr] = torch.stack([torch.eye(b, device = L.device, dtype = H.dtype)] * m)

    return L, H  # , D.to(DL.device)


def ldlq(
    weight: torch.Tensor,
    L: torch.Tensor,
    quant_args: dict,
    pb: ProgressBar | None = None
):
    """
    :param weight:
        Input weights, shape (k, n). If device is "cpu", result is collected on CPU as well, saving a bunch of
        VRAM but adding a little PCIe overhead and many sync points

    :param L:
        LDL decomposition of regularized H

    :param quant_args:
        dict:
         - K: bitrate
         - buf_size_k: buffer size for LDLQ, along k

    :param pb:
        Optional ProgressPar to update, k // 16 steps

    :return:
        tuple:
         - quantized weight, shape (k, n)
         - indices (unpacked), shape (k // 16, n // 16, 256), uint16_t
    """

    devices = quant_args["devices"]
    for device in devices:
        torch.cuda.synchronize(device)
    main_stream = get_quant_stream(devices[0])
    with torch.cuda.stream(main_stream):

        devices = quant_args["devices"]
        device = L.device
        assert device == torch.device(devices[0])

        buffer_device = weight.device
        size_k, size_n = weight.shape  # Row-major
        assert size_k % 16 == 0
        assert size_n % 128 == 0
        tiles_k = size_k // 16
        tiles_n = size_n // 16

        buf_size_k = max(quant_args.get("buf_size_k", 128), 16)
        assert buf_size_k % 16 == 0
        assert size_n % buf_size_k == 0

        p_row = 0

        # Work buffers
        prod_cache = torch.zeros((size_k, size_n), dtype = torch.float, device = device)
        weight_q = torch.zeros((size_k, size_n), dtype = torch.float, device = buffer_device)
        encoded = torch.zeros((tiles_k, tiles_n, 256), dtype = torch.short, device = buffer_device)

        for j in range(size_k, 0, -buf_size_k):
            i = j - buf_size_k

            # Current span is rows i:j
            b_weight = weight[i:j].to(device)
            b_weight_q = weight_q[i:j] if device == buffer_device else \
                torch.zeros_like(weight_q[i:j], device = device)
            b_encoded = encoded[i // 16 : j // 16] if device == buffer_device else \
                torch.zeros_like(encoded[i // 16 : j // 16], device = device)
            b_prod_cache = prod_cache[i:j]
            b_L = L[i:j]

            # Iterate over rows of blocks in current span
            for bj in range(buf_size_k, 0, -16):
                bi = bj - 16

                # Error so far for the current span
                bb_err = b_weight[bj:] - b_weight_q[bj:]

                # Corresponding slice of LDL decomposition of H
                bb_L = b_L[bj:, i + bi:i + bj]

                # Input tiles for quantization
                compensation_term = b_prod_cache[bi:bj]
                compensation_term.addmm_(bb_L.T, bb_err,  alpha = 1.0, beta = 1.0)
                rows = b_weight[bi:bj] + compensation_term

                tiles = rows.reshape(16, tiles_n, 16).permute(1, 0, 2).reshape(tiles_n, 256)

                # Pre-permute to tensor core layout
                tiles = tiles[:, tensor_core_perm(device)]

                # Quantize
                quant_w, quant_i = quantize_tiles_multigpu(tiles, quant_args)

                # Undo permutation on reconstructed tiles, but keep indices in tensor core layout
                quant_w = quant_w[:, tensor_core_perm_i(device)]

                # Store result
                quant_w = quant_w.reshape(tiles_n, 16, 16).permute(1, 0, 2).reshape(16, size_n)
                b_weight_q[bi:bj] = quant_w
                b_encoded[bi // 16 : bj // 16] = quant_i.unsqueeze(0)

                # Update progress
                if pb:
                    p_row += 1
                    pb.update(p_row)

            # Collect output
            if device != buffer_device:
                weight_q[i:j] = b_weight_q.to(buffer_device)
                encoded[i // 16 : j // 16] = b_encoded.to(buffer_device)

            # Cache error term for the rest of the matrix
            b_err = b_weight - b_weight_q
            prod_cache.addmm_(b_L.T, b_err, alpha = 1.0, beta = 1.0)

        for device in devices:
            torch.cuda.synchronize(device)

    return weight_q, encoded


def mixed_rate_spec(size_k: int, size_n: int, quant_args: dict):
    """Validate and normalize a heterogeneous K2/K3/K4 tile-rate map.

    ``mixed_rate_axis`` names the logical EXL tile axis whose entries receive
    distinct trellis rates.  ``mixed_tile_bits`` contains one K value per tile
    on that axis.  The rate map changes only the tile quantizer; the LDLQ
    recursion remains one dense traversal over the complete input Hessian.
    """

    rate_axis = quant_args.get("mixed_rate_axis")
    if rate_axis not in ("k", "n"):
        raise ValueError("mixed_rate_axis must be 'k' or 'n'")
    if size_k % 16 or size_n % 16:
        raise ValueError("mixed-rate LDLQ input must be 16x16 tile aligned")
    raw_tile_bits = quant_args.get("mixed_tile_bits")
    if raw_tile_bits is None:
        raise ValueError("mixed_tile_bits is required for mixed-rate LDLQ")
    tile_bits = tuple(raw_tile_bits)
    rate_tiles = size_k // 16 if rate_axis == "k" else size_n // 16
    if len(tile_bits) != rate_tiles:
        raise ValueError(
            f"mixed_tile_bits has {len(tile_bits)} entries; expected {rate_tiles}"
        )
    if any(
        isinstance(bits, bool) or not isinstance(bits, int) or bits not in (2, 3, 4)
        for bits in tile_bits
    ):
        raise ValueError("mixed-rate LDLQ supports only integer K2, K3, and K4")
    return rate_axis, tile_bits


def ldlq_mixed(
    weight: torch.Tensor,
    L: torch.Tensor,
    quant_args: dict,
    pb: ProgressBar | None = None,
):
    """Dense-H LDLQ with K selected per tile on one logical matrix axis.

    This is the heterogeneous-rate equivalent of :func:`ldlq`.  It preserves
    the same backward block traversal and error-feedback state.  In
    particular, K2/K3/K4 partitions are not quantized as independent matrix
    calls, so off-diagonal Hessian coupling remains active across rate
    boundaries.
    """

    devices = quant_args["devices"]
    for quant_device in devices:
        torch.cuda.synchronize(quant_device)
    main_stream = get_quant_stream(devices[0])
    with torch.cuda.stream(main_stream):
        device = L.device
        if device != torch.device(devices[0]):
            raise ValueError("mixed-rate LDLQ decomposition is on the wrong device")
        buffer_device = weight.device
        size_k, size_n = weight.shape
        rate_axis, tile_bits = mixed_rate_spec(size_k, size_n, quant_args)
        tiles_k = size_k // 16
        tiles_n = size_n // 16

        buffer_rows = max(quant_args.get("buf_size_k", 128), 16)
        if buffer_rows % 16 or size_k % buffer_rows:
            raise ValueError("mixed-rate LDLQ buffer must tile the EXL K dimension")

        product_cache = torch.zeros(
            (size_k, size_n), dtype=torch.float, device=device
        )
        weight_q = torch.zeros(
            (size_k, size_n), dtype=torch.float, device=buffer_device
        )
        encoded_q = torch.empty(
            (tiles_k, tiles_n, 256), dtype=torch.int16, device=buffer_device
        )
        perm = tensor_core_perm(device)
        perm_i = tensor_core_perm_i(device)
        n_positions = None
        if rate_axis == "n":
            bits_tensor = torch.tensor(tile_bits, dtype=torch.int64, device=device)
            n_positions = {
                bit_width: torch.nonzero(bits_tensor == bit_width).flatten()
                for bit_width in sorted(set(tile_bits))
            }

        progress_row = 0
        for end in range(size_k, 0, -buffer_rows):
            begin = end - buffer_rows
            block_weight = weight[begin:end].to(device)
            block_weight_q = (
                weight_q[begin:end]
                if buffer_device == device
                else torch.zeros_like(block_weight, device=device)
            )
            block_encoded = (
                encoded_q[begin // 16:end // 16]
                if buffer_device == device
                else torch.empty(
                    (buffer_rows // 16, tiles_n, 256),
                    dtype=torch.int16,
                    device=device,
                )
            )
            block_product = product_cache[begin:end]
            block_ldl = L[begin:end]

            for block_end in range(buffer_rows, 0, -16):
                block_begin = block_end - 16
                error = block_weight[block_end:] - block_weight_q[block_end:]
                local_ldl = block_ldl[
                    block_end:, begin + block_begin:begin + block_end
                ]
                compensation = block_product[block_begin:block_end]
                compensation.addmm_(local_ldl.T, error, alpha=1.0, beta=1.0)
                rows = block_weight[block_begin:block_end] + compensation
                tiles = (
                    rows.reshape(16, tiles_n, 16)
                    .permute(1, 0, 2)
                    .reshape(tiles_n, 256)
                )
                tiles = tiles[:, perm]

                if rate_axis == "k":
                    local_args = dict(quant_args)
                    local_args["K"] = tile_bits[(begin + block_begin) // 16]
                    quantized, indices = quantize_tiles_multigpu(tiles, local_args)
                else:
                    quantized = torch.empty_like(tiles)
                    indices = torch.empty_like(tiles, dtype=torch.int16)
                    assert n_positions is not None
                    for bit_width, positions in n_positions.items():
                        local_args = dict(quant_args)
                        local_args["K"] = bit_width
                        local_quantized, local_indices = quantize_tiles_multigpu(
                            tiles.index_select(0, positions), local_args
                        )
                        quantized.index_copy_(0, positions, local_quantized)
                        indices.index_copy_(0, positions, local_indices)

                block_encoded[block_begin // 16] = indices
                quantized = quantized[:, perm_i]
                quantized = (
                    quantized.reshape(tiles_n, 16, 16)
                    .permute(1, 0, 2)
                    .reshape(16, size_n)
                )
                block_weight_q[block_begin:block_end] = quantized
                if pb:
                    progress_row += 1
                    pb.update(progress_row)

            if buffer_device != device:
                weight_q[begin:end] = block_weight_q.to(buffer_device)
                encoded_q[begin // 16:end // 16] = block_encoded.to(buffer_device)
            block_error = block_weight - block_weight_q
            product_cache.addmm_(block_ldl.T, block_error, alpha=1.0, beta=1.0)

    for quant_device in devices:
        torch.cuda.synchronize(quant_device)
    return weight_q, encoded_q


def _make_mixed_ldlq_grouping(position_bits, device):
    """Build one immutable gather plan outside CUDA graph capture."""

    values = tuple(int(value) for value in position_bits)
    widths = tuple(sorted(set(values)))
    if len(widths) == 1:
        return None, None, ((widths[0], 0, len(values)),)
    order = tuple(
        position
        for bit_width in widths
        for position, value in enumerate(values)
        if value == bit_width
    )
    inverse = [0] * len(order)
    for grouped, original in enumerate(order):
        inverse[original] = grouped
    slices = []
    begin = 0
    for bit_width in widths:
        end = begin + values.count(bit_width)
        slices.append((bit_width, begin, end))
        begin = end
    return (
        torch.tensor(order, dtype=torch.int64, device=device),
        torch.tensor(inverse, dtype=torch.int64, device=device),
        tuple(slices),
    )


def _prepare_mixed_ldlq_groupings(rate_axis, tile_bits, tiles_k, tiles_n, device):
    if rate_axis == "n":
        return (
            _make_mixed_ldlq_grouping(
                (value for member in tile_bits for value in member), device
            ),
            None,
        )

    grouping_cache = {}
    input_groupings = []
    for rate_tile in range(tiles_k):
        member_bits = tuple(bits[rate_tile] for bits in tile_bits)
        grouping = grouping_cache.get(member_bits)
        if grouping is None:
            grouping = _make_mixed_ldlq_grouping(
                (
                    bit_width
                    for bit_width in member_bits
                    for _ in range(tiles_n)
                ),
                device,
            )
            grouping_cache[member_bits] = grouping
        input_groupings.append(grouping)
    return None, input_groupings


def _baddbmm_ldlq(input, batch1, batch2, *, tf32, inplace=False):
    """Dispatch one LDLQ feedback product at the requested FP32 policy.

    PyTorch 2.13 defaults float32 matmuls to IEEE FP32 on this host.  TF32 is
    an encoder-only control: tensors and accumulation outputs remain FP32, but
    Blackwell Tensor Cores consume ten mantissa bits for the two multiplicands.
    Restore the process-global backend flag immediately after dispatch so
    functional scoring and unrelated encoder operations retain their policy.
    """

    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    try:
        if inplace:
            return input.baddbmm_(batch1, batch2)
        return torch.baddbmm(input, batch1, batch2)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def ldlq_mixed_batched(
    weights: torch.Tensor,
    Ls: torch.Tensor,
    quant_args_list: list[dict],
    pb: ProgressBar | None = None,
    *,
    synchronize: bool = True,
    prepared_groupings=None,
    process_begin: int = 0,
    process_end: int | None = None,
    initial_weights_q: torch.Tensor | None = None,
    initial_encoded_q: torch.Tensor | None = None,
    initial_product_cache: torch.Tensor | None = None,
    return_product_cache: bool = False,
    source_indices: torch.Tensor | None = None,
):
    """Batch independent heterogeneous-rate LDLQ traversals.

    Every batch member retains its own reconstruction error and dense-H LDL
    factor.  At each 16-row step, tiles from all members are gathered by K and
    submitted to at most one K2, K3, and K4 kernel call.  The batching changes
    launch granularity, not the candidate's rate map or error-feedback graph.

    ``process_begin``/``process_end`` optionally restrict the backwards
    traversal to one buffer-aligned K interval.  A lower interval must receive
    all three state tensors returned by an earlier upper-interval call.  This
    supports exact common-prefix reuse across several rate candidates without
    approximating or resetting dense-H error feedback.

    Ordinarily ``weights`` and ``Ls`` use shapes ``[B, k, n]`` and
    ``[B, k, k]``.  A continuation trie can instead retain one immutable
    source/Hessian pair per expert and pass ``source_indices`` with shape
    ``[B]``.  Only the small row slices needed by the current LDLQ step are
    then gathered; the multi-gigabyte source and dense-H tensors are never
    copied once per live rate state.  All members must use the same logical
    rate axis, matrix shape, and device, but may carry different K2/K3/K4 maps
    on that axis.
    """

    if weights.ndim != 3 or Ls.ndim != 3:
        raise ValueError("batched mixed LDLQ expects rank-three weights and Ls")
    source_count, size_k, size_n = weights.shape
    batch = len(quant_args_list)
    if source_count <= 0 or batch <= 0 or Ls.shape != (source_count, size_k, size_k):
        raise ValueError("batched mixed LDLQ weight/L shapes do not agree")
    if weights.dtype != torch.float or Ls.dtype != torch.float:
        raise TypeError("batched mixed LDLQ weights and Ls must be float32")
    if weights.device != Ls.device or weights.device.type != "cuda":
        raise ValueError("batched mixed LDLQ inputs must share one CUDA device")
    if source_indices is None:
        if source_count != batch:
            raise ValueError(
                "batched mixed LDLQ needs one source per member without source_indices"
            )
    else:
        if (
            source_indices.shape != (batch,)
            or source_indices.dtype != torch.long
            or source_indices.device != weights.device
        ):
            raise ValueError("mixed LDLQ source_indices must be device-local int64 [B]")
        if bool(torch.any(source_indices < 0)) or bool(
            torch.any(source_indices >= source_count)
        ):
            raise ValueError("mixed LDLQ source_indices are out of range")

    devices = quant_args_list[0]["devices"]
    device = weights.device
    if device != torch.device(devices[0]):
        raise ValueError("batched mixed LDLQ inputs are on the wrong device")
    if any(args.get("devices") != devices for args in quant_args_list):
        raise ValueError("batched mixed LDLQ members must share devices")
    ldlq_tf32 = bool(quant_args_list[0].get("ldlq_tf32", False))
    if any(bool(args.get("ldlq_tf32", False)) != ldlq_tf32 for args in quant_args_list):
        raise ValueError("batched mixed LDLQ members must share TF32 policy")
    if synchronize:
        for quant_device in devices:
            torch.cuda.synchronize(quant_device)

    normalized = [mixed_rate_spec(size_k, size_n, args) for args in quant_args_list]
    rate_axis = normalized[0][0]
    if any(axis != rate_axis for axis, _ in normalized):
        raise ValueError("batched mixed LDLQ members must share one rate axis")
    tile_bits = [bits for _, bits in normalized]
    tiles_k = size_k // 16
    tiles_n = size_n // 16
    buffer_rows = max(quant_args_list[0].get("buf_size_k", 128), 16)
    if any(max(args.get("buf_size_k", 128), 16) != buffer_rows for args in quant_args_list):
        raise ValueError("batched mixed LDLQ members must share one buffer size")
    if buffer_rows % 16 or size_k % buffer_rows:
        raise ValueError("batched mixed LDLQ buffer must tile the EXL K dimension")
    if process_end is None:
        process_end = size_k
    if (
        isinstance(process_begin, bool)
        or isinstance(process_end, bool)
        or not isinstance(process_begin, int)
        or not isinstance(process_end, int)
        or not 0 <= process_begin < process_end <= size_k
        or process_begin % buffer_rows
        or process_end % buffer_rows
    ):
        raise ValueError("mixed LDLQ process interval must be buffer-aligned")

    initial_state = (
        initial_weights_q,
        initial_encoded_q,
        initial_product_cache,
    )
    if any(value is not None for value in initial_state) and not all(
        value is not None for value in initial_state
    ):
        raise ValueError("mixed LDLQ continuation requires all three state tensors")
    if initial_weights_q is None and process_end != size_k:
        raise ValueError("a lower mixed LDLQ interval requires continuation state")

    if prepared_groupings is None:
        static_grouping, input_groupings = _prepare_mixed_ldlq_groupings(
            rate_axis, tile_bits, tiles_k, tiles_n, device
        )
    else:
        static_grouping, input_groupings = prepared_groupings

    main_stream = get_quant_stream(devices[0])
    with torch.cuda.stream(main_stream):
        if initial_weights_q is None:
            product_cache = weights.new_zeros((batch, size_k, size_n))
            weights_q = weights.new_zeros((batch, size_k, size_n))
            encoded_q = torch.zeros(
                (batch, tiles_k, tiles_n, 256), dtype=torch.int16, device=device
            )
        else:
            weights_q = initial_weights_q
            encoded_q = initial_encoded_q
            product_cache = initial_product_cache
            assert encoded_q is not None and product_cache is not None
            expected_weights = (batch, size_k, size_n)
            expected_encoded = (batch, tiles_k, tiles_n, 256)
            if (
                weights_q.shape != expected_weights
                or weights_q.dtype != weights.dtype
                or weights_q.device != device
                or product_cache.shape != expected_weights
                or product_cache.dtype != weights.dtype
                or product_cache.device != device
                or encoded_q.shape != expected_encoded
                or encoded_q.dtype != torch.int16
                or encoded_q.device != device
            ):
                raise ValueError("mixed LDLQ continuation state has invalid shape or type")
        perm = tensor_core_perm(device)
        perm_i = tensor_core_perm_i(device)
        progress_row = 0

        for end in range(process_end, process_begin, -buffer_rows):
            begin = end - buffer_rows
            for block_end in range(buffer_rows, 0, -16):
                block_begin = block_end - 16
                row_begin = begin + block_begin
                row_end = begin + block_end
                source_tail = weights[:, row_end:end]
                local_ldl = Ls[:, row_end:end, row_begin:row_end]
                source_rows = weights[:, row_begin:row_end]
                if source_indices is not None:
                    source_tail = source_tail.index_select(0, source_indices)
                    local_ldl = local_ldl.index_select(0, source_indices)
                    source_rows = source_rows.index_select(0, source_indices)
                error = source_tail - weights_q[:, row_end:end]
                compensation = _baddbmm_ldlq(
                    product_cache[:, row_begin:row_end],
                    local_ldl.transpose(1, 2),
                    error,
                    tf32=ldlq_tf32,
                )
                rows = source_rows + compensation
                tiles = (
                    rows.reshape(batch, 16, tiles_n, 16)
                    .permute(0, 2, 1, 3)
                    .reshape(batch * tiles_n, 256)
                )
                tiles = tiles[:, perm]
                if rate_axis == "n":
                    assert static_grouping is not None
                    grouping = static_grouping
                else:
                    rate_tile = row_begin // 16
                    assert input_groupings is not None
                    grouping = input_groupings[rate_tile]
                order, inverse, rate_slices = grouping
                grouped_tiles = tiles if order is None else tiles.index_select(0, order)
                quantized_parts = []
                index_parts = []
                for bit_width, part_begin, part_end in rate_slices:
                    local_args = dict(quant_args_list[0])
                    local_args["K"] = bit_width
                    local_quantized, local_indices = quantize_tiles_multigpu(
                        grouped_tiles[part_begin:part_end], local_args
                    )
                    quantized_parts.append(local_quantized)
                    index_parts.append(local_indices)
                if len(quantized_parts) == 1:
                    quantized = quantized_parts[0]
                    indices = index_parts[0]
                else:
                    quantized = torch.cat(quantized_parts, dim=0)
                    indices = torch.cat(index_parts, dim=0)
                    assert inverse is not None
                    quantized = quantized.index_select(0, inverse)
                    indices = indices.index_select(0, inverse)

                encoded_q[:, row_begin // 16] = indices.reshape(
                    batch, tiles_n, 256
                )
                quantized = quantized[:, perm_i]
                weights_q[:, row_begin:row_end] = (
                    quantized.reshape(batch, tiles_n, 16, 16)
                    .permute(0, 2, 1, 3)
                    .reshape(batch, 16, size_n)
                )
                if pb:
                    progress_row += 1
                    pb.update(progress_row)

            # This buffer has now been committed and LDLQ only visits rows
            # below ``begin`` from here on.  Feedback into rows at or above
            # ``begin`` is therefore dead: those product-cache entries are
            # never read again, including when this call returns continuation
            # state to the mixed-K prefix trie.  Limit the dense update to the
            # still-unprocessed prefix.  This preserves every subsequently
            # observed value while avoiding roughly half of the block-update
            # GEMM work over a complete backwards traversal.
            if begin:
                source_block = weights[:, begin:end]
                block_ldl = Ls[:, begin:end, :begin]
                if source_indices is not None:
                    source_block = source_block.index_select(0, source_indices)
                    block_ldl = block_ldl.index_select(0, source_indices)
                block_error = source_block - weights_q[:, begin:end]
                _baddbmm_ldlq(
                    product_cache[:, :begin],
                    block_ldl.transpose(1, 2),
                    block_error,
                    tf32=ldlq_tf32,
                    inplace=True,
                )

    if synchronize:
        for quant_device in devices:
            torch.cuda.synchronize(quant_device)
    if return_product_cache:
        return weights_q, encoded_q, product_cache
    return weights_q, encoded_q


def ldlq_mixed_n_candidates_reuse(
    weights: torch.Tensor,
    Ls: torch.Tensor,
    quant_args_groups: list[list[dict]],
):
    """Encode output-axis candidates while reusing their common K3 tiles.

    LDLQ error feedback couples rows of the EXL-oriented matrix (the input
    axis), but output columns are independent.  Consequently two candidates
    that differ only in ``mixed_tile_bits`` on the ``n`` axis produce exactly
    the same reconstruction and trellis states for every unchanged output
    tile.  The phase-one R0/R1/R2 ladder can therefore encode the full K3
    matrix once, encode the union of its K2/K4 replacement records once, and
    assemble all candidates without changing their LDLQ graphs.

    The helper intentionally accepts only an all-K3 first candidate and
    rejects maps that assign more than one non-K3 width to the same tile.  It
    returns candidates in source-major order, matching the flattened order in
    :func:`quantize_qsrt_batch`.
    """

    if weights.ndim != 3 or Ls.ndim != 3:
        raise ValueError("output-axis reuse expects rank-three weights and Ls")
    source_count, size_k, size_n = weights.shape
    if source_count <= 0 or Ls.shape != (source_count, size_k, size_k):
        raise ValueError("output-axis reuse weight/L shapes do not agree")
    if len(quant_args_groups) != source_count or any(
        not group for group in quant_args_groups
    ):
        raise ValueError("output-axis reuse needs candidates for every source")

    normalized_groups = []
    for group in quant_args_groups:
        normalized = [mixed_rate_spec(size_k, size_n, args) for args in group]
        if any(axis != "n" for axis, _ in normalized):
            raise ValueError("output-axis reuse supports only mixed_rate_axis='n'")
        if any(bits != (3,) * (size_n // 16) for bits in [normalized[0][1]]):
            raise ValueError("output-axis reuse requires an all-K3 first candidate")
        normalized_groups.append([bits for _, bits in normalized])

    # R0 is common to every candidate and every unchanged output record.
    base_args = [group[0] for group in quant_args_groups]
    base_weights_q, base_encoded_q = ldlq_mixed_batched(
        weights, Ls, base_args
    )

    # Quantize each source/width union once.  Group unions of equal size so
    # K2 and K4 records from both w1 and w3 share large trellis launches.
    override_specs = []
    for source, candidate_bits in enumerate(normalized_groups):
        width_by_tile = {}
        for bits in candidate_bits[1:]:
            for tile, bit_width in enumerate(bits):
                if bit_width == 3:
                    continue
                previous = width_by_tile.setdefault(tile, bit_width)
                if previous != bit_width:
                    raise ValueError(
                        "one output tile uses multiple non-K3 widths across candidates"
                    )
        for bit_width in (2, 4):
            positions = tuple(
                tile for tile, width in width_by_tile.items() if width == bit_width
            )
            if positions:
                override_specs.append((source, bit_width, positions))

    overrides = {}
    counts = sorted({len(positions) for _, _, positions in override_specs})
    for tile_count in counts:
        members = [
            spec for spec in override_specs if len(spec[2]) == tile_count
        ]
        member_weights = []
        member_Ls = []
        member_args = []
        for source, bit_width, positions in members:
            columns = torch.tensor(
                [
                    column
                    for tile in positions
                    for column in range(tile * 16, (tile + 1) * 16)
                ],
                dtype=torch.int64,
                device=weights.device,
            )
            member_weights.append(weights[source].index_select(1, columns))
            member_Ls.append(Ls[source])
            args = dict(quant_args_groups[source][0])
            args["mixed_tile_bits"] = (bit_width,) * tile_count
            member_args.append(args)
        override_weights_q, override_encoded_q = ldlq_mixed_batched(
            torch.stack(member_weights),
            torch.stack(member_Ls),
            member_args,
        )
        for member, (source, bit_width, positions) in enumerate(members):
            overrides[(source, bit_width)] = (
                positions,
                override_weights_q[member],
                override_encoded_q[member],
            )

    candidates_weights_q = []
    candidates_encoded_q = []
    for source, candidate_bits in enumerate(normalized_groups):
        for candidate, bits in enumerate(candidate_bits):
            if candidate == 0:
                candidates_weights_q.append(base_weights_q[source])
                candidates_encoded_q.append(base_encoded_q[source])
                continue
            weight_q = base_weights_q[source].clone()
            encoded_q = base_encoded_q[source].clone()
            for bit_width in (2, 4):
                target_tiles = tuple(
                    tile for tile, width in enumerate(bits) if width == bit_width
                )
                if not target_tiles:
                    continue
                positions, replacement_weight, replacement_encoded = overrides[
                    (source, bit_width)
                ]
                local_by_tile = {tile: local for local, tile in enumerate(positions)}
                local_tiles = tuple(local_by_tile[tile] for tile in target_tiles)
                target_columns = torch.tensor(
                    [
                        column
                        for tile in target_tiles
                        for column in range(tile * 16, (tile + 1) * 16)
                    ],
                    dtype=torch.int64,
                    device=weights.device,
                )
                local_columns = torch.tensor(
                    [
                        column
                        for tile in local_tiles
                        for column in range(tile * 16, (tile + 1) * 16)
                    ],
                    dtype=torch.int64,
                    device=weights.device,
                )
                target_tiles_tensor = torch.tensor(
                    target_tiles, dtype=torch.int64, device=weights.device
                )
                local_tiles_tensor = torch.tensor(
                    local_tiles, dtype=torch.int64, device=weights.device
                )
                weight_q.index_copy_(
                    1,
                    target_columns,
                    replacement_weight.index_select(1, local_columns),
                )
                encoded_q.index_copy_(
                    1,
                    target_tiles_tensor,
                    replacement_encoded.index_select(1, local_tiles_tensor),
                )
            candidates_weights_q.append(weight_q)
            candidates_encoded_q.append(encoded_q)

    return torch.stack(candidates_weights_q), torch.stack(candidates_encoded_q)


def ldlq_mixed_k_candidates_prefix_reuse(
    weights: torch.Tensor,
    Ls: torch.Tensor,
    quant_args_groups: list[list[dict]],
):
    """Encode input-axis candidates with an exact shared backwards prefix.

    For every source, candidate rate maps must share a buffer-aligned suffix
    on the EXL K axis.  LDLQ traverses that suffix first.  We quantize it once,
    retain the complete reconstruction and dense-H product cache, fan those
    states out source-major across candidates, and continue the remaining K
    rows independently.  The result is numerically identical to independent
    full traversals in the same coordinate order.

    This is particularly useful for an R0--R5 layout that places the ten
    donor/recipient records below the fourteen common K3 records: record work
    falls from ``6 * 24`` to ``14 + 6 * 10`` without changing any path search.
    """

    if weights.ndim != 3 or Ls.ndim != 3:
        raise ValueError("input-axis prefix reuse expects rank-three weights and Ls")
    source_count, size_k, size_n = weights.shape
    if source_count <= 0 or Ls.shape != (source_count, size_k, size_k):
        raise ValueError("input-axis prefix reuse weight/L shapes do not agree")
    if len(quant_args_groups) != source_count or any(
        not group for group in quant_args_groups
    ):
        raise ValueError("input-axis prefix reuse needs candidates for every source")
    if not any(len(group) > 1 for group in quant_args_groups):
        raise ValueError("input-axis prefix reuse needs at least one multi-candidate source")

    normalized_groups = []
    suffix_starts = []
    for group in quant_args_groups:
        normalized = [mixed_rate_spec(size_k, size_n, args) for args in group]
        if any(axis != "k" for axis, _ in normalized):
            raise ValueError("input-axis prefix reuse supports only mixed_rate_axis='k'")
        maps = [bits for _, bits in normalized]
        suffix_start = len(maps[0])
        while suffix_start > 0 and len(
            {bits[suffix_start - 1] for bits in maps}
        ) == 1:
            suffix_start -= 1
        normalized_groups.append(maps)
        suffix_starts.append(suffix_start)

    buffer_rows = max(quant_args_groups[0][0].get("buf_size_k", 128), 16)
    if any(
        max(args.get("buf_size_k", 128), 16) != buffer_rows
        for group in quant_args_groups
        for args in group
    ):
        raise ValueError("input-axis prefix reuse requires one LDLQ buffer size")
    # Use the suffix common to every source and round toward a smaller shared
    # region so the split remains a valid LDLQ buffer boundary.
    split_row = max(suffix_starts) * 16
    split_row = ((split_row + buffer_rows - 1) // buffer_rows) * buffer_rows
    if split_row >= size_k:
        flat_members = [
            (source, candidate)
            for source, group in enumerate(quant_args_groups)
            for candidate in range(len(group))
        ]
        return ldlq_mixed_batched(
            torch.stack([weights[source] for source, _ in flat_members]),
            torch.stack([Ls[source] for source, _ in flat_members]),
            [
                quant_args_groups[source][candidate]
                for source, candidate in flat_members
            ],
        )

    shared_weights_q, shared_encoded_q, shared_product_cache = ldlq_mixed_batched(
        weights,
        Ls,
        [group[0] for group in quant_args_groups],
        process_begin=split_row,
        process_end=size_k,
        return_product_cache=True,
    )

    if split_row == 0:
        flat_source = torch.tensor(
            [
                source
                for source, group in enumerate(quant_args_groups)
                for _ in group
            ],
            dtype=torch.long,
            device=weights.device,
        )
        return (
            shared_weights_q.index_select(0, flat_source),
            shared_encoded_q.index_select(0, flat_source),
        )

    # Traverse the variable records as a state trie.  A node carries one exact
    # LDLQ state and all candidates whose rate maps have been identical so far.
    # At a rate boundary it is copied only once per distinct child map.  Nested
    # R ladders therefore grow from one to six live states gradually instead of
    # paying for six complete variable regions.
    nodes = [
        (source, tuple(range(len(group))))
        for source, group in enumerate(quant_args_groups)
    ]
    node_weights_q = shared_weights_q
    node_encoded_q = shared_encoded_q
    node_product_cache = shared_product_cache
    devices = quant_args_groups[0][0]["devices"]
    main_stream = get_quant_stream(devices[0])
    with torch.cuda.stream(main_stream):
        for end in range(split_row, 0, -buffer_rows):
            begin = end - buffer_rows
            tile_begin = begin // 16
            tile_end = end // 16
            child_nodes = []
            parent_indices = []
            child_args = []
            for parent, (source, candidates) in enumerate(nodes):
                partitions = {}
                for candidate in candidates:
                    key = normalized_groups[source][candidate][tile_begin:tile_end]
                    partitions.setdefault(key, []).append(candidate)
                for members in partitions.values():
                    child_nodes.append((source, tuple(members)))
                    parent_indices.append(parent)
                    child_args.append(quant_args_groups[source][members[0]])

            if len(child_nodes) != len(nodes) or any(
                parent != index for index, parent in enumerate(parent_indices)
            ):
                parent_index = torch.tensor(
                    parent_indices,
                    dtype=torch.long,
                    device=weights.device,
                )
                node_weights_q = node_weights_q.index_select(0, parent_index)
                node_encoded_q = node_encoded_q.index_select(0, parent_index)
                node_product_cache = node_product_cache.index_select(
                    0, parent_index
                )
            nodes = child_nodes
            node_source_indices = torch.tensor(
                [source for source, _ in nodes],
                dtype=torch.long,
                device=weights.device,
            )
            node_weights_q, node_encoded_q, node_product_cache = ldlq_mixed_batched(
                weights,
                Ls,
                child_args,
                synchronize=False,
                process_begin=begin,
                process_end=end,
                initial_weights_q=node_weights_q,
                initial_encoded_q=node_encoded_q,
                initial_product_cache=node_product_cache,
                return_product_cache=True,
                source_indices=node_source_indices,
            )
    for quant_device in devices:
        torch.cuda.synchronize(quant_device)

    node_by_candidate = {}
    for node, (source, candidates) in enumerate(nodes):
        for candidate in candidates:
            node_by_candidate[(source, candidate)] = node
    flat_members = [
        (source, candidate)
        for source, group in enumerate(quant_args_groups)
        for candidate in range(len(group))
    ]
    if set(node_by_candidate) != set(flat_members):
        raise AssertionError("input-axis prefix trie lost a rate candidate")
    output_index = torch.tensor(
        [node_by_candidate[member] for member in flat_members],
        dtype=torch.long,
        device=weights.device,
    )
    return (
        node_weights_q.index_select(0, output_index),
        node_encoded_q.index_select(0, output_index),
    )


def fallback_quant(
    weight: torch.Tensor,
    q_device: torch.Tensor,
    quant_args: dict,
    pb: ProgressBar | None = None
):
    """
    Perform the same quantization as ldlq() but without an LDL decomposition

    :param weight:
        Input weights, shape (k, n). If device is "cpu", result is collected on CPU as well, saving a bunch of
        VRAM but adding a little PCIe overhead and many sync points

    :param q_device:
        Target device

    :param quant_args:
        dict:
         - K: bitrate
         - buf_size_k: buffer size for faux-LDLQ, along k

    :param pb:
        Optional ProgressPar to update, k // 16 steps

    :return:
        tuple:
         - quantized weight, shape (k, n)
         - indices (unpacked), shape (k // 16, n // 16, 256), uint16_t
    """

    devices = quant_args["devices"]
    for device in devices:
        torch.cuda.synchronize(device)
    main_stream = get_quant_stream(devices[0])
    with torch.cuda.stream(main_stream):

        devices = quant_args["devices"]
        device = weight.device
        assert device == torch.device(devices[0])

        buffer_device = weight.device
        size_k, size_n = weight.shape  # Row-major
        assert size_k % 16 == 0
        assert size_n % 128 == 0
        tiles_k = size_k // 16
        tiles_n = size_n // 16

        buf_size_k = max(quant_args.get("buf_size_k", 128), 16)
        assert buf_size_k % 16 == 0
        assert size_n % buf_size_k == 0

        p_row = 0

        # Work buffers
        weight_q = torch.zeros((size_k, size_n), dtype = torch.float, device = buffer_device)
        encoded = torch.zeros((tiles_k, tiles_n, 256), dtype = torch.short, device = buffer_device)

        # Accumulate sum of squared error on-device to avoid per-block CPU syncs
        mse_sum = torch.zeros((), dtype=torch.float64, device=device)

        for j in range(size_k, 0, -buf_size_k):
            i = j - buf_size_k

            # Current span is rows i:j
            b_weight = weight[i:j].to(device)
            b_weight_q = weight_q[i:j] if device == buffer_device else \
                torch.zeros_like(weight_q[i:j], device = device)
            b_encoded = encoded[i // 16 : j // 16] if device == buffer_device else \
                torch.zeros_like(encoded[i // 16 : j // 16], device = device)

            # Iterate over rows of blocks in current span
            for bj in range(buf_size_k, 0, -16):
                bi = bj - 16

                # Input tiles for quantization
                rows = b_weight[bi:bj]
                tiles = rows.reshape(16, tiles_n, 16).permute(1, 0, 2).reshape(tiles_n, 256)

                # Pre-permute to tensor core layout
                tiles = tiles[:, tensor_core_perm(device)]

                # Quantize
                quant_w, quant_i = quantize_tiles_multigpu(tiles, quant_args)

                # Undo permutation on reconstructed tiles, but keep indices in tensor core layout
                quant_w = quant_w[:, tensor_core_perm_i(device)]

                # Restore row-major block layout
                quant_w = quant_w.reshape(tiles_n, 16, 16).permute(1, 0, 2).reshape(16, size_n)

                # Accumulate squared error for this block
                mse_sum += torch.sum(
                    (rows.float() - quant_w.float()).pow(2),
                    dtype=torch.float64,
                )

                # Store result
                b_weight_q[bi:bj] = quant_w
                b_encoded[bi // 16 : bj // 16] = quant_i.unsqueeze(0)

                # Update progress
                if pb:
                    p_row += 1
                    pb.update(p_row)

            # Collect output
            if device != buffer_device:
                weight_q[i:j] = b_weight_q.to(buffer_device)
                encoded[i // 16 : j // 16] = b_encoded.to(buffer_device)

        for device in devices:
            torch.cuda.synchronize(device)

        mse = (mse_sum / weight.numel()).item()

    return weight_q, encoded, mse


def block_trace_parts(A, B, block_size = 1024):
    """Return the device-resident column-block terms of ``trace(A.T @ B @ A)``."""

    return torch.stack(
        [
            torch.einsum(
                "ik,ij,jk->",
                A,
                B[:, j_start : min(j_start + block_size, B.shape[1])],
                A[j_start : min(j_start + block_size, B.shape[1]), :],
            )
            for j_start in range(0, B.shape[1], block_size)
        ]
    )


def block_trace(A, B, block_size = 1024):
    """
    Compute trace(A.T @ B @ A) in column blocks of B to bound the temporary
    """
    total = 0.0
    # Preserve the historical Python summation order while paying for only
    # one device synchronization.  This helper is also the scalar reference
    # for the batched mixed-candidate proxy path below.
    for partial in block_trace_parts(A, B, block_size).cpu():
        total += partial.item()
    return total


def ldlq_batched(
    weights: torch.Tensor,
    Ls: torch.Tensor,
    quant_args: dict,
    pb: ProgressBar | None = None
):
    """
    LDLQ over a stack of same-shape tensors with per-tensor L, batched so each 16-row step quantizes all
    tensors' tiles in one quantize_tiles call. The recursion is identical to ldlq() per batch element; only
    the matmuls become bmm. All buffers stay on-device (this path is only used for groups of small tensors).

    :param weights:
        Input weights, shape (B, k, n), float, on the quant device

    :param Ls:
        LDL decompositions of the regularized Hessians, shape (B, k, k), same device

    :return:
        tuple:
         - quantized weights, shape (B, k, n)
         - indices (unpacked), shape (B, k // 16, n // 16, 256), int16
    """
    devices = quant_args["devices"]
    for device in devices:
        torch.cuda.synchronize(device)
    main_stream = get_quant_stream(devices[0])
    with torch.cuda.stream(main_stream):

        device = weights.device
        assert device == torch.device(devices[0]) and Ls.device == device
        B, size_k, size_n = weights.shape
        assert Ls.shape == (B, size_k, size_k)
        assert size_k % 16 == 0
        assert size_n % 128 == 0
        tiles_k = size_k // 16
        tiles_n = size_n // 16

        buf_size_k = max(quant_args.get("buf_size_k", 128), 16)
        assert buf_size_k % 16 == 0
        assert size_k % buf_size_k == 0

        p_row = 0
        perm = tensor_core_perm(device)
        perm_i = tensor_core_perm_i(device)

        prod_cache = torch.zeros((B, size_k, size_n), dtype = torch.float, device = device)
        weight_q = torch.zeros_like(weights)
        encoded = torch.zeros((B, tiles_k, tiles_n, 256), dtype = torch.short, device = device)

        for j in range(size_k, 0, -buf_size_k):
            i = j - buf_size_k

            for bj in range(buf_size_k, 0, -16):
                bi = bj - 16
                gi, gj = i + bi, i + bj

                # Error so far for the remaining rows of the current span
                bb_err = weights[:, gj:j] - weight_q[:, gj:j]

                # Corresponding slice of the LDL decompositions
                bb_L = Ls[:, gj:j, gi:gj]

                # Input tiles for quantization; out-of-place equivalent of ldlq()'s in-place accumulation
                # (the mutated prod_cache rows are never read again there)
                compensation = torch.baddbmm(prod_cache[:, gi:gj], bb_L.transpose(1, 2), bb_err)
                rows = weights[:, gi:gj] + compensation

                tiles = rows.reshape(B, 16, tiles_n, 16).permute(0, 2, 1, 3).reshape(B * tiles_n, 256)
                tiles = tiles[:, perm]

                quant_w, quant_i = quantize_tiles_multigpu(tiles, quant_args)

                quant_w = quant_w[:, perm_i]
                quant_w = quant_w.reshape(B, tiles_n, 16, 16).permute(0, 2, 1, 3).reshape(B, 16, size_n)
                weight_q[:, gi:gj] = quant_w
                encoded[:, gi // 16] = quant_i.view(B, tiles_n, 256)

                if pb:
                    p_row += 1
                    pb.update(p_row)

            # Cache error term for the rest of the matrices
            b_err = weights[:, i:j] - weight_q[:, i:j]
            prod_cache.baddbmm_(Ls[:, i:j].transpose(1, 2), b_err)

        for device in devices:
            torch.cuda.synchronize(device)

    return weight_q, encoded


finalize_capture_H_mutex = threading.Lock()

def finalize_capture_H(H_data: dict, quant_args: dict, verbose: bool):
    with finalize_capture_H_mutex:

        if H_data["H"].is_meta:
            H_data["L"] = None
            H_data["finalized"] = True
            H_data["diag"] = None
            H_data["q_fallback"] = True

            H = H_data["H"]
            k = H.shape[0]
            su = (torch.randn(k, device = H_data["device"]).sign() + 1e-5).sign().to(torch.float).unsqueeze(1)
            H_data["su"] = su

            return True, None, None, su, None

        # Unswap H
        if "H_swap_device" in H_data:
            H_data["H"] = H_data["H"].to(H_data["H_swap_device"])
            del H_data["H_swap_device"]

        H = H_data["H"]
        if H_data["finalized"]:
            return H_data["q_fallback"], H, H_data["L"], H_data["su"], H_data["diag"]

        # Mean of samples summed up during forward pass
        # Switch to uncalibrated fallback if no input activations or diagonal is too small (few activations)
        count = H_data["count"]
        if count == 0:
            q_fallback = True
            diag_mean = 0.0
        else:
            H /= count
            diag_mean = torch.diag(H).mean()
            q_fallback = diag_mean.item() < 1e-20

        # Regularize diagonal
        H.diagonal().add_(quant_args.get("sigma_reg", 0.025) * diag_mean)

        # Some tests
        diag = H.diagonal().clone()

        if verbose:
            print(f"     - H min/max: {H.min().item():.6f}   {H.max().item():.6f}")
            print(f"     - H mean/std: {H.mean().item():.6f}   {H.std().item():.6f}")
            print(f"     - H diag min/max: {diag.min():.6f}   {diag.max():.6f} ")

        # Random sign flips for input channel, fixed for the first linear layer to quantize with this H
        k = H.shape[0]
        su = (torch.randn(k, device = H.device).sign() + 1e-5).sign().to(torch.float).unsqueeze(1)
        H_data["su"] = su

        # Input had
        H *= su.T
        blockwise_preapply_had_r_(H, had_k)
        H *= su
        blockwise_preapply_had_l_(H, had_k)

        # Get block LDL decomposition of H, zero diagonal
        if q_fallback:
            L = None
        else:
            L, H = block_ldl(H, 16, quant_args, verbose)
            dr = torch.arange(k)
            L[dr, dr] = 0

        H_data["L"] = L

        # H is no longer needed except to compute proxy error so move to CPU
        H = H.cpu()
        H_data["H"] = H.cpu()

        H_data["finalized"] = True
        H_data["diag"] = diag
        H_data["q_fallback"] = q_fallback
        return q_fallback, H, L, su, diag


def pack_trellis(encoded: torch.Tensor, quant_args: dict) -> torch.Tensor:
    K = quant_args["K"]
    shape = encoded.shape
    assert len(shape) == 3 and shape[2] == 256
    assert encoded.dtype == torch.int16
    packed_shape = (shape[0], shape[1], 256 * K // 16)
    packed = torch.zeros(packed_shape, dtype = torch.int16, device = encoded.device)
    ext.pack_trellis(packed, encoded.contiguous(), K)
    # unpacked = torch.zeros_like(encoded)
    # ext.unpack_trellis(unpacked, packed, K)
    # assert torch.equal(unpacked, encoded)
    return packed


def pack_signs(signs: torch.Tensor, quant_args: dict) -> torch.Tensor:
    signs = signs.half().flatten().contiguous()
    assert signs.shape[0] % 16 == 0
    packed = torch.zeros(signs.shape[0] // 16, dtype = torch.int16, device = signs.device)
    ext.pack_signs(packed, signs)
    return packed


def sample_scale_tiles(weight_r: torch.Tensor, width: int = 3) -> torch.Tensor:
    """
    Sample tiles for the global scale search: a wrapped diagonal, guaranteeing every tile row and column is
    sampled at least once (outliers are typically whole input or output channels), plus the tiles with the
    highest and lowest RMS as explicit outlier insurance for anything a diagonal of the given width misses.
    """
    device = weight_r.device
    tiles_k = weight_r.shape[0] // 16
    tiles_n = weight_r.shape[1] // 16
    w4 = weight_r.view(tiles_k, 16, tiles_n, 16)

    diag_len = max(tiles_k, tiles_n)
    ii = torch.arange(diag_len, device = device).repeat_interleave(width)
    ww = torch.arange(width, device = device).repeat(diag_len)
    kk = ii % tiles_k
    nn = (ii + ww) % tiles_n

    # Tiles with extreme RMS, by flat tile index
    num_x = max(8, (diag_len * width) // 16)
    tile_ms = w4.square().mean(dim = (1, 3)).flatten()
    num_x = min(num_x, (tile_ms.shape[0] + 1) // 2)
    hi = torch.topk(tile_ms, num_x).indices
    lo = torch.topk(tile_ms, num_x, largest = False).indices
    xk = torch.cat((hi, lo)) // tiles_n
    xn = torch.cat((hi, lo)) % tiles_n

    tiles = w4[torch.cat((kk, xk)), :, torch.cat((nn, xn)), :].reshape(-1, 256)
    return tiles[:, tensor_core_perm(device)].contiguous()


def g_scale_search_batch(
    samples: list[torch.Tensor],
    quant_args: dict,
) -> list[tuple[float, torch.Tensor]]:
    """
    Global scale search over a group of tensors' tile samples, batching all evaluations into as few
    quantize_tiles calls as possible (two host syncs per group, regardless of group size).

    The sampled error curve is flat near the optimum but not perfectly unimodal, so a sequential golden-section
    search can converge on a local wiggle well away from the true minimum. Instead: a coarse grid over the full
    range on a subsample of the tiles, then a fine grid around the coarse minimum on the full sample, refined by
    parabolic interpolation; the curve's flatness makes the grid spacing precise enough.
    """
    n_t = len(samples)
    device = samples[0].device
    max_tiles = 65536

    def eval_pairs(pairs, tile_sets):
        # pairs: list of (tensor_idx, scale); one mse per pair, kernel calls chunked to the batch limit
        out = torch.empty(len(pairs), dtype = torch.float, device = device)
        i = 0
        while i < len(pairs):
            j, tot = i, 0
            while j < len(pairs) and tot + tile_sets[pairs[j][0]].shape[0] <= max_tiles:
                tot += tile_sets[pairs[j][0]].shape[0]
                j += 1
            j = max(j, i + 1)
            batch = torch.cat([tile_sets[t] * s for t, s in pairs[i:j]])
            quant_w, _ = quantize_tiles_multigpu(batch, quant_args)
            offset = 0
            for p, (t, s) in enumerate(pairs[i:j]):
                cnt = tile_sets[t].shape[0]
                out[i + p] = (quant_w[offset : offset + cnt] / s - tile_sets[t]).square().mean()
                offset += cnt
            i = j
        return out

    # Stage 1: coarse grid over the full search range, on a subsample of each tensor's tiles
    coarse = [0.1 + 0.2 * i for i in range(10)]  # 0.1 .. 1.9, same range as the old golden-section bracket
    subs = [s[::3] for s in samples]
    pairs1 = [(t, s) for t in range(n_t) for s in coarse]
    mse1 = eval_pairs(pairs1, subs).view(n_t, len(coarse))
    centers = [coarse[c] for c in mse1.argmin(dim = 1).tolist()]

    # Stage 2: fine grid around each tensor's coarse minimum, on its full sample
    step = 0.075
    fine = [[c + step * (i - 2) for i in range(5)] for c in centers]
    pairs2 = [(t, s) for t in range(n_t) for s in fine[t]]
    mse2 = eval_pairs(pairs2, samples).view(n_t, 5)
    mse2_h = mse2.tolist()

    results = []
    for t in range(n_t):
        best = min(range(5), key = lambda i: mse2_h[t][i])
        if 0 < best < 4:
            y0, y1, y2 = mse2_h[t][best - 1], mse2_h[t][best], mse2_h[t][best + 1]
            denom = y0 - 2.0 * y1 + y2
            offset = 0.5 * (y0 - y2) / denom if denom > 0 else 0.0
            offset = max(-0.5, min(0.5, offset))
        else:
            offset = 0.0
        best_scale = max(fine[t][best] + offset * step, 0.01)
        results.append((best_scale, mse2[t, best]))
    return results


def g_scale_gss(
    weight_r: torch.Tensor,
    verbose: bool,
    quant_args: dict,
    width: int = 3,
    pb: ProgressBar = None
):
    devices = quant_args["devices"]
    for device in devices:
        torch.cuda.synchronize(device)

    main_stream = get_quant_stream(devices[0])
    # TODO: Figure out why Torch always initializes cuda:0 when exiting this CM, even when it's not used
    with torch.cuda.stream(main_stream):
        tiles = sample_scale_tiles(weight_r, width)
        if pb:
            pb.update(50)
        best_scale, best_mse = g_scale_search_batch([tiles], quant_args)[0]
        if pb:
            pb.update(100)
        if verbose:
            print(f"     - scale search: min = {best_scale:.6f}, mse: {best_mse.item():.6f}")

    for device in devices:
        torch.cuda.synchronize(device)

    return best_scale, best_mse


def block_rms(x: torch.Tensor, dim: int, keepdim: bool = False, blocksize: int = 32):
    """
    Compute blockwise x.square().mean(dim, keepdim).sqrt()
    """
    n = x.size(dim)
    sq = None
    for block in torch.split(x, blocksize, dim = dim):
        block_sq = block.square().sum(dim = dim, keepdim = keepdim)
        if sq is None:
            sq = block_sq
        else:
            sq += block_sq
    mean_sq = sq / n
    return mean_sq.sqrt()


def block_rms_n(x: torch.Tensor, dim: int = 0, blocksize: int = 32):
    """
    Compute blockwise x.square().mean().sqrt()
    """
    n = 0
    sq = None
    for block in torch.split(x, blocksize, dim = dim):
        block_sq = block.square().sum()
        n += block.numel()
        if sq is None:
            sq = block_sq
        else:
            sq += block_sq
    mean_sq = sq / n
    return mean_sq.sqrt()


def block_nmse(x: torch.Tensor, y: torch.Tensor, dim: int = 0, blocksize: int = 32):
    """
    Compute blockwise (x - y).square().mean().item() / y.square().mean().item()
    """
    sq = None
    diff_sq = None
    for block_x, block_y in zip(torch.split(x, blocksize, dim = dim), torch.split(y, blocksize, dim = dim)):
        block_sq = block_y.square().sum()
        block_diff_sq = (block_x - block_y).square().sum()
        if sq is None:
            sq = block_sq
            diff_sq = block_diff_sq
        else:
            sq += block_sq
            diff_sq += block_diff_sq
    return diff_sq.item() / (sq.item() + 1e-20)


_SHARED_INPUT_SCALES: dict = {}


def output_signs(n: int, device: torch.device, quant_args: dict) -> torch.Tensor:
    """Draw output signs from an optional transform-specific RNG stream."""

    if "sv_seed" in quant_args:
        torch.manual_seed(quant_args["sv_seed"])
    return (
        (torch.randn(n, device=device).sign() + 1e-5)
        .sign()
        .to(torch.float)
        .unsqueeze(0)
    )


def regularize(
    weight: torch.Tensor,
    su: torch.Tensor,
    sv: torch.Tensor,
    quant_args: dict,
    verbose: bool,
    H_diag: torch.Tensor | None,
    pb: ProgressBar | None,
    skip_g_scale: bool = False,
    q_fallback: bool = False
):
    """
    Transform weights into the distribution expected by EXL3 tile quantization.

    The routine chooses whether to apply output-channel scaling, folds output and input sign/scale vectors into the
    matrix, applies blockwise Hadamard transforms, and optionally searches for a global scale that minimizes sample
    quantization error. It returns the transformed weight plus the scale metadata needed to reconstruct the original
    linear layer behavior after quantization.
    """
    force_out_scales = quant_args["apply_out_scales"]

    # dist_ref = torch.empty((512,), dtype = torch.float, device = weight.device)
    # dist_r = torch.empty_like(dist_ref)
    def jsd(h1, h2):
        m = (h1 + h2) / 2
        eps = 1e-12
        js = F.kl_div((h1 + eps).log(), m, reduction = "sum") + \
             F.kl_div((h2 + eps).log(), m, reduction = "sum")
        return js / 2

    # From experiments, it seems the deciding factor in when scaling output channels is beneficial is when
    # the input to the linear layer is very irregular. After some testing, set the cutoff at 15% of the RMS sum
    # on 2% of the channels
    # TODO: More science
    if not q_fallback and H_diag is not None:
        diag = H_diag.sqrt()
        diag, _ = torch.sort(diag, descending = True)
        cutoff = diag.shape[0] // 50
        skew_factor = diag[:cutoff].sum() / diag.sum()
        if verbose:
            print(f"     - input state skew: {skew_factor.item():.6f}")

        if force_out_scales is None:
            apply_out_scales = skew_factor.item() < 0.15
        else:
            apply_out_scales = force_out_scales

    else:
        apply_out_scales = True if force_out_scales is None else force_out_scales

    if q_fallback:
        apply_out_scales = force_out_scales

    # Apply output scales
    out_channel_scales = block_rms(weight, dim = 0, keepdim = True)
    mean = out_channel_scales.mean().item()
    if mean > 1e-30:
        out_channel_scales /= mean
        quant_args["zeros"] = False
    else:
        quant_args["zeros"] = True
        if force_out_scales is not None:
            apply_out_scales = True
    zero_out_scales = out_channel_scales.abs() < 1e-30

    if apply_out_scales:
        out_channel_scales[zero_out_scales] = 0.1
        sv = (sv * out_channel_scales + 1e-10).float()
        if verbose:
            out_channel_std = out_channel_scales.std().item()
            out_channel_mean = out_channel_scales.mean().item()
            print(f"     - out ch scales std/mean: {out_channel_std:.6f}   {out_channel_mean:.6f}")

    # Output sign flips (and scales)
    weight /= sv

    # Force zero output channels to zero
    sv[zero_out_scales] = 0.0

    # Output hadamard transform
    blockwise_preapply_had_r_(weight, had_n)

    # Input sign flips and scales
    in_channel_scales = block_rms(weight, dim = 1, keepdim = True)
    in_channel_scales[in_channel_scales.abs() < 1e-30] = 0.1
    # kquant shared-su mode: reuse the first tensor's channel-scale profile for
    # every tensor with the same key (e.g. all experts of a MoE layer/matrix),
    # so the stored su vector is identical across them and TP replicas can
    # broadcast one copy. Trellis codes re-fit against the shared profile.
    shared_key = quant_args.get("shared_input_scales_key")
    if shared_key is not None:
        in_channel_scales = _SHARED_INPUT_SCALES.setdefault(
            shared_key, in_channel_scales)
    su = (su * in_channel_scales / (-codebook_scale) + 1e-10).float()  # mustn't be inplace
    weight /= su
    blockwise_preapply_had_l_(weight, had_k)

    # Determine best scale for matrix by test quantizing a sample of tiles along a wrapped diagonal
    if not skip_g_scale:
        g_scale, mse_scale = g_scale_gss(weight, False, quant_args, pb = pb)
    else:
        g_scale = 1.0
    weight, su, sv = apply_g_scale(
        weight,
        su,
        sv,
        g_scale,
        into_sv=bool(quant_args.get("g_scale_into_sv")),
    )

    # ext.test_distribution(weight_os, dist_r, dist_ref, -3.8, 3.8)
    # js_os = jsd(dist_r, dist_ref)

    if verbose:
        print(f"     - su/sv std: {su.std().item():.6f}   {sv.std().item():.6f}")
        print(f"     - global scale: {g_scale:.6f}")
        print(f"     - sample mse: {mse_scale.item():.6f}")
        print(f"     - apply_out_scales: {str(apply_out_scales)}")

    return apply_out_scales, weight, g_scale, su, sv


def relocate_g_scale_to_sv(
    su: torch.Tensor, sv: torch.Tensor, g_scale: float
):
    """Move regularize's global scale from ``su`` to ``sv`` exactly.

    ``regularize`` returns ``su = base_su / g_scale``.  Some serving layouts
    share ``su`` across many tensors while keeping ``sv`` tensor-local.  This
    relocation restores ``base_su`` and divides ``sv`` by the same factor, so
    the outer-product dequantization scale is unchanged.
    """

    if not math.isfinite(float(g_scale)) or float(g_scale) <= 0.0:
        raise ValueError("g_scale must be positive and finite")
    return su * g_scale, sv / g_scale


def apply_g_scale(
    weight: torch.Tensor,
    su: torch.Tensor,
    sv: torch.Tensor,
    g_scale: float,
    *,
    into_sv: bool = False,
):
    """Apply the searched scale without perturbing a shared ``su`` tensor.

    Moving a scale from ``su`` to ``sv`` after first dividing ``su`` is only
    algebraically exact: the divide/multiply round trip can change a float32
    value by one ULP and, at a rounding boundary, change the packed FP16
    value.  When ``into_sv`` is set, leave ``su`` untouched from the outset
    and put the inverse scale directly into the tensor-local ``sv`` vector.
    """

    if not math.isfinite(float(g_scale)) or float(g_scale) <= 0.0:
        raise ValueError("g_scale must be positive and finite")
    weight *= g_scale
    if into_sv:
        sv /= g_scale
    else:
        su /= g_scale
    return weight, su, sv


def quantize_qsrt(
    weight: torch.Tensor,
    H_data: dict,
    quant_args: dict,
    return_weight_q: bool,
    progress_str: str | None = None,
    verbose: bool = False,
    swap_to_device: torch.device | None = None,
    save_reg: str = None
):
    """
    :param weight:
        Input tensor, row major shape (in_features, out_features)

    :param H_data:
        Dictionary of hessian tensor and related data, as collected by Linear wrapper class. May be reused between
        linear layers with the same input (e.g. Q, K and V projections)

    :param quant_args:
        dict:
         - K: bitrate
         - seed: integer seed for random sign flips etc.
         - sigma_reg: regularization factor
         - mixed_rate_axis: optional ``"k"`` or ``"n"`` tile-rate axis
         - mixed_tile_bits: optional K2/K3/K4 value per tile on that axis
         - pack_trellis_fn: optional callback for a heterogeneous payload

    :param return_weight_q:
        Return quantized weight

    :param progress_str:
        Show progress bar during quantization

    :param verbose:
        Dump extra stats

    :param swap_to_device:
        If input tensor is on CPU, move to this device before quantization

    :param save_reg:
        Save regularized tensor as image to the provided path

    :return:
        tuple:
          - quantized weight
          - proxy error: trace(err @ H @ err.T) / (W @ H @ W.T)
          - quantized and packed tensors
    """

    progress_text = None if not progress_str else progress_str.replace("<step>", "Preparing")
    with (ProgressBar(progress_text, 100) as pb):

        assert weight.dtype == torch.float
        tiles_k = weight.shape[0] // 16

        if "seed" in quant_args:
            torch.manual_seed(quant_args["seed"])

        devices = quant_args["devices"]
        if weight.device != torch.device(devices[0]):
            weight = weight.to(devices[0])

        device = weight.device if swap_to_device is None else swap_to_device
        k, n = weight.shape

        # Get H, LDL decomp. and input/output sign flips
        q_fallback, H, L, su, H_diag = finalize_capture_H(H_data, quant_args, verbose)
        if H is not None and H.is_cuda:
            H = H.to(device)
        if L is not None and L.is_cuda:
            L = L.to(device)
        if su.is_cuda:
            su = su.to(device)
        if H_diag is not None and H_diag.is_cuda:
            H_diag = H_diag.to(device)
        sv = output_signs(n, device, quant_args)

        # Move stored L to CPU (if not already), move working L to device
        if H_data["L"] is not None:
            H_data["L"] = H_data["L"].cpu()
        if L is not None:
            L = L.to(device)

        if swap_to_device is not None:
            weight = weight.to(swap_to_device)
        if verbose:
            weight_copy = weight.cpu()
        weight_r = weight
        del weight

        if verbose:
            rms = block_rms_n(weight_r, dim = 0)
            print(f"     - input tensor rms: {rms:.6f}")

        # Regularization
        apply_out_scales, weight_r, g_scale, su, sv = regularize(
            weight_r,
            su,
            sv,
            quant_args,
            verbose,
            H_diag,
            pb,
            q_fallback = q_fallback
        )
        if save_reg:
            save_tensor_image(weight_r, save_reg)

        if verbose:
            rms = weight_r.square().mean().sqrt()
            print(f"     - regularized rms:  {rms:.6f}")

        progress_text = None if not progress_str else progress_str.replace("<step>", "Quantizing")
        pb.update(0)
        pb.new_task(progress_text, tiles_k)

        # Select device for work buffers (CPU is slower for small tensors but saves a lot of VRAM on big ones)
        # TODO: Use pynvml or mem_get_info to predict whether CPU buffer is needed
        if weight_r.numel() > 5e8:
            weight_r = weight_r.cpu()

        # Quantize
        mixed_rate = (
            "mixed_rate_axis" in quant_args or "mixed_tile_bits" in quant_args
        )
        if not q_fallback:
            if mixed_rate:
                weight_q, encoded_q = ldlq_mixed(weight_r, L, quant_args, pb)
            else:
                weight_q, encoded_q = ldlq(weight_r, L, quant_args, pb)  #zxc
            del L
        else:
            if mixed_rate:
                raise ValueError(
                    "mixed-rate quantization requires a non-fallback dense Hessian"
                )
            weight_q, encoded_q, mse_err = fallback_quant(weight_r, device, quant_args, pb)  # zxc

        pb.update(tiles_k)

        # Metrics
        if not q_fallback:
            try:
                E = weight_r - weight_q  # may run on CPU
                W = weight_r
                Hd = H.to(device)
                weight_r = None
                E = E.to(device)
                num = block_trace(E, Hd)
                E = None
                W = W.to(device)
                den = block_trace(W, Hd)
                W = None
                Hd = None
                proxy_err = num / max(den, 1e-8)
            except torch.OutOfMemoryError:
                weight_r = None
                E = None
                W = None
                Hd = None
                proxy_err = -1.0
        else:
            proxy_err = mse_err

        # free_mem()

        if return_weight_q or verbose:
            weight_q = weight_q.to(device)
            weight_q = preapply_had_l(weight_q, had_k)
            weight_q *= su
            weight_q = preapply_had_r(weight_q, had_n)
            weight_q *= sv

            if verbose:
                weight = weight_copy.to(device)
                nmse = block_nmse(weight_q, weight)
                print(f"     - quant nmse: {nmse:.6f}")

        # Compile packed tensor
        suh = su.flatten().contiguous().to(dtype = torch.half, copy = True)
        svh = sv.flatten().contiguous().to(dtype = torch.half, copy = True)
        pack_trellis_fn = quant_args.get("pack_trellis_fn", pack_trellis)
        trellis = pack_trellis_fn(encoded_q.to(device), quant_args)

        out_tensors = {
            # "scale": weight_scale.to(dtype = torch.float, copy = True),
            # "su": pack_signs(su, quant_args),
            "suh": suh,
            # "sv": pack_signs(sv, quant_args),
            "svh": svh,
            "trellis": trellis,
        }

        # Safetensors doesn't know what to do with a torch.uint32 tensor. Anyway, since the multipliers are now
        # locked, the values in these tensors are never read, but they need to be present in the model files to
        # indicate which codebook to use during inference, per individual tensor.
        if quant_args.get("mcg"):
            out_tensors.update({
                "mcg": torch.tensor(codebook_mcg_mult, dtype = torch.uint32).view(torch.int)
            })
        if quant_args.get("mul1"):
            out_tensors.update({
                "mul1": torch.tensor(codebook_mul1_mult, dtype = torch.uint32).view(torch.int)
            })
        quant_args.update({
            "apply_out_scales": apply_out_scales,
            "g_scale": g_scale,
            "q_fallback": q_fallback,
        })

    return weight_q, proxy_err, out_tensors


def quantize_qsrt_batch(
    weights: list[torch.Tensor],
    H_datas: list[dict],
    quant_args_groups: list[list[dict]],
    *,
    return_weight_q: bool = True,
    verbose: bool = False,
):
    """Prepare each source once and batch all of its mixed-rate candidates.

    The ordinary API repeats output signs, channel scaling, Hadamards, global
    scale search, and the complete LDLQ launch loop for every rate map.  This
    API accepts one group of rate maps per source matrix, performs the common
    preparation once, and sends all candidates through
    :func:`ldlq_mixed_batched`.

    Results are returned as one list per source.  Each result contains the
    unpacked trellis states, persisted FP16 scale vectors, the proxy error, and
    (when requested) a reconstruction made with those persisted scales.  It
    intentionally does not pack a physical bitstream; callers can score all
    candidates cheaply and close only the selected payload.
    """

    source_count = len(weights)
    if source_count <= 0 or len(H_datas) != source_count or len(quant_args_groups) != source_count:
        raise ValueError("QSRT batch inputs must have equal nonzero lengths")
    if any(not group for group in quant_args_groups):
        raise ValueError("every QSRT source needs at least one rate candidate")
    if any(weight.ndim != 2 or weight.dtype != torch.float for weight in weights):
        raise TypeError("QSRT batch weights must be float32 matrices")
    shape = weights[0].shape
    if any(weight.shape != shape for weight in weights):
        raise ValueError("QSRT batch sources must have one matrix shape")

    preparation_keys = (
        "K",
        "seed",
        "sv_seed",
        "sigma_reg",
        "devices",
        "device_ratios",
        "apply_out_scales",
        "mcg",
        "mul1",
        "shared_input_scales_key",
        "g_scale_into_sv",
        "buf_size_k",
        "ldlq_tf32",
        "tailbite_context",
        "g_scale_override",
    )
    for group in quant_args_groups:
        reference = group[0]
        for args in group:
            mixed_rate_spec(shape[0], shape[1], args)
            for key in preparation_keys:
                if args.get(key) != reference.get(key):
                    raise ValueError(
                        f"QSRT candidates changed preparation argument {key}"
                    )

    devices = quant_args_groups[0][0]["devices"]
    device = torch.device(devices[0])
    if any(group[0].get("devices") != devices for group in quant_args_groups):
        raise ValueError("QSRT source groups must share devices")
    if any(weight.device != device for weight in weights):
        raise ValueError("QSRT batch weights must already be on the quant device")

    prepared = []
    for source, (weight, H_data, group) in enumerate(
        zip(weights, H_datas, quant_args_groups)
    ):
        args = group[0]
        if "seed" in args:
            torch.manual_seed(args["seed"])
        q_fallback, H, L, su, H_diag = finalize_capture_H(H_data, args, verbose)
        if q_fallback or H is None or L is None:
            raise ValueError("mixed-rate batching requires a non-fallback dense Hessian")
        H = H.to(device)
        L = L.to(device)
        su = su.to(device)
        if H_diag is not None:
            H_diag = H_diag.to(device)
        if H_data["L"] is not None:
            H_data["L"] = H_data["L"].cpu()
        sv = output_signs(weight.shape[1], device, args)
        apply_out_scales, weight_r, _, su, sv = regularize(
            weight,
            su,
            sv,
            args,
            verbose,
            H_diag,
            None,
            skip_g_scale=True,
            q_fallback=False,
        )
        prepared.append(
            {
                "source": source,
                "H": H,
                "L": L,
                "weight_r": weight_r,
                "su": su,
                "sv": sv,
                "apply_out_scales": apply_out_scales,
            }
        )

    # One batched global-scale search still returns an independent optimum for
    # every source and is the same operation used by the serial search.  A
    # forced value is an experiment hook for path-aware scale-oracle studies;
    # it remains source-local and is folded into the ordinary persisted scale
    # vectors exactly like a searched value.
    scales: list[tuple[float, torch.Tensor | None] | None] = [None] * source_count
    search_sources = []
    search_samples = []
    for source, (item, group) in enumerate(zip(prepared, quant_args_groups)):
        override = group[0].get("g_scale_override")
        if override is None:
            search_sources.append(source)
            search_samples.append(sample_scale_tiles(item["weight_r"]))
        else:
            if (
                isinstance(override, bool)
                or not isinstance(override, (int, float))
                or not math.isfinite(float(override))
                or float(override) <= 0.0
            ):
                raise ValueError("g_scale_override must be positive and finite")
            scales[source] = (float(override), None)
    if search_samples:
        searched = g_scale_search_batch(
            search_samples, quant_args_groups[search_sources[0]][0]
        )
        for source, result in zip(search_sources, searched):
            scales[source] = result
    del search_samples
    if any(result is None for result in scales):
        raise AssertionError("global scale search did not cover every source")
    for item, group, result in zip(prepared, quant_args_groups, scales):
        assert result is not None
        g_scale, _ = result
        item["weight_r"], item["su"], item["sv"] = apply_g_scale(
            item["weight_r"],
            item["su"],
            item["sv"],
            g_scale,
            into_sv=bool(group[0].get("g_scale_into_sv")),
        )
        item["g_scale"] = g_scale

    flat_members = [
        (source, candidate)
        for source, group in enumerate(quant_args_groups)
        for candidate in range(len(group))
    ]
    can_reuse_output_tiles = all(
        all(
            mixed_rate_spec(shape[0], shape[1], args)[0] == "n"
            for args in group
        )
        and mixed_rate_spec(shape[0], shape[1], group[0])[1]
        == (3,) * (shape[1] // 16)
        for group in quant_args_groups
    )
    if can_reuse_output_tiles:
        source_weights = torch.stack([item["weight_r"] for item in prepared])
        source_Ls = torch.stack([item["L"] for item in prepared])
        weights_q, encoded_q = ldlq_mixed_n_candidates_reuse(
            source_weights,
            source_Ls,
            quant_args_groups,
        )
        del source_Ls
    elif any(len(group) > 1 for group in quant_args_groups) and all(
        all(
            mixed_rate_spec(shape[0], shape[1], args)[0] == "k"
            for args in group
        )
        for group in quant_args_groups
    ):
        source_weights = torch.stack([item["weight_r"] for item in prepared])
        Ls_batch = torch.stack([item["L"] for item in prepared])
        weights_q, encoded_q = ldlq_mixed_k_candidates_prefix_reuse(
            source_weights,
            Ls_batch,
            quant_args_groups,
        )
        del Ls_batch
    else:
        source_weights = None
        weights_batch = torch.stack(
            [prepared[source]["weight_r"] for source, _ in flat_members]
        )
        Ls_batch = torch.stack([prepared[source]["L"] for source, _ in flat_members])
        flat_args = [
            quant_args_groups[source][candidate]
            for source, candidate in flat_members
        ]
        weights_q, encoded_q = ldlq_mixed_batched(
            weights_batch, Ls_batch, flat_args
        )
        del Ls_batch

    # Dense-H proxy values are descriptive evidence; the Viterbi traversal is
    # already complete.  Reading every column-block scalar with ``.item()``
    # here used to introduce thousands of serial GPU synchronizations per
    # expert batch.  Queue the identical block terms, transfer them together,
    # then sum each row in the same Python order as ``block_trace``.
    denominator_parts = [
        block_trace_parts(item["weight_r"], item["H"])
        for item in prepared
    ]
    numerator_parts = []
    pending_results = []
    for flat, (source, candidate) in enumerate(flat_members):
        item = prepared[source]
        args = quant_args_groups[source][candidate]
        source_weight = (
            source_weights[source]
            if source_weights is not None
            else weights_batch[flat]
        )
        error = source_weight - weights_q[flat]
        numerator_parts.append(block_trace_parts(error, item["H"]))
        del error
        suh = item.get("suh")
        svh = item.get("svh")
        if suh is None or svh is None:
            suh = item["su"].flatten().contiguous().to(dtype=torch.half, copy=True)
            svh = item["sv"].flatten().contiguous().to(dtype=torch.half, copy=True)
            item["suh"] = suh
            item["svh"] = svh
        reconstruction = None
        if return_weight_q:
            reconstruction = preapply_had_l(weights_q[flat], had_k)
            reconstruction *= suh.float().unsqueeze(1)
            reconstruction = preapply_had_r(reconstruction, had_n)
            reconstruction *= svh.float().unsqueeze(0)
            reconstruction = reconstruction.contiguous()
        args.update(
            {
                "apply_out_scales": item["apply_out_scales"],
                "g_scale": item["g_scale"],
                "q_fallback": False,
                "zeros": quant_args_groups[source][0].get("zeros", False),
            }
        )
        pending_results.append(
            (source, reconstruction, encoded_q[flat].contiguous(), suh, svh)
        )

    trace_rows = torch.stack((*denominator_parts, *numerator_parts)).cpu()
    trace_values = []
    for row in trace_rows:
        total = 0.0
        for partial in row:
            total += partial.item()
        trace_values.append(total)
    denominators = trace_values[:source_count]
    numerators = trace_values[source_count:]

    results = [[] for _ in range(source_count)]
    for numerator, pending in zip(numerators, pending_results):
        source, reconstruction, encoded, suh, svh = pending
        proxy_err = numerator / max(denominators[source], 1e-8)
        results[source].append(
            {
                "weight_q": reconstruction,
                "encoded": encoded,
                "suh": suh,
                "svh": svh,
                "proxy": proxy_err,
                "g_scale": prepared[source]["g_scale"],
            }
        )
    return results


def quantize_uniform_batch(
    weights: list[torch.Tensor],
    H_datas: list[dict],
    quant_args_list: list[dict],
    progress_str: str | None = None,
    verbose: bool = False,
):
    """
    Quantize a group of same-K linears together, batching the scale search and the LDLQ recursion so the
    quantize_tiles kernel sees large batches instead of one small tensor's worth of tiles per step.

    Two layouts, chosen by whether the H_data dicts are the same object:
      - shared Hessian (e.g. all gate/up projections of a block-sparse MLP, or fused q/k/v): the regularized
        weights are concatenated along out_features and run through a single ldlq() pass with the shared L.
        This is exact: LDLQ treats columns independently given L, and su (drawn per qmap) is identical
      - distinct Hessians over same-shape tensors (e.g. per-expert down projections): weights are stacked and
        run through ldlq_batched() with per-tensor L

    RNG is re-seeded per tensor (finalize's su draw on first call for the qmap, then the per-tensor sv draw),
    so per-tensor results should match quantize_qsrt() in the wider matmuls. Tensors whose Hessian fails to
    finalize (q_fallback) are routed through quantize_qsrt() individually.

    :return:
        list of (proxy_err, out_tensors) per input tensor; quant_args_list entries are updated in place like
        quantize_qsrt updates quant_args
    """
    if any(
        "mixed_rate_axis" in quant_args or "mixed_tile_bits" in quant_args
        for quant_args in quant_args_list
    ):
        raise ValueError(
            "mixed-rate LDLQ is currently supported by quantize_qsrt, not the "
            "same-K batch API"
        )
    n_t = len(weights)
    qa0 = quant_args_list[0]
    devices = qa0["devices"]
    device = torch.device(devices[0])
    shared_H = all(hd is H_datas[0] for hd in H_datas)
    assert shared_H or all(w.shape == weights[0].shape for w in weights)
    assert all(w.shape[0] == weights[0].shape[0] for w in weights)
    assert all(qa["K"] == qa0["K"] for qa in quant_args_list)

    size_k = weights[0].shape[0]
    tiles_k = size_k // 16
    results: list = [None] * n_t

    progress_text = None if not progress_str else progress_str.replace("<step>", "Preparing")
    with ProgressBar(progress_text, 100) as pb:

        # Finalize Hessians, replicating the serial path's per-tensor RNG stream. Fallback tensors are
        # handled individually by quantize_qsrt
        finalized = []
        batch_idx = []
        for t in range(n_t):
            qa = quant_args_list[t]
            if "seed" in qa:
                torch.manual_seed(qa["seed"])
            q_fallback, H, L, su, H_diag = finalize_capture_H(H_datas[t], qa, verbose)
            if q_fallback:
                finalized.append(None)
            else:
                finalized.append((H, L, su, H_diag))
                batch_idx.append(t)

        for t in range(n_t):
            if finalized[t] is None:
                results[t] = quantize_qsrt(weights[t], H_datas[t], quant_args_list[t], False, None, verbose)[1:]

        if not batch_idx:
            return results

        # Regularize each tensor with the scale search deferred, then search all scales in one batch
        regs = {}
        for t in batch_idx:
            qa = quant_args_list[t]
            if "seed" in qa:
                torch.manual_seed(qa["seed"])
            H, L, su, H_diag = finalized[t]
            weight = weights[t]
            if weight.device != device:
                weight = weight.to(device)
            if su.is_cuda:
                su = su.to(device)
            if H_diag is not None and H_diag.is_cuda:
                H_diag = H_diag.to(device)
            sv = output_signs(weight.shape[1], device, qa)
            apply_out_scales, weight_r, _, su, sv = regularize(
                weight, su, sv, qa, verbose, H_diag, None, skip_g_scale = True)
            regs[t] = [weight_r, su, sv, apply_out_scales]
            weights[t] = None

        samples = [sample_scale_tiles(regs[t][0]) for t in batch_idx]
        scales = g_scale_search_batch(samples, qa0)
        del samples
        g_scales = {}
        for t, (g_scale, _) in zip(batch_idx, scales):
            regs[t][0] *= g_scale
            # kquant shared-su mode: fold the per-tensor global scale into the
            # sv vector (sharded under TP for gate/up) instead of su
            # (replicated), keeping su identical across experts.
            if quant_args_list[t].get("g_scale_into_sv"):
                regs[t][2] /= g_scale
            else:
                regs[t][1] /= g_scale
            g_scales[t] = g_scale
        pb.update(100)

        progress_text = None if not progress_str else progress_str.replace("<step>", "Quantizing")
        pb.new_task(progress_text, tiles_k)

        # Quantize
        if shared_H:
            L = finalized[batch_idx[0]][1].to(device)
            widths = [regs[t][0].shape[1] for t in batch_idx]
            weight_r_cat = torch.cat([regs[t][0] for t in batch_idx], dim = 1)
            for t in batch_idx:
                regs[t][0] = None
            weight_q_cat, encoded_cat = ldlq(weight_r_cat, L, qa0, pb)
            del L
            weight_rs = list(torch.split(weight_r_cat, widths, dim = 1))
            weight_qs = list(torch.split(weight_q_cat, widths, dim = 1))
            encodeds = list(torch.split(encoded_cat, [w // 16 for w in widths], dim = 1))
        else:
            Ls = torch.stack([finalized[t][1].to(device) for t in batch_idx])
            weight_r_stack = torch.stack([regs[t][0] for t in batch_idx])
            for t in batch_idx:
                regs[t][0] = None
            weight_q_stack, encoded_stack = ldlq_batched(weight_r_stack, Ls, qa0, pb)
            del Ls
            weight_rs = list(weight_r_stack.unbind(0))
            weight_qs = list(weight_q_stack.unbind(0))
            encodeds = list(encoded_stack.unbind(0))

        pb.update(tiles_k)

        # Per-tensor metrics and packing
        Hd = None
        for bi, t in enumerate(batch_idx):
            qa = quant_args_list[t]
            _, su, sv, apply_out_scales = regs[t]
            if shared_H:
                if Hd is None:
                    Hd = finalized[t][0].to(device)
            else:
                Hd = finalized[t][0].to(device)
            try:
                E = weight_rs[bi] - weight_qs[bi]
                num = block_trace(E, Hd)
                E = None
                den = block_trace(weight_rs[bi], Hd)
                proxy_err = num / max(den, 1e-8)
            except torch.OutOfMemoryError:
                E = None
                proxy_err = -1.0

            # Optional calibration metrics for allocation experiments.  Keep
            # the public return value unchanged; callers opt in through their
            # per-tensor quant_args and receive the metrics there.  The dense-H
            # numerator is measured in the encoder's regularized coordinate
            # system.  Reconstructing both weights lets callers optionally
            # evaluate the actual error against an untouched canonical
            # Hessian, and also exposes a per-input-channel residual vector
            # that can be reweighted by expert-conditional moments without
            # requantizing the candidate.
            if qa.get("return_error_metrics") and proxy_err >= 0.0:
                wr = weight_rs[bi].to(device)
                wq = weight_qs[bi].to(device)
                wr = preapply_had_l(wr, had_k)
                wq = preapply_had_l(wq, had_k)
                wr *= su
                wq *= su
                wr = preapply_had_r(wr, had_n)
                wq = preapply_had_r(wq, had_n)
                wr *= sv
                wq *= sv
                canonical_error = wr - wq
                residual_by_input = canonical_error.square().sum(dim=1).cpu()
                error_hessian = qa.get("error_hessian")
                if error_hessian is None:
                    canonical_num = num
                    canonical_den = den
                else:
                    if error_hessian.shape != (wr.shape[0], wr.shape[0]):
                        raise ValueError(
                            "error_hessian shape does not match quantized input "
                            f"dimension: {error_hessian.shape} vs {wr.shape[0]}"
                        )
                    error_hessian = error_hessian.to(device)
                    canonical_num = block_trace(canonical_error, error_hessian)
                    canonical_den = block_trace(wr, error_hessian)
                qa["error_metrics"] = {
                    "numerator": float(canonical_num),
                    "denominator": float(canonical_den),
                    "encoder_numerator": float(num),
                    "encoder_denominator": float(den),
                    "residual_by_input": residual_by_input,
                }
                del wr, wq, canonical_error, residual_by_input
            weight_rs[bi] = None
            weight_qs[bi] = None

            suh = su.flatten().contiguous().to(dtype = torch.half, copy = True)
            svh = sv.flatten().contiguous().to(dtype = torch.half, copy = True)
            trellis = pack_trellis(encodeds[bi].contiguous(), qa)
            encodeds[bi] = None

            out_tensors = {
                "suh": suh,
                "svh": svh,
                "trellis": trellis,
            }
            if qa.get("mcg"):
                out_tensors.update({
                    "mcg": torch.tensor(codebook_mcg_mult, dtype = torch.uint32).view(torch.int)
                })
            if qa.get("mul1"):
                out_tensors.update({
                    "mul1": torch.tensor(codebook_mul1_mult, dtype = torch.uint32).view(torch.int)
                })
            qa.update({
                "apply_out_scales": apply_out_scales,
                "g_scale": g_scales[t],
                "q_fallback": False,
            })
            results[t] = (proxy_err, out_tensors)

    return results
