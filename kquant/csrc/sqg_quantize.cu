#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

#include "util.cuh"
#include "quant/codebook.cuh"

// Keep only the table pointer in broadcast-friendly constant memory.  The
// labels themselves remain in ordinary read-only global memory: a warp's
// Viterbi lanes normally request different states, which would serialize a
// 64-KiB constant-memory table despite its attractive nominal footprint.
__device__ __constant__ const uint8_t* sqg_e4m3_lut;

template <>
__device__ inline half decode_3inst<4>(uint32_t state)
{
    const __nv_fp8_storage_t fp8 = sqg_e4m3_lut[state & 0xffffu];
    const __half_raw raw = __nv_cvt_fp8_to_halfraw(fp8, __NV_E4M3);
    return __ushort_as_half(raw.x);
}

template <>
__device__ inline half2 decode_3inst_2<4>(uint32_t state0, uint32_t state1)
{
    return __halves2half2(decode_3inst<4>(state0), decode_3inst<4>(state1));
}

#include "quant/quantize_tiles_kernel.cuh"

namespace
{

template <int K>
void launch_sqg(
    const float* input,
    float* output,
    uint16_t* indices,
    half* costs,
    uint8_t* edges,
    int tiles,
    int max_batch,
    int shared_memory,
    int tailbite_context,
    cudaStream_t stream)
{
    const int threads = K == 2 ? 1024 : 512;
    cudaFuncSetAttribute(
        quantize_tiles_kernel<K, 4>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_memory);
    cudaFuncSetAttribute(
        quantize_tiles_kernel<K, 4>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared);
    for (int start = 0; start < tiles; start += max_batch)
    {
        const int batch = min(max_batch, tiles - start);
        quantize_tiles_kernel<K, 4><<<batch, threads, shared_memory, stream>>>(
            input + 256 * start,
            output + 256 * start,
            indices + 256 * start,
            costs,
            edges,
            tailbite_context);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

}  // namespace

void quantize_tiles_sqg_cuda(
    at::Tensor input_tiles,
    at::Tensor output_tiles,
    at::Tensor output_indices,
    at::Tensor temp_costs,
    at::Tensor temp_edges,
    at::Tensor codebook,
    int64_t K,
    int64_t tailbite_context)
{
    const at::cuda::OptionalCUDAGuard guard(input_tiles.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(input_tiles.is_cuda(), "input_tiles must be CUDA");
    TORCH_CHECK(input_tiles.scalar_type() == at::kFloat, "input_tiles must be FP32");
    TORCH_CHECK(input_tiles.dim() == 2 && input_tiles.size(1) == 256,
                "input_tiles must have shape [tiles, 256]");
    TORCH_CHECK(output_tiles.sizes() == input_tiles.sizes(),
                "output_tiles shape mismatch");
    TORCH_CHECK(output_tiles.scalar_type() == at::kFloat,
                "output_tiles must be FP32");
    TORCH_CHECK(output_indices.sizes() == input_tiles.sizes(),
                "output_indices shape mismatch");
    TORCH_CHECK(output_indices.scalar_type() == at::kShort,
                "output_indices must be int16");
    TORCH_CHECK(K >= 2 && K <= 4, "SQG validation kernel supports K2--K4");
    TORCH_CHECK(tailbite_context >= 1 && tailbite_context <= 128,
                "tailbite_context must be in 1..128");
    TORCH_CHECK(codebook.is_cuda() && codebook.device() == input_tiles.device(),
                "codebook must be on the input CUDA device");
    TORCH_CHECK(codebook.scalar_type() == at::kByte && codebook.numel() == 65536,
                "codebook must contain 65,536 uint8 E4M3 values");
    TORCH_CHECK(codebook.is_contiguous(), "codebook must be contiguous");

    const int edges_per_state = 65536 >> K;
    const int decision_entries = K <= 4 ? edges_per_state / 2 : edges_per_state;
    TORCH_CHECK(temp_costs.scalar_type() == at::kHalf && temp_costs.dim() == 3,
                "temp_costs must be rank-3 FP16");
    TORCH_CHECK(temp_costs.size(1) == 2 && temp_costs.size(2) == edges_per_state,
                "temp_costs shape mismatch");
    TORCH_CHECK(temp_edges.scalar_type() == at::kByte && temp_edges.dim() == 3,
                "temp_edges must be rank-3 uint8");
    TORCH_CHECK(temp_edges.size(1) == 256 && temp_edges.size(2) == decision_entries,
                "temp_edges shape mismatch");

    const uint8_t* codebook_pointer = codebook.data_ptr<uint8_t>();
    C10_CUDA_CHECK(cudaMemcpyToSymbolAsync(
        sqg_e4m3_lut,
        &codebook_pointer,
        sizeof(codebook_pointer),
        0,
        cudaMemcpyHostToDevice,
        stream));

    int device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    int multiprocessors = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &multiprocessors, cudaDevAttrMultiProcessorCount, device));
    const int max_batch = min(
        static_cast<int>(temp_costs.size(0)), 3 * multiprocessors);
    const int shared_memory =
        2 * edges_per_state * static_cast<int>(sizeof(half)) + 512 + 64 + 128;
    const int tiles = static_cast<int>(input_tiles.size(0));
    if (!tiles) return;

    switch (K)
    {
        case 2:
            launch_sqg<2>(
                input_tiles.data_ptr<float>(), output_tiles.data_ptr<float>(),
                reinterpret_cast<uint16_t*>(output_indices.data_ptr<int16_t>()),
                reinterpret_cast<half*>(temp_costs.data_ptr<at::Half>()),
                temp_edges.data_ptr<uint8_t>(), tiles, max_batch, shared_memory,
                static_cast<int>(tailbite_context), stream);
            break;
        case 3:
            launch_sqg<3>(
                input_tiles.data_ptr<float>(), output_tiles.data_ptr<float>(),
                reinterpret_cast<uint16_t*>(output_indices.data_ptr<int16_t>()),
                reinterpret_cast<half*>(temp_costs.data_ptr<at::Half>()),
                temp_edges.data_ptr<uint8_t>(), tiles, max_batch, shared_memory,
                static_cast<int>(tailbite_context), stream);
            break;
        case 4:
            launch_sqg<4>(
                input_tiles.data_ptr<float>(), output_tiles.data_ptr<float>(),
                reinterpret_cast<uint16_t*>(output_indices.data_ptr<int16_t>()),
                reinterpret_cast<half*>(temp_costs.data_ptr<at::Half>()),
                temp_edges.data_ptr<uint8_t>(), tiles, max_batch, shared_memory,
                static_cast<int>(tailbite_context), stream);
            break;
    }
}
