#include <torch/extension.h>

void quantize_tiles_sqg_cuda(
    at::Tensor input_tiles,
    at::Tensor output_tiles,
    at::Tensor output_indices,
    at::Tensor temp_costs,
    at::Tensor temp_edges,
    at::Tensor codebook,
    int64_t K,
    int64_t tailbite_context);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
    module.def(
        "quantize_tiles_sqg",
        &quantize_tiles_sqg_cuda,
        "Quantize EXL L16 tiles against a supplied E4M3 SQG table");
}

