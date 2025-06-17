#include <cstdint>
#include "conv3d_implicit_gemm.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/script.h>

static std::vector<std::unique_ptr<Conv3DImplicitGemmKernel>> kernels;


void setup_kernels(
    std::string ptx_dir
) {
    kernels = setup(ptx_dir);
    TORCH_CHECK(!kernels.empty(), "No kernels were loaded from the provided PTX directory");
}

torch::Tensor conv3d_implicit_gemm_torch(
    torch::Tensor features, // [N, D]
    torch::Tensor indices, // [N', K**3]
    torch::Tensor weights, // [K**3, D, D']
    int64_t K,
    int64_t kernel_idx
) {
    TORCH_CHECK(features.is_cuda(),  "features must be a CUDA tensor");
    TORCH_CHECK(indices.is_cuda() , "indices must be a CUDA tensor");
    TORCH_CHECK(weights.is_cuda() , "weights must be a CUDA tensor");


    int N = features.size(0);
    int D = features.size(1);
    int NPrime = indices.size(0);
    int DPrime = weights.size(2);

    auto output = torch::zeros({NPrime, DPrime}, features.options());

    auto kernel = kernels[kernel_idx].get();
    auto dtype = kernel->get_dtype();

    CUdeviceptr features_ptr;
    CUdeviceptr indices_ptr;
    CUdeviceptr weights_ptr;
    CUdeviceptr output_ptr;

    if (dtype == "fp16") {
        TORCH_CHECK(features.dtype() == torch::kHalf, "features must be of type torch::kHalf");
        TORCH_CHECK(weights.dtype() == torch::kHalf, "weights must be of type torch::kHalf");
        features_ptr = reinterpret_cast<CUdeviceptr>(features.data_ptr<at::Half>());
        indices_ptr = reinterpret_cast<CUdeviceptr>(indices.data_ptr<int32_t>());
        weights_ptr = reinterpret_cast<CUdeviceptr>(weights.data_ptr<at::Half>());
        output_ptr = reinterpret_cast<CUdeviceptr>(output.data_ptr<at::Half>());
    } else if (dtype == "fp32") {
        TORCH_CHECK(features.dtype() == torch::kFloat, "features must be of type torch::kFloat");
        TORCH_CHECK(weights.dtype() == torch::kFloat, "weights must be of type torch::kFloat");
        features_ptr = reinterpret_cast<CUdeviceptr>(features.data_ptr<float>());
        indices_ptr = reinterpret_cast<CUdeviceptr>(indices.data_ptr<int32_t>());
        weights_ptr = reinterpret_cast<CUdeviceptr>(weights.data_ptr<float>());
        output_ptr = reinterpret_cast<CUdeviceptr>(output.data_ptr<float>());
    } else {
        throw std::runtime_error("Unsupported dtype: " + dtype);
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    kernel->run(
        features_ptr, indices_ptr, weights_ptr, output_ptr,
        N, NPrime, D, DPrime, K, stream
    );

    return output;
}

int64_t get_num_kernels() {
    return kernels.size();
}

TORCH_LIBRARY(conv3d_implicit_gemm, m) {
    m.def("setup_kernels", &setup_kernels);
    m.def("get_num_kernels", &get_num_kernels);
    m.def("conv3d_implicit_gemm_torch", &conv3d_implicit_gemm_torch);
}

