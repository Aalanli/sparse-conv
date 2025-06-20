#include <cstdint>
#include "conv3d_implicit_gemm.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <memory>
#include <torch/script.h>


thread_local std::unique_ptr<Conv3DKernels> kernels = nullptr;

void setup_kernels(
    std::string ptx_dir
) {
    kernels = std::make_unique<Conv3DKernels>(ptx_dir);
}

void save_kernel_map(std::string kernel_map_file) {
    kernels->save_kernel_map(kernel_map_file);
}

torch::Tensor conv3d_implicit_gemm_torch(
    torch::Tensor features, // [N, D]
    torch::Tensor indices, // [N', K**3]
    torch::Tensor weights, // [K**3, D, D']
    int64_t K,
    std::string acc_dtype
) {
    TORCH_CHECK(features.is_cuda(),  "features must be a CUDA tensor");
    TORCH_CHECK(indices.is_cuda() , "indices must be a CUDA tensor");
    TORCH_CHECK(weights.is_cuda() , "weights must be a CUDA tensor");
    TORCH_CHECK(kernels != nullptr, "Kernels must be set up before calling conv3d_implicit_gemm_torch");


    int N = features.size(0);
    int D = features.size(1);
    int NPrime = indices.size(0);
    int DPrime = weights.size(2);

    auto output = torch::zeros({NPrime, DPrime}, features.options());

    CUdeviceptr features_ptr;
    CUdeviceptr indices_ptr;
    CUdeviceptr weights_ptr;
    CUdeviceptr output_ptr;
    std::string dtype;

    if (features.dtype() == torch::kHalf) {
        TORCH_CHECK(features.dtype() == torch::kHalf, "features must be of type torch::kHalf");
        TORCH_CHECK(weights.dtype() == torch::kHalf, "weights must be of type torch::kHalf");
        features_ptr = reinterpret_cast<CUdeviceptr>(features.data_ptr<at::Half>());
        indices_ptr = reinterpret_cast<CUdeviceptr>(indices.data_ptr<int32_t>());
        weights_ptr = reinterpret_cast<CUdeviceptr>(weights.data_ptr<at::Half>());
        output_ptr = reinterpret_cast<CUdeviceptr>(output.data_ptr<at::Half>());
        dtype = "fp16";
    } else if (features.dtype() == torch::kFloat) {
        TORCH_CHECK(features.dtype() == torch::kFloat, "features must be of type torch::kFloat");
        TORCH_CHECK(weights.dtype() == torch::kFloat, "weights must be of type torch::kFloat");
        features_ptr = reinterpret_cast<CUdeviceptr>(features.data_ptr<float>());
        indices_ptr = reinterpret_cast<CUdeviceptr>(indices.data_ptr<int32_t>());
        weights_ptr = reinterpret_cast<CUdeviceptr>(weights.data_ptr<float>());
        output_ptr = reinterpret_cast<CUdeviceptr>(output.data_ptr<float>());
        dtype = "fp32";
    } else {
        throw std::runtime_error(std::string("Unsupported dtype: ") + std::string(features.dtype().TypeName<std::string>()));
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    kernels->run(
        features_ptr, // [N, D]
        indices_ptr, // [N', K**3]
        weights_ptr, // [K**3, D, D']
        output_ptr, // [N', D']
        N,
        NPrime,
        D,
        DPrime,
        K,
        acc_dtype,
        dtype,
        stream
    );

    return output;
}


TORCH_LIBRARY(conv3d_implicit_gemm, m) {
    m.def("setup_kernels", &setup_kernels);
    m.def("save_kernel_map", &save_kernel_map);
    m.def("conv3d_implicit_gemm_torch", &conv3d_implicit_gemm_torch);
}

