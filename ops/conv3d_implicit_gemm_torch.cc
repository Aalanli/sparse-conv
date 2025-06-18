#include "conv3d_implicit_gemm.h"
#include <cstdint>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <map>
#include <memory>
#include <torch/script.h>
#include <tuple>

thread_local static std::vector<std::unique_ptr<Conv3DImplicitGemmKernel>>
    kernels;

using kernel_hash_t = std::tuple<int, int, int, int, int, std::string>;
thread_local static std::unique_ptr<std::map<kernel_hash_t, int>> kernel_map;

template <typename F> double record(F &&func, CUstream stream) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);
    func();
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<double>(milliseconds);
}

template <typename F> double benchmark(F &&func, CUstream stream) {
    c10::cuda::CUDAGuard device_guard(at::cuda::current_device());
    const int n_warmup = 3;
    const double target_time = 10; // target time for recording
    // Warmup
    double approx_iter_time = record(
                                  [&]() {
                                      for (int64_t i = 0; i < n_warmup; ++i) {
                                          func();
                                      }
                                  },
                                  stream) /
                              n_warmup;

    int num_iterations =
        std::max(static_cast<int>(target_time / (approx_iter_time + 0.1)), 2);

    return record(
               [&]() {
                   for (int64_t i = 0; i < num_iterations; ++i) {
                       func();
                   }
               },
               stream) /
           num_iterations;
}

void setup_kernels(std::string ptx_dir) {
    kernels = setup(ptx_dir);
    kernel_map = std::make_unique<std::map<kernel_hash_t, int>>();
    TORCH_CHECK(!kernels.empty(),
                "No kernels were loaded from the provided PTX directory");
}

torch::Tensor conv3d_implicit_gemm_torch(torch::Tensor features, // [N, D]
                                         torch::Tensor indices,  // [N', K**3]
                                         torch::Tensor weights, // [K**3, D, D']
                                         int64_t K) {
    TORCH_CHECK(features.is_cuda(), "features must be a CUDA tensor");
    TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
    TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");

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
        TORCH_CHECK(features.dtype() == torch::kHalf,
                    "features must be of type torch::kHalf");
        TORCH_CHECK(weights.dtype() == torch::kHalf,
                    "weights must be of type torch::kHalf");
        features_ptr =
            reinterpret_cast<CUdeviceptr>(features.data_ptr<at::Half>());
        indices_ptr =
            reinterpret_cast<CUdeviceptr>(indices.data_ptr<int32_t>());
        weights_ptr =
            reinterpret_cast<CUdeviceptr>(weights.data_ptr<at::Half>());
        output_ptr = reinterpret_cast<CUdeviceptr>(output.data_ptr<at::Half>());
        dtype = "fp16";
    } else if (features.dtype() == torch::kFloat) {
        TORCH_CHECK(features.dtype() == torch::kFloat,
                    "features must be of type torch::kFloat");
        TORCH_CHECK(weights.dtype() == torch::kFloat,
                    "weights must be of type torch::kFloat");
        features_ptr =
            reinterpret_cast<CUdeviceptr>(features.data_ptr<float>());
        indices_ptr =
            reinterpret_cast<CUdeviceptr>(indices.data_ptr<int32_t>());
        weights_ptr = reinterpret_cast<CUdeviceptr>(weights.data_ptr<float>());
        output_ptr = reinterpret_cast<CUdeviceptr>(output.data_ptr<float>());
        dtype = "fp32";
    } else {
        throw std::runtime_error(
            std::string("Unsupported dtype: ") +
            std::string(features.dtype().TypeName<std::string>()));
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int seq_len_quant = 2500;
    kernel_hash_t kernel_hash = std::make_tuple(
        N / seq_len_quant, NPrime / seq_len_quant, D, DPrime, K, dtype);
    if (kernel_map->find(kernel_hash) == kernel_map->end()) {
        // Find a suitable kernel
        double best_time = 1e9;
        int best_kernel_index = -1;
        for (size_t i = 0; i < kernels.size(); ++i) {
            if (kernels[i]->can_run(N, NPrime, D, DPrime, K, dtype)) {
                double time = benchmark(
                    [&]() {
                        kernels[i]->run(features_ptr, indices_ptr, weights_ptr,
                                        output_ptr, N, NPrime, D, DPrime, K,
                                        stream);
                    },
                    stream);

                if (time < best_time) {
                    best_time = time;
                    best_kernel_index = i;
                }
            }
        }

        TORCH_CHECK(best_kernel_index != -1,
                    "No suitable kernel found for the given parameters");
        (*kernel_map)[kernel_hash] = best_kernel_index;
    }

    int kernel_index = (*kernel_map)[kernel_hash];
    kernels[kernel_index]->run(features_ptr, indices_ptr, weights_ptr,
                               output_ptr, N, NPrime, D, DPrime, K, stream);

    return output;
}

TORCH_LIBRARY(conv3d_implicit_gemm, m) {
    m.def("setup_kernels", &setup_kernels);
    m.def("conv3d_implicit_gemm_torch", &conv3d_implicit_gemm_torch);
}
