#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "json.hpp"

using json = nlohmann::json;

class Conv3DImplicitGemmKernel {
    std::string ptx;

    int shared;
    int global_scratch_size;
    int num_warps;

    int block_n;
    int block_k;
    int block_dp;
    bool div_k;
    bool div_d;
    bool div_dp;
    int parallel_k;
    std::string acc_dtype;
    std::string dtype;  // "fp16", "fp32"

    bool valid;
    CUmodule mod;
    CUfunction func;

   public:
    Conv3DImplicitGemmKernel(const json &config);

    ~Conv3DImplicitGemmKernel();

    void run(CUdeviceptr features,  // [N, D]
             CUdeviceptr indices,   // [N', K**3]
             CUdeviceptr weights,   // [K**3, D, D']
             CUdeviceptr output,    // [N', D']

             int N, int NPrime, int D, int DPrime, int K, CUstream stream);

    std::string get_dtype() const { return dtype; }

    std::string get_signature() const {
        std::ostringstream oss;
        oss << "{"
            << "BLOCK_N: " << block_n << ", "
            << "BLOCK_Dp: " << block_dp << ", "
            << "BLOCK_K: " << block_k << ", "
            << "PARALLEL_K: " << parallel_k << ", "
            << "acc_dtype: " << acc_dtype << ", "
            << "dtype: " << dtype << "}";
        return oss.str();
    }

    bool can_run(int N, int NPrime, int D, int DPrime, int K, std::string acc_dtype, std::string dtype) const {
        return valid && (dtype == this->dtype) && (acc_dtype == this->acc_dtype) && (!div_k || K % 16 == 0) &&
               (!div_d || D % 16 == 0) && (!div_dp || DPrime % 16 == 0);
    }
};

template <typename F>
double record(F &&func, CUstream stream) {
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

template <typename F>
double benchmark(F &&func, CUstream stream, const int n_warmup = 3,
                 const double target_time = 30  // target time for recording
) {
    // benchmark function inspired by triton
    // for micro-benchmarking kernels, we should take care to invalidate the L2 cache
    // Warmup
    double approx_iter_time = record(
                                  [&]() {
                                      for (int64_t i = 0; i < n_warmup; ++i) {
                                          func();
                                      }
                                  },
                                  stream) /
                              n_warmup;

    int num_iterations = std::max(static_cast<int>(target_time / (approx_iter_time + 0.1)), 2);

    std::vector<cudaEvent_t> start_events(num_iterations);
    std::vector<cudaEvent_t> stop_events(num_iterations);
    for (int i = 0; i < num_iterations; ++i) {
        cudaEventCreate(&start_events[i]);
        cudaEventCreate(&stop_events[i]);
    }
    int *buf;
    cudaMalloc(&buf, sizeof(int) * 1e6);
    for (int i = 0; i < num_iterations; ++i) {
        // this clears the l2 cache of the GPU
        cudaMemsetAsync(buf, 0, sizeof(int) * 1e6, stream);
        cudaEventRecord(start_events[i], stream);
        func();
        cudaEventRecord(stop_events[i], stream);
    }
    cudaEventSynchronize(stop_events[num_iterations - 1]);
    double total_time = 0.0;
    for (int i = 0; i < num_iterations; ++i) {
        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start_events[i], stop_events[i]);
        total_time += static_cast<double>(milliseconds);
        cudaEventDestroy(start_events[i]);
        cudaEventDestroy(stop_events[i]);
    }
    cudaFree(buf);
    return total_time / num_iterations;
}

class Conv3DKernels {
    //  N, N', D, D', K, sm, acc_dtype, dtype
    using kernel_hash_t = std::tuple<int, int, int, int, int, int, std::string, std::string>;

    std::vector<std::unique_ptr<Conv3DImplicitGemmKernel>> kernels;
    std::map<kernel_hash_t, int> kernel_map;
    std::map<kernel_hash_t, std::vector<double>> kernel_times;
    std::map<kernel_hash_t, std::vector<int>> kernel_indices;
    std::string metadata;
    int sm;

   public:
    void load_kernel_map();
    void load_kernel_map(const std::string &kernel_map_json);
    void save_kernel_map(std::string kernel_map_file);

    // this loads the kernels using the embedded metadata, through the cuda driver API
    // however, it doesn't initialize a cuda context, as we want to use the cuda context from pytorch.
    // therefore, it must be called after the pytorch context is initialized, which happens when you use the first CUDA
    // function
    Conv3DKernels();

    void run(CUdeviceptr features,  // [N, D]
             CUdeviceptr indices,   // [N', K**3]
             CUdeviceptr weights,   // [K**3, D, D']
             CUdeviceptr output,    // [N', D']
             int N, int NPrime, int D, int DPrime, int K, std::string acc_dtype, std::string dtype, CUstream stream);
};
