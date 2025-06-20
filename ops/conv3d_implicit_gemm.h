#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "json.hpp"
#include <string>
#include <cassert>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <map>


using json = nlohmann::json;

class Conv3DImplicitGemmKernel {
    std::string ptx_path;
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
    std::string dtype; // "fp16", "fp32"
    
    bool valid;
    CUmodule mod;
    CUfunction func;
public: 
    Conv3DImplicitGemmKernel(
        json config, std::string ptx_path
    );

    ~Conv3DImplicitGemmKernel();

    void run(
        CUdeviceptr features, // [N, D]
        CUdeviceptr indices, // [N', K**3]
        CUdeviceptr weights, // [K**3, D, D']
        CUdeviceptr output, // [N', D']
        
        int N,
        int NPrime,
        int D,
        int DPrime,
        int K,
        CUstream stream
    );

    std::string get_dtype() const {
        return dtype;
    }

    std::string get_signature() const{
        std::ostringstream oss;
        oss << "{"
            << "N: " << block_n << ", "
            << "D: " << block_dp << ", "
            << "K: " << block_k << ", "
            << "acc_dtype: " << acc_dtype << ", "
            << "dtype: " << dtype
            << "}";
        return oss.str();
    }

    bool can_run(
        int N,
        int NPrime,
        int D,
        int DPrime,
        int K,
        std::string acc_dtype,
        std::string dtype
    ) const {
        return valid && (dtype == this->dtype) && (acc_dtype == this->acc_dtype) &&
               (!div_k || K % 16 == 0) &&
               (!div_d || D % 16 == 0) &&
               (!div_dp || DPrime % 16 == 0);
    }
};


template <typename F>
double record(
    F&& func,
    CUstream stream
) {
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
double benchmark(
    F&& func,
    CUstream stream,
    const int n_warmup = 3,
    const double target_time = 30 // target time for recording
) {
    // Warmup
    double approx_iter_time = record(
        [&]() {
            for (int64_t i = 0; i < n_warmup; ++i) {
                func();
            }
        },
        stream
    ) / n_warmup;

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
    //  N, D, N', D', K, sm, acc_dtype, dtype
    using kernel_hash_t = std::tuple<int, int, int, int, int, int, std::string, std::string>;

    std::vector<std::unique_ptr<Conv3DImplicitGemmKernel>> kernels;
    std::map<kernel_hash_t, int> kernel_map;
    std::string ptx_dir;
    std::string metadata;
    int sm;
public:
    void load_kernel_map(std::string kernel_map_file);
    void save_kernel_map(std::string kernel_map_file);

    Conv3DKernels(std::string ptx_dir);

    void run(
        CUdeviceptr features, // [N, D]
        CUdeviceptr indices, // [N', K**3]
        CUdeviceptr weights, // [K**3, D, D']
        CUdeviceptr output, // [N', D']
        int N,
        int NPrime,
        int D,
        int DPrime,
        int K,
        std::string acc_dtype,
        std::string dtype,
        CUstream stream
    );
};
