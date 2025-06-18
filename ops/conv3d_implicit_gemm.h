#pragma once
#include <cuda.h>
#include "json.hpp"
#include <string>
#include <cassert>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

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

    bool can_run(
        int N,
        int NPrime,
        int D,
        int DPrime,
        int K,
        std::string dtype
    ) const {
        return valid && (dtype == this->dtype) &&
               (!div_k || K % 16 == 0) &&
               (!div_d || D % 16 == 0) &&
               (!div_dp || DPrime % 16 == 0);
    }
};

std::vector<std::unique_ptr<Conv3DImplicitGemmKernel>> setup(
    std::string ptx_dir
);
