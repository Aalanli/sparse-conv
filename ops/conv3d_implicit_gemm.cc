
#include <cuda.h>
#include "json.hpp"
#include <string>
#include <cassert>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <filesystem>

#include "conv3d_implicit_gemm.h"

using json = nlohmann::json;
#define CHECK(call)                                               \
    do {                                                          \
        CUresult _e = call;                                       \
        if (_e != CUDA_SUCCESS) {                                 \
            const char *err, *str;                                \
            cuGetErrorName(_e, &err);                             \
            cuGetErrorString(_e, &str);                           \
            std::cerr << "CUDA Error: " << err << " – " << str    \
                      << " at line " << __LINE__ << std::endl;    \
            std::exit(EXIT_FAILURE);                              \
        }                                                         \
    } while (0)

Conv3DImplicitGemmKernel::Conv3DImplicitGemmKernel(
    json config, std::string ptx_path
) {
    shared = config["meta"]["shared"];
    global_scratch_size = config["meta"]["global_scratch_size"];
    num_warps = config["meta"]["num_warps"];

    block_n = config["config"]["block_sizes"]["BLOCK_N"];
    block_k = config["config"]["block_sizes"]["BLOCK_K"];
    block_dp = config["config"]["block_sizes"]["BLOCK_Dp"];
    div_k = config["config"]["div_k"];
    div_d = config["config"]["div_d"];
    div_dp = config["config"]["div_dp"];
    dtype = config["config"]["dtype"];
    int sm = config["config"]["sm"];
    
    // Get current device
    int device = 0;
    CHECK(cuCtxGetDevice(&device));

    // Get device properties
    CUdevice cuDevice;
    CHECK(cuDeviceGet(&cuDevice, device));
    int device_major = 0, device_minor = 0;
    CHECK(cuDeviceGetAttribute(&device_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
    CHECK(cuDeviceGetAttribute(&device_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
    int device_sm = device_major * 10 + device_minor;

    // Check if device SM version is >= compiled SM version
    valid = (device_sm >= sm);
    if (!valid) {
        std::cerr << "Device SM " << device_sm << " is less than required SM " << sm
                  << ". Kernel will not be loaded from " << ptx_path << std::endl;
        return;
    }

    this->ptx_path = ptx_path;
    std::ifstream ptx_file(ptx_path);
    this->ptx = std::string(
        (std::istreambuf_iterator<char>(ptx_file)),
        std::istreambuf_iterator<char>()
    );

    assert(global_scratch_size == 0 && "global_scratch_size must be 0 for now");

    CHECK(cuModuleLoadDataEx(
        &mod, ptx.c_str(), 0, nullptr, nullptr
    ));
    CHECK(cuModuleGetFunction(
        &func, mod, "implicit_conv3d_kernel"
    ));
}

Conv3DImplicitGemmKernel::~Conv3DImplicitGemmKernel() {
    cuModuleUnload(mod);
}

void Conv3DImplicitGemmKernel::run(
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
) {
    assert((!div_k || K % 16 == 0) && "K must be divisible by 16 if div_k is true");
    assert((!div_d || D % 16 == 0) && "D must be divisible by 16 if div_d is true");
    assert((!div_dp || DPrime % 16 == 0) && "DPrime must be divisible by 16 if div_dp is true");

    int threads = num_warps * 32;
    int blocks = ((NPrime + block_n - 1) / block_n) * ((DPrime + block_dp - 1) / block_dp);

    // we dont have global scratch for now
    CUdeviceptr global_scratch;

    void *args[] = {
        &features,
        &indices,
        &weights,
        &output,

        const_cast<int*>(&N),
        const_cast<int*>(&NPrime),
        const_cast<int*>(&D),
        const_cast<int*>(&DPrime),
        const_cast<int*>(&K),
        &global_scratch,
    };

    CHECK(cuLaunchKernel(
        func,
        blocks, 1, 1, // grid
        threads, 1, 1, // block
        shared, stream, // shared memory and stream
        args, nullptr // kernel arguments and extra
    ));
}

std::vector<std::unique_ptr<Conv3DImplicitGemmKernel>> setup(
    std::string ptx_dir
) {
    ptx_dir = std::filesystem::absolute(ptx_dir).string();
    std::string config_path = ptx_dir + "/meta.json";
    std::cout << "Loading kernels from: " << config_path << std::endl;
    std::ifstream config_file (config_path);
    std::string config_file_content(
        (std::istreambuf_iterator<char>(config_file)),
        std::istreambuf_iterator<char>()
    );
    json config = json::parse(
        config_file_content
    );
    
    std::vector<std::unique_ptr<Conv3DImplicitGemmKernel>> kernels;
    for (auto& [key, value] : config.items()) {
        auto ptx_path = ptx_dir + "/" + key;
        auto ker = new Conv3DImplicitGemmKernel(
            value, ptx_path
        );
        kernels.push_back(std::unique_ptr<Conv3DImplicitGemmKernel>(ker));
    }

    return kernels;
}

