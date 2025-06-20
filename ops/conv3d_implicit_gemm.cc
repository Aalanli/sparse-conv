
#include <cuda.h>
#include <cuda_fp16.h>
#include "json.hpp"
#include <sstream>
#include <string>
#include <cassert>

#include <iostream>
#include <fstream>
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

    block_n = config["config"]["BLOCK_N"];
    block_k = config["config"]["BLOCK_K"];
    block_dp = config["config"]["BLOCK_Dp"];
    div_k = config["config"]["div_k"];
    div_d = config["config"]["div_d"];
    div_dp = config["config"]["div_dp"];
    dtype = config["config"]["dtype"];
    parallel_k = config["config"]["parallel_k"];
    acc_dtype = config["config"]["acc_dtype"];
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

    valid = (device_sm == sm);
    if (!valid) {
        // std::cerr << "Device SM " << device_sm << " is not equal to the required SM " << sm
        //           << ". Kernel will not be loaded from " << ptx_path << std::endl;
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
    if (!valid) {
        return; // No need to unload if the kernel is not valid
    }
    CHECK(cuModuleUnload(mod));
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
    int blocks = ((NPrime + block_n - 1) / block_n) * ((DPrime + block_dp - 1) / block_dp) * parallel_k;

    if (parallel_k > 1) {
        int elems = NPrime * DPrime;
        if (dtype == "fp16") {
            cuMemsetD16Async(output, 0, elems, stream);
        } else if (dtype == "fp32") {
            cuMemsetD32Async(output, 0, elems, stream);
        } else {
            throw std::runtime_error("Unsupported dtype: " + dtype);
        }
    }

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

void Conv3DKernels::load_kernel_map(std::string kernel_map_file) {
    std::ifstream file(kernel_map_file);
    if (!file.is_open()) {
        std::cerr << "Cached kernel map file not found: " << kernel_map_file << std::endl;
        return;
    }
    json kmap = json::parse(file);
    for (auto& value : kmap) {
        int N = value["N"];
        int NPrime = value["NPrime"];
        int D = value["D"];
        int DPrime = value["DPrime"];
        int K = value["K"];
        int sm = value["sm"];
        std::string acc_dtype = value["acc_dtype"];
        std::string dtype = value["dtype"];
        kernel_hash_t key = std::make_tuple(N, NPrime, D, DPrime, K, sm, acc_dtype, dtype);
        int index = value["index"];
        if (kernel_map.find(key) != kernel_map.end()) {
            std::cerr << "Warning: Duplicate kernel map entry "
                      << value.dump() << std::endl;
        } else {
            kernel_map[key] = index;
            std::vector<int> kidx;
            for (const auto& k : value["kidx"]) {
                kidx.push_back(k.get<int>());
            }
            std::vector<double> times;
            for (const auto& time : value["time"]) {
                times.push_back(time.get<double>());
            }
            kernel_times[key] = times;
            kernel_indices[key] = kidx;
        }
    }
}

void Conv3DKernels::save_kernel_map(std::string kernel_map_file) {
    if (std::filesystem::exists(kernel_map_file)) {
        load_kernel_map(kernel_map_file);
    }
    json kmap;
    for (const auto& [key, index] : kernel_map) {
        auto [N, NPrime, D, DPrime, K, sm, acc_dtype, dtype] = key;
        json idx_time;
        for (const auto& time : kernel_times[key]) {
            idx_time.push_back(time);
        }
        json kidx;
        for (const auto& k : kernel_indices[key]) {
            kidx.push_back(k);
        }
        kmap.push_back({
            {"N", N},
            {"NPrime", NPrime},
            {"D", D},
            {"DPrime", DPrime},
            {"K", K},
            {"sm", sm},
            {"acc_dtype", acc_dtype},
            {"dtype", dtype},
            {"index", index},
            {"time", idx_time},
            {"kidx", kidx},
            {"signature", kernels[index]->get_signature()}
        });
    }
    std::ofstream file(kernel_map_file);
    file << kmap.dump(4);
}

Conv3DKernels::Conv3DKernels(
    std::string ptx_dir
) {

    int device = 0;
    CHECK(cuDeviceGet(&device, 0));
    int device_major = 0, device_minor = 0;
    CHECK(cuDeviceGetAttribute(&device_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    CHECK(cuDeviceGetAttribute(&device_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    sm = device_major * 10 + device_minor;

    ptx_dir = std::filesystem::absolute(ptx_dir).string();
    std::string config_path = ptx_dir + "/meta.json";
    std::cout << "Loading kernels from: " << config_path << std::endl;
    std::ifstream config_file (config_path);
    metadata = std::string(
        (std::istreambuf_iterator<char>(config_file)),
        std::istreambuf_iterator<char>()
    );
    json config = json::parse(
        metadata
    );
    
    for (auto& [key, value] : config.items()) {
        auto ptx_path = ptx_dir + "/" + key;
        auto ker = new Conv3DImplicitGemmKernel(
            value, ptx_path
        );
        kernels.push_back(std::unique_ptr<Conv3DImplicitGemmKernel>(ker));
    }

    load_kernel_map(ptx_dir + "/kernel_map.json");
}

int quant_N(int N) {
    std::vector<int> thresholds = {1000, 10000, 100000, 600000};
    for (size_t i = 0; i < thresholds.size(); ++i) {
        if (N <= thresholds[i]) {
            return thresholds[i];
        }
    }
    return thresholds.back();
}

void Conv3DKernels::run(
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
) {
    // quantize N 
    kernel_hash_t key = std::make_tuple(quant_N(N), quant_N(NPrime), D, DPrime, K, sm, acc_dtype, dtype);
    
    if (kernel_map.find(key) == kernel_map.end()) {
        // Find a suitable kernel
        double best_time = 1e9;
        int best_kernel_index = -1;
        std::vector<int> kidx;
        std::vector<double> times;
        for (size_t i = 0; i < kernels.size(); ++i) {
            if (kernels[i]->can_run(N, NPrime, D, DPrime, K, acc_dtype, dtype)) {
                double time = benchmark([&]() {
                    kernels[i]->run(
                        features, indices, weights, output,
                        N, NPrime, D, DPrime, K, stream
                    );
                }, stream);
                kidx.push_back(i);
                times.push_back(time);

                if (time < best_time) {
                    best_time = time;
                    best_kernel_index = i;
                }
            }
        }
        if (best_kernel_index == -1) {
            std::ostringstream oss;
            oss << "No suitable kernel found for parameters: "

                << "N: " << N << ", "
                << "NPrime: " << NPrime << ", "
                << "D: " << D << ", "
                << "DPrime: " << DPrime << ", "
                << "K: " << K << ", "
                << "acc_dtype: " << acc_dtype << ", "
                << "dtype: " << dtype;
            throw std::runtime_error(oss.str());
        }
        kernel_map[key] = best_kernel_index;
        kernel_times[key] = times;
        kernel_indices[key] = kidx;
    }
    
    int index = kernel_map[key];
    kernels[index]->run(features, indices, weights, output, N, NPrime, D, DPrime, K, stream);
}

