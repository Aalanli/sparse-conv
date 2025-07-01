#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <map>
#include <iostream>

#include "json.hpp"

#define CHECK_CUDA_CALL(call)                                                                                     \
    do {                                                                                                \
        CUresult _e = call;                                                                             \
        if (_e != CUDA_SUCCESS) {                                                                       \
            const char *err, *str;                                                                      \
            cuGetErrorName(_e, &err);                                                                   \
            cuGetErrorString(_e, &str);                                                                 \
            std::cerr << "CUDA Error: " << err << " - " << str << " at line " << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE);                                                                    \
        }                                                                                               \
    } while (0)


using json = nlohmann::json;

int get_cur_sm_version();

int cdiv(int a, int b);

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


class Dtype {
public:
    enum Type {
        FP16,
        FP32,
        INT32,
        INT64,
        BOOL
    };
    const Type type;
    const bool is_ptr;
    Dtype(Type type, bool is_ptr = false) : type(type), is_ptr(is_ptr) {}

    static Dtype from_string(const std::string &str) {
        if (str == "fp16") {
            return Dtype(FP16);
        } else if (str == "fp32") {
            return Dtype(FP32);
        } else if (str == "i32") {
            return Dtype(INT32);
        } else if (str == "i64") {
            return Dtype(INT64);
        } else if (str == "i1") {
            return Dtype(BOOL);
        } else if (str[0] == '*') {
            return Dtype(from_string(str.substr(1)).type, true);
        } else {
            throw std::runtime_error("Unsupported dtype: " + str);
        }
    }

    std::string to_string() const {
        std::string type_str;
        switch (type) {
            case FP16:
                type_str = "fp16";
                break;
            case FP32:
                type_str = "fp32";
                break;
            case INT32:
                type_str = "i32";
                break;
            case INT64:
                type_str = "i64";
                break;
            case BOOL:
                type_str = "i1";
                break;
        }
        return is_ptr ? "*" + type_str : type_str;
    }

    bool operator==(const Dtype &other) const {
        return type == other.type && is_ptr == other.is_ptr;
    }

    bool operator!=(const Dtype &other) const {
        return !(*this == other);
    }

    void operator=(const Dtype &other) {
        if (this != &other) {
            throw std::runtime_error("Dtype assignment is not allowed");
        }
    }

    bool operator<(const Dtype &other) const {
        return std::tie(type, is_ptr) < std::tie(other.type, other.is_ptr); 
    }
};

bool is_div_16(const std::pair<Dtype, void*> &arg);


/// KernelArgs is a more flexible representation of kernel parameters. And must satisfy the following interface:
/// typename KHash_t; a cheap type that is hashable and identifies the performance sensitive parameters of the kernel, such as dimensions
/// KHash_t khash() const; get the hash of the kernel parameters
/// std::vector<std::pair<Dtype, void*>> get_args() const; 
/// static KHash_t deserialize(const json &);  // Deserialize the hash from JSON
/// static json serialize(KHash_t) const;  // Serialize the kernel parameters to JSON

/// The "raw" arguments to the kernel, which only contains data, such as dimension bounds
/// and pointers to data.
using KArg_t = std::vector<std::pair<Dtype, void*>>;  // (dtype, pointer to data)



/// Think of this class like the equivalent of a cuda kernel
/// except triton kernels do not have "threads"
template <typename KernelArgs_t>
class TritonKernel {
    std::string ptx;

    int shared;
    int global_scratch_size;
    int num_warps;

    std::string kernel_name;
    std::vector<std::pair<std::string, Dtype>> args;  // (name, dtype)
    std::vector<int> divisible_by_16;

    CUmodule mod;
    CUfunction func;

public:
    using KernelArgs = KernelArgs_t;
    using self_t = TritonKernel<KernelArgs>;

    TritonKernel(const std::string &ptx, int shared, int global_scratch_size, int num_warps,
            const std::string &kernel_name,
            const std::vector<std::pair<std::string, Dtype>> &args,
            const std::vector<int> &divisible_by_16) 
    : ptx(ptx), shared(shared), global_scratch_size(global_scratch_size), num_warps(num_warps),
        kernel_name(kernel_name),
        args(args), divisible_by_16(divisible_by_16)
    {
        CHECK_CUDA_CALL(cuModuleLoadDataEx(&mod, ptx.data(), 0, nullptr, nullptr));
        CHECK_CUDA_CALL(cuModuleGetFunction(&func, mod, kernel_name.c_str()));
    }
    
    TritonKernel(const json &config) {
        // std::vector<std::pair<std::string, Dtype>> args;
        // std::vector<int> divisible_by_16;
        for (const auto &arg : config["args"]) {
            std::string name = arg["name"];
            Dtype dtype = Dtype::from_string(arg["dtype"]);
            args.push_back({name, dtype});
        }
        for (const auto &div : config["divisible_by_16"]) {
            divisible_by_16.push_back(div.get<int>());
        }
        ptx = config["ptx"];
        shared = config["shared"];
        global_scratch_size = config["global_scratch_size"];
        num_warps = config["num_warps"];
        kernel_name = config["kernel_name"];
        CHECK_CUDA_CALL(cuModuleLoadDataEx(&mod, ptx.data(), 0, nullptr, nullptr));
        CHECK_CUDA_CALL(cuModuleGetFunction(&func, mod, kernel_name.c_str()));
    }


    ~TritonKernel() {
        CHECK_CUDA_CALL(cuModuleUnload(mod));
    }

    void run(KernelArgs &kargs, CUstream stream) {
        auto args = kargs.get_args();
        if (args.size() != this->args.size()) {
            throw std::runtime_error("Argument size mismatch: expected " + std::to_string(this->args.size()) +
                                    ", got " + std::to_string(args.size()));
        }
        std::vector<void*> arg_ptrs;
        arg_ptrs.reserve(args.size() + 1);
        int i = 0;
        for (auto i : divisible_by_16) {
            if (!(i < (int) args.size() && is_div_16(args[i]))) {
                throw std::runtime_error("Argument at index " + std::to_string(i) +
                                        " must be divisible by 16");
            }
        }
        for (auto [dtype, data] : args) {
            if (dtype != this->args[i].second) {
                throw std::runtime_error("Argument type mismatch at index " + std::to_string(i) +
                                        ": expected " + this->args[i].second.to_string() +
                                        ", got " + dtype.to_string());
            }
            arg_ptrs.push_back(data);
            i += 1;
        }
        assert(global_scratch_size == 0 && "global_scratch_size must be 0 for now");
        arg_ptrs.push_back(nullptr);  // Placeholder for global scratch, not used currently

        auto [gx, gy, gz] = blocks(kargs);
        CHECK_CUDA_CALL(cuLaunchKernel(
            func, 
            gx, gy, gz,
            num_warps * 32, 1, 1,
            shared, stream, arg_ptrs.data(), nullptr));
    }


    bool can_run(const KernelArgs &kargs) const {
        auto args = kargs.get_args();
        if (args.size() != this->args.size()) {
            return false; 
        }
        for (auto i : divisible_by_16) {
            if (!(i < (int) args.size() && is_div_16(args[i]))) {
                return false;  // Argument at index i must be divisible by 16
            }
        }
        std::vector<void*> arg_ptrs;
        arg_ptrs.reserve(args.size());
        int i = 0;
        for (auto [dtype, data] : args) {
            if (dtype != this->args[i].second) {
                return false;
            }
            i += 1;
        }
        return true;
    }

    std::string signature() const {
        std::ostringstream oss;
        oss << "{";
        for (const auto &[name, dtype] : args) {
            oss << name << ": " << dtype.to_string() << ", ";
        }
        oss << "shared: " << shared << ", "
            << "global_scratch_size: " << global_scratch_size << ", "
            << "num_warps: " << num_warps << ", "
            << "kernel_name: " << kernel_name
            << "}";
        return oss.str();
    }

    virtual std::tuple<int, int, int> blocks(const KernelArgs &args) const = 0;
};

template <typename KernelT>
class TritonAotKernels {
public:
    using KernelArgs = typename KernelT::KernelArgs;
    using KHash_t = typename KernelArgs::KHash_t;
private:
    std::vector<std::unique_ptr<KernelT>> kernels;
    

    using KHash_t_ = std::tuple<KHash_t, int>;  // (khash, sm_version)
    std::map<KHash_t_, int> kernel_map;
    std::map<KHash_t_, std::vector<double>> kernel_times;
    std::map<KHash_t_, std::vector<int>> kernel_indices;
    int sm;
public:
    json dump_kernel_map() const {
        json res;
        for (const auto &[khash, index] : kernel_map) {
            json kmap;
            kmap["khash"] = KernelArgs::serialize(std::get<0>(khash));
            kmap["index"] = index;
            kmap["signature"] = kernels[index]->signature();
            kmap["times"] = kernel_times.at(khash);
            kmap["kidx"] = kernel_indices.at(khash);
            kmap["sm"] = std::get<1>(khash);
            res.push_back(kmap);
        }
        return res;
    }

    void load_kernel_map(const json& kmap) {
        for (const auto &value : kmap) {
            KHash_t khash = KernelArgs::deserialize(value["khash"]);
            int index = value["index"];
            if (kernel_map.find(khash) != kernel_map.end()) {
                std::cerr << "Warning: Duplicate kernel map entry " << value.dump() << std::endl;
            } else {
                auto khash_with_sm = std::make_tuple(khash, value["sm"].get<int>());
                kernel_map[khash_with_sm] = index;
                kernel_times[khash_with_sm] = value["times"].get<std::vector<double>>();
                kernel_indices[khash_with_sm] = value["kidx"].get<std::vector<int>>();
            }
        }
    }
    
    TritonAotKernels(const json &kernel_meta) {
        sm = get_cur_sm_version();
        unsigned int num_skipped = 0;
        for (const auto &value : kernel_meta) {
            int ptx_sm = value["sm"];
            if (ptx_sm > sm) {
                num_skipped += 1;
                // this preserves the indices in the kernel map
                kernels.push_back(nullptr);
            }
            auto ker = std::make_unique<KernelT>(value);
            kernels.push_back(std::move(ker));
        }
        if (num_skipped > 0) {
            std::cerr << "Warning: " << num_skipped << " kernels were skipped due to SM version mismatch." << std::endl;
        } 
        if (kernels.size() == num_skipped) {
            throw std::runtime_error("No kernels available for the current SM version: " + std::to_string(sm));
        }
    }

    void run(KernelArgs args, CUstream stream) {
        auto khash = args.khash();
        auto khash_with_sm = std::make_tuple(khash, sm);
        if (kernel_map.find(khash_with_sm) == kernel_map.end()) {
            // Find a suitable kernel
            double best_time = 1e9;
            int best_kernel_index = -1;
            std::vector<int> kidx;
            std::vector<double> times;
            for (size_t i = 0; i < kernels.size(); ++i) {
                if (kernels[i] != nullptr && kernels[i]->can_run(args)) {
                    double time = benchmark(
                        [&]() { kernels[i]->run(args, stream); },
                        stream);
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
                    << KernelArgs::serialize(khash)
                    << ", SM version: " << sm << std::endl;
                throw std::runtime_error(oss.str());
            }
            kernel_map[khash_with_sm] = best_kernel_index;
            kernel_times[khash_with_sm] = times;
            kernel_indices[khash_with_sm] = kidx;
        }

        int kernel_index = kernel_map[khash_with_sm];
        kernels[kernel_index]->run(args, stream);
    }
};

