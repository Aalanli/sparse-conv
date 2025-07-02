#include "triton_aot_utils.h"

#include <mutex>

// rust-like Mutex class for managing unique_ptr instances, cuz it's convenient
template <typename T>
class MutexGuard {
private:
    std::lock_guard<std::mutex> lock;
    T *instance;
public:
    MutexGuard(std::unique_ptr<T> &instance, std::mutex &mtx)
        : lock(mtx), instance(instance.get()) {}
    T* get() {
        return instance;
    }
};

template <typename T>
class Mutex {
private:
    std::mutex mtx;
    std::unique_ptr<T> instance;
public:
    Mutex() : instance(nullptr) {}
    void operator=(std::unique_ptr<T> &&new_instance) {
        std::lock_guard<std::mutex> lock(mtx);
        instance = std::move(new_instance);
    }
    MutexGuard<T> lock() {
        return MutexGuard<T>(instance, mtx);
    }
};

int quant_N(int N);

struct IdxSortKernelArgs {
    using KHash_t = std::tuple<int, int, std::string>;
    void *indices; // [K3, N]
    void *line_mask; // [N]
    Dtype mask_dtype;
    int N;
    int N_stride;
    int K3;

    static json serialize(const KHash_t &khash) {
        json res;
        res["N"] = std::get<0>(khash);
        res["K3"] = std::get<1>(khash);
        res["mask_dtype"] = std::get<2>(khash);
        return res;
    }

    static KHash_t deserialize(const json &j) {
        return std::make_tuple(j["N"].get<int>(), j["K3"].get<int>(), j["mask_dtype"].get<std::string>());
    }

    KHash_t khash() const {
        return std::make_tuple(quant_N(N), K3, mask_dtype.to_string());
    }

    KArg_t get_args() const {
        return {
            {Dtype(Dtype::INT32, true), indices},
            {Dtype(mask_dtype.type, true), line_mask},
            {Dtype(Dtype::INT32), (void*) &N},
            {Dtype(Dtype::INT32), (void*) &N_stride},
            {Dtype(Dtype::INT32), (void*) &K3}
        };
    }
};

class ImplicitGemmSortKernel : public TritonKernel<IdxSortKernelArgs> {
    int BLOCK_K;
    int BLOCK_N;
public:
    ImplicitGemmSortKernel(const json &config) : 
        TritonKernel<IdxSortKernelArgs>(config), 
        BLOCK_K(config["BLOCK_K"]),
        BLOCK_N(config["BLOCK_N"]) {}
    
    std::tuple<int, int, int> blocks(const IdxSortKernelArgs &args) const override {
        return {
            cdiv(args.N, BLOCK_N) * cdiv(args.K3, BLOCK_K),
            1,
            1
        };
    }

    bool can_run(IdxSortKernelArgs &args) const {
        return args.K3 <= BLOCK_K && super_t::can_run(args);
    }
};

struct GemmMaskKernelArgs {
    using KHash_t = std::tuple<int, int, int>;
    void* indices; // [K3, N]
    void* mask; // [N', K3]
    int N;
    int N_stride;
    int K3;
    int BLOCK_N;

    static json serialize(const KHash_t &khash) {
        json res;
        res["N"] = std::get<0>(khash);
        res["K3"] = std::get<1>(khash);
        res["BLOCK_N"] = std::get<2>(khash);
        return res;
    }

    static KHash_t deserialize(const json &j) {
        return std::make_tuple(j["N"].get<int>(), j["K3"].get<int>(), j["BLOCK_N"].get<int>());
    }

    KHash_t khash() const {
        return std::make_tuple(quant_N(N), K3, BLOCK_N);
    }

    KArg_t get_args() {
        return {
            {Dtype(Dtype::INT32, true),indices},
            {Dtype(Dtype::BOOL, true), mask},
            {Dtype(Dtype::INT32), (void*) &N},
            {Dtype(Dtype::INT32), (void*) &N_stride},
            {Dtype(Dtype::INT32), (void*) &K3}
        };
    }
};

class ImplicitGemmMaskKernel : public TritonKernel<GemmMaskKernelArgs> {
    int BLOCK_N;
    int BLOCK_K;
public:
    ImplicitGemmMaskKernel(const json &config) : 
        TritonKernel<GemmMaskKernelArgs>(config), 
        BLOCK_N(config["BLOCK_N"]),
        BLOCK_K(config["BLOCK_K"]) {}
    
    std::tuple<int, int, int> blocks(const GemmMaskKernelArgs &args) const override {
        return {
            cdiv(args.N, BLOCK_N) * cdiv(args.K3, BLOCK_K),
            1,
            1
        };
    }

    bool can_run(GemmMaskKernelArgs &args) const {
        return args.BLOCK_N == BLOCK_N && super_t::can_run(args);
    }
};

struct ImplicitGemmConv3dKernelTArgs {
    Dtype feat_dtype;
    Dtype weight_dtype;
    Dtype acc_dtype;

    void *features; // [N, D]
    void *indices;  // [N', K**3]
    void *mask_ind; // [NP, K**3]
    void *weights;  // [K**3, D, D']
    void *out_perm; // [N']
    void *out;      // [N', D']
    int N;
    int NPrime;
    int N_prime_stride;
    int D;
    int DPrime;
    int K;
    int BLOCK_N;
    bool sorted;
    // feat_dtype, weight_dtype, acc_dtype, N, NPrime, D, DPrime, K, BLOCK_N, PARALLEL_K, sorted
    using KHash_t = std::tuple<Dtype, Dtype, Dtype, int, int, int, int, int, int, bool>;

    static json serialize(const KHash_t& args) {
        json res;
        res["feat_dtype"] = std::get<0>(args).to_string();
        res["weight_dtype"] = std::get<1>(args).to_string();
        res["acc_dtype"] = std::get<2>(args).to_string();
        res["N"] = std::get<3>(args);
        res["NPrime"] = std::get<4>(args);
        res["D"] = std::get<5>(args);
        res["DPrime"] = std::get<6>(args);
        res["K"] = std::get<7>(args);
        res["BLOCK_N"] = std::get<8>(args);
        res["sorted"] = std::get<9>(args);
        return res;
    }

    static KHash_t deserialize(const json &j) {
        return std::make_tuple(
            Dtype::from_string(j["feat_dtype"].get<std::string>()),
            Dtype::from_string(j["weight_dtype"].get<std::string>()),
            Dtype::from_string(j["acc_dtype"].get<std::string>()),
            j["N"].get<int>(),
            j["NPrime"].get<int>(),
            j["D"].get<int>(),
            j["DPrime"].get<int>(),
            j["K"].get<int>(),
            j["BLOCK_N"].get<int>(),
            j["sorted"].get<bool>()
        );
    }

    KHash_t khash() const {
        return std::make_tuple(
            feat_dtype, weight_dtype, acc_dtype,
            quant_N(N), quant_N(NPrime), D, DPrime, K, BLOCK_N, sorted
        );
    }

    KArg_t get_args() const {
        return {
            {Dtype(feat_dtype.type, true), features},
            {Dtype(Dtype::INT32, true), indices},
            {Dtype(Dtype::BOOL, true), mask_ind},
            {Dtype(weight_dtype.type, true), weights},
            {Dtype(Dtype::INT32, true), out_perm},
            {Dtype(feat_dtype.type, true), out},
            {Dtype(Dtype::INT32), (void*) &N},
            {Dtype(Dtype::INT32), (void*) &NPrime},
            {Dtype(Dtype::INT32), (void*) &N_prime_stride},
            {Dtype(Dtype::INT32), (void*) &D},
            {Dtype(Dtype::INT32), (void*) &DPrime},
            {Dtype(Dtype::INT32), (void*) &K},
            {Dtype(Dtype::BOOL), (void*) &sorted},
        };
    }

};


class ImplicitGemmConv3dKernelT : public TritonKernel<ImplicitGemmConv3dKernelTArgs> {
    int BLOCK_N;
    int BLOCK_K;
    int BLOCK_Dp;
    int PARALLEL_K;
    Dtype acc_dtype;
public:
    
    ImplicitGemmConv3dKernelT(const json &config) : 
        TritonKernel<ImplicitGemmConv3dKernelTArgs>(config),
        BLOCK_N(config["BLOCK_N"]),
        BLOCK_K(config["BLOCK_K"]),
        BLOCK_Dp(config["BLOCK_Dp"]),
        PARALLEL_K(config["PARALLEL_K"]),
        acc_dtype(Dtype::from_string(config["acc_dtype"].get<std::string>())) {}
    
    std::tuple<int, int, int> blocks(const ImplicitGemmConv3dKernelTArgs &args) const override {
        return {
            cdiv(args.NPrime, BLOCK_N) * cdiv(args.DPrime, BLOCK_Dp),
            1, 1
        };
    }

    bool can_run(ImplicitGemmConv3dKernelTArgs &args) const {
        return args.BLOCK_N == BLOCK_N &&
            args.acc_dtype == acc_dtype && super_t::can_run(args);
    }

    void run(KernelArgs kargs, CUstream stream) {
        if (PARALLEL_K > 1) {
            if (kargs.feat_dtype.type == Dtype::FP16) {
                cuMemsetD16((CUdeviceptr) kargs.out, 0, kargs.NPrime * kargs.DPrime);
            } else if (kargs.feat_dtype.type == Dtype::FP32) {
                cuMemsetD32((CUdeviceptr) kargs.out, 0, kargs.NPrime * kargs.DPrime);
            } else {
                throw std::runtime_error("Unsupported feature dtype for implicit gemm conv3d: " + kargs.feat_dtype.to_string());
            }
        }
        super_t::run(kargs, stream);
    }
};

struct ImplicitGemmConv3dGradKernelArgs {
    void* dout;
    void* features;
    void* weights;
    void* indices;
    void* dfeatures;
    void* dweights;
    int N;
    int N_prime;
    int N_prime_stride;
    int D;
    int DPrime;
    int K3;
    Dtype feat_dtype;
    Dtype weight_dtype;
    Dtype acc_dtype;
    using KHash_t = std::tuple<Dtype, Dtype, Dtype, int, int, int, int, int>;

    static json serialize(const KHash_t& args) {
        json res;
        res["feat_dtype"] = std::get<0>(args).to_string();
        res["weight_dtype"] = std::get<1>(args).to_string();
        res["acc_dtype"] = std::get<2>(args).to_string();
        res["N"] = std::get<3>(args);
        res["NPrime"] = std::get<4>(args);
        res["D"] = std::get<5>(args);
        res["DPrime"] = std::get<6>(args);
        res["K3"] = std::get<7>(args);
        return res;
    }
    static KHash_t deserialize(const json &j) {
        return std::make_tuple(
            Dtype::from_string(j["feat_dtype"].get<std::string>()),
            Dtype::from_string(j["weight_dtype"].get<std::string>()),
            Dtype::from_string(j["acc_dtype"].get<std::string>()),
            j["N"].get<int>(),
            j["NPrime"].get<int>(),
            j["D"].get<int>(),
            j["DPrime"].get<int>(),
            j["K3"].get<int>()
        );
    }

    KHash_t khash() const {
        return std::make_tuple(
            feat_dtype, weight_dtype, acc_dtype,
            quant_N(N), quant_N(N_prime), D, DPrime, K3
        );
    }
};

struct ImplicitGemmConv3dDFKernelArgs : ImplicitGemmConv3dGradKernelArgs {
    KArg_t get_args() const {
        return {
            {Dtype(feat_dtype.type, true), dout},
            {Dtype(weight_dtype.type, true), weights},
            {Dtype(Dtype::INT32, true), indices},
            {Dtype(feat_dtype.type, true), dfeatures},
            {Dtype(Dtype::INT32), (void*) &N},
            {Dtype(Dtype::INT32), (void*) &N_prime},
            {Dtype(Dtype::INT32), (void*) &N_prime_stride},
            {Dtype(Dtype::INT32), (void*) &D},
            {Dtype(Dtype::INT32), (void*) &DPrime}
        };
    }
};

struct ImplicitGemmConv3dDWKernelArgs : ImplicitGemmConv3dGradKernelArgs {
    KArg_t get_args() const {
        return {
            {Dtype(feat_dtype.type, true), dout},
            {Dtype(feat_dtype.type, true), features},
            {Dtype(Dtype::INT32, true), indices},
            {Dtype(weight_dtype.type, true), dweights},
            {Dtype(Dtype::INT32), (void*) &N},
            {Dtype(Dtype::INT32), (void*) &N_prime},
            {Dtype(Dtype::INT32), (void*) &N_prime_stride},
            {Dtype(Dtype::INT32), (void*) &D},
            {Dtype(Dtype::INT32), (void*) &DPrime},
            {Dtype(Dtype::INT32), (void*) &K3}
        };
    }
};

class ImplicitGemmConv3dDFKernel : public TritonKernel<ImplicitGemmConv3dDFKernelArgs> {
    Dtype acc_dtype;
    int BLOCK_DPrime;
    int BLOCK_NPrime;
    int BLOCK_D;
public:
    ImplicitGemmConv3dDFKernel(const json &config) : 
        TritonKernel<ImplicitGemmConv3dDFKernelArgs>(config),
        acc_dtype(Dtype::from_string(config["acc_dtype"].get<std::string>())),
        BLOCK_DPrime(config["BLOCK_DPrime"]),
        BLOCK_NPrime(config["BLOCK_NPrime"]),
        BLOCK_D(config["BLOCK_D"]) {}
    
    std::tuple<int, int, int> blocks(const ImplicitGemmConv3dDFKernelArgs &args) const override {
        return {
            cdiv(args.N_prime, BLOCK_NPrime) * cdiv(args.D, BLOCK_D) * args.K3,
            1, 1
        };
    }

    void run(KernelArgs kargs, CUstream stream) {
        if (kargs.feat_dtype.type == Dtype::FP16) {
            cuMemsetD16((CUdeviceptr) kargs.dfeatures, 0, kargs.N * kargs.D);
        } else if (kargs.feat_dtype.type == Dtype::FP32) {
            cuMemsetD32((CUdeviceptr) kargs.dfeatures, 0, kargs.N * kargs.D);
        } else {
            throw std::runtime_error("Unsupported feature dtype for implicit gemm conv3d DF: " + kargs.feat_dtype.to_string());
        }
        super_t::run(kargs, stream);
    }
};

class ImplicitGemmConv3dDWKernel : public TritonKernel<ImplicitGemmConv3dDWKernelArgs> {
    Dtype acc_dtype;
    int BLOCK_DPrime;
    int BLOCK_NPrime;
    int BLOCK_D;
    int PARALLEL_K;
public:
    ImplicitGemmConv3dDWKernel(const json &config) :
        TritonKernel<ImplicitGemmConv3dDWKernelArgs>(config),
        acc_dtype(Dtype::from_string(config["acc_dtype"].get<std::string>())),
        BLOCK_DPrime(config["BLOCK_DPrime"]),
        BLOCK_NPrime(config["BLOCK_NPrime"]),
        BLOCK_D(config["BLOCK_D"]),
        PARALLEL_K(config["PARALLEL_K"])    
    {}
    
    std::tuple<int, int, int> blocks(const ImplicitGemmConv3dDWKernelArgs &args) const override {
        return {
            cdiv(args.DPrime, BLOCK_DPrime) * cdiv(args.D, BLOCK_D) * args.K3 * PARALLEL_K,
            1, 1
        };
    }

    void run(KernelArgs kargs, CUstream stream) {
        if (PARALLEL_K > 1) {
            if (kargs.weight_dtype.type == Dtype::FP16) {
                cuMemsetD16((CUdeviceptr) kargs.dweights, 0, kargs.K3 * kargs.D * kargs.DPrime);
            } else if (kargs.weight_dtype.type == Dtype::FP32) {
                cuMemsetD32((CUdeviceptr) kargs.dweights, 0, kargs.K3 * kargs.D * kargs.DPrime);
            } else {
                throw std::runtime_error("Unsupported feature dtype for implicit gemm conv3d DW: " + kargs.feat_dtype.to_string());
            }
        }
        super_t::run(kargs, stream);
    }
};

void save_kernel_map(std::string kernel_map_file);

MutexGuard<TritonAotKernels<ImplicitGemmConv3dKernelT>> get_implicit_gemm_kernels();
MutexGuard<TritonAotKernels<ImplicitGemmSortKernel>> get_implicit_sort_kernels();
MutexGuard<TritonAotKernels<ImplicitGemmMaskKernel>> get_implicit_gemm_mask_kernels();
MutexGuard<TritonAotKernels<ImplicitGemmConv3dDFKernel>> get_implicit_gemm_df_kernels();
MutexGuard<TritonAotKernels<ImplicitGemmConv3dDWKernel>> get_implicit_gemm_dw_kernels();

