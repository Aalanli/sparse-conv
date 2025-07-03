#include "conv3d_implicit_gemm_T.h"
#include "triton_aot_utils.h"
#include <fstream>
#include <mutex>

int quant_N(int N) {
    std::vector<int> thresholds = {1000, 10000, 100000, 600000};
    for (size_t i = 0; i < thresholds.size(); ++i) {
        if (N <= thresholds[i]) {
            return thresholds[i];
        }
    }
    return thresholds.back();
}


void set_zero(void* ptr, Dtype dtype, int size, CUstream stream) {
    if (dtype.type == Dtype::FP16) {
        cuMemsetD16Async((CUdeviceptr) ptr, 0, size, stream);
    } else if (dtype.type == Dtype::FP32) {
        cuMemsetD32Async((CUdeviceptr) ptr, 0, size, stream);
    } else if (dtype.type == Dtype::INT32) {
        cuMemsetD32Async((CUdeviceptr) ptr, 0, size, stream);
    } else if (dtype.type == Dtype::INT64) {
        cuMemsetD32Async((CUdeviceptr) ptr, 0, size * 2, stream);
    } else if (dtype.type == Dtype::BOOL) {
        cuMemsetD8Async((CUdeviceptr) ptr, 0, size, stream);
    } else {
        throw std::runtime_error("Unsupported dtype for set_zero: " + dtype.to_string());
    }
}

extern "C" {
extern const unsigned char _binary_meta_json_start[];
extern const unsigned char _binary_meta_json_end[];

extern const unsigned char _binary_kernel_map_json_start[];
extern const unsigned char _binary_kernel_map_json_end[];
}

static Mutex<TritonAotKernels<ImplicitGemmConv3dKernelT>> implicit_gemm_kernels;
static Mutex<TritonAotKernels<ImplicitGemmSortKernel>> implicit_sort_kernels;
static Mutex<TritonAotKernels<ImplicitGemmMaskKernel>> implicit_gemm_mask_kernels;
static Mutex<TritonAotKernels<ImplicitGemmConv3dDFKernel>> implicit_gemm_df_kernels;
static Mutex<TritonAotKernels<ImplicitGemmConv3dDWKernel>> implicit_gemm_dw_kernels;

void setup_kernels() {
    if (implicit_gemm_kernels.lock().get() != nullptr) {
        return;  // already initialized
    }
    std::cerr << "Setting up implicit gemm kernels..." << std::endl;
    std::string meta_json(reinterpret_cast<const char *>(_binary_meta_json_start),
                            _binary_meta_json_end - _binary_meta_json_start);
    json meta = json::parse(meta_json);
    // implicit_gemm_kernels = std::make_unique<TritonAotKernels<ImplicitGemmConv3dKernelTArgs>>(meta["conv3d_implicit_gemm"]);
    implicit_gemm_kernels = std::make_unique<TritonAotKernels<ImplicitGemmConv3dKernelT>>(meta["implicit_conv3d_kernel_T"]);
    implicit_sort_kernels = std::make_unique<TritonAotKernels<ImplicitGemmSortKernel>>(meta["implicit_gemm_idx_sort_kernel"]);
    implicit_gemm_mask_kernels = std::make_unique<TritonAotKernels<ImplicitGemmMaskKernel>>(meta["implicit_gemm_mask_kernel"]);
    implicit_gemm_df_kernels = std::make_unique<TritonAotKernels<ImplicitGemmConv3dDFKernel>>(meta["implicit_gemm_dF_kernel"]);
    implicit_gemm_dw_kernels = std::make_unique<TritonAotKernels<ImplicitGemmConv3dDWKernel>>(meta["implicit_gemm_dW_kernel"]);

    std::string kernel_map_json(reinterpret_cast<const char *>(_binary_kernel_map_json_start),
                                    _binary_kernel_map_json_end - _binary_kernel_map_json_start);
    json kernel_map = json::parse(kernel_map_json);
    if (kernel_map.contains("implicit_conv3d_kernel_T")) {
        implicit_gemm_kernels.lock().get()->load_kernel_map(kernel_map["implicit_conv3d_kernel_T"]);
    }
    if (kernel_map.contains("implicit_gemm_idx_sort_kernel")) {
        implicit_sort_kernels.lock().get()->load_kernel_map(kernel_map["implicit_gemm_idx_sort_kernel"]);
    }
    if (kernel_map.contains("implicit_gemm_mask_kernel")) {
        implicit_gemm_mask_kernels.lock().get()->load_kernel_map(kernel_map["implicit_gemm_mask_kernel"]);
    }
    if (kernel_map.contains("implicit_gemm_dF_kernel")) {
        implicit_gemm_df_kernels.lock().get()->load_kernel_map(kernel_map["implicit_gemm_dF_kernel"]);
    }
    if (kernel_map.contains("implicit_gemm_dW_kernel")) {
        implicit_gemm_dw_kernels.lock().get()->load_kernel_map(kernel_map["implicit_gemm_dW_kernel"]);
    }
}

MutexGuard<TritonAotKernels<ImplicitGemmConv3dKernelT>> get_implicit_gemm_kernels() {
    setup_kernels();
    return implicit_gemm_kernels.lock();
}

MutexGuard<TritonAotKernels<ImplicitGemmSortKernel>> get_implicit_sort_kernels() {
    setup_kernels();
    return implicit_sort_kernels.lock();
}

MutexGuard<TritonAotKernels<ImplicitGemmMaskKernel>> get_implicit_gemm_mask_kernels() {
    setup_kernels();
    return implicit_gemm_mask_kernels.lock();
}

MutexGuard<TritonAotKernels<ImplicitGemmConv3dDFKernel>> get_implicit_gemm_df_kernels() {
    setup_kernels();
    return implicit_gemm_df_kernels.lock();
}

MutexGuard<TritonAotKernels<ImplicitGemmConv3dDWKernel>> get_implicit_gemm_dw_kernels() {
    setup_kernels();
    return implicit_gemm_dw_kernels.lock();
}

void save_kernel_map(std::string kernel_map_file) {
    json kmap;
    kmap["implicit_conv3d_kernel_T"] = get_implicit_gemm_kernels().get()->dump_kernel_map();
    kmap["implicit_gemm_idx_sort_kernel"] = get_implicit_sort_kernels().get()->dump_kernel_map();
    kmap["implicit_gemm_mask_kernel"] = get_implicit_gemm_mask_kernels().get()->dump_kernel_map();
    kmap["implicit_gemm_dF_kernel"] = get_implicit_gemm_df_kernels().get()->dump_kernel_map();
    kmap["implicit_gemm_dW_kernel"] = get_implicit_gemm_dw_kernels().get()->dump_kernel_map();
    std::ofstream file(kernel_map_file);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open kernel map file: " + kernel_map_file);
    }
    file << kmap.dump(-1);
    file.close();
}

