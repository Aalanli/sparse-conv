#include "conv3d_implicit_gemm_T.h"
#include "triton_aot_utils.h"
#include <fstream>

int quant_N(int N) {
    std::vector<int> thresholds = {1000, 10000, 100000, 600000};
    for (size_t i = 0; i < thresholds.size(); ++i) {
        if (N <= thresholds[i]) {
            return thresholds[i];
        }
    }
    return thresholds.back();
}

extern "C" {
extern const unsigned char _binary_meta_json_start[];
extern const unsigned char _binary_meta_json_end[];

extern const unsigned char _binary_kernel_map_json_start[];
extern const unsigned char _binary_kernel_map_json_end[];
}

static thread_local std::unique_ptr<TritonAotKernels<ImplicitGemmConv3dKernelT>> implicit_gemm_kernels = nullptr;
static thread_local std::unique_ptr<TritonAotKernels<ImplicitGemmSortKernel>> implicit_sort_kernels = nullptr;
static thread_local std::unique_ptr<TritonAotKernels<ImplicitGemmMaskKernel>> implicit_gemm_mask_kernels = nullptr;

void setup_kernels() {
    if (implicit_gemm_kernels) {
        return;  // already initialized
    }
    std::string meta_json(reinterpret_cast<const char *>(_binary_meta_json_start),
                            _binary_meta_json_end - _binary_meta_json_start);
    json meta = json::parse(meta_json);
    // implicit_gemm_kernels = std::make_unique<TritonAotKernels<ImplicitGemmConv3dKernelTArgs>>(meta["conv3d_implicit_gemm"]);
    implicit_gemm_kernels = std::make_unique<TritonAotKernels<ImplicitGemmConv3dKernelT>>(meta["implicit_conv3d_kernel_T"]);
    implicit_sort_kernels = std::make_unique<TritonAotKernels<ImplicitGemmSortKernel>>(meta["implicit_gemm_idx_sort_kernel"]);
    implicit_gemm_mask_kernels = std::make_unique<TritonAotKernels<ImplicitGemmMaskKernel>>(meta["implicit_gemm_mask_kernel"]);
}

TritonAotKernels<ImplicitGemmConv3dKernelT>* get_implicit_gemm_kernels() {
    setup_kernels();
    return implicit_gemm_kernels.get();
}

TritonAotKernels<ImplicitGemmSortKernel>* get_implicit_sort_kernels() {
    setup_kernels();
    return implicit_sort_kernels.get();
}

TritonAotKernels<ImplicitGemmMaskKernel>* get_implicit_gemm_mask_kernels() {
    setup_kernels();
    return implicit_gemm_mask_kernels.get();
}

void save_kernel_map(std::string kernel_map_file) {
    json kmap;
    kmap["implicit_conv3d_kernel_T"] = get_implicit_gemm_kernels()->dump_kernel_map();
    kmap["implicit_gemm_idx_sort_kernel"] = get_implicit_sort_kernels()->dump_kernel_map();
    kmap["implicit_gemm_mask_kernel"] = get_implicit_gemm_mask_kernels()->dump_kernel_map();
    std::ofstream file(kernel_map_file);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open kernel map file: " + kernel_map_file);
    }
    file << kmap.dump(4);
}
