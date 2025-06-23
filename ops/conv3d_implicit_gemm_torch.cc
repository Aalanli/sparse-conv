#include <cstdint>
#include "conv3d_implicit_gemm.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <memory>
#include <torch/script.h>
#include <tuple>
#include <vector>


thread_local std::unique_ptr<Conv3DKernels> kernels = nullptr;

void setup_kernels() {
    if (kernels != nullptr) {
        return; // Already set up
    }
    kernels = std::make_unique<Conv3DKernels>();
}

void save_kernel_map(std::string kernel_map_file) {
    setup_kernels();
    kernels->save_kernel_map(kernel_map_file);
}

torch::Tensor conv3d_implicit_gemm_torch_forward(
    torch::Tensor features, // [N, D]
    torch::Tensor indices, // [N', K**3]
    torch::Tensor weights, // [K**3, D, D']
    int64_t K,
    std::string acc_dtype
) {
    setup_kernels();
    TORCH_CHECK(features.is_cuda(),  "features must be a CUDA tensor");
    TORCH_CHECK(indices.is_cuda() , "indices must be a CUDA tensor");
    TORCH_CHECK(weights.is_cuda() , "weights must be a CUDA tensor");
    TORCH_CHECK(kernels != nullptr, "Kernels must be set up before calling conv3d_implicit_gemm_torch");


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
        TORCH_CHECK(features.dtype() == torch::kHalf, "features must be of type torch::kHalf");
        TORCH_CHECK(weights.dtype() == torch::kHalf, "weights must be of type torch::kHalf");
        features_ptr = reinterpret_cast<CUdeviceptr>(features.data_ptr<at::Half>());
        indices_ptr = reinterpret_cast<CUdeviceptr>(indices.data_ptr<int32_t>());
        weights_ptr = reinterpret_cast<CUdeviceptr>(weights.data_ptr<at::Half>());
        output_ptr = reinterpret_cast<CUdeviceptr>(output.data_ptr<at::Half>());
        dtype = "fp16";
    } else if (features.dtype() == torch::kFloat) {
        TORCH_CHECK(features.dtype() == torch::kFloat, "features must be of type torch::kFloat");
        TORCH_CHECK(weights.dtype() == torch::kFloat, "weights must be of type torch::kFloat");
        features_ptr = reinterpret_cast<CUdeviceptr>(features.data_ptr<float>());
        indices_ptr = reinterpret_cast<CUdeviceptr>(indices.data_ptr<int32_t>());
        weights_ptr = reinterpret_cast<CUdeviceptr>(weights.data_ptr<float>());
        output_ptr = reinterpret_cast<CUdeviceptr>(output.data_ptr<float>());
        dtype = "fp32";
    } else {
        throw std::runtime_error(std::string("Unsupported dtype: ") + std::string(features.dtype().TypeName<std::string>()));
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    kernels->run(
        features_ptr, // [N, D]
        indices_ptr, // [N', K**3]
        weights_ptr, // [K**3, D, D']
        output_ptr, // [N', D']
        N,
        NPrime,
        D,
        DPrime,
        K,
        acc_dtype,
        dtype,
        stream
    );

    return output;
}

std::tuple<torch::Tensor, torch::Tensor> conv3d_implicit_gemm_torch_backward(
    torch::Tensor dout, // [N', D'] 
    torch::Tensor feats, // [N, D]
    torch::Tensor indices, // [N', K**3]
    torch::Tensor weights // [K**3, D, D']
) {
    TORCH_CHECK(dout.dim() == 2, "dout must be a 2D tensor");
    TORCH_CHECK(feats.dim() == 2, "feats must be a 2D tensor");
    TORCH_CHECK(indices.dim() == 2, "indices must be a 2D tensor");
    TORCH_CHECK(weights.dim() == 3, "weights must be a 3D tensor");

    int N = feats.size(0);
    int K3 = weights.size(0);
    int D = weights.size(1);
    int DPrime = weights.size(2);

    auto feats_padded   = at::zeros({N + 1, D}, feats.options());
    feats_padded.narrow(/*dim=*/0, /*start=*/0, /*length=*/N).copy_(feats);

    auto indices_flat = indices.view({-1});
    auto indices_ = at::where(indices_flat < 0, N, indices_flat);
    auto weights_view = weights.view({-1, DPrime});

    auto feats_gathered = feats_padded.index_select(0, indices_).view({-1, K3 * D});
    auto dfeats_gathered = dout.mm(weights_view.t());
    auto dweights = feats_gathered.t().mm(dout).view({K3, D, DPrime});
    auto dfeats = at::zeros_like(feats_padded);
    dfeats.index_add_(0, indices_, dfeats_gathered.view({-1, D}));
    dfeats = dfeats.narrow(/*dim=*/0, /*start=*/0, /*length=*/N);
    return std::make_tuple(dfeats, dweights);
}

class Conv3dImplicitGemm : public torch::autograd::Function<Conv3dImplicitGemm> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor features, // [N, D]
        torch::Tensor indices, // [N', K**3]
        torch::Tensor weights, // [K**3, D, D']
        int64_t K,
        std::string acc_dtype
    ) {
        ctx->save_for_backward({features, indices, weights});
        return conv3d_implicit_gemm_torch_forward(features, indices, weights, K, acc_dtype);
    }

    static std::vector<torch::Tensor> backward(
        torch::autograd::AutogradContext *ctx,
        std::vector<torch::Tensor> dout // [N', D']
    ) {
        auto saved = ctx->get_saved_variables();
        TORCH_CHECK(saved.size() == 3, "Expected 3 saved tensors, got ", saved.size());
        auto feats = saved[0];
        auto indices = saved[1];
        auto weights = saved[2];

        auto dfeats_and_dweights = conv3d_implicit_gemm_torch_backward(dout[0], feats, indices, weights);
        return {
            std::get<0>(dfeats_and_dweights), // dfeats
            torch::Tensor(), // No gradient for indices
            std::get<1>(dfeats_and_dweights), // dweights
            torch::Tensor(), // No gradient for K
            torch::Tensor() // No gradient for acc_dtype
        };
    }
    
};

torch::Tensor conv3d_implicit_gemm_torch(
    torch::Tensor features, // [N, D]
    torch::Tensor indices, // [N', K**3]
    torch::Tensor weights, // [K**3, D, D']
    int64_t K,
    std::string acc_dtype
) {
    return Conv3dImplicitGemm::apply(
        features, // [N, D]
        indices, // [N', K**3]
        weights, // [K**3, D, D']
        K,
        acc_dtype
    );
}


TORCH_LIBRARY(conv3d_implicit_gemm, m) {
    m.def("save_kernel_map", &save_kernel_map);
    m.def("conv3d_implicit_gemm_torch", &conv3d_implicit_gemm_torch);
}

