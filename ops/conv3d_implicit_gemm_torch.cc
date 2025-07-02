#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <sstream>
#include <torch/script.h>

#include <cstdint>
#include <tuple>
#include <vector>

#include "c10/core/ScalarType.h"
#include "conv3d_implicit_gemm_T.h"

Dtype get_dtype(const torch::Tensor &tensor) {
    if (tensor.dtype() == torch::kFloat32) {
        return Dtype(Dtype::FP32, true);
    } else if (tensor.dtype() == torch::kFloat16) {
        return Dtype(Dtype::FP16, true);
    } else if (tensor.dtype() == torch::kInt32) {
        return Dtype(Dtype::INT32, true);
    } else if (tensor.dtype() == torch::kInt64) {
        return Dtype(Dtype::INT64, true);
    } else if (tensor.dtype() == torch::kBool) {
        return Dtype(Dtype::BOOL, true);
    } else {
        std::ostringstream oss;
        oss << "Unsupported tensor dtype: " << tensor.dtype();
        throw std::runtime_error(oss.str());
    }
}

torch::Tensor conv3d_implicit_gemm_torch_forward(torch::Tensor features,  // [N, D]
                                                 torch::Tensor indices,   // [K**3, N']
                                                 torch::Tensor weights,   // [K**3, D, D']
                                                 int64_t K, std::string acc_dtype, int64_t BLOCK_N, bool sorted) {
    TORCH_CHECK(features.is_cuda(), "features must be a CUDA tensor");
    TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
    TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");
    TORCH_CHECK(features.dim() == 2, "features must be a 2D tensor");
    TORCH_CHECK(indices.dim() == 2, "indices must be a 2D tensor");
    TORCH_CHECK(weights.dim() == 3, "weights must be a 3D tensor");
    TORCH_CHECK(features.dtype() == torch::kFloat32 || features.dtype() == torch::kFloat16,
               "features must be of type float32 or float16");
    TORCH_CHECK(indices.dtype() == torch::kInt32, "indices must be of type int32");
    TORCH_CHECK(weights.dtype() == torch::kFloat32 || weights.dtype() == torch::kFloat16,
               "weights must be of type float32 or float16");
               TORCH_CHECK(K * K * K <= 64, "K**3 must be less than or equal to 64");
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (!features.is_contiguous()) {
        features = features.contiguous();
    }
    if (!weights.is_contiguous()) {
        weights = weights.contiguous();
    }

    int N = features.size(0);
    int D = features.size(1);
    int NPrime = indices.size(1);
    int NPrimeStride = indices.stride(0);
    int DPrime = weights.size(2);

    
    int K3 = K * K * K;
    int NP = cdiv(NPrime, BLOCK_N);
    auto output = torch::zeros({NPrime, DPrime}, features.options());
    auto mask_i = torch::empty({NP, K3}, features.options().dtype(torch::kBool));
    
    torch::Tensor out_perm;
    if (sorted) {
        auto sort_dtype = K3 <= 32 ? torch::kInt32 : torch::kInt64;
        auto sort_inds = torch::empty({NPrime}, features.options().dtype(sort_dtype));
        get_implicit_sort_kernels()->run({
            indices.data_ptr(),
            sort_inds.data_ptr(),
            get_dtype(sort_inds),
            NPrime,
            NPrimeStride,
            K3
        }, stream);
        out_perm = torch::argsort(sort_inds, 0, false).to(torch::kInt32);
        indices = indices.index_select(1, out_perm);
    }

    get_implicit_gemm_mask_kernels()->run({
        indices.data_ptr(),
        mask_i.data_ptr(),
        NPrime,
        NPrimeStride,
        K3,
        (int) BLOCK_N,
    }, stream);
    
    ImplicitGemmConv3dKernelTArgs args{
        get_dtype(features),
        get_dtype(weights),
        Dtype::from_string(acc_dtype),

        features.data_ptr(),
        indices.data_ptr(),
        mask_i.data_ptr(),
        weights.data_ptr(),
        sorted ? out_perm.data_ptr() : nullptr,
        output.data_ptr(),
        N,
        NPrime,
        NPrimeStride,
        D,
        DPrime,
        (int) K,
        (int) BLOCK_N,
        sorted
    };

    get_implicit_gemm_kernels()->run(args, stream);
    return output;
}

std::tuple<torch::Tensor, torch::Tensor> conv3d_implicit_gemm_torch_backward(torch::Tensor dout,     // [N', D']
                                                                             torch::Tensor feats,    // [N, D]
                                                                             torch::Tensor indices,  // [N', K**3]
                                                                             torch::Tensor weights   // [K**3, D, D']
) {
    TORCH_CHECK(dout.dim() == 2, "dout must be a 2D tensor");
    TORCH_CHECK(feats.dim() == 2, "feats must be a 2D tensor");
    TORCH_CHECK(indices.dim() == 2, "indices must be a 2D tensor");
    TORCH_CHECK(weights.dim() == 3, "weights must be a 3D tensor");

    int N = feats.size(0);
    int K3 = weights.size(0);
    int D = weights.size(1);
    int DPrime = weights.size(2);

    auto feats_padded = at::zeros({N + 1, D}, feats.options());
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
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                                 torch::Tensor features,  // [N, D]
                                 torch::Tensor indices,   // [N', K**3]
                                 torch::Tensor weights,   // [K**3, D, D']
                                 int64_t K, std::string acc_dtype, int64_t BLOCK_N, bool sorted) {
        ctx->save_for_backward({features, indices, weights});
        return conv3d_implicit_gemm_torch_forward(features, indices, weights, K, acc_dtype, BLOCK_N, sorted);
    }

    static std::vector<torch::Tensor> backward(torch::autograd::AutogradContext *ctx,
                                               std::vector<torch::Tensor> dout  // [N', D']
    ) {
        auto saved = ctx->get_saved_variables();
        TORCH_CHECK(saved.size() == 3, "Expected 3 saved tensors, got ", saved.size());
        auto feats = saved[0];
        auto indices = saved[1];
        auto weights = saved[2];

        auto dfeats_and_dweights = conv3d_implicit_gemm_torch_backward(dout[0], feats, indices, weights);
        return {
            std::get<0>(dfeats_and_dweights),  // dfeats
            torch::Tensor(),                   // No gradient for indices
            std::get<1>(dfeats_and_dweights),  // dweights
            torch::Tensor(),                   // No gradient for K
            torch::Tensor(),                   // No gradient for acc_dtype
            torch::Tensor(),                   // No gradient for BLOCK_N
            torch::Tensor()                    // No gradient for sorted
        };
    }
};

torch::Tensor conv3d_implicit_gemm_torch(torch::Tensor features,  // [N, D]
                                         torch::Tensor indices,   // [N', K**3]
                                         torch::Tensor weights,   // [K**3, D, D']
                                         int64_t K, std::string acc_dtype, int64_t BLOCK_N, bool sorted) {
    return Conv3dImplicitGemm::apply(features,  // [N, D]
                                     indices,   // [N', K**3]
                                     weights,   // [K**3, D, D']
                                     K, acc_dtype, BLOCK_N, sorted);  // [N', D']
}

TORCH_LIBRARY(conv3d_implicit_gemm, m) {
    m.def("save_kernel_map", &save_kernel_map);
    m.def("conv3d_implicit_gemm_torch", &conv3d_implicit_gemm_torch);
}
