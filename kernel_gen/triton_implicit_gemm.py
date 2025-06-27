# %%
import torch
from kernel_gen.implicit_gemm_kernel import implicit_conv3d_kernel, implicit_conv3d_kernel_T, implicit_gemm_mask_kernel, implicit_gemm_idx_sort_kernel
from kernel_gen.implicit_gemm_kernel import implicit_gemm_dF_kernel, implicit_gemm_dW_kernel
from triton import cdiv
import triton.language as tl


def conv3d_implicit_gemm(feats: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor, kernel_size: int, acc_dtype=tl.float32):
    N, D = feats.shape
    N_prime, K3 = indices.shape
    out = torch.zeros((N_prime, weights.shape[2]), device=feats.device, dtype=feats.dtype)
    grid = lambda meta: (cdiv(N_prime, meta["BLOCK_N"]) * cdiv(meta["D_prime"], meta["BLOCK_Dp"]) * meta['PARALLEL_K'],)
    implicit_conv3d_kernel[grid](
        feats,  # [N, D]
        indices,  # [N', K**3]
        weights,  # [K**3, D, D']
        out,  # [N', D']
        N,
        N_prime,
        D,
        weights.shape[2],
        kernel_size,
        acc_dtype=acc_dtype,
    )
    return out


def sort_indices(indices: torch.Tensor):
    K3, N_prime = indices.shape
    if K3 <= 32:
        sort_dtype = tl.int32
        sort_inds = torch.empty(N_prime, device=indices.device, dtype=torch.int32)
        BLOCK_K = 32
    elif K3 <= 64:
        sort_dtype = tl.int64
        sort_inds = torch.empty(N_prime, device=indices.device, dtype=torch.int64)
        BLOCK_K = 64
    else:
        raise ValueError(f"Unsupported K3 size: {K3}")
    grid = lambda meta: (cdiv(N_prime, meta["BLOCK_N"]),)
    N_stride = indices.stride(0)
    implicit_gemm_idx_sort_kernel[grid](
        indices,  # [K**3, N']
        sort_inds,  # [N']
        N_prime,
        N_stride,
        K3,
        BLOCK_K,
        mask_dtype=sort_dtype,
    )
    sort_idx = torch.argsort(sort_inds)
    indices = indices[:, sort_idx]
    return indices, sort_idx

def conv3d_implicit_gemm_T(
    feats: torch.Tensor, 
    indices: torch.Tensor, 
    weights: torch.Tensor, 
    kernel_size: int, 
    acc_dtype=tl.float32,
    BLOCK_N: int = 32,
    sort: bool = True
):
    N, D = feats.shape
    K3, N_prime = indices.shape
    NP = cdiv(N_prime, BLOCK_N)

    if sort:
        indices, inv = sort_indices(indices)
    else:
        inv = None
    N_prime_stride = indices.stride(0)
    torch.cuda.synchronize()
    mask_i = torch.empty((NP, K3), device=feats.device, dtype=torch.bool)    
    implicit_gemm_mask_kernel[(NP,)](
        indices,  # [K**3, N']
        mask_i,  # [NP, K**3]
        N_prime,
        N_prime_stride,
        kernel_size,
        BLOCK_N
    )

    torch.cuda.synchronize()

    D_Prime = weights.shape[2]
    out = torch.empty((N_prime, D_Prime), device=feats.device, dtype=feats.dtype)
    grid = lambda meta: (NP * cdiv(meta["D_prime"], meta["BLOCK_Dp"])
                         * meta['PARALLEL_K'],)
    implicit_conv3d_kernel_T[grid](
        feats,  # [N, D]
        indices,  # [K**3, N']
        mask_i, # [NP, K**3]
        weights,  # [K**3, D, D']
        inv,
        out, # [N', D']
        N,
        N_prime,
        N_prime_stride, 
        D,
        D_Prime,
        kernel_size,
        BLOCK_N,
        acc_dtype=acc_dtype,
        sorted=sort
    )
    torch.cuda.synchronize()
    return out


def implicit_gemm_grad(dout: torch.Tensor, features: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor, acc_dtype=tl.float32):
    N_prime, D_prime = dout.shape
    K3, D, _ = weights.shape
    N, _ = features.shape
    N_prime_stride = indices.stride(0)

    dfeatures = torch.zeros_like(features)
    grid = lambda meta: (cdiv(N_prime, meta['BLOCK_NPrime']) * cdiv(D, meta['BLOCK_D']) * K3,)
    implicit_gemm_dF_kernel[grid](
        dout, weights, indices, dfeatures, N, N_prime, N_prime_stride, 
        D, D_prime, acc_dtype=acc_dtype
    )

    dweight = torch.empty_like(weights)
    grid = lambda meta: (cdiv(D_prime, meta['BLOCK_DPrime']) * cdiv(D, meta['BLOCK_D']) * K3,)
    implicit_gemm_dW_kernel[grid](
        dout, features, indices, dweight,
        N, N_prime, N_prime_stride,
        D, D_prime, K3, acc_dtype=acc_dtype
    )

    return dfeatures, dweight


