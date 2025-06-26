# %%
import torch
from implicit_gemm_kernel import implicit_conv3d_kernel, implicit_conv3d_kernel_T, implicit_gemm_mask_kernel, implicit_gemm_idx_sort_kernel
from triton import cdiv
import triton.language as tl


def conv3d_implicit_gemm(feats: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor, kernel_size: int, acc_dtype=tl.float32):
    N, D = feats.shape
    N_prime, K3 = indices.shape
    out = torch.empty((N_prime, weights.shape[2]), device=feats.device, dtype=feats.dtype)
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
    implicit_gemm_idx_sort_kernel[grid](
        indices,  # [K**3, N']
        sort_inds,  # [N']
        N_prime,
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

    mask_i = torch.empty((NP, K3), device=feats.device, dtype=torch.bool)    
    implicit_gemm_mask_kernel[(NP,)](
        indices,  # [K**3, N']
        mask_i,  # [NP, K**3]
        N_prime,
        kernel_size,
        BLOCK_N
    )


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
        D,
        D_Prime,
        kernel_size,
        BLOCK_N,
        acc_dtype=acc_dtype,
        sorted=sort
    )
    return out
