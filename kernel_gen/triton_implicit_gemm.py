# %%
import torch
from implicit_gemm_kernel import implicit_conv3d_kernel
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

