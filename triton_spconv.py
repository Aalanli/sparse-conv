# %%
import torch
from implicit_gemm_kernel import implicit_conv3d_kernel, implicit_conv3d_kernel_T, implicit_gemm_mask_kernel, implicit_gemm_idx_sort_kernel
from triton import cdiv
import triton.language as tl

import ops
import ops.idx_gen


def conv3d_implicit_gemm(feats: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor, kernel_size: int, acc_dtype=tl.float32):
    N, D = feats.shape
    K3, N_prime = indices.shape
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


def reference_conv3d_subm(feats: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor, kernel_size: int):
    n = feats.shape[0]
    feats = torch.nn.functional.pad(feats, (0, 0, 0, 1))
    indices = torch.where(indices < 0, n, indices)
    out = feats[indices.flatten()].reshape(-1, kernel_size**3 * feats.shape[1]) @ weights.view(-1, weights.shape[-1])
    return out


def compare_conv3d_subm():
    feats = torch.randn(1000, 64, device="cuda")
    coords = torch.randint(0, 10, (1000, 4), device="cuda").int()
    weights = torch.randn(27, 64, 64, device="cuda")  # 27 = 3**3 for kernel size 3
    kernel_size = 3
    indices = torch.full((10, kernel_size**3), -1, device="cuda", dtype=torch.int32)
    indices[:, 27 // 2] = torch.arange(10, device="cuda", dtype=torch.int32)
    # indices = torch.tensor([[1, 0, 1, 0] + [-1] * (27-4)], device="cuda", dtype=torch.int32)
    # print(indices)

    out_triton2 = conv3d_implicit_gemm_T(feats, indices.T.contiguous(), weights, kernel_size, sort=True)
    out_triton = conv3d_implicit_gemm(feats, indices, weights, kernel_size)
    out_ref = reference_conv3d_subm(feats, indices, weights, kernel_size)

    if not torch.allclose(out_triton, out_ref, atol=1e-1, rtol=1e-3):
        print("Outputs do not match!")
        print((out_triton - out_ref).abs().max())
        # print(out_triton)
        # print(out_ref)
    
    if not torch.allclose(out_triton2, out_ref, atol=1e-1, rtol=1e-3):
        print("Outputs from transposed kernel do not match!")
        print((out_triton2 - out_ref).abs().max())
        print(out_triton2)
        print(out_ref)

# compare_conv3d_subm()


class Conv3DSubmModule(torch.nn.Module):
    def __init__(self, kernel_size: int, in_channels: int = 64, out_channels: int = 64, acc_dtype=tl.float32):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = torch.nn.Parameter(
            torch.randn(kernel_size**3, in_channels, out_channels, device="cuda", dtype=torch.float16)
        )
        self.acc_dtype = acc_dtype

    def forward(self, feats: torch.Tensor, coords: torch.Tensor):
        # coords: [N, 4] where last dimension is (x, y, z, batch_id)
        # feats: [N, D]
        # indices: [batch_size, kernel_size**3]
        indices = ops.idx_gen.gen_conv3d_subm_indices(coords, self.kernel_size)

        # indices: [batch_size, kernel_size**3]
        out = conv3d_implicit_gemm(feats, indices, self.weight, self.kernel_size, acc_dtype=self.acc_dtype)
        return out

