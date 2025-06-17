# %%
import ops
import torch
import triton
import triton.language as tl
from triton import cdiv

import ops.idx_gen


@triton.jit
def or_combine(a, b):
    return a | b


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 16, "BLOCK_Dp": 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 32, 'BLOCK_K': 16, 'BLOCK_Dp': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 32, 'BLOCK_K': 32, 'BLOCK_Dp': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 32, 'BLOCK_K': 16, 'BLOCK_Dp': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 32, 'BLOCK_Dp': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 16, 'BLOCK_Dp': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 32, 'BLOCK_Dp': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 32, 'BLOCK_Dp': 32}, num_warps=4, num_stages=2),

        triton.Config({"BLOCK_N": 16, "BLOCK_K": 16, "BLOCK_Dp": 16}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_N': 32, 'BLOCK_K': 16, 'BLOCK_Dp': 16}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_N': 32, 'BLOCK_K': 32, 'BLOCK_Dp': 16}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_N': 32, 'BLOCK_K': 16, 'BLOCK_Dp': 32}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 32, 'BLOCK_Dp': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 16, 'BLOCK_Dp': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 32, 'BLOCK_Dp': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 32, 'BLOCK_Dp': 32}, num_warps=4, num_stages=3),

        triton.Config({"BLOCK_N": 16, "BLOCK_K": 32, "BLOCK_Dp": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 32, "BLOCK_Dp": 32}, num_warps=2, num_stages=3),
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 32, "BLOCK_Dp": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 64, "BLOCK_Dp": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 32, "BLOCK_Dp": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 32, "BLOCK_Dp": 128}, num_warps=8, num_stages=3),

    ],
    key=["N", "N_prime", "K", "D", "D_prime"],
)
@triton.jit
def implicit_conv3d_kernel(
    features,  # [N, D]
    indices,  # [N', K**3]
    weights,  # [K**3, D, D']
    output,  # [N', D']
    N,
    N_prime,
    K: tl.constexpr,
    D: tl.constexpr,
    D_prime: tl.constexpr,
    BLOCK_N: tl.constexpr,  # tile size for N
    BLOCK_K: tl.constexpr,  # tile size for K
    BLOCK_Dp: tl.constexpr,  # tile size for D
    acc_dtype: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_n = tl.cdiv(N_prime, BLOCK_N)
    # TODO: potential optimization here: grid-rasterization for better l2-cache reuse
    pid_n = pid % grid_n
    pid_dp = pid // grid_n

    stride_k = K * K * K
    num_blocks_k_per_d = tl.cdiv(D, BLOCK_K)

    ind_ptr = indices + (tl.arange(0, BLOCK_N) + pid_n * BLOCK_N) * stride_k
    acc = tl.zeros((BLOCK_N, BLOCK_Dp), dtype=acc_dtype)  # TODO: adjust acc dtype
    for k in range(stride_k):
        # [BLOCK_N]
        inds = tl.load(ind_ptr, mask=(tl.arange(0, BLOCK_N) + pid_n * BLOCK_N) < N_prime, other=-1)
        # tl.device_print("inds", inds)
        if tl.reduce(inds != -1, 0, or_combine):
            for ki in range(tl.cdiv(D, BLOCK_K)):
                offset_k = k * D + ki * BLOCK_K

                ptr_f = features + (inds[:, None] * D + tl.arange(0, BLOCK_K)[None, :] + ki * BLOCK_K)
                mask_f = (inds[:, None] != -1) & (tl.arange(0, BLOCK_K)[None, :] + ki * BLOCK_K < D)

                ptr_w = weights + (
                    (tl.arange(0, BLOCK_K)[:, None] + offset_k) * D_prime
                    + tl.arange(0, BLOCK_Dp)[None, :]
                    + pid_dp * BLOCK_Dp
                )
                mask_w = ((tl.arange(0, BLOCK_K)[:, None] + offset_k) < stride_k * D) & (
                    tl.arange(0, BLOCK_Dp)[None, :] + pid_dp * BLOCK_Dp < D_prime
                )

                feats = tl.load(ptr_f, mask_f, other=0.0).to(acc_dtype)
                ws = tl.load(ptr_w, mask_w, other=0.0).to(acc_dtype)

                acc += tl.dot(feats, ws, out_dtype=acc_dtype).to(acc_dtype)
        ind_ptr += 1
    out_ptr = output + (
        (tl.arange(0, BLOCK_N)[:, None] + pid_n * BLOCK_N) * D_prime
        + tl.arange(0, BLOCK_Dp)[None, :]
        + pid_dp * BLOCK_Dp
    )
    out_mask = (tl.arange(0, BLOCK_N)[:, None] + pid_n * BLOCK_N < N_prime) & (
        tl.arange(0, BLOCK_Dp)[None, :] + pid_dp * BLOCK_Dp < D_prime
    )

    tl.store(out_ptr, acc.to(out_ptr.dtype.element_ty), out_mask)


def conv3d_subm(feats: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor, kernel_size: int):
    N, D = feats.shape
    N_prime, K3 = indices.shape
    out = torch.zeros((N_prime, weights.shape[2]), device=feats.device, dtype=feats.dtype)
    grid = lambda meta: (cdiv(N_prime, meta["BLOCK_N"]) * cdiv(meta["D_prime"], meta["BLOCK_Dp"]),)
    implicit_conv3d_kernel[grid](
        feats,  # [N, D]
        indices,  # [N', K**3]
        weights,  # [K**3, D, D']
        out,  # [N', D']
        N,
        N_prime,
        kernel_size,
        D,
        weights.shape[2],
        acc_dtype=tl.float32,
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
    print(indices)

    out_triton = conv3d_subm(feats, indices, weights, kernel_size)
    out_ref = reference_conv3d_subm(feats, indices, weights, kernel_size)

    print((out_triton - out_ref).abs().max())
    if not torch.allclose(out_triton, out_ref, atol=1e-3, rtol=1e-3):
        print("Outputs do not match!")
        print(out_triton)
        print(out_ref)


class Conv3DSubmModule(torch.nn.Module):
    def __init__(self, kernel_size: int, in_channels: int = 64, out_channels: int = 64):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = torch.nn.Parameter(
            torch.randn(kernel_size**3, in_channels, out_channels, device="cuda", dtype=torch.float16)
        )

    def forward(self, feats: torch.Tensor, coords: torch.Tensor):
        # coords: [N, 4] where last dimension is (x, y, z, batch_id)
        # feats: [N, D]
        # indices: [batch_size, kernel_size**3]
        indices = ops.idx_gen.gen_conv3d_subm_indices(coords, self.kernel_size)

        # indices: [batch_size, kernel_size**3]
        out = conv3d_subm(feats, indices, self.weight, self.kernel_size)
        return out
    


