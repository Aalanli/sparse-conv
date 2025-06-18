# %%
import torch
from implicit_gemm_kernel import implicit_conv3d_kernel
from triton import cdiv
import triton.language as tl

import ops
import ops.idx_gen


def conv3d_implicit_gemm(feats: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor, kernel_size: int):
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
        D,
        weights.shape[2],
        kernel_size,
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

    out_triton = conv3d_implicit_gemm(feats, indices, weights, kernel_size)
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
        out = conv3d_implicit_gemm(feats, indices, self.weight, self.kernel_size)
        return out

