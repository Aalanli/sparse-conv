# %%
import ops
import glob
import os
import torch

import ops.conv3d_implicit_gemm as aot_implicit_gemm
import ops.idx_gen
import triton.language as tl
from triton_spconv import conv3d_implicit_gemm
from implicit_gemm_kernel import implicit_conv3d_kernel
from utils import get_voxel_coords


def reference_conv3d_subm(feats: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor, kernel_size: int):
    n = feats.shape[0]
    feats = torch.nn.functional.pad(feats, (0, 0, 0, 1))
    indices = torch.where(indices < 0, n, indices)
    out = feats[indices.flatten()].reshape(-1, kernel_size**3 * feats.shape[1]) @ weights.view(-1, weights.shape[-1])
    return out

def compare_conv3d_subm(coords, dim_in, dim_out, kernel_size):
    n = coords.shape[0]
    feats = torch.randn(n, dim_in, device="cuda", dtype=torch.float16)
    weights = torch.randn(kernel_size ** 3, dim_in, dim_out, device="cuda", dtype=torch.float16) / dim_in**0.5
    indices = ops.idx_gen.gen_conv3d_subm_indices(coords, kernel_size)

    _ = conv3d_implicit_gemm(feats, indices, weights, kernel_size)
    out_triton = conv3d_implicit_gemm(feats, indices, weights, kernel_size)
    out_ref = reference_conv3d_subm(feats, indices, weights, kernel_size)

    if not torch.allclose(out_triton, out_ref, atol=1e-1, rtol=1e-3):
        diffs = (out_triton - out_ref).abs()
        print(f"Max diff: {diffs.max()}")
        print("Outputs do not match!")
        print(out_triton)
        print(out_ref)

    out_aot = aot_implicit_gemm.conv3d_implicit_gemm(
        feats, indices, weights, kernel_size
    )
    # print(out_aot)
    if not (torch.allclose(out_ref, out_aot, atol=1e-1, rtol=1e-2)):
        print(f"Outputs do not match! ref vs aot", out_ref.shape)
        diffs = (out_aot - out_ref).abs()
        print(diffs.max())
        idx = int(diffs.flatten().argmax().item())
        row = idx // out_triton.shape[1]
        print(row, idx % out_triton.shape[1])
        # print(out_aot[row])
        # print(out_ref[row])
        print(out_aot.flatten()[idx], out_ref.flatten()[idx])
        print(diffs.mean())   


def test():
    idx = get_voxel_coords(10000, device='cuda')

    compare_conv3d_subm(idx, 16, 32, 3)
    compare_conv3d_subm(idx, 64, 64, 3)
    compare_conv3d_subm(idx, 128, 128, 3)
    compare_conv3d_subm(idx, 64, 128, 3)

test()