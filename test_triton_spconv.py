# %%
import ops
import glob
import os
import torch

import ops.conv3d_implicit_gemm as aot_implicit_gemm
import ops.idx_gen
from triton_spconv import conv3d_implicit_gemm

voxel_data = []
for p in glob.glob(os.path.expanduser('~/voxel_data/*.pt')):
    with open(p, 'rb') as f:
        voxel_data.append(torch.load(f))


def reference_conv3d_subm(feats: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor, kernel_size: int):
    n = feats.shape[0]
    feats = torch.nn.functional.pad(feats, (0, 0, 0, 1))
    indices = torch.where(indices < 0, n, indices)
    out = feats[indices.flatten()].reshape(-1, kernel_size**3 * feats.shape[1]) @ weights.view(-1, weights.shape[-1])
    return out

def compare_conv3d_subm(coords, dim_in, dim_out, kernel_size):
    n = coords.shape[0]
    feats = torch.randn(n, dim_in, device="cuda").half()
    weights = torch.randn(kernel_size ** 3, dim_in, dim_out, device="cuda").half()
    indices = ops.idx_gen.gen_conv3d_subm_indices(coords, kernel_size)

    out_triton = conv3d_implicit_gemm(feats, indices, weights, kernel_size)
    out_ref = reference_conv3d_subm(feats, indices, weights, kernel_size)

    # print((out_triton - out_ref).abs().max())
    if not torch.allclose(out_triton, out_ref, atol=1e-1, rtol=1e-3):
        print("Outputs do not match!")
        print(out_triton)
        print(out_ref)

    out_aot = aot_implicit_gemm.conv3d_implicit_gemm(
        feats, indices, weights, kernel_size, 0
    )

    print((out_aot - out_ref).abs().max())
    print(out_aot)
    print(torch.allclose(out_ref, out_aot, atol=1e-1, rtol=1e-3))
    


compare_conv3d_subm(voxel_data[0], 16, 32, 3)
compare_conv3d_subm(voxel_data[0], 64, 64, 3)
compare_conv3d_subm(voxel_data[1], 128, 128, 3)
compare_conv3d_subm(voxel_data[2], 64, 128, 3)


