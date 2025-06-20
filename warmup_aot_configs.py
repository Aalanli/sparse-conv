import torch
import ops
import ops.idx_gen
import ops.conv3d_implicit_gemm

from utils import get_voxel_coords

def run(coords, dim_in, dim_out, kernel_size, dtype, acc_dtype):
    n = coords.shape[0]
    feats = torch.randn(n, dim_in, device="cuda", dtype=dtype)
    weights = torch.randn(kernel_size ** 3, dim_in, dim_out, device="cuda", dtype=dtype)
    indices = ops.idx_gen.gen_conv3d_subm_indices(coords, kernel_size)
    out = ops.conv3d_implicit_gemm.conv3d_implicit_gemm(feats, indices, weights, kernel_size, acc_dtype=acc_dtype)
    return out

coords = get_voxel_coords(800_000, device='cuda')
indices = ops.idx_gen.gen_conv3d_subm_indices(coords, kernel_size=3)


for N in [5000, 25_000, 50_000, 100_000, 200_000, 400_000]:
    for out_dtype in [torch.float32, torch.float16]:
        for acc_dtype in ['fp32', 'fp16']:
            for kernel_size in [3]:
                if acc_dtype == 'fp16' and out_dtype == torch.float32:
                    continue
                cn = coords[:N]
                run(cn, 16, 16, kernel_size, out_dtype, acc_dtype)
                run(cn, 16, 32, kernel_size, out_dtype, acc_dtype)
                run(cn, 32, 32, kernel_size, out_dtype, acc_dtype)
                run(cn, 32, 64, kernel_size, out_dtype, acc_dtype)
                run(cn, 64, 64, kernel_size, out_dtype, acc_dtype)
                run(cn, 64, 128, kernel_size, out_dtype, acc_dtype)
                run(cn, 128, 128, kernel_size, out_dtype, acc_dtype)
                run(cn, 128, 256, kernel_size, out_dtype, acc_dtype)
                run(cn, 256, 256, kernel_size, out_dtype, acc_dtype)
ops.conv3d_implicit_gemm.save_kernel_map()