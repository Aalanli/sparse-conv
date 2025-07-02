# %%
import sys
import os
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)


import torch
import ops
import ops.idx_gen
import ops.conv3d_implicit_gemm


from utils import get_voxel_coords

def run(coords, N, dim_in, dim_out, kernel_size, dtype, weight_dtype, acc_dtype):
    feats = torch.randn(N, dim_in, device="cuda", dtype=dtype, requires_grad=True)
    weights = torch.randn(kernel_size ** 3, dim_in, dim_out, device="cuda", dtype=weight_dtype, requires_grad=True)
    indices = ops.idx_gen.gen_conv3d_subm_indices(coords, kernel_size)
    out = ops.conv3d_implicit_gemm.conv3d_implicit_gemm(feats, indices, weights, kernel_size, acc_dtype=acc_dtype)
    out.sum().backward()
    return out


def warmup_full():
    coords = get_voxel_coords(800_000, device="cuda")
    SEQS = [1000, 10000, 100_000, 600_000]
    # run(coords[:300], 300, 16, 16, 3, torch.float32, "fp32")

    configs = (
        (out_dtype, weight_dtype, acc_dtype, N, N_PRIME, kernel_size)
        for out_dtype in [torch.float32, torch.float16]
        for acc_dtype in ['fp32']
        for weight_dtype in [torch.float16, torch.float32]
        for N in SEQS
        for N_PRIME in SEQS
        for kernel_size in [3]
        if not (acc_dtype == 'fp16' and out_dtype == torch.float32) and not (weight_dtype == torch.float16 and out_dtype == torch.float32)
    )

    for out_dtype, weight_dtype, acc_dtype, N, N_PRIME, kernel_size in tqdm(list(configs), desc="Warmup configs"):
        cn = coords[:N]
        run(cn, N_PRIME, 16, 16, kernel_size, out_dtype, weight_dtype, acc_dtype)
        run(cn, N_PRIME, 16, 32, kernel_size, out_dtype, weight_dtype, acc_dtype)
        run(cn, N_PRIME, 32, 32, kernel_size, out_dtype, weight_dtype, acc_dtype)
        run(cn, N_PRIME, 32, 64, kernel_size, out_dtype, weight_dtype, acc_dtype)
        run(cn, N_PRIME, 64, 64, kernel_size, out_dtype, weight_dtype, acc_dtype)
        run(cn, N_PRIME, 64, 128, kernel_size, out_dtype, weight_dtype, acc_dtype)
        run(cn, N_PRIME, 128, 128, kernel_size, out_dtype, weight_dtype, acc_dtype)
        run(cn, N_PRIME, 128, 256, kernel_size, out_dtype, weight_dtype, acc_dtype)
        run(cn, N_PRIME, 256, 256, kernel_size, out_dtype, weight_dtype, acc_dtype)
    ops.conv3d_implicit_gemm.save_kernel_map()

def warmup_toy():
    coords = get_voxel_coords(800_000, device="cuda")
    run(coords[:300], 300, 16, 16, 3, torch.float32, torch.float32, "fp32")
    ops.conv3d_implicit_gemm.save_kernel_map()

warmup_full()