
# %%
import ops
import torch

import ops.idx_gen
import triton.language as tl
from triton_spconv import conv3d_implicit_gemm
from implicit_gemm_kernel import implicit_conv3d_kernel
from utils import get_voxel_coords

from collections import namedtuple

ConfigKey = namedtuple("ConfigKey", ["N", "D", "D_prime", "K", "dtype", "acc_dtype"])

def print_cache(N, dim_in, dim_out, kernel_size, dtype, acc_dtype):

    coords = get_voxel_coords(N, device='cuda')
    n = coords.shape[0]
    feats = torch.randn(n, dim_in, device="cuda", dtype=dtype)
    weights = torch.randn(kernel_size ** 3, dim_in, dim_out, device="cuda", dtype=dtype) / dim_in**0.5
    indices = ops.idx_gen.gen_conv3d_subm_indices(coords, kernel_size)

    out = conv3d_implicit_gemm(feats, indices, weights, kernel_size, acc_dtype=acc_dtype)


    for k, v in implicit_conv3d_kernel.cache.items():
        config_key = ConfigKey(
            N=N,
            D=dim_in,
            D_prime=dim_out,
            K=kernel_size**3,
            dtype=str(dtype),
            acc_dtype=str(acc_dtype)
        )
        config_val = v.kwargs
        config_val['num_warps'] = v.num_warps
        config_val['num_stages'] = v.num_stages
        implicit_conv3d_kernel.cache.clear()
        return config_key, config_val
    
key_vals = []

for N in [5000, 10_000, 20_000, 50_000, 100_000, 200_000, 400_000]:
    for acc_dtype in [tl.float32, tl.float16]:
        key_vals.append(print_cache(N, 16, 16, 3, torch.float16, acc_dtype))
        key_vals.append(print_cache(N, 16, 32, 3, torch.float16, acc_dtype))
        key_vals.append(print_cache(N, 32, 32, 3, torch.float16, acc_dtype))
        key_vals.append(print_cache(N, 32, 64, 3, torch.float16, acc_dtype))
        key_vals.append(print_cache(N, 64, 64, 3, torch.float16, acc_dtype))
        key_vals.append(print_cache(N, 64, 128, 3, torch.float16, acc_dtype))
        key_vals.append(print_cache(N, 128, 128, 3, torch.float16, acc_dtype))
        key_vals.append(print_cache(N, 128, 256, 3, torch.float16, acc_dtype))
        key_vals.append(print_cache(N, 256, 256, 3, torch.float16, acc_dtype))

import json
import argparse

parser = argparse.ArgumentParser(description="Dump extracted configs to a file.")
parser.add_argument("--out", type=str, required=True, help="Output JSON file name")
args = parser.parse_args()

with open(args.out, "w") as f:
    json.dump(
        [{"key": k._asdict(), "val": v} for k, v in key_vals],
        f,
        indent=2
    )

