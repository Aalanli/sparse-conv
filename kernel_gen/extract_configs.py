# %%
from tqdm import tqdm
import itertools
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import ops
import torch
import copy

import ops.idx_gen
import triton
import triton.language as tl
from kernel_gen.triton_implicit_gemm import conv3d_implicit_gemm, conv3d_implicit_gemm_T, implicit_gemm_grad
from kernel_gen.implicit_gemm_kernel import implicit_conv3d_kernel, implicit_conv3d_kernel_T, implicit_gemm_dF_kernel, implicit_gemm_dW_kernel, implicit_gemm_idx_sort_kernel, implicit_gemm_mask_kernel
from kernel_gen.utils import get_voxel_coords
import json
import argparse


def get_cache(kernel: triton.runtime.Autotuner, constexprs=None):
    configs = []
    for c, t in kernel.configs_timings.items():
        config_key = {}
        config_key['constexprs'] = copy.copy(c.kwargs)
        config_key['num_warps'] = c.num_warps
        config_key['num_stages'] = c.num_stages
        if constexprs is not None:
            config_key['constexprs'].update(constexprs)
        configs.append({
            "config": config_key,
            "time": t, # a list of 3 floats [0.5, 0.2, 0.8] quantiles
        })
    kernel.cache.clear()
    kernel.configs_timings.clear()
    return configs

def print_cache(N, dim_in, dim_out, kernel_size, dtype, weight_dtype, acc_dtype, BLOCK_N):
    coords = get_voxel_coords(N, device='cuda')
    n = coords.shape[0]
    feats = torch.randn(n, dim_in, device="cuda", dtype=dtype)
    weights = torch.randn(kernel_size ** 3, dim_in, dim_out, device="cuda", dtype=weight_dtype) / dim_in**0.5
    # indices = ops.idx_gen.gen_conv3d_subm_indices(coords, kernel_size)
    K3 = kernel_size ** 3

    # out = conv3d_implicit_gemm(feats, indices, weights, kernel_size, acc_dtype=acc_dtype)
    indices_T = ops.idx_gen.gen_conv3d_subm_indices_v2(coords, kernel_size)
    out = conv3d_implicit_gemm_T(feats, indices_T, weights, kernel_size, acc_dtype=acc_dtype, BLOCK_N=BLOCK_N, sort=True)
    implicit_gemm_grad(out, feats, indices_T, weights, acc_dtype)
    normalize_dtypes = {
        torch.float32: tl.float32,
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        tl.float32: tl.float32,
        tl.float16: tl.float16,
        tl.bfloat16: tl.bfloat16,
    }
    to_dtype_str = lambda x: str(normalize_dtypes[x])
    config_key = {
        "N": N,
        "dim_in": dim_in,
        "dim_out": dim_out,
        "kernel_size": kernel_size,
        "dtype": to_dtype_str(dtype),
        "weight_dtype": to_dtype_str(weight_dtype),
        "acc_dtype": to_dtype_str(acc_dtype),
        "BLOCK_N": BLOCK_N
    }

    config = {}
    config['key'] = config_key
    constexprs = {'acc_dtype': to_dtype_str(acc_dtype)}
    
    config['kernels'] = dict(
        implicit_conv3d_kernel_T=get_cache(implicit_conv3d_kernel_T, constexprs=dict(BLOCK_N=BLOCK_N, **constexprs)),
        implicit_gemm_dF_kernel=get_cache(implicit_gemm_dF_kernel, constexprs=constexprs),
        implicit_gemm_dW_kernel=get_cache(implicit_gemm_dW_kernel, constexprs=constexprs),
        implicit_gemm_idx_sort_kernel=get_cache(implicit_gemm_idx_sort_kernel, constexprs={'BLOCK_K': triton.next_power_of_2(K3)}),
        implicit_gemm_mask_kernel=get_cache(implicit_gemm_mask_kernel, constexprs={'BLOCK_N': BLOCK_N}),
    )
    return config

def get_sm():
    if torch.cuda.is_available():
        sm = torch.cuda.get_device_capability()
        return sm[0] * 10 + sm[1]
    else:
        raise RuntimeError("CUDA is not available. Cannot determine SM version.")

def extract_full_configs(key_vals: list):
    Ns = [1000, 10000, 100000, 600000]
    dtypes = [
            (torch.float32, tl.float32, torch.float32),
            (torch.float16, tl.float32, torch.float16),
            (torch.float16, tl.float32, torch.float32),
        ]
    BLOCK_Ns = [32]
    dims = [
        (16, 16), (32, 32), (64, 64), (64, 128), (128, 128), (256, 256)
    ]
    kernel_sizes = [3]
    params = list(itertools.product(Ns, dtypes, BLOCK_Ns, dims, kernel_sizes))
    for N, (out_dtype, weight_dtype, acc_dtype), BLOCK_N, (dim_in, dim_out), K in tqdm(params, desc="Extracting configs"):
        key_vals.append(print_cache(N, dim_in, dim_out, K, out_dtype, weight_dtype, acc_dtype, BLOCK_N))

def extract_test_configs(key_vals: list):
    key_vals.append(
        print_cache(100000, 128, 128, 3, torch.float16, torch.float32, tl.float32, 32)
    )
    key_vals.append(
        print_cache(400000, 32, 32, 3, torch.float16, torch.float32, tl.float32, 32)
    )
    key_vals.append(
        print_cache(400000, 16, 16, 3, torch.float16, torch.float32, tl.float32, 32)
    )


if __name__ == "__main__":
    sm = get_sm()
    key_vals = []

    extract_test_configs(key_vals)

    parser = argparse.ArgumentParser(description="Dump extracted configs to a file.")
    parser.add_argument("--out", type=str, required=True, help="Output JSON file name")
    args = parser.parse_args()
    
    for kv in key_vals:
        kv['sm'] = sm

    with open(args.out, "w") as f:
        json.dump(
            key_vals,
            f,
            indent=2
        )

