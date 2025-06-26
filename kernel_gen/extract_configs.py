
# %%
import ops
import torch

import ops.idx_gen
import triton.language as tl
from triton_spconv import conv3d_implicit_gemm
from implicit_gemm_kernel import implicit_conv3d_kernel
from utils import get_voxel_coords
import json
import argparse


def print_cache(N, dim_in, dim_out, kernel_size, dtype, acc_dtype):

    coords = get_voxel_coords(N, device='cuda')
    n = coords.shape[0]
    feats = torch.randn(n, dim_in, device="cuda", dtype=dtype)
    weights = torch.randn(kernel_size ** 3, dim_in, dim_out, device="cuda", dtype=dtype) / dim_in**0.5
    indices = ops.idx_gen.gen_conv3d_subm_indices(coords, kernel_size)

    out = conv3d_implicit_gemm(feats, indices, weights, kernel_size, acc_dtype=acc_dtype)

    torch_dtype_to_triton_dtype = {
        torch.float32: tl.float32,
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
    }

    for k, v in implicit_conv3d_kernel.cache.items():
        config_key = dict(
            N=N,
            D=dim_in,
            D_prime=dim_out,
            K=kernel_size**3,
            dtype=str(torch_dtype_to_triton_dtype[dtype]),
            acc_dtype=str(acc_dtype)
        )
        config_key.update(v.kwargs)
        config_key['num_warps'] = v.num_warps
        config_key['num_stages'] = v.num_stages
        implicit_conv3d_kernel.cache.clear()
        print(config_key)
        return config_key

def get_sm():
    if torch.cuda.is_available():
        sm = torch.cuda.get_device_capability()
        return sm[0] * 10 + sm[1]
    else:
        raise RuntimeError("CUDA is not available. Cannot determine SM version.")

if __name__ == "__main__":
    sm = get_sm()
    key_vals = []

    for N in [5000, 25_000, 50_000, 100_000, 200_000, 400_000]:
        for out_dtype in [torch.float32, torch.float16]:
            for acc_dtype in [tl.float32, tl.float16]:
                if out_dtype == torch.float32 and acc_dtype == tl.float16:
                    continue
                key_vals.append(print_cache(N, 16, 16, 3, out_dtype, acc_dtype))
                key_vals.append(print_cache(N, 16, 32, 3, out_dtype, acc_dtype))
                key_vals.append(print_cache(N, 32, 32, 3, out_dtype, acc_dtype))
                key_vals.append(print_cache(N, 32, 64, 3, out_dtype, acc_dtype))
                key_vals.append(print_cache(N, 64, 64, 3, out_dtype, acc_dtype))
                key_vals.append(print_cache(N, 64, 128, 3, out_dtype, acc_dtype))
                key_vals.append(print_cache(N, 128, 128, 3, out_dtype, acc_dtype))
                key_vals.append(print_cache(N, 128, 256, 3, out_dtype, acc_dtype))
                key_vals.append(print_cache(N, 256, 256, 3, out_dtype, acc_dtype))

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

