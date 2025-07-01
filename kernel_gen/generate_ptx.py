# %%
from tqdm import tqdm
import json
import os
from collections import defaultdict

import triton
from triton import JITFunction
import triton.backends
import triton.language as tl
from collections import namedtuple
import argparse

from pathlib import Path


def generate_ptx(
    mod: triton.JITFunction,
    signature: dict[str, str],
    constexprs: dict[str, int | str],
    divisibility: dict[int, int],
    warps: int,
    stages: int,
    sm_version: int,
    ptx_version: int | None = None
):
    constexpr_map = {
        "fp32": tl.float32,
        "fp16": tl.float16,
        "i32": tl.int32,
        "i64": tl.int64,
        "i1": tl.int1
    }
    constexprs = {k: constexpr_map[v] if isinstance(v, str) else v for k, v in constexprs.items()}
    attrs = {(i,): [['tt.divisibility', j]] for i, j in divisibility.items()}
    src = triton.compiler.ASTSource(mod, signature=signature, constexprs=constexprs, attrs=attrs)
    target = triton.backends.compiler.GPUTarget(backend='cuda', arch=sm_version, warp_size=32)
    opts = {'num_warps': warps, 'num_stages': stages}
    if ptx_version is not None:
        opts['ptx_version'] = ptx_version
    ccker = triton.compile(src, target, options=opts)
    return ccker
    

def get_sig(key: dict):
    dtype = key['dtype']
    K3 = key['kernel_size'] ** 3
    weight_dtype = key['weight_dtype']
    sig = {}
    sig['implicit_conv3d_kernel_T'] = {
        'features': f'*{dtype}',
        'indices': f'*i32',
        'mask_ind': '*i1',
        'weights': f'*{weight_dtype}',
        'out_perm': f'*i32',
        'output': f'*{dtype}',
        'N': 'i32',
        'N_prime': 'i32',
        'N_prime_stride': 'i32',
        'D': 'i32',
        'D_prime': 'i32',
        'K': 'i32',
        'sorted': 'i1',
        'BLOCK_N': 'constexpr',
        'BLOCK_K': 'constexpr',
        'BLOCK_Dp': 'constexpr',
        'PARALLEL_K': 'constexpr',
        'acc_dtype': 'constexpr'
    }
    sig['implicit_gemm_idx_sort_kernel'] = {
        'indices': f'*i32',
        'line_mask': f'*i32' if K3 <= 32 else '*i64',
        'N': 'i32',
        'N_stride': 'i32',
        'K3': 'i32',
        'BLOCK_K': 'constexpr',
        'BLOCK_N': 'constexpr',
    }
    sig['implicit_gemm_mask_kernel'] = {
        'indices': f'*i32',
        'mask': '*i1',
        'N': 'i32',
        'N_stride': 'i32',
        'K3': 'i32',
        'BLOCK_N': 'constexpr',
        'BLOCK_K': 'constexpr',
    }
    sig['implicit_gemm_dF_kernel'] = {
        'dout': f'*{dtype}',
        'weights': f'*{weight_dtype}',
        'indices': f'*i32',
        'dfeatures': f'*{dtype}',
        'N': 'i32',
        'N_prime': 'i32',
        'N_prime_stride': 'i32',
        'D': 'i32',
        'D_prime': 'i32',
        'BLOCK_DPrime': 'constexpr',
        'BLOCK_NPrime': 'constexpr',
        'BLOCK_D': 'constexpr',
        'acc_dtype': 'constexpr'
    }
    sig['implicit_gemm_dW_kernel'] = {
        'dout': f'*{dtype}',
        'features': f'*{dtype}',
        'indices': f'*i32',
        'dweight': f'*{weight_dtype}',
        'N': 'i32',
        'N_prime': 'i32',
        'N_prime_stride': 'i32',
        'D': 'i32',
        'D_prime': 'i32',
        'K3': 'i32',
        'BLOCK_NPrime': 'constexpr',
        'BLOCK_DPrime': 'constexpr',
        'BLOCK_D': 'constexpr',
        'PARALLEL_K': 'constexpr',
        'acc_dtype': 'constexpr'
    }
    return sig


def get_divisibility():
    divisibility = {}
    divisibility['implicit_conv3d_kernel_T'] = [
        ['features', 'indices', 'mask_ind', 'weights', 'out_perm', 'output', 'D', 'D_prime', 'N_prime_stride'],
        ['features', 'indices', 'mask_ind', 'weights', 'out_perm', 'output', 'D', 'D_prime'],
        []
    ]
    divisibility['implicit_gemm_idx_sort_kernel'] = [
        ['indices', 'line_mask'], []
    ]
    divisibility['implicit_gemm_mask_kernel'] = [
        ['indices', 'mask'], []
    ]
    divisibility['implicit_gemm_dF_kernel'] = [
        ['dout', 'weights', 'indices', 'dfeatures', 'D', 'D_prime'],
        []
    ]
    divisibility['implicit_gemm_dW_kernel'] = [
        ['dout', 'features', 'indices', 'dweight', 'D', 'D_prime'],
        []
    ]
    return divisibility

def get_divisibility_indices(sig: dict[str, str], div_args: list[str]):
    div = {}
    for i, key in enumerate(sig):
        if key in div_args:
            div[i] = 16
    return div

def generate_ptx_from_config(configs: list[dict], functions: dict[str, JITFunction], ptx_version: int | None = None):
    compiled_kernels = defaultdict(set) # ker_name -> set of constexprs
    ccinfo = defaultdict(list)
    t = tqdm(desc="generating kernels")
    for config in configs:
        # ker_name -> sig
        sm = config['sm']
        sigs = get_sig(config['key'])
        # ker_name -> divisible args
        divisibility = get_divisibility()
        div_indices = {}
        for ker_name, divs in divisibility.items():
            div_indices[ker_name] = [get_divisibility_indices(sigs[ker_name], div) for div in divs]

        # config["kernels"]: ker_name -> [{"config": {"contexprs": ..., "num_warps": ..., "num_stages": ...}, "time": [0.5, 0.2, 0.8] quantiles}]
        for ker_name, ker_configs in config['kernels'].items():
            min_idx = min(range(len(ker_configs)), key=lambda i: ker_configs[i]['time'][0])
            ker_config = ker_configs[min_idx]
            
            ker_config_str = str(ker_config['config']) + " sm: " + str(sm)
            if ker_config_str in compiled_kernels[ker_name]:
                continue
            compiled_kernels[ker_name].add(ker_config_str)
            for div in div_indices[ker_name]:
                constexprs = ker_config['config']['constexprs']
                ker = generate_ptx(
                    functions[ker_name],
                    signature=sigs[ker_name],
                    constexprs=constexprs,
                    divisibility=div,
                    warps=ker_config['config']['num_warps'],
                    stages=ker_config['config']['num_stages'],
                    sm_version=sm,
                    ptx_version=ptx_version
                )
                ker_ccinfo = ker.metadata._asdict()
                ker_meta = {}
                ker_meta['kernel_name'] = ker_name
                ker_meta['args'] = [
                    {'name': arg, 'dtype': sigs[ker_name][arg]} for arg in sigs[ker_name] if arg not in constexprs
                ]
                ker_meta['divisible_by_16'] = list(div.keys())
                ker_meta['sm'] = sm
                ker_meta['num_warps'] = ker_config['config']['num_warps']
                ker_meta['num_stages'] = ker_config['config']['num_stages']
                ker_meta['shared'] = ker_ccinfo['shared']
                ker_meta['global_scratch_size'] = ker_ccinfo['global_scratch_size']
                ker_meta.update(ker_config['config']['constexprs'])
                ker_meta['ptx'] = ker.asm['ptx']
                ccinfo[ker_name].append(ker_meta)
                t.update(1)
    t.close()
    return ccinfo
            
from implicit_gemm_kernel import implicit_conv3d_kernel_T, implicit_gemm_dF_kernel, implicit_gemm_dW_kernel, implicit_gemm_idx_sort_kernel, implicit_gemm_mask_kernel
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate PTX files for implicit_conv3d_kernel with various configs.")
    parser.add_argument('--ptx_version', type=int, default=82, help='PTX version to use (optional)')
    args = parser.parse_args()

    configs = []
    for file in Path(__file__).parent.glob("tuning_configs/*.json"):
        with open(file, 'r') as f:
            config = json.load(f)
            configs.extend(config)
    functions = {
        'implicit_conv3d_kernel_T': implicit_conv3d_kernel_T.fn,
        'implicit_gemm_dF_kernel': implicit_gemm_dF_kernel.fn,
        'implicit_gemm_dW_kernel': implicit_gemm_dW_kernel.fn,
        'implicit_gemm_idx_sort_kernel': implicit_gemm_idx_sort_kernel.fn,
        'implicit_gemm_mask_kernel': implicit_gemm_mask_kernel.fn,
    }
    ccinfo = generate_ptx_from_config(configs, functions, ptx_version=args.ptx_version)
    output_dir = Path(__file__).parent.parent / 'ops'
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'meta.json', 'w') as f:
        json.dump(ccinfo, f)
    with open(output_dir / 'kernel_map.json', 'w') as f:
        json.dump([], f, indent=1)