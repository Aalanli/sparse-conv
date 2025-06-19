# %%
import json
import os

import triton
import triton.backends
import triton.language as tl
from collections import namedtuple
import argparse

from pathlib import Path

def generate_ptx(
    mod: triton.JITFunction,
    signature: dict[str, str],
    constexprs: dict[str, int],
    divisibility: dict[int, int],
    warps: int,
    stages: int,
    sm_version: int,
    output_name: str = 'kernel.ptx',
    ptx_version: int | None = None
):
    attrs = {(i,): [['tt.divisibility', j]] for i, j in divisibility.items()}
    src = triton.compiler.ASTSource(mod, signature=signature, constexprs=constexprs, attrs=attrs)
    target = triton.backends.compiler.GPUTarget(backend='cuda', arch=sm_version, warp_size=32)
    opts = {'num_warps': warps, 'num_stages': stages}
    if ptx_version is not None:
        opts['ptx_version'] = ptx_version
    ccker = triton.compile(src, target, options=opts)
    with open(output_name, 'w') as f:
        print("Compiled Triton kernel")
        print(ccker.metadata)

        f.write(ccker.asm['ptx'])
    return ccker
    


# {'num_warps': 2, 'num_ctas': 1, 'num_stages': 2, 'num_buffers_warp_spec': 0, 'num_consumer_groups': 0, 'reg_dec_producer': 0, 'reg_inc_consumer': 0, 'maxnreg': None, 'cluster_dims': (1, 1, 1), 'ptx_version': None, 'enable_fp_fusion': True, 'launch_cooperative_grid': False, 'supported_fp8_dtypes': ('fp8e4b15', 'fp8e4nv', 'fp8e5'), 'deprecated_fp8_dtypes': (), 'default_dot_input_precision': 'tf32', 'allowed_dot_input_precisions': ('tf32', 'tf32x3', 'ieee'), 'max_num_imprecise_acc_default': 0, 'extern_libs': ((...),), 'debug': False, 'backend_name': 'cuda', 'sanitize_overflow': True, 'arch': 'sm89'}

def get_sig(dtype: str):
    sig = {
        'features': f'*{dtype}',
        'indices': f'*i32',
        'weights': f'*{dtype}',
        'output': f'*{dtype}',
        'N': 'i32',
        'N_prime': 'i32',
        'D': 'i32',
        'D_prime': 'i32',
        'K': 'i32',
        'BLOCK_N': 'constexpr',
        'BLOCK_K': 'constexpr',
        'BLOCK_Dp': 'constexpr',
        'acc_dtype': 'constexpr',
        'PARALLEL_K': 'constexpr',
    }
    return sig

def get_divisibility(div_k: bool, div_d: bool, div_dp: bool):
    divisibility = {}
    if div_d:
        divisibility[0] = 16 # features
        divisibility[6] = 16 # D
    if div_dp:
        divisibility[2] = 16 # weights
        divisibility[7] = 16 # D_prime
    if div_k:
        divisibility[1] = 16 # indices
        divisibility[8] = 16 # K
    return divisibility

Config = namedtuple('Config', ['BLOCK_N', 'BLOCK_K', 'BLOCK_Dp', 'num_warps', 'num_stages', 'parallel_k', 'div_k', 'div_d', 'div_dp', 'acc_dtype', 'dtype', 'sm'])


def load_configs(div_k: bool = False, div_d: bool = True, div_dp: bool = True, pk: int = 1):
    params = set()
    for file in Path(__file__).parent.glob('tuning_configs/*.json'):
        with open(file, 'r') as f:
            data = json.load(f)
            for kv in data:
                params.add(
                    Config(
                        BLOCK_N=kv['BLOCK_N'],
                        BLOCK_K=kv['BLOCK_K'],
                        BLOCK_Dp=kv['BLOCK_Dp'],
                        num_warps=kv['num_warps'],
                        num_stages=kv['num_stages'],
                        parallel_k=pk,
                        div_k=div_k,
                        div_d=div_d,
                        div_dp=div_dp,
                        acc_dtype=kv['acc_dtype'],
                        dtype=kv['dtype'],
                        sm=kv["sm"]
                    )
                )
    return params


from implicit_gemm_kernel import implicit_conv3d_kernel

def generate_ptx_from_config(config: Config, output_name: str = 'kernel.ptx', ptx_version: int | None = None):
    signature = get_sig(config.dtype)
    dtype_to_triton = {
        'fp16': tl.float16,
        'fp32': tl.float32,
    }
    constexprs = {
        'BLOCK_N': config.BLOCK_N,
        'BLOCK_K': config.BLOCK_K,
        'BLOCK_Dp': config.BLOCK_Dp,
        'acc_dtype': dtype_to_triton[config.dtype],
        'PARALLEL_K': config.parallel_k,
    }
    divisibility = get_divisibility(config.div_k, config.div_d, config.div_dp)

    return generate_ptx(
        implicit_conv3d_kernel.fn,
        signature=signature,
        constexprs=constexprs,
        divisibility=divisibility,
        warps=config.num_warps,
        stages=config.num_stages,
        sm_version=config.sm,
        output_name=output_name,
        ptx_version=ptx_version
    )



def generate_ptx_from_configs(configs: list[Config], output_path: str = 'kernel_ptx', ptx_version: int | None = None):
    os.makedirs(output_path, exist_ok=True)
    meta = {}
    for i, config in enumerate(configs):
        output_name = f"{config}.ptx"
        ccinfo = generate_ptx_from_config(config, output_name=os.path.join(output_path, output_name), ptx_version=ptx_version)
        ker_meta = ccinfo.metadata._asdict()
        ker_meta['target'] = None # ker_meta['target'].asdict()
        meta[output_name] = {
            'config': config._asdict(),
            'meta': ker_meta,
        }
    
    with open(os.path.join(output_path, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)
    print(f"Generated PTX files and metadata in {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate PTX files for implicit_conv3d_kernel with various configs.")
    parser.add_argument('--output', type=str, default='kernel_ptx', help='Output directory for PTX files')
    parser.add_argument('--ptx_version', type=int, default=None, help='PTX version to use (optional)')
    args = parser.parse_args()

    configs = list()
    configs.extend(load_configs(div_k=False, div_d=True,  div_dp=True , pk=1)) # fully divisible
    configs.extend(load_configs(div_k=False, div_d=False, div_dp=False, pk=1)) # not divisible
    configs.extend(load_configs(div_k=False, div_d=True,  div_dp=True , pk=2)) # fully divisible
    configs.extend(load_configs(div_k=False, div_d=False, div_dp=False, pk=2)) # not divisible
    
    generate_ptx_from_configs(configs, output_path=args.output, ptx_version=args.ptx_version)

