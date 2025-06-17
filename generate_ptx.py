# %%
import json
import os

import triton
import triton.backends
import triton.language as tl
from collections import namedtuple


def generate_ptx(
    mod: triton.JITFunction,
    signature: dict[str, str],
    constexprs: dict[str, int],
    divisibility: dict[int, int],
    warps: int,
    stages: int,
    sm_version: int,
    output_name: str = 'kernel.ptx'
):
    attrs = {(i,): [['tt.divisibility', j]] for i, j in divisibility.items()}
    src = triton.compiler.ASTSource(mod, signature=signature, constexprs=constexprs, attrs=attrs)
    target = triton.backends.compiler.GPUTarget(backend='cuda', arch=sm_version, warp_size=32)
    opts = {'num_warps': warps, 'num_stages': stages}
    ccker = triton.compile(src, target, options=opts)
    with open(output_name, 'w') as f:
        print("Compiled Triton kernel")
        print(ccker.metadata)

        f.write(ccker.asm['ptx'])
    return ccker
    
from triton_spconv import implicit_conv3d_kernel

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

Config = namedtuple('Config', ['block_sizes', 'num_warps', 'num_stages', 'div_k', 'div_d', 'div_dp', 'dtype', 'sm'])

sm = 89

configs = [
    Config({"BLOCK_N": 32, "BLOCK_K": 16, "BLOCK_Dp": 16}, num_warps=2, num_stages=2, div_k=False, div_d=True,  div_dp=True, dtype='fp16', sm=sm),
    Config({"BLOCK_N": 16, "BLOCK_K": 16, "BLOCK_Dp": 16}, num_warps=2, num_stages=2, div_k=False, div_d=True,  div_dp=True, dtype='fp16', sm=sm),
    Config({"BLOCK_N": 16, "BLOCK_K": 32, "BLOCK_Dp": 64}, num_warps=4, num_stages=3, div_k=False, div_d=True,  div_dp=True, dtype='fp16', sm=sm),
    Config({"BLOCK_N": 32, "BLOCK_K": 32, "BLOCK_Dp": 16}, num_warps=2, num_stages=2, div_k=False, div_d=True,  div_dp=True, dtype='fp16', sm=sm),
    Config({"BLOCK_N": 16, "BLOCK_K": 32, "BLOCK_Dp": 128},num_warps=8, num_stages=3, div_k=False, div_d=True,  div_dp=True, dtype='fp16', sm=sm),
    Config({"BLOCK_N": 128, "BLOCK_K": 32, "BLOCK_Dp": 32},num_warps=4, num_stages=2, div_k=False, div_d=True,  div_dp=True, dtype='fp16', sm=sm),
    Config({"BLOCK_N": 16, "BLOCK_K": 64, "BLOCK_Dp": 64}, num_warps=4, num_stages=3, div_k=False, div_d=True,  div_dp=True, dtype='fp16', sm=sm),
    Config({"BLOCK_N": 16, "BLOCK_K": 32, "BLOCK_Dp": 32}, num_warps=2, num_stages=2, div_k=False, div_d=True,  div_dp=True, dtype='fp16', sm=sm),
    Config({"BLOCK_N": 32, "BLOCK_K": 16, "BLOCK_Dp": 32}, num_warps=2, num_stages=2, div_k=False, div_d=True,  div_dp=True, dtype='fp16', sm=sm),
    Config({"BLOCK_N": 32, "BLOCK_K": 16, "BLOCK_Dp": 16}, num_warps=2, num_stages=2, div_k=False, div_d=False, div_dp=True, dtype='fp16', sm=sm),
    Config({"BLOCK_N": 16, "BLOCK_K": 16, "BLOCK_Dp": 16}, num_warps=2, num_stages=2, div_k=False, div_d=False, div_dp=True, dtype='fp16', sm=sm),
    Config({"BLOCK_N": 16, "BLOCK_K": 32, "BLOCK_Dp": 64}, num_warps=4, num_stages=3, div_k=False, div_d=False, div_dp=True, dtype='fp16', sm=sm),
    Config({"BLOCK_N": 32, "BLOCK_K": 32, "BLOCK_Dp": 16}, num_warps=2, num_stages=2, div_k=False, div_d=False, div_dp=True, dtype='fp16', sm=sm),
    Config({"BLOCK_N": 16, "BLOCK_K": 32, "BLOCK_Dp": 128},num_warps=8, num_stages=3, div_k=False, div_d=False, div_dp=True, dtype='fp16', sm=sm),
    Config({"BLOCK_N": 128, "BLOCK_K": 32, "BLOCK_Dp": 32},num_warps=4, num_stages=2, div_k=False, div_d=False, div_dp=True, dtype='fp16', sm=sm),
    Config({"BLOCK_N": 16, "BLOCK_K": 64, "BLOCK_Dp": 64}, num_warps=4, num_stages=3, div_k=False, div_d=False, div_dp=True, dtype='fp16', sm=sm),
    Config({"BLOCK_N": 16, "BLOCK_K": 32, "BLOCK_Dp": 32}, num_warps=2, num_stages=2, div_k=False, div_d=False, div_dp=True, dtype='fp16', sm=sm),
    Config({"BLOCK_N": 32, "BLOCK_K": 16, "BLOCK_Dp": 32}, num_warps=2, num_stages=2, div_k=False, div_d=False, div_dp=True, dtype='fp16', sm=sm),
    Config({"BLOCK_N": 32, "BLOCK_K": 16, "BLOCK_Dp": 16}, num_warps=2, num_stages=2, div_k=False, div_d=False, div_dp=False, dtype='fp16', sm=sm),
    Config({"BLOCK_N": 16, "BLOCK_K": 16, "BLOCK_Dp": 16}, num_warps=2, num_stages=2, div_k=False, div_d=False, div_dp=False, dtype='fp16', sm=sm),
    Config({"BLOCK_N": 16, "BLOCK_K": 32, "BLOCK_Dp": 64}, num_warps=4, num_stages=3, div_k=False, div_d=False, div_dp=False, dtype='fp16', sm=sm),
    Config({"BLOCK_N": 32, "BLOCK_K": 32, "BLOCK_Dp": 16}, num_warps=2, num_stages=2, div_k=False, div_d=False, div_dp=False, dtype='fp16', sm=sm),
    Config({"BLOCK_N": 16, "BLOCK_K": 32, "BLOCK_Dp": 128},num_warps=8, num_stages=3, div_k=False, div_d=False, div_dp=False, dtype='fp16', sm=sm),
    Config({"BLOCK_N": 128, "BLOCK_K": 32, "BLOCK_Dp": 32},num_warps=4, num_stages=2, div_k=False, div_d=False, div_dp=False, dtype='fp16', sm=sm),
    Config({"BLOCK_N": 16, "BLOCK_K": 64, "BLOCK_Dp": 64}, num_warps=4, num_stages=3, div_k=False, div_d=False, div_dp=False, dtype='fp16', sm=sm),
    Config({"BLOCK_N": 16, "BLOCK_K": 32, "BLOCK_Dp": 32}, num_warps=2, num_stages=2, div_k=False, div_d=False, div_dp=False, dtype='fp16', sm=sm),
    Config({"BLOCK_N": 32, "BLOCK_K": 16, "BLOCK_Dp": 32}, num_warps=2, num_stages=2, div_k=False, div_d=False, div_dp=False, dtype='fp16', sm=sm),
]

def generate_ptx_from_config(config: Config, output_name: str = 'kernel.ptx'):
    signature = get_sig(config.dtype)
    dtype_to_triton = {
        'fp16': tl.float16,
        'fp32': tl.float32,
    }
    constexprs = {
        'BLOCK_N': config.block_sizes['BLOCK_N'],
        'BLOCK_K': config.block_sizes['BLOCK_K'],
        'BLOCK_Dp': config.block_sizes['BLOCK_Dp'],
        'acc_dtype': dtype_to_triton[config.dtype],
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
        output_name=output_name
    )



def generate_ptx_from_configs(configs: list[Config], output_path: str = 'kernel_ptx'):
    os.makedirs(output_path, exist_ok=True)
    meta = {}
    for i, config in enumerate(configs):
        output_name = f"{config}.ptx"
        ccinfo = generate_ptx_from_config(config, output_name=os.path.join(output_path, output_name))
        import triton.tools.compile
        ker_meta = ccinfo.metadata._asdict()
        ker_meta['target'] = None # ker_meta['target'].asdict()
        meta[output_name] = {
            'config': config._asdict(),
            'meta': ker_meta,
        }
    
    with open(os.path.join(output_path, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)
    print(f"Generated PTX files and metadata in {output_path}")

    
generate_ptx_from_configs(configs, output_path='kernel_ptx')

