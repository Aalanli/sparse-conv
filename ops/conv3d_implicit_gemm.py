# %%
from pathlib import Path
import torch
from torch import Tensor


def conv3d_implicit_gemm(input: Tensor, indices: Tensor, weight: Tensor, kernel_size: int, acc_dtype: str = "fp32") -> Tensor:
    return torch.ops.conv3d_implicit_gemm.conv3d_implicit_gemm_torch(
        input, indices, weight, kernel_size, acc_dtype
    )


def save_kernel_map():
    torch.ops.conv3d_implicit_gemm.save_kernel_map(str(Path(__file__).parent / 'kernel_map.json'))

