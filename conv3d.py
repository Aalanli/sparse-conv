# Copyright (c) 2021 Waabi Innovation. All rights reserved.
from typing import List, Optional

import torch
from torch import Tensor

from ops import conv3d_implicit_gemm, idx_gen


# don't use torchsparse here, we want to avoid the dependency
class SparseTensor:
    def __init__(self, features: Tensor, coords: Tensor, spatial_range: List[int]):
        assert features.ndim == 2, "Features should be a 2D tensor (N, C)"
        assert coords.ndim == 2, "Coords should be a 2D tensor (N, 4)"
        assert coords.shape[1] == 4, "Coords should have 4 columns (batch, x, y, z)"
        assert features.shape[0] == coords.shape[0], "Features and coords must have the same number of elements"
        assert coords.dtype == torch.int32, "Coords should be of type int32"
        self.features = features
        self.coords = coords
        self.spatial_range = spatial_range

    def dense(self):
        D = self.features.shape[1]
        buf = torch.zeros(
            (self.spatial_range[0], self.spatial_range[1], self.spatial_range[2], self.spatial_range[3], D),
            dtype=self.features.dtype,
            device=self.features.device,
        )
        lin = self.coords[:, 3] + self.spatial_range[3] * (
            self.coords[:, 2] + self.spatial_range[2] * (self.coords[:, 1] + self.spatial_range[1] * self.coords[:, 0])
        )
        buf.view(-1, D).index_add_(0, lin, self.features)
        return buf


# the design of this module is such that it can be compiled with tensorRT.
# during AOT compilation, we lift all the indices outside the compiled module


class SparseSubmConv3d(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bias: bool = False, acc_dtype="fp32"):
        super().__init__()
        weight = torch.nn.Linear(kernel_size**3 * in_channels, out_channels, bias)
        self.weight_master = torch.nn.Parameter(weight.weight.view(-1, in_channels, out_channels).contiguous())
        self.register_buffer("weight", self.weight_master.half())
        self.bias = weight.bias if bias else None
        self.kernel_size = kernel_size
        self.acc_dtype = acc_dtype
        self.need_copy = False

    def forward(self, x: SparseTensor, indices: Optional[torch.Tensor] = None):
        # x: (N, C)
        # indices: (N)
        if indices is None:
            indices = idx_gen.gen_conv3d_subm_indices(x.coords, self.kernel_size)
        if self.training:
            self.need_copy = True
            h = self.weight_master.half()
            f = conv3d_implicit_gemm.conv3d_implicit_gemm(
                x.features.half(), indices, h, self.kernel_size, acc_dtype=self.acc_dtype
            )
        else:
            if self.need_copy:
                self.weight.copy_(self.weight_master.half())
                self.need_copy = False
            f = conv3d_implicit_gemm.conv3d_implicit_gemm(
                x.features.half(), indices, self.weight, self.kernel_size, acc_dtype=self.acc_dtype
            )
        if self.bias is not None:
            f = f + self.bias[None, :]
        return SparseTensor(features=f, coords=x.coords, spatial_range=x.spatial_range)


class SparseConv3d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int | List[int] = 1,
        padding: int | List[int] = 1,
        bias: bool = False,
        acc_dtype: str = "fp32",
    ):
        super().__init__()
        weight = torch.nn.Linear(kernel_size**3 * in_channels, out_channels, bias)
        self.weight_master = torch.nn.Parameter(weight.weight.view(-1, in_channels, out_channels).contiguous())
        self.register_buffer("weight", self.weight_master.half())
        self.bias = weight.bias if bias else None
        self.kernel_size = kernel_size
        self.stride = [stride] * 3 if isinstance(stride, int) else stride
        self.pad = [padding] * 3 if isinstance(padding, int) else padding
        assert len(self.stride) == 3, "Stride must be a list of three integers."
        assert len(self.pad) == 3, "Padding must be a list of three integers."
        self.dim = in_channels
        self.out_channels = out_channels
        self.acc_dtype = acc_dtype
        self.need_copy = False

    def calc_res_range(self, i: int, p: int, s: int):
        return (i + p * 2 - self.kernel_size) // s + 1

    def forward(self, x: SparseTensor, indices: Optional[torch.Tensor] = None):
        # x: (N, C)
        # indices: (N)
        if indices is None:
            new_coords, indices = idx_gen.gen_conv3d_indices(
                x.coords, x.spatial_range, self.kernel_size, self.stride, self.pad
            )
        else:
            new_coords = x.coords  # if indices are provided, we assume coords are already aligned

        if self.training:
            self.need_copy = True
            f = conv3d_implicit_gemm.conv3d_implicit_gemm(
                x.features.half(), indices, self.weight_master.half(), self.kernel_size, acc_dtype=self.acc_dtype
            )
        else:
            if self.need_copy:
                self.weight.copy_(self.weight_master.half())
                self.need_copy = False
            f = conv3d_implicit_gemm.conv3d_implicit_gemm(
                x.features.half(), indices, self.weight, self.kernel_size, acc_dtype=self.acc_dtype
            )
        if self.bias is not None:
            f = f + self.bias[None, :]
        return SparseTensor(
            features=f,
            coords=new_coords,
            spatial_range=[
                x.spatial_range[0],  # don't change batch size
                self.calc_res_range(x.spatial_range[1], self.pad[0], self.stride[0]),
                self.calc_res_range(x.spatial_range[2], self.pad[1], self.stride[1]),
                self.calc_res_range(x.spatial_range[3], self.pad[2], self.stride[2]),
            ],
        )
