from typing import List

import torch
from torch import Tensor

def gen_conv3d_subm_indices(coords: Tensor, kernel_size: int = 3, hash_map_multiplier: float = 4.0, threads: int = 128, lookup_tries: int = 32) -> Tensor:
    return torch.ops.convIdx.generate_conv3d_subm_indices(coords, kernel_size, hash_map_multiplier, threads, lookup_tries)


def gen_conv3d_subm_indices_v2(
    coords: Tensor,
    kernel_size: int = 3,
    hash_map_multiplier: float = 8.0,
    threads: int = 128,
    lookup_tries: int = 32
) -> Tensor:
    """
    Generate indices for 3D sparse convolution with submanifold sampling.
    Returns new_coords and indices.
    """
    return torch.ops.convIdx.generate_conv3d_subm_indices_v2(
        coords,
        kernel_size,
        hash_map_multiplier,
        threads,
        lookup_tries
    )


def gen_conv3d_indices(
    coords: Tensor,
    spatial_range: List[int],
    kernel_size: int = 3,
    stride: int | List[int] = 1,
    padding: int | List[int] = 0,
    hash_map_multiplier: float = 8.0,
    threads: int = 256,
    lookup_tries: int = 128
) -> tuple[Tensor, Tensor]:
    """
    Generate indices for 3D sparse convolution.
    Returns new_coords and indices.

    Note: this function does not match torchsparse behavior for stride = 1 or 1 in stride. This is I think a bug in torchsparse
        For example, (simplified to 1D case):

        coords: [2]
        kernel_size: 3
        max_x: 5
        stride: 1
        padding: 0

        torchsparse generates indices: [1, 2, 3]
        but we generate indices:       [0, 1, 2]
        The second is correct, as it matches what we expect from dense convolution. Eg.
            y[0] = x[0] * w[0] + x[1] * w[1] + x[2] * w[2]
            y[1] = x[1] * w[0] + x[2] * w[1] + x[3] * w[2]
            y[2] = x[2] * w[0] + x[3] * w[1] + x[4] * w[2]
            with everything else being 0. Hence the only non-zero indices in the output are [0, 1, 2].
    """
    batch_size, max_x, max_y, max_z = spatial_range
    if isinstance(stride, int):
        stride = [stride] * 3
    if isinstance(padding, int):
        padding = [padding] * 3
    assert len(stride) == 3, "Stride must be a list of three integers."
    assert len(padding) == 3, "Padding must be a list of three integers."

    return torch.ops.convIdx.generate_conv3d_indices(
        coords,
        batch_size,
        kernel_size,
        stride[0],
        stride[1],
        stride[2],
        padding[0],
        padding[1],
        padding[2],
        max_x,
        max_y,
        max_z,
        hash_map_multiplier,
        threads,
        lookup_tries
    )


def gen_conv3d_indices_v2(
    coords: Tensor,
    spatial_range: List[int],
    kernel_size: int = 3,
    stride: int | List[int] = 1,
    padding: int | List[int] = 0,
    hash_map_multiplier: float = 2.0,
    threads: int = 256,
    lookup_tries: int = 128
) -> tuple[Tensor, Tensor]:
    """
    Generate indices for 3D sparse convolution.
    Returns new_coords and indices.

    Note: this function does not match torchsparse behavior for stride = 1 or 1 in stride. This is I think a bug in torchsparse
        For example, (simplified to 1D case):

        coords: [2]
        kernel_size: 3
        max_x: 5
        stride: 1
        padding: 0

        torchsparse generates indices: [1, 2, 3]
        but we generate indices:       [0, 1, 2]
        The second is correct, as it matches what we expect from dense convolution. Eg.
            y[0] = x[0] * w[0] + x[1] * w[1] + x[2] * w[2]
            y[1] = x[1] * w[0] + x[2] * w[1] + x[3] * w[2]
            y[2] = x[2] * w[0] + x[3] * w[1] + x[4] * w[2]
            with everything else being 0. Hence the only non-zero indices in the output are [0, 1, 2].
    """
    batch_size, max_x, max_y, max_z = spatial_range
    if isinstance(stride, int):
        stride = [stride] * 3
    if isinstance(padding, int):
        padding = [padding] * 3
    assert len(stride) == 3, "Stride must be a list of three integers."
    assert len(padding) == 3, "Padding must be a list of three integers."

    return torch.ops.convIdx.generate_conv3d_indices_v2(
        coords,
        batch_size,
        kernel_size,
        stride[0],
        stride[1],
        stride[2],
        padding[0],
        padding[1],
        padding[2],
        max_x,
        max_y,
        max_z,
        hash_map_multiplier,
        threads,
        lookup_tries
    )