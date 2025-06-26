# Copyright (c) 2021 Waabi Innovation. All rights reserved.
# %%
import pytest
import torch
import torchsparse
import torchsparse.nn as spnn


torchsparse.backends.hash_rsv_ratio = 16.0
from ops.idx_gen import gen_conv3d_indices, gen_conv3d_subm_indices


def make_unique_coords(b, N, batch_size=1):
    xs = torch.randint(0, b**3, (N,), dtype=torch.int32)
    xs = torch.unique(xs)
    return torch.stack(
        [torch.randint(0, batch_size, (xs.shape[0],), dtype=torch.int32), xs % b, xs // b % b, xs // (b * b)], dim=1
    )


def canonicalize_coords(c):
    """
    sparse convolution is permutation equivariant, eg.
    sparse_conv3d(features[perm], coords[perm]) == sparse_conv3d(features, coords)[perm]
    this function returns a canonical ordering of the coordinates, so that we can compare them
    """
    mi, ma = c.min(dim=0).values, c.max(dim=0).values
    c = c - mi[None]
    ms = torch.cat([ma.new_ones(1), ma - mi + 1]).cumprod(0)[None, :-1]
    return torch.argsort((c * ms).sum(-1))


def allclose_coords(c1, c2):
    if (c1.shape != c2.shape) or (c1.dtype != c2.dtype):
        return False
    if c1.shape[0] == 0 and c2.shape[0] == 0:
        return True
    c1 = c1.cpu()
    c2 = c2.cpu()
    m1 = c1.max(dim=0).values
    m2 = c2.max(dim=0).values
    if (m1 != m2).any():
        return False
    ms = torch.cat([m1.new_ones(1), m1 + 1]).cumprod(0)[:-1][None, :]
    cl1 = torch.sort((c1 * ms).sum(-1))
    cl2 = torch.sort((c2 * ms).sum(-1))
    return bool((cl1.values == cl2.values).all())


def get_torchsparse_coords(c, kernel_size=3, stride=1, padding=0, spatial_range=None):
    sparse_tensor = torchsparse.SparseTensor(
        feats=torch.ones([c.shape[0], 1], dtype=torch.float32),  # Dummy features
        coords=c.int(),
        spatial_range=(c.max(dim=0).values + 1).tolist() if spatial_range is None else spatial_range,
    ).cuda()
    mod = spnn.Conv3d(
        in_channels=1, out_channels=1, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
    ).cuda()
    sparse_tensor = mod(sparse_tensor)
    return sparse_tensor.C


def _test_subm_indices_cuda_vs_cpu(c, kernel_size=3):
    c = c.int()
    c2 = gen_conv3d_subm_indices(c.cpu(), kernel_size=kernel_size, hash_map_multiplier=8.0).cpu()
    c3 = gen_conv3d_subm_indices(c.cuda(), kernel_size=kernel_size, hash_map_multiplier=8.0).cpu()
    assert allclose_coords(c2, c3), f"Subm indices mismatch: {c2} vs {c3}"


@pytest.mark.parametrize("kernel_size", [3, 5])
def test_subm_unit(kernel_size):
    # this shows that padding doesn't matter for subm convolution
    c = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)
    _test_subm_indices_cuda_vs_cpu(c, kernel_size=kernel_size)

    c = torch.tensor([[0, 2, 2, 2]], dtype=torch.int32)
    _test_subm_indices_cuda_vs_cpu(c, kernel_size=kernel_size)


@pytest.mark.parametrize("kernel_size", [3, 5])
@pytest.mark.parametrize("seq", [1, 100, 5000])
def test_subm_random(kernel_size, seq):
    b = 100
    c = make_unique_coords(b, seq)
    _test_subm_indices_cuda_vs_cpu(c, kernel_size=kernel_size)


@pytest.mark.parametrize("kernel_size", [3, 5])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("stride", [2])  # torchsparse for stride 1 is subm convolution
def test_conv3d_unit(kernel_size, padding, stride):
    c = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)
    c2, _ = gen_conv3d_indices(
        c.cpu(), spatial_range=[1, 1, 1, 1], kernel_size=kernel_size, stride=stride, padding=padding
    )
    c3, _ = gen_conv3d_indices(
        c.cuda(), spatial_range=[1, 1, 1, 1], kernel_size=kernel_size, stride=stride, padding=padding
    )
    assert allclose_coords(c2, c3), f"Conv3D indices mismatch: {c2} vs {c3}"

    c = torch.tensor([[0, 2, 2, 2]], dtype=torch.int32)
    c2, _ = gen_conv3d_indices(
        c.cpu(), spatial_range=[1, 10, 10, 10], kernel_size=kernel_size, stride=stride, padding=padding
    )
    c3, _ = gen_conv3d_indices(
        c.cuda(), spatial_range=[1, 10, 10, 10], kernel_size=kernel_size, stride=stride, padding=padding
    )
    assert allclose_coords(c2, c3), f"Conv3D indices mismatch: {c2} vs {c3}"
    c4 = get_torchsparse_coords(
        c, kernel_size=kernel_size, stride=stride, padding=padding, spatial_range=[1, 10, 10, 10]
    )
    assert allclose_coords(c3, c4), f"Conv3D indices mismatch with torchsparse: {c3} vs {c4}"

    c = torch.tensor(
        [
            [0, 79, 90, 0],
            [0, 21, 19, 1],
            [0, 61, 15, 6],
            [0, 58, 56, 32],
            [0, 65, 10, 46],
            [0, 89, 26, 49],
            [0, 53, 77, 49],
            [0, 73, 10, 83],
            [0, 28, 30, 84],
            [0, 30, 76, 99],
        ],
        dtype=torch.int32,
    )
    c2, _ = gen_conv3d_indices(
        c.cpu(), spatial_range=[1, 100, 100, 100], kernel_size=kernel_size, stride=stride, padding=padding
    )
    c3, _ = gen_conv3d_indices(
        c.cuda(), spatial_range=[1, 100, 100, 100], kernel_size=kernel_size, stride=stride, padding=padding
    )
    assert allclose_coords(c2, c3), f"Conv3D indices mismatch: {c2} vs {c3}"
    c4 = get_torchsparse_coords(
        c, kernel_size=kernel_size, stride=stride, padding=padding, spatial_range=[1, 100, 100, 100]
    )
    assert allclose_coords(c3, c4), f"Conv3D indices mismatch with torchsparse: {c2} vs {c4}"


@pytest.mark.parametrize("kernel_size", [3, 5])
@pytest.mark.parametrize("padding", [0, 1, [1, 2, 1]])
@pytest.mark.parametrize("stride", [2, [2, 2, 3], [2, 3, 2]])
@pytest.mark.parametrize("seq", [1, 100, 5000])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_conv3d_random(kernel_size, padding, stride, seq, batch_size):
    torch.manual_seed(42)
    b = 100
    c = make_unique_coords(b, seq, batch_size)
    c2, _ = gen_conv3d_indices(
        c.cpu(), spatial_range=[batch_size, b, b, b], kernel_size=kernel_size, stride=stride, padding=padding
    )
    c3, _ = gen_conv3d_indices(
        c.cuda(), spatial_range=[batch_size, b, b, b], kernel_size=kernel_size, stride=stride, padding=padding
    )
    assert allclose_coords(c2, c3), f"Conv3D indices mismatch: {c2} vs {c3}"
    c4 = get_torchsparse_coords(
        c, kernel_size=kernel_size, stride=stride, padding=padding, spatial_range=[batch_size, b, b, b]
    )
    assert allclose_coords(c3, c4), f"Conv3D indices mismatch with torchsparse: {c2} vs {c4}"
