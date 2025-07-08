# %%
import pytest
import torch
import torchsparse
import torchsparse.nn as spnn


torchsparse.backends.hash_rsv_ratio = 16.0
from ops import conv3d_implicit_gemm, idx_gen
from conv3d import SparseConv3d, SparseSubmConv3d, SparseTensor
from idx_gen_test import (
    allclose_coords,
    canonicalize_coords,
    make_unique_coords,
)


def allclose_sparse_tensor(t1: SparseTensor, t2: SparseTensor) -> bool:
    if t1.features.shape != t2.features.shape or t1.coords.shape != t2.coords.shape:
        return False
    if not allclose_coords(t1.coords, t2.coords):
        return False
    p1 = canonicalize_coords(t1.coords)
    p2 = canonicalize_coords(t2.coords)
    if not torch.allclose(t1.features[p1], t2.features[p2], atol=1e-3, rtol=1e-3):
        print("Features mismatch")
        print((t1.features[p1] - t2.features[p2]).abs().max())
        return False
    return True


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("kernel_size", [3])
@pytest.mark.parametrize("seq", [1, 100, 1000])
@pytest.mark.parametrize("channels", [(1, 1), (4, 4), (16, 16), (16, 32)])
@pytest.mark.parametrize("stride_pad", [(1, 0), (2, 0), (2, 1), (2, 2), ([2, 3, 2], [1, 2, 1])])
@torch.no_grad
def test_conv3d(batch_size, kernel_size, seq, channels, stride_pad):
    torch.manual_seed(0)
    in_channels, out_channels = channels
    stride, pad = stride_pad
    b = 100
    c = make_unique_coords(b, seq, batch_size)
    c = c.cuda()
    sparse_tensor = SparseTensor(
        features=torch.randn([c.shape[0], in_channels], dtype=torch.float32).cuda(),  # Dummy features
        coords=c.int().cuda(),
        spatial_range=[batch_size, b, b, b],
    )
    spnn_tensor = torchsparse.SparseTensor(
        feats=sparse_tensor.features, coords=sparse_tensor.coords, spatial_range=list(sparse_tensor.spatial_range)
    )

    sp_conv = spnn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=pad,
        bias=False,
    ).cuda()

    if stride == 1:
        gather_conv = SparseSubmConv3d(
            in_channels=in_channels, out_channels=out_channels, bias=False, kernel_size=kernel_size
        ).cuda()
    else:
        gather_conv = SparseConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            bias=False,
        ).cuda()
    gather_conv.weight_master = torch.nn.Parameter(
        sp_conv.kernel.reshape(kernel_size, kernel_size, kernel_size, -1)
        .permute(2, 1, 0, 3)
        .reshape(kernel_size**3, in_channels, out_channels)
        .contiguous()
    )

    out1 = sp_conv(spnn_tensor)
    out1 = SparseTensor(features=out1.F.half(), coords=out1.C, spatial_range=(out1.spatial_range))
    out2 = gather_conv(sparse_tensor)
    assert allclose_sparse_tensor(out1, out2), f"SubmConv3D output mismatch: {out1.features} vs {out2.features}"


def reference_conv3d_subm(feats: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor, kernel_size: int):
    n = feats.shape[0]
    feats = torch.nn.functional.pad(feats, (0, 0, 0, 1))
    indices = torch.where(indices < 0, n, indices)
    out = feats[indices.flatten()].reshape(-1, kernel_size**3 * feats.shape[1]) @ weights.view(-1, weights.shape[-1])
    return out


def reference_backwards(
    dout: torch.Tensor, feats: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor, kernel_size: int
):
    # dout: [N', D']
    # feats: [N, D]
    # weights: [K**3, D, D']
    K3, D, DPrime = weights.shape
    n = feats.shape[0]

    # [N+1, D]
    feats = torch.nn.functional.pad(feats, (0, 0, 0, 1))
    indices = torch.where(indices < 0, n, indices).flatten()
    weights_view = weights.view(-1, DPrime)
    # [N', K**3 * D]
    feats_gathered = feats[indices].reshape(-1, kernel_size**3 * D)
    dfeats_gathered = dout @ weights_view.T
    dweights_view = feats_gathered.T @ dout
    dfeats = torch.zeros_like(feats)
    dfeats.index_add_(0, indices, dfeats_gathered.reshape(-1, D))
    dfeats = dfeats[:-1]
    dweights = dweights_view.view(K3, D, DPrime)
    return dfeats, dweights


@pytest.mark.parametrize("N", [100, 1000])
@pytest.mark.parametrize("D_DPrime", [(16, 16), (16, 32), (32, 16), (32, 32)])
@pytest.mark.parametrize("kernel_size", [3, 5])
def test_reference_backwards(N, D_DPrime, kernel_size):
    D, DPrime = D_DPrime
    coords = make_unique_coords(100, N, batch_size=1).cuda()
    indices = idx_gen.gen_conv3d_subm_indices(coords, kernel_size)
    N = indices.shape[0]

    feats = torch.randn(N, D, device="cuda", requires_grad=True)
    weights = torch.randn(kernel_size**3, D, DPrime, device="cuda", requires_grad=True)

    out1 = reference_conv3d_subm(feats, indices, weights, kernel_size)
    dout = torch.randn_like(out1, device="cuda")
    (out1 * dout).sum().backward()

    dfeats_ref = feats.grad
    dweights_ref = weights.grad

    with torch.no_grad():
        dfeats, dweights = reference_backwards(dout, feats, indices, weights, kernel_size)

    assert torch.allclose(dfeats, dfeats_ref, atol=1e-3, rtol=1e-3)
    assert torch.allclose(dweights, dweights_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("N", [100, 1000])
@pytest.mark.parametrize("D_DPrime", [(16, 16), (16, 32), (32, 16), (32, 32)])
@pytest.mark.parametrize("kernel_size", [3, 5])
@pytest.mark.parametrize(
    "dtype_weight_dtype",
    [(torch.float32, torch.float32), (torch.float16, torch.float16), (torch.float16, torch.float32)],
)
@pytest.mark.parametrize("sorted", [False, True])
def test_backwards(N, D_DPrime, kernel_size, dtype_weight_dtype, sorted):
    from kernel_gen.triton_implicit_gemm import implicit_gemm_grad
    D, DPrime = D_DPrime
    dtype, weight_dtype = dtype_weight_dtype
    coords = make_unique_coords(100, N, batch_size=1).cuda()
    indices = idx_gen.gen_conv3d_subm_indices(coords, kernel_size)
    N = indices.shape[0]

    feats = torch.randn(N, D, device="cuda", requires_grad=True, dtype=dtype)
    weights = torch.tensor(
        torch.randn(kernel_size**3, D, DPrime, device="cuda", dtype=weight_dtype) / D**0.5, requires_grad=True, dtype=weight_dtype
    )
    dout = torch.randn(N, DPrime, device="cuda", dtype=dtype)
    with torch.no_grad():
        dfeats_ref, dweights_ref = reference_backwards(
            dout.float(), feats.float(), indices, weights.float(), kernel_size
        )

    res = conv3d_implicit_gemm.conv3d_implicit_gemm(feats, indices.T.contiguous(), weights.half(), kernel_size, sorted=sorted, parallel_k=True)
    (res * dout).sum().backward()

    dfeats = feats.grad
    dweights = weights.grad
    atol = 1e-2 if dtype == torch.float32 else 1e-1
    rtol = 1e-2 if dtype == torch.float32 else 1e-1

    diff_feats = (dfeats - dfeats_ref).abs()
    print("max feature gradient mismatch:", diff_feats.max(), diff_feats.mean())

    # if not torch.allclose(dfeats.float(), dfeats_ref, atol=atol, rtol=rtol):
    #     assert False, f"Feature gradients mismatch: {dfeats} vs {dfeats_ref}"

    atol = 1e-1  # due to parallel-k this is less precise (we use atomic instructions)
    rtol = 1e-2 if dtype == torch.float32 else 1e-1
    diff_weights = (dweights - dweights_ref).abs()
    print("max weight gradient mismatch:", diff_weights.max(), diff_weights.mean())
    # if not torch.allclose(dweights.float(), dweights_ref, atol=atol, rtol=rtol):
    #     assert False, f"Weight gradients mismatch: {dweights} vs {dweights_ref}"
    
    dfeats_triton, dweights_triton = implicit_gemm_grad(dout, feats, indices.T.contiguous(), weights.half())

    print("max triton feature gradient mismatch:", (dfeats_triton - dfeats_ref).abs().mean())
    print("max triton weight gradient mismatch:", (dweights_triton - dweights_ref).abs().mean())

test_backwards(100000, (128, 128), 3, (torch.float16, torch.float32), False)  # Example call to test function

@pytest.mark.parametrize("N", [100, 1000])
@pytest.mark.parametrize("D", [16, 32])
def test_to_dense(N, D):
    coords = make_unique_coords(50, N, batch_size=1).cuda()
    features = torch.randn(coords.shape[0], D, device="cuda")

    sparse_tensor = SparseTensor(
        features=features,
        coords=coords.int().cuda(),
        spatial_range=[1, 50, 50, 50],
    )
    sparse_tensor2 = torchsparse.SparseTensor(
        feats=sparse_tensor.features, coords=sparse_tensor.coords, spatial_range=[1, 50, 50, 50]
    )
    dense_tensor = sparse_tensor.dense()
    dense_tensor2 = sparse_tensor2.dense()

    assert torch.allclose(dense_tensor, dense_tensor2), "Dense conversion mismatch"
