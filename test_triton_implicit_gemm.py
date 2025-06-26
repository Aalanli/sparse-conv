# %%
import ops
import glob
import os
import torch

import ops.conv3d_implicit_gemm as aot_implicit_gemm
import ops.conv3d_implicit_gemm
import ops.idx_gen
from kernel_gen.triton_implicit_gemm import conv3d_implicit_gemm
from kernel_gen.utils import get_voxel_coords


def reference_conv3d_subm(feats: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor, kernel_size: int):
    n = feats.shape[0]
    feats = torch.nn.functional.pad(feats, (0, 0, 0, 1))
    indices = torch.where(indices < 0, n, indices)
    out = feats[indices.flatten()].reshape(-1, kernel_size**3 * feats.shape[1]) @ weights.view(-1, weights.shape[-1])
    return out

def reference_backwards(dout: torch.Tensor, feats: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor, kernel_size: int):
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


def test_reference_backwards(coords, D, DPrime, kernel_size):
    indices = ops.idx_gen.gen_conv3d_subm_indices(coords, kernel_size)
    N = indices.shape[0]

    feats = torch.randn(N, D, device="cuda", requires_grad=True)
    weights = torch.randn(kernel_size ** 3, D, DPrime, device="cuda", requires_grad=True)

    out1 = reference_conv3d_subm(feats, indices, weights, kernel_size)
    dout = torch.randn_like(out1, device="cuda")
    (out1 * dout).sum().backward()

    dfeats_ref = feats.grad
    dweights_ref = weights.grad

    with torch.no_grad():
        dfeats, dweights = reference_backwards(dout, feats, indices, weights, kernel_size)
    
    if not torch.allclose(dfeats, dfeats_ref, atol=1e-3, rtol=1e-3):
        print("dfeats do not match!")
        print("Reference:", dfeats_ref)
        print("Computed:", dfeats)
        print("Max diff:", (dfeats - dfeats_ref).abs().max())
    
    if not torch.allclose(dweights, dweights_ref, atol=1e-3, rtol=1e-3):
        print("dweights do not match!")
        print("Reference:", dweights_ref)
        print("Computed:", dweights)
        print("Max diff:", (dweights - dweights_ref).abs().max())


def test_backwards(coords, D, DPrime, kernel_size):
    indices = ops.idx_gen.gen_conv3d_subm_indices(coords, kernel_size)
    N = indices.shape[0]

    feats = torch.randn(N, D, device="cuda", requires_grad=True)
    weights = torch.randn(kernel_size ** 3, D, DPrime, device="cuda", requires_grad=True)
    dout = torch.randn(N, DPrime, device="cuda")
    with torch.no_grad():
        dfeats_ref, dweights_ref = reference_backwards(dout, feats, indices, weights, kernel_size)

    res = ops.conv3d_implicit_gemm.conv3d_implicit_gemm(
        feats, indices, weights, kernel_size
    )
    (res * dout).sum().backward()

    dfeats = feats.grad
    dweights = weights.grad

    if not torch.allclose(dfeats, dfeats_ref, atol=1e-3, rtol=1e-3):
        print("dfeats do not match!")
        print("Reference:", dfeats_ref)
        print("Computed:", dfeats)
        print("Max diff:", (dfeats - dfeats_ref).abs().max())
    
    if not torch.allclose(dweights, dweights_ref, atol=1e-3, rtol=1e-3):
        print("dweights do not match!")
        print("Reference:", dweights_ref)
        print("Computed:", dweights)
        print("Max diff:", (dweights - dweights_ref).abs().max())

def compare_conv3d_subm(coords, dim_in, dim_out, kernel_size):
    n = coords.shape[0]
    feats = torch.randn(n, dim_in, device="cuda", dtype=torch.float16)
    weights = torch.randn(kernel_size ** 3, dim_in, dim_out, device="cuda", dtype=torch.float16) / dim_in**0.5
    indices = ops.idx_gen.gen_conv3d_subm_indices(coords, kernel_size)

    _ = conv3d_implicit_gemm(feats, indices, weights, kernel_size)
    out_triton = conv3d_implicit_gemm(feats, indices, weights, kernel_size)
    out_ref = reference_conv3d_subm(feats, indices, weights, kernel_size)

    if not torch.allclose(out_triton, out_ref, atol=1e-1, rtol=1e-3):
        diffs = (out_triton - out_ref).abs()
        print(f"Max diff: {diffs.max()}")
        print("Outputs do not match!")
        print(out_triton)
        print(out_ref)

    out_aot = aot_implicit_gemm.conv3d_implicit_gemm(
        feats, indices, weights, kernel_size
    )
    if not (torch.allclose(out_ref, out_aot, atol=1e-1, rtol=1e-2)):
        print(f"Outputs do not match! ref vs aot", out_ref.shape)
        diffs = (out_aot - out_ref).abs()
        print(diffs.max())
        idx = int(diffs.flatten().argmax().item())
        row = idx // out_triton.shape[1]
        print(row, idx % out_triton.shape[1])
        # print(out_aot[row])
        # print(out_ref[row])
        print(out_aot.flatten()[idx], out_ref.flatten()[idx])
        print(diffs.mean())   


def test_triton_jit_kernels_conv3d_subm(coords, dim_in, dim_out, kernel_size):
    from kernel_gen.triton_implicit_gemm import conv3d_implicit_gemm, conv3d_implicit_gemm_T
    n = coords.shape[0]
    feats = torch.randn(n, dim_in, device="cuda", dtype=torch.float16)
    weights = torch.randn(kernel_size ** 3, dim_in, dim_out, device="cuda", dtype=torch.float16) / dim_in**0.5
    indices = ops.idx_gen.gen_conv3d_subm_indices(coords, kernel_size)
    indices_T = ops.idx_gen.gen_conv3d_subm_indices_v2(coords, kernel_size)
    assert (indices == indices_T.T).all()
    N = indices_T.shape[1]
    indices_T_ = torch.empty([indices_T.shape[0], N + 1000], device=indices_T.device, dtype=indices_T.dtype)
    indices_T_[:, :N] = indices.T
    indices_T = indices_T_.narrow(1, 0, N)

    out_ref = reference_conv3d_subm(feats, indices, weights, kernel_size)
    out_triton = conv3d_implicit_gemm(feats, indices, weights, kernel_size)
    out_triton_T = conv3d_implicit_gemm_T(feats, indices_T, weights, kernel_size)

    if not torch.allclose(out_triton, out_ref, atol=1e-1, rtol=1e-3):
        diffs = (out_triton - out_ref).abs()
        print(f"Max diff: {diffs.max()}")
        print("Triton Jit. Outputs do not match!")
        print(out_triton)
        print(out_ref)
        print(out_triton[-1])
        print(indices[-1])
    
    if not torch.allclose(out_triton_T, out_ref, atol=1e-1, rtol=1e-3):
        diffs = (out_triton_T - out_ref).abs()
        print(f"Max diff: {diffs.max()}")
        print("Triton Jit T. Outputs do not match!")
        print(out_triton_T)
        print(out_ref)
        print(out_triton_T[-1])
        print(indices_T[:, -1])

def test():
    idx = get_voxel_coords(10000, device='cuda')

    # compare_conv3d_subm(idx, 16, 32, 3)
    # compare_conv3d_subm(idx, 64, 64, 3)
    # compare_conv3d_subm(idx, 128, 128, 3)
    # compare_conv3d_subm(idx, 64, 128, 3)

    # test_reference_backwards(idx, 16, 16, 3)
    # test_reference_backwards(idx, 16, 32, 3)
    # test_reference_backwards(idx, 64, 64, 3)

    # test_backwards(idx, 16, 16, 3)
    # test_backwards(idx, 16, 32, 3)
    # test_backwards(idx, 64, 64, 3)

    test_triton_jit_kernels_conv3d_subm(idx, 16, 16, 3)
    test_triton_jit_kernels_conv3d_subm(idx, 64, 64, 3)

test()

