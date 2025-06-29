# %%

"""
This script shows that eventhough we may want to maximizing the number
of skipped zeros (for example, via sorting in spconv) in implicit gemm, 
it is not always the best idea.

Sometimes, cache performance is more important, and we should not sort the indices.
"""
import torch
import ops.conv3d_implicit_gemm as igemm
from triton.testing import do_bench
from utils import get_voxel_coords
import ops.idx_gen as idx_gen
import ops.encode as encode


def get_indices(N, K, density):
    indices = torch.randint(0, N, (N, K ** 3), device='cuda', dtype=torch.int32)
    flip = torch.rand((N, K ** 3), device='cuda') < density
    indices = torch.where(flip, indices, -1)
    return indices.int()

def benchmark_indices(indices, D, run_once=False):
    _, K1 = indices.shape
    N = (indices.max() + 1).item()
    feats = torch.randn(N, D, device='cuda', dtype=torch.float16)
    weights = torch.randn(K1, D, D, device='cuda', dtype=torch.float16)
    f = lambda: igemm.conv3d_implicit_gemm(feats, indices, weights, K)
    if run_once:
        f()
        return 0.0
    else:
        return do_bench(f)

def sort_indices(indices):
    N, K1 = indices.shape
    mask = indices >= 0
    powers = 2 ** torch.arange(K1, device=indices.device).view(1, -1)
    idx = (mask * powers).sum(dim=1)
    perm = idx.argsort()
    sorted_indices = indices[perm]
    return sorted_indices, perm

def sort_indices_encode(indices, coords):
    assert indices.shape[0] == coords.shape[0]
    code = encode.encode_fused(coords, torch.tensor([1], device=indices.device, dtype=torch.int32), 16)
    code = code[0]
    perm = code.argsort()
    sorted_indices = indices[perm]
    return sorted_indices

N = 400_000
K = 3

coords = get_voxel_coords(max_seq=N, device='cuda')
print("bench indices", do_bench(lambda: idx_gen.gen_conv3d_subm_indices(coords, K)))
print("bench indices v2", do_bench(lambda: idx_gen.gen_conv3d_subm_indices_v2(coords, K)))

idx = idx_gen.gen_conv3d_subm_indices(coords, K)
perm = torch.randperm(idx.size(0))
idx_perm = idx[perm]
sidx = sort_indices(idx)[0]
sidx2 = sort_indices_encode(idx, coords)


run_once = False
print(benchmark_indices(idx,      128, run_once=run_once))
print(benchmark_indices(sidx,     128, run_once=run_once))
print(benchmark_indices(idx_perm, 128, run_once=run_once))
print(benchmark_indices(sidx2,    128, run_once=run_once))

def calculate_skipped_zeros(indices, tile_size):
    N, K1 = indices.shape
    if N % tile_size != 0:
        indices = indices[:N - N % tile_size]
    indices = indices.view(-1, tile_size, K1)
    mask = indices >= 0
    mask = mask.sum(dim=1) > 0
    skipped_zeros = K1 - mask.sum(dim=1)
    return skipped_zeros

def summarize_implicit_gemm(indices, tile_size):
    skipped_zeros = calculate_skipped_zeros(indices, tile_size).float()
    print("Average skipped zeros:", skipped_zeros.mean().item())
    print("Max skipped zeros:", skipped_zeros.max().item())

T = 64
summarize_implicit_gemm(idx, T)
summarize_implicit_gemm(sidx, T)
summarize_implicit_gemm(idx_perm, T)
summarize_implicit_gemm(sidx2, T)

