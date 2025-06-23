import glob
import torch
import ops.idx_gen
import ops.conv3d_implicit_gemm as implicit_gemm

from bench import TorchsparseSubM, SpconvSubM

def generate_random_coords(N, max_coord=100, batch_size=1, device='cuda'):
    coords = torch.randint(0, max_coord, (N, 3), device=device)
    batch_ids = torch.randint(0, batch_size, (N, 1), device=device)
    return torch.cat([batch_ids, coords], dim=1).int()

def get_voxel_coords(max_seq: int, device='cuda'):
    coords = []
    for f in glob.glob('example_pts/*.pt'):
        data = torch.load(f, map_location=device)
        coords.append(data)

    coords = torch.cat(coords, dim=0)
    return coords[:max_seq].int()

def get_spatial_range(coords):
    return (coords.max(dim=0).values + 1).int().tolist()


# ncu --set full --target-processes all -o idx_gen -f --profile-from-start off python ncu.py 

N = 40_000
D = 64
K = 3
S = 2
with torch.autocast('cuda'):
    with torch.no_grad():
        spconv = SpconvSubM(D, D, K)
        torchsparse = TorchsparseSubM(D, D, K)

        f = torch.randn(N, D, device='cuda', dtype=torch.float16)
        w = torch.randn(K ** 3, D, D, device='cuda', dtype=torch.float16)
        idx_rand = generate_random_coords(N, max_coord=1000, batch_size=1, device='cuda')
        spatial_range_rand = get_spatial_range(idx_rand)
        idx_actual = get_voxel_coords(N, device='cuda')
        idx_actual_perm = idx_actual[torch.randperm(idx_actual.size(0))]
        spatial_range_actual = get_spatial_range(idx_actual)

        ind0 = ops.idx_gen.gen_conv3d_subm_indices(idx_rand, K)
        ind1 = ops.idx_gen.gen_conv3d_subm_indices(idx_actual, K)
        ind2 = ops.idx_gen.gen_conv3d_subm_indices(idx_actual_perm, K)

        implicit_gemm.conv3d_implicit_gemm(f, ind0, w, K)
        spconv.forward(f, idx_rand, spatial_range_rand)
        torchsparse.forward(f, idx_rand, spatial_range_rand)

        torch.cuda.cudart().cudaProfilerStart()


        # ind0 = ops.idx_gen.gen_conv3d_subm_indices(idx_rand, K)
        # implicit_gemm.conv3d_implicit_gemm(f, ind0, w, K)

        ind1 = ops.idx_gen.gen_conv3d_subm_indices(idx_actual, K)
        implicit_gemm.conv3d_implicit_gemm(f, ind1, w, K)

        # ind2 = ops.idx_gen.gen_conv3d_subm_indices(idx_actual_perm, K)
        # implicit_gemm.conv3d_implicit_gemm(f, ind2, w, K)

        # spconv.forward(f, idx_rand, spatial_range_rand)
        spconv.forward(f, idx_actual, spatial_range_actual)
        # spconv.forward(f, idx_actual_perm, spatial_range_actual)


        # torchsparse.forward(f, idx_rand, spatial_range_rand)
        torchsparse.forward(f, idx_actual, spatial_range_actual)
        # torchsparse.forward(f, idx_actual_perm, spatial_range_actual)

        torch.cuda.cudart().cudaProfilerStop()

