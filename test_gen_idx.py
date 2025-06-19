import glob
import torch
import ops.idx_gen

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


# ncu --set full --target-processes all -o idx_gen -f --profile-from-start off python test_gen_idx.py 

N = 400_000
D = 32
K = 3
S = 2

idx_rand = generate_random_coords(N, max_coord=1000, batch_size=1, device='cuda')
spatial_range_rand = get_spatial_range(idx_rand)
idx_actual = get_voxel_coords(N, device='cuda')
idx_actual = idx_actual[torch.randperm(idx_actual.size(0))]
spatial_range_actual = get_spatial_range(idx_actual)
print(spatial_range_actual, spatial_range_rand)

torch.cuda.cudart().cudaProfilerStart()
ops.idx_gen.gen_conv3d_subm_indices(idx_rand, K)
ops.idx_gen.gen_conv3d_subm_indices(idx_actual, K, threads=512)


c1, r1 = ops.idx_gen.gen_conv3d_indices(idx_rand, spatial_range_rand, K, S)
c2, r2 = ops.idx_gen.gen_conv3d_indices(idx_actual, spatial_range_actual, K, S)

torch.cuda.cudart().cudaProfilerStop()

