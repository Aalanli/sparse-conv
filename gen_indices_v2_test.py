# %%

import torch
from kernel_gen.utils import get_voxel_coords
from idx_gen_test import canonicalize_coords
from ops.idx_gen import gen_conv3d_subm_indices, gen_conv3d_subm_indices_v2, gen_conv3d_indices, gen_conv3d_indices_v2
from triton.testing import do_bench

K = 3
coords = get_voxel_coords(400_000)
# coords = torch.tensor([[0, 1, 1, 1]], device='cuda', dtype=torch.int32)

# idx1 = gen_conv3d_subm_indices(coords, K)
# idx2, mask = gen_conv3d_subm_indices_v2(coords, K)
# idx2 = idx2.T
# print(torch.all(idx1 == idx2))
# mask_ref = (idx2 >= 0).any(dim=1).int()
# print(torch.all(mask_ref == (mask != 0)))



# print("v2 time: ", do_bench(lambda: gen_conv3d_subm_indices_v2(coords, K)[0]))
# print("v1 time: ", do_bench(lambda: gen_conv3d_subm_indices(coords, K)))


spatial_range = (coords.max(dim=0).values + 10).tolist()
c1, idx1 = gen_conv3d_indices(coords, spatial_range, K, stride=2)
c2, idx2 = gen_conv3d_indices_v2(coords, spatial_range, K, stride=2)
idx2 = idx2.T

p1 = canonicalize_coords(c1)
p2 = canonicalize_coords(c2)

c1 = c1[p1]
c2 = c2[p2]
idx1 = idx1[p1]
idx2 = idx2[p2]

print(torch.all(c1 == c2))
print(torch.all(idx1 == idx2))
print(idx1)
print(idx2)
print(c1)
print(c2)

print("v1 time", do_bench(lambda: gen_conv3d_indices(coords, spatial_range, K, stride=2)))
print("v2 time", do_bench(lambda: gen_conv3d_indices_v2(coords, spatial_range, K, stride=2)))

