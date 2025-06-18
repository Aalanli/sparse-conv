
import torch
import ops.idx_gen

idx = torch.randint(0, 100, (100000, 4), device='cuda', dtype=torch.int32)

ops.idx_gen.gen_conv3d_subm_indices(idx, 3)

ops.idx_gen.gen_conv3d_indices(idx, [1, 100, 100, 100], 3, stride=2, padding=0)
torch.cuda.synchronize()
print(idx)
