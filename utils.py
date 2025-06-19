import glob
import torch

def get_voxel_coords(max_seq: int, device='cuda'):
    coords = []
    for f in glob.glob('example_pts/*.pt'):
        data = torch.load(f, map_location=device)
        coords.append(data)

    coords = torch.cat(coords, dim=0)
    return coords[:max_seq].int()
