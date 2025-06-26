from pathlib import Path
import torch

def get_voxel_coords(max_seq: int, device='cuda'):
    coords = []
    for f in (Path(__file__).parent / "example_pts").glob("*.pt"):
        data = torch.load(f, map_location=device)
        coords.append(data)

    coords = torch.cat(coords, dim=0)
    return coords[:max_seq].int()
