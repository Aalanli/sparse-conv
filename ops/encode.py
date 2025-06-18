#Copyright(c) 2021 Waabi Innovation.All rights reserved.
import torch from torch import Tensor

    def encode_z(x:torch.Tensor, depth: int) :assert x.dim() == 2 assert x.shape[1] == 3 assert x.shape[1] * depth < 64
    return torch.ops.spacy_curves.encode_z(x, depth)


def encode_hilbert(pts: torch.Tensor, depth: int):
    assert pts.dim() == 2
    assert pts.shape[1] == 3
    assert pts.shape[1] * depth < 64
    return torch.ops.spacy_curves.encode_hilbert(pts, depth)


def encode_fused(pts: torch.Tensor, enc_ids: torch.Tensor, depth: int):
    """
    pts: (N, 3) or (N, 4) int32 (first dim is batch id)
    enc_ids: (K) between 0 and 3
        - 0: z
        - 1: z-trans
        - 2: hilbert
        - 3: hilbert-trans
    output: (K, N) int64
    """
    return torch.ops.spacy_curves.fused_encoding(pts, depth, enc_ids, 1)


class FastSerializer:
    def __init__(self, encoding_schemes: list[str]):
        enc_id_map = {"z": 0, "z-trans": 1, "hilbert": 2, "hilbert-trans": 3}
        encoding_ids: list[int] = []
        for enc in encoding_schemes:
            if enc not in enc_id_map:
                raise ValueError(f"Unknown encoding scheme: {enc}")
            encoding_ids.append(enc_id_map[enc])
        self.enc_id_map = enc_id_map
        self.id_to_name = {v: k for k, v in enc_id_map.items()}
        self.encodings = {"cpu": torch.tensor(encoding_ids, dtype=torch.int32)}

    def encode(self, grid_coord: Tensor, depth: int = 16, shuffle_orders: bool = False):
        dev_str = str(grid_coord.device)
        if dev_str not in self.encodings:
            self.encodings[dev_str] = self.encodings["cpu"].to(grid_coord.device)
        encodings = self.encodings[dev_str]
        if grid_coord.dtype != torch.int32:
            grid_coord = grid_coord.to(torch.int32)
        enc_id_map = self.enc_id_map
        if shuffle_orders:
            perm = torch.randperm(encodings.shape[0])
            enc_id_map = {self.id_to_name[i]: int(perm[i]) for i in range(encodings.shape[0])}
            encodings = encodings[perm]

        codes = encode_fused(grid_coord, encodings, depth)
        return codes, enc_id_map
