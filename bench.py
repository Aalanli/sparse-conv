# %%
import time
import glob
import ops
import argparse

import torch
import matplotlib.pyplot as plt
import triton.testing
import triton.language as tl

# Import implementations
try:
    from torchsparse.nn import conv as ts_conv
    from torchsparse import SparseTensor
    import torchsparse
    torchsparse.backends.allow_fp16 = True
    torchsparse.backends.hash_rsv_ratio = 8.0
except ImportError:
    ts_conv = None

try:
    from spconv.pytorch import SparseConvTensor, SubMConv3d as SPConv3d
    from spconv.pytorch import ops as spconv_ops
except ImportError:
    SPConv3d = None


import ops.conv3d_implicit_gemm
import ops.idx_gen

def generate_random_coords(N, max_coord=50, batch_size=2, device='cuda'):
    # batch indices [0, batch_size)
    batch = torch.randint(0, batch_size, (N, 1), device=device)
    xyz = torch.randint(0, max_coord, (N, 3), device=device)
    return torch.cat([batch, xyz], dim=1).int()


def get_voxel_coords(max_seq: int, device='cuda'):
    coords = []
    for f in glob.glob('example_pts/*.pt'):
        data = torch.load(f, map_location=device)
        coords.append(data)

    coords = torch.cat(coords, dim=0)
    return coords[:max_seq].int()

class ImplBase:
    name = 'base'

    def __init__(self, in_channels, out_channels, kernel_size=3):
        raise NotImplementedError

    def forward(self, feats, coords, spatial_range):
        raise NotImplementedError


class TorchsparseSubM(ImplBase):
    name = 'torchsparse'

    def __init__(self, in_channels, out_channels, kernel_size=3):
        assert ts_conv is not None, "torchsparse not installed"
        # torchsparse SubMConv3d
        
        self.conv = ts_conv.Conv3d(in_channels, out_channels, kernel_size, bias=False).cuda()

    def forward(self, feats, coords, spatial_range):
        # expect coords as [N,4] int tensor
        # torchsparse expects a SparseTensor
        x = SparseTensor(feats, coords.int(), spatial_range=spatial_range)
        out = self.conv(x)
        return out.F


class SpconvSubM(ImplBase):
    name = 'spconv'

    def __init__(self, in_channels, out_channels, kernel_size=3):
        assert SPConv3d is not None, "spconv not installed"
        # spconv SubMConv3d needs spatial shape and tensor
        self.layer = SPConv3d(in_channels, out_channels, kernel_size, bias=False).cuda()

    def forward(self, feats, coords, spatial_range):
        sp_t = SparseConvTensor(feats, coords.int(), spatial_shape=spatial_range[1:], batch_size=spatial_range[0])
        out = self.layer(sp_t)
        return out.features

class Conv3DSubmAot(ImplBase):
    name = 'conv3d_subm_aot'

    def __init__(self, in_channels, out_channels, kernel_size=3):
        self.weight = torch.nn.Parameter(
            torch.randn(kernel_size**3, in_channels, out_channels, device='cuda', dtype=torch.float16)
        )
        self.kernel_size = kernel_size

    def forward(self, feats, coords, spatial_range):
        # coords: [N, 4] where last dimension is (x, y, z, batch_id)
        # feats: [N, D]
        indices = ops.idx_gen.gen_conv3d_subm_indices(coords, self.kernel_size)
        return ops.conv3d_implicit_gemm.conv3d_implicit_gemm(
            feats, indices, self.weight, self.kernel_size
        )


import ops.idx_gen
from triton_spconv import Conv3DSubmModule
import pandas as pd
from tabulate import tabulate
class ImplicitGemmAccf32(ImplBase):
    name = 'implicit_gemm_accf32'

    def __init__(self, in_channels, out_channels, kernel_size=3):
        self.layer = Conv3DSubmModule(kernel_size, in_channels, out_channels, acc_dtype=tl.float32).cuda()

    def forward(self, feats, coords, spatial_range):
        # coords: [N, 4] where last dimension is (x, y, z, batch_id)
        # feats: [N, D]
        return self.layer(feats, coords)
    
class ImplicitGemmAccf16(ImplBase):
    name = 'implicit_gemm_accf16'

    def __init__(self, in_channels, out_channels, kernel_size=3):
        self.layer = Conv3DSubmModule(kernel_size, in_channels, out_channels, acc_dtype=tl.float16).cuda()

    def forward(self, feats, coords, spatial_range):
        # coords: [N, 4] where last dimension is (x, y, z, batch_id)
        # feats: [N, D]
        return self.layer(feats, coords)
    

class NaiveConv3D(ImplBase):
    name = 'naive_conv3d'

    def __init__(self, in_channels, out_channels, kernel_size=3):
        self.kernel_size = kernel_size
        self.weight = torch.nn.Parameter(
            torch.randn(kernel_size**3, in_channels, out_channels, device='cuda', dtype=torch.float16)
        )

    def forward(self, feats, coords, spatial_range):
        indices = ops.idx_gen.gen_conv3d_subm_indices(coords, self.kernel_size)
        n = feats.shape[0]
        feats = torch.nn.functional.pad(feats, (0, 0, 0, 1))  # pad to handle indices
        indices = torch.where(indices < 0, n, indices)  # replace -1 with N
        out = feats[indices.flatten()].reshape(-1, self.kernel_size**3 * feats.shape[1]) @ self.weight.view(-1, self.weight.shape[-1])
        return out


@torch.no_grad
def benchmark_impl(impl_cls: type[ImplBase], Ns, Ds, warmup=10, runs=50, device='cuda', dtype=torch.float16):
    results = {D: [] for D in Ds}
    vox_coords = get_voxel_coords(max_seq=max(Ns), device=device)
    spatial_range = (vox_coords.max(dim=0).values + 1).tolist()  # [batch_size, x, y, z]

    with torch.autocast(device_type=device, dtype=torch.float16):
        for D in Ds:
            # instantiate implementation
            impl = impl_cls(in_channels=D, out_channels=D, kernel_size=3)
            for N in Ns:
                # generate data
                feats = torch.randn(N, D, device=device, dtype=dtype)
                # coords = generate_random_coords(N, max_coord=max_coord, batch_size=batch, device=device)
                coords = vox_coords[:N]
                
                torch.cuda.synchronize() 
                torch.cuda.nvtx.range_push(f"Benchmark {impl.name} D={D} N={N}")
                for _ in range(3):
                    _ = impl.forward(feats, coords, spatial_range)
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_push(f"Benchmark {impl.name} D={D} N={N} benchmarks")
                elapsed = triton.testing.do_bench(
                    lambda: impl.forward(feats, coords, spatial_range),
                    warmup=warmup, rep=runs
                )
                torch.cuda.nvtx.range_pop()
                results[D].append(elapsed)
                print(f"Impl={impl.name}, D={D}, N={N}, time={elapsed:.3f} ms")
    return results


def plot_results(all_results, Ns, Ds, out_file='benchmark.png'):
    for impl_name, results in all_results.items():
        for D in Ds:
            plt.plot(Ns, results[D], marker='o', label=f"{impl_name}-D{D}")
    plt.xlabel('Sequence length N')
    plt.ylabel('Avg execution time (ms)')
    sm_version = torch.cuda.get_device_capability()

    plt.title(f'SubM 3D Sparse Conv Benchmark, SM{sm_version[0]}{sm_version[1]}' )
    plt.legend()
    plt.grid(True)
    plt.savefig(out_file)
    print(f"Saved plot to {out_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Ns', type=int, nargs='+', default=[1000, 5000, 10000, 50000, 100000, 200000, 300000, 400000])
    parser.add_argument('--Ds', type=int, nargs='+', default=[32])
    parser.add_argument('--runs', type=int, default=50)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--plot_file', type=str, required=False, default=None)
    args = parser.parse_args()

    # list of implementations
    implementations = [SpconvSubM, TorchsparseSubM, Conv3DSubmAot]

    all_results = {}
    for impl in implementations:
        all_results[impl.name] = benchmark_impl(
            impl, args.Ns, args.Ds,
            warmup=args.warmup, runs=args.runs
        )
    # all_results['implicit_gemm'] = benchmark_impl(
    #     ImplicitGemm, args.Ns, args.Ds,
    #     warmup=args.warmup, runs=args.runs
    # )
    if args.plot_file is not None:
        plot_results(all_results, args.Ns, args.Ds, out_file=args.plot_file)
    
    for D in args.Ds:
        print(f"Results for D={D}:")
        # Build a table: rows indexed by N, columns by implementation name
        data = {impl_name: results[D] for impl_name, results in all_results.items()}
        df = pd.DataFrame(data, index=args.Ns)
        df.index.name = 'N'
        print(tabulate(df, headers='keys', tablefmt='psql'))
        print("")


if __name__ == '__main__':
    main()
