# %%
import json
import torch

kernel_map_path = 'kernel_ptx/kernel_map.json'
kernel_maps = json.load(open(kernel_map_path, 'r'))


# %%

def collect_by(
    D, DPrime, kernel_size, acc_dtype, dtype, sm=None
):
    times = []
    indices = []
    sequences = []
    for v in kernel_maps:
        if (
            v['D'] == D
            and v['DPrime'] == DPrime
            and v['K'] == kernel_size
            and v['acc_dtype'] == acc_dtype
            and v['dtype'] == dtype
        ):
            if sm is not None and v['sm'] != sm:
                continue
            times.append(v['time'])
            indices.append(v["kidx"])
            sequences.append(v["N"])
    return times, sequences, indices

t1, s1, i1 = collect_by(
    D=32, DPrime=64, kernel_size=3, acc_dtype='fp32', dtype='fp16'
)
times = torch.tensor(t1, dtype=torch.float32)
amin = times.argmin(-1)
residuals = times - times[:, amin[-1]].unsqueeze(-1)
residuals = residuals.clamp(max=0.0)
print(amin)
print(residuals.min(dim=-1).values)
print(residuals.min())
