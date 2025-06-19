import triton
import triton.language as tl

@triton.jit
def or_combine(a, b):
    return a | b



@triton.autotune(
    configs=[
        # good for float16
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 16, "BLOCK_Dp": 16 , 'PARALLEL_K': 1},  num_warps=2, num_stages=2),
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 16, "BLOCK_Dp": 16 , 'PARALLEL_K': 1},  num_warps=2, num_stages=2),
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 32, "BLOCK_Dp": 64 , 'PARALLEL_K': 1},  num_warps=2, num_stages=2),
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 32, "BLOCK_Dp": 64 , 'PARALLEL_K': 1},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 32, "BLOCK_Dp": 16 , 'PARALLEL_K': 1},  num_warps=2, num_stages=2),
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 32, "BLOCK_Dp": 128, 'PARALLEL_K': 1}, num_warps=4,  num_stages=2),
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 32, "BLOCK_Dp": 128, 'PARALLEL_K': 1}, num_warps=8,  num_stages=2),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 32, "BLOCK_Dp": 32, 'PARALLEL_K': 1}, num_warps=4,  num_stages=2),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 32, "BLOCK_Dp": 32, 'PARALLEL_K': 1}, num_warps=8,  num_stages=2),
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 64, "BLOCK_Dp": 64 , 'PARALLEL_K': 1},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 32, "BLOCK_Dp": 32 , 'PARALLEL_K': 1},  num_warps=2, num_stages=2),
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 16, "BLOCK_Dp": 32 , 'PARALLEL_K': 1},  num_warps=2, num_stages=2),
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 128, "BLOCK_Dp": 16 , 'PARALLEL_K': 1},  num_warps=2, num_stages=2),
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 128, "BLOCK_Dp": 16 , 'PARALLEL_K': 1},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 128, "BLOCK_Dp": 32 , 'PARALLEL_K': 1},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 128, "BLOCK_Dp": 32 , 'PARALLEL_K': 1},  num_warps=8, num_stages=2),

        # triton.Config({"BLOCK_N": 32, "BLOCK_K": 16, "BLOCK_Dp": 16 , 'PARALLEL_K': 2},  num_warps=2, num_stages=2),
        # triton.Config({"BLOCK_N": 16, "BLOCK_K": 16, "BLOCK_Dp": 16 , 'PARALLEL_K': 2},  num_warps=2, num_stages=2),
        # triton.Config({"BLOCK_N": 16, "BLOCK_K": 32, "BLOCK_Dp": 64 , 'PARALLEL_K': 2},  num_warps=4, num_stages=3),
        # triton.Config({"BLOCK_N": 32, "BLOCK_K": 32, "BLOCK_Dp": 16 , 'PARALLEL_K': 2},  num_warps=2, num_stages=2),
        # triton.Config({"BLOCK_N": 16, "BLOCK_K": 32, "BLOCK_Dp": 128, 'PARALLEL_K': 2}, num_warps=8, num_stages=3),
        # triton.Config({"BLOCK_N": 128, "BLOCK_K": 32, "BLOCK_Dp": 32, 'PARALLEL_K': 2}, num_warps=4, num_stages=2),
        # triton.Config({"BLOCK_N": 16, "BLOCK_K": 64, "BLOCK_Dp": 64 , 'PARALLEL_K': 2},  num_warps=4, num_stages=3),
        # triton.Config({"BLOCK_N": 16, "BLOCK_K": 32, "BLOCK_Dp": 32 , 'PARALLEL_K': 2},  num_warps=2, num_stages=2),
        # triton.Config({"BLOCK_N": 32, "BLOCK_K": 16, "BLOCK_Dp": 32 , 'PARALLEL_K': 2},  num_warps=2, num_stages=2),

        # triton.Config({"BLOCK_N": 32, "BLOCK_K": 16, "BLOCK_Dp": 16 , 'PARALLEL_K': 3},  num_warps=2, num_stages=2),
        # triton.Config({"BLOCK_N": 16, "BLOCK_K": 16, "BLOCK_Dp": 16 , 'PARALLEL_K': 3},  num_warps=2, num_stages=2),
        # triton.Config({"BLOCK_N": 16, "BLOCK_K": 32, "BLOCK_Dp": 64 , 'PARALLEL_K': 3},  num_warps=4, num_stages=3),
        # triton.Config({"BLOCK_N": 32, "BLOCK_K": 32, "BLOCK_Dp": 16 , 'PARALLEL_K': 3},  num_warps=2, num_stages=2),
        # triton.Config({"BLOCK_N": 16, "BLOCK_K": 32, "BLOCK_Dp": 128, 'PARALLEL_K': 3}, num_warps=8, num_stages=3),
        # triton.Config({"BLOCK_N": 128, "BLOCK_K": 32, "BLOCK_Dp": 32, 'PARALLEL_K': 3}, num_warps=4, num_stages=2),
        # triton.Config({"BLOCK_N": 16, "BLOCK_K": 64, "BLOCK_Dp": 64 , 'PARALLEL_K': 3},  num_warps=4, num_stages=3),
        # triton.Config({"BLOCK_N": 16, "BLOCK_K": 32, "BLOCK_Dp": 32 , 'PARALLEL_K': 3},  num_warps=2, num_stages=2),
        # triton.Config({"BLOCK_N": 32, "BLOCK_K": 16, "BLOCK_Dp": 32 , 'PARALLEL_K': 3},  num_warps=2, num_stages=2),
    ],
    key=["N", "N_prime", "K", "D", "D_prime", "acc_dtype"],
)
@triton.jit
def implicit_conv3d_kernel(
    features,  # [N, D]
    indices,  # [N', K**3]
    weights,  # [K**3, D, D']
    output,  # [N', D']
    N,
    N_prime,
    D,
    D_prime,
    K,
    BLOCK_N: tl.constexpr,  # tile size for N
    BLOCK_K: tl.constexpr,  # tile size for K
    BLOCK_Dp: tl.constexpr,  # tile size for D
    PARALLEL_K: tl.constexpr, # whether to parallelize over K
    acc_dtype: tl.constexpr,
):
    pid = tl.program_id(axis=0) // PARALLEL_K
    pid_k = tl.program_id(axis=0) % PARALLEL_K
    grid_n = tl.cdiv(N_prime, BLOCK_N)
    # TODO: potential optimization here: grid-rasterization for better l2-cache reuse
    pid_n = pid % grid_n
    pid_dp = pid // grid_n

    stride_k = K * K * K

    ind_ptr = indices + (tl.arange(0, BLOCK_N) + pid_n * BLOCK_N) * stride_k
    acc = tl.zeros((BLOCK_N, BLOCK_Dp), dtype=acc_dtype)
    for k_it in range(pid_k, stride_k, PARALLEL_K):
        k = k_it
        # [BLOCK_N]
        inds = tl.load(ind_ptr + k, mask=(tl.arange(0, BLOCK_N) + pid_n * BLOCK_N) < N_prime, other=-1)
        # tl.device_print("inds", inds)
        if tl.reduce((0 <= inds) & (inds < N), 0, or_combine):
            for ki in range(tl.cdiv(D, BLOCK_K)):
                offset_k = k * D + ki * BLOCK_K

                ptr_f = features + (inds[:, None] * D + tl.arange(0, BLOCK_K)[None, :] + ki * BLOCK_K)
                mask_f = (inds[:, None] != -1) & (tl.arange(0, BLOCK_K)[None, :] + ki * BLOCK_K < D)

                ptr_w = weights + (
                    (tl.arange(0, BLOCK_K)[:, None] + offset_k) * D_prime
                    + tl.arange(0, BLOCK_Dp)[None, :]
                    + pid_dp * BLOCK_Dp
                )
                mask_w = ((tl.arange(0, BLOCK_K)[:, None] + offset_k) < stride_k * D) & (
                    tl.arange(0, BLOCK_Dp)[None, :] + pid_dp * BLOCK_Dp < D_prime
                )

                feats = tl.load(ptr_f, mask_f, other=0.0).to(acc_dtype)
                ws = tl.load(ptr_w, mask_w, other=0.0).to(acc_dtype)

                acc += tl.dot(feats, ws, out_dtype=acc_dtype).to(acc_dtype)
    out_ptr = output + (
        (tl.arange(0, BLOCK_N)[:, None] + pid_n * BLOCK_N) * D_prime
        + tl.arange(0, BLOCK_Dp)[None, :]
        + pid_dp * BLOCK_Dp
    )
    out_mask = (tl.arange(0, BLOCK_N)[:, None] + pid_n * BLOCK_N < N_prime) & (
        tl.arange(0, BLOCK_Dp)[None, :] + pid_dp * BLOCK_Dp < D_prime
    )
    if PARALLEL_K == 1:
        tl.store(out_ptr, acc.to(out_ptr.dtype.element_ty), out_mask)
    else:
        tl.atomic_add(
            out_ptr,
            acc.to(out_ptr.dtype.element_ty),
            out_mask,
        )
