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
    ],
    key=["N", "N_prime", "K", "D", "D_prime", "acc_dtype"],
)
@triton.jit
def implicit_conv3d_kernel(
    features,  # [N, D]
    indices,  # [K**3, N']
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

    ind_ptr = indices + (tl.arange(0, BLOCK_N) + pid_n * BLOCK_N)
    acc = tl.zeros((BLOCK_N, BLOCK_Dp), dtype=acc_dtype)
    for k_it in range(pid_k, stride_k, PARALLEL_K):
        k = k_it
        # [BLOCK_N]
        inds = tl.load(ind_ptr + k * N_prime, mask=(tl.arange(0, BLOCK_N) + pid_n * BLOCK_N) < N_prime, other=-1)
        # tl.device_print("inds", inds)
        if tl.reduce((0 <= inds) & (inds < N), 0, or_combine):
            for ki in range(tl.cdiv(D, BLOCK_K)):
                offset_k = k * D + ki * BLOCK_K

                ptr_f = features + (inds[:, None] * D + tl.arange(0, BLOCK_K)[None, :] + ki * BLOCK_K)
                mask_f = ((0 <= inds) & (inds < N))[:, None] & (tl.arange(0, BLOCK_K)[None, :] + ki * BLOCK_K < D)

                ptr_w = weights + (
                    (tl.arange(0, BLOCK_K)[:, None] + offset_k) * D_prime
                    + tl.arange(0, BLOCK_Dp)[None, :]
                    + pid_dp * BLOCK_Dp
                )
                mask_w = ((tl.arange(0, BLOCK_K)[:, None] + offset_k) < stride_k * D) & (
                    tl.arange(0, BLOCK_Dp)[None, :] + pid_dp * BLOCK_Dp < D_prime
                )

                feats = tl.load(ptr_f, mask_f, other=0.0)
                ws = tl.load(ptr_w, mask_w, other=0.0).to(feats.dtype)

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

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 16},  num_warps=2),
        triton.Config({"BLOCK_N": 32},  num_warps=4),
        triton.Config({"BLOCK_N": 64},  num_warps=4),
        triton.Config({"BLOCK_N": 128}, num_warps=8),
    ],
    key=["N", "K3", "BLOCK_K"],
)
@triton.jit
def implicit_gemm_idx_sort_kernel(
    indices, # [K**3, N]
    lin_mask, # [N]
    N,
    K3,
    BLOCK_K: tl.constexpr,  # kernel size
    mask_dtype: tl.constexpr, # int32 / int64
    BLOCK_N: tl.constexpr,  # tile size for N
):
    pid = tl.program_id(axis=0)

    offset_n = (pid * BLOCK_N + tl.arange(0, BLOCK_N))
    offset_k = tl.arange(0, BLOCK_K)
    idx_ptr = indices + ((offset_k[:, None] * N + offset_n[None, :]))
    idx_mask = (offset_k[:, None] < K3) & (offset_n[None, :] < N)
    # [BLOCK_K, BLOCK_N]
    inds = tl.load(idx_ptr, mask=idx_mask, other=-1)
    inds_density_mask = (inds >= 0) & (inds < N)
    inds_density = tl.full([1], 1, mask_dtype) << offset_k[:, None]
    inds_density = tl.where(inds_density_mask, inds_density, 0)
    inds_density = tl.reduce(inds_density, 0, or_combine)
    tl.store(
        lin_mask + offset_n,
        inds_density,
        mask=(offset_n < N),
    )

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_K": 16}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_K": 32}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_K": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_K": 32}, num_warps=4, num_stages=2),
    ],
    key=["N", "K", "BLOCK_N"],
)
@triton.jit
def implicit_gemm_mask_kernel(
    indices, # [K**3, N]
    mask, # [N', K**3]
    N,
    K: tl.constexpr,  # kernel size
    BLOCK_N: tl.constexpr,  # tile size for N
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    K3: tl.constexpr = K * K * K
    offset_n = (pid * BLOCK_N + tl.arange(0, BLOCK_N))
    offset_k = tl.arange(0, BLOCK_K)
    for i in range(0, K3, BLOCK_K):
        ptr_i = indices + ((offset_k + i)[:, None] * N + offset_n[None, :])
        # [BLOCK_K, BLOCK_N]
        inds = tl.load(ptr_i, mask=(offset_k + i < K3)[:, None] & (offset_n < N)[None, :], other=-1)
        imask = tl.reduce((inds < N) & (inds >= 0), 1, or_combine) # [BLOCK_K]
        mask_ptr = mask + (offset_k + i + pid * K3)
        tl.store(mask_ptr, imask, mask=(offset_k + i < K3))


@triton.autotune(
    configs=[
        # good for float16
        triton.Config({"BLOCK_K": 16, "BLOCK_Dp": 16 , 'PARALLEL_K': 1},  num_warps=2, num_stages=2),
        triton.Config({"BLOCK_K": 32, "BLOCK_Dp": 64 , 'PARALLEL_K': 1},  num_warps=2, num_stages=2),
        triton.Config({"BLOCK_K": 32, "BLOCK_Dp": 64 , 'PARALLEL_K': 1},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_K": 32, "BLOCK_Dp": 16 , 'PARALLEL_K': 1},  num_warps=2, num_stages=2),
        triton.Config({"BLOCK_K": 32, "BLOCK_Dp": 128, 'PARALLEL_K': 1}, num_warps=4,  num_stages=2),
        triton.Config({"BLOCK_K": 32, "BLOCK_Dp": 128, 'PARALLEL_K': 1}, num_warps=8,  num_stages=2),
        triton.Config({"BLOCK_K": 32, "BLOCK_Dp": 32, 'PARALLEL_K': 1}, num_warps=4,  num_stages=2),
        triton.Config({"BLOCK_K": 32, "BLOCK_Dp": 32, 'PARALLEL_K': 1}, num_warps=8,  num_stages=2),
        triton.Config({"BLOCK_K": 64, "BLOCK_Dp": 64 , 'PARALLEL_K': 1},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_K": 32, "BLOCK_Dp": 32 , 'PARALLEL_K': 1},  num_warps=2, num_stages=2),
        triton.Config({"BLOCK_K": 16, "BLOCK_Dp": 32 , 'PARALLEL_K': 1},  num_warps=2, num_stages=2),
        triton.Config({"BLOCK_K": 128, "BLOCK_Dp": 16 , 'PARALLEL_K': 1},  num_warps=2, num_stages=2),
        triton.Config({"BLOCK_K": 128, "BLOCK_Dp": 16 , 'PARALLEL_K': 1},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_K": 128, "BLOCK_Dp": 32 , 'PARALLEL_K': 1},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_K": 128, "BLOCK_Dp": 32 , 'PARALLEL_K': 1},  num_warps=8, num_stages=2),
    ],
    key=["N", "N_prime", "K", "D", "D_prime", "acc_dtype"],
)
@triton.jit
def implicit_conv3d_kernel_T(
    features,  # [N, D]
    indices,  # [K**3, N']
    mask_ind, # [NP, K**3]
    weights,  # [K**3, D, D']
    out_perm, # [N']
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
    sorted: tl.constexpr
):
    pid = tl.program_id(axis=0) // PARALLEL_K
    pid_k = tl.program_id(axis=0) % PARALLEL_K
    grid_n = tl.cdiv(N_prime, BLOCK_N)
    # TODO: potential optimization here: grid-rasterization for better l2-cache reuse
    pid_n = pid % grid_n
    pid_dp = pid // grid_n

    stride_k = K * K * K

    ind_ptr = indices + (tl.arange(0, BLOCK_N) + pid_n * BLOCK_N)
    acc = tl.zeros((BLOCK_N, BLOCK_Dp), dtype=acc_dtype)
    for k_it in range(pid_k, stride_k, PARALLEL_K):
        k = k_it
        # [BLOCK_N]
        mask_i = tl.load(mask_ind + k + pid_n * stride_k)
        if mask_i:
            # tl.device_print("inds", inds)
            inds = tl.load(ind_ptr + k * N_prime, mask=(tl.arange(0, BLOCK_N) + pid_n * BLOCK_N) < N_prime, other=-1)
            for ki in range(tl.cdiv(D, BLOCK_K)):
                offset_k = k * D + ki * BLOCK_K

                ptr_f = features + (inds[:, None] * D + tl.arange(0, BLOCK_K)[None, :] + ki * BLOCK_K)
                mask_f = ((0 <= inds) & (inds < N))[:, None] & (tl.arange(0, BLOCK_K)[None, :] + ki * BLOCK_K < D)

                ptr_w = weights + (
                    (tl.arange(0, BLOCK_K)[:, None] + offset_k) * D_prime
                    + tl.arange(0, BLOCK_Dp)[None, :]
                    + pid_dp * BLOCK_Dp
                )
                mask_w = ((tl.arange(0, BLOCK_K)[:, None] + offset_k) < stride_k * D) & (
                    tl.arange(0, BLOCK_Dp)[None, :] + pid_dp * BLOCK_Dp < D_prime
                )

                feats = tl.load(ptr_f, mask_f, other=0.0)
                ws = tl.load(ptr_w, mask_w, other=0.0).to(feats.dtype)

                acc += tl.dot(feats, ws, out_dtype=acc_dtype).to(acc_dtype)
    
    offsets_sort = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    mask_n = offsets_sort < N_prime
    if sorted:
        offsets_n = tl.load(out_perm + offsets_sort, mask=mask_n, other=-1)
    else:
        offsets_n = offsets_sort
        
    out_ptr = output + (
        offsets_n[:, None] * D_prime
        + tl.arange(0, BLOCK_Dp)[None, :]
        + pid_dp * BLOCK_Dp
    )
    out_mask = (offsets_n[:, None] < N_prime) & (
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



