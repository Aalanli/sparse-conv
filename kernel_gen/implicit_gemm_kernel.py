import triton
import triton.language as tl

@triton.jit
def or_combine(a, b):
    return a | b

@triton.autotune(
    configs=[
        # good for float16
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 16, "BLOCK_Dp": 16, "PARALLEL_K": 1}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 16, "BLOCK_Dp": 16, "PARALLEL_K": 1}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 32, "BLOCK_Dp": 64, "PARALLEL_K": 1}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 32, "BLOCK_Dp": 64, "PARALLEL_K": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 32, "BLOCK_Dp": 16, "PARALLEL_K": 1}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 32, "BLOCK_Dp": 128, "PARALLEL_K": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 32, "BLOCK_Dp": 128, "PARALLEL_K": 1}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 32, "BLOCK_Dp": 32, "PARALLEL_K": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 32, "BLOCK_Dp": 32, "PARALLEL_K": 1}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 64, "BLOCK_Dp": 64, "PARALLEL_K": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 32, "BLOCK_Dp": 32, "PARALLEL_K": 1}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 16, "BLOCK_Dp": 32, "PARALLEL_K": 1}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 128, "BLOCK_Dp": 16, "PARALLEL_K": 1}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 128, "BLOCK_Dp": 16, "PARALLEL_K": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 128, "BLOCK_Dp": 32, "PARALLEL_K": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 128, "BLOCK_Dp": 32, "PARALLEL_K": 1}, num_warps=8, num_stages=2),
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
    PARALLEL_K: tl.constexpr,  # whether to parallelize over K
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
    indices, # [K**3, N] maybe strided in N
    lin_mask, # [N]
    N,
    N_stride,
    K3,
    BLOCK_K: tl.constexpr,  # kernel size
    mask_dtype: tl.constexpr, # int32 / int64
    BLOCK_N: tl.constexpr,  # tile size for N
):
    pid = tl.program_id(axis=0)

    offset_n = (pid * BLOCK_N + tl.arange(0, BLOCK_N))
    offset_k = tl.arange(0, BLOCK_K)
    idx_ptr = indices + ((offset_k[:, None] * N_stride + offset_n[None, :]))
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
    stride_n,
    K: tl.constexpr,  # kernel size
    BLOCK_N: tl.constexpr,  # tile size for N
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    K3: tl.constexpr = K * K * K
    offset_n = (pid * BLOCK_N + tl.arange(0, BLOCK_N))
    offset_k = tl.arange(0, BLOCK_K)
    for i in range(0, K3, BLOCK_K):
        ptr_i = indices + ((offset_k + i)[:, None] * stride_n + offset_n[None, :])
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
    indices,  # [K**3, N'] may be strided in N'
    mask_ind, # [NP, K**3]
    weights,  # [K**3, D, D']
    out_perm, # [N']
    output,  # [N', D']
    N,
    N_prime,
    N_prime_stride,
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
            inds = tl.load(ind_ptr + k * N_prime_stride, mask=(tl.arange(0, BLOCK_N) + pid_n * BLOCK_N) < N_prime, other=-1)
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
    out_mask = ((offsets_n[:, None] < N_prime) & (offsets_n[:, None] >= 0)) & (
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


# out = F' @ W = F[indices] @ W
# out: [N', D']
# features: [N, D]
# indices: [K**3, N']
# weights: [K**3, D, D']
# 

# weights' = weights (K**3 * D, D')
# df = dout @ weights'^T
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_DPrime": 16, "BLOCK_NPrime": 16, "BLOCK_D": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_DPrime": 64, "BLOCK_NPrime": 32, "BLOCK_D": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_DPrime": 64, "BLOCK_NPrime": 16, "BLOCK_D": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_DPrime": 16, "BLOCK_NPrime": 32, "BLOCK_D": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_DPrime": 16, "BLOCK_NPrime": 64, "BLOCK_D": 32}, num_warps=4, num_stages=3),
    ],
    key=["N", "N_prime", "D", "D_prime", "acc_dtype"]
)
@triton.jit
def implicit_gemm_dF_kernel(
    dout,     # [N', D']
    weights,  # [K**3 * D, D']
    indices,  # [K**3, N'] may be strided in N'
    dfeatures,# [N, D]
    N,        # number of features
    N_prime,  # number of indices
    N_prime_stride, 
    D,        # feature dimension
    D_prime,  # output dimension
    BLOCK_DPrime: tl.constexpr, # the "k" reduction dimension
    BLOCK_NPrime: tl.constexpr,
    BLOCK_D: tl.constexpr,
    acc_dtype: tl.constexpr
):
    pid = tl.program_id(0)

    grid_np = tl.cdiv(N_prime, BLOCK_NPrime)
    pid_np = pid % grid_np
    pid_d = pid // grid_np

    blocks_per_d = tl.cdiv(D, BLOCK_D)

    acc = tl.zeros((BLOCK_NPrime, BLOCK_D), dtype=acc_dtype)
    offset_np = tl.arange(0, BLOCK_NPrime) + pid_np * BLOCK_NPrime
    offset_dp = tl.arange(0, BLOCK_DPrime)
    offset_d = tl.arange(0, BLOCK_D) + (pid_d % blocks_per_d) * BLOCK_D + (pid_d // blocks_per_d) * D
    for _ in range(0, tl.cdiv(D_prime, BLOCK_DPrime)):
        ptr_dout = dout + (offset_np[:, None] * D_prime + offset_dp[None, :])
        mask_dout = (offset_np[:, None] < N_prime) & (offset_dp[None, :] < D_prime)

        ptr_weights = weights + (offset_dp[:, None] + offset_d[None, :] * D_prime)
        mask_weights = (offset_dp[:, None] < D_prime) & \
            ((tl.arange(0, BLOCK_D)[None, :] + (pid_d % blocks_per_d) * BLOCK_D) < D)
        
        dout_v = tl.load(ptr_dout, mask_dout, other=0.0)
        W = tl.load(ptr_weights, mask_weights, other=0.0).to(dout_v.dtype)

        acc += tl.dot(dout_v, W, out_dtype=acc_dtype).to(acc_dtype)
        offset_dp += BLOCK_DPrime
    
    kprime = (pid_d // blocks_per_d)
    inds_ptr = indices + offset_np + kprime * N_prime_stride
    inds = tl.load(inds_ptr, mask=(offset_np < N_prime), other=-1)
    inds_mask = (inds >= 0) & (inds < N)

    offset_d_out = tl.arange(0, BLOCK_D) + (pid_d % blocks_per_d) * BLOCK_D
    df_ptr = dfeatures + (inds[:, None] * D + offset_d_out[None, :])
    df_mask = inds_mask[:, None] & (offset_d_out[None, :] < D)

    tl.atomic_add(df_ptr, acc.to(dfeatures.dtype.element_ty), df_mask)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_DPrime": 16, "BLOCK_NPrime": 16, "BLOCK_D": 32, "PARALLEL_K": 1}, num_warps=4, num_stages=3),
    ],
    key=["N", "N_prime", "D", "D_prime", "acc_dtype"]
)
@triton.jit
def implicit_gemm_dW_kernel(
    dout,  # [N', D']
    features, # [N, D]
    indices, # [K**3, N'] may be strided in N'
    dweight, # [K**3, D, D']
    N, 
    N_prime,
    N_prime_stride,
    D,
    D_prime,
    K3,
    BLOCK_NPrime: tl.constexpr, # "k" reduction dimension
    BLOCK_DPrime: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PARALLEL_K: tl.constexpr,
    acc_dtype: tl.constexpr
):
    grid_dp = tl.cdiv(D_prime, BLOCK_DPrime)
    grid_d = tl.cdiv(D, BLOCK_D) * K3
    grid_size = grid_dp * grid_d
    id = tl.program_id(0)
    kid = id // grid_size
    pid = id % grid_size
    

    pid_dp = pid % grid_dp
    pid_d = pid // grid_dp

    blocks_per_d = tl.cdiv(D, BLOCK_D)

    acc = tl.zeros((BLOCK_DPrime, BLOCK_D), dtype=acc_dtype)
    offset_dp = tl.arange(0, BLOCK_DPrime) + pid_dp * BLOCK_DPrime
    offset_d = tl.arange(0, BLOCK_D) + (pid_d % blocks_per_d) * BLOCK_D
    
    for k in range(kid, tl.cdiv(N_prime, BLOCK_NPrime), PARALLEL_K):
        offset_np_ind = tl.arange(0, BLOCK_NPrime) + k * BLOCK_NPrime
        dout_ptr = dout + (offset_dp[:, None] + offset_np_ind[None, :] * D_prime)
        dout_mask = (offset_dp[:, None] < D_prime) & (offset_np_ind[None, :] < N_prime)

        inds_ptr = indices + (offset_np_ind + (pid_d // blocks_per_d) * N_prime_stride)
        inds = tl.load(inds_ptr, offset_np_ind < N_prime, other=-1)
        
        f_ptr = features + (inds[:, None] * D + offset_d[None, :])
        f_mask = ((inds >= 0) & (inds < N))[:, None] & (offset_d < D[None, :])
        
        dout_v = tl.load(dout_ptr, dout_mask, other=0.0)
        f_v = tl.load(f_ptr, f_mask, other=0.0)

        acc += tl.dot(dout_v, f_v, out_dtype=acc_dtype).to(acc_dtype)

    acc_T = tl.trans(acc)
    offset_dp = tl.arange(0, BLOCK_DPrime) + pid_dp * BLOCK_DPrime
    offset_d = tl.arange(0, BLOCK_D) + (pid_d % blocks_per_d) * BLOCK_D + (pid_d // blocks_per_d) * D
    dweight_ptr = dweight + (offset_d[:, None] * D_prime + offset_dp[None, :])
    mask_dweight = (tl.arange(0, BLOCK_D) + (pid_d % blocks_per_d) * BLOCK_D < D)[:, None] &\
        (offset_dp < D_prime)[None, :]
    
    tl.store(dweight_ptr, acc_T, mask_dweight)


