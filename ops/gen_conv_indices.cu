#include "torch/types.h"
#include <cuda.h>

#include <cassert>
#include <cstdint>
#include <cub/block/block_scan.cuh>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/script.h>

__device__ __host__ uint64_t pack_coords_device(const int *coord) {
    uint64_t packed = 0;
    int4 coord4 = ((int4 *)coord)[0];  // Assuming coord is aligned to int4
    packed |= static_cast<uint64_t>(coord4.x) << 48;
    packed |= static_cast<uint64_t>(coord4.y) << 32;
    packed |= static_cast<uint64_t>(coord4.z) << 16;
    packed |= static_cast<uint64_t>(coord4.w);
    return packed;
}

/*
there are two types of sparse convolution: sub-manifold and regular, subm convolution does not change the number of
coordinates, while regular convolution does. Intuitively, subm convolution convolves the filter only at the center of
non-zero components. While regular convolution mimicks the dense version.

For example (simplified to 1D case)

Input coordinates: [2]
(dense) Tensor: [0, 0, x, 0, 0]
Filter: [a, b, c]

subm:
    output = [0, 0, b*x, 0, 0]
    out_coords = [2]

pad = 0, stride = 1
regular:
    output = [c * x, b * x, a * x]
    out_coords = [0, 1, 2]
*/

extern "C" int64_t generate_conv3d_indices_cpu(const int *coords,  // (N, 4)
                                               int *new_coords,    // (N', 4)
                                               int *indices,       // (N', K ** 3)
                                               const int64_t N,    // number of coordinates
                                               const int64_t K,    // kernel size
                                               const int3 stride, const int3 pad, const int3 bounds) {
    // N' = N * K ** 3 upper bound of output size
    int64_t output_size = 0;
    std::unordered_map<int64_t, int> output_map;

    for (int64_t i = 0; i < N; ++i) {
        int b, x, y, z;
        b = coords[i * 4];
        x = coords[i * 4 + 1];
        y = coords[i * 4 + 2];
        z = coords[i * 4 + 3];

        // y[b, i, j, k] = x[b, i * sx + dx - pad, j * sy + dy - pad, k * sz + dz - pad] * w[dx, dy, dz]
        // exists i, j, k such that
        // i * sx + dx - pad == x, etc.
        for (int dx = 0; dx < K; ++dx) {
            for (int dy = 0; dy < K; ++dy) {
                for (int dz = 0; dz < K; ++dz) {
                    bool exists = true;
                    exists &= (x + pad.x - dx) % stride.x == 0 && x + pad.x - dx >= 0;
                    exists &= (y + pad.y - dy) % stride.y == 0 && y + pad.y - dy >= 0;
                    exists &= (z + pad.z - dz) % stride.z == 0 && z + pad.z - dz >= 0;
                    if (!exists) continue;
                    int new_x = (x + pad.x - dx) / stride.x;
                    int new_y = (y + pad.y - dy) / stride.y;
                    int new_z = (z + pad.z - dz) / stride.z;
                    exists &= new_x * stride.x + K - 1 < bounds.x + 2 * pad.x;
                    exists &= new_y * stride.y + K - 1 < bounds.y + 2 * pad.y;
                    exists &= new_z * stride.z + K - 1 < bounds.z + 2 * pad.z;
                    exists &= !(new_x < 0 || new_y < 0 || new_z < 0 || new_x >= bounds.x || new_y >= bounds.y ||
                                new_z >= bounds.z);
                    if (!exists) {
                        continue;  // out of bounds
                    }
                    int new_coord[4] = {b, new_x, new_y, new_z};
                    auto packed_new_coord = pack_coords_device(new_coord);
                    auto ks = dz + K * (dy + K * dx);
                    if (output_map.count(packed_new_coord) == 0) {
                        output_map[packed_new_coord] = output_size;
                        new_coords[output_size * 4] = b;
                        new_coords[output_size * 4 + 1] = new_x;
                        new_coords[output_size * 4 + 2] = new_y;
                        new_coords[output_size * 4 + 3] = new_z;
                        output_size += 1;
                    }
                    int64_t output_index = output_map[packed_new_coord];
                    indices[output_index * (K * K * K) + ks] = i;
                }
            }
        }
    }

    return output_size;
}

extern "C" void generate_conv3d_subm_indices_cpu(const int *coords,  // (N, 4)
                                                 int *indices,       // (N, K ** 3)
                                                 int64_t N,          // number of coordinates
                                                 int64_t K           // kernel size
) {
    std::unordered_map<int64_t, int> coord_map;
    for (int64_t i = 0; i < N; ++i) {
        int64_t packed_coord = pack_coords_device(coords + i * 4);
        if (coord_map.find(packed_coord) == coord_map.end()) {
            coord_map[packed_coord] = i;
        }
    }

    for (int i = 0; i < N; ++i) {
        int b, x, y, z;
        b = coords[i * 4];
        x = coords[i * 4 + 1];
        y = coords[i * 4 + 2];
        z = coords[i * 4 + 3];
        int offset = 0;
        for (int dx = 0; dx < K; ++dx) {
            for (int dy = 0; dy < K; ++dy) {
                for (int dz = 0; dz < K; ++dz) {
                    int new_x = x + dx - K / 2;
                    int new_y = y + dy - K / 2;
                    int new_z = z + dz - K / 2;

                    int new_coord[4] = {b, new_x, new_y, new_z};
                    int64_t packed_new_coord = pack_coords_device(new_coord);

                    if (coord_map.find(packed_new_coord) != coord_map.end()) {
                        indices[i * K * K * K + offset] = coord_map[packed_new_coord];
                    } else {
                        indices[i * K * K * K + offset] = -1;
                    }
                    offset++;
                }
            }
        }
    }
}

__device__ __host__ uint64_t fmix64(uint64_t h) {
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

constexpr unsigned long long EMPTY_HASH = 0xffffffffffffffffULL;  // -1 in unsigned long long

__global__ void insert_indices_hash_kernel(const int *coords,   // (N, 4)
                                           uint64_t *hash_key,  // (T,)
                                           int *hash_value,     // (T,)
                                           int N,               // number of coordinates
                                           int T,                // kernel size
                                           int max_lookup,
                                           int *dropped_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int4 coord4 = ((int4 *)coords)[idx];
    uint64_t packed_coord = pack_coords_device(&coord4.x);
    auto hash = fmix64(packed_coord);

    int hash_index = hash % T;
    for (int l = 0; l < max_lookup; ++l) {
        auto hkey = atomicCAS((unsigned long long *)hash_key + hash_index, EMPTY_HASH, packed_coord);
        if (hkey == packed_coord) {
            return;
        } else if (hkey == EMPTY_HASH) {
            // if there are duplicate indices, then this becomes non-deterministic
            hash_value[hash_index] = idx;
            return;
        }
        hash_index = (hash_index + 1) % T;  // Linear probing
    }
    atomicAdd(dropped_points, 1);
}

__device__ __forceinline__ int lookup_hashtable(
    uint64_t val,
    volatile uint64_t *hash_key,  // (T,)
    int *hash_value,     // (T,)
    int T,               // table size
    int lookup_tries
) {
    auto hash = fmix64(val);
    int hash_index = hash % T;

    // bound the number of lookups
    for (int l = 0; l < lookup_tries; l++) {
        auto hash_key_val = hash_key[hash_index];
        if (hash_key_val == val) {  // found
            return hash_value[hash_index];
        } else if (hash_key_val == EMPTY_HASH) {  // not found
            return -1;  // Not found
        }
        hash_index = (hash_index + 1) % T;  // Linear probing
    }
    return -2;  // Not found after max lookups
}

__global__ void generate_conv3d_subm_indices_kernel(const int *coords,
                                                    uint64_t *hash_key,  // (T,)
                                                    int *hash_value,     // (T,)
                                                    int *indices,        // (N, K ** 3)
                                                    const int N,         // number of coordinates
                                                    const int T,         // table size
                                                    const int K,          // kernel size
                                                    const int lookup_tries,
                                                    int* dropped_points
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx = tid / (K * K * K);
    const int offset = tid % (K * K * K);
    if (idx >= N) return;

    int4 coord4 = ((int4 *)coords)[idx];
    int b = coord4.x;
    int x = coord4.y;
    int y = coord4.z;
    int z = coord4.w;

    const int dz = offset % K;
    const int dy = (offset / K) % K;
    const int dx = offset / (K * K);
    int new_x = x + dx - K / 2;
    int new_y = y + dy - K / 2;
    int new_z = z + dz - K / 2;
    if (new_x < 0 || new_y < 0 || new_z < 0) {
        // indices[idx * K * K * K + offset] = -1;  // Not found
        return;                                  // Skip out of bounds
    }

    int new_coord[4] = {b, new_x, new_y, new_z};
    uint64_t packed_new_coord = pack_coords_device(new_coord);

    int found_index = lookup_hashtable(packed_new_coord, hash_key, hash_value, T, lookup_tries);
    if (found_index == -2) {
        atomicAdd(dropped_points, 1);
    } else {
        indices[idx * K * K * K + offset] = found_index;
    }
}

extern "C" void generate_conv3d_subm_indices_gpu(const int *coords,  // (N, 4)
                                                 int64_t *hash_key,  // (T,)
                                                 int *hash_value,    // (T,)
                                                 int *indices,       // (N, K ** 3)
                                                 int* dropped_points, // (2,) zero
                                                 int N,              // number of coordinates
                                                 int K,              // kernel size
                                                 int T, 
                                                 int lookup_tries, int threads, CUstream stream) {
    int threads_per_block = threads;
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    insert_indices_hash_kernel<<<blocks, threads_per_block, 0, stream>>>(coords, (uint64_t *)hash_key, hash_value, N,
                                                                         T, lookup_tries, dropped_points);
    int blocks_subm = (N * K * K * K + threads_per_block - 1) / threads_per_block;
    generate_conv3d_subm_indices_kernel<<<blocks_subm, threads_per_block, 0, stream>>>(coords, (uint64_t *)hash_key,
                                                                                       hash_value, indices, N, T, K, lookup_tries, dropped_points + 1);
}

template <typename F>
__device__ void printv(F func) {
    for (int t = 0; t < blockDim.x; ++t) {
        if (t == threadIdx.x) {
            printf("tid[%d]=", t);
            func();
            printf("\n");
        }
        __syncthreads();
    }
}

template <int THREADS>
__global__ void generate_conv3d_indices_kernel_one_stage(const int *coords,  // (N, 4)
                                                         int *new_coords,    // (N', 4)
                                                         int *indices,       // (N', K * K * K)

                                                         int *global_offset,         // (1,) init = 0
                                                         uint64_t *hash_keys,        // (T,) init = -1
                                                         volatile int *hash_values,  // (T,) init = -1

                                                         const int N,  // number of coordinates
                                                         const int Nprime,
                                                         const int K,  // kernel size
                                                         const int T, const int3 stride, const int3 pad,
                                                         const int3 bounds,
                                                         const int lookup_tries,
                                                         int* dropped_points) {
    const int KerStride = K * K * K;
    const int tid = (blockIdx.x * blockDim.x + threadIdx.x);
    const int idx = tid / KerStride;
    const int offset = tid % KerStride;
    const int dx = offset / (K * K);
    const int dy = offset / K % K;
    const int dz = offset % K;

    int4 coord4 = (idx < N) ? ((int4 *)coords)[idx] : int4();
    int b = coord4.x;
    int x = coord4.y;
    int y = coord4.z;
    int z = coord4.w;
    bool exists = idx < N;
    exists &= (x + pad.x - dx) % stride.x == 0 && x + pad.x - dx >= 0;
    exists &= (y + pad.y - dy) % stride.y == 0 && y + pad.y - dy >= 0;
    exists &= (z + pad.z - dz) % stride.z == 0 && z + pad.z - dz >= 0;
    int new_x = (x + pad.x - dx) / stride.x;
    int new_y = (y + pad.y - dy) / stride.y;
    int new_z = (z + pad.z - dz) / stride.z;
    exists &= !(new_x < 0 || new_y < 0 || new_z < 0 || new_x >= bounds.x || new_y >= bounds.y || new_z >= bounds.z);
    // this boundary condition may drop points unexpectedly (without the correct pad)
    // this is to replicate torchsparse behavior
    exists &= new_x * stride.x + K - 1 < bounds.x + 2 * pad.x;
    exists &= new_y * stride.y + K - 1 < bounds.y + 2 * pad.y;
    exists &= new_z * stride.z + K - 1 < bounds.z + 2 * pad.z;
    // printv([=]() {printf("exists %d (%d, %d, %d)", exists, new_x, new_y, new_z);});
    int new_coord[4] = {b, new_x, new_y, new_z};
    auto packed_new_coord = pack_coords_device(new_coord);
    // optimization opportunity here to first write to shared
    auto hash = fmix64(packed_new_coord);
    int hash_idx = hash % T;
    uint64_t hkey = 0;
    if (exists) {  // skip for all items that does not exist
        int lookups = 0;
        while (lookups < lookup_tries) {
            hkey = atomicCAS((unsigned long long *)hash_keys + hash_idx, EMPTY_HASH, packed_new_coord);
            if (hkey == packed_new_coord || hkey == EMPTY_HASH) {
                break;
            }
            hash_idx = (hash_idx + 1) % T;
            lookups += 1;
        }

        if (lookups >= lookup_tries) {
            atomicAdd(dropped_points, 1);
            exists = false;
        }
    }
    // if hkey == EMPTY_HASH, then this thread inserts into the hashmap
    __shared__ int local_offset_shared;
    // a thread that does not "Exist" can never trigger
    int local_offset = (hkey == EMPTY_HASH) ? 1 : 0;

    int offset_final;
    using blockscan = cub::BlockScan<int, THREADS>;
    __shared__ typename blockscan::TempStorage temp_storage;
    blockscan(temp_storage).ExclusiveSum(local_offset, offset_final);
    // printv([=]() { printf("%d, %d", local_offset, offset_final); });
    if (threadIdx.x == blockDim.x - 1) {
        local_offset_shared = atomicAdd(global_offset, offset_final + local_offset);
    }
    __syncthreads();
    offset_final += local_offset_shared;
    if (hkey == EMPTY_HASH) {
        hash_values[hash_idx] = offset_final;
    } else if (exists) {
        int lookups = 0;
        while (lookups < lookup_tries) {
            offset_final = hash_values[hash_idx];
            if (offset_final != -1) {
                break;
            }
            lookups += 1;
        }
        if (lookups >= lookup_tries) {
            exists = false; // exceeded maximum lookups, do not write 
            atomicAdd(dropped_points, 1);
        }
    }
    // printv([=]() {printf("offset_final %d, idx  %d, offset %d", offset_final, idx, offset);});
    if (offset_final < Nprime && exists && idx < N) {
        indices[offset_final * KerStride + offset] = idx;
        ((int4 *)new_coords)[offset_final] = *(int4 *)new_coord;
    }
}

extern "C" void generate_conv3d_indices_kernel_gpu(const int *coords,  // (N, 4)
                                                   int *new_coords,    // (N', 4)
                                                   int *indices,       // (N', K * K * K)

                                                   int *global_offset,   // (1,) init = 0
                                                   uint64_t *hash_keys,  // (T,) init = -1
                                                   int *hash_values,     // (T,) init = -1

                                                   const int N,  // number of coordinates
                                                   const int Nprime,
                                                   const int K,  // kernel size
                                                   const int T, const int3 stride, const int3 pad, const int3 bounds, const int lookup_tries,
                                                   int* dropped_points, // (1,)
                                                   const int64_t threads, CUstream stream) {
    int blocks = (N * K * K * K + threads - 1) / threads;

    switch (threads) {
        case 128:
            generate_conv3d_indices_kernel_one_stage<128>
                <<<blocks, threads, 0, stream>>>(coords, new_coords, indices, global_offset, hash_keys, hash_values, N,
                                                 Nprime, K, T, stride, pad, bounds, lookup_tries, dropped_points);
            break;
        case 256:
            generate_conv3d_indices_kernel_one_stage<256>
                <<<blocks, threads, 0, stream>>>(coords, new_coords, indices, global_offset, hash_keys, hash_values, N,
                                                 Nprime, K, T, stride, pad, bounds, lookup_tries, dropped_points);
            break;
        case 512:
            generate_conv3d_indices_kernel_one_stage<512>
                <<<blocks, threads, 0, stream>>>(coords, new_coords, indices, global_offset, hash_keys, hash_values, N,
                                                 Nprime, K, T, stride, pad, bounds, lookup_tries, dropped_points);
            break;
        case 1024:
            generate_conv3d_indices_kernel_one_stage<1024>
                <<<blocks, threads, 0, stream>>>(coords, new_coords, indices, global_offset, hash_keys, hash_values, N,
                                                 Nprime, K, T, stride, pad, bounds, lookup_tries, dropped_points);
            break;
        default:
            std::cout << "Invalid thread size, no kernels launched" << std::endl;
            break;
    }
}


template <int K>
__launch_bounds__(K*32) __global__ void generate_conv3d_subm_indices_kernel_3(const int *coords,
                                                    uint64_t *hash_key,  // (T,)
                                                    int *hash_value,     // (T,)
                                                    int *indices,        // (K ** 3, N)
                                                    const int N,         // number of coordinates
                                                    const int T,         // table size
                                                    const int lookup_tries,
                                                    int* dropped_points
) {
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    constexpr unsigned int items = 32;
    __shared__ int4 coords_s[items]; // [items]

    const int ni = lane_id + blockIdx.x * items;
    if (ni < N && warp_id == 0) {
        coords_s[lane_id] = ((int4*) &coords)[ni];
    } else if (ni >= N) {
        return;
    }
    __syncthreads();
    int4 coord4 = coords_s[lane_id];

    int b = coord4.x;
    int x = coord4.y;
    int y = coord4.z;
    int z = coord4.w;

    #pragma unroll
    for (int offset = warp_id; offset < K * K * K; offset += K) {
        const int dz = offset % K;
        const int dy = (offset / K) % K;
        const int dx = offset / (K * K);
        int new_x = x + dx - K / 2;
        int new_y = y + dy - K / 2;
        int new_z = z + dz - K / 2;
        
        if (new_x < 0 || new_y < 0 || new_z < 0) {
            // indices[idx * K * K * K + offset] = -1;  // Not found
            continue;                                  // Skip out of bounds
        }
    
        int new_coord[4] = {b, new_x, new_y, new_z};
        uint64_t packed_new_coord = pack_coords_device(new_coord);
    
        int found_index = lookup_hashtable(packed_new_coord, hash_key, hash_value, T, lookup_tries);
        if (found_index == -2) {
            atomicAdd(dropped_points, 1);
        } else {
            indices[offset * N + ni] = found_index;
        }
    }
}


torch::Tensor generate_conv3d_subm_indicesV2(torch::Tensor &coords, int64_t K, double hash_table_multiplier, int64_t threads, int64_t lookup_tries) {
    TORCH_CHECK(coords.dim() == 2 && coords.size(1) == 4, "coords must be of shape (N, 4)");
    TORCH_CHECK(coords.dtype() == torch::kInt32, "coords must be of dtype int32");
    TORCH_CHECK(K > 0, "Kernel size K must be positive");
    TORCH_CHECK(K % 2 == 1, "Subm kernel size K must be odd");

    int64_t N = coords.size(0);
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(coords.device());
    auto indices = torch::empty({K * K * K, N}, options);
    
    if (coords.device().is_cuda()) {
        at::cuda::CUDAGuard device_guard(coords.device());

        // Call the GPU implementation
        int T = static_cast<int>(N * hash_table_multiplier);
        TORCH_CHECK(hash_table_multiplier > 1, "Hash table multiplier must be > 1");
        TORCH_CHECK(T > N * 2, "Hash table size must be greater than number of coordinates * 2");

        auto dropped_points = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(coords.device()));
        indices.fill_(-1);

        auto hash_keys = torch::empty({T}, torch::TensorOptions().dtype(torch::kInt64).device(coords.device()));
        hash_keys.fill_(-1);
        auto hash_values = torch::empty({T}, torch::TensorOptions().dtype(torch::kInt32).device(coords.device()));
        hash_values.fill_(-1);

        auto stream = at::cuda::getCurrentCUDAStream().stream();
        int threads_per_block = threads;
        int blocks = (N + threads_per_block - 1) / threads_per_block;
        insert_indices_hash_kernel<<<blocks, threads_per_block, 0, stream>>>((int*) coords.data_ptr(), (uint64_t *)hash_keys.data_ptr(), 
                                        (int*) hash_values.data_ptr(), N,
                                                                                T, lookup_tries, (int*) dropped_points.data_ptr());

        if (K == 3) {
            int threads = 32 * K;
            generate_conv3d_subm_indices_kernel_3<3><<<((N + 32 - 1) / 32), threads, 0, stream>>>(
                (int *)coords.data_ptr(), (uint64_t *)hash_keys.data_ptr(), (int *)hash_values.data_ptr(),
                (int *)indices.data_ptr(), N, T, lookup_tries, (int*) dropped_points.data_ptr()
            );
        } else {
            throw std::runtime_error("Only K=3 is supported in generate_conv3d_subm_indicesV2");
        }

        if (dropped_points[0].item<int>() != 0) {
            TORCH_CHECK(false, "Error: dropped points detected in generate_conv3d_subm_indicesV2 (hash table overflow or collision).");
        }
    } else {
        throw std::runtime_error("CPU implementation is not supported for generate_conv3d_subm_indicesV2");
    }

    return indices;
}

// returned indices that are null are filled with -1
torch::Tensor generate_conv3d_subm_indices(const torch::Tensor &coords,  // (N, 4)
                                           int64_t K,                    // kernel size
                                           double hash_table_multiplier,
                                            int64_t threads, 
                                        int64_t lookup_tries) {
    TORCH_CHECK(coords.dim() == 2 && coords.size(1) == 4, "coords must be of shape (N, 4)");
    TORCH_CHECK(coords.dtype() == torch::kInt32, "coords must be of dtype int32");
    TORCH_CHECK(K > 0, "Kernel size K must be positive");
    TORCH_CHECK(K % 2 == 1, "Subm kernel size K must be odd");

    int64_t N = coords.size(0);
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(coords.device());
    auto indices = torch::empty({N, K * K * K}, options);

    if (coords.device().is_cuda()) {
        at::cuda::CUDAGuard device_guard(coords.device());

        // Call the GPU implementation
        int T = static_cast<int>(N * hash_table_multiplier);
        TORCH_CHECK(hash_table_multiplier > 1, "Hash table multiplier must be > 1");
        TORCH_CHECK(T > N * 2, "Hash table size must be greater than number of coordinates * 2");

        auto dropped_points = torch::zeros({2}, torch::TensorOptions().dtype(torch::kInt32).device(coords.device()));
        indices.fill_(-1);
        auto hash_keys = torch::empty({T}, torch::TensorOptions().dtype(torch::kInt64).device(coords.device()));
        hash_keys.fill_(-1);
        auto hash_values = torch::empty({T}, torch::TensorOptions().dtype(torch::kInt32).device(coords.device()));
        generate_conv3d_subm_indices_gpu(coords.data_ptr<int>(), hash_keys.data_ptr<int64_t>(),
                                         hash_values.data_ptr<int>(), indices.data_ptr<int>(), dropped_points.data_ptr<int>(), N, K, T, lookup_tries, threads,
                                         at::cuda::getCurrentCUDAStream().stream());
        if (dropped_points[0].item<int>() != 0 || dropped_points[1].item<int>() != 0) {
            TORCH_CHECK(false, "Error: dropped points detected in generate_conv3d_subm_indices (hash table overflow or collision).");
        }
    } else {
        generate_conv3d_subm_indices_cpu(coords.data_ptr<int>(), indices.data_ptr<int>(), N, K);
    }

    return indices;
}


std::tuple<torch::Tensor, torch::Tensor> generate_conv3d_indices(const torch::Tensor coords,  // (N, 4)
                                                                 const int64_t batch_size, const int64_t K,
                                                                 const int64_t stride_x, const int64_t stride_y,
                                                                 const int64_t stride_z, const int64_t pad_x,
                                                                 const int64_t pad_y, const int64_t pad_z,
                                                                 const int64_t max_x, const int64_t max_y,
                                                                 const int64_t max_z, double hash_table_multiplier, int64_t threads, int64_t lookup_tries) {
    TORCH_CHECK(coords.dim() == 2 && coords.size(1) == 4, "coords must be of shape (N, 4)");
    TORCH_CHECK(coords.dtype() == torch::kInt32, "coords must be of dtype int32");
    TORCH_CHECK(K > 0, "Kernel size K must be positive");
    TORCH_CHECK(stride_x > 0, "Stride must be positive");
    TORCH_CHECK(stride_y > 0, "Stride must be positive");
    TORCH_CHECK(stride_z > 0, "Stride must be positive");
    TORCH_CHECK(pad_x >= 0, "Padding must be non-negative");
    TORCH_CHECK(pad_y >= 0, "Padding must be non-negative");
    TORCH_CHECK(pad_z >= 0, "Padding must be non-negative");
    TORCH_CHECK(max_x >= 0 && max_y >= 0 && max_z >= 0, "Max coordinates must be non-negative");

    int64_t N = coords.size(0);
    int num_x = std::max(max_x + 2 * pad_x - K, 0l) / stride_x + 1;
    int num_y = std::max(max_y + 2 * pad_y - K, 0l) / stride_y + 1;
    int num_z = std::max(max_z + 2 * pad_z - K, 0l) / stride_z + 1;

    // number of additional points each non-zero point can theoretically add, alone each dimension
    int num_Kx = (K + stride_x - 1) / stride_x;
    int num_Ky = (K + stride_y - 1) / stride_y;
    int num_Kz = (K + stride_z - 1) / stride_z;
    
    // theoretical maximum number of indices
    int NPrime = std::min(N * num_Kx * num_Ky * num_Kz, num_x * num_y * num_z * batch_size);

    auto options = torch::TensorOptions().dtype(torch::kInt32).device(coords.device());
    auto new_coords = torch::empty({NPrime, 4}, options);
    auto indices = torch::empty({NPrime, K * K * K}, options);
    // treat -1 as zero/null indices
    indices.fill_(-1);

    int3 stride = {static_cast<int>(stride_x), static_cast<int>(stride_y), static_cast<int>(stride_z)};
    int3 pad = {static_cast<int>(pad_x), static_cast<int>(pad_y), static_cast<int>(pad_z)};
    int3 max_coords = {static_cast<int>(max_x), static_cast<int>(max_y), static_cast<int>(max_z)};

    if (coords.device().is_cuda()) {
        // Call the GPU implementation
        int T = static_cast<int>(NPrime * hash_table_multiplier);
        // TODO: tune T so it's not too big
        // this is data dependent. If T is too small, the kernel gets stuck
        // TODO: make kernel resilient to T being too small
        // TORCH_CHECK(hash_table_multiplier > 1, "Hash table multiplier must be > 1");
        // TORCH_CHECK(T > NPrime, "Hash table size must be greater than number of coordinates * 2");

        auto global_offset = torch::empty({2}, torch::TensorOptions().dtype(torch::kInt32).device(coords.device()));
        global_offset.fill_(0);

        auto hash_keys = torch::empty({T}, torch::TensorOptions().dtype(torch::kInt64).device(coords.device()));
        hash_keys.fill_(-1);
        auto hash_values = torch::empty({T}, torch::TensorOptions().dtype(torch::kInt32).device(coords.device()));
        hash_values.fill_(-1);
        generate_conv3d_indices_kernel_gpu(coords.data_ptr<int>(), new_coords.data_ptr<int>(), indices.data_ptr<int>(),

                                           global_offset.data_ptr<int>(), (uint64_t *)hash_keys.data_ptr<int64_t>(),
                                           hash_values.data_ptr<int>(), N, NPrime, K, T, stride, pad, max_coords,
                                           lookup_tries,
                                           global_offset.data_ptr<int>() + 1,  // dropped points
                                           threads,  // threads
                                           at::cuda::getCurrentCUDAStream().stream());
        int actual_NPrime = global_offset[0].item<int>();
        int drooped_points = global_offset[1].item<int>();
        if (drooped_points > 0) {
            TORCH_CHECK(false, "Error: dropped points detected in generate_conv3d_indices (hash table overflow or collision).", drooped_points, " points were dropped.");
        }
        new_coords.resize_({actual_NPrime, 4});
        indices.resize_({actual_NPrime, K * K * K});
    } else {
        int64_t NPrime = generate_conv3d_indices_cpu(coords.data_ptr<int>(), new_coords.data_ptr<int>(),
                                                     indices.data_ptr<int>(), N, K, stride, pad, max_coords);
        new_coords.resize_({NPrime, 4});
        indices.resize_({NPrime, K * K * K});
    }
    return {new_coords, indices};
}

TORCH_LIBRARY(convIdx, m) {
    m.def("generate_conv3d_subm_indices",
          &generate_conv3d_subm_indices);
    m.def("generate_conv3d_indices",
          &generate_conv3d_indices);
    m.def("generate_conv3d_subm_indices_v2",
          &generate_conv3d_subm_indicesV2);
}
