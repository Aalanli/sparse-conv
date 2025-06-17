#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/script.h>

#include <iostream>

__device__ __host__ int64_t key_i(int x, int depth, int i) {
    int64_t key = 0;
    for (int j = 0; j < depth; j++) {
        key = key | ((x & (1 << j)) << (2 * j + i));
    }
    return key;
}

__device__ __host__ int64_t key_x(int x, int depth) { return key_i(x, depth, 2); }

__device__ __host__ int64_t key_y(int y, int depth) { return key_i(y, depth, 1); }

__device__ __host__ int64_t key_z(int z, int depth) { return key_i(z, depth, 0); }

__device__ __host__ int64_t encode_z_pt(int x, int y, int z, int depth) {
    int mask;
    if (depth > 8) {
        mask = 255;
    } else {
        mask = (1 << depth) - 1;
    }
    auto key = key_x(x & mask, 8) | key_y(y & mask, 8) | key_z(z & mask, 8);
    if (depth > 8) {
        mask = (1 << (depth - 8)) - 1;
        auto key16 = (key_x((x >> 8) & mask, 8)) | (key_y((y >> 8) & mask, 8)) | (key_z((z >> 8) & mask, 8));
        key = key | (key16 << 24);
    }

    return key;
}

void encode_z_kernel_cpu(int *pts, int n, int depth, int64_t *res) {
    for (int i = 0; i < n; ++i) {
        res[i] = encode_z_pt(pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2], depth);
    }
}

__global__ void encode_z_kernel_cuda(const int *__restrict__ pts, const int n, const int depth,
                                     int64_t *__restrict__ res) {
    for (int nid = threadIdx.x + blockIdx.x * blockDim.x; nid < n; nid += blockDim.x * gridDim.x) {
        int x = pts[nid * 3];
        int y = pts[nid * 3 + 1];
        int z = pts[nid * 3 + 2];

        res[nid] = encode_z_pt(x, y, z, depth);
    }
}

torch::Tensor encode_z(torch::Tensor pts, int64_t depth) {
    at::cuda::CUDAGuard device_guard(pts.device());
    auto dtype = pts.scalar_type();
    TORCH_CHECK(pts.dim() == 2, "pts must be a 2D tensor");
    TORCH_CHECK(pts.size(1) == 3, "pts must have 3 channels");
    TORCH_CHECK(dtype == torch::kInt32, "pts must be of type int32");
    TORCH_CHECK(pts.is_contiguous(), "pts must be contiguous");
    auto n = pts.size(0);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(pts.device());
    auto keys = torch::empty({n}, options);
    if (n == 0) {
        return keys;
    }
    if (pts.device().is_cuda()) {
        auto stream = c10::cuda::getCurrentCUDAStream();
        dim3 block_size(512);
        dim3 grid_size((n + 512 - 1) / 512);
        encode_z_kernel_cuda<<<grid_size, block_size, 0, stream>>>(pts.data_ptr<int>(), n, int(depth),
                                                                   keys.data_ptr<int64_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        encode_z_kernel_cpu(pts.data_ptr<int>(), pts.size(0), int(depth), keys.data_ptr<int64_t>());
    }
    return keys;
}

__device__ __host__ int64_t transpose_pack(int *locs, int ndims, int num_bits) {
    int64_t key = 0;
    for (int dim = 0; dim < ndims; dim++) {
        for (int bit = 0; bit < num_bits; bit++) {
            key |= int64_t((locs[dim] & (1 << bit)) >> bit) << (bit * ndims + (ndims - 1 - dim));
        }
    }
    return key;
}

__device__ __host__ int bit_length(int x) {
    int count = 0;
    while (x) {
        x >>= 1;
        count++;
    }
    return count;
}

template <int ndims>
__device__ __host__ int64_t encode_hilbert_pt(int *locs, int num_bits) {
    int buf[ndims];
    for (int i = 0; i < ndims; i++) {
        buf[i] = locs[i];
    }

    for (int bit = 0; bit < num_bits; bit++) {
        for (int dim = 0; dim < ndims; dim++) {
            int bit_i = num_bits - bit - 1;
            int mask = buf[dim] & (1 << bit_i);
            int lower_bit_mask = (1 << bit_i) - 1;
            if (mask) {
                buf[0] = (lower_bit_mask & (~buf[0])) | ((~lower_bit_mask) & buf[0]);
            } else {
                int to_flip = (buf[0] ^ buf[dim]) & lower_bit_mask;
                buf[0] = (to_flip ^ buf[0]);
                buf[dim] = (to_flip ^ buf[dim]);
            }
        }
    }
    auto res = transpose_pack(buf, ndims, num_bits);
    int shift = 1 << (bit_length(num_bits * ndims) - 1);
    while (shift > 0) {
        res ^= (res >> shift);
        shift >>= 1;
    }
    return res;
}

__device__ __host__ int64_t encode_hilbert_3d(int x, int y, int z, int num_bits) {
    int locs[3] = {x, y, z};
    return encode_hilbert_pt<3>(locs, num_bits);
}

void encode_hilbert_kernel_cpu(int *pts, int n, int num_bits, int64_t *res) {
    for (int i = 0; i < n; ++i) {
        res[i] = encode_hilbert_3d(pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2], num_bits);
    }
}

__global__ void encode_hilbert_kernel_cuda(const int *__restrict__ pts, const int n, const int num_bits,
                                           int64_t *__restrict__ res) {
    for (int nid = threadIdx.x + blockIdx.x * blockDim.x; nid < n; nid += blockDim.x * gridDim.x) {
        int x = pts[nid * 3];
        int y = pts[nid * 3 + 1];
        int z = pts[nid * 3 + 2];

        res[nid] = encode_hilbert_3d(x, y, z, num_bits);
    }
}

torch::Tensor encode_hilbert(torch::Tensor pts, int64_t num_bits) {
    at::cuda::CUDAGuard device_guard(pts.device());
    auto dtype = pts.scalar_type();
    TORCH_CHECK(pts.dim() == 2, "pts must be a 2D tensor");
    TORCH_CHECK(pts.size(1) == 3, "pts must have 3 channels");
    TORCH_CHECK(dtype == torch::kInt32, "pts must be of type int32");
    TORCH_CHECK(pts.is_contiguous(), "pts must be contiguous");
    auto n = pts.size(0);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(pts.device());
    auto keys = torch::empty({n}, options);
    if (n == 0) {
        return keys;
    }
    if (pts.device().is_cuda()) {
        auto stream = c10::cuda::getCurrentCUDAStream();
        dim3 block_size(512);
        dim3 grid_size((n + 512 - 1) / 512);
        encode_hilbert_kernel_cuda<<<grid_size, block_size, 0, stream>>>(pts.data_ptr<int>(), n, int(num_bits),
                                                                         keys.data_ptr<int64_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        encode_hilbert_kernel_cpu(pts.data_ptr<int>(), pts.size(0), int(num_bits), keys.data_ptr<int64_t>());
    }
    return keys;
}

__global__ void fused_encoding_kernel(
    const int *__restrict__ pts,  // [n, 3]
    const int n, const int num_bits, const int num_encodings,
    const int *__restrict__ encoding_ids,  // [num_encodings] {0 = z, 1 = z-trans, 2 = hilbert, 3 = hilbert-trans}
    int64_t *__restrict__ enc_out          // [num_encodings, n]
) {
    extern __shared__ int encoding_ids_shared[];
    if (threadIdx.x < num_encodings) {
        encoding_ids_shared[threadIdx.x] = encoding_ids[threadIdx.x];
    }
    __syncthreads();
    for (int nid = threadIdx.x + blockIdx.x * blockDim.x; nid < n; nid += blockDim.x * gridDim.x) {
        int x = pts[nid * 3];
        int y = pts[nid * 3 + 1];
        int z = pts[nid * 3 + 2];
        int64_t res;
        for (int i = 0; i < num_encodings; ++i) {
            const int enc_id = encoding_ids_shared[i];
            if (enc_id == 0) {
                res = encode_z_pt(x, y, z, num_bits);
            } else if (enc_id == 1) {
                res = encode_z_pt(y, x, z, num_bits);
            } else if (enc_id == 2) {
                res = encode_hilbert_3d(x, y, z, num_bits);
            } else if (enc_id == 3) {
                res = encode_hilbert_3d(y, x, z, num_bits);
            }

            enc_out[i * n + nid] = res;
        }
    }
}

__global__ void fused_encoding_batch_kernel(
    const int *__restrict__ pts,  // [n, 4]
    const int n, const int num_bits, const int num_encodings,
    const int *__restrict__ encoding_ids,  // [num_encodings] {0 = z, 1 = z-trans, 2 = hilbert, 3 = hilbert-trans}
    int64_t *__restrict__ enc_out          // [num_encodings, n]
) {
    extern __shared__ int encoding_ids_shared[];
    if (threadIdx.x < num_encodings) {
        encoding_ids_shared[threadIdx.x] = encoding_ids[threadIdx.x];
    }
    __syncthreads();
    for (int nid = threadIdx.x + blockIdx.x * blockDim.x; nid < n; nid += blockDim.x * gridDim.x) {
        int4 data = ((int4 *)pts)[nid];
        int64_t b = int64_t(data.x) << (num_bits * 3);
        int x = data.y;
        int y = data.z;
        int z = data.w;
        int64_t res;
        for (int i = 0; i < num_encodings; ++i) {
            const int enc_id = encoding_ids_shared[i];
            if (enc_id == 0) {
                res = encode_z_pt(x, y, z, num_bits);
            } else if (enc_id == 1) {
                res = encode_z_pt(y, x, z, num_bits);
            } else if (enc_id == 2) {
                res = encode_hilbert_3d(x, y, z, num_bits);
            } else if (enc_id == 3) {
                res = encode_hilbert_3d(y, x, z, num_bits);
            }

            enc_out[i * n + nid] = res | b;
        }
    }
}

torch::Tensor fused_encoding(torch::Tensor pts, int64_t num_bits, torch::Tensor encoding_ids,
                             int64_t nelem_per_worker) {
    at::cuda::CUDAGuard device_guard(pts.device());
    auto dtype = pts.scalar_type();
    TORCH_CHECK(pts.dim() == 2, "pts must be a 2D tensor");
    TORCH_CHECK(pts.size(1) == 3 || pts.size(1) == 4, "pts must have 3 or 4 channels");
    TORCH_CHECK(dtype == torch::kInt32, "pts must be of type int32");
    TORCH_CHECK(pts.is_contiguous(), "pts must be contiguous");
    TORCH_CHECK(encoding_ids.dim() == 1, "encoding_ids must be a 1D tensor");
    TORCH_CHECK(encoding_ids.scalar_type() == torch::kInt32, "encoding_ids must be of type int32");
    TORCH_CHECK(encoding_ids.is_contiguous(), "encoding_ids must be contiguous");
    TORCH_CHECK(encoding_ids.size(0) < 512, "todo");
    TORCH_CHECK(pts.device() == encoding_ids.device());
    TORCH_CHECK(num_bits * 3 <= 64, "bits_overflow");

    auto n = pts.size(0);
    auto num_encodings = encoding_ids.size(0);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(pts.device());
    auto keys = torch::empty({num_encodings, n}, options);
    if (n == 0 || num_encodings == 0) {
        return keys;
    }

    bool is_batch_mode = pts.size(1) == 4;

    if (pts.device().is_cuda() && encoding_ids.device().is_cuda()) {
        auto stream = c10::cuda::getCurrentCUDAStream();
        dim3 block_size(512);
        dim3 grid_size(((n + 512 - 1) / 512 + nelem_per_worker - 1) / nelem_per_worker);
        size_t shared_mem_size = num_encodings * sizeof(int);
        if (!is_batch_mode) {
            fused_encoding_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(
                pts.data_ptr<int>(), n, int(num_bits), num_encodings, encoding_ids.data_ptr<int>(),
                keys.data_ptr<int64_t>());
        } else {
            fused_encoding_batch_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(
                pts.data_ptr<int>(), n, int(num_bits), num_encodings, encoding_ids.data_ptr<int>(),
                keys.data_ptr<int64_t>());
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        auto pts_data = pts.data_ptr<int>();
        auto encoding_ids_data = encoding_ids.data_ptr<int>();
        auto keys_data = keys.data_ptr<int64_t>();

        for (int nid = 0; nid < n; ++nid) {
            int b, x, y, z;
            if (is_batch_mode) {
                b = pts_data[nid * 4];
                x = pts_data[nid * 4 + 1];
                y = pts_data[nid * 4 + 2];
                z = pts_data[nid * 4 + 3];
            } else {
                b = 0;
                x = pts_data[nid * 3];
                y = pts_data[nid * 3 + 1];
                z = pts_data[nid * 3 + 2];
            }
            for (int i = 0; i < num_encodings; ++i) {
                int64_t res = -1;
                const int enc_id = encoding_ids_data[i];
                if (enc_id == 0) {
                    res = encode_z_pt(x, y, z, num_bits);
                } else if (enc_id == 1) {
                    res = encode_z_pt(y, x, z, num_bits);
                } else if (enc_id == 2) {
                    res = encode_hilbert_3d(x, y, z, num_bits);
                } else if (enc_id == 3) {
                    res = encode_hilbert_3d(y, x, z, num_bits);
                }

                keys_data[i * n + nid] = res | (int64_t(b) << (num_bits * 3));
            }
        }
    }
    return keys;
}

TORCH_LIBRARY(spacy_curves, m) {
    m.def("encode_z(Tensor pts, int depth) -> Tensor", &encode_z);
    m.def("encode_hilbert(Tensor pts, int num_bits) -> Tensor", &encode_hilbert);
    m.def("fused_encoding(Tensor pts, int num_bits, Tensor encoding_ids, int nelem_per_worker) -> Tensor", &fused_encoding);
}
