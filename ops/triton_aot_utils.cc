// Copyright (c) 2025 Waabi Innovation. All rights reserved.
#ifndef PLATFORM_CPU
#include "triton_aot_utils.h"

#include <cstddef>

int get_cur_sm_version() {
    int device = 0;
    CHECK_CU_CALL(cuCtxGetDevice(&device));

    // Get the current sm version
    CUdevice cuDevice;
    CHECK_CU_CALL(cuDeviceGet(&cuDevice, device));
    int device_major = 0, device_minor = 0;
    CHECK_CU_CALL(cuDeviceGetAttribute(&device_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
    CHECK_CU_CALL(cuDeviceGetAttribute(&device_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
    int device_sm = device_major * 10 + device_minor;
    return device_sm;
}

int cdiv(int a, int b) { return (a + b - 1) / b; }

bool is_div_16(const std::pair<Dtype, void *> &arg) {
    if (arg.first.is_ptr) {
        return ((size_t)(arg.second)) % 16 == 0;
    } else if (arg.first == Dtype::INT32) {
        int data = *reinterpret_cast<int *>(arg.second);
        return data % 16 == 0;
    } else if (arg.first == Dtype::INT64) {
        int64_t data = *reinterpret_cast<int64_t *>(arg.second);
        return data % 16 == 0;
    }
    return false;  // For other dtypes, we assume they are not divisible by 16
}

void set_zero(void *ptr, Dtype dtype, int size, CUstream stream) {
    if (dtype.type == Dtype::FP16) {
        cuMemsetD16Async((CUdeviceptr)ptr, 0, size, stream);
    } else if (dtype.type == Dtype::FP32) {
        cuMemsetD32Async((CUdeviceptr)ptr, 0, size, stream);
    } else if (dtype.type == Dtype::INT32) {
        cuMemsetD32Async((CUdeviceptr)ptr, 0, size, stream);
    } else if (dtype.type == Dtype::INT64) {
        cuMemsetD32Async((CUdeviceptr)ptr, 0, size * 2, stream);
    } else if (dtype.type == Dtype::BOOL) {
        cuMemsetD8Async((CUdeviceptr)ptr, 0, size, stream);
    } else {
        throw std::runtime_error("Unsupported dtype for set_zero: " + dtype.to_string());
    }
}

#endif  // PLATFORM_CPU
