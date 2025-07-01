#include "triton_aot_utils.h"
#include <cstddef>

int get_cur_sm_version() {
    int device = 0;
    CHECK_CUDA_CALL(cuCtxGetDevice(&device));

    // Get the current sm version
    CUdevice cuDevice;
    CHECK_CUDA_CALL(cuDeviceGet(&cuDevice, device));
    int device_major = 0, device_minor = 0;
    CHECK_CUDA_CALL(cuDeviceGetAttribute(&device_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
    CHECK_CUDA_CALL(cuDeviceGetAttribute(&device_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
    int device_sm = device_major * 10 + device_minor;
    return device_sm;
}

int cdiv(int a, int b) {
    return (a + b - 1) / b;
}

bool is_div_16(const std::pair<Dtype, void*> &arg) {
    if (arg.first.is_ptr) {
        return size_t(arg.second) % 16 == 0;
    } else if (arg.first == Dtype::INT32) {
        int data = *reinterpret_cast<int*>(arg.second);
        return data % 16 == 0;
    } else if (arg.first == Dtype::INT64) {
        int64_t data = *reinterpret_cast<int64_t*>(arg.second);
        return data % 16 == 0;
    }
    return false;  // For other dtypes, we assume they are not divisible by 16
}


