
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    // Print CUDA Toolkit version
    // printf("CUDA Toolkit version: %d.%d\n", CUDA_VERSION / 1000, (CUDA_VERSION % 1000) / 10);

    // Print CUDA Driver version
    int driverVersion = 0;
    cudaDriverGetVersion(&driverVersion);
    printf("CUDA Driver version: %d.%d\n", driverVersion / 1000, (driverVersion % 1000) / 10);

    // Get number of devices
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("Number of CUDA devices: %d\n", deviceCount);

    // Print properties for each device
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("\nDevice %d: %s\n", i, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total global memory: %zu MB\n", prop.totalGlobalMem / (1024 * 1024));
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Clock rate: %.2f MHz\n", prop.clockRate / 1000.0f);
    }

    return 0;
}