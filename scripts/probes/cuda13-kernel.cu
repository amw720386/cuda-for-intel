// Compile to an object with nvcc and link against the genuine CUDA 13 cudart.
#include <cuda_runtime.h>
#include <windows.h>
#include <cstdio>
#include <cstdlib>

static void check(cudaError_t result, const char *operation) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "%s: %d (%s)\n", operation, result, cudaGetErrorString(result));
        std::exit(1);
    }
}

__global__ void write_value(int *output, int value) {
    *output = value;
}

int wmain(int argc, wchar_t **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: kernel.exe ABSOLUTE_PATH_TO_NVCUDA_DLL\n");
        return 1;
    }
    if (!LoadLibraryExW(argv[1], nullptr,
                       LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)) {
        std::fprintf(stderr, "LoadLibraryExW(nvcuda.dll): %lu\n", GetLastError());
        return 1;
    }
    int *device = nullptr;
    int result = 0;
    constexpr int expected = 0x12345678;
    check(cudaSetDevice(0), "cudaSetDevice");
    check(cudaMalloc(reinterpret_cast<void **>(&device), sizeof(result)), "cudaMalloc");
    check(cudaMemset(device, 0x7e, sizeof(result)), "cudaMemset");
    check(cudaMemcpy(&result, device, sizeof(result), cudaMemcpyDeviceToHost), "cudaMemcpy after fill");
    if (result != 0x7e7e7e7e) {
        std::fprintf(stderr, "Unexpected fill result %08x\n", static_cast<unsigned>(result));
        return 2;
    }
    write_value<<<1, 1>>>(device, expected);
    check(cudaGetLastError(), "write_value launch");
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    check(cudaMemcpy(&result, device, sizeof(result), cudaMemcpyDeviceToHost), "cudaMemcpy");
    check(cudaFree(device), "cudaFree");
    check(cudaDeviceReset(), "cudaDeviceReset");
    std::printf("Kernel result %08x\n", static_cast<unsigned>(result));
    return result == expected ? 0 : 2;
}
