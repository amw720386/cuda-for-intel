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

__global__ void write_value(int *output, int value, long long left, long long right) {
    asm("neg.s32 %0, %1;" : "=r"(output[0]) : "r"(value));
    float signed_value;
    asm("copysign.f32 %0, %1, %2;" : "=f"(signed_value) : "f"(-0.0f), "f"(1.5f));
    output[1] = __float_as_int(signed_value);
    constexpr unsigned a = 0x1a1, b = 0x2b2, c = 0x3c3, d = 0x4d4;
    asm volatile("st.global.v4.u8 [%0], {%1, %2, %3, %4};" ::
                 "l"(output + 2), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
    asm volatile("st.u16 [%0], %1;" :: "l"(output + 3), "r"(0x15e6f) : "memory");
    asm volatile("st.u8 [%0], %1;" :: "l"(reinterpret_cast<char *>(output + 3) + 3),
                 "r"(0x1a5) : "memory");
    long long minimum;
    asm("min.s64 %0, %1, %2;" : "=l"(minimum) : "l"(left), "l"(right));
    reinterpret_cast<long long *>(output + 4)[0] = minimum;
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
    int result[6] = {};
    constexpr int expected = 0x12345678;
    check(cudaSetDevice(0), "cudaSetDevice");
    check(cudaMalloc(reinterpret_cast<void **>(&device), sizeof(result)), "cudaMalloc");
    check(cudaMemset(device, 0x7e, sizeof(result)), "cudaMemset");
    check(cudaMemcpy(result, device, sizeof(result), cudaMemcpyDeviceToHost), "cudaMemcpy after fill");
    if (result[0] != 0x7e7e7e7e) {
        std::fprintf(stderr, "Unexpected fill result %08x\n", static_cast<unsigned>(result[0]));
        return 2;
    }
    cudaStream_t stream = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed), "cudaStreamBeginCapture");
    int captured_value = -expected;
    write_value<<<1, 1, 0, stream>>>(device, captured_value, 0x1234567800000000LL, -0x12345678aabbccddLL);
    check(cudaGetLastError(), "write_value capture");
    captured_value = 0;
    cudaGraph_t graph = nullptr;
    check(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
    check(cudaMemcpy(result, device, sizeof(result), cudaMemcpyDeviceToHost), "cudaMemcpy before replay");
    if (result[0] != 0x7e7e7e7e) {
        std::fprintf(stderr, "Capture modified device memory before replay\n");
        return 2;
    }
    cudaGraphExec_t execution = nullptr;
    check(cudaGraphInstantiateWithFlags(&execution, graph, 0), "cudaGraphInstantiateWithFlags");
    check(cudaGraphDestroy(graph), "cudaGraphDestroy");
    check(cudaGraphLaunch(execution, stream), "cudaGraphLaunch");
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    check(cudaGraphExecDestroy(execution), "cudaGraphExecDestroy");
    check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    check(cudaMemcpy(result, device, sizeof(result), cudaMemcpyDeviceToHost), "cudaMemcpy");
    check(cudaFree(device), "cudaFree");
    check(cudaDeviceReset(), "cudaDeviceReset");
    std::printf("Kernel results %08x %08x %08x\n", static_cast<unsigned>(result[0]),
                static_cast<unsigned>(result[1]), static_cast<unsigned>(result[2]));
    std::printf("Signed minimum words %08x %08x\n", static_cast<unsigned>(result[5]),
                static_cast<unsigned>(result[4]));
    return result[0] == expected && static_cast<unsigned>(result[1]) == 0xbfc00000
        && static_cast<unsigned>(result[2]) == 0xd4c3b2a1
        && static_cast<unsigned>(result[3]) == 0xa57e5e6f
        && static_cast<unsigned>(result[4]) == 0x55443323
        && static_cast<unsigned>(result[5]) == 0xedcba987 ? 0 : 2;
}
