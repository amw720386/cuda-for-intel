#ifndef NVRTC_SHIM_STDIO_H
#define NVRTC_SHIM_STDIO_H
#include "nvrtc_shim_builtins.h"
extern "C" __device__ int printf(const char*, ...);
#endif
