#ifndef NVRTC_SHIM_MATH_H
#define NVRTC_SHIM_MATH_H
#include "nvrtc_shim_builtins.h"
#define M_PI 3.14159265358979323846
#define M_E 2.7182818284590452354
#define M_LN2 0.69314718055994530942
#define M_SQRT2 1.41421356237309504880
#define INFINITY __builtin_inff()
#define NAN __builtin_nanf("")
#define HUGE_VALF __builtin_inff()
#define CUDART_INF_F __builtin_inff()
#define CUDART_NAN_F __builtin_nanf("")
#define CUDART_PI_F 3.141592654f
#endif
