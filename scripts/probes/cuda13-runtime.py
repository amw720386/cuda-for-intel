"""Probe the genuine Windows CUDA 13 Runtime or the stock Ollama CUDA backend."""

import argparse
import ctypes as c
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def function(dll, name, result, *args):
    fn = getattr(dll, name)
    fn.restype = result
    fn.argtypes = args
    return fn


def check(dll, name, arg_types, *args):
    status = function(dll, name, c.c_int, *arg_types)(*args)
    if status:
        raise SystemExit(f"{name} returned CUDA Runtime error {status}")
    print(f"{name} passed", flush=True)


def runtime_probe(runtime):
    int_ptr = c.POINTER(c.c_int)
    version = c.c_int()
    check(runtime, "cudaDriverGetVersion", (int_ptr,), c.byref(version))
    print(f"CUDA driver ABI: {version.value}", flush=True)
    count = c.c_int()
    check(runtime, "cudaGetDeviceCount", (int_ptr,), c.byref(count))
    if count.value < 1:
        raise SystemExit("The CUDA Runtime enumerated zero devices")
    print(f"CUDA devices: {count.value}")
    def fresh_thread_memory():
        free = c.c_size_t()
        total = c.c_size_t()
        size_ptr = c.POINTER(c.c_size_t)
        check(runtime, "cudaMemGetInfo", (size_ptr, size_ptr), c.byref(free), c.byref(total))
        if not 0 < free.value <= total.value:
            raise SystemExit("Invalid device memory reported on a fresh host thread")

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(fresh_thread_memory).result()
    check(runtime, "cudaSetDevice", (c.c_int,), 0)

    # An aligned buffer for cudaDeviceProp. This probe only reads the leading name.
    props = (c.c_uint64 * 512)()
    check(runtime, "cudaGetDeviceProperties", (c.c_void_p, c.c_int), props, 0)
    name = c.string_at(c.addressof(props), 256).split(bytes(1), 1)[0]
    print(f"Device 0: {name.decode()}")

    allocation = c.c_void_p()
    check(runtime, "cudaMalloc", (c.POINTER(c.c_void_p), c.c_size_t), c.byref(allocation), 4096)
    try:
        check(runtime, "cudaDeviceSynchronize", ())
    finally:
        check(runtime, "cudaFree", (c.c_void_p,), allocation)


def ggml_probe(base, plugin):
    registry = function(plugin, "ggml_backend_init", c.c_void_p)()
    if not registry:
        raise SystemExit("ggml_backend_init returned a null registry")
    count = function(base, "ggml_backend_reg_dev_count", c.c_size_t, c.c_void_p)(registry)
    if not count:
        raise SystemExit("The stock GGML CUDA registry contains zero devices")
    get_device = function(base, "ggml_backend_reg_dev_get", c.c_void_p, c.c_void_p, c.c_size_t)
    name = function(base, "ggml_backend_dev_name", c.c_char_p, c.c_void_p)
    for index in range(count):
        print(f"GGML CUDA device {index}: {name(get_device(registry, index)).decode()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("driver", type=Path)
    parser.add_argument("cuda_dll", type=Path, help="cudart64_13.dll or ggml-cuda.dll")
    parser.add_argument("--ggml-base", type=Path, help="ggml-base.dll from the same Ollama release")
    args = parser.parse_args()
    driver_path, cuda_path = (path.resolve(strict=True) for path in (args.driver, args.cuda_dll))
    with os.add_dll_directory(str(cuda_path.parent)):
        # ctypes keeps the module mapped for this process.
        c.WinDLL(str(driver_path))
        if args.ggml_base:
            base = c.CDLL(str(args.ggml_base.resolve(strict=True)))
            ggml_probe(base, c.CDLL(str(cuda_path)))
        else:
            runtime_probe(c.WinDLL(str(cuda_path)))


if __name__ == "__main__":
    main()
