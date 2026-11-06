# CUDA for Intel

An experimental, independent Rust implementation of a CUDA Driver API subset for Intel GPUs. The initial target is Windows x64 on Intel Iris Xe (Alder Lake). The project separates the CUDA ABI from the Intel Level Zero backend and will translate a small PTX subset to SPIR-V.

```powershell
cargo build --locked
cargo test --locked
cargo run --locked --example devices
```

The build emits `target/debug/nvcuda.dll` on Windows. Device discovery requires the system Level Zero loader and an Intel graphics driver with Level Zero support. The Linux loader path is included for portability. Windows x64 is the validation target.

The current API supports initialization and device enumeration, names, memory size and selected workgroup attributes. GPU execution, CUDA contexts, memory transfers and module loading are under development.

## References and license

MIT licensed. Architectural references include [Intel-era ZLUDA](https://github.com/em-winterschon/ZLUDA-CUDA-on-Intel-GPU), its [upstream v1 snapshot](https://github.com/vosen/ZLUDA/tree/v1), [modern ZLUDA](https://github.com/vosen/ZLUDA), and [CUDA for AMD on Windows](https://github.com/Speedstu/CUDA-for-AMD-Windows). The backend uses the [Level Zero ABI](https://github.com/oneapi-src/level-zero) exposed by the installed loader and driver. [Intel Compute Runtime](https://github.com/intel/compute-runtime) and [Intel Graphics Compiler](https://github.com/intel/intel-graphics-compiler) describe the driver and compiler stack.
