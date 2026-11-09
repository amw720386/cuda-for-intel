//! Test DLL for the redirect CI job: imports `cublas64_12.dll` by name at load time (static
//! import), as `torch_cuda.dll` does. Placed next to an "NVIDIA" `cublas64_12.dll`.

#[cfg(windows)]
#[link(name = "cublas64_12", kind = "raw-dylib")]
unsafe extern "C" {
    fn cfi_dummy_id() -> u32;
}

#[cfg(windows)]
#[unsafe(no_mangle)]
pub extern "C" fn consumer_call() -> u32 {
    unsafe { cfi_dummy_id() }
}
