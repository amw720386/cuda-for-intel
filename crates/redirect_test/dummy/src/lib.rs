//! Test DLL for the redirect CI job. The job copies it into the app directory (as "NVIDIA's"
//! `cublas64_12.dll`) and into the install directory (as ours); the probe tells them apart by the
//! path the loader reports.

#[unsafe(no_mangle)]
pub extern "C" fn cfi_dummy_id() -> u32 {
    42
}
