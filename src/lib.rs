//! Experimental CUDA Driver API compatibility library.
#![cfg_attr(not(target_pointer_width = "64"), allow(unused))]

#[cfg(not(all(target_pointer_width = "64", any(windows, target_os = "linux"))))]
compile_error!("CUDA for Intel requires 64-bit Windows or Linux");

mod cuda;
mod level_zero;

pub use level_zero::{DeviceInfo, Error, intel_devices};
