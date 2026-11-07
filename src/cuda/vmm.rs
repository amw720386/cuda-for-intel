//! CUDA Driver Virtual Memory Management stubs.
//!
//! Stock Ollama's `ggml-cuda.dll` imports these symbols even when the device
//! reports `CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 0`.
//! Exporting NotSupported stubs unblocks LoadLibrary; GGML skips the VMM pool
//! after the attribute check.
use super::CuResult;
use std::ffi::{c_int, c_ulonglong};

type CUdeviceptr = u64;
type CUmemGenericAllocationHandle = u64;

#[repr(C)]
pub struct CUmemAllocationProp {
    _opaque: [u8; 64],
}

#[repr(C)]
pub struct CUmemAccessDesc {
    _opaque: [u8; 16],
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemAddressReserve(
    ptr: *mut CUdeviceptr,
    _size: usize,
    _alignment: usize,
    _addr: CUdeviceptr,
    _flags: c_ulonglong,
) -> CuResult {
    if !ptr.is_null() {
        unsafe { ptr.write(0) };
    }
    CuResult::NotSupported
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemAddressFree(_ptr: CUdeviceptr, _size: usize) -> CuResult {
    CuResult::NotSupported
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemCreate(
    handle: *mut CUmemGenericAllocationHandle,
    _size: usize,
    _prop: *const CUmemAllocationProp,
    _flags: c_ulonglong,
) -> CuResult {
    if !handle.is_null() {
        unsafe { handle.write(0) };
    }
    CuResult::NotSupported
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemRelease(_handle: CUmemGenericAllocationHandle) -> CuResult {
    CuResult::NotSupported
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemMap(
    _ptr: CUdeviceptr,
    _size: usize,
    _offset: usize,
    _handle: CUmemGenericAllocationHandle,
    _flags: c_ulonglong,
) -> CuResult {
    CuResult::NotSupported
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemUnmap(_ptr: CUdeviceptr, _size: usize) -> CuResult {
    CuResult::NotSupported
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemSetAccess(
    _ptr: CUdeviceptr,
    _size: usize,
    _desc: *const CUmemAccessDesc,
    _count: usize,
) -> CuResult {
    CuResult::NotSupported
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemGetAllocationGranularity(
    granularity: *mut usize,
    _prop: *const CUmemAllocationProp,
    _option: c_int,
) -> CuResult {
    if !granularity.is_null() {
        unsafe { granularity.write(0) };
    }
    CuResult::NotSupported
}
