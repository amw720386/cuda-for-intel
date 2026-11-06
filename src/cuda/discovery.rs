use super::primary::*;
use super::streams::events::*;
use super::streams::{cuStreamCreate, cuStreamDestroy_v2, cuStreamQuery, cuStreamSynchronize};
use super::*;

// Entry-point negotiation ceiling. Feature and ABI support are per symbol.
const DRIVER_ABI: c_int = 13000;

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuDriverGetVersion(out: *mut c_int) -> CuResult {
    trace::call("cuDriverGetVersion", || {
        if out.is_null() {
            return InvalidValue;
        }
        unsafe { out.write(DRIVER_ABI) };
        Success
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuDeviceGetPCIBusId(
    out: *mut c_char,
    len: c_int,
    ordinal: c_int,
) -> CuResult {
    trace::call("cuDeviceGetPCIBusId", || {
        if out.is_null() || len <= 0 {
            return InvalidValue;
        }
        device(ordinal, |info| {
            let Some([domain, bus, slot, function]) = info.pci_address else {
                return NotSupported;
            };
            let id = format!("{domain:04x}:{bus:02x}:{slot:02x}.{function:x}\0");
            if (len as usize) < id.len() {
                return InvalidValue;
            }
            unsafe { std::ptr::copy_nonoverlapping(id.as_ptr().cast(), out, id.len()) };
            Success
        })
    })
}

fn error_info(code: c_int) -> Option<(&'static CStr, &'static CStr)> {
    Some(match code {
        0 => (c"CUDA_SUCCESS", c"no error"),
        1 => (c"CUDA_ERROR_INVALID_VALUE", c"invalid argument"),
        2 => (c"CUDA_ERROR_OUT_OF_MEMORY", c"out of memory"),
        3 => (c"CUDA_ERROR_NOT_INITIALIZED", c"initialization error"),
        4 => (c"CUDA_ERROR_DEINITIALIZED", c"driver shutting down"),
        100 => (
            c"CUDA_ERROR_NO_DEVICE",
            c"no CUDA-capable device is detected",
        ),
        101 => (c"CUDA_ERROR_INVALID_DEVICE", c"invalid device ordinal"),
        200 => (c"CUDA_ERROR_INVALID_IMAGE", c"invalid kernel image"),
        201 => (c"CUDA_ERROR_INVALID_CONTEXT", c"invalid device context"),
        209 => (
            c"CUDA_ERROR_NO_BINARY_FOR_GPU",
            c"no kernel image is available",
        ),
        218 => (c"CUDA_ERROR_INVALID_PTX", c"invalid PTX"),
        222 => (
            c"CUDA_ERROR_UNSUPPORTED_PTX_VERSION",
            c"unsupported PTX version",
        ),
        301 => (c"CUDA_ERROR_FILE_NOT_FOUND", c"file not found"),
        304 => (
            c"CUDA_ERROR_OPERATING_SYSTEM",
            c"operating system call failed",
        ),
        400 => (c"CUDA_ERROR_INVALID_HANDLE", c"invalid resource handle"),
        500 => (c"CUDA_ERROR_NOT_FOUND", c"named symbol not found"),
        600 => (c"CUDA_ERROR_NOT_READY", c"operation is not ready"),
        708 => (
            c"CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE",
            c"primary context is active",
        ),
        709 => (c"CUDA_ERROR_CONTEXT_IS_DESTROYED", c"context is destroyed"),
        719 => (c"CUDA_ERROR_LAUNCH_FAILED", c"launch failed"),
        801 => (c"CUDA_ERROR_NOT_SUPPORTED", c"operation not supported"),
        999 => (c"CUDA_ERROR_UNKNOWN", c"unknown error"),
        _ => return None,
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGetErrorName(code: c_int, out: *mut *const c_char) -> CuResult {
    trace::call("cuGetErrorName", || {
        if out.is_null() {
            return InvalidValue;
        }
        let info = error_info(code);
        unsafe { out.write(info.map_or(std::ptr::null(), |(name, _)| name.as_ptr())) };
        if info.is_some() {
            Success
        } else {
            InvalidValue
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGetErrorString(code: c_int, out: *mut *const c_char) -> CuResult {
    trace::call("cuGetErrorString", || {
        if out.is_null() {
            return InvalidValue;
        }
        let info = error_info(code);
        unsafe { out.write(info.map_or(std::ptr::null(), |(_, text)| text.as_ptr())) };
        if info.is_some() {
            Success
        } else {
            InvalidValue
        }
    })
}

const FOUND: c_int = 0;
const MISSING: c_int = 1;
const VERSION: c_int = 2;

fn lookup(symbol: &[u8], version: c_int, per_thread: bool) -> Result<Opaque, c_int> {
    macro_rules! entry {
        ($function:ident, $since:literal) => {
            entry!($function, $since, DRIVER_ABI + 1, false)
        };
        ($function:ident, $since:literal, $until:expr, $has_ptds:expr) => {
            if !($since..$until).contains(&version) {
                Err(VERSION)
            } else if per_thread && $has_ptds {
                Err(MISSING)
            } else {
                Ok($function as *const () as Opaque)
            }
        };
    }

    // These keys are CUDA base names. Each range describes the ABI of the
    // callable function, including upper bounds where CUDA changed its ABI.
    match symbol {
        b"cuInit" => entry!(cuInit, 2000),
        b"cuDriverGetVersion" => entry!(cuDriverGetVersion, 2020),
        b"cuGetErrorName" => entry!(cuGetErrorName, 6000),
        b"cuGetErrorString" => entry!(cuGetErrorString, 6000),
        b"cuGetProcAddress" if version >= 12000 => entry!(cuGetProcAddress_v2, 12000),
        b"cuGetProcAddress" => entry!(cuGetProcAddress, 11030),
        b"cuDeviceGet" => entry!(cuDeviceGet, 2000),
        b"cuDeviceGetCount" => entry!(cuDeviceGetCount, 2000),
        b"cuDeviceGetName" => entry!(cuDeviceGetName, 2000),
        b"cuDeviceGetAttribute" => entry!(cuDeviceGetAttribute, 2000),
        b"cuDeviceGetPCIBusId" => entry!(cuDeviceGetPCIBusId, 4010),
        b"cuDeviceTotalMem" => entry!(cuDeviceTotalMem_v2, 3020),
        b"cuDevicePrimaryCtxGetState" => entry!(cuDevicePrimaryCtxGetState, 7000),
        b"cuDevicePrimaryCtxRetain" => entry!(cuDevicePrimaryCtxRetain, 7000),
        b"cuDevicePrimaryCtxRelease" if version >= 11000 => {
            entry!(cuDevicePrimaryCtxRelease_v2, 11000)
        }
        b"cuDevicePrimaryCtxRelease" => entry!(cuDevicePrimaryCtxRelease, 7000),
        b"cuDevicePrimaryCtxReset" if version >= 11000 => entry!(cuDevicePrimaryCtxReset_v2, 11000),
        b"cuDevicePrimaryCtxReset" => entry!(cuDevicePrimaryCtxReset, 7000),
        b"cuDevicePrimaryCtxSetFlags" if version >= 11000 => {
            entry!(cuDevicePrimaryCtxSetFlags_v2, 11000)
        }
        b"cuDevicePrimaryCtxSetFlags" => entry!(cuDevicePrimaryCtxSetFlags, 7000),
        b"cuCtxCreate" => entry!(cuCtxCreate_v2, 3020, 11040, false),
        b"cuCtxDestroy" => entry!(cuCtxDestroy_v2, 4000),
        b"cuCtxGetCurrent" => entry!(cuCtxGetCurrent, 4000),
        b"cuCtxSetCurrent" => entry!(cuCtxSetCurrent, 4000),
        b"cuCtxGetDevice" => entry!(cuCtxGetDevice, 2000, 13000, false),
        b"cuCtxSynchronize" => entry!(cuCtxSynchronize, 2000, 13000, false),
        b"cuMemAlloc" => entry!(cuMemAlloc_v2, 3020),
        b"cuMemFree" => entry!(cuMemFree_v2, 3020),
        b"cuMemHostAlloc" => entry!(cuMemHostAlloc, 2020),
        b"cuMemFreeHost" => entry!(cuMemFreeHost, 2000),
        b"cuMemcpyHtoD" => entry!(cuMemcpyHtoD_v2, 3020, DRIVER_ABI + 1, true),
        b"cuMemcpyDtoH" => entry!(cuMemcpyDtoH_v2, 3020, DRIVER_ABI + 1, true),
        b"cuMemcpyHtoDAsync" => entry!(cuMemcpyHtoDAsync_v2, 3020, DRIVER_ABI + 1, true),
        b"cuMemcpyDtoHAsync" => entry!(cuMemcpyDtoHAsync_v2, 3020, DRIVER_ABI + 1, true),
        b"cuModuleLoad" => entry!(cuModuleLoad, 2000),
        b"cuModuleLoadData" => entry!(cuModuleLoadData, 2000),
        b"cuModuleGetFunction" => entry!(cuModuleGetFunction, 2000),
        b"cuModuleUnload" => entry!(cuModuleUnload, 2000),
        b"cuLaunchKernel" => entry!(cuLaunchKernel, 4000, DRIVER_ABI + 1, true),
        b"cuStreamCreate" => entry!(cuStreamCreate, 2000),
        b"cuStreamDestroy" => entry!(cuStreamDestroy_v2, 4000),
        b"cuStreamSynchronize" => entry!(cuStreamSynchronize, 2000, DRIVER_ABI + 1, true),
        b"cuStreamQuery" => entry!(cuStreamQuery, 2000, DRIVER_ABI + 1, true),
        b"cuEventCreate" => entry!(cuEventCreate, 2000),
        b"cuEventRecord" => entry!(cuEventRecord, 2000, DRIVER_ABI + 1, true),
        b"cuEventQuery" => entry!(cuEventQuery, 2000),
        b"cuEventSynchronize" => entry!(cuEventSynchronize, 2000),
        b"cuEventDestroy" => entry!(cuEventDestroy_v2, 4000),
        b"cuStreamWaitEvent" => entry!(cuStreamWaitEvent, 3020, DRIVER_ABI + 1, true),
        _ => Err(MISSING),
    }
}

unsafe fn resolve(
    symbol: *const c_char,
    out: *mut Opaque,
    version: c_int,
    flags: u64,
    query: *mut c_int,
    legacy: bool,
) -> CuResult {
    if symbol.is_null() || out.is_null() || !(1..=DRIVER_ABI).contains(&version) || flags > 2 {
        if trace::enabled() {
            trace::record(format_args!(
                "resolve invalid request version={version} flags={flags}"
            ));
        }
        return InvalidValue;
    }
    let symbol = unsafe { CStr::from_ptr(symbol) }.to_bytes();
    let (pointer, outcome) = match lookup(symbol, version, flags == 2) {
        Ok(pointer) => (pointer, FOUND),
        Err(outcome) => (std::ptr::null_mut(), outcome),
    };
    unsafe {
        out.write(pointer);
        if !query.is_null() {
            query.write(outcome);
        }
    }
    if trace::enabled() {
        let name = trace::symbol(symbol);
        trace::record(format_args!(
            "resolve {name} version={version} flags={flags} status={outcome}"
        ));
    }
    if legacy && outcome != FOUND {
        NotFound
    } else {
        Success
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGetProcAddress(
    symbol: *const c_char,
    out: *mut Opaque,
    version: c_int,
    flags: u64,
) -> CuResult {
    trace::call("cuGetProcAddress", || unsafe {
        resolve(symbol, out, version, flags, std::ptr::null_mut(), true)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGetProcAddress_v2(
    symbol: *const c_char,
    out: *mut Opaque,
    version: c_int,
    flags: u64,
    query: *mut c_int,
) -> CuResult {
    trace::call("cuGetProcAddress_v2", || unsafe {
        resolve(symbol, out, version, flags, query, false)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr::null_mut;

    #[test]
    fn resolves_only_the_requested_abi() {
        let mut function = null_mut();
        let mut query = -1;
        let resolve = |symbol: &CStr, function: &mut Opaque, version, flags, query: &mut c_int| unsafe {
            cuGetProcAddress_v2(symbol.as_ptr(), function, version, flags, query)
        };
        assert_eq!(
            resolve(c"cuGetProcAddress", &mut function, 12000, 0, &mut query),
            Success
        );
        assert_eq!(function, cuGetProcAddress_v2 as *const () as Opaque);
        assert_eq!(query, FOUND);
        assert_eq!(
            resolve(c"cuGetProcAddress", &mut function, 11030, 1, &mut query),
            Success
        );
        assert_eq!(function, cuGetProcAddress as *const () as Opaque);

        for (symbol, version, flags, expected) in [
            (c"cuMemAlloc", 3010, 0, VERSION),
            (c"cuCtxCreate", 12000, 0, VERSION),
            (c"cuCtxGetDevice", 13000, 0, VERSION),
            (c"cuMemcpyHtoD", 12000, 2, MISSING),
            (c"cuMemAlloc_v2", 12000, 0, MISSING),
            (c"cuGetExportTable", 13000, 0, MISSING),
        ] {
            function = cuInit as *const () as Opaque;
            assert_eq!(
                resolve(symbol, &mut function, version, flags, &mut query),
                Success
            );
            assert!(function.is_null());
            assert_eq!(query, expected);
        }
        assert_eq!(
            resolve(c"cuInit", &mut function, DRIVER_ABI + 10, 0, &mut query),
            InvalidValue
        );
        assert_eq!(
            resolve(c"cuInit", &mut function, DRIVER_ABI, 3, &mut query),
            InvalidValue
        );
        assert_eq!(
            unsafe { cuGetProcAddress(c"cuGetExportTable".as_ptr(), &mut function, 13000, 0) },
            NotFound
        );
        assert!(function.is_null());
        assert_eq!(
            resolve(c"cuInit", &mut function, DRIVER_ABI, 2, &mut query),
            Success
        );
        let init: extern "system" fn(c_uint) -> CuResult = unsafe { std::mem::transmute(function) };
        assert_eq!(init(1), InvalidValue);
    }

    #[test]
    fn version_and_error_strings_work_before_initialization() {
        let mut version = 0;
        assert_eq!(unsafe { cuDriverGetVersion(&mut version) }, Success);
        assert_eq!(version, DRIVER_ABI);
        assert_eq!(unsafe { cuDriverGetVersion(null_mut()) }, InvalidValue);
        let mut out = std::ptr::null();
        assert_eq!(unsafe { cuGetErrorName(801, &mut out) }, Success);
        assert_eq!(unsafe { CStr::from_ptr(out) }, c"CUDA_ERROR_NOT_SUPPORTED");
        assert_eq!(unsafe { cuGetErrorString(801, &mut out) }, Success);
        assert_eq!(unsafe { CStr::from_ptr(out) }, c"operation not supported");
        assert_eq!(unsafe { cuGetErrorName(-1, &mut out) }, InvalidValue);
        assert!(out.is_null());
    }
}
