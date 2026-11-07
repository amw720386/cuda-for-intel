use super::export_table::cuGetExportTable;
use super::gaps::*;
use super::graphs::*;
use super::library::*;
use super::primary::*;
use super::streams::events::*;
use super::streams::{cuStreamCreate, cuStreamDestroy_v2, cuStreamQuery, cuStreamSynchronize};
use super::vmm::*;
use super::*;

// Entry-point negotiation ceiling. Feature and ABI support are per symbol.
pub(super) const DRIVER_ABI: c_int = 13000;

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
        216 => (
            c"CUDA_ERROR_PEER_ACCESS_UNSUPPORTED",
            c"peer access is not supported across the given devices",
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
        401 => (c"CUDA_ERROR_ILLEGAL_STATE", c"illegal state"),
        500 => (c"CUDA_ERROR_NOT_FOUND", c"named symbol not found"),
        600 => (c"CUDA_ERROR_NOT_READY", c"operation is not ready"),
        704 => (
            c"CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED",
            c"peer access is already enabled",
        ),
        705 => (
            c"CUDA_ERROR_PEER_ACCESS_NOT_ENABLED",
            c"peer access has not been enabled",
        ),
        708 => (
            c"CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE",
            c"primary context is active",
        ),
        709 => (c"CUDA_ERROR_CONTEXT_IS_DESTROYED", c"context is destroyed"),
        712 => (
            c"CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED",
            c"part or all of the requested memory range is already mapped",
        ),
        713 => (
            c"CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED",
            c"pointer does not correspond to any currently registered memory region",
        ),
        719 => (c"CUDA_ERROR_LAUNCH_FAILED", c"launch failed"),
        801 => (c"CUDA_ERROR_NOT_SUPPORTED", c"operation not supported"),
        900 => (
            c"CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED",
            c"operation not permitted during stream capture",
        ),
        901 => (
            c"CUDA_ERROR_STREAM_CAPTURE_INVALIDATED",
            c"stream capture invalidated",
        ),
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
        // cudart 12 resolves this at driver ABI 3000.
        b"cuGetExportTable" => entry!(cuGetExportTable, 3000),
        b"cuDeviceGet" => entry!(cuDeviceGet, 2000),
        b"cuDeviceGetCount" => entry!(cuDeviceGetCount, 2000),
        b"cuDeviceGetName" => entry!(cuDeviceGetName, 2000),
        b"cuDeviceGetAttribute" => entry!(cuDeviceGetAttribute, 2000),
        b"cuDeviceGetPCIBusId" => entry!(cuDeviceGetPCIBusId, 4010),
        b"cuDeviceGetUuid" => entry!(cuDeviceGetUuid, 9020),
        b"cuDeviceGetLuid" => entry!(cuDeviceGetLuid, 10000),
        b"cuDeviceTotalMem" => entry!(cuDeviceTotalMem_v2, 3020),
        b"cuMemGetInfo" => entry!(cuMemGetInfo_v2, 3020),
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
        b"cuCtxGetApiVersion" => entry!(cuCtxGetApiVersion, 3020),
        b"cuCtxGetFlags" => entry!(cuCtxGetFlags, 7000),
        b"cuCtxGetDevice" if version >= 13000 => entry!(cuCtxGetDevice_v2, 13000),
        b"cuCtxGetDevice" => entry!(cuCtxGetDevice, 2000, 13000, false),
        b"cuCtxSynchronize" if version >= 13000 => entry!(cuCtxSynchronize_v2, 13000),
        b"cuCtxSynchronize" => entry!(cuCtxSynchronize, 2000, 13000, false),
        b"cuMemAlloc" => entry!(cuMemAlloc_v2, 3020),
        b"cuMemFree" => entry!(cuMemFree_v2, 3020),
        b"cuMemHostAlloc" => entry!(cuMemHostAlloc, 2020),
        b"cuMemFreeHost" => entry!(cuMemFreeHost, 2000),
        b"cuMemcpyHtoD" => entry!(cuMemcpyHtoD_v2, 3020, DRIVER_ABI + 1, true),
        b"cuMemcpyDtoH" => entry!(cuMemcpyDtoH_v2, 3020, DRIVER_ABI + 1, true),
        b"cuMemcpyHtoDAsync" => entry!(cuMemcpyHtoDAsync_v2, 3020, DRIVER_ABI + 1, true),
        b"cuMemcpyDtoHAsync" => entry!(cuMemcpyDtoHAsync_v2, 3020, DRIVER_ABI + 1, true),
        b"cuMemsetD8" => entry!(cuMemsetD8_v2, 3020, DRIVER_ABI + 1, true),
        b"cuMemsetD8Async" => entry!(cuMemsetD8Async, 3020, DRIVER_ABI + 1, true),
        // Memory and copies. Entries that take a stream have PTDS variants.
        b"cuMemcpyDtoD" => entry!(cuMemcpyDtoD_v2, 3020, DRIVER_ABI + 1, true),
        b"cuMemcpyDtoDAsync" => entry!(cuMemcpyDtoDAsync_v2, 3020, DRIVER_ABI + 1, true),
        b"cuMemsetD16" => entry!(cuMemsetD16_v2, 3020, DRIVER_ABI + 1, true),
        b"cuMemsetD16Async" => entry!(cuMemsetD16Async, 3020, DRIVER_ABI + 1, true),
        b"cuMemsetD32" => entry!(cuMemsetD32_v2, 3020, DRIVER_ABI + 1, true),
        b"cuMemsetD32Async" => entry!(cuMemsetD32Async, 3020, DRIVER_ABI + 1, true),
        b"cuMemcpy" => entry!(cuMemcpy, 7000, DRIVER_ABI + 1, true),
        b"cuMemcpyAsync" => entry!(cuMemcpyAsync, 7000, DRIVER_ABI + 1, true),
        b"cuMemcpy2D" | b"cuMemcpy2DUnaligned" => {
            entry!(cuMemcpy2D_v2, 3020, DRIVER_ABI + 1, true)
        }
        b"cuMemcpy2DAsync" => entry!(cuMemcpy2DAsync_v2, 3020, DRIVER_ABI + 1, true),
        b"cuMemcpy3D" => entry!(cuMemcpy3D_v2, 3020, DRIVER_ABI + 1, true),
        b"cuMemcpy3DAsync" => entry!(cuMemcpy3DAsync_v2, 3020, DRIVER_ABI + 1, true),
        b"cuMemcpyPeer" => entry!(cuMemcpyPeer, 4000, DRIVER_ABI + 1, true),
        b"cuMemcpyPeerAsync" => entry!(cuMemcpyPeerAsync, 4000, DRIVER_ABI + 1, true),
        b"cuMemGetAddressRange" => entry!(cuMemGetAddressRange_v2, 3020),
        b"cuPointerGetAttribute" => entry!(cuPointerGetAttribute, 4000),
        b"cuPointerGetAttributes" => entry!(cuPointerGetAttributes, 7000),
        // cuMemHostRegister gained its _v2 flags in CUDA 6.5.
        b"cuMemHostRegister" => entry!(cuMemHostRegister_v2, 6050),
        b"cuMemHostUnregister" => entry!(cuMemHostUnregister, 4000),
        b"cuMemAllocHost" => entry!(cuMemAllocHost_v2, 3020),
        b"cuMemAllocAsync" => entry!(cuMemAllocAsync, 11020, DRIVER_ABI + 1, true),
        b"cuMemFreeAsync" => entry!(cuMemFreeAsync, 11020, DRIVER_ABI + 1, true),
        b"cuMemAllocFromPoolAsync" => entry!(cuMemAllocFromPoolAsync, 11020, DRIVER_ABI + 1, true),
        b"cuDeviceGetDefaultMemPool" => entry!(cuDeviceGetDefaultMemPool, 11020),
        b"cuDeviceGetMemPool" => entry!(cuDeviceGetMemPool, 11020),
        b"cuDeviceSetMemPool" => entry!(cuDeviceSetMemPool, 11020),
        b"cuMemPoolCreate" => entry!(cuMemPoolCreate, 11020),
        b"cuMemPoolDestroy" => entry!(cuMemPoolDestroy, 11020),
        b"cuMemPoolSetAttribute" => entry!(cuMemPoolSetAttribute, 11020),
        b"cuMemPoolGetAttribute" => entry!(cuMemPoolGetAttribute, 11020),
        b"cuMemPoolTrimTo" => entry!(cuMemPoolTrimTo, 11020),
        // CUDA 12.2 changed advice and prefetch to take a CUmemLocation.
        b"cuMemAdvise" if version >= 12020 => entry!(cuMemAdvise_v2, 12020, DRIVER_ABI + 1, false),
        b"cuMemAdvise" => entry!(cuMemAdvise, 8000, 12020, false),
        b"cuMemPrefetchAsync" if version >= 12020 => {
            entry!(cuMemPrefetchAsync_v2, 12020, DRIVER_ABI + 1, true)
        }
        b"cuMemPrefetchAsync" => entry!(cuMemPrefetchAsync, 8000, 12020, true),
        // Functions, occupancy and extended launch.
        b"cuFuncGetAttribute" => entry!(cuFuncGetAttribute, 2020),
        b"cuFuncSetAttribute" => entry!(cuFuncSetAttribute, 9000),
        b"cuFuncSetCacheConfig" => entry!(cuFuncSetCacheConfig, 3000),
        b"cuOccupancyMaxActiveBlocksPerMultiprocessor" => {
            entry!(cuOccupancyMaxActiveBlocksPerMultiprocessor, 6050)
        }
        b"cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags" => {
            entry!(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, 7000)
        }
        b"cuOccupancyMaxPotentialBlockSize" => entry!(cuOccupancyMaxPotentialBlockSize, 6050),
        b"cuOccupancyMaxPotentialBlockSizeWithFlags" => {
            entry!(cuOccupancyMaxPotentialBlockSizeWithFlags, 7000)
        }
        b"cuLaunchKernelEx" => entry!(cuLaunchKernelEx, 12000, DRIVER_ABI + 1, true),
        // Peers, context stack, stream priorities.
        b"cuDeviceCanAccessPeer" => entry!(cuDeviceCanAccessPeer, 4000),
        b"cuDeviceGetP2PAttribute" => entry!(cuDeviceGetP2PAttribute, 8000),
        b"cuCtxEnablePeerAccess" => entry!(cuCtxEnablePeerAccess, 4000),
        b"cuCtxDisablePeerAccess" => entry!(cuCtxDisablePeerAccess, 4000),
        b"cuCtxPushCurrent" => entry!(cuCtxPushCurrent_v2, 4000),
        b"cuCtxPopCurrent" => entry!(cuCtxPopCurrent_v2, 4000),
        b"cuCtxGetStreamPriorityRange" => entry!(cuCtxGetStreamPriorityRange, 5050),
        b"cuStreamCreateWithPriority" => entry!(cuStreamCreateWithPriority, 5050),
        b"cuModuleLoad" => entry!(cuModuleLoad, 2000),
        b"cuModuleLoadData" => entry!(cuModuleLoadData, 2000),
        b"cuModuleGetFunction" => entry!(cuModuleGetFunction, 2000),
        b"cuModuleUnload" => entry!(cuModuleUnload, 2000),
        b"cuLibraryLoadData" => entry!(cuLibraryLoadData, 12000),
        b"cuLibraryUnload" => entry!(cuLibraryUnload, 12000),
        b"cuModuleGetLoadingMode" => entry!(cuModuleGetLoadingMode, 11070),
        b"cuLibraryGetKernel" => entry!(cuLibraryGetKernel, 12000),
        b"cuLibraryGetModule" => entry!(cuLibraryGetModule, 12000),
        b"cuKernelGetFunction" => entry!(cuKernelGetFunction, 12000),
        b"cuLaunchKernel" => entry!(cuLaunchKernel, 4000, DRIVER_ABI + 1, true),
        b"cuStreamCreate" => entry!(cuStreamCreate, 2000),
        b"cuStreamDestroy" => entry!(cuStreamDestroy_v2, 4000),
        b"cuStreamSynchronize" => entry!(cuStreamSynchronize, 2000, DRIVER_ABI + 1, true),
        b"cuStreamQuery" => entry!(cuStreamQuery, 2000, DRIVER_ABI + 1, true),
        b"cuStreamBeginCapture" if version < 10010 => {
            entry!(cuStreamBeginCapture, 10000, 10010, true)
        }
        b"cuStreamBeginCapture" => entry!(cuStreamBeginCapture_v2, 10010, DRIVER_ABI + 1, true),
        b"cuThreadExchangeStreamCaptureMode" => entry!(cuThreadExchangeStreamCaptureMode, 10010),
        b"cuStreamGetCaptureInfo" if version >= 12030 => {
            entry!(cuStreamGetCaptureInfo_v3, 12030, DRIVER_ABI + 1, true)
        }
        b"cuStreamGetCaptureInfo" if version >= 11030 => {
            entry!(cuStreamGetCaptureInfo_v2, 11030, 12030, true)
        }
        b"cuStreamGetCaptureInfo" => entry!(cuStreamGetCaptureInfo, 10010, 11030, true),
        b"cuStreamEndCapture" => entry!(cuStreamEndCapture, 10000, DRIVER_ABI + 1, true),
        b"cuStreamIsCapturing" => entry!(cuStreamIsCapturing, 10000, DRIVER_ABI + 1, true),
        b"cuGraphDestroy" => entry!(cuGraphDestroy, 10000),
        b"cuGraphInstantiate" if version >= 12000 => entry!(cuGraphInstantiateWithFlags, 12000),
        b"cuGraphInstantiate" => entry!(cuGraphInstantiate_v1, 10000, 12000, false),
        b"cuGraphInstantiateWithFlags" => entry!(cuGraphInstantiateWithFlags, 11040),
        b"cuGraphInstantiateWithParams" => {
            entry!(cuGraphInstantiateWithParams, 12000, DRIVER_ABI + 1, true)
        }
        b"cuGraphExecUpdate" if version >= 12000 => entry!(cuGraphExecUpdate_v2, 12000),
        b"cuGraphExecUpdate" => entry!(cuGraphExecUpdate, 10020, 12000, false),
        b"cuGraphUpload" => entry!(cuGraphUpload, 11010, DRIVER_ABI + 1, true),
        b"cuGraphExecDestroy" => entry!(cuGraphExecDestroy, 10000),
        b"cuGraphLaunch" => entry!(cuGraphLaunch, 10000, DRIVER_ABI + 1, true),
        b"cuEventCreate" => entry!(cuEventCreate, 2000),
        b"cuEventRecord" => entry!(cuEventRecord, 2000, DRIVER_ABI + 1, true),
        b"cuEventRecordWithFlags" => entry!(cuEventRecordWithFlags, 11010, DRIVER_ABI + 1, true),
        b"cuEventQuery" => entry!(cuEventQuery, 2000),
        b"cuEventSynchronize" => entry!(cuEventSynchronize, 2000),
        b"cuEventDestroy" => entry!(cuEventDestroy_v2, 4000),
        b"cuStreamWaitEvent" => entry!(cuStreamWaitEvent, 3020, DRIVER_ABI + 1, true),
        // Virtual memory management (exported as NotSupported stubs; attr 102 = 0).
        b"cuMemAddressReserve" => entry!(cuMemAddressReserve, 10020),
        b"cuMemAddressFree" => entry!(cuMemAddressFree, 10020),
        b"cuMemCreate" => entry!(cuMemCreate, 10020),
        b"cuMemRelease" => entry!(cuMemRelease, 10020),
        b"cuMemMap" => entry!(cuMemMap, 10020),
        b"cuMemUnmap" => entry!(cuMemUnmap, 10020),
        b"cuMemSetAccess" => entry!(cuMemSetAccess, 10020),
        b"cuMemGetAllocationGranularity" => entry!(cuMemGetAllocationGranularity, 10020),
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
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("cuda-for-intel-export.log")
    {
        use std::io::Write;
        let _ = writeln!(
            file,
            "cuGetProcAddress {} ver={version} flags={flags} outcome={outcome} found={}",
            trace::symbol(symbol),
            !pointer.is_null()
        );
    }
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
            (c"cuMemcpyHtoD", 12000, 2, MISSING),
            (c"cuMemAlloc_v2", 12000, 0, MISSING),
        ] {
            function = cuInit as *const () as Opaque;
            assert_eq!(
                resolve(symbol, &mut function, version, flags, &mut query),
                Success
            );
            assert!(function.is_null());
            assert_eq!(query, expected);
        }
        for (name, legacy, explicit) in [
            (
                c"cuCtxGetDevice",
                cuCtxGetDevice as Opaque,
                cuCtxGetDevice_v2 as Opaque,
            ),
            (
                c"cuCtxSynchronize",
                cuCtxSynchronize as Opaque,
                cuCtxSynchronize_v2 as Opaque,
            ),
        ] {
            assert_eq!(lookup(name.to_bytes(), 12000, false), Ok(legacy));
            assert_eq!(lookup(name.to_bytes(), 13000, false), Ok(explicit));
        }
        assert_eq!(
            resolve(c"cuGetExportTable", &mut function, 3000, 0, &mut query),
            Success
        );
        assert!(!function.is_null());
        assert_eq!(query, FOUND);
        assert_eq!(
            resolve(c"cuInit", &mut function, DRIVER_ABI + 10, 0, &mut query),
            InvalidValue
        );
        assert_eq!(
            resolve(c"cuInit", &mut function, DRIVER_ABI, 3, &mut query),
            InvalidValue
        );
        assert_eq!(
            unsafe { cuGetProcAddress(c"cuGetExportTable".as_ptr(), &mut function, 3000, 0) },
            Success
        );
        assert!(!function.is_null());
        assert_eq!(
            resolve(c"cuInit", &mut function, DRIVER_ABI, 2, &mut query),
            Success
        );
        let init: extern "system" fn(c_uint) -> CuResult = unsafe { std::mem::transmute(function) };
        assert_eq!(init(1), InvalidValue);
    }

    #[test]
    fn resolves_driver_gap_entry_points_with_versioned_names() {
        let at = |name: &CStr, version, per_thread| lookup(name.to_bytes(), version, per_thread);
        // Unversioned base names resolve to the current _v2 implementation.
        assert_eq!(at(c"cuMemGetAddressRange", 12000, false), Ok(cuMemGetAddressRange_v2 as Opaque));
        assert_eq!(at(c"cuMemHostRegister", 12000, false), Ok(cuMemHostRegister_v2 as Opaque));
        assert_eq!(at(c"cuMemcpyDtoD", 12000, false), Ok(cuMemcpyDtoD_v2 as Opaque));
        assert_eq!(at(c"cuMemcpyDtoDAsync", 12000, false), Ok(cuMemcpyDtoDAsync_v2 as Opaque));
        assert_eq!(at(c"cuMemcpy2D", 12000, false), Ok(cuMemcpy2D_v2 as Opaque));
        assert_eq!(at(c"cuMemsetD32", 12000, false), Ok(cuMemsetD32_v2 as Opaque));
        assert_eq!(at(c"cuCtxPushCurrent", 12000, false), Ok(cuCtxPushCurrent_v2 as Opaque));
        // The suffixed names are not base names.
        assert_eq!(at(c"cuMemcpy2D_v2", 12000, false), Err(MISSING));
        // Advice and prefetch changed signature in CUDA 12.2.
        assert_eq!(at(c"cuMemAdvise", 12010, false), Ok(cuMemAdvise as Opaque));
        assert_eq!(at(c"cuMemAdvise", 12020, false), Ok(cuMemAdvise_v2 as Opaque));
        assert_eq!(at(c"cuMemPrefetchAsync", 12010, false), Ok(cuMemPrefetchAsync as Opaque));
        assert_eq!(at(c"cuMemPrefetchAsync", 13000, false), Ok(cuMemPrefetchAsync_v2 as Opaque));
        assert_eq!(at(c"cuMemAdvise", 7000, false), Err(VERSION));
        // Entry points are absent before the release that introduced them.
        assert_eq!(at(c"cuLaunchKernelEx", 11080, false), Err(VERSION));
        assert_eq!(at(c"cuLaunchKernelEx", 12000, false), Ok(cuLaunchKernelEx as Opaque));
        assert_eq!(at(c"cuMemAllocAsync", 11010, false), Err(VERSION));
        assert_eq!(at(c"cuMemAllocAsync", 11020, false), Ok(cuMemAllocAsync as Opaque));
        assert_eq!(at(c"cuFuncSetAttribute", 8000, false), Err(VERSION));
        assert_eq!(at(c"cuOccupancyMaxActiveBlocksPerMultiprocessor", 6040, false), Err(VERSION));
        // Stream-aware entries have no PTDS variant; others resolve for PTDS callers.
        assert_eq!(at(c"cuMemcpyAsync", 13000, true), Err(MISSING));
        assert_eq!(at(c"cuLaunchKernelEx", 13000, true), Err(MISSING));
        assert!(at(c"cuFuncGetAttribute", 13000, true).is_ok());
        assert!(at(c"cuPointerGetAttributes", 13000, true).is_ok());
        for name in [
            c"cuFuncGetAttribute",
            c"cuOccupancyMaxPotentialBlockSize",
            c"cuOccupancyMaxPotentialBlockSizeWithFlags",
            c"cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
            c"cuMemFreeAsync",
            c"cuMemPoolSetAttribute",
            c"cuDeviceGetDefaultMemPool",
            c"cuMemcpyPeer",
            c"cuCtxEnablePeerAccess",
            c"cuStreamCreateWithPriority",
        ] {
            assert!(at(name, 13000, false).is_ok(), "{name:?}");
        }
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
