//! CUDA Driver ABI. Public pointers follow CUDA's caller validity requirements.
use crate::level_zero::{Api, Context, DeviceInfo, Error, Runtime};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::ffi::{c_char, c_int, c_uint, c_void};
use std::sync::Mutex;

#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CuResult {
    Success = 0,
    InvalidValue = 1,
    OutOfMemory = 2,
    NotInitialized = 3,
    NoDevice = 100,
    InvalidDevice = 101,
    InvalidContext = 201,
    LaunchFailed = 719,
    NotSupported = 801,
    Unknown = 999,
}

use CuResult::*;

type Opaque = *mut c_void;

struct CudaContext {
    backend: Context,
    device: c_int,
    allocations: BTreeMap<usize, usize>,
}

struct State {
    runtime: Runtime,
    next_handle: usize,
    contexts: HashMap<usize, CudaContext>,
}

static RUNTIME: Mutex<Option<State>> = Mutex::new(None);
thread_local! {
    static CURRENT: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

fn backend_error(error: Error) -> CuResult {
    match error {
        Error::Loader(_) | Error::Driver("zeInit", 0x78000001) => NoDevice,
        Error::Driver(_, 0x70000001 | 0x70000006) => LaunchFailed,
        Error::Driver(_, 0x70000002 | 0x70000003) => OutOfMemory,
        Error::Driver(_, 0x78000002 | 0x78000003) => NotSupported,
        Error::Driver(_, 0x78000004 | 0x78000007..=0x7800000a) => InvalidValue,
        Error::Driver(_, 0x78000005) => InvalidContext,
        _ => Unknown,
    }
}

fn runtime<T>(action: impl FnOnce(&Runtime) -> Result<T, CuResult>) -> Result<T, CuResult> {
    state(|state| action(&state.runtime))
}

fn state<T>(action: impl FnOnce(&mut State) -> Result<T, CuResult>) -> Result<T, CuResult> {
    let mut guard = RUNTIME.lock().map_err(|_| Unknown)?;
    action(guard.as_mut().ok_or(NotInitialized)?)
}

fn status(result: Result<(), CuResult>) -> CuResult {
    result.map_or_else(|error| error, |()| Success)
}

fn current() -> Option<usize> {
    CURRENT.with(|stack| stack.borrow().last().copied())
}

fn with_context<T>(
    action: impl FnOnce(&Runtime, &mut CudaContext) -> Result<T, CuResult>,
) -> Result<T, CuResult> {
    state(|state| {
        let key = current().ok_or(InvalidContext)?;
        let context = state.contexts.get_mut(&key).ok_or(InvalidContext)?;
        action(&state.runtime, context)
    })
}

fn device(ordinal: c_int, action: impl FnOnce(&DeviceInfo) -> CuResult) -> CuResult {
    runtime(|rt| {
        let device = rt.devices.get(ordinal as usize).ok_or(InvalidDevice)?;
        Ok(action(&device.info))
    })
    .unwrap_or_else(|error| error)
}

#[unsafe(no_mangle)]
pub extern "system" fn cuInit(flags: c_uint) -> CuResult {
    if flags != 0 {
        return InvalidValue;
    }
    let Ok(mut guard) = RUNTIME.lock() else {
        return Unknown;
    };
    if guard.is_some() {
        return Success;
    }
    match Runtime::load() {
        Ok(rt) if rt.devices.is_empty() => NoDevice,
        Ok(rt) => {
            *guard = Some(State {
                runtime: rt,
                next_handle: 1,
                contexts: HashMap::new(),
            });
            Success
        }
        Err(error) => backend_error(error),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuCtxCreate_v2(
    out: *mut Opaque,
    flags: c_uint,
    ordinal: c_int,
) -> CuResult {
    status(state(|state| {
        if out.is_null() {
            return Err(InvalidValue);
        }
        // Only the default scheduling policy and no additional flags are supported.
        if flags != 0 {
            return Err(NotSupported);
        }
        let device = state
            .runtime
            .devices
            .get(ordinal as usize)
            .ok_or(InvalidDevice)?;
        let key = state.next_handle;
        let next = key.checked_add(1).ok_or(OutOfMemory)?;
        let backend = Context::create(&state.runtime.api, device).map_err(backend_error)?;
        state.next_handle = next;
        state.contexts.insert(
            key,
            CudaContext {
                backend,
                device: ordinal,
                allocations: BTreeMap::new(),
            },
        );
        CURRENT.with(|stack| stack.borrow_mut().push(key));
        unsafe { out.write(std::ptr::without_provenance_mut(key)) };
        Ok(())
    }))
}

#[unsafe(no_mangle)]
pub extern "system" fn cuCtxDestroy_v2(handle: Opaque) -> CuResult {
    status(state(|state| {
        let key = handle.addr();
        let context = state.contexts.get_mut(&key).ok_or(InvalidContext)?;
        let api = &state.runtime.api;
        while let Some((&base, _)) = context.allocations.first_key_value() {
            context.free(api, base)?;
        }
        context.backend.destroy(api).map_err(backend_error)?;
        state.contexts.remove(&key);
        CURRENT.with(|stack| {
            let mut stack = stack.borrow_mut();
            if stack.last() == Some(&key) {
                stack.pop();
            }
        });
        Ok(())
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuCtxGetCurrent(out: *mut Opaque) -> CuResult {
    status(state(|state| {
        if out.is_null() {
            return Err(InvalidValue);
        }
        let key = current().unwrap_or(0);
        if key != 0 && !state.contexts.contains_key(&key) {
            return Err(InvalidContext);
        }
        unsafe { out.write(std::ptr::without_provenance_mut(key)) };
        Ok(())
    }))
}

#[unsafe(no_mangle)]
pub extern "system" fn cuCtxSetCurrent(handle: Opaque) -> CuResult {
    status(state(|state| {
        let key = handle.addr();
        if key != 0 && !state.contexts.contains_key(&key) {
            return Err(InvalidContext);
        }
        CURRENT.with(|stack| {
            let mut stack = stack.borrow_mut();
            stack.pop();
            if key != 0 {
                stack.push(key);
            }
        });
        Ok(())
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuCtxGetDevice(out: *mut c_int) -> CuResult {
    status(with_context(|_, context| {
        if out.is_null() {
            return Err(InvalidValue);
        }
        unsafe { out.write(context.device) };
        Ok(())
    }))
}

#[unsafe(no_mangle)]
pub extern "system" fn cuCtxSynchronize() -> CuResult {
    status(with_context(|runtime, context| {
        context
            .backend
            .synchronize(&runtime.api)
            .map_err(backend_error)
    }))
}

impl CudaContext {
    fn free(&mut self, api: &Api, base: usize) -> Result<(), CuResult> {
        if !self.allocations.contains_key(&base) {
            return Err(InvalidValue);
        }
        self.backend.free(api, base).map_err(backend_error)?;
        self.allocations.remove(&base);
        Ok(())
    }
}

fn contains_range(allocations: &BTreeMap<usize, usize>, address: usize, size: usize) -> bool {
    allocations
        .range(..=address)
        .next_back()
        .is_some_and(|(&base, &length)| {
            let offset = address - base;
            offset < length && size <= length - offset
        })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemAlloc_v2(out: *mut u64, size: usize) -> CuResult {
    status(with_context(|runtime, context| {
        if out.is_null() || size == 0 {
            return Err(InvalidValue);
        }
        if size as u64 > runtime.devices[context.device as usize].info.max_allocation {
            return Err(OutOfMemory);
        }
        let address = context
            .backend
            .alloc(&runtime.api, size)
            .map_err(backend_error)?;
        context.allocations.insert(address, size);
        unsafe { out.write(address as u64) };
        Ok(())
    }))
}

#[unsafe(no_mangle)]
pub extern "system" fn cuMemFree_v2(address: u64) -> CuResult {
    status(with_context(|runtime, context| {
        if address == 0 {
            return Ok(());
        }
        context.free(&runtime.api, address as usize)
    }))
}

unsafe fn copy(host: Opaque, device: u64, size: usize, to_device: bool) -> CuResult {
    status(with_context(|runtime, context| {
        if size == 0 {
            return Ok(());
        }
        let address = device as usize;
        if host.is_null() || !contains_range(&context.allocations, address, size) {
            return Err(InvalidValue);
        }
        let device = std::ptr::with_exposed_provenance_mut(address);
        let (dst, src) = if to_device {
            (device, host)
        } else {
            (host, device)
        };
        unsafe { context.backend.copy(&runtime.api, dst, src, size) }.map_err(backend_error)
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemcpyHtoD_v2(
    dst: u64,
    src: *const c_void,
    size: usize,
) -> CuResult {
    unsafe { copy(src.cast_mut(), dst, size, true) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemcpyDtoH_v2(dst: Opaque, src: u64, size: usize) -> CuResult {
    unsafe { copy(dst, src, size, false) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuDeviceGetCount(count: *mut c_int) -> CuResult {
    runtime(|rt| {
        if count.is_null() {
            return Err(InvalidValue);
        }
        let value = c_int::try_from(rt.devices.len()).map_err(|_| Unknown)?;
        unsafe { count.write(value) };
        Ok(Success)
    })
    .unwrap_or_else(|error| error)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuDeviceGet(out: *mut c_int, ordinal: c_int) -> CuResult {
    device(ordinal, |_| {
        if out.is_null() {
            return InvalidValue;
        }
        unsafe { out.write(ordinal) };
        Success
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuDeviceGetName(
    out: *mut c_char,
    len: c_int,
    ordinal: c_int,
) -> CuResult {
    device(ordinal, |info| {
        if out.is_null() || len <= 0 {
            return InvalidValue;
        }
        let bytes = info.name.as_bytes();
        let copied = bytes.len().min(len as usize - 1);
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.cast::<u8>(), copied);
            out.add(copied).write(0);
        }
        Success
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuDeviceTotalMem_v2(out: *mut usize, ordinal: c_int) -> CuResult {
    device(ordinal, |info| {
        if out.is_null() {
            return InvalidValue;
        }
        unsafe { out.write(info.total_memory as usize) };
        Success
    })
}

fn attribute(info: &DeviceInfo, attr: c_int) -> Result<c_int, CuResult> {
    let value = match attr {
        1 => info.max_workgroup_size,
        2..=4 => info.max_workgroup_dimensions[(attr - 2) as usize],
        5..=7 => info.max_grid_dimensions[(attr - 5) as usize],
        8 => info.max_shared_local_memory,
        _ => return Err(NotSupported),
    };
    // CUDA uses a signed int even where the backend can represent larger limits.
    Ok(value.min(c_int::MAX as u32) as c_int)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuDeviceGetAttribute(
    out: *mut c_int,
    attr: c_int,
    ordinal: c_int,
) -> CuResult {
    device(ordinal, |info| {
        if out.is_null() {
            return InvalidValue;
        }
        match attribute(info, attr) {
            Ok(value) => {
                unsafe { out.write(value) };
                Success
            }
            Err(error) => error,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_reserved_init_flags_without_a_driver() {
        assert_eq!(cuInit(1), InvalidValue);
    }

    #[test]
    fn device_queries_require_initialization() {
        let mut out = -1;
        assert_eq!(unsafe { cuDeviceGetCount(&mut out) }, NotInitialized);
        assert_eq!(out, -1);
    }

    #[test]
    fn device_ranges_must_fit_one_allocation() {
        let allocations = BTreeMap::from([(100, 10), (110, 5), (usize::MAX - 8, 8)]);
        assert!(contains_range(&allocations, 102, 8));
        assert!(!contains_range(&allocations, 102, 9));
        assert!(!contains_range(&allocations, 115, 1));
        assert!(!contains_range(&allocations, 99, 1));
        assert!(!contains_range(&allocations, usize::MAX - 1, usize::MAX));
    }
}
