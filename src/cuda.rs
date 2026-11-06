//! CUDA Driver ABI. Public pointers follow CUDA's caller validity requirements.
use crate::level_zero::{Api, Context, DeviceInfo, Error, Module, Runtime};
use crate::ptx::{self, Param};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::sync::Mutex;

mod streams;
use streams::Streams;

#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CuResult {
    Success = 0,
    InvalidValue = 1,
    OutOfMemory = 2,
    NotInitialized = 3,
    NoDevice = 100,
    InvalidDevice = 101,
    InvalidImage = 200,
    InvalidContext = 201,
    InvalidPtx = 218,
    FileNotFound = 301,
    OperatingSystem = 304,
    InvalidHandle = 400,
    NotFound = 500,
    NotReady = 600,
    ContextIsDestroyed = 709,
    LaunchFailed = 719,
    NotSupported = 801,
    Unknown = 999,
}

use CuResult::*;

type Opaque = *mut c_void;

struct CudaContext {
    backend: Context,
    streams: Streams,
    device: c_int,
    allocations: BTreeMap<usize, usize>,
    host_allocations: BTreeMap<usize, usize>,
    modules: HashMap<usize, CudaModule>,
    functions: HashMap<usize, usize>,
}

struct CudaModule {
    backend: Module,
    name: CString,
    params: Vec<Param>,
    max_threads: Option<u32>,
    function: Option<usize>,
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
    if !matches!(error, Error::Driver(_, 1)) {
        diagnostic(&error);
    }
    match error {
        Error::Driver(_, 1) => NotReady,
        Error::Loader(_) | Error::Driver("zeInit", 0x78000001) => NoDevice,
        Error::Driver(_, 0x70000001 | 0x70000006) | Error::Build(0x70000001 | 0x70000006, _) => {
            LaunchFailed
        }
        Error::Driver(_, 0x70000002 | 0x70000003) | Error::Build(0x70000002 | 0x70000003, _) => {
            OutOfMemory
        }
        Error::Driver(_, 0x78000002 | 0x78000003) | Error::Build(0x78000002 | 0x78000003, _) => {
            NotSupported
        }
        Error::Driver(_, 0x78000004 | 0x78000007..=0x7800000a) => InvalidValue,
        Error::Driver(_, 0x78000005) => InvalidContext,
        Error::Driver(_, 0x78000011) => NotFound,
        Error::Driver(_, 0x7800001c) => InvalidHandle,
        Error::Driver(_, 0x78000013..=0x78000017) => InvalidValue,
        Error::Build(0x70000004 | 0x70000005, _) => InvalidPtx,
        _ => Unknown,
    }
}

fn diagnostic(error: &impl std::fmt::Display) {
    if std::env::var_os("CUDA_FOR_INTEL_LOG").as_deref() == Some(std::ffi::OsStr::new("1")) {
        use std::io::Write;
        let _ = writeln!(std::io::stderr().lock(), "CUDA for Intel: {error}");
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
    with_context_handles(|runtime, context, _| action(runtime, context))
}

fn with_context_handles<T>(
    action: impl FnOnce(&Runtime, &mut CudaContext, &mut usize) -> Result<T, CuResult>,
) -> Result<T, CuResult> {
    state(|state| {
        let key = current().ok_or(InvalidContext)?;
        let context = state.contexts.get_mut(&key).ok_or(ContextIsDestroyed)?;
        action(&state.runtime, context, &mut state.next_handle)
    })
}

fn reserve_handle(next: &mut usize) -> Result<usize, CuResult> {
    let key = *next;
    *next = key.checked_add(1).ok_or(OutOfMemory)?;
    Ok(key)
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
                next_handle: 3, // CUDA reserves stream values 0, 1 and 2.
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
        let key = reserve_handle(&mut state.next_handle)?;
        let api = &state.runtime.api;
        let mut backend = Context::create(api, device).map_err(backend_error)?;
        let streams = match Streams::new(api, &backend) {
            Ok(streams) => streams,
            Err(error) => {
                if let Err(cleanup) = backend.destroy(api) {
                    diagnostic(&cleanup);
                }
                return Err(error);
            }
        };
        state.contexts.insert(
            key,
            CudaContext {
                backend,
                streams,
                device: ordinal,
                allocations: BTreeMap::new(),
                host_allocations: BTreeMap::new(),
                modules: HashMap::new(),
                functions: HashMap::new(),
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
        // Context teardown cannot invalidate a list another host call is using.
        if context.streams.has_waiters() {
            return Err(NotReady);
        }
        context.streams.wait_all(api)?;
        while let Some(&key) = context.modules.keys().next() {
            context.unload(api, key)?;
        }
        while let Some((&base, _)) = context.allocations.first_key_value() {
            context.free(api, base)?;
        }
        while let Some((&base, _)) = context.host_allocations.first_key_value() {
            context.free_host(api, base)?;
        }
        context.streams.destroy(api)?;
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
    status(state(|_| {
        if out.is_null() {
            return Err(InvalidValue);
        }
        let key = current().unwrap_or(0);
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
    streams::synchronize_context()
}

impl CudaContext {
    fn free_host(&mut self, api: &Api, base: usize) -> Result<(), CuResult> {
        self.backend.free(api, base).map_err(backend_error)?;
        self.host_allocations.remove(&base);
        Ok(())
    }

    fn free(&mut self, api: &Api, base: usize) -> Result<(), CuResult> {
        if !self.allocations.contains_key(&base) {
            return Err(InvalidValue);
        }
        self.backend.free(api, base).map_err(backend_error)?;
        self.allocations.remove(&base);
        Ok(())
    }

    fn unload(&mut self, api: &Api, key: usize) -> Result<(), CuResult> {
        let module = self.modules.get_mut(&key).ok_or(InvalidHandle)?;
        module.backend.destroy(api).map_err(backend_error)?;
        if let Some(function) = module.function {
            self.functions.remove(&function);
        }
        self.modules.remove(&key);
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
        if !context.allocations.contains_key(&(address as usize)) {
            return Err(InvalidValue);
        }
        context.streams.wait_all(&runtime.api)?;
        context.free(&runtime.api, address as usize)
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemHostAlloc(
    out: *mut Opaque,
    size: usize,
    flags: c_uint,
) -> CuResult {
    status(with_context(|runtime, context| {
        if out.is_null() || size == 0 || flags & !7 != 0 {
            return Err(InvalidValue);
        }
        if flags != 0 {
            return Err(NotSupported);
        }
        if size as u64 > runtime.devices[context.device as usize].info.max_allocation {
            return Err(OutOfMemory);
        }
        let base = context
            .backend
            .alloc_host(&runtime.api, size)
            .map_err(backend_error)?;
        context.host_allocations.insert(base, size);
        unsafe { out.write(std::ptr::with_exposed_provenance_mut(base)) };
        Ok(())
    }))
}

#[unsafe(no_mangle)]
pub extern "system" fn cuMemFreeHost(host: Opaque) -> CuResult {
    status(with_context(|runtime, context| {
        if !context.host_allocations.contains_key(&host.addr()) {
            return Err(InvalidValue);
        }
        context.streams.wait_all(&runtime.api)?;
        context.free_host(&runtime.api, host.addr())
    }))
}

unsafe fn copy_async(
    host: Opaque,
    device: u64,
    size: usize,
    stream: Opaque,
    to_device: bool,
) -> CuResult {
    status(with_context(|runtime, context| {
        let key = context.streams.launch_key(stream)?;
        if size == 0 {
            return Ok(());
        }
        if !contains_range(&context.allocations, device as usize, size)
            || !contains_range(&context.host_allocations, host.addr(), size)
        {
            return Err(InvalidValue);
        }
        let device = std::ptr::with_exposed_provenance_mut(device as usize);
        let (dst, src) = if to_device {
            (device, host)
        } else {
            (host, device)
        };
        unsafe {
            context
                .streams
                .copy(&runtime.api, &context.backend, key, (dst, src), size)
        }
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemcpyHtoDAsync_v2(
    dst: u64,
    src: *const c_void,
    size: usize,
    stream: Opaque,
) -> CuResult {
    unsafe { copy_async(src.cast_mut(), dst, size, stream, true) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemcpyDtoHAsync_v2(
    dst: Opaque,
    src: u64,
    size: usize,
    stream: Opaque,
) -> CuResult {
    unsafe { copy_async(dst, src, size, stream, false) }
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
        context.streams.wait_legacy(&runtime.api, u64::MAX)?;
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

fn load_ptx(
    runtime: &Runtime,
    context: &mut CudaContext,
    next: &mut usize,
    image: &[u8],
) -> Result<usize, CuResult> {
    let ptx = std::str::from_utf8(image).map_err(|_| InvalidImage)?;
    let translated = ptx::translate(ptx).map_err(|error| {
        diagnostic(&error);
        InvalidPtx
    })?;
    let device = &runtime.devices[context.device as usize];
    let info = &device.info;
    if info.spirv_version < 0x10000
        || (translated.fp32
            && (!info.fp32_denorms || !info.fp32_round_to_nearest || !device.fp32_inf_nan))
        || (translated.fp32_fma && !device.fp32_fma)
    {
        return Err(NotSupported);
    }
    if translated.device_atomics && !device.device_memory_atomics {
        return Err(NotSupported);
    }
    let argument_bytes = translated.params.iter().fold(0usize, |offset, &param| {
        offset.next_multiple_of(param.size()) + param.size()
    });
    if argument_bytes > info.max_kernel_arguments_size as usize
        || translated.shared_bytes > info.max_shared_local_memory
    {
        return Err(NotSupported);
    }
    let name = CString::new(translated.name).map_err(|_| InvalidPtx)?;
    let key = reserve_handle(next)?;
    let backend = context
        .backend
        .load_module(&runtime.api, &translated.words)
        .map_err(backend_error)?;
    context.modules.insert(
        key,
        CudaModule {
            backend,
            name,
            params: translated.params,
            max_threads: translated.max_threads,
            function: None,
        },
    );
    Ok(key)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuModuleLoad(out: *mut Opaque, path: *const c_char) -> CuResult {
    status(with_context_handles(|runtime, context, next| {
        if out.is_null() || path.is_null() {
            return Err(InvalidValue);
        }
        let path = unsafe { CStr::from_ptr(path) }
            .to_str()
            .map_err(|_| InvalidValue)?;
        let image = std::fs::read(path).map_err(|error| match error.kind() {
            std::io::ErrorKind::NotFound => FileNotFound,
            _ => OperatingSystem,
        })?;
        let key = load_ptx(runtime, context, next, &image)?;
        unsafe { out.write(std::ptr::without_provenance_mut(key)) };
        Ok(())
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuModuleLoadData(out: *mut Opaque, image: *const c_void) -> CuResult {
    status(with_context_handles(|runtime, context, next| {
        if out.is_null() || image.is_null() {
            return Err(InvalidValue);
        }
        // Recognize ELF and CUDA fatbin headers before treating the input as C text.
        if matches!(unsafe { image.cast::<u8>().read() }, 0x7f | 0x50 | 0xb1) {
            return Err(InvalidImage);
        }
        let image = unsafe { CStr::from_ptr(image.cast()) }.to_bytes();
        let key = load_ptx(runtime, context, next, image)?;
        unsafe { out.write(std::ptr::without_provenance_mut(key)) };
        Ok(())
    }))
}

#[unsafe(no_mangle)]
pub extern "system" fn cuModuleUnload(module: Opaque) -> CuResult {
    status(with_context(|runtime, context| {
        if !context.modules.contains_key(&module.addr()) {
            return Err(InvalidHandle);
        }
        context.streams.wait_all(&runtime.api)?;
        context.unload(&runtime.api, module.addr())
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuModuleGetFunction(
    out: *mut Opaque,
    module: Opaque,
    name: *const c_char,
) -> CuResult {
    status(with_context_handles(|runtime, context, next| {
        if out.is_null() || name.is_null() {
            return Err(InvalidValue);
        }
        let module_key = module.addr();
        let module = context.modules.get_mut(&module_key).ok_or(InvalidHandle)?;
        if unsafe { CStr::from_ptr(name) } != module.name.as_c_str() {
            return Err(NotFound);
        }
        let function = match module.function {
            Some(key) => key,
            None => reserve_handle(next)?,
        };
        module
            .backend
            .create_kernel(&runtime.api, &module.name)
            .map_err(backend_error)?;
        module.function = Some(function);
        context.functions.insert(function, module_key);
        unsafe { out.write(std::ptr::without_provenance_mut(function)) };
        Ok(())
    }))
}

fn validate_launch(
    info: &DeviceInfo,
    grid: [u32; 3],
    block: [u32; 3],
    max_threads: Option<u32>,
) -> Result<(), CuResult> {
    let fits = |size: [u32; 3], limits: [u32; 3]| {
        size.into_iter()
            .zip(limits)
            .all(|(size, limit)| (1..=limit).contains(&size))
    };
    if !fits(grid, info.max_grid_dimensions) || !fits(block, info.max_workgroup_dimensions) {
        return Err(InvalidValue);
    }
    let total = block
        .into_iter()
        .try_fold(1u32, u32::checked_mul)
        .ok_or(InvalidValue)?;
    if total > info.max_workgroup_size || max_threads.is_some_and(|limit| total > limit) {
        return Err(InvalidValue);
    }
    Ok(())
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuLaunchKernel(
    function: Opaque,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    block_x: u32,
    block_y: u32,
    block_z: u32,
    shared_bytes: u32,
    stream: Opaque,
    kernel_params: *mut Opaque,
    extra: *mut Opaque,
) -> CuResult {
    status(with_context(|runtime, context| {
        if shared_bytes != 0 || !extra.is_null() {
            return Err(NotSupported);
        }
        let stream = context.streams.launch_key(stream)?;
        let module_key = context
            .functions
            .get(&function.addr())
            .ok_or(InvalidHandle)?;
        let module = context.modules.get(module_key).ok_or(InvalidHandle)?;
        let grid = [grid_x, grid_y, grid_z];
        let block = [block_x, block_y, block_z];
        validate_launch(
            &runtime.devices[context.device as usize].info,
            grid,
            block,
            module.max_threads,
        )?;
        if kernel_params.is_null() && !module.params.is_empty() {
            return Err(InvalidValue);
        }
        let mut args = Vec::with_capacity(module.params.len());
        for (index, &param) in module.params.iter().enumerate() {
            let value = unsafe { kernel_params.add(index).read() };
            if value.is_null() {
                return Err(InvalidValue);
            }
            if param == Param::GlobalPtr {
                let address = unsafe { value.cast::<u64>().read_unaligned() };
                if !contains_range(&context.allocations, address as usize, 1) {
                    return Err(InvalidValue);
                }
            }
            args.push(unsafe { std::slice::from_raw_parts(value.cast::<u8>(), param.size()) });
        }
        context.streams.launch(
            &runtime.api,
            &context.backend,
            &module.backend,
            stream,
            (grid, block),
            &args,
        )
    }))
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
