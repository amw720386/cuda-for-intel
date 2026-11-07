//! CUDA Driver ABI. Public pointers follow CUDA's caller validity requirements.
use crate::level_zero::{Api, Context, DeviceInfo, Error, Handle, Module, Runtime};
use crate::ptx::{self, Param};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::sync::Mutex;

mod current_stack;
mod discovery;
#[cfg(windows)]
mod dxgi_luid;
mod export_table;
mod gap_math;
mod gaps;
mod graphs;
#[cfg(windows)]
mod integrity;
mod launch_args;
mod library;
mod primary;
mod streams;
mod trace;
mod vmm;
use streams::Streams;

#[cfg(all(test, windows))]
#[path = "../tests/gpu/device.rs"]
mod gpu_device;
#[cfg(all(test, windows))]
mod ollama_bench;
#[cfg(all(test, windows))]
mod transfer_bench;
#[cfg(all(test, windows))]
use gpu_device::ordinal as test_device;

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
    NoBinaryForGpu = 209,
    InvalidContext = 201,
    InvalidPtx = 218,
    PeerAccessUnsupported = 216,
    FileNotFound = 301,
    OperatingSystem = 304,
    InvalidHandle = 400,
    IllegalState = 401,
    NotFound = 500,
    NotReady = 600,
    PeerAccessAlreadyEnabled = 704,
    PeerAccessNotEnabled = 705,
    PrimaryContextActive = 708,
    ContextIsDestroyed = 709,
    HostMemoryAlreadyRegistered = 712,
    HostMemoryNotRegistered = 713,
    LaunchFailed = 719,
    NotSupported = 801,
    StreamCaptureUnsupported = 900,
    StreamCaptureInvalidated = 901,
    StreamCaptureMerge = 902,
    StreamCaptureUnmatched = 903,
    StreamCaptureUnjoined = 904,
    StreamCaptureIsolation = 905,
    CapturedEvent = 907,
    GraphExecUpdateFailure = 910,
    Unknown = 999,
}

use CuResult::*;

type Opaque = *mut c_void;

struct CudaContext {
    backend: Context,
    streams: Streams,
    device: c_int,
    teardown_error: Option<CuResult>,
    allocations: BTreeMap<usize, usize>,
    host_allocations: BTreeMap<usize, usize>,
    /// Application memory registered with cuMemHostRegister (bookkeeping only).
    host_registered: BTreeMap<usize, usize>,
    modules: HashMap<usize, CudaModule>,
    library_modules: HashMap<usize, usize>,
    functions: HashMap<usize, (usize, CString)>,
    local_storage: HashMap<usize, export_table::ContextStorage>,
}

struct CudaModule {
    backend: Module,
    entries: HashMap<CString, CudaEntry>,
    unloading: bool,
}

struct CudaEntry {
    params: Vec<Param>,
    max_threads: Option<u32>,
    /// SPIR-V workgroup shared buffer size (bytes); bounds CUDA dynamic shared launches.
    shared_bytes: u32,
    function: Option<(usize, Handle)>,
}

struct State {
    runtime: Runtime,
    next_handle: usize,
    contexts: HashMap<usize, CudaContext>,
    /// Explicit stream handle -> owning context key. Entries can go stale after a
    /// stream or context is destroyed, so every hit is re-validated.
    stream_owners: HashMap<usize, usize>,
    incomplete_contexts: Vec<Context>,
    primary: HashMap<c_int, primary::Primary>,
    libraries: library::Libraries,
    graphs: graphs::Graphs,
}

static RUNTIME: Mutex<Option<State>> = Mutex::new(None);
#[cfg(test)]
static TEST_RUNTIME: Mutex<()> = Mutex::new(());
thread_local! {
    static CURRENT: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

fn backend_error(error: Error) -> CuResult {
    if trace::enabled() {
        match &error {
            Error::Driver(_, 1) => {}
            Error::Driver(call, status) => {
                trace::record(format_args!("Level Zero {call} -> {status:#010x}"))
            }
            Error::Build(status, _) => {
                trace::record(format_args!("Level Zero zeModuleCreate -> {status:#010x}"))
            }
            Error::Loader(_) => trace::record(format_args!("Level Zero loader unavailable")),
            Error::InvalidProperties => {
                trace::record(format_args!("Level Zero invalid device properties"))
            }
        }
    }
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

fn dump_launch_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("CUDA_FOR_INTEL_DUMP_LAUNCH").is_some())
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
    CURRENT.with(|stack| current_stack::top(&stack.borrow()))
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
        context.check_live()?;
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

/// Bring-up log file. Opening it costs a syscall per call, and the Runtime shim
/// retains the primary context on every launch, so it is opt-in
/// (`CUDA_FOR_INTEL_LOG=1`, `CUDA_FOR_INTEL_TRACE=1` or `CUDA_FOR_INTEL_EXPORT_LOG=1`).
fn export_log_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        ["CUDA_FOR_INTEL_LOG", "CUDA_FOR_INTEL_TRACE", "CUDA_FOR_INTEL_EXPORT_LOG"]
            .iter()
            .any(|name| std::env::var_os(name).as_deref() == Some(std::ffi::OsStr::new("1")))
    })
}

pub(crate) fn export_log(message: impl std::fmt::Display) {
    if !export_log_enabled() {
        return;
    }
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("cuda-for-intel-export.log")
    {
        use std::io::Write;
        let _ = writeln!(file, "{message}");
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn cuInit(flags: c_uint) -> CuResult {
    export_log(format_args!("cuInit flags={flags} enter"));
    let result = trace::call("cuInit", || {
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
                if trace::enabled() {
                    trace::record(format_args!(
                        "initialized {} Level Zero device(s)",
                        rt.devices.len()
                    ));
                }
                *guard = Some(State {
                    runtime: rt,
                    next_handle: 3, // CUDA reserves stream values 0, 1 and 2.
                    contexts: HashMap::new(),
                    stream_owners: HashMap::new(),
                    incomplete_contexts: Vec::new(),
                    primary: HashMap::new(),
                    libraries: library::Libraries::default(),
                    graphs: graphs::Graphs::default(),
                });
                Success
            }
            Err(error) => backend_error(error),
        }
    });
    export_log(format_args!("cuInit flags={flags} -> {}", result as i32));
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuCtxCreate_v2(
    out: *mut Opaque,
    flags: c_uint,
    ordinal: c_int,
) -> CuResult {
    trace::call("cuCtxCreate_v2", || {
        status(state(|state| {
            if out.is_null() {
                return Err(InvalidValue);
            }
            // Only the default scheduling policy and no additional flags are supported.
            if flags != 0 {
                return Err(NotSupported);
            }
            let key = state.create_context(ordinal)?;
            CURRENT.with(|stack| stack.borrow_mut().push(key));
            unsafe { out.write(std::ptr::without_provenance_mut(key)) };
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuCtxDestroy_v2(handle: Opaque) -> CuResult {
    trace::call("cuCtxDestroy_v2", || {
        let destroyed = state(|state| {
            let key = handle.addr();
            if state
                .primary
                .values()
                .any(|primary| primary.key == Some(key))
            {
                return Err(InvalidContext);
            }
            let teardown = state.destroy_context(key)?;
            if !state.contexts.contains_key(&key) {
                state.graphs.drop_context(key);
                CURRENT.with_borrow_mut(|stack| {
                    if stack.last() == Some(&key) {
                        stack.pop();
                    }
                });
            }
            Ok(teardown)
        });
        finish_destroy(destroyed)
    })
}

type ContextTeardown = (
    Result<(), CuResult>,
    Option<std::thread::JoinHandle<Result<(), CuResult>>>,
    Option<(usize, HashMap<usize, export_table::ContextStorage>)>,
);

fn finish_destroy(destroyed: Result<ContextTeardown, CuResult>) -> CuResult {
    status(destroyed.and_then(|(result, worker, storage)| {
        let stopped = worker.map_or(Ok(()), |worker| {
            worker.join().map_err(|_| Unknown).and_then(|result| result)
        });
        if let Some((context, storage)) = storage {
            for (key, value) in storage {
                value.destroy(context, key);
            }
        }
        result.and(stopped)
    }))
}

impl State {
    fn create_context(&mut self, ordinal: c_int) -> Result<usize, CuResult> {
        let device = self
            .runtime
            .devices
            .get(ordinal as usize)
            .ok_or(InvalidDevice)?;
        let key = reserve_handle(&mut self.next_handle)?;
        let api = &self.runtime.api;
        self.incomplete_contexts
            .retain_mut(|context| context.destroy(api).inspect_err(diagnostic).is_err());
        let mut backend = Context::create(api, device).map_err(backend_error)?;
        let streams = match backend
            .initialize(api)
            .map_err(backend_error)
            .and_then(|()| Streams::new(api, &backend, key))
        {
            Ok(streams) => streams,
            Err(error) => {
                if let Err(cleanup) = backend.destroy(api) {
                    diagnostic(&cleanup);
                    self.incomplete_contexts.push(backend);
                }
                return Err(error);
            }
        };
        self.contexts.insert(
            key,
            CudaContext {
                backend,
                streams,
                device: ordinal,
                teardown_error: None,
                allocations: BTreeMap::new(),
                host_allocations: BTreeMap::new(),
                host_registered: BTreeMap::new(),
                modules: HashMap::new(),
                library_modules: HashMap::new(),
                functions: HashMap::new(),
                local_storage: HashMap::new(),
            },
        );
        if trace::enabled() {
            trace::record(format_args!("context created ordinal={ordinal}"));
        }
        Ok(key)
    }

    fn destroy_context(&mut self, key: usize) -> Result<ContextTeardown, CuResult> {
        let context = self.contexts.get_mut(&key).ok_or(InvalidContext)?;
        // An explicit host wait can still be using a native list.
        if context.streams.has_waiters() {
            return Err(NotReady);
        }
        let result = context.destroy(&self.runtime.api);
        let worker = context.streams.stop();
        if let Err(error) = result {
            // Some native resources may already have been released.
            context.teardown_error.get_or_insert(error);
            return Ok((result, worker, None));
        }
        let result = context.streams.execution_error().map_or(Ok(()), Err);
        let storage = std::mem::take(&mut context.local_storage);
        self.contexts.remove(&key);
        Ok((result, worker, Some((key, storage))))
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuCtxGetCurrent(out: *mut Opaque) -> CuResult {
    trace::call("cuCtxGetCurrent", || {
        status(state(|_| {
            if out.is_null() {
                return Err(InvalidValue);
            }
            let key = current().unwrap_or(0);
            unsafe { out.write(std::ptr::without_provenance_mut(key)) };
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuCtxSetCurrent(handle: Opaque) -> CuResult {
    trace::call("cuCtxSetCurrent", || {
        status(state(|state| {
            let key = handle.addr();
            if key != 0 {
                state
                    .contexts
                    .get(&key)
                    .ok_or(InvalidContext)?
                    .check_live()?;
            }
            CURRENT.with(|stack| current_stack::replace_top(&mut stack.borrow_mut(), key));
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuCtxGetDevice(out: *mut c_int) -> CuResult {
    trace::call("cuCtxGetDevice", || unsafe {
        context_device(out, std::ptr::null_mut())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuCtxGetDevice_v2(out: *mut c_int, handle: Opaque) -> CuResult {
    trace::call("cuCtxGetDevice_v2", || unsafe {
        context_device(out, handle)
    })
}

unsafe fn context_device(out: *mut c_int, handle: Opaque) -> CuResult {
    status(state(|state| {
        let key = context_key(handle)?;
        let context = state.contexts.get(&key).ok_or(if handle.is_null() {
            ContextIsDestroyed
        } else {
            InvalidContext
        })?;
        context.check_live()?;
        if out.is_null() {
            return Err(InvalidValue);
        }
        unsafe { out.write(context.device) };
        Ok(())
    }))
}

fn context_key(handle: Opaque) -> Result<usize, CuResult> {
    if handle.is_null() {
        current().ok_or(InvalidContext)
    } else {
        Ok(handle.addr())
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn cuCtxSynchronize() -> CuResult {
    trace::call("cuCtxSynchronize", || {
        streams::synchronize_context(std::ptr::null_mut())
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuCtxSynchronize_v2(handle: Opaque) -> CuResult {
    trace::call("cuCtxSynchronize_v2", || {
        streams::synchronize_context(handle)
    })
}

impl CudaContext {
    fn check_live(&self) -> Result<(), CuResult> {
        self.teardown_error.map_or(Ok(()), Err)?;
        if self.backend.health.lost() {
            return Err(LaunchFailed);
        }
        Ok(())
    }

    fn destroy(&mut self, api: &Api) -> Result<(), CuResult> {
        self.streams.drain_for_teardown(api)?;
        self.streams.destroy(api)?;
        while let Some(&key) = self.modules.keys().next() {
            self.unload(api, key)?;
        }
        while let Some((&base, _)) = self.allocations.first_key_value() {
            self.free(api, base)?;
        }
        while let Some((&base, _)) = self.host_allocations.first_key_value() {
            self.free_host(api, base)?;
        }
        self.backend.destroy(api).map_err(backend_error)
    }

    fn free_host(&mut self, api: &Api, base: usize) -> Result<(), CuResult> {
        self.backend
            .free(api, base)
            .inspect_err(|error| self.backend.health.observe(error))
            .map_err(backend_error)?;
        self.host_allocations.remove(&base);
        Ok(())
    }

    fn free(&mut self, api: &Api, base: usize) -> Result<(), CuResult> {
        if !self.allocations.contains_key(&base) {
            return Err(InvalidValue);
        }
        self.backend
            .free(api, base)
            .inspect_err(|error| self.backend.health.observe(error))
            .map_err(backend_error)?;
        self.allocations.remove(&base);
        Ok(())
    }

    fn unload(&mut self, api: &Api, key: usize) -> Result<(), CuResult> {
        let module = self.modules.get_mut(&key).ok_or(InvalidHandle)?;
        // A failed module destroy can follow a successful kernel destroy.
        // Keep the module for a retry, while retiring its function handles.
        module.unloading = true;
        for entry in module.entries.values_mut() {
            if let Some((function, _)) = entry.function.take() {
                self.functions.remove(&function);
            }
        }
        module
            .backend
            .destroy(api)
            .inspect_err(|error| self.backend.health.observe(error))
            .map_err(backend_error)?;
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
    trace::call("cuMemAlloc_v2", || {
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
                .inspect_err(|error| context.backend.health.observe(error))
                .map_err(backend_error)?;
            context.allocations.insert(address, size);
            unsafe { out.write(address as u64) };
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuMemFree_v2(address: u64) -> CuResult {
    trace::call("cuMemFree_v2", || {
        if address != 0 {
            streams::drain_unlocked(None);
        }
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
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemHostAlloc(
    out: *mut Opaque,
    size: usize,
    flags: c_uint,
) -> CuResult {
    trace::call("cuMemHostAlloc", || {
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
                .inspect_err(|error| context.backend.health.observe(error))
                .map_err(backend_error)?;
            context.host_allocations.insert(base, size);
            unsafe { out.write(std::ptr::with_exposed_provenance_mut(base)) };
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuMemFreeHost(host: Opaque) -> CuResult {
    trace::call("cuMemFreeHost", || {
        streams::drain_unlocked(None);
        status(with_context(|runtime, context| {
            if !context.host_allocations.contains_key(&host.addr()) {
                return Err(InvalidValue);
            }
            context.streams.wait_all(&runtime.api)?;
            context.free_host(&runtime.api, host.addr())
        }))
    })
}

unsafe fn copy_async(
    host: Opaque,
    device: u64,
    size: usize,
    stream: Opaque,
    to_device: bool,
) -> CuResult {
    // Pageable fallback waits on the stream; do that wait without the lock.
    if size != 0
        && !host.is_null()
        && let Ok(key) = streams::stream_key(stream)
        && with_context(|_, context| {
            Ok(!contains_range(&context.host_allocations, host.addr(), size))
        })
        .unwrap_or(false)
    {
        streams::drain_unlocked(Some(key));
    }
    status(with_context(|runtime, context| {
        let key = context.streams.launch_key(stream)?;
        if size == 0 {
            return Ok(());
        }
        if host.is_null() || !contains_range(&context.allocations, device as usize, size) {
            return Err(InvalidValue);
        }
        if let Some(capture) = context.streams.capture(key)? {
            // The host pointer is read (or written) at graph launch, as with a
            // CUDA memcpy node.
            let kind = if to_device {
                graphs::CopyKind::HostToDevice
            } else {
                graphs::CopyKind::DeviceToHost
            };
            let (dst, src) = if to_device {
                (device as usize, host.addr())
            } else {
                (host.addr(), device as usize)
            };
            return capture.push(key, graphs::Node::Copy(graphs::Copy { dst, src, size, kind }));
        }
        unsafe { copy_host_async(runtime, context, key, host, device, size, to_device) }
    }))
}

/// Issue a validated host/device copy on a stream (also used by graph replay).
unsafe fn copy_host_async(
    runtime: &Runtime,
    context: &mut CudaContext,
    key: usize,
    host: Opaque,
    device: u64,
    size: usize,
    to_device: bool,
) -> Result<(), CuResult> {
    let device_ptr = std::ptr::with_exposed_provenance_mut(device as usize);
    let (dst, src) = if to_device {
        (device_ptr, host)
    } else {
        (host, device_ptr)
    };
    // NVIDIA Runtime allows async copies from pageable host memory (may
    // synchronize). Our Level Zero path needs registered/pinned host for
    // true async; fall back to a blocking copy for ordinary host pointers.
    //
    // Critical: wait for *this* stream before the fallback copy. Waiting only
    // for legacy/blocking streams races with ggml's common pattern of
    // non-blocking stream compute followed by cudaMemcpyAsync(DtoH) of
    // logits / BatchedEx pointer tables into pageable host memory.
    if contains_range(&context.host_allocations, host.addr(), size) {
        unsafe {
            context
                .streams
                .copy(&runtime.api, &context.backend, key, (dst, src), size)
        }
    } else {
        context
            .streams
            .wait_stream(&runtime.api, key, u64::MAX)?;
        if context.streams.stream_is_blocking(key)? {
            context.streams.wait_legacy(&runtime.api, u64::MAX)?;
        }
        unsafe { context.backend.copy(&runtime.api, dst, src, size) }
            .inspect_err(|error| context.backend.health.observe(error))
            .map_err(backend_error)
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemcpyHtoDAsync_v2(
    dst: u64,
    src: *const c_void,
    size: usize,
    stream: Opaque,
) -> CuResult {
    trace::call("cuMemcpyHtoDAsync_v2", || unsafe {
        copy_async(src.cast_mut(), dst, size, stream, true)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemcpyDtoHAsync_v2(
    dst: Opaque,
    src: u64,
    size: usize,
    stream: Opaque,
) -> CuResult {
    trace::call("cuMemcpyDtoHAsync_v2", || unsafe {
        copy_async(dst, src, size, stream, false)
    })
}

unsafe fn copy(host: Opaque, device: u64, size: usize, to_device: bool) -> CuResult {
    if size != 0 {
        streams::drain_unlocked(Some(0));
    }
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
        unsafe { context.backend.copy(&runtime.api, dst, src, size) }
            .inspect_err(|error| context.backend.health.observe(error))
            .map_err(backend_error)
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemcpyHtoD_v2(
    dst: u64,
    src: *const c_void,
    size: usize,
) -> CuResult {
    trace::call("cuMemcpyHtoD_v2", || unsafe {
        copy(src.cast_mut(), dst, size, true)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemcpyDtoH_v2(dst: Opaque, src: u64, size: usize) -> CuResult {
    trace::call("cuMemcpyDtoH_v2", || unsafe { copy(dst, src, size, false) })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemcpyDtoD_v2(dst: u64, src: u64, size: usize) -> CuResult {
    trace::call("cuMemcpyDtoD_v2", || {
        if size != 0 {
            streams::drain_unlocked(Some(0));
        }
        status(with_context(|runtime, context| {
            if size == 0 {
                return Ok(());
            }
            let dst_addr = dst as usize;
            let src_addr = src as usize;
            if !contains_range(&context.allocations, dst_addr, size)
                || !contains_range(&context.allocations, src_addr, size)
            {
                return Err(InvalidValue);
            }
            let dst_ptr = std::ptr::with_exposed_provenance_mut(dst_addr);
            let src_ptr = std::ptr::with_exposed_provenance(src_addr);
            context.streams.wait_legacy(&runtime.api, u64::MAX)?;
            unsafe { context.backend.copy(&runtime.api, dst_ptr, src_ptr, size) }
                .inspect_err(|error| context.backend.health.observe(error))
                .map_err(backend_error)
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemcpyDtoDAsync_v2(
    dst: u64,
    src: u64,
    size: usize,
    stream: Opaque,
) -> CuResult {
    trace::call("cuMemcpyDtoDAsync_v2", || {
        status(with_context(|runtime, context| {
            let key = context.streams.launch_key(stream)?;
            if size == 0 {
                return Ok(());
            }
            let dst_addr = dst as usize;
            let src_addr = src as usize;
            if !contains_range(&context.allocations, dst_addr, size)
                || !contains_range(&context.allocations, src_addr, size)
            {
                return Err(InvalidValue);
            }
            if let Some(capture) = context.streams.capture(key)? {
                return capture.push(
                    key,
                    graphs::Node::Copy(graphs::Copy {
                        dst: dst_addr,
                        src: src_addr,
                        size,
                        kind: graphs::CopyKind::DeviceToDevice,
                    }),
                );
            }
            let dst_ptr = std::ptr::with_exposed_provenance_mut(dst_addr);
            let src_ptr = std::ptr::with_exposed_provenance(src_addr);
            unsafe {
                context
                    .streams
                    .copy(&runtime.api, &context.backend, key, (dst_ptr, src_ptr), size)
            }
        }))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuMemsetD8_v2(dst: u64, value: u8, size: usize) -> CuResult {
    trace::call("cuMemsetD8_v2", || {
        memset_bytes(dst, value, size, std::ptr::null_mut())
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuMemsetD8Async(dst: u64, value: u8, size: usize, stream: Opaque) -> CuResult {
    trace::call("cuMemsetD8Async", || memset_bytes(dst, value, size, stream))
}

fn memset_bytes(dst: u64, value: u8, size: usize, stream: Opaque) -> CuResult {
    status(with_context(|runtime, context| {
        let key = context.streams.launch_key(stream)?;
        if size == 0 {
            return Ok(());
        }
        let dst = dst as usize;
        if !contains_range(&context.allocations, dst, size) {
            return Err(InvalidValue);
        }
        if let Some(capture) = context.streams.capture(key)? {
            return capture.push(key, graphs::Node::Fill(graphs::Fill { dst, value, size }));
        }
        context.streams.fill(
            &runtime.api,
            &context.backend,
            key,
            (std::ptr::with_exposed_provenance_mut(dst), value, size),
        )
    }))
}

fn load_ptx(
    runtime: &Runtime,
    context: &mut CudaContext,
    next: &mut usize,
    image: &[u8],
) -> Result<usize, CuResult> {
    let ptx = std::str::from_utf8(image).map_err(|_| InvalidImage)?;
    let translated = ptx::translate(ptx).map_err(|error| {
        if trace::enabled() {
            let preview: String = ptx.chars().take(120).collect();
            trace::record(format_args!("PTX translation rejected: {error}; preview={preview:?}"));
        }
        diagnostic(&error);
        // Always persist a short reject breadcrumb for fatbin bring-up.
        export_log(format_args!("PTX reject: {error}"));
        InvalidPtx
    })?;
    let device = &runtime.devices[context.device as usize];
    let info = &device.info;
    if info.spirv_version < 0x10000 {
        return Err(NotSupported);
    }
    let mut entries = HashMap::with_capacity(translated.kernels.len());
    for kernel in translated.kernels {
        let argument_bytes = launch_args::size(&kernel.params);
        if (kernel.fp32 && !info.supports_fp32())
            || (kernel.fp32_fma && !info.fp32_fma)
            || (kernel.device_atomics && !info.device_memory_atomics)
            || argument_bytes > info.max_kernel_arguments_size as usize
            || kernel.shared_bytes > info.max_shared_local_memory
        {
            return Err(NotSupported);
        }
        let name = CString::new(kernel.name).map_err(|_| InvalidPtx)?;
        entries.insert(
            name,
            CudaEntry {
                params: kernel.params,
                max_threads: kernel.max_threads,
                shared_bytes: kernel.shared_bytes,
                function: None,
            },
        );
    }
    let key = reserve_handle(next)?;
    let backend = context
        .backend
        .load_module(&runtime.api, &translated.words)
        .inspect_err(|error| context.backend.health.observe(error))
        .map_err(backend_error)?;
    context.modules.insert(
        key,
        CudaModule {
            backend,
            entries,
            unloading: false,
        },
    );
    Ok(key)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuModuleLoad(out: *mut Opaque, path: *const c_char) -> CuResult {
    trace::call("cuModuleLoad", || {
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
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuModuleLoadData(out: *mut Opaque, image: *const c_void) -> CuResult {
    trace::call("cuModuleLoadData", || {
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
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuModuleUnload(module: Opaque) -> CuResult {
    trace::call("cuModuleUnload", || {
        streams::drain_unlocked(None);
        status(with_context(|runtime, context| {
            if !context.modules.contains_key(&module.addr())
                || context.library_modules.values().any(|&key| key == module.addr())
            {
                return Err(InvalidHandle);
            }
            context.streams.wait_all(&runtime.api)?;
            context.unload(&runtime.api, module.addr())
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuModuleGetFunction(
    out: *mut Opaque,
    module: Opaque,
    name: *const c_char,
) -> CuResult {
    trace::call("cuModuleGetFunction", || {
        status(with_context_handles(|runtime, context, next| {
            if out.is_null() || name.is_null() {
                return Err(InvalidValue);
            }
            let name = unsafe { CStr::from_ptr(name) };
            let function = module_function(&runtime.api, context, next, module.addr(), name)?;
            unsafe { out.write(std::ptr::without_provenance_mut(function)) };
            Ok(())
        }))
    })
}

fn module_function(
    api: &Api,
    context: &mut CudaContext,
    next: &mut usize,
    module_key: usize,
    name: &CStr,
) -> Result<usize, CuResult> {
    let module = context.modules.get_mut(&module_key).ok_or(InvalidHandle)?;
    if module.unloading {
        return Err(InvalidHandle);
    }
    let entry = module.entries.get_mut(name).ok_or(NotFound)?;
    if let Some((function, _)) = entry.function {
        return Ok(function);
    }
    let function = reserve_handle(next)?;
    let kernel = module
        .backend
        .create_kernel(api, name)
        .inspect_err(|error| context.backend.health.observe(error))
        .map_err(backend_error)?;
    entry.function = Some((function, kernel));
    context.functions.insert(function, (module_key, name.to_owned()));
    Ok(function)
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
    trace::call("cuLaunchKernel", || {
        status(state(|state| {
            let mut function = function.addr();
            if state.libraries.contains_kernel(function) {
                function = library::kernel_function(state, function)?;
            }
            let runtime = &state.runtime;
            let key = current().ok_or(InvalidContext)?;
            let context = state.contexts.get_mut(&key).ok_or(ContextIsDestroyed)?;
            context.check_live()?;
            let stream = context.streams.launch_key(stream)?;
            let (module_key, name) = context
                .functions
                .get(&function)
                .ok_or(InvalidHandle)?;
            let module = context.modules.get(module_key).ok_or(InvalidHandle)?;
            let entry = module.entries.get(name).ok_or(InvalidHandle)?;
            let (_, kernel) = entry.function.ok_or(InvalidHandle)?;
            // Dynamic shared is backed by a SPIR-V workgroup array sized at translate time.
            // NVIDIA allows a non-zero sharedMemBytes launch even when the kernel has no
            // `extern __shared__` / `.shared` (GGML mmf passes 128B unused); ignore it.
            if entry.shared_bytes != 0 && shared_bytes > entry.shared_bytes {
                return Err(InvalidValue);
            }
            let grid = [grid_x, grid_y, grid_z];
            let block = [block_x, block_y, block_z];
            validate_launch(
                &runtime.devices[context.device as usize].info,
                grid,
                block,
                entry.max_threads,
            )?;
            let args = unsafe {
                launch_args::read(&entry.params, kernel_params, extra, |address| {
                    // Pinned host allocations (cuMemAllocHost/cuMemHostAlloc) are Level Zero
                    // host USM and valid kernel arguments, like in CUDA's unified addressing.
                    contains_range(&context.allocations, address as usize, 1)
                        || contains_range(&context.host_allocations, address as usize, 1)
                })
            };
            let args = match args {
                Ok(args) => args,
                Err(error) => {
                    export_log(format_args!(
                        "cuLaunchKernel args rejected for {} ({} params)",
                        name.to_string_lossy(),
                        entry.params.len()
                    ));
                    return Err(error);
                }
            };
            // Optional launch dump for soft_max bring-up (CUDA_FOR_INTEL_DUMP_LAUNCH=1).
            if dump_launch_enabled()
                && name.to_string_lossy().contains("soft_max")
            {
                let mut summary = format!(
                    "soft_max launch args={} shared_bytes={shared_bytes} grid={grid:?} block={block:?}",
                    args.len()
                );
                if args.len() >= 4 {
                    for (i, label) in ["x", "mask", "sinks", "dst"].iter().enumerate() {
                        let ptr = unsafe { args[i].value.cast::<u64>().read_unaligned() };
                        summary.push_str(&format!(" {label}={ptr:#x}"));
                    }
                }
                if args.len() >= 4 + 32 {
                    let word = |offset: usize| -> u32 {
                        unsafe { args[4 + offset / 4].value.cast::<u32>().read_unaligned() }
                    };
                    let word64 = |offset: usize| -> u64 {
                        let lo = word(offset) as u64;
                        let hi = word(offset + 4) as u64;
                        lo | (hi << 32)
                    };
                    let scale = f32::from_bits(word(112));
                    summary.push_str(&format!(
                        " ncols={} ne01={} scale={scale} n_head_log2={}",
                        word64(16) as i64,
                        word64(48) as i64,
                        word(8)
                    ));
                }
                export_log(summary.clone());
                eprintln!("CUDA for Intel {summary}");
            }
            if let Some(capture) = context.streams.capture(stream)? {
                // The caller's argument storage may expire before graph launch.
                capture.push(
                    stream,
                    graphs::Node::Kernel(unsafe {
                        graphs::Kernel::capture(function, (grid, block), &args)
                    }),
                )?;
                return Ok(());
            }
            context.streams.launch(
                &runtime.api,
                &context.backend,
                kernel,
                stream,
                (grid, block),
                &args,
            )
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuDeviceGetCount(count: *mut c_int) -> CuResult {
    export_log("cuDeviceGetCount enter");
    let result = trace::call("cuDeviceGetCount", || {
        runtime(|rt| {
            if count.is_null() {
                return Err(InvalidValue);
            }
            let value = c_int::try_from(rt.devices.len()).map_err(|_| Unknown)?;
            unsafe { count.write(value) };
            Ok(Success)
        })
        .unwrap_or_else(|error| error)
    });
    export_log(format_args!("cuDeviceGetCount -> {}", result as i32));
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuDeviceGet(out: *mut c_int, ordinal: c_int) -> CuResult {
    trace::call("cuDeviceGet", || {
        device(ordinal, |_| {
            if out.is_null() {
                return InvalidValue;
            }
            unsafe { out.write(ordinal) };
            Success
        })
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuDeviceGetName(
    out: *mut c_char,
    len: c_int,
    ordinal: c_int,
) -> CuResult {
    trace::call("cuDeviceGetName", || {
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
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuDeviceTotalMem_v2(out: *mut usize, ordinal: c_int) -> CuResult {
    trace::call("cuDeviceTotalMem_v2", || {
        device(ordinal, |info| {
            if out.is_null() {
                return InvalidValue;
            }
            unsafe { out.write(info.total_memory as usize) };
            Success
        })
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CuResult {
    trace::call("cuMemGetInfo_v2", || {
        status(state(|state| {
            if free.is_null() || total.is_null() {
                return Err(InvalidValue);
            }
            let key = current().ok_or(InvalidContext)?;
            let device = state.contexts.get(&key).ok_or(ContextIsDestroyed)?.device;
            let info = state
                .runtime
                .devices
                .get(device as usize)
                .ok_or(InvalidDevice)?;
            let total_mem = info.info.total_memory as usize;
            // Level Zero core exposes no free-memory counter (that needs Sysman), so
            // report the capacity minus device bytes this driver has handed out.
            let used = used_device_bytes(&state.contexts, device);
            unsafe {
                free.write(total_mem.saturating_sub(used));
                total.write(total_mem);
            }
            Ok(())
        }))
    })
}

fn used_device_bytes(contexts: &HashMap<usize, CudaContext>, device: c_int) -> usize {
    contexts
        .values()
        .filter(|context| context.device == device)
        .flat_map(|context| context.allocations.values())
        .fold(0usize, |total, &size| total.saturating_add(size))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuDeviceGetUuid(out: *mut [u8; 16], ordinal: c_int) -> CuResult {
    trace::call("cuDeviceGetUuid", || {
        device(ordinal, |info| {
            if out.is_null() {
                return InvalidValue;
            }
            unsafe { out.write(info.uuid) };
            Success
        })
    })
}

/// Windows LUID for the device — 8-byte value plus a node mask.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuDeviceGetLuid(
    luid: *mut [u8; 8],
    device_node_mask: *mut c_uint,
    ordinal: c_int,
) -> CuResult {
    trace::call("cuDeviceGetLuid", || {
        device(ordinal, |info| {
            if luid.is_null() || device_node_mask.is_null() {
                return InvalidValue;
            }
            // CUDA Runtime matches devices to WDDM adapters by LUID. A synthetic
            // value makes cudart report cudaErrorInsufficientDriver (35).
            #[cfg(windows)]
            let bytes =
                dxgi_luid::adapter_identity(info.vendor_id, info.device_id).map(|(luid, _)| luid);
            #[cfg(not(windows))]
            let bytes: Option<[u8; 8]> = None;
            let Some(bytes) = bytes else {
                return NotSupported;
            };
            if trace::enabled() {
                trace::record(format_args!(
                    "LUID vendor={:#x} device={:#x} bytes={:02x?}",
                    info.vendor_id, info.device_id, bytes
                ));
            }
            unsafe {
                luid.write(bytes);
                device_node_mask.write(1);
            }
            Success
        })
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuCtxGetApiVersion(handle: Opaque, version: *mut c_uint) -> CuResult {
    trace::call("cuCtxGetApiVersion", || {
        status(state(|state| {
            if version.is_null() {
                return Err(InvalidValue);
            }
            let key = handle.addr();
            if key != 0 {
                state
                    .contexts
                    .get(&key)
                    .ok_or(InvalidContext)?
                    .check_live()?;
            } else {
                // Null context → current context, matching NVIDIA.
                let current = current().ok_or(InvalidContext)?;
                state
                    .contexts
                    .get(&current)
                    .ok_or(InvalidContext)?
                    .check_live()?;
            }
            // Report the Driver ABI floor used for contexts created here.
            unsafe { version.write(3020) };
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuCtxGetFlags(flags: *mut c_uint) -> CuResult {
    trace::call("cuCtxGetFlags", || {
        status(with_context(|_, _| {
            if flags.is_null() {
                return Err(InvalidValue);
            }
            unsafe { flags.write(0) };
            Ok(())
        }))
    })
}

/// PTX feature level advertised to CUDA Runtime / applications.
///
/// This is a compatibility contract, not a claim of NVIDIA silicon: kernels are
/// accepted and lowered at this ISA surface (Turing-class PTX used by GGML).
const COMPUTE_CAPABILITY_MAJOR: u32 = 7;
const COMPUTE_CAPABILITY_MINOR: u32 = 5;

fn multiprocessor_count(info: &DeviceInfo) -> u32 {
    // Level Zero has no CUDA SM count; derive a stable positive stand-in so
    // cudart / GGML occupancy math does not see zero devices worth of SMs.
    let subgroups = info.subgroup_sizes.iter().copied().max().unwrap_or(32).max(1);
    (info.max_workgroup_size / subgroups).clamp(1, 256)
}

fn attribute(info: &DeviceInfo, attr: c_int) -> Result<c_int, CuResult> {
    // Enumerants match CUDA's `CUdevice_attribute` (cuda-bindings DeviceAttribute).
    let value = match attr {
        1 => info.max_workgroup_size, // MAX_THREADS_PER_BLOCK
        2..=4 => info.max_workgroup_dimensions[(attr - 2) as usize], // MAX_BLOCK_DIM_*
        5..=7 => info.max_grid_dimensions[(attr - 5) as usize], // MAX_GRID_DIM_*
        8 => info.max_shared_local_memory, // MAX_SHARED_MEMORY_PER_BLOCK
        9 => 64 * 1024,                    // TOTAL_CONSTANT_MEMORY
        10 => 32,                          // WARP_SIZE
        11 => 1 << 20,                     // MAX_PITCH
        12 => 65536,                       // MAX_REGISTERS_PER_BLOCK
        13 => 1_000_000,                   // CLOCK_RATE (kHz)
        14 => 512,                         // TEXTURE_ALIGNMENT
        15 => 1,                           // GPU_OVERLAP
        16 => multiprocessor_count(info),  // MULTIPROCESSOR_COUNT
        17 => 0,                           // KERNEL_EXEC_TIMEOUT
        18 => info.device_flags & 1, // INTEGRATED
        19 => 1,                     // CAN_MAP_HOST_MEMORY
        20 => 0,                     // COMPUTE_MODE_DEFAULT
        // Texture/surface limits: cudart device bring-up rejects an all-zero
        // texture capability surface. Report Turing-class portable limits.
        21 => 32768,                 // MAXIMUM_TEXTURE1D_WIDTH
        22 | 23 => 32768,            // MAXIMUM_TEXTURE2D_WIDTH/HEIGHT
        24..=26 => 16384,            // MAXIMUM_TEXTURE3D_*
        27 | 28 => 32768,            // MAXIMUM_TEXTURE2D_LAYERED_WIDTH/HEIGHT
        29 => 2048,                  // MAXIMUM_TEXTURE2D_LAYERED_LAYERS
        30 => 512,                   // SURFACE_ALIGNMENT
        31 => 0,                     // CONCURRENT_KERNELS
        32 => 0,                     // ECC_ENABLED
        33 | 34 | 50 => {
            let [domain, bus, device, _] = info.pci_address.ok_or(NotSupported)?;
            match attr {
                33 => bus,    // PCI_BUS_ID
                34 => device, // PCI_DEVICE_ID
                _ => domain,  // PCI_DOMAIN_ID
            }
        }
        35 => 0,                            // TCC_DRIVER
        36 => 1_000_000,                    // MEMORY_CLOCK_RATE
        37 => 256,                          // GLOBAL_MEMORY_BUS_WIDTH
        38 => info.max_shared_local_memory, // L2_CACHE_SIZE
        39 => info.max_workgroup_size,      // MAX_THREADS_PER_MULTIPROCESSOR
        40 => 1,                            // ASYNC_ENGINE_COUNT
        41 => 0,                            // UNIFIED_ADDRESSING
        42 => 32768,                        // MAXIMUM_TEXTURE1D_LAYERED_WIDTH
        43 => 2048,                         // MAXIMUM_TEXTURE1D_LAYERED_LAYERS
        44 => 1,                            // CAN_TEX2D_GATHER
        45 | 46 => 16384,                   // MAXIMUM_TEXTURE2D_GATHER_*
        47..=49 => 2048,                    // MAXIMUM_TEXTURE3D_*_ALTERNATE
        51 => 32,                           // TEXTURE_PITCH_ALIGNMENT
        52 => 32768,                        // MAXIMUM_TEXTURECUBEMAP_WIDTH
        53 => 32768,                        // MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH
        54 => 2048,                         // MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS
        55 => 32768,                        // MAXIMUM_SURFACE1D_WIDTH
        56 | 57 => 32768,                   // MAXIMUM_SURFACE2D_*
        58..=60 => 16384,                   // MAXIMUM_SURFACE3D_*
        61 => 32768,                        // MAXIMUM_SURFACE1D_LAYERED_WIDTH
        62 => 2048,                         // MAXIMUM_SURFACE1D_LAYERED_LAYERS
        63 | 64 => 32768,                   // MAXIMUM_SURFACE2D_LAYERED_WIDTH/HEIGHT
        65 => 2048,                         // MAXIMUM_SURFACE2D_LAYERED_LAYERS
        66 | 67 => 32768,                   // MAXIMUM_SURFACECUBEMAP_*
        68 => 2048,                         // MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS
        69 => 2 * 1024 * 1024 * 1024,       // MAXIMUM_TEXTURE1D_LINEAR_WIDTH
        70 | 71 => 32768,                   // MAXIMUM_TEXTURE2D_LINEAR_WIDTH/HEIGHT
        72 => 1 << 20,                      // MAXIMUM_TEXTURE2D_LINEAR_PITCH
        73 | 74 => 32768,                   // MAXIMUM_TEXTURE2D_MIPMAPPED_*
        75 => COMPUTE_CAPABILITY_MAJOR,     // COMPUTE_CAPABILITY_MAJOR
        76 => COMPUTE_CAPABILITY_MINOR,     // COMPUTE_CAPABILITY_MINOR
        77 => 32768,                        // MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH
        78 => 0,                            // STREAM_PRIORITIES_SUPPORTED
        79 => 1,                            // GLOBAL_L1_CACHE_SUPPORTED
        80 => 1,                            // LOCAL_L1_CACHE_SUPPORTED
        81 => info.max_shared_local_memory, // MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
        82 => 65536,                        // MAX_REGISTERS_PER_MULTIPROCESSOR
        83 => 0,                            // MANAGED_MEMORY
        84 => 0,                            // MULTI_GPU_BOARD
        85 => 0,                            // MULTI_GPU_BOARD_GROUP_ID
        86 => 0,                            // HOST_NATIVE_ATOMIC_SUPPORTED
        87 => 32,                           // SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO
        88 => 0,                            // PAGEABLE_MEMORY_ACCESS
        89 => 0,                            // CONCURRENT_MANAGED_ACCESS
        90 => 1,                            // COMPUTE_PREEMPTION_SUPPORTED
        91 => 0,                            // CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM
        92..=94 => 0,                       // deprecated stream memops
        95 => 0,                            // COOPERATIVE_LAUNCH
        96 => 0,                            // COOPERATIVE_MULTI_DEVICE_LAUNCH
        97 => info.max_shared_local_memory, // MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
        98 => 0,                            // CAN_FLUSH_REMOTE_WRITES
        99 => 1,                            // HOST_REGISTER_SUPPORTED
        100 | 101 => 0,                     // pageable / direct managed access
        102 => 0,                           // VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
        103..=105 => 0,                     // shareable handle types
        106 => 16,                          // MAX_BLOCKS_PER_MULTIPROCESSOR
        107 => 0,                           // GENERIC_COMPRESSION_SUPPORTED
        108 => info.max_shared_local_memory, // MAX_PERSISTING_L2_CACHE_SIZE
        109 => 128 * 1024 * 1024,           // MAX_ACCESS_POLICY_WINDOW_SIZE
        110 => 0,                           // GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED
        111 => 0,                           // RESERVED_SHARED_MEMORY_PER_BLOCK
        112..=114 => 0,                     // sparse / readonly host register / timeline
        115 => 0,                           // MEMORY_POOLS_SUPPORTED
        116..=119 => 0,                     // GPUDirect RDMA / mempool handles
        120..=134 => 0,                     // cluster / dma-buf / IPC / NUMA / MPS
        139 => {
            // PCI device ID in the upper half, vendor ID in the lower half.
            let device = u16::try_from(info.device_id).map_err(|_| NotSupported)?;
            let vendor = u16::try_from(info.vendor_id).map_err(|_| NotSupported)?;
            return Ok(((u32::from(device) << 16) | u32::from(vendor)) as c_int);
        }
        #[cfg(windows)]
        140 => {
            let (_, subsystem) =
                dxgi_luid::adapter_identity(info.vendor_id, info.device_id).ok_or(NotSupported)?;
            return subsystem.map(|id| id as c_int).ok_or(NotSupported);
        }
        // CUDA 12/13 enumerants with no Intel equivalent: decompression,
        // Vulkan CIG, host NUMA VMM, host pools, host DMA-BUF, partial atomics.
        135..=138 | 141..=145 => 0,
        _ => {
            if trace::enabled() {
                trace::record(format_args!(
                    "unsupported device attribute {attr} on {}",
                    info.name
                ));
            }
            return Err(NotSupported);
        }
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
    trace::call("cuDeviceGetAttribute", || {
        if trace::enabled() {
            trace::record(format_args!(
                "device attribute ordinal={ordinal} attribute={attr}"
            ));
        }
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
        let _serial = TEST_RUNTIME.lock().unwrap();
        let mut out = -1;
        assert_eq!(unsafe { cuDeviceGetCount(&mut out) }, NotInitialized);
        assert_eq!(out, -1);
    }

    #[test]
    fn host_pinned_kernel_pointer_args_are_valid() {
        let device = BTreeMap::from([(0x1000usize, 0x100usize)]);
        let host = BTreeMap::from([(0x9000usize, 0x40usize)]);
        let valid = |address: usize| {
            contains_range(&device, address, 1) || contains_range(&host, address, 1)
        };
        assert!(valid(0x1010) && valid(0x9020));
        assert!(!valid(0x9040) && !valid(0x5000));
    }

    #[test]
    fn device_reset_status_counts_as_device_lost() {
        assert!(crate::level_zero::Error::Driver("zeCommandQueueSynchronize", 0x70000006).device_lost());
        assert!(!crate::level_zero::Error::Driver("zeCommandQueueSynchronize", 0x70000002).device_lost());
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
