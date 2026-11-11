//! Minimal bindings for the Level Zero core API used by this crate.
use libloading::Library;
use std::ffi::{CStr, c_char, c_void};
use std::fmt;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[cfg(test)]
pub(crate) mod fake;
#[cfg(test)]
pub(crate) mod fault;
mod native_cache;
mod regular;
pub(crate) use regular::{Queue, QueueFence};
mod streams;
pub(crate) use streams::WaitFn;
pub(crate) use streams::KernelArg;
pub(crate) use streams::{CommandList, Fence, Fences};
// Kernel timestamp events: hardware tests, and the opt-in CFI_GPU_PROFILE runtime profiler.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) mod profiling;

type Raw = *mut c_void;
type Status = u32;

#[derive(Clone, Copy)]
pub(crate) struct Handle(Raw);

// The Level Zero handles are opaque tokens. The CUDA frontend serializes their use.
unsafe impl Send for Handle {}

#[derive(Debug)]
pub enum Error {
    Loader(libloading::Error),
    Driver(&'static str, u32),
    Build(u32, String),
    InvalidProperties,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Loader(error) => write!(f, "Level Zero loader: {error}"),
            Self::Driver(call, status) => {
                write!(f, "{call} failed with Level Zero status {status:#010x}")
            }
            Self::Build(status, log) => {
                write!(
                    f,
                    "zeModuleCreate failed with Level Zero status {status:#010x}"
                )?;
                if !log.is_empty() {
                    write!(f, "\n{log}")?;
                }
                Ok(())
            }
            Self::InvalidProperties => f.write_str("Level Zero returned invalid device properties"),
        }
    }
}

impl std::error::Error for Error {}

impl Error {
    pub(crate) fn device_lost(&self) -> bool {
        matches!(
            self,
            Self::Driver(_, 0x70000001 | 0x70000006) | Self::Build(0x70000001 | 0x70000006, _)
        )
    }
}

fn check(call: &'static str, status: Status) -> Result<(), Error> {
    match status {
        0 => Ok(()),
        _ => Err(Error::Driver(call, status)),
    }
}

// Fail before entering the driver so tests can also use real hardware safely.
fn invoke(_name: &'static str, native: impl FnOnce() -> Status) -> Status {
    #[cfg(test)]
    if let Some(status) = fault::status(_name) {
        return status;
    }
    if crate::cuda::profile_on() {
        let started = std::time::Instant::now();
        let status = native();
        crate::cuda::profile_note(_name, started.elapsed().as_nanos() as u64);
        return status;
    }
    native()
}

fn call(name: &'static str, native: impl FnOnce() -> Status) -> Result<(), Error> {
    check(name, invoke(name, native))
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Header {
    stype: u32,
    next: *mut c_void,
}

impl Header {
    const fn new(stype: u32) -> Self {
        Self {
            stype,
            next: std::ptr::null_mut(),
        }
    }
}

#[repr(C)]
struct DriverProperties {
    header: Header,
    version: u32,
    uuid: [u8; 16],
}

#[repr(C)]
struct FloatAtomics {
    header: Header,
    flags: [u32; 3],
}

#[repr(C)]
struct Properties {
    header: Header,
    kind: u32,
    vendor: u32,
    device: u32,
    flags: u32,
    subdevice_id: u32,
    core_clock_rate: u32,
    max_alloc_size: u64,
    max_hardware_contexts: u32,
    max_queue_priority: u32,
    threads_per_eu: u32,
    physical_eu_simd_width: u32,
    eus_per_subslice: u32,
    subslices_per_slice: u32,
    slices: u32,
    timer_resolution: u64,
    timestamp_valid_bits: u32,
    kernel_timestamp_valid_bits: u32,
    uuid: [u8; 16],
    name: [c_char; 256],
}

#[repr(C)]
struct ComputeProperties {
    header: Header,
    max_group_size: u32,
    group_size: [u32; 3],
    group_count: [u32; 3],
    shared_local_memory: u32,
    subgroup_count: u32,
    subgroup_sizes: [u32; 8],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MemoryProperties {
    header: Header,
    flags: u32,
    max_clock_rate: u32,
    max_bus_width: u32,
    total_size: u64,
    name: [c_char; 256],
}

#[repr(C)]
struct ModuleProperties {
    header: Header,
    spirv_version: u32,
    flags: u32,
    fp16_flags: u32,
    fp32_flags: u32,
    fp64_flags: u32,
    max_arguments_size: u32,
    printf_buffer_size: u32,
    native_kernel_uuid: [u8; 16],
}

#[repr(C)]
struct MemoryAccessProperties {
    header: Header,
    host_alloc: u32,
    device_alloc: u32,
    shared_single_device_alloc: u32,
    shared_cross_device_alloc: u32,
    shared_system_alloc: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ExtensionProperties {
    name: [c_char; 256],
    version: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct QueueGroupProperties {
    header: Header,
    flags: u32,
    max_fill_pattern_size: usize,
    queues: u32,
}

#[repr(C)]
struct ContextDesc {
    header: Header,
    flags: u32,
}

#[repr(C)]
struct QueueDesc {
    header: Header,
    ordinal: u32,
    index: u32,
    flags: u32,
    mode: u32,
    priority: u32,
}

#[repr(C)]
struct AllocDesc {
    header: Header,
    flags: u32,
    ordinal: u32,
}

#[repr(C)]
struct HostAllocDesc {
    header: Header,
    flags: u32,
}

#[repr(C)]
struct ModuleDesc {
    header: Header,
    format: u32,
    input_size: usize,
    input: *const u8,
    build_flags: *const c_char,
    constants: *const c_void,
}

#[repr(C)]
struct KernelDesc {
    header: Header,
    flags: u32,
    name: *const c_char,
}

#[repr(C)]
struct GroupCount {
    x: u32,
    y: u32,
    z: u32,
}

#[repr(C)]
struct EventPoolDesc {
    header: Header,
    flags: u32,
    count: u32,
}

#[repr(C)]
struct EventDesc {
    header: Header,
    index: u32,
    signal: u32,
    wait: u32,
}

#[repr(C)]
struct PciProperties {
    header: Header,
    address: [u32; 4],
    generation: i32,
    width: i32,
    bandwidth: i64,
}

type GetPci = unsafe extern "system" fn(Raw, *mut PciProperties) -> Status;
/// `zeCommandListImmediateAppendCommandListsExp(immediate, count, lists, signal, wait count, waits)`
pub(crate) type AppendLists = unsafe extern "system" fn(Raw, u32, *mut Raw, Raw, u32, *mut Raw) -> Status;
/// `zeDeviceCanAccessPeer(device, peer, *mut ze_bool_t)`
type CanAccessPeer = unsafe extern "system" fn(Raw, Raw, *mut u8) -> Status;

macro_rules! loader_api {
    ($( $name:ident($($arg:ty),*) ),+ $(,)?) => {
        #[allow(non_snake_case)]
        pub(crate) struct Api {
            $( $name: unsafe extern "system" fn($($arg),*) -> Status, )+
            get_pci: Option<GetPci>,
            can_access_peer: Option<CanAccessPeer>,
            /// Experimental immediate-list extension; `None` when the loader lacks it.
            pub(crate) append_lists: Option<AppendLists>,
            #[cfg(not(test))]
            _library: Library,
            #[cfg(test)]
            _library: Option<Library>,
        }

        impl Api {
            fn open() -> Result<Self, Error> {
                #[cfg(windows)]
                let library: Library = {
                    use libloading::os::windows::{Library, LOAD_LIBRARY_SEARCH_SYSTEM32};
                    // Restrict the loader lookup to System32, even beside a CUDA application.
                    unsafe { Library::load_with_flags("ze_loader.dll", LOAD_LIBRARY_SEARCH_SYSTEM32) }
                        .map_err(Error::Loader)?.into()
                };
                #[cfg(target_os = "linux")]
                let library = unsafe { Library::new("libze_loader.so.1") }.map_err(Error::Loader)?;
                // Host-only unit-test builds (for example macOS) run on the fake backend.
                #[cfg(not(any(windows, target_os = "linux")))]
                let library: Library = unsafe { Library::new("libze_loader.so.1") }.map_err(Error::Loader)?;

                Ok(Self {
                    $( $name: unsafe { library.get(concat!(stringify!($name), "\0").as_bytes()) }
                        .map(|symbol| *symbol).map_err(Error::Loader)?, )+
                    get_pci: unsafe { library.get::<GetPci>(b"zeDevicePciGetPropertiesExt\0") }
                        .ok().map(|symbol| *symbol),
                    can_access_peer: unsafe { library.get::<CanAccessPeer>(b"zeDeviceCanAccessPeer\0") }
                        .ok().map(|symbol| *symbol),
                    append_lists: unsafe {
                        library.get::<AppendLists>(b"zeCommandListImmediateAppendCommandListsExp\0")
                    }
                    .ok().map(|symbol| *symbol),
                    #[cfg(not(test))]
                    _library: library,
                    #[cfg(test)]
                    _library: Some(library),
                })
            }

            #[cfg(test)]
            fn stub() -> Self {
                Self {
                    $( $name: {
                        unsafe extern "system" fn unsupported($(_: $arg),*) -> Status {
                            fake::unexpected(stringify!($name))
                        }
                        unsupported
                    }, )+
                    get_pci: None,
                    can_access_peer: None,
                    append_lists: None,
                    _library: None,
                }
            }

            /// True for the in-process fake backend used by unit tests.
            #[cfg(test)]
            pub(crate) fn is_fake(&self) -> bool {
                self._library.is_none()
            }

            /// Declare that the kernel may dereference device/host/shared USM pointers it was not
            /// given as arguments (pointer tables, linked structures), as ggml's batched GEMM does.
            /// Without this the driver need not keep those allocations resident for the dispatch.
            fn set_indirect_access(&self, kernel: Raw) {
                if std::env::var_os("CUDA_FOR_INTEL_NO_INDIRECT").is_some() {
                    return;
                }
                #[cfg(not(test))]
                let library = Some(&self._library);
                #[cfg(test)]
                let library = self._library.as_ref();
                let Some(library) = library else { return };
                type SetIndirect = unsafe extern "system" fn(Raw, u32) -> Status;
                if let Ok(set) = unsafe { library.get::<SetIndirect>(b"zeKernelSetIndirectAccess\0") } {
                    // ZE_KERNEL_INDIRECT_ACCESS_FLAG_HOST | DEVICE | SHARED
                    let _ = unsafe { set(kernel, 0b111) };
                }
            }

            /// Print private/spill/SLM sizes of a kernel when `CUDA_FOR_INTEL_LOG` is set.
            /// Uses a direct loader lookup so the fake backend needs no extra entry point.
            fn log_kernel_properties(&self, kernel: Raw, name: &CStr) {
                if !crate::cuda::export_log_enabled() {
                    return;
                }
                #[cfg(not(test))]
                let library = Some(&self._library);
                #[cfg(test)]
                let library = self._library.as_ref();
                let Some(library) = library else { return };
                type GetProps = unsafe extern "system" fn(Raw, *mut u8) -> Status;
                let Ok(get) = (unsafe { library.get::<GetProps>(b"zeKernelGetProperties\0") }) else {
                    return;
                };
                let mut buffer = [0u8; 256];
                buffer[..4].copy_from_slice(&0x15u32.to_le_bytes()); // ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES
                if unsafe { get(kernel, buffer.as_mut_ptr()) } != 0 {
                    return;
                }
                let word = |i: usize| u32::from_le_bytes(buffer[16 + 4 * i..20 + 4 * i].try_into().unwrap());
                eprintln!(
                    "CUDA for Intel: kernel {} args={} subgroup={}..{} slm={} private={} spill={}",
                    name.to_string_lossy(),
                    word(0),
                    word(5),
                    word(6),
                    word(8),
                    word(9),
                    word(10)
                );
            }

            #[cfg(all(test, windows))]
            fn library(&self) -> &Library {
                self._library.as_ref().expect("hardware profiling requires the native loader")
            }

            /// The native loader (`CFI_GPU_PROFILE` resolves optional timestamp entry points).
            #[cfg(all(not(test), windows))]
            fn library(&self) -> &Library {
                &self._library
            }
        }
    };
}

loader_api! {
    zeInit(u32),
    zeDriverGet(*mut u32, *mut Raw),
    zeDriverGetApiVersion(Raw, *mut u32),
    zeDriverGetProperties(Raw, *mut DriverProperties),
    zeDeviceGet(Raw, *mut u32, *mut Raw),
    zeDeviceGetProperties(Raw, *mut Properties),
    zeDeviceGetComputeProperties(Raw, *mut ComputeProperties),
    zeDeviceGetMemoryProperties(Raw, *mut u32, *mut MemoryProperties),
    zeDeviceGetMemoryAccessProperties(Raw, *mut MemoryAccessProperties),
    zeDeviceGetModuleProperties(Raw, *mut ModuleProperties),
    zeDriverGetExtensionProperties(Raw, *mut u32, *mut ExtensionProperties),
    zeDeviceGetCommandQueueGroupProperties(Raw, *mut u32, *mut QueueGroupProperties),
    zeContextCreate(Raw, *const ContextDesc, *mut Raw),
    zeContextDestroy(Raw),
    zeCommandListCreateImmediate(Raw, Raw, *const QueueDesc, *mut Raw),
    zeCommandListCreate(Raw, Raw, *const regular::ListDesc, *mut Raw),
    zeCommandListClose(Raw),
    zeCommandListReset(Raw),
    zeCommandQueueCreate(Raw, Raw, *const QueueDesc, *mut Raw),
    zeCommandQueueDestroy(Raw),
    zeCommandQueueSynchronize(Raw, u64),
    zeCommandQueueExecuteCommandLists(Raw, u32, *mut Raw, Raw),
    zeFenceCreate(Raw, *const regular::FenceDesc, *mut Raw),
    zeFenceDestroy(Raw),
    zeFenceReset(Raw),
    zeFenceQueryStatus(Raw),
    zeFenceHostSynchronize(Raw, u64),
    zeCommandListDestroy(Raw),
    zeCommandListHostSynchronize(Raw, u64),
    zeCommandListAppendSignalEvent(Raw, Raw),
    zeCommandListAppendBarrier(Raw, Raw, u32, *mut Raw),
    zeCommandListAppendMemoryRangesBarrier(
        Raw,
        u32,
        *const usize,
        *const *const c_void,
        Raw,
        u32,
        *mut Raw
    ),
    zeEventPoolCreate(Raw, *const EventPoolDesc, u32, *mut Raw, *mut Raw),
    zeEventPoolDestroy(Raw),
    zeEventCreate(Raw, *const EventDesc, *mut Raw),
    zeEventDestroy(Raw),
    zeEventHostReset(Raw),
    zeEventQueryStatus(Raw),
    zeEventHostSynchronize(Raw, u64),
    zeMemAllocDevice(Raw, *const AllocDesc, usize, usize, Raw, *mut Raw),
    zeMemAllocHost(Raw, *const HostAllocDesc, usize, usize, *mut Raw),
    zeMemFree(Raw, Raw),
    zeCommandListAppendMemoryCopy(Raw, Raw, *const c_void, usize, Raw, u32, *mut Raw),
    zeCommandListAppendMemoryFill(Raw, Raw, *const c_void, usize, usize, Raw, u32, *mut Raw),
    zeModuleCreate(Raw, Raw, *const ModuleDesc, *mut Raw, *mut Raw),
    zeModuleDestroy(Raw),
    zeModuleGetNativeBinary(Raw, *mut usize, *mut u8),
    zeModuleGetGlobalPointer(Raw, *const c_char, *mut usize, *mut Raw),
    zeCommandListAppendWriteGlobalTimestamp(Raw, *mut u64, Raw, u32, *mut Raw),
    zeModuleBuildLogGetString(Raw, *mut usize, *mut c_char),
    zeModuleBuildLogDestroy(Raw),
    zeKernelCreate(Raw, *const KernelDesc, *mut Raw),
    zeKernelDestroy(Raw),
    zeKernelSetGroupSize(Raw, u32, u32, u32),
    zeKernelSetArgumentValue(Raw, u32, usize, *const c_void),
    zeCommandListAppendLaunchKernel(Raw, Raw, *const GroupCount, Raw, u32, *mut Raw),
}

/// Capabilities reported by the installed Level Zero driver.
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_flags: u32,
    /// PCI domain, bus, device and function, when the driver exposes them.
    pub pci_address: Option<[u32; 4]>,
    pub driver_version: u32,
    pub api_version: u32,
    pub uuid: [u8; 16],
    pub total_memory: u64,
    pub max_allocation: u64,
    pub max_workgroup_size: u32,
    pub max_workgroup_dimensions: [u32; 3],
    pub max_grid_dimensions: [u32; 3],
    pub max_shared_local_memory: u32,
    pub subgroup_sizes: Vec<u32>,
    pub spirv_version: u32,
    pub fp32_denorms: bool,
    pub fp32_round_to_nearest: bool,
    pub fp32_inf_nan: bool,
    pub fp32_fma: bool,
    pub device_memory_atomics: bool,
    pub module_flags: u32,
    /// Floating-point capability flags in FP16, FP32, FP64 order.
    pub floating_point_flags: [u32; 3],
    /// Atomic-operation flags in FP16, FP32, FP64 order, if queried successfully.
    pub float_atomic_flags: Option<[u32; 3]>,
    /// Host, device, single-device shared, cross-device shared, system shared.
    pub memory_access_flags: [u32; 5],
    pub memory_heaps: Vec<MemoryHeap>,
    pub max_kernel_arguments_size: u32,
    pub driver_extensions: Vec<(String, u32)>,
    pub queue_groups: Vec<QueueGroup>,
    /// Timestamp counter frequency, when the driver supports the 1.2 properties.
    pub timestamp_frequency_hz: Option<u64>,
    pub timestamp_valid_bits: u32,
    pub kernel_timestamp_valid_bits: u32,
}

impl DeviceInfo {
    pub fn compute_queue(&self) -> Option<QueueGroup> {
        select_compute_queue(&self.queue_groups)
    }

    pub fn supports_fp32(&self) -> bool {
        self.fp32_denorms && self.fp32_round_to_nearest && self.fp32_inf_nan
    }
}

#[derive(Clone, Debug)]
pub struct MemoryHeap {
    pub name: String,
    pub size: u64,
    pub clock_mhz: u32,
    pub bus_width: u32,
    pub flags: u32,
}

/// An ordinal and its capabilities as reported by Level Zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueueGroup {
    pub ordinal: u32,
    pub flags: u32,
    pub queues: u32,
}

impl QueueGroup {
    pub fn compute(self) -> bool {
        self.flags & 1 != 0
    }

    pub fn copy(self) -> bool {
        self.flags & 2 != 0
    }

    pub fn copy_only(self) -> bool {
        self.copy() && !self.compute()
    }
}

fn select_compute_queue(groups: &[QueueGroup]) -> Option<QueueGroup> {
    groups
        .iter()
        .copied()
        .find(|group| group.queues != 0 && group.compute())
}

pub(crate) struct Device {
    pub(crate) info: DeviceInfo,
    driver: Handle,
    handle: Handle,
    /// Position of the Level Zero driver and of the device inside it, before any
    /// CUDA-style reordering or filtering.
    pub(crate) driver_index: u32,
    pub(crate) device_index: u32,
    /// Per-device translator probe results (see `cuda::devices::probe`).
    pub(crate) probe: std::sync::OnceLock<crate::cuda::devices::probe::DeviceProbe>,
}

impl Device {
    /// Raw driver handle address; equal values mean the same Level Zero driver.
    pub(crate) fn driver_id(&self) -> usize {
        self.driver.0.addr()
    }

    /// `zeDeviceCanAccessPeer`, or `None` when the loader does not export it or the
    /// devices belong to different drivers (the call is only defined within one).
    pub(crate) fn can_access_peer(&self, api: &Api, peer: &Device) -> Option<bool> {
        if self.driver_id() != peer.driver_id() {
            return None;
        }
        let query = api.can_access_peer?;
        let mut value = 0u8;
        (unsafe { query(self.handle.0, peer.handle.0, &mut value) } == 0).then_some(value != 0)
    }
}

pub(crate) struct Runtime {
    pub(crate) api: Api,
    pub(crate) devices: Vec<Device>,
}

#[derive(Default)]
pub(crate) struct DeviceHealth(AtomicBool);

impl DeviceHealth {
    pub(crate) fn observe(&self, error: &Error) {
        if error.device_lost() {
            self.0.store(true, Ordering::Relaxed);
        }
    }

    pub(crate) fn lost(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

pub(crate) struct Context {
    pub(crate) health: Arc<DeviceHealth>,
    handle: Handle,
    device: Handle,
    commands: Option<Handle>,
    build_logs: Vec<Handle>,
    ordinal: u32,
    /// Device and driver identity mixed into native-binary cache keys.
    cache_tag: String,
}

/// Device and driver identity mixed into native-binary cache keys. Per device: two
/// different GPUs never share a binary, identical models can.
pub(crate) fn cache_tag(info: &DeviceInfo) -> String {
    format!(
        "{:x}:{:x}:{}:{:x}:{:x}",
        info.vendor_id, info.device_id, info.name, info.driver_version, info.api_version
    )
}

pub(crate) struct Module {
    handle: Handle,
    kernels: Vec<Handle>,
}

impl Module {
    pub(crate) fn create_kernel(&mut self, api: &Api, name: &CStr) -> Result<Handle, Error> {
        let desc = KernelDesc {
            header: Header::new(0x1d),
            flags: 0,
            name: name.as_ptr(),
        };
        let mut kernel = std::ptr::null_mut();
        call("zeKernelCreate", || unsafe {
            (api.zeKernelCreate)(self.handle.0, &desc, &mut kernel)
        })?;
        api.set_indirect_access(kernel);
        api.log_kernel_properties(kernel, name);
        let kernel = Handle(kernel);
        self.kernels.push(kernel);
        Ok(kernel)
    }

    /// The driver's compiled binary for this module, when it can produce one.
    fn native_binary(&self, api: &Api) -> Option<Vec<u8>> {
        let mut size = 0usize;
        if unsafe { (api.zeModuleGetNativeBinary)(self.handle.0, &mut size, std::ptr::null_mut()) } != 0
            || size == 0
        {
            return None;
        }
        let mut data = vec![0u8; size];
        (unsafe { (api.zeModuleGetNativeBinary)(self.handle.0, &mut size, data.as_mut_ptr()) } == 0)
            .then(|| {
                data.truncate(size);
                data
            })
    }

    /// Device address and size of an exported module variable.
    pub(crate) fn global_pointer(&self, api: &Api, name: &CStr) -> Result<(usize, usize), Error> {
        let (mut size, mut pointer) = (0usize, std::ptr::null_mut());
        call("zeModuleGetGlobalPointer", || unsafe {
            (api.zeModuleGetGlobalPointer)(self.handle.0, name.as_ptr(), &mut size, &mut pointer)
        })?;
        Ok((pointer.expose_provenance(), size))
    }

    pub(crate) fn destroy(&mut self, api: &Api) -> Result<(), Error> {
        while let Some(kernel) = self.kernels.last() {
            call("zeKernelDestroy", || unsafe {
                (api.zeKernelDestroy)(kernel.0)
            })?;
            self.kernels.pop();
        }
        call("zeModuleDestroy", || unsafe {
            (api.zeModuleDestroy)(self.handle.0)
        })
    }
}

/// Intel compiler options for a SPIR-V module of `words` words. `CUDA_FOR_INTEL_BUILD_FLAGS`
/// applies to every module and `CUDA_FOR_INTEL_BIG_FLAGS` to modules above
/// `CUDA_FOR_INTEL_BIG_WORDS` (default 400000) words.
fn build_flags(words: usize) -> String {
    let get = |name: &str| std::env::var(name).unwrap_or_default();
    compose_build_flags(words, &get)
}

/// Flag every module is built with unless `CUDA_FOR_INTEL_FAST_DIV=1`: PTX `div.rn.f32` and
/// `sqrt.rn.f32` are IEEE correctly rounded on NVIDIA, but Intel's OpenCL compiler lowers
/// SPIR-V `OpFDiv` / `OpenCL.std sqrt` to 2.5 / 3 ulp versions without this option. The 1-ulp
/// quotient errors flip q8_1 activation bytes (`x / d` in quantize_q8_1) and decorrelate greedy
/// decoding from CUDA.
pub(crate) const CORRECT_DIV_FLAG: &str = "-cl-fp32-correctly-rounded-divide-sqrt";

fn compose_build_flags(words: usize, get: &dyn Fn(&str) -> String) -> String {
    let mut flags = get("CUDA_FOR_INTEL_BUILD_FLAGS");
    if get("CUDA_FOR_INTEL_FAST_DIV") != "1" && !flags.contains(CORRECT_DIV_FLAG) {
        if !flags.is_empty() {
            flags.push(' ');
        }
        flags.push_str(CORRECT_DIV_FLAG);
    }
    let threshold = get("CUDA_FOR_INTEL_BIG_WORDS").parse().unwrap_or(400_000);
    if words > threshold {
        let big = get("CUDA_FOR_INTEL_BIG_FLAGS");
        if !big.is_empty() {
            if !flags.is_empty() {
                flags.push(' ');
            }
            flags.push_str(&big);
        }
    }
    flags
}

fn build_log(api: &Api, handle: Raw) -> String {
    let mut size = 0;
    if unsafe { (api.zeModuleBuildLogGetString)(handle, &mut size, std::ptr::null_mut()) } != 0
        || !(1..=1024 * 1024).contains(&size)
    {
        return String::new();
    }
    let mut bytes = vec![0; size];
    if unsafe { (api.zeModuleBuildLogGetString)(handle, &mut size, bytes.as_mut_ptr().cast()) } != 0
    {
        return String::new();
    }
    CStr::from_bytes_until_nul(&bytes)
        .map(|log| log.to_string_lossy().into_owned())
        .unwrap_or_default()
}

impl Context {
    pub(crate) fn create(api: &Api, device: &Device) -> Result<Self, Error> {
        let ordinal = device
            .info
            .compute_queue()
            .ok_or(Error::Driver("select compute queue", 0x78000003))?
            .ordinal;
        let desc = ContextDesc {
            header: Header::new(0xd),
            flags: 0,
        };
        let mut handle = std::ptr::null_mut();
        call("zeContextCreate", || unsafe {
            (api.zeContextCreate)(device.driver.0, &desc, &mut handle)
        })?;
        Ok(Self {
            health: Arc::default(),
            handle: Handle(handle),
            device: device.handle,
            commands: None,
            build_logs: Vec::new(),
            ordinal,
            cache_tag: cache_tag(&device.info),
        })
    }

    pub(crate) fn initialize(&mut self, api: &Api) -> Result<(), Error> {
        let queue = QueueDesc {
            header: Header::new(0xe),
            ordinal: self.ordinal,
            index: 0,
            flags: 0,
            mode: 1, // ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS
            priority: 0,
        };
        let mut commands = std::ptr::null_mut();
        call("zeCommandListCreateImmediate", || unsafe {
            (api.zeCommandListCreateImmediate)(self.handle.0, self.device.0, &queue, &mut commands)
        })?;
        self.commands = Some(Handle(commands));
        Ok(())
    }

    fn commands(&self) -> Result<Raw, Error> {
        self.commands
            .map(|handle| handle.0)
            .ok_or(Error::Driver("use closed command list", 0x78000005))
    }

    pub(crate) fn destroy(&mut self, api: &Api) -> Result<(), Error> {
        while let Some(log) = self.build_logs.last() {
            call("zeModuleBuildLogDestroy", || unsafe {
                (api.zeModuleBuildLogDestroy)(log.0)
            })?;
            self.build_logs.pop();
        }
        if let Some(commands) = self.commands {
            call("zeCommandListDestroy", || unsafe {
                (api.zeCommandListDestroy)(commands.0)
            })?;
            self.commands = None;
        }
        call("zeContextDestroy", || unsafe {
            (api.zeContextDestroy)(self.handle.0)
        })
    }

    pub(crate) fn alloc(&self, api: &Api, size: usize) -> Result<usize, Error> {
        self.commands()?;
        let desc = AllocDesc {
            header: Header::new(0x15),
            flags: 0,
            ordinal: 0,
        };
        let mut out = std::ptr::null_mut();
        call("zeMemAllocDevice", || unsafe {
            (api.zeMemAllocDevice)(self.handle.0, &desc, size, 0, self.device.0, &mut out)
        })?;
        Ok(out.expose_provenance())
    }

    pub(crate) fn alloc_host(&self, api: &Api, size: usize) -> Result<usize, Error> {
        self.commands()?;
        // Prefer uncached host pages for CUDA-style pinned staging. Cached host
        // memory can retain a prior DtoH pattern in the CPU cache after a later
        // DMA overwrite when completion used HostSynchronize without a host-scope
        // event; Arc B390 reproduced that as an intermittent stale read.
        let desc = HostAllocDesc {
            header: Header::new(0x16),
            flags: 1 << 1, // ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED
        };
        let mut out = std::ptr::null_mut();
        call("zeMemAllocHost", || unsafe {
            (api.zeMemAllocHost)(self.handle.0, &desc, size, 4096, &mut out)
        })?;
        Ok(out.expose_provenance())
    }

    pub(crate) fn free(&self, api: &Api, address: usize) -> Result<(), Error> {
        call("zeMemFree", || unsafe {
            (api.zeMemFree)(
                self.handle.0,
                std::ptr::with_exposed_provenance_mut(address),
            )
        })
    }

    pub(crate) unsafe fn copy(
        &self,
        api: &Api,
        dst: Raw,
        src: *const c_void,
        size: usize,
    ) -> Result<(), Error> {
        let commands = self.commands()?;
        let signal = std::ptr::null_mut();
        #[cfg(all(test, windows))]
        let signal = profiling::operation_signal(signal);
        call("zeCommandListAppendMemoryCopy", || unsafe {
            (api.zeCommandListAppendMemoryCopy)(
                commands,
                dst,
                src,
                size,
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
            )
        })?;
        // Make the destination range coherent before the synchronous list
        // returns, including host visibility for DtoH into pinned memory.
        let range = dst.cast_const();
        call("zeCommandListAppendMemoryRangesBarrier", || unsafe {
            (api.zeCommandListAppendMemoryRangesBarrier)(
                commands,
                1,
                &size,
                &range,
                signal,
                0,
                std::ptr::null_mut(),
            )
        })
    }

    pub(crate) fn load_module(&mut self, api: &Api, words: &[u32]) -> Result<Module, Error> {
        self.commands()?;
        // Fake backends cannot produce native binaries.
        #[cfg(test)]
        let cache = if api.is_fake() { None } else { native_cache::directory() };
        #[cfg(not(test))]
        let cache = native_cache::directory();
        let flags = build_flags(words.len());
        let key = cache.map(|_| native_cache::key(words, &self.cache_tag, &flags));
        if let (Some(dir), Some(key)) = (cache, &key) {
            if let Some(binary) = native_cache::load(dir, key) {
                match self.create_module(api, 1, binary.as_ptr(), binary.len(), "") {
                    Ok(module) => return Ok(module),
                    Err(Error::Build(..)) | Err(Error::Driver(..)) => native_cache::discard(dir, key),
                    Err(other) => return Err(other),
                }
            }
        }
        let module = self.create_module(api, 0, words.as_ptr().cast(), std::mem::size_of_val(words), &flags)?;
        if let (Some(dir), Some(key)) = (cache, &key) {
            if let Some(binary) = module.native_binary(api) {
                native_cache::store(dir, key, &binary);
            }
        }
        Ok(module)
    }

    /// A thread-safe handle for compiling SPIR-V into the native-binary cache off the launch
    /// path. `None` on fake backends or when the native cache is disabled.
    pub(crate) fn native_compiler(&self, api: &Api) -> Option<NativeCompiler> {
        #[cfg(test)]
        if api.is_fake() {
            return None;
        }
        native_cache::directory()?;
        Some(NativeCompiler {
            create: api.zeModuleCreate,
            destroy: api.zeModuleDestroy,
            native_binary: api.zeModuleGetNativeBinary,
            log_destroy: api.zeModuleBuildLogDestroy,
            context: self.handle,
            device: self.device,
            tag: self.cache_tag.clone(),
        })
    }

    /// `format` 0 is SPIR-V and 1 is a native binary (`ZE_MODULE_FORMAT_*`).
    fn create_module(&mut self, api: &Api, format: u32, input: *const u8, size: usize, flags: &str) -> Result<Module, Error> {
        let flags = std::ffi::CString::new(flags).unwrap_or_default();
        let desc = ModuleDesc {
            header: Header::new(0x1b),
            format,
            input_size: size,
            input,
            build_flags: if flags.as_bytes().is_empty() { std::ptr::null() } else { flags.as_ptr() },
            constants: std::ptr::null(),
        };
        let mut module = std::ptr::null_mut();
        let mut log = std::ptr::null_mut();
        let status = invoke("zeModuleCreate", || unsafe {
            (api.zeModuleCreate)(self.handle.0, self.device.0, &desc, &mut module, &mut log)
        });
        let message = if !log.is_null() {
            let message = if status == 0 {
                String::new()
            } else {
                build_log(api, log)
            };
            if call("zeModuleBuildLogDestroy", || unsafe {
                (api.zeModuleBuildLogDestroy)(log)
            })
            .is_err()
            {
                self.build_logs.push(Handle(log));
            }
            message
        } else {
            String::new()
        };
        if status != 0 {
            return Err(Error::Build(status, message));
        }
        Ok(Module {
            handle: Handle(module),
            kernels: Vec::new(),
        })
    }
}

fn enumerate<T: Clone>(
    call: &'static str,
    initial: T,
    mut query: impl FnMut(*mut u32, *mut T) -> Status,
) -> Result<Vec<T>, Error> {
    let mut count = 0;
    check(call, query(&mut count, std::ptr::null_mut()))?;
    let mut values = vec![initial; count as usize];
    if count != 0 {
        check(call, query(&mut count, values.as_mut_ptr()))?;
        if count as usize > values.len() {
            return Err(Error::InvalidProperties);
        }
        values.truncate(count as usize);
    }
    Ok(values)
}

fn name(value: &[c_char]) -> Result<String, Error> {
    // C char is one byte on both supported platforms. Bound the NUL search to the field.
    let bytes = unsafe { std::slice::from_raw_parts(value.as_ptr().cast::<u8>(), value.len()) };
    let value = CStr::from_bytes_until_nul(bytes).map_err(|_| Error::InvalidProperties)?;
    Ok(value.to_string_lossy().into_owned())
}

impl Runtime {
    pub(crate) fn load() -> Result<Self, Error> {
        let api = Api::open()?;
        // ZE_INIT_FLAG_GPU_ONLY
        check("zeInit", unsafe { (api.zeInit)(1) })?;
        let drivers = enumerate("zeDriverGet", std::ptr::null_mut(), |count, out| unsafe {
            (api.zeDriverGet)(count, out)
        })?;
        let mut devices = Vec::new();
        let mut skipped = None;
        for (driver_index, driver) in drivers.into_iter().enumerate() {
            // One broken driver (for example a non-GPU one) must not hide the others.
            let handles = match enumerate("zeDeviceGet", std::ptr::null_mut(), |count, out| unsafe {
                (api.zeDeviceGet)(driver, count, out)
            }) {
                Ok(handles) => handles,
                Err(error) => {
                    skipped.get_or_insert(error);
                    continue;
                }
            };
            let mut metadata = None;
            let mut device_index = 0u32;
            for handle in handles {
                // These C output structures contain only integers and pointers.
                let mut properties: Properties = unsafe { std::mem::zeroed() };
                properties.header = Header::new(3);
                check("zeDeviceGetProperties", unsafe {
                    (api.zeDeviceGetProperties)(handle, &mut properties)
                })?;
                if properties.vendor != 0x8086 || properties.kind != 1 {
                    device_index += 1;
                    continue;
                }

                let mut compute: ComputeProperties = unsafe { std::mem::zeroed() };
                compute.header = Header::new(4);
                check("zeDeviceGetComputeProperties", unsafe {
                    (api.zeDeviceGetComputeProperties)(handle, &mut compute)
                })?;
                let mut module: ModuleProperties = unsafe { std::mem::zeroed() };
                module.header = Header::new(5);
                check("zeDeviceGetModuleProperties", unsafe {
                    (api.zeDeviceGetModuleProperties)(handle, &mut module)
                })?;
                let mut access: MemoryAccessProperties = unsafe { std::mem::zeroed() };
                access.header = Header::new(8);
                check("zeDeviceGetMemoryAccessProperties", unsafe {
                    (api.zeDeviceGetMemoryAccessProperties)(handle, &mut access)
                })?;
                let mut memory: MemoryProperties = unsafe { std::mem::zeroed() };
                memory.header = Header::new(7);
                let memory =
                    enumerate("zeDeviceGetMemoryProperties", memory, |count, out| unsafe {
                        (api.zeDeviceGetMemoryProperties)(handle, count, out)
                    })?;
                let total_memory = memory
                    .iter()
                    .try_fold(0u64, |total, region| total.checked_add(region.total_size))
                    .ok_or(Error::InvalidProperties)?;
                if metadata.is_none() {
                    let mut driver_info = DriverProperties {
                        header: Header::new(1),
                        version: 0,
                        uuid: [0; 16],
                    };
                    let mut version = 0;
                    check("zeDriverGetProperties", unsafe {
                        (api.zeDriverGetProperties)(driver, &mut driver_info)
                    })?;
                    check("zeDriverGetApiVersion", unsafe {
                        (api.zeDriverGetApiVersion)(driver, &mut version)
                    })?;
                    let initial: ExtensionProperties = unsafe { std::mem::zeroed() };
                    let values = enumerate(
                        "zeDriverGetExtensionProperties",
                        initial,
                        |count, out| unsafe {
                            (api.zeDriverGetExtensionProperties)(driver, count, out)
                        },
                    )?;
                    metadata = Some((
                        driver_info.version,
                        version,
                        values
                            .iter()
                            .map(|ext| Ok((name(&ext.name)?, ext.version)))
                            .collect::<Result<Vec<_>, Error>>()?,
                    ));
                }
                let subgroup_sizes = compute
                    .subgroup_sizes
                    .get(..compute.subgroup_count as usize)
                    .ok_or(Error::InvalidProperties)?
                    .to_vec();
                let mut group: QueueGroupProperties = unsafe { std::mem::zeroed() };
                group.header = Header::new(6);
                let groups = enumerate(
                    "zeDeviceGetCommandQueueGroupProperties",
                    group,
                    |count, out| unsafe {
                        (api.zeDeviceGetCommandQueueGroupProperties)(handle, count, out)
                    },
                )?;
                let queue_groups: Vec<_> = groups
                    .into_iter()
                    .enumerate()
                    .map(|(ordinal, group)| QueueGroup {
                        ordinal: ordinal as u32,
                        flags: group.flags,
                        queues: group.queues,
                    })
                    .collect();
                let (driver_version, api_version, extensions) = metadata.as_ref().unwrap();
                let mut atomic = FloatAtomics {
                    header: Header::new(0x10003),
                    flags: [0; 3],
                };
                let float_atomic_flags = if extensions
                    .iter()
                    .any(|(name, _)| name == "ZE_extension_float_atomics")
                {
                    // Query independently so an optional extension failure does
                    // not erase the core floating-point properties.
                    let mut query: ModuleProperties = unsafe { std::mem::zeroed() };
                    query.header = Header {
                        stype: 5,
                        next: std::ptr::from_mut(&mut atomic).cast(),
                    };
                    (unsafe { (api.zeDeviceGetModuleProperties)(handle, &mut query) } == 0)
                        .then_some(atomic.flags)
                } else {
                    None
                };
                let mut timer: Properties = unsafe { std::mem::zeroed() };
                timer.header = Header::new(0x0002_0006); // ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2
                let timestamp_frequency_hz =
                    (unsafe { (api.zeDeviceGetProperties)(handle, &mut timer) } == 0
                        && timer.timer_resolution != 0)
                        .then_some(timer.timer_resolution);
                let pci_address = api.get_pci.and_then(|get_pci| {
                    let mut pci = PciProperties {
                        header: Header::new(0x10008),
                        address: [0; 4],
                        generation: 0,
                        width: 0,
                        bandwidth: 0,
                    };
                    (unsafe { get_pci(handle, &mut pci) } == 0).then_some(pci.address)
                });
                devices.push(Device {
                    driver: Handle(driver),
                    handle: Handle(handle),
                    driver_index: driver_index as u32,
                    device_index,
                    probe: Default::default(),

                    info: DeviceInfo {
                        name: name(&properties.name)?,
                        vendor_id: properties.vendor,
                        device_id: properties.device,
                        device_flags: properties.flags,
                        pci_address,
                        driver_version: *driver_version,
                        api_version: *api_version,
                        uuid: properties.uuid,
                        total_memory,
                        max_allocation: properties.max_alloc_size,
                        max_workgroup_size: compute.max_group_size,
                        max_workgroup_dimensions: compute.group_size,
                        max_grid_dimensions: compute.group_count,
                        max_shared_local_memory: compute.shared_local_memory,
                        subgroup_sizes,
                        spirv_version: module.spirv_version,
                        fp32_denorms: module.fp32_flags & 1 != 0,
                        fp32_round_to_nearest: module.fp32_flags & 4 != 0,
                        fp32_inf_nan: module.fp32_flags & (1 << 1) != 0,
                        fp32_fma: module.fp32_flags & (1 << 5) != 0,
                        device_memory_atomics: access.device_alloc & (1 << 1) != 0,
                        module_flags: module.flags,
                        floating_point_flags: [
                            module.fp16_flags,
                            module.fp32_flags,
                            module.fp64_flags,
                        ],
                        float_atomic_flags,
                        memory_access_flags: [
                            access.host_alloc,
                            access.device_alloc,
                            access.shared_single_device_alloc,
                            access.shared_cross_device_alloc,
                            access.shared_system_alloc,
                        ],
                        memory_heaps: memory
                            .iter()
                            .map(|heap| {
                                Ok(MemoryHeap {
                                    name: name(&heap.name)?,
                                    size: heap.total_size,
                                    clock_mhz: heap.max_clock_rate,
                                    bus_width: heap.max_bus_width,
                                    flags: heap.flags,
                                })
                            })
                            .collect::<Result<_, Error>>()?,
                        max_kernel_arguments_size: module.max_arguments_size,
                        driver_extensions: extensions.clone(),
                        queue_groups,
                        timestamp_frequency_hz,
                        timestamp_valid_bits: properties.timestamp_valid_bits,
                        kernel_timestamp_valid_bits: properties.kernel_timestamp_valid_bits,
                    },
                });
                device_index += 1;
            }
        }
        if devices.is_empty()
            && let Some(error) = skipped
        {
            return Err(error);
        }
        Ok(Self { api, devices })
    }
}

/// Enumerate Intel root GPUs and their Level Zero capabilities.
pub fn intel_devices() -> Result<Vec<DeviceInfo>, Error> {
    Ok(Runtime::load()?
        .devices
        .into_iter()
        .map(|device| device.info)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_property_layouts_match_the_64_bit_c_abi() {
        use std::mem::{offset_of, size_of};
        assert_eq!(size_of::<Properties>(), 368);
        assert_eq!(offset_of!(Properties, max_alloc_size), 40);
        assert_eq!(offset_of!(Properties, name), 112);
        assert_eq!(size_of::<DriverProperties>(), 40);
        assert_eq!(size_of::<FloatAtomics>(), 32);
        assert_eq!(size_of::<PciProperties>(), 48);
        assert_eq!(offset_of!(PciProperties, bandwidth), 40);
        assert_eq!(size_of::<ComputeProperties>(), 88);
        assert_eq!(size_of::<MemoryProperties>(), 296);
        assert_eq!(size_of::<ModuleProperties>(), 64);
        assert_eq!(size_of::<MemoryAccessProperties>(), 40);
        assert_eq!(size_of::<ExtensionProperties>(), 260);
        assert_eq!(size_of::<QueueGroupProperties>(), 40);
        assert_eq!(offset_of!(QueueGroupProperties, queues), 32);
    }

    #[test]
    fn queue_selection_uses_capabilities_and_real_ordinals() {
        let groups = [
            QueueGroup {
                ordinal: 2,
                flags: 1,
                queues: 0,
            },
            QueueGroup {
                ordinal: 4,
                flags: 2,
                queues: 1,
            },
            QueueGroup {
                ordinal: 7,
                flags: 15,
                queues: 1,
            },
        ];
        assert_eq!(select_compute_queue(&groups), Some(groups[2]));
        assert_eq!(select_compute_queue(&groups[..2]), None);
        assert_eq!(select_compute_queue(&groups[2..]), Some(groups[2]));
        assert!(groups[1].copy_only());
        assert!(groups[2].copy() && !groups[2].copy_only());
    }

    #[test]
    fn enumeration_propagates_query_failure() {
        assert!(matches!(
            enumerate("test", 0u8, |_, _| 0x78000001),
            Err(Error::Driver("test", 0x78000001))
        ));
    }
}

/// Compiles SPIR-V with the Intel compiler and stores the native binary in the persistent
/// cache, so a later `load_module` of the same SPIR-V skips the compiler. `zeModuleCreate`
/// and `zeModuleDestroy` are thread-safe in Level Zero; the function pointers stay valid
/// because the loader library lives for the whole process.
pub(crate) struct NativeCompiler {
    create: unsafe extern "system" fn(Raw, Raw, *const ModuleDesc, *mut Raw, *mut Raw) -> Status,
    destroy: unsafe extern "system" fn(Raw) -> Status,
    native_binary: unsafe extern "system" fn(Raw, *mut usize, *mut u8) -> Status,
    log_destroy: unsafe extern "system" fn(Raw) -> Status,
    context: Handle,
    device: Handle,
    tag: String,
}

// Function pointers and opaque handles only.
unsafe impl Send for NativeCompiler {}
unsafe impl Sync for NativeCompiler {}

impl NativeCompiler {
    /// Whether the native cache already holds a binary for `words` on this device.
    pub(crate) fn is_cached(&self, words: &[u32]) -> bool {
        native_cache::directory().is_some_and(|dir| native_cache::contains(dir, &native_cache::key(words, &self.tag, &build_flags(words.len()))))
    }

    /// Returns true when a binary for `words` is in the cache afterwards.
    pub(crate) fn precompile(&self, words: &[u32]) -> bool {
        let Some(dir) = native_cache::directory() else { return false };
        let flags = build_flags(words.len());
        let key = native_cache::key(words, &self.tag, &flags);
        if native_cache::contains(dir, &key) {
            return true;
        }
        let flags = std::ffi::CString::new(flags).unwrap_or_default();
        let desc = ModuleDesc {
            header: Header::new(0x1b),
            format: 0,
            input_size: std::mem::size_of_val(words),
            input: words.as_ptr().cast(),
            build_flags: if flags.as_bytes().is_empty() { std::ptr::null() } else { flags.as_ptr() },
            constants: std::ptr::null(),
        };
        let (mut module, mut log) = (std::ptr::null_mut(), std::ptr::null_mut());
        let status = unsafe { (self.create)(self.context.0, self.device.0, &desc, &mut module, &mut log) };
        if !log.is_null() {
            unsafe { (self.log_destroy)(log) };
        }
        if status != 0 || module.is_null() {
            return false;
        }
        let mut size = 0usize;
        let mut stored = false;
        if unsafe { (self.native_binary)(module, &mut size, std::ptr::null_mut()) } == 0 && size != 0 {
            let mut data = vec![0u8; size];
            if unsafe { (self.native_binary)(module, &mut size, data.as_mut_ptr()) } == 0 {
                data.truncate(size);
                native_cache::store(dir, &key, &data);
                stored = true;
            }
        }
        unsafe { (self.destroy)(module) };
        stored
    }
}

#[cfg(test)]
mod build_flag_tests {
    use super::*;
    fn env(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> String {
        move |k| pairs.iter().find(|p| p.0 == k).map(|p| p.1.to_string()).unwrap_or_default()
    }
    #[test]
    fn correctly_rounded_div_is_default() {
        assert_eq!(compose_build_flags(10, &env(&[])), CORRECT_DIV_FLAG);
        assert_eq!(compose_build_flags(10, &env(&[("CUDA_FOR_INTEL_BUILD_FLAGS", "-x")])), format!("-x {CORRECT_DIV_FLAG}"));
        assert_eq!(compose_build_flags(10, &env(&[("CUDA_FOR_INTEL_BUILD_FLAGS", CORRECT_DIV_FLAG)])), CORRECT_DIV_FLAG);
    }
    #[test]
    fn fast_div_opt_out() {
        assert_eq!(compose_build_flags(10, &env(&[("CUDA_FOR_INTEL_FAST_DIV", "1")])), "");
    }
}
