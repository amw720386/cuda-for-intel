//! Minimal bindings for the Level Zero core API used by this crate.
use libloading::Library;
use std::ffi::{CStr, c_char, c_void};
use std::fmt;

mod streams;
pub(crate) use streams::{CommandList, Fence, Fences};
#[cfg(test)]
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

fn check(call: &'static str, status: Status) -> Result<(), Error> {
    match status {
        0 => Ok(()),
        _ => Err(Error::Driver(call, status)),
    }
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

macro_rules! loader_api {
    ($( $name:ident($($arg:ty),*) ),+ $(,)?) => {
        #[allow(non_snake_case)]
        pub(crate) struct Api {
            $( $name: unsafe extern "system" fn($($arg),*) -> Status, )+
            _library: Library,
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

                Ok(Self {
                    $( $name: unsafe { library.get(concat!(stringify!($name), "\0").as_bytes()) }
                        .map(|symbol| *symbol).map_err(Error::Loader)?, )+
                    _library: library,
                })
            }
        }
    };
}

loader_api! {
    zeInit(u32),
    zeDriverGet(*mut u32, *mut Raw),
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
    zeCommandListDestroy(Raw),
    zeCommandListHostSynchronize(Raw, u64),
    zeCommandListAppendSignalEvent(Raw, Raw),
    zeCommandListAppendBarrier(Raw, Raw, u32, *mut Raw),
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
    zeModuleCreate(Raw, Raw, *const ModuleDesc, *mut Raw, *mut Raw),
    zeModuleDestroy(Raw),
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
    pub device_id: u32,
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
    pub max_kernel_arguments_size: u32,
    pub driver_extensions: Vec<(String, u32)>,
    pub queue_groups: Vec<QueueGroup>,
    /// Timestamp counter frequency, when the driver supports the 1.2 properties.
    pub timestamp_frequency_hz: Option<u64>,
    pub timestamp_valid_bits: u32,
    pub kernel_timestamp_valid_bits: u32,
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

fn select_compute_queue(groups: &[QueueGroup]) -> Option<u32> {
    groups
        .iter()
        .find(|group| group.queues != 0 && group.compute())
        .map(|group| group.ordinal)
}

pub(crate) struct Device {
    pub(crate) info: DeviceInfo,
    pub(crate) device_memory_atomics: bool,
    pub(crate) fp32_inf_nan: bool,
    pub(crate) fp32_fma: bool,
    driver: Handle,
    handle: Handle,
    compute_queue: Option<u32>,
}

pub(crate) struct Runtime {
    pub(crate) api: Api,
    pub(crate) devices: Vec<Device>,
}

pub(crate) struct Context {
    handle: Handle,
    device: Handle,
    commands: Option<Handle>,
    ordinal: u32,
}

pub(crate) struct Module {
    handle: Handle,
    kernel: Option<Handle>,
}

impl Module {
    pub(crate) fn create_kernel(&mut self, api: &Api, name: &CStr) -> Result<(), Error> {
        if self.kernel.is_none() {
            let desc = KernelDesc {
                header: Header::new(0x1d),
                flags: 0,
                name: name.as_ptr(),
            };
            let mut kernel = std::ptr::null_mut();
            check("zeKernelCreate", unsafe {
                (api.zeKernelCreate)(self.handle.0, &desc, &mut kernel)
            })?;
            self.kernel = Some(Handle(kernel));
        }
        Ok(())
    }

    pub(crate) fn destroy(&mut self, api: &Api) -> Result<(), Error> {
        if let Some(kernel) = self.kernel {
            check("zeKernelDestroy", unsafe {
                (api.zeKernelDestroy)(kernel.0)
            })?;
            self.kernel = None;
        }
        check("zeModuleDestroy", unsafe {
            (api.zeModuleDestroy)(self.handle.0)
        })
    }
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
            .compute_queue
            .ok_or(Error::Driver("select compute queue", 0x78000003))?;
        let desc = ContextDesc {
            header: Header::new(0xd),
            flags: 0,
        };
        let mut handle = std::ptr::null_mut();
        check("zeContextCreate", unsafe {
            (api.zeContextCreate)(device.driver.0, &desc, &mut handle)
        })?;
        let queue = QueueDesc {
            header: Header::new(0xe),
            ordinal,
            index: 0,
            flags: 0,
            mode: 1, // ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS
            priority: 0,
        };
        let mut commands = std::ptr::null_mut();
        if let Err(error) = check("zeCommandListCreateImmediate", unsafe {
            (api.zeCommandListCreateImmediate)(handle, device.handle.0, &queue, &mut commands)
        }) {
            unsafe { (api.zeContextDestroy)(handle) };
            return Err(error);
        }
        Ok(Self {
            handle: Handle(handle),
            device: device.handle,
            commands: Some(Handle(commands)),
            ordinal,
        })
    }

    fn commands(&self) -> Result<Raw, Error> {
        self.commands
            .map(|handle| handle.0)
            .ok_or(Error::Driver("use closed command list", 0x78000005))
    }

    pub(crate) fn destroy(&mut self, api: &Api) -> Result<(), Error> {
        if let Some(commands) = self.commands {
            check("zeCommandListDestroy", unsafe {
                (api.zeCommandListDestroy)(commands.0)
            })?;
            self.commands = None;
        }
        check("zeContextDestroy", unsafe {
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
        check("zeMemAllocDevice", unsafe {
            (api.zeMemAllocDevice)(self.handle.0, &desc, size, 0, self.device.0, &mut out)
        })?;
        Ok(out.expose_provenance())
    }

    pub(crate) fn alloc_host(&self, api: &Api, size: usize) -> Result<usize, Error> {
        self.commands()?;
        let desc = HostAllocDesc {
            header: Header::new(0x16),
            flags: 0,
        };
        let mut out = std::ptr::null_mut();
        check("zeMemAllocHost", unsafe {
            (api.zeMemAllocHost)(self.handle.0, &desc, size, 4096, &mut out)
        })?;
        Ok(out.expose_provenance())
    }

    pub(crate) fn free(&self, api: &Api, address: usize) -> Result<(), Error> {
        check("zeMemFree", unsafe {
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
        check("zeCommandListAppendMemoryCopy", unsafe {
            (api.zeCommandListAppendMemoryCopy)(
                commands,
                dst,
                src,
                size,
                signal,
                0,
                std::ptr::null_mut(),
            )
        })
    }

    pub(crate) fn load_module(&self, api: &Api, words: &[u32]) -> Result<Module, Error> {
        self.commands()?;
        let desc = ModuleDesc {
            header: Header::new(0x1b),
            format: 0, // ZE_MODULE_FORMAT_IL_SPIRV
            input_size: std::mem::size_of_val(words),
            input: words.as_ptr().cast(),
            build_flags: std::ptr::null(),
            constants: std::ptr::null(),
        };
        let mut module = std::ptr::null_mut();
        let mut log = std::ptr::null_mut();
        let status = unsafe {
            (api.zeModuleCreate)(self.handle.0, self.device.0, &desc, &mut module, &mut log)
        };
        let message = if !log.is_null() {
            let message = if status == 0 {
                String::new()
            } else {
                build_log(api, log)
            };
            unsafe { (api.zeModuleBuildLogDestroy)(log) };
            message
        } else {
            String::new()
        };
        if status != 0 {
            return Err(Error::Build(status, message));
        }
        Ok(Module {
            handle: Handle(module),
            kernel: None,
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
        for driver in drivers {
            let handles = enumerate("zeDeviceGet", std::ptr::null_mut(), |count, out| unsafe {
                (api.zeDeviceGet)(driver, count, out)
            })?;
            let mut extensions = None;
            for handle in handles {
                // These C output structures contain only integers and pointers.
                let mut properties: Properties = unsafe { std::mem::zeroed() };
                properties.header = Header::new(3);
                check("zeDeviceGetProperties", unsafe {
                    (api.zeDeviceGetProperties)(handle, &mut properties)
                })?;
                if properties.vendor != 0x8086 || properties.kind != 1 {
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
                if extensions.is_none() {
                    let initial: ExtensionProperties = unsafe { std::mem::zeroed() };
                    let values = enumerate(
                        "zeDriverGetExtensionProperties",
                        initial,
                        |count, out| unsafe {
                            (api.zeDriverGetExtensionProperties)(driver, count, out)
                        },
                    )?;
                    extensions = Some(
                        values
                            .iter()
                            .map(|ext| Ok((name(&ext.name)?, ext.version)))
                            .collect::<Result<Vec<_>, Error>>()?,
                    );
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
                let compute_queue = select_compute_queue(&queue_groups);
                let mut timer: Properties = unsafe { std::mem::zeroed() };
                timer.header = Header::new(0x0002_0006); // ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2
                let timestamp_frequency_hz =
                    (unsafe { (api.zeDeviceGetProperties)(handle, &mut timer) } == 0
                        && timer.timer_resolution != 0)
                        .then_some(timer.timer_resolution);
                devices.push(Device {
                    driver: Handle(driver),
                    handle: Handle(handle),
                    compute_queue,
                    device_memory_atomics: access.device_alloc & (1 << 1) != 0,
                    fp32_inf_nan: module.fp32_flags & (1 << 1) != 0,
                    fp32_fma: module.fp32_flags & (1 << 5) != 0,
                    info: DeviceInfo {
                        name: name(&properties.name)?,
                        device_id: properties.device,
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
                        max_kernel_arguments_size: module.max_arguments_size,
                        driver_extensions: extensions.as_ref().cloned().unwrap_or_default(),
                        queue_groups,
                        timestamp_frequency_hz,
                        timestamp_valid_bits: properties.timestamp_valid_bits,
                        kernel_timestamp_valid_bits: properties.kernel_timestamp_valid_bits,
                    },
                });
            }
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
        assert_eq!(select_compute_queue(&groups), Some(7));
        assert_eq!(select_compute_queue(&groups[..2]), None);
        assert_eq!(select_compute_queue(&groups[2..]), Some(7));
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
