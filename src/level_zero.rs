//! Minimal bindings for the Level Zero core API used by this crate.
use libloading::Library;
use std::ffi::{CStr, c_char, c_void};
use std::fmt;

type Raw = *mut c_void;
type Status = u32;

#[derive(Debug)]
pub enum Error {
    Loader(libloading::Error),
    Driver(&'static str, u32),
    InvalidProperties,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Loader(error) => write!(f, "Level Zero loader: {error}"),
            Self::Driver(call, status) => {
                write!(f, "{call} failed with Level Zero status {status:#010x}")
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
#[derive(Clone, Copy)]
struct ExtensionProperties {
    name: [c_char; 256],
    version: u32,
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
    zeDeviceGetModuleProperties(Raw, *mut ModuleProperties),
    zeDriverGetExtensionProperties(Raw, *mut u32, *mut ExtensionProperties),
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
    pub max_kernel_arguments_size: u32,
    pub driver_extensions: Vec<(String, u32)>,
}

pub(crate) struct Device {
    pub(crate) info: DeviceInfo,
}

pub(crate) struct Runtime {
    _api: Api,
    pub(crate) devices: Vec<Device>,
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
                devices.push(Device {
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
                        max_kernel_arguments_size: module.max_arguments_size,
                        driver_extensions: extensions.as_ref().cloned().unwrap_or_default(),
                    },
                });
            }
        }
        Ok(Self { _api: api, devices })
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
        assert_eq!(size_of::<ExtensionProperties>(), 260);
    }

    #[test]
    fn enumeration_propagates_query_failure() {
        assert!(matches!(
            enumerate("test", 0u8, |_, _| 0x78000001),
            Err(Error::Driver("test", 0x78000001))
        ));
    }
}
