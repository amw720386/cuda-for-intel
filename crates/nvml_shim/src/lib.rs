//! NVML shim. Applications (PyTorch `torch.cuda.mem_get_info`, nvidia-smi style tools,
//! `pynvml`, CuPy, ONNX Runtime) query NVML next to the CUDA driver. This library answers
//! what the Driver API can answer (device count, name, memory, PCI id, UUID, compute
//! capability, driver versions) and returns `NVML_ERROR_NOT_SUPPORTED` for sensors,
//! clocks, power, ECC, processes and everything else (see `stubs.rs`). Nothing here
//! panics or aborts: bad arguments return `NVML_ERROR_INVALID_ARGUMENT`.
#![allow(non_snake_case)]

mod backend;
mod stubs;

use backend::{backend, Backend};
use std::ffi::{c_char, c_int, c_uint, c_void, CStr};
use std::sync::Mutex;

pub type NvmlReturn = c_uint;
pub const SUCCESS: NvmlReturn = 0;
pub const UNINITIALIZED: NvmlReturn = 1;
pub const INVALID_ARGUMENT: NvmlReturn = 2;
pub const NOT_SUPPORTED: NvmlReturn = 3;
pub const NOT_FOUND: NvmlReturn = 6;
pub const INSUFFICIENT_SIZE: NvmlReturn = 7;
pub const DRIVER_NOT_LOADED: NvmlReturn = 9;
pub const GPU_NOT_FOUND: NvmlReturn = 28;
pub const UNKNOWN: NvmlReturn = 999;

pub const NVML_VERSION: &str = "12.9.79";

static INIT_COUNT: Mutex<usize> = Mutex::new(0);

fn initialized() -> bool {
    INIT_COUNT.lock().map(|count| *count > 0).unwrap_or(false)
}

/// Text for `nvmlErrorString`.
pub fn error_text(code: NvmlReturn) -> &'static CStr {
    match code {
        0 => c"Success",
        1 => c"Uninitialized",
        2 => c"Invalid Argument",
        3 => c"Not Supported",
        4 => c"Insufficient Permissions",
        5 => c"Already Initialized",
        6 => c"Not Found",
        7 => c"Insufficient Size",
        8 => c"Insufficient External Power",
        9 => c"Driver Not Loaded",
        10 => c"Timeout",
        11 => c"Interrupt Request Issue",
        12 => c"NVML Shared Library Not Found",
        13 => c"Function Not Found",
        14 => c"Corrupted infoROM",
        15 => c"GPU is lost",
        16 => c"GPU requires restart",
        17 => c"The operating system has blocked the request.",
        18 => c"RM has detected an NVML/RM version mismatch.",
        19 => c"In use by another client",
        20 => c"Insufficient Memory",
        21 => c"No data",
        22 => c"The requested vgpu operation is not available on target device, because ECC is enabled",
        23 => c"Ran out of critical resources, other than memory",
        24 => c"Ran out of critical resources, other than memory",
        25 => c"The provided version is invalid/unsupported",
        26 => c"The requested functionality has been deprecated",
        27 => c"The system is not ready for the request",
        28 => c"No GPUs were found",
        29 => c"Resource not in correct state to perform requested operation",
        _ => c"Unknown Error",
    }
}

/// Copy `text` plus a NUL into `buffer` of `length` bytes.
unsafe fn write_text(buffer: *mut c_char, length: c_uint, text: &str) -> NvmlReturn {
    if buffer.is_null() {
        return INVALID_ARGUMENT;
    }
    let bytes = text.as_bytes();
    if (length as usize) < bytes.len() + 1 {
        return INSUFFICIENT_SIZE;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.cast::<u8>(), bytes.len());
        buffer.add(bytes.len()).cast::<u8>().write(0);
    }
    SUCCESS
}

/// Device handles encode `index + 1` so a null handle is always invalid.
pub fn handle_for(index: usize) -> *mut c_void {
    std::ptr::without_provenance_mut(index + 1)
}

fn index_of(handle: *mut c_void) -> Result<usize, NvmlReturn> {
    if !initialized() {
        return Err(UNINITIALIZED);
    }
    let address = handle.addr();
    if address == 0 {
        return Err(INVALID_ARGUMENT);
    }
    let index = address - 1;
    match backend().device_count() {
        Ok(count) if index < count => Ok(index),
        Ok(_) => Err(INVALID_ARGUMENT),
        Err(code) => Err(code),
    }
}

fn with_device(handle: *mut c_void, action: impl FnOnce(&dyn Backend, usize) -> NvmlReturn) -> NvmlReturn {
    match index_of(handle) {
        Ok(index) => action(&*backend(), index),
        Err(code) => code,
    }
}

// ---------------------------------------------------------------------------
// Lifecycle and system queries
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "system" fn nvmlInit_v2() -> NvmlReturn {
    // Without a loadable driver NVML reports it the way the real library does.
    if let Err(code) = backend().device_count() {
        return if code == NOT_FOUND { GPU_NOT_FOUND } else { code };
    }
    match INIT_COUNT.lock() {
        Ok(mut count) => {
            *count += 1;
            SUCCESS
        }
        Err(_) => UNKNOWN,
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn nvmlInit() -> NvmlReturn {
    nvmlInit_v2()
}

#[unsafe(no_mangle)]
pub extern "system" fn nvmlInitWithFlags(_flags: c_uint) -> NvmlReturn {
    nvmlInit_v2()
}

#[unsafe(no_mangle)]
pub extern "system" fn nvmlShutdown() -> NvmlReturn {
    match INIT_COUNT.lock() {
        Ok(mut count) if *count > 0 => {
            *count -= 1;
            SUCCESS
        }
        Ok(_) => UNINITIALIZED,
        Err(_) => UNKNOWN,
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn nvmlErrorString(code: NvmlReturn) -> *const c_char {
    error_text(code).as_ptr()
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn nvmlSystemGetDriverVersion(version: *mut c_char, length: c_uint) -> NvmlReturn {
    if !initialized() {
        return UNINITIALIZED;
    }
    unsafe { write_text(version, length, &backend().driver_version()) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn nvmlSystemGetNVMLVersion(version: *mut c_char, length: c_uint) -> NvmlReturn {
    unsafe { write_text(version, length, NVML_VERSION) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn nvmlSystemGetCudaDriverVersion(version: *mut c_int) -> NvmlReturn {
    if version.is_null() {
        return INVALID_ARGUMENT;
    }
    if !initialized() {
        return UNINITIALIZED;
    }
    unsafe { version.write(backend().cuda_driver_version()) };
    SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn nvmlSystemGetCudaDriverVersion_v2(version: *mut c_int) -> NvmlReturn {
    unsafe { nvmlSystemGetCudaDriverVersion(version) }
}

/// Process names are not tracked.
#[unsafe(no_mangle)]
pub extern "system" fn nvmlSystemGetProcessName(_pid: c_uint, _name: *mut c_char, _length: c_uint) -> NvmlReturn {
    if initialized() { NOT_FOUND } else { UNINITIALIZED }
}

// ---------------------------------------------------------------------------
// Devices
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "system" fn nvmlDeviceGetCount_v2(count: *mut c_uint) -> NvmlReturn {
    if count.is_null() {
        return INVALID_ARGUMENT;
    }
    if !initialized() {
        return UNINITIALIZED;
    }
    match backend().device_count() {
        Ok(n) => {
            unsafe { count.write(n as c_uint) };
            SUCCESS
        }
        Err(code) => code,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn nvmlDeviceGetCount(count: *mut c_uint) -> NvmlReturn {
    unsafe { nvmlDeviceGetCount_v2(count) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn nvmlDeviceGetHandleByIndex_v2(index: c_uint, device: *mut *mut c_void) -> NvmlReturn {
    if device.is_null() {
        return INVALID_ARGUMENT;
    }
    if !initialized() {
        return UNINITIALIZED;
    }
    match backend().device_count() {
        Ok(count) if (index as usize) < count => {
            unsafe { device.write(handle_for(index as usize)) };
            SUCCESS
        }
        Ok(_) => INVALID_ARGUMENT,
        Err(code) => code,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn nvmlDeviceGetHandleByIndex(index: c_uint, device: *mut *mut c_void) -> NvmlReturn {
    unsafe { nvmlDeviceGetHandleByIndex_v2(index, device) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn nvmlDeviceGetIndex(device: *mut c_void, index: *mut c_uint) -> NvmlReturn {
    if index.is_null() {
        return INVALID_ARGUMENT;
    }
    match index_of(device) {
        Ok(i) => {
            unsafe { index.write(i as c_uint) };
            SUCCESS
        }
        Err(code) => code,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn nvmlDeviceGetMinorNumber(device: *mut c_void, minor: *mut c_uint) -> NvmlReturn {
    unsafe { nvmlDeviceGetIndex(device, minor) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn nvmlDeviceGetName(device: *mut c_void, name: *mut c_char, length: c_uint) -> NvmlReturn {
    with_device(device, |b, i| match b.name(i) {
        Some(text) => unsafe { write_text(name, length, &text) },
        None => NOT_SUPPORTED,
    })
}

/// `GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`
pub fn format_uuid(bytes: [u8; 16]) -> String {
    let hex: Vec<String> = bytes.iter().map(|b| format!("{b:02x}")).collect();
    format!(
        "GPU-{}-{}-{}-{}-{}",
        hex[0..4].concat(),
        hex[4..6].concat(),
        hex[6..8].concat(),
        hex[8..10].concat(),
        hex[10..16].concat()
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn nvmlDeviceGetUUID(device: *mut c_void, uuid: *mut c_char, length: c_uint) -> NvmlReturn {
    with_device(device, |b, i| match b.uuid(i) {
        Some(bytes) => unsafe { write_text(uuid, length, &format_uuid(bytes)) },
        None => NOT_SUPPORTED,
    })
}

#[repr(C)]
pub struct Memory {
    pub total: u64,
    pub free: u64,
    pub used: u64,
}

/// (total, free, used); free falls back to total when no context is current.
fn memory_of(b: &dyn Backend, i: usize) -> Option<Memory> {
    let total = b.total_memory(i)?;
    let free = b.free_memory(i).unwrap_or(total).min(total);
    Some(Memory { total, free, used: total - free })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn nvmlDeviceGetMemoryInfo(device: *mut c_void, memory: *mut Memory) -> NvmlReturn {
    if memory.is_null() {
        return INVALID_ARGUMENT;
    }
    with_device(device, |b, i| match memory_of(b, i) {
        Some(m) => {
            unsafe { memory.write(m) };
            SUCCESS
        }
        None => NOT_SUPPORTED,
    })
}

#[repr(C)]
pub struct MemoryV2 {
    pub version: c_uint,
    pub total: u64,
    pub reserved: u64,
    pub free: u64,
    pub used: u64,
}

/// `nvmlMemory_v2_t` version word: sizeof(struct) | (2 << 24).
pub const MEMORY_V2_VERSION: c_uint = (std::mem::size_of::<MemoryV2>() as c_uint) | (2 << 24);

#[unsafe(no_mangle)]
pub unsafe extern "system" fn nvmlDeviceGetMemoryInfo_v2(device: *mut c_void, memory: *mut MemoryV2) -> NvmlReturn {
    if memory.is_null() {
        return INVALID_ARGUMENT;
    }
    if unsafe { (*memory).version } != MEMORY_V2_VERSION {
        return 25; // NVML_ERROR_ARGUMENT_VERSION_MISMATCH
    }
    with_device(device, |b, i| match memory_of(b, i) {
        Some(m) => {
            unsafe { memory.write(MemoryV2 { version: MEMORY_V2_VERSION, total: m.total, reserved: 0, free: m.free, used: m.used }) };
            SUCCESS
        }
        None => NOT_SUPPORTED,
    })
}

/// `nvmlPciInfo_t`
#[repr(C)]
pub struct PciInfo {
    pub bus_id_legacy: [c_char; 16],
    pub domain: c_uint,
    pub bus: c_uint,
    pub device: c_uint,
    pub pci_device_id: c_uint,
    pub pci_subsystem_id: c_uint,
    pub bus_id: [c_char; 32],
}

/// Parse `dddd:bb:ss.f` into (domain, bus, device).
pub fn parse_bus_id(text: &str) -> Option<(u32, u32, u32)> {
    let mut parts = text.split(':');
    let domain = u32::from_str_radix(parts.next()?, 16).ok()?;
    let bus = u32::from_str_radix(parts.next()?, 16).ok()?;
    let device = u32::from_str_radix(parts.next()?.split('.').next()?, 16).ok()?;
    Some((domain, bus, device))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn nvmlDeviceGetPciInfo_v3(device: *mut c_void, pci: *mut PciInfo) -> NvmlReturn {
    if pci.is_null() {
        return INVALID_ARGUMENT;
    }
    with_device(device, |b, i| {
        let Some(id) = b.pci_bus_id(i) else { return NOT_SUPPORTED };
        let Some((domain, bus, dev)) = parse_bus_id(&id) else { return NOT_SUPPORTED };
        let mut info = PciInfo { bus_id_legacy: [0; 16], domain, bus, device: dev, pci_device_id: 0, pci_subsystem_id: 0, bus_id: [0; 32] };
        for (slot, byte) in info.bus_id.iter_mut().zip(id.bytes().take(31)) {
            *slot = byte as c_char;
        }
        for (slot, byte) in info.bus_id_legacy.iter_mut().zip(id.bytes().skip(4).take(15)) {
            *slot = byte as c_char;
        }
        unsafe { pci.write(info) };
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn nvmlDeviceGetPciInfo_v2(device: *mut c_void, pci: *mut PciInfo) -> NvmlReturn {
    unsafe { nvmlDeviceGetPciInfo_v3(device, pci) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn nvmlDeviceGetPciInfo(device: *mut c_void, pci: *mut PciInfo) -> NvmlReturn {
    unsafe { nvmlDeviceGetPciInfo_v3(device, pci) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn nvmlDeviceGetHandleByPciBusId_v2(bus_id: *const c_char, device: *mut *mut c_void) -> NvmlReturn {
    if bus_id.is_null() || device.is_null() {
        return INVALID_ARGUMENT;
    }
    if !initialized() {
        return UNINITIALIZED;
    }
    let wanted = unsafe { CStr::from_ptr(bus_id) }.to_string_lossy().to_ascii_lowercase();
    let b = backend();
    let count = match b.device_count() {
        Ok(count) => count,
        Err(code) => return code,
    };
    for i in 0..count {
        if b.pci_bus_id(i).is_some_and(|id| id.to_ascii_lowercase() == wanted) {
            unsafe { device.write(handle_for(i)) };
            return SUCCESS;
        }
    }
    NOT_FOUND
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn nvmlDeviceGetHandleByPciBusId(bus_id: *const c_char, device: *mut *mut c_void) -> NvmlReturn {
    unsafe { nvmlDeviceGetHandleByPciBusId_v2(bus_id, device) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn nvmlDeviceGetHandleByUUID(uuid: *const c_char, device: *mut *mut c_void) -> NvmlReturn {
    if uuid.is_null() || device.is_null() {
        return INVALID_ARGUMENT;
    }
    if !initialized() {
        return UNINITIALIZED;
    }
    let wanted = unsafe { CStr::from_ptr(uuid) }.to_string_lossy().to_ascii_lowercase();
    let b = backend();
    let count = match b.device_count() {
        Ok(count) => count,
        Err(code) => return code,
    };
    for i in 0..count {
        if b.uuid(i).is_some_and(|bytes| format_uuid(bytes).to_ascii_lowercase() == wanted) {
            unsafe { device.write(handle_for(i)) };
            return SUCCESS;
        }
    }
    NOT_FOUND
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn nvmlDeviceGetCudaComputeCapability(device: *mut c_void, major: *mut c_int, minor: *mut c_int) -> NvmlReturn {
    if major.is_null() || minor.is_null() {
        return INVALID_ARGUMENT;
    }
    with_device(device, |b, i| match b.compute_capability(i) {
        Some((high, low)) => {
            unsafe {
                major.write(high);
                minor.write(low);
            }
            SUCCESS
        }
        None => NOT_SUPPORTED,
    })
}

/// Compute mode DEFAULT (0): the device is shared.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn nvmlDeviceGetComputeMode(device: *mut c_void, mode: *mut c_uint) -> NvmlReturn {
    if mode.is_null() {
        return INVALID_ARGUMENT;
    }
    with_device(device, |_, _| {
        unsafe { mode.write(0) };
        SUCCESS
    })
}

/// Running-process lists are not tracked: report an empty list.
unsafe fn no_processes(device: *mut c_void, count: *mut c_uint) -> NvmlReturn {
    if count.is_null() {
        return INVALID_ARGUMENT;
    }
    with_device(device, |_, _| {
        unsafe { count.write(0) };
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn nvmlDeviceGetComputeRunningProcesses_v3(device: *mut c_void, count: *mut c_uint, _infos: *mut c_void) -> NvmlReturn {
    unsafe { no_processes(device, count) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn nvmlDeviceGetComputeRunningProcesses_v2(device: *mut c_void, count: *mut c_uint, _infos: *mut c_void) -> NvmlReturn {
    unsafe { no_processes(device, count) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn nvmlDeviceGetComputeRunningProcesses(device: *mut c_void, count: *mut c_uint, _infos: *mut c_void) -> NvmlReturn {
    unsafe { no_processes(device, count) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn nvmlDeviceGetGraphicsRunningProcesses_v3(device: *mut c_void, count: *mut c_uint, _infos: *mut c_void) -> NvmlReturn {
    unsafe { no_processes(device, count) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn nvmlDeviceGetMPSComputeRunningProcesses_v3(device: *mut c_void, count: *mut c_uint, _infos: *mut c_void) -> NvmlReturn {
    unsafe { no_processes(device, count) }
}

#[cfg(test)]
mod tests;
