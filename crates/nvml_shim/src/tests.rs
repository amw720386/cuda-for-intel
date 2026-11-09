use super::backend::{set_backend, Backend};
use super::*;
use std::ptr::null_mut;
use std::sync::{Arc, Mutex, MutexGuard};

static SERIAL: Mutex<()> = Mutex::new(());

struct Fake {
    devices: usize,
    free: Option<u64>,
}

impl Backend for Fake {
    fn device_count(&self) -> Result<usize, u32> {
        if self.devices == usize::MAX { Err(9) } else { Ok(self.devices) }
    }
    fn name(&self, i: usize) -> Option<String> {
        Some(format!("Intel(R) Arc(TM) Fake {i}"))
    }
    fn total_memory(&self, _: usize) -> Option<u64> {
        Some(16 << 30)
    }
    fn free_memory(&self, _: usize) -> Option<u64> {
        self.free
    }
    fn pci_bus_id(&self, i: usize) -> Option<String> {
        Some(format!("0000:{:02x}:00.0", i + 3))
    }
    fn uuid(&self, i: usize) -> Option<[u8; 16]> {
        let mut bytes = [0xab; 16];
        bytes[0] = i as u8;
        Some(bytes)
    }
    fn compute_capability(&self, _: usize) -> Option<(i32, i32)> {
        Some((7, 5))
    }
    fn cuda_driver_version(&self) -> i32 {
        13000
    }
    fn driver_version(&self) -> String {
        "13.0".into()
    }
}

struct Session(#[allow(dead_code)] MutexGuard<'static, ()>);

impl Session {
    fn new(devices: usize, free: Option<u64>) -> Self {
        let guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        while nvmlShutdown() == SUCCESS {}
        set_backend(Some(Arc::new(Fake { devices, free })));
        Self(guard)
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        while nvmlShutdown() == SUCCESS {}
        set_backend(None);
    }
}

fn text(buffer: &[c_char]) -> String {
    let bytes: Vec<u8> = buffer.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
    String::from_utf8(bytes).unwrap()
}

#[test]
fn queries_fail_cleanly_before_init() {
    let _s = Session::new(1, None);
    let mut count = 7;
    assert_eq!(unsafe { nvmlDeviceGetCount_v2(&mut count) }, UNINITIALIZED);
    let mut handle = null_mut();
    assert_eq!(unsafe { nvmlDeviceGetHandleByIndex_v2(0, &mut handle) }, UNINITIALIZED);
    assert_eq!(nvmlShutdown(), UNINITIALIZED);
}

#[test]
fn init_is_reference_counted() {
    let _s = Session::new(1, None);
    assert_eq!(nvmlInit_v2(), SUCCESS);
    assert_eq!(nvmlInit(), SUCCESS);
    assert_eq!(nvmlShutdown(), SUCCESS);
    let mut count = 0;
    assert_eq!(unsafe { nvmlDeviceGetCount_v2(&mut count) }, SUCCESS);
    assert_eq!(count, 1);
    assert_eq!(nvmlShutdown(), SUCCESS);
    assert_eq!(nvmlShutdown(), UNINITIALIZED);
}

#[test]
fn init_reports_missing_driver() {
    let _s = Session::new(usize::MAX, None);
    assert_eq!(nvmlInit_v2(), DRIVER_NOT_LOADED);
}

#[test]
fn device_identity_queries() {
    let _s = Session::new(2, Some(4 << 30));
    assert_eq!(nvmlInit_v2(), SUCCESS);
    let mut handle = null_mut();
    assert_eq!(unsafe { nvmlDeviceGetHandleByIndex_v2(1, &mut handle) }, SUCCESS);
    assert_eq!(unsafe { nvmlDeviceGetHandleByIndex_v2(2, &mut handle) }, INVALID_ARGUMENT);
    assert_eq!(unsafe { nvmlDeviceGetHandleByIndex_v2(1, &mut handle) }, SUCCESS);

    let mut name = [0 as c_char; 96];
    assert_eq!(unsafe { nvmlDeviceGetName(handle, name.as_mut_ptr(), 96) }, SUCCESS);
    assert_eq!(text(&name), "Intel(R) Arc(TM) Fake 1");
    let mut tiny = [0 as c_char; 4];
    assert_eq!(unsafe { nvmlDeviceGetName(handle, tiny.as_mut_ptr(), 4) }, INSUFFICIENT_SIZE);
    assert_eq!(unsafe { nvmlDeviceGetName(null_mut(), name.as_mut_ptr(), 96) }, INVALID_ARGUMENT);
    assert_eq!(unsafe { nvmlDeviceGetName(handle, std::ptr::null_mut(), 96) }, INVALID_ARGUMENT);

    let mut index = 9;
    assert_eq!(unsafe { nvmlDeviceGetIndex(handle, &mut index) }, SUCCESS);
    assert_eq!(index, 1);

    let mut uuid = [0 as c_char; 96];
    assert_eq!(unsafe { nvmlDeviceGetUUID(handle, uuid.as_mut_ptr(), 96) }, SUCCESS);
    assert_eq!(text(&uuid), "GPU-01ababab-abab-abab-abab-abababababab");
    let mut found = null_mut();
    assert_eq!(unsafe { nvmlDeviceGetHandleByUUID(uuid.as_ptr(), &mut found) }, SUCCESS);
    assert_eq!(found, handle);
    assert_eq!(unsafe { nvmlDeviceGetHandleByUUID(c"GPU-nope".as_ptr(), &mut found) }, NOT_FOUND);

    let (mut major, mut minor) = (0, 0);
    assert_eq!(unsafe { nvmlDeviceGetCudaComputeCapability(handle, &mut major, &mut minor) }, SUCCESS);
    assert_eq!((major, minor), (7, 5));
    assert_eq!(nvmlShutdown(), SUCCESS);
}

#[test]
fn memory_and_pci() {
    let _s = Session::new(1, Some(4 << 30));
    assert_eq!(nvmlInit_v2(), SUCCESS);
    let mut handle = null_mut();
    assert_eq!(unsafe { nvmlDeviceGetHandleByIndex_v2(0, &mut handle) }, SUCCESS);

    let mut memory = Memory { total: 0, free: 0, used: 0 };
    assert_eq!(unsafe { nvmlDeviceGetMemoryInfo(handle, &mut memory) }, SUCCESS);
    assert_eq!((memory.total, memory.free, memory.used), (16 << 30, 4 << 30, 12 << 30));
    assert_eq!(unsafe { nvmlDeviceGetMemoryInfo(handle, std::ptr::null_mut()) }, INVALID_ARGUMENT);

    let mut v2 = MemoryV2 { version: MEMORY_V2_VERSION, total: 0, reserved: 0, free: 0, used: 0 };
    assert_eq!(unsafe { nvmlDeviceGetMemoryInfo_v2(handle, &mut v2) }, SUCCESS);
    assert_eq!((v2.total, v2.free, v2.used), (16 << 30, 4 << 30, 12 << 30));
    v2.version = 1;
    assert_eq!(unsafe { nvmlDeviceGetMemoryInfo_v2(handle, &mut v2) }, 25);

    let mut pci: PciInfo = unsafe { std::mem::zeroed() };
    assert_eq!(unsafe { nvmlDeviceGetPciInfo_v3(handle, &mut pci) }, SUCCESS);
    assert_eq!((pci.domain, pci.bus, pci.device), (0, 3, 0));
    assert_eq!(text(&pci.bus_id), "0000:03:00.0");
    let mut found = null_mut();
    assert_eq!(unsafe { nvmlDeviceGetHandleByPciBusId_v2(c"0000:03:00.0".as_ptr(), &mut found) }, SUCCESS);
    assert_eq!(found, handle);
    assert_eq!(unsafe { nvmlDeviceGetHandleByPciBusId_v2(c"0000:09:00.0".as_ptr(), &mut found) }, NOT_FOUND);
    assert_eq!(nvmlShutdown(), SUCCESS);
}

#[test]
fn free_memory_defaults_to_total_without_a_context() {
    let _s = Session::new(1, None);
    assert_eq!(nvmlInit_v2(), SUCCESS);
    let mut handle = null_mut();
    unsafe { nvmlDeviceGetHandleByIndex_v2(0, &mut handle) };
    let mut memory = Memory { total: 0, free: 0, used: 1 };
    assert_eq!(unsafe { nvmlDeviceGetMemoryInfo(handle, &mut memory) }, SUCCESS);
    assert_eq!((memory.free, memory.used), (16 << 30, 0));
}

#[test]
fn system_strings_and_processes() {
    let _s = Session::new(1, None);
    assert_eq!(nvmlInit_v2(), SUCCESS);
    let mut buffer = [0 as c_char; 80];
    assert_eq!(unsafe { nvmlSystemGetDriverVersion(buffer.as_mut_ptr(), 80) }, SUCCESS);
    assert_eq!(text(&buffer), "13.0");
    assert_eq!(unsafe { nvmlSystemGetNVMLVersion(buffer.as_mut_ptr(), 80) }, SUCCESS);
    assert_eq!(text(&buffer), NVML_VERSION);
    let mut cuda = 0;
    assert_eq!(unsafe { nvmlSystemGetCudaDriverVersion(&mut cuda) }, SUCCESS);
    assert_eq!(cuda, 13000);
    let mut handle = null_mut();
    unsafe { nvmlDeviceGetHandleByIndex_v2(0, &mut handle) };
    let mut count = 5;
    assert_eq!(unsafe { nvmlDeviceGetComputeRunningProcesses_v3(handle, &mut count, null_mut()) }, SUCCESS);
    assert_eq!(count, 0);
    assert_eq!(nvmlSystemGetProcessName(1, buffer.as_mut_ptr(), 80), NOT_FOUND);
}

#[test]
fn sensors_are_not_supported_not_crashes() {
    let _s = Session::new(1, None);
    assert_eq!(nvmlInit_v2(), SUCCESS);
    // Stubs ignore their arguments and report NOT_SUPPORTED.
    assert_eq!(stubs::nvmlDeviceGetTemperature(), NOT_SUPPORTED);
    assert_eq!(stubs::nvmlDeviceGetPowerUsage(), NOT_SUPPORTED);
    assert_eq!(stubs::nvmlDeviceGetFanSpeed(), NOT_SUPPORTED);
    assert_eq!(stubs::nvmlDeviceGetUtilizationRates(), NOT_SUPPORTED);
}

#[test]
fn error_strings() {
    assert_eq!(error_text(0).to_str().unwrap(), "Success");
    assert_eq!(error_text(3).to_str().unwrap(), "Not Supported");
    assert_eq!(error_text(12345).to_str().unwrap(), "Unknown Error");
    assert!(!nvmlErrorString(UNINITIALIZED).is_null());
}

#[test]
fn bus_id_and_uuid_formatting() {
    assert_eq!(parse_bus_id("0000:03:00.0"), Some((0, 3, 0)));
    assert_eq!(parse_bus_id("00ab:1f:1d.2"), Some((0xab, 0x1f, 0x1d)));
    assert_eq!(parse_bus_id("garbage"), None);
    assert_eq!(format_uuid([0; 16]), "GPU-00000000-0000-0000-0000-000000000000");
}
