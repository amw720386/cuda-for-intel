//! Source of device facts: the loaded CUDA driver, or a fake in tests.
use libloading::Library;
use std::ffi::{c_char, c_int, c_void};
use std::sync::{Arc, Mutex, OnceLock};

pub trait Backend: Send + Sync {
    /// `Err` carries an NVML error code (driver missing: 9).
    fn device_count(&self) -> Result<usize, u32>;
    fn name(&self, index: usize) -> Option<String>;
    fn total_memory(&self, index: usize) -> Option<u64>;
    /// `None` when no CUDA context is current (NVML must not create one).
    fn free_memory(&self, index: usize) -> Option<u64>;
    fn pci_bus_id(&self, index: usize) -> Option<String>;
    fn uuid(&self, index: usize) -> Option<[u8; 16]>;
    fn compute_capability(&self, index: usize) -> Option<(i32, i32)>;
    fn cuda_driver_version(&self) -> i32;
    fn driver_version(&self) -> String;
}

static OVERRIDE: Mutex<Option<Arc<dyn Backend>>> = Mutex::new(None);

pub fn backend() -> Arc<dyn Backend> {
    if let Ok(guard) = OVERRIDE.lock() {
        if let Some(fake) = guard.as_ref() {
            return fake.clone();
        }
    }
    static DRIVER: OnceLock<Arc<dyn Backend>> = OnceLock::new();
    DRIVER.get_or_init(|| Arc::new(DriverBackend::load())).clone()
}

/// Replace the backend (tests). `None` restores the driver backend.
pub fn set_backend(backend: Option<Arc<dyn Backend>>) {
    if let Ok(mut guard) = OVERRIDE.lock() {
        *guard = backend;
    }
}

type Status = c_int;

struct DriverBackend {
    lib: Option<Library>,
}

impl DriverBackend {
    fn load() -> Self {
        let name = if cfg!(windows) { "nvcuda.dll" } else { "libcuda.so.1" };
        let lib = unsafe { Library::new(name) }.ok();
        if let Some(lib) = &lib {
            if let Ok(init) = unsafe { lib.get::<unsafe extern "system" fn(u32) -> Status>(b"cuInit\0") } {
                if unsafe { init(0) } != 0 {
                    return Self { lib: None };
                }
            }
        }
        Self { lib }
    }

    fn sym<T: Copy>(&self, name: &[u8]) -> Option<T> {
        let lib = self.lib.as_ref()?;
        unsafe { lib.get::<T>(name) }.ok().map(|s| *s)
    }
}

impl Backend for DriverBackend {
    fn device_count(&self) -> Result<usize, u32> {
        let count = self.sym::<unsafe extern "system" fn(*mut c_int) -> Status>(b"cuDeviceGetCount\0").ok_or(9u32)?;
        let mut n = 0;
        match unsafe { count(&mut n) } {
            0 => Ok(n.max(0) as usize),
            _ => Err(9),
        }
    }

    fn name(&self, index: usize) -> Option<String> {
        let get = self.sym::<unsafe extern "system" fn(*mut c_char, c_int, c_int) -> Status>(b"cuDeviceGetName\0")?;
        let mut buffer = [0 as c_char; 256];
        (unsafe { get(buffer.as_mut_ptr(), 256, index as c_int) } == 0).then(|| {
            let bytes: Vec<u8> = buffer.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
            String::from_utf8_lossy(&bytes).into_owned()
        })
    }

    fn total_memory(&self, index: usize) -> Option<u64> {
        let get = self.sym::<unsafe extern "system" fn(*mut usize, c_int) -> Status>(b"cuDeviceTotalMem_v2\0")?;
        let mut total = 0;
        (unsafe { get(&mut total, index as c_int) } == 0).then_some(total as u64)
    }

    fn free_memory(&self, _index: usize) -> Option<u64> {
        let current = self.sym::<unsafe extern "system" fn(*mut *mut c_void) -> Status>(b"cuCtxGetCurrent\0")?;
        let mut ctx = std::ptr::null_mut();
        if unsafe { current(&mut ctx) } != 0 || ctx.is_null() {
            return None;
        }
        let info = self.sym::<unsafe extern "system" fn(*mut usize, *mut usize) -> Status>(b"cuMemGetInfo_v2\0")?;
        let (mut free, mut total) = (0, 0);
        (unsafe { info(&mut free, &mut total) } == 0).then_some(free as u64)
    }

    fn pci_bus_id(&self, index: usize) -> Option<String> {
        let get = self.sym::<unsafe extern "system" fn(*mut c_char, c_int, c_int) -> Status>(b"cuDeviceGetPCIBusId\0")?;
        let mut buffer = [0 as c_char; 64];
        (unsafe { get(buffer.as_mut_ptr(), 64, index as c_int) } == 0).then(|| {
            let bytes: Vec<u8> = buffer.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
            String::from_utf8_lossy(&bytes).into_owned()
        })
    }

    fn uuid(&self, index: usize) -> Option<[u8; 16]> {
        let get = self.sym::<unsafe extern "system" fn(*mut [u8; 16], c_int) -> Status>(b"cuDeviceGetUuid_v2\0")?;
        let mut bytes = [0u8; 16];
        (unsafe { get(&mut bytes, index as c_int) } == 0).then_some(bytes)
    }

    fn compute_capability(&self, index: usize) -> Option<(i32, i32)> {
        let get = self.sym::<unsafe extern "system" fn(*mut c_int, c_int, c_int) -> Status>(b"cuDeviceGetAttribute\0")?;
        let (mut major, mut minor) = (0, 0);
        let ok = unsafe { get(&mut major, 75, index as c_int) } == 0 && unsafe { get(&mut minor, 76, index as c_int) } == 0;
        ok.then_some((major, minor))
    }

    fn cuda_driver_version(&self) -> i32 {
        let Some(get) = self.sym::<unsafe extern "system" fn(*mut c_int) -> Status>(b"cuDriverGetVersion\0") else { return 0 };
        let mut version = 0;
        if unsafe { get(&mut version) } == 0 { version } else { 0 }
    }

    fn driver_version(&self) -> String {
        // The CUDA driver version stands in for the display driver version.
        let v = self.cuda_driver_version();
        format!("{}.{}", v / 1000, (v % 1000) / 10)
    }
}
