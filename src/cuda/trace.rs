use super::CuResult;
use std::fmt;
use std::io::Write;
use std::sync::OnceLock;

pub(super) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var_os("CUDA_FOR_INTEL_TRACE").is_some_and(|value| value == "1"))
}

pub(super) fn record(message: fmt::Arguments<'_>) {
    let _ = writeln!(std::io::stderr().lock(), "CUDA for Intel trace {message}");
}

pub(super) fn call(api: &'static str, action: impl FnOnce() -> CuResult) -> CuResult {
    static ENTER: OnceLock<bool> = OnceLock::new();
    if enabled() && *ENTER.get_or_init(|| std::env::var_os("CUDA_FOR_INTEL_TRACE_ENTER").is_some()) {
        record(format_args!("{api} enter"));
    }
    let result = action();
    if enabled() {
        record(format_args!("{api} -> {}", result as i32));
    }
    result
}

pub(super) fn symbol(value: &[u8]) -> &str {
    if value.starts_with(b"cu")
        && value.len() <= 128
        && value
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
    {
        std::str::from_utf8(value).unwrap()
    } else {
        "<redacted>"
    }
}

/// Under CUDA_FOR_INTEL_TRACE, log access violations with the faulting module
/// and offset (lets a null call from a stock library be located without a debugger).
#[cfg(windows)]
pub(super) fn install_crash_logger() {
    use std::ffi::c_void;
    #[repr(C)]
    struct ExceptionRecord {
        code: u32,
        flags: u32,
        next: *mut ExceptionRecord,
        address: *mut c_void,
        n: u32,
        info: [usize; 15],
    }
    #[repr(C)]
    struct ExceptionPointers {
        record: *mut ExceptionRecord,
        context: *mut u8,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn AddVectoredExceptionHandler(
            first: u32,
            handler: unsafe extern "system" fn(*mut ExceptionPointers) -> i32,
        ) -> *mut c_void;
        fn GetModuleHandleExA(flags: u32, addr: *const c_void, module: *mut *mut c_void) -> i32;
        fn GetModuleFileNameA(module: *mut c_void, buf: *mut u8, size: u32) -> u32;
    }
    unsafe extern "system" fn handler(p: *mut ExceptionPointers) -> i32 {
        unsafe {
            let r = &*(*p).record;
            if r.code == 0xC000_0005 {
                let mut module = std::ptr::null_mut();
                // FROM_ADDRESS | UNCHANGED_REFCOUNT
                GetModuleHandleExA(4 | 2, r.address, &mut module);
                let mut name = [0u8; 260];
                let n = GetModuleFileNameA(module, name.as_mut_ptr(), 260) as usize;
                let name = String::from_utf8_lossy(&name[..n]);
                let name = name.rsplit('\\').next().unwrap_or("?");
                // Context RIP-relative return address is at [rsp]; Rsp is context+0x98.
                let rsp = *((*p).context.add(0x98) as *const usize);
                let ret = *(rsp as *const usize);
                let mut rmod = std::ptr::null_mut();
                GetModuleHandleExA(4 | 2, ret as *const c_void, &mut rmod);
                let mut rname = [0u8; 260];
                let rn = GetModuleFileNameA(rmod, rname.as_mut_ptr(), 260) as usize;
                let rname = String::from_utf8_lossy(&rname[..rn]);
                let rname = rname.rsplit('\\').next().unwrap_or("?");
                record(format_args!(
                    "ACCESS VIOLATION at {name}+{:#x} (addr {:#x}, access {} {:#x}); [rsp] returns into {rname}+{:#x}",
                    r.address as usize - module as usize,
                    r.address as usize,
                    r.info[0],
                    r.info[1],
                    ret.wrapping_sub(rmod as usize)
                ));
            }
        }
        0
    }
    unsafe { AddVectoredExceptionHandler(1, handler) };
}

#[cfg(not(windows))]
pub(super) fn install_crash_logger() {}
