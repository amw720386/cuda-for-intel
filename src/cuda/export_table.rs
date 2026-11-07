//! Undocumented `cuGetExportTable` interfaces used by CUDA Runtime.
//!
//! Layout and GUID coverage follow the publicly documented ZLUDA dark-API map
//! (MIT). Only the slots required for cudart device enumeration / primary
//! context retain are wired initially; remaining slots stay null.
use super::primary::cuDevicePrimaryCtxRetain;
use super::*;
use std::ffi::CStr;
use std::sync::OnceLock;

/// CUDART_INTERFACE — `{6BD5FB6C-5BF4-E74A-8987-D93912FD9DF9}`
const CUDART_INTERFACE: [u8; 16] = [
    0x6B, 0xD5, 0xFB, 0x6C, 0x5B, 0xF4, 0xE7, 0x4A, 0x89, 0x87, 0xD9, 0x39, 0x12, 0xFD, 0x9D, 0xF9,
];

/// TOOLS_RUNTIME_CALLBACK_HOOKS — `{A094798C-2E74-2E74-93F2-0800200C0A66}`
const TOOLS_RUNTIME_CALLBACK_HOOKS: [u8; 16] = [
    0xA0, 0x94, 0x79, 0x8C, 0x2E, 0x74, 0x2E, 0x74, 0x93, 0xF2, 0x08, 0x00, 0x20, 0x0C, 0x0A, 0x66,
];

const TOOLS_TLS: [u8; 16] = [
    0x42, 0xD8, 0x5A, 0x81, 0x23, 0xF6, 0xCB, 0x47, 0x82, 0x98, 0xF6, 0xE7, 0x8A, 0x3A, 0xEC, 0xDC,
];

/// CONTEXT_LOCAL_STORAGE_INTERFACE_V0301 — `{C693336E-1121-DF11-A8C3-68F355D89593}`
const CONTEXT_LOCAL_STORAGE: [u8; 16] = [
    0xC6, 0x93, 0x33, 0x6E, 0x11, 0x21, 0xDF, 0x11, 0xA8, 0xC3, 0x68, 0xF3, 0x55, 0xD8, 0x95, 0x93,
];

const CONTEXT_CHECKS: [u8; 16] = [
    0x26, 0x3E, 0x88, 0x60, 0x7C, 0xD2, 0x61, 0x43, 0x92, 0xF6, 0xBB, 0xD5, 0x00, 0x6D, 0xFA, 0x7E,
];

const INTEGRITY_CHECK: [u8; 16] = [
    0xD4, 0x08, 0x20, 0x55, 0xBD, 0xE6, 0x70, 0x4B, 0x8D, 0x34, 0xBA, 0x12, 0x3C, 0x66, 0xE1, 0xF2,
];

/// HYBRID_CUDART — `{DD9E98A4-F410-4844-99B0-E2AEB7B432B5}`
///
/// Windows CUDA 13 static cudart requests this table, then calls slot 1 to
/// obtain a `__cudaGetProcAddress` resolver (ZLUDA dark-api map, MIT).
const HYBRID_CUDART: [u8; 16] = [
    0xDD, 0x9E, 0x98, 0xA4, 0xF4, 0x10, 0x48, 0x44, 0x99, 0xB0, 0xE2, 0xAE, 0xB7, 0xB4, 0x32, 0xB5,
];

/// Token returned by `hybrid_runtime_load_get_proc_address` (ZLUDA value).
const HYBRID_RUNTIME_TOKEN: usize = 302_100_128;

const PTR_SIZE: usize = size_of::<*const c_void>();

fn tables() -> &'static Tables {
    static TABLES: OnceLock<Tables> = OnceLock::new();
    TABLES.get_or_init(Tables::new)
}

struct Tables {
    cudart: [*const c_void; 13],
    tools_hooks: [*const c_void; 7],
    tools_tls: [*const c_void; 4],
    context_local: [*const c_void; 4],
    context_checks: [*const c_void; 4],
    integrity_check: [*const c_void; 3],
    hybrid_cudart: [*const c_void; 3],
}

// Function-pointer tables are immutable after init; raw pointers are Sync/Send by contract.
unsafe impl Sync for Tables {}
unsafe impl Send for Tables {}

impl Tables {
    fn new() -> Self {
        let mut cudart = [std::ptr::null(); 13];
        cudart[0] = (13 * PTR_SIZE) as *const c_void;
        // Slot 2: primary-context retain used by cudart device bring-up.
        // Point directly at the Driver API export — a thin wrapper can be
        // ICF-merged onto the same symbol and recurse forever.
        cudart[2] = cuDevicePrimaryCtxRetain as *const c_void;
        // Slot 7: cudart teardown hook — no-op for now.
        cudart[7] = cudart_interface_fn7 as *const c_void;
        // Slot 12: optional compiler load — report success without loading NVVM.
        cudart[12] = load_compilers as *const c_void;

        let mut tools_hooks = [std::ptr::null(); 7];
        tools_hooks[0] = (7 * PTR_SIZE) as *const c_void;
        // The counts returned by slots 2 and 6 are in u32 elements.
        tools_hooks[2] = tools_hook_fn2 as *const c_void;
        tools_hooks[6] = tools_hook_fn6 as *const c_void;

        let mut tools_tls = [std::ptr::null(); 4];
        tools_tls[0] = (4 * PTR_SIZE) as *const c_void;

        let mut context_local = [std::ptr::null(); 4];
        // Slot 0 size is not used the same way; provide put/get/delete stubs.
        context_local[0] = context_local_storage_put as *const c_void;
        context_local[1] = context_local_storage_delete as *const c_void;
        context_local[2] = context_local_storage_get as *const c_void;

        let mut context_checks = [std::ptr::null(); 4];
        context_checks[0] = (4 * PTR_SIZE) as *const c_void;
        context_checks[2] = context_check as *const c_void;
        context_checks[3] = context_check_mode as *const c_void;

        let mut integrity_check = [std::ptr::null(); 3];
        integrity_check[0] = (3 * PTR_SIZE) as *const c_void;
        integrity_check[1] = runtime_integrity_check as *const c_void;

        let mut hybrid_cudart = [std::ptr::null(); 3];
        hybrid_cudart[0] = (3 * PTR_SIZE) as *const c_void;
        hybrid_cudart[1] = hybrid_runtime_load_get_proc_address as *const c_void;
        hybrid_cudart[2] = hybrid_runtime_free as *const c_void;

        Self {
            cudart,
            tools_hooks,
            tools_tls,
            context_local,
            context_checks,
            integrity_check,
            hybrid_cudart,
        }
    }
}

unsafe extern "system" fn context_check_mode() -> u32 {
    // The driver has no special context checking mode enabled.
    0
}

unsafe extern "system" fn context_check(
    context: Opaque,
    mode: *mut u32,
    alternate: *mut Opaque,
) -> CuResult {
    if mode.is_null() || alternate.is_null() {
        return InvalidValue;
    }
    status(state(|state| {
        if !context.is_null() {
            state
                .contexts
                .get(&context.addr())
                .ok_or(InvalidContext)?
                .check_live()?;
        }
        // All contexts on this driver use ordinary synchronization.
        unsafe {
            mode.write(0);
            alternate.write(std::ptr::null_mut());
        }
        Ok(())
    }))
}

unsafe extern "system" fn runtime_integrity_check(
    version: u32,
    unix_seconds: u64,
    out: *mut [u64; 2],
) -> CuResult {
    if out.is_null() {
        return InvalidValue;
    }
    super::export_log(format!("runtime_integrity_check version={version}"));
    #[cfg(windows)]
    {
        let result = match version {
            13000 => [0x3341181C03CB675C, 0x8ED383AA1F4CD1E8],
            13001 => [0x1841181C03CB675C, 0x8ED383AA1F4CD1E8],
            13002 => {
                #[link(name = "kernel32")]
                unsafe extern "system" {
                    fn GetCurrentThreadId() -> u32;
                }
                let tables = tables();
                let mut inputs = [0; 48];
                inputs[..4].copy_from_slice(&discovery::DRIVER_ABI.to_le_bytes());
                inputs[4..8].copy_from_slice(&version.to_le_bytes());
                inputs[8..12].copy_from_slice(&std::process::id().to_le_bytes());
                inputs[12..16].copy_from_slice(&unsafe { GetCurrentThreadId() }.to_le_bytes());
                inputs[16..24].copy_from_slice(&(tables.cudart.as_ptr() as u64).to_le_bytes());
                inputs[24..32]
                    .copy_from_slice(&(tables.integrity_check.as_ptr() as u64).to_le_bytes());
                inputs[32..40].copy_from_slice(&(tables.integrity_check[1] as u64).to_le_bytes());
                inputs[40..].copy_from_slice(&unix_seconds.to_le_bytes());
                match runtime(|runtime| {
                    let devices = runtime
                        .devices
                        .iter()
                        .map(|device| {
                            let info = &device.info;
                            let [domain, bus, slot, _] = info.pci_address.ok_or(NotSupported)?;
                            let mut identity = [0; 28];
                            identity[..16].copy_from_slice(&info.uuid);
                            identity[16..20].copy_from_slice(&domain.to_le_bytes());
                            identity[20..24].copy_from_slice(&bus.to_le_bytes());
                            identity[24..].copy_from_slice(&slot.to_le_bytes());
                            Ok(identity)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(integrity::digest(&inputs, devices))
                }) {
                    Ok(digest) => digest,
                    Err(error) => return error,
                }
            }
            _ => return NotSupported,
        };
        unsafe { out.write(result) };
        Success
    }
    #[cfg(not(windows))]
    {
        let _ = unix_seconds;
        NotSupported
    }
}

unsafe extern "system" fn cudart_interface_fn7(_arg: usize) {}

unsafe extern "system" fn load_compilers() -> CuResult {
    Success
}

const TOOLS_HOOK_FN2_COUNT: usize = 1024;
const TOOLS_HOOK_FN6_COUNT: usize = 14;
static mut TOOLS_HOOK_FN2_SPACE: [u32; TOOLS_HOOK_FN2_COUNT] = [0; TOOLS_HOOK_FN2_COUNT];
static mut TOOLS_HOOK_FN6_SPACE: [u32; TOOLS_HOOK_FN6_COUNT] = [0; TOOLS_HOOK_FN6_COUNT];

unsafe extern "system" fn tools_hook_fn2(ptr: *mut *mut u32, count: *mut usize) {
    super::export_log("tools_hook_fn2");
    unsafe {
        if !ptr.is_null() {
            *ptr = std::ptr::addr_of_mut!(TOOLS_HOOK_FN2_SPACE).cast();
        }
        if !count.is_null() {
            *count = TOOLS_HOOK_FN2_COUNT;
        }
    }
}

unsafe extern "system" fn tools_hook_fn6(ptr: *mut *mut u32, count: *mut usize) {
    super::export_log("tools_hook_fn6");
    unsafe {
        let space = std::ptr::addr_of_mut!(TOOLS_HOOK_FN6_SPACE).cast();
        if !ptr.is_null() {
            *ptr = space;
        }
        if !count.is_null() {
            *count = TOOLS_HOOK_FN6_COUNT;
        }
    }
}

type ContextDtor = Option<unsafe extern "system" fn(Opaque, *mut c_void, *mut c_void)>;

unsafe extern "system" fn context_local_storage_put(
    context: Opaque,
    key: *mut c_void,
    value: *mut c_void,
    _dtor: ContextDtor,
) -> CuResult {
    super::export_log(format!(
        "context_local_storage_put ctx={context:?} key={key:?} value={value:?}"
    ));
    // Accept and ignore — enough for cudart init probes that park TLS state.
    Success
}

unsafe extern "system" fn context_local_storage_delete(
    context: Opaque,
    key: *mut c_void,
) -> CuResult {
    super::export_log(format!(
        "context_local_storage_delete ctx={context:?} key={key:?}"
    ));
    Success
}

unsafe extern "system" fn context_local_storage_get(
    value: *mut *mut c_void,
    context: Opaque,
    key: *mut c_void,
) -> CuResult {
    super::export_log(format!(
        "context_local_storage_get ctx={context:?} key={key:?}"
    ));
    if value.is_null() {
        return InvalidValue;
    }
    unsafe { *value = std::ptr::null_mut() };
    // Report success with a null value; NotFound aborts cudart bring-up.
    Success
}

#[cfg(windows)]
type HybridResolver = unsafe extern "system" fn(*const c_void) -> *const c_void;

#[cfg(windows)]
fn hybrid_resolver(name: &str) -> Result<HybridResolver, CuResult> {
    use libloading::os::windows::{
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LOAD_LIBRARY_SEARCH_SYSTEM32, Library,
    };
    use std::{cell::Cell, path::PathBuf};

    struct HybridRuntime {
        _library: Library,
        resolver: HybridResolver,
    }

    static HYBRID64: OnceLock<HybridRuntime> = OnceLock::new();
    static HYBRID64A: OnceLock<HybridRuntime> = OnceLock::new();
    thread_local! { static LOADING: Cell<bool> = const { Cell::new(false) }; }

    let slot = match name {
        "nvcudart_hybrid64.dll" => &HYBRID64,
        "nvcudart_hybrid64a.dll" => &HYBRID64A,
        _ => return Err(InvalidValue),
    };
    if let Some(runtime) = slot.get() {
        return Ok(runtime.resolver);
    }
    let directory = std::env::var_os("CUDA_FOR_INTEL_HYBRID_DIR").ok_or(FileNotFound)?;
    let path = PathBuf::from(directory).join(name);
    if !path.is_absolute() {
        return Err(InvalidValue);
    }

    // Load outside the OnceLock initializer. A DLL can call back into the
    // driver during process attach, when waiting on its own load would deadlock.
    LOADING.with(|loading| {
        if loading.replace(true) {
            return Err(OperatingSystem);
        }
        let result = (|| {
            let library = unsafe {
                Library::load_with_flags(
                    &path,
                    LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32,
                )
            }
            .map_err(|error| {
                super::export_log(format!("hybrid load {name}: {error}"));
                FileNotFound
            })?;
            let resolver = *unsafe { library.get::<HybridResolver>(b"__cudaGetProcAddress") }
                .map_err(|error| {
                    super::export_log(format!("hybrid resolver {name}: {error}"));
                    OperatingSystem
                })?;
            // Keep the module mapped for the lifetime of any returned pointer.
            let runtime = slot.get_or_init(|| HybridRuntime {
                _library: library,
                resolver,
            });
            Ok(runtime.resolver)
        })();
        loading.set(false);
        result
    })
}

unsafe extern "system" fn hybrid_runtime_load_get_proc_address(
    name: *const c_char,
    fn_ptr: *mut *const c_void,
    token: *mut usize,
) -> CuResult {
    if name.is_null() || fn_ptr.is_null() || token.is_null() {
        return InvalidValue;
    }
    unsafe {
        fn_ptr.write(std::ptr::null());
        token.write(0);
    }
    let Ok(name) = unsafe { CStr::from_ptr(name) }.to_str() else {
        return InvalidValue;
    };
    super::export_log(format!("hybrid_runtime_load_get_proc_address name={name}"));
    #[cfg(windows)]
    {
        let resolver = match hybrid_resolver(name) {
            Ok(resolver) => resolver,
            Err(error) => return error,
        };
        unsafe {
            fn_ptr.write(resolver as *const c_void);
            token.write(HYBRID_RUNTIME_TOKEN);
        }
        Success
    }
    #[cfg(not(windows))]
    {
        NotSupported
    }
}

unsafe extern "system" fn hybrid_runtime_free(token: usize) -> CuResult {
    super::export_log(format!("hybrid_runtime_free token={token}"));
    if token != HYBRID_RUNTIME_TOKEN {
        return InvalidHandle;
    }
    Success
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGetExportTable(
    table: *mut *const c_void,
    id: *const [u8; 16],
) -> CuResult {
    trace::call("cuGetExportTable", || {
        if table.is_null() || id.is_null() {
            return InvalidValue;
        }
        let guid = unsafe { *id };
        let name = if guid == CUDART_INTERFACE {
            "CUDART_INTERFACE"
        } else if guid == TOOLS_RUNTIME_CALLBACK_HOOKS {
            "TOOLS_RUNTIME_CALLBACK_HOOKS"
        } else if guid == TOOLS_TLS {
            "TOOLS_TLS"
        } else if guid == CONTEXT_LOCAL_STORAGE {
            "CONTEXT_LOCAL_STORAGE"
        } else if guid == CONTEXT_CHECKS {
            "CONTEXT_CHECKS"
        } else if guid == INTEGRITY_CHECK {
            "INTEGRITY_CHECK"
        } else if guid == HYBRID_CUDART {
            "HYBRID_CUDART"
        } else {
            "UNKNOWN"
        };
        let hex: String = guid
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<Vec<_>>()
            .join("");
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("cuda-for-intel-export.log")
        {
            use std::io::Write;
            let _ = writeln!(file, "cuGetExportTable {name} guid={hex}");
        }
        diagnostic(&format!("cuGetExportTable {name} guid={hex}"));
        let tables = tables();
        let ptr = if guid == CUDART_INTERFACE {
            tables.cudart.as_ptr().cast()
        } else if guid == TOOLS_RUNTIME_CALLBACK_HOOKS {
            tables.tools_hooks.as_ptr().cast()
        } else if guid == TOOLS_TLS {
            tables.tools_tls.as_ptr().cast()
        } else if guid == CONTEXT_LOCAL_STORAGE {
            tables.context_local.as_ptr().cast()
        } else if guid == CONTEXT_CHECKS {
            tables.context_checks.as_ptr().cast()
        } else if guid == INTEGRITY_CHECK {
            tables.integrity_check.as_ptr().cast()
        } else if guid == HYBRID_CUDART {
            tables.hybrid_cudart.as_ptr().cast()
        } else {
            unsafe { table.write(std::ptr::null()) };
            return NotFound;
        };
        unsafe { table.write(ptr) };
        Success
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tools_callbacks_return_u32_buffers_with_cuda13_counts() {
        type ToolsBuffer = unsafe extern "system" fn(*mut *mut u32, *mut usize);
        let mut table = std::ptr::null();
        assert_eq!(
            unsafe { cuGetExportTable(&mut table, &TOOLS_RUNTIME_CALLBACK_HOOKS) },
            Success
        );
        let slots = table.cast::<*const c_void>();
        for (index, expected) in [(2, 1024), (6, 14)] {
            let callback: ToolsBuffer = unsafe { std::mem::transmute(*slots.add(index)) };
            let mut buffer = std::ptr::null_mut();
            let mut count = 0;
            unsafe { callback(&mut buffer, &mut count) };
            assert_eq!(count, expected);
            assert!(!buffer.is_null() && buffer.is_aligned());
            assert_eq!(unsafe { buffer.add(count - 1).read() }, 0);
        }
    }

    #[cfg(windows)]
    #[test]
    fn invalid_hybrid_library_name_clears_outputs() {
        let mut resolver = std::ptr::dangling::<u8>().cast();
        let mut token = usize::MAX;
        let status = unsafe {
            hybrid_runtime_load_get_proc_address(
                c"cudart64_13.dll".as_ptr(),
                &mut resolver,
                &mut token,
            )
        };
        assert_eq!(status, InvalidValue);
        assert!(resolver.is_null());
        assert_eq!(token, 0);
    }
}
