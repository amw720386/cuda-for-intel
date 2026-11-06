//! Undocumented `cuGetExportTable` interfaces used by CUDA Runtime.
//!
//! Layout and GUID coverage follow the publicly documented ZLUDA dark-API map
//! (MIT). Only the slots required for cudart device enumeration / primary
//! context retain are wired initially; remaining slots stay null.
use super::primary::cuDevicePrimaryCtxRetain;
use super::*;
use std::sync::OnceLock;

/// CUDART_INTERFACE — `{6BD5FB6C-5BF4-E74A-8987-D93912FD9DF9}`
const CUDART_INTERFACE: [u8; 16] = [
    0x6B, 0xD5, 0xFB, 0x6C, 0x5B, 0xF4, 0xE7, 0x4A, 0x89, 0x87, 0xD9, 0x39, 0x12, 0xFD, 0x9D, 0xF9,
];

/// TOOLS_RUNTIME_CALLBACK_HOOKS — `{A094798C-2E74-2E74-93F2-0800200C0A66}`
const TOOLS_RUNTIME_CALLBACK_HOOKS: [u8; 16] = [
    0xA0, 0x94, 0x79, 0x8C, 0x2E, 0x74, 0x2E, 0x74, 0x93, 0xF2, 0x08, 0x00, 0x20, 0x0C, 0x0A, 0x66,
];

/// CONTEXT_LOCAL_STORAGE_INTERFACE_V0301 — `{C693336E-1121-DF11-A8C3-68F355D89593}`
const CONTEXT_LOCAL_STORAGE: [u8; 16] = [
    0xC6, 0x93, 0x33, 0x6E, 0x11, 0x21, 0xDF, 0x11, 0xA8, 0xC3, 0x68, 0xF3, 0x55, 0xD8, 0x95, 0x93,
];

const PTR_SIZE: usize = size_of::<*const c_void>();

fn tables() -> &'static Tables {
    static TABLES: OnceLock<Tables> = OnceLock::new();
    TABLES.get_or_init(Tables::new)
}

struct Tables {
    cudart: [*const c_void; 13],
    tools_hooks: [*const c_void; 7],
    context_local: [*const c_void; 4],
    /// Oversized null stub for GUIDs we have not reverse-engineered yet.
    /// Slot 0 is the byte size of the table; remaining slots stay null so a
    /// probing cudart neither sees a missing export-table mechanism nor
    /// indexes off the end of a size-only vector.
    unknown_stub: [*const c_void; 16],
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
        // ZLUDA layout: slot 2 → 512×usize scratch; slot 6 → 2-byte scratch.
        tools_hooks[2] = tools_hook_fn2 as *const c_void;
        tools_hooks[6] = tools_hook_fn6 as *const c_void;

        let mut context_local = [std::ptr::null(); 4];
        // Slot 0 size is not used the same way; provide put/get/delete stubs.
        context_local[0] = context_local_storage_put as *const c_void;
        context_local[1] = context_local_storage_delete as *const c_void;
        context_local[2] = context_local_storage_get as *const c_void;

        let mut unknown_stub = [std::ptr::null(); 16];
        unknown_stub[0] = (16 * PTR_SIZE) as *const c_void;

        Self {
            cudart,
            tools_hooks,
            context_local,
            unknown_stub,
        }
    }
}

unsafe extern "system" fn cudart_interface_fn7(_arg: usize) {}

unsafe extern "system" fn load_compilers() -> CuResult {
    Success
}

/// Scratch space for TOOLS_RUNTIME_CALLBACK_HOOKS slot 2 (ZLUDA: 512 usizes).
static mut TOOLS_HOOK_FN2_SPACE: [usize; 512] = [0; 512];
/// Scratch space for TOOLS_RUNTIME_CALLBACK_HOOKS slot 6 (ZLUDA: 2 bytes).
static mut TOOLS_HOOK_FN6_SPACE: [u8; 2] = [0; 2];

unsafe extern "system" fn tools_hook_fn2(ptr: *mut *mut usize, size: *mut usize) {
    super::export_log("tools_hook_fn2");
    unsafe {
        if !ptr.is_null() {
            *ptr = std::ptr::addr_of_mut!(TOOLS_HOOK_FN2_SPACE).cast();
        }
        if !size.is_null() {
            *size = 512;
        }
    }
}

unsafe extern "system" fn tools_hook_fn6(ptr: *mut *mut u8, size: *mut usize) -> *mut u8 {
    super::export_log("tools_hook_fn6");
    unsafe {
        let space = std::ptr::addr_of_mut!(TOOLS_HOOK_FN6_SPACE).cast();
        if !ptr.is_null() {
            *ptr = space;
        }
        if !size.is_null() {
            *size = 2;
        }
        space
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
        } else if guid == CONTEXT_LOCAL_STORAGE {
            "CONTEXT_LOCAL_STORAGE"
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
        } else if guid == CONTEXT_LOCAL_STORAGE {
            tables.context_local.as_ptr().cast()
        } else {
            // Present a sized null stub. Returning NotFound here made cudart 12
            // report cudaErrorInsufficientDriver (35); a 1-slot stub let it
            // index off the end into cudaErrorInitializationError (3).
            unsafe { table.write(tables.unknown_stub.as_ptr().cast()) };
            return Success;
        };
        unsafe { table.write(ptr) };
        Success
    })
}
