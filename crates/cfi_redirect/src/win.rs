//! Hooks and `DllMain` (Windows only). Uses the vendored Microsoft Detours for the inline hooks
//! and for adding this DLL to a suspended child's import table.

use crate::{map, pe, shared};
use cfi_detours_sys as detours;
use std::cell::Cell;
use std::collections::HashSet;
use std::ffi::c_void;
use std::os::windows::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

type Handle = *mut c_void;
type Bool = i32;

const CREATE_SUSPENDED: u32 = 0x4;
const LOAD_WITH_ALTERED_SEARCH_PATH: u32 = 0x8;
const DLL_PROCESS_ATTACH: u32 = 1;

#[repr(C)]
struct UnicodeString {
    length: u16,
    maximum_length: u16,
    buffer: *mut u16,
}

#[repr(C)]
struct ProcessInformation {
    process: Handle,
    thread: Handle,
    process_id: u32,
    thread_id: u32,
}

type LdrLoadDll = unsafe extern "system" fn(*const u16, *mut u32, *const UnicodeString, *mut Handle) -> i32;
type CreateProcessW = unsafe extern "system" fn(
    *const u16, *mut u16, *const c_void, *const c_void, Bool, u32, *const c_void, *const u16, *const c_void,
    *mut ProcessInformation,
) -> Bool;
type CreateProcessA = unsafe extern "system" fn(
    *const u8, *mut u8, *const c_void, *const c_void, Bool, u32, *const c_void, *const u8, *const c_void,
    *mut ProcessInformation,
) -> Bool;

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetModuleHandleW(name: *const u16) -> Handle;
    fn GetProcAddress(module: Handle, name: *const u8) -> *mut c_void;
    fn GetModuleFileNameW(module: Handle, buffer: *mut u16, size: u32) -> u32;
    fn GetShortPathNameW(long: *const u16, short: *mut u16, size: u32) -> u32;
    fn GetCurrentThread() -> Handle;
    fn ResumeThread(thread: Handle) -> u32;
    fn TerminateProcess(process: Handle, code: u32) -> Bool;
    fn CloseHandle(handle: Handle) -> Bool;
    fn LoadLibraryExW(name: *const u16, file: Handle, flags: u32) -> Handle;
    fn DisableThreadLibraryCalls(module: Handle) -> Bool;
}

// Originals; Detours rewrites each to point at its trampoline.
static mut REAL_LDR_LOAD_DLL: *mut c_void = std::ptr::null_mut();
static mut REAL_CREATE_PROCESS_W: *mut c_void = std::ptr::null_mut();
static mut REAL_CREATE_PROCESS_A: *mut c_void = std::ptr::null_mut();

struct State {
    dir: PathBuf,
    /// This DLL's path in the ANSI form `DetourUpdateProcessWithDll` takes, NUL-terminated.
    self_ansi: Option<Vec<u8>>,
    log: bool,
    system_root: String,
}

static STATE: OnceLock<State> = OnceLock::new();
static SCANNED: Mutex<Option<HashSet<String>>> = Mutex::new(None);
static NVCUDA_LOADED: AtomicBool = AtomicBool::new(false);

thread_local! {
    static SCANNING: Cell<bool> = const { Cell::new(false) };
}

fn wide(s: &std::ffi::OsStr) -> Vec<u16> {
    s.encode_wide().chain(std::iter::once(0)).collect()
}

fn log(state: &State, message: std::fmt::Arguments) {
    if state.log {
        eprintln!("cfi_redirect: {message}");
    }
}

impl State {
    fn has(&self, name: &str) -> bool {
        self.dir.join(name).is_file()
    }

    /// The file to load instead of `requested`, or `None` to load what was asked for.
    fn target_for(&self, requested: &str) -> Option<PathBuf> {
        let (file, alias) = map::resolve(requested, |c| self.has(c))?;
        let source = self.dir.join(&file);
        if !alias {
            return Some(source);
        }
        self.alias(&source, map::base_name(requested))
    }

    /// A copy of `source` named `name`, kept under `<install>\aliases` (or the temp directory when
    /// the install directory is read-only). Refreshed when `source` changes size or gets newer.
    fn alias(&self, source: &Path, name: &str) -> Option<PathBuf> {
        let src_meta = std::fs::metadata(source).ok()?;
        let fresh = |p: &Path| {
            std::fs::metadata(p).is_ok_and(|m| {
                m.len() == src_meta.len()
                    && matches!((m.modified(), src_meta.modified()), (Ok(a), Ok(b)) if a >= b)
            })
        };
        let mut dirs = vec![self.dir.join("aliases")];
        let tag: u32 = self.dir.to_string_lossy().bytes().fold(0x811c_9dc5, |h, b| (h ^ u32::from(b)).wrapping_mul(0x0100_0193));
        dirs.push(std::env::temp_dir().join("cfi-redirect-aliases").join(format!("{tag:08x}")));
        for dir in dirs {
            let dest = dir.join(name);
            if fresh(&dest) {
                return Some(dest);
            }
            if std::fs::create_dir_all(&dir).is_err() {
                continue;
            }
            let tmp = dir.join(format!("{name}.{}.tmp", std::process::id()));
            if std::fs::copy(source, &tmp).is_ok() {
                // A copy that another process has loaded cannot be replaced; reuse it if fresh.
                if std::fs::rename(&tmp, &dest).is_ok() || fresh(&dest) {
                    let _ = std::fs::remove_file(&tmp);
                    return Some(dest);
                }
                let _ = std::fs::remove_file(&tmp);
            }
        }
        None
    }

    /// Our nvcuda.dll first: the stand-ins import it by name, and the loader would otherwise
    /// search the exe directory, System32 and PATH for it.
    fn ensure_nvcuda(&self) {
        if NVCUDA_LOADED.load(Ordering::Relaxed) || !self.has("nvcuda.dll") {
            return;
        }
        NVCUDA_LOADED.store(true, Ordering::Relaxed);
        let path = wide(self.dir.join("nvcuda.dll").as_os_str());
        if unsafe { LoadLibraryExW(path.as_ptr(), std::ptr::null_mut(), LOAD_WITH_ALTERED_SEARCH_PATH) }.is_null() {
            log(self, format_args!("preloading our nvcuda.dll failed"));
        }
    }

    /// Load our stand-ins for every redirected library that `requested` (and its same-directory
    /// dependencies, three levels deep) imports by name.
    fn preload_imports(&self, requested: &str) {
        if !requested.contains(['\\', '/']) {
            return;
        }
        let lower = requested.to_ascii_lowercase();
        if !self.system_root.is_empty() && lower.starts_with(&self.system_root) {
            return;
        }
        let mut wanted = Vec::new();
        {
            let mut seen = SCANNED.lock().unwrap_or_else(|e| e.into_inner());
            let seen = seen.get_or_insert_with(HashSet::new);
            let mut todo = vec![(PathBuf::from(requested), 0u32)];
            while let Some((path, depth)) = todo.pop() {
                if !seen.insert(path.to_string_lossy().to_ascii_lowercase()) {
                    continue;
                }
                let Ok(mut file) = std::fs::File::open(&path) else { continue };
                let Some(names) = pe::imports(&mut file) else { continue };
                for name in names {
                    if map::candidates(&name).is_some() {
                        if let Some(target) = self.target_for(&name) {
                            wanted.push((name, target));
                        }
                    } else if depth < 3 {
                        let sibling = path.with_file_name(&name);
                        if sibling.is_file() {
                            todo.push((sibling, depth + 1));
                        }
                    }
                }
            }
        }
        for (name, target) in wanted {
            if !name.eq_ignore_ascii_case("nvcuda.dll") {
                self.ensure_nvcuda();
            }
            let path = wide(target.as_os_str());
            let ok = !unsafe { LoadLibraryExW(path.as_ptr(), std::ptr::null_mut(), LOAD_WITH_ALTERED_SEARCH_PATH) }.is_null();
            log(self, format_args!("preload {name} for {requested} -> {} ({})", target.display(), if ok { "ok" } else { "failed" }));
        }
    }

    unsafe fn inject(&self, process: Handle) -> bool {
        let Some(dll) = &self.self_ansi else { return false };
        let dlls = [dll.as_ptr()];
        if unsafe { detours::DetourUpdateProcessWithDll(process, dlls.as_ptr(), 1) } == 0 {
            log(self, format_args!("child not injected (DetourUpdateProcessWithDll failed)"));
            return false;
        }
        let payload: Vec<u16> = self.dir.as_os_str().encode_wide().collect();
        let guid = guid();
        unsafe {
            detours::DetourCopyPayloadToProcess(process, &guid, payload.as_ptr().cast(), (payload.len() * 2) as u32) != 0
        }
    }
}

fn guid() -> detours::Guid {
    let (data1, data2, data3, data4) = shared::PAYLOAD_GUID;
    detours::Guid { data1, data2, data3, data4 }
}

unsafe extern "system" fn hook_ldr_load_dll(
    search: *const u16,
    characteristics: *mut u32,
    name: *const UnicodeString,
    handle: *mut Handle,
) -> i32 {
    let real: LdrLoadDll = unsafe { std::mem::transmute(REAL_LDR_LOAD_DLL) };
    let (Some(state), Some(us)) = (STATE.get(), unsafe { name.as_ref() }) else {
        return unsafe { real(search, characteristics, name, handle) };
    };
    if us.buffer.is_null() || us.length == 0 {
        return unsafe { real(search, characteristics, name, handle) };
    }
    let requested = String::from_utf16_lossy(unsafe { std::slice::from_raw_parts(us.buffer, usize::from(us.length / 2)) });
    if let Some(target) = state.target_for(&requested) {
        if target.as_os_str().to_string_lossy().eq_ignore_ascii_case(&requested) {
            return unsafe { real(search, characteristics, name, handle) };
        }
        if !map::base_name(&requested).eq_ignore_ascii_case("nvcuda.dll") {
            state.ensure_nvcuda();
        }
        log(state, format_args!("{requested} -> {}", target.display()));
        let buffer: Vec<u16> = target.as_os_str().encode_wide().collect();
        let bytes = (buffer.len() * 2) as u16;
        let rewritten = UnicodeString { length: bytes, maximum_length: bytes, buffer: buffer.as_ptr().cast_mut() };
        return unsafe { real(search, characteristics, &rewritten, handle) };
    }
    if !SCANNING.get() {
        SCANNING.set(true);
        state.preload_imports(&requested);
        SCANNING.set(false);
    }
    unsafe { real(search, characteristics, name, handle) }
}

/// Create suspended, add this DLL and the payload, then resume unless the caller asked for a
/// suspended process.
unsafe fn spawn_injected(flags: u32, info: *mut ProcessInformation, create: impl FnOnce(u32, *mut ProcessInformation) -> Bool) -> Bool {
    let Some(state) = STATE.get() else { return create(flags, info) };
    let mut local = ProcessInformation { process: std::ptr::null_mut(), thread: std::ptr::null_mut(), process_id: 0, thread_id: 0 };
    let target = if info.is_null() { &raw mut local } else { info };
    if create(flags | CREATE_SUSPENDED, target) == 0 {
        return 0;
    }
    let pi = unsafe { &*target };
    let injected = unsafe { state.inject(pi.process) };
    log(state, format_args!("child pid {} injected: {injected}", pi.process_id));
    let mut ok = 1;
    if flags & CREATE_SUSPENDED == 0 && unsafe { ResumeThread(pi.thread) } == u32::MAX {
        unsafe { TerminateProcess(pi.process, 1) };
        ok = 0;
    }
    if info.is_null() {
        unsafe {
            CloseHandle(pi.process);
            CloseHandle(pi.thread);
        }
    }
    ok
}

#[allow(clippy::too_many_arguments)]
unsafe extern "system" fn hook_create_process_w(
    app: *const u16, cmd: *mut u16, pa: *const c_void, ta: *const c_void, inherit: Bool, flags: u32,
    env: *const c_void, cwd: *const u16, si: *const c_void, info: *mut ProcessInformation,
) -> Bool {
    let real: CreateProcessW = unsafe { std::mem::transmute(REAL_CREATE_PROCESS_W) };
    unsafe { spawn_injected(flags, info, |f, i| real(app, cmd, pa, ta, inherit, f, env, cwd, si, i)) }
}

#[allow(clippy::too_many_arguments)]
unsafe extern "system" fn hook_create_process_a(
    app: *const u8, cmd: *mut u8, pa: *const c_void, ta: *const c_void, inherit: Bool, flags: u32,
    env: *const c_void, cwd: *const u8, si: *const c_void, info: *mut ProcessInformation,
) -> Bool {
    let real: CreateProcessA = unsafe { std::mem::transmute(REAL_CREATE_PROCESS_A) };
    unsafe { spawn_injected(flags, info, |f, i| real(app, cmd, pa, ta, inherit, f, env, cwd, si, i)) }
}

fn module_path(module: Handle) -> Option<Vec<u16>> {
    let mut buf = vec![0u16; 32768];
    let n = unsafe { GetModuleFileNameW(module, buf.as_mut_ptr(), buf.len() as u32) } as usize;
    (n > 0 && n < buf.len()).then(|| buf[..n].to_vec())
}

/// ANSI (here: ASCII) path for Detours; falls back to the 8.3 short name for non-ASCII paths.
fn ansi_path(path: &[u16]) -> Option<Vec<u8>> {
    let ascii = |p: &[u16]| p.iter().all(|&c| c < 0x80).then(|| p.iter().map(|&c| c as u8).chain([0]).collect());
    ascii(path).or_else(|| {
        let long: Vec<u16> = path.iter().copied().chain([0]).collect();
        let mut short = vec![0u16; 32768];
        let n = unsafe { GetShortPathNameW(long.as_ptr(), short.as_mut_ptr(), short.len() as u32) } as usize;
        (n > 0 && n < short.len()).then(|| ascii(&short[..n])).flatten()
    })
}

fn install_dir() -> Option<PathBuf> {
    let mut size = 0u32;
    let guid = guid();
    let payload = unsafe { detours::DetourFindPayloadEx(&guid, &mut size) };
    if !payload.is_null() && size >= 2 {
        let units = unsafe { std::slice::from_raw_parts(payload.cast::<u16>(), size as usize / 2) };
        return Some(PathBuf::from(std::ffi::OsString::from_wide(units)));
    }
    std::env::var_os(shared::ENV_DIR).filter(|d| !d.is_empty()).map(PathBuf::from)
}

unsafe fn proc_address(module: &str, name: &std::ffi::CStr) -> *mut c_void {
    let m = unsafe { GetModuleHandleW(wide(module.as_ref()).as_ptr()) };
    if m.is_null() { std::ptr::null_mut() } else { unsafe { GetProcAddress(m, name.as_ptr().cast()) } }
}

unsafe fn attach_hooks() -> bool {
    unsafe {
        REAL_LDR_LOAD_DLL = proc_address("ntdll.dll", c"LdrLoadDll");
        for (slot, name) in [(&raw mut REAL_CREATE_PROCESS_W, c"CreateProcessW"), (&raw mut REAL_CREATE_PROCESS_A, c"CreateProcessA")] {
            let mut p = proc_address("kernelbase.dll", name);
            if p.is_null() {
                p = proc_address("kernel32.dll", name);
            }
            *slot = p;
        }
        if REAL_LDR_LOAD_DLL.is_null() || detours::DetourTransactionBegin() != 0 {
            return false;
        }
        detours::DetourUpdateThread(GetCurrentThread());
        let hooks: [(*mut *mut c_void, *mut c_void); 3] = [
            (&raw mut REAL_LDR_LOAD_DLL, hook_ldr_load_dll as *mut c_void),
            (&raw mut REAL_CREATE_PROCESS_W, hook_create_process_w as *mut c_void),
            (&raw mut REAL_CREATE_PROCESS_A, hook_create_process_a as *mut c_void),
        ];
        for (slot, hook) in hooks {
            if !(*slot).is_null() && detours::DetourAttach(slot, hook) != 0 {
                detours::DetourTransactionAbort();
                return false;
            }
        }
        detours::DetourTransactionCommit() == 0
    }
}

/// Exported so the DLL has an ordinal-1 export, which Detours' import-table injection requires.
#[unsafe(no_mangle)]
pub extern "system" fn cfi_redirect_active() -> i32 {
    STATE.get().is_some() as i32
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "system" fn DllMain(module: Handle, reason: u32, _reserved: *mut c_void) -> Bool {
    if reason != DLL_PROCESS_ATTACH {
        return 1;
    }
    unsafe {
        detours::DetourRestoreAfterWith();
        DisableThreadLibraryCalls(module);
    }
    let Some(dir) = install_dir() else { return 1 };
    let log = std::env::var_os(shared::ENV_LOG).is_some_and(|v| v == "1");
    let system_root = std::env::var("SystemRoot").map(|s| format!("{}\\", s.to_ascii_lowercase())).unwrap_or_default();
    let self_ansi = module_path(module).and_then(|p| ansi_path(&p));
    let state = State { dir, self_ansi, log, system_root };
    if state.log {
        eprintln!("cfi_redirect: active in pid {}, install dir {}", std::process::id(), state.dir.display());
    }
    let _ = STATE.set(state);
    if !unsafe { attach_hooks() } {
        eprintln!("cfi_redirect: installing the load hooks failed; CUDA libraries load unredirected");
    }
    1
}
