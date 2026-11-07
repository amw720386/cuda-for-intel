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

/// `CUDA_FOR_INTEL_PROFILE=1`: per-API call counts and wall time, appended to the export log
/// as cumulative `PROF` lines about every two seconds.
pub(super) fn profile_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("CUDA_FOR_INTEL_PROFILE").as_deref() == Some(std::ffi::OsStr::new("1")))
}

type ProfileTable = std::collections::HashMap<&'static str, (u64, u64)>;

pub(super) fn profile_record(api: &'static str, nanos: u64) {
    static TABLE: std::sync::Mutex<Option<(ProfileTable, std::time::Instant)>> = std::sync::Mutex::new(None);
    let Ok(mut guard) = TABLE.lock() else { return };
    let (table, last) = guard.get_or_insert_with(|| (ProfileTable::new(), std::time::Instant::now()));
    if api == "cuLaunchKernel" {
        sampler::note_launch_thread();
    }
    let entry = table.entry(api).or_default();
    entry.0 += 1;
    entry.1 += nanos;
    if last.elapsed().as_secs() >= 2 {
        *last = std::time::Instant::now();
        let mut rows: Vec<_> = table.iter().collect();
        rows.sort();
        let mut text = String::from("PROF-BEGIN");
        text.push_str(&format!("\nPROF-CPU {}", process_cpu_ms()));
        for (name, (count, total)) in rows {
            text.push_str(&format!("\nPROF {name} {count} {}", total / 1000));
        }
        super::export_log_always(text);
    }
}

#[cfg(windows)]
fn process_cpu_ms() -> String {
    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    struct FileTime(u32, u32);
    unsafe extern "system" {
        fn GetCurrentProcess() -> isize;
        fn GetProcessTimes(process: isize, create: *mut FileTime, exit: *mut FileTime, kernel: *mut FileTime, user: *mut FileTime) -> i32;
    }
    let (mut c, mut e, mut k, mut u) = (FileTime::default(), FileTime::default(), FileTime::default(), FileTime::default());
    unsafe { GetProcessTimes(GetCurrentProcess(), &mut c, &mut e, &mut k, &mut u) };
    let ms = |t: FileTime| (((t.1 as u64) << 32) | t.0 as u64) / 10_000;
    format!("kernel_ms={} user_ms={} wall_ms={}", ms(k), ms(u), {
        static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        START.get_or_init(std::time::Instant::now).elapsed().as_millis()
    })
}

#[cfg(not(windows))]
fn process_cpu_ms() -> String {
    String::new()
}

pub(super) fn call(api: &'static str, action: impl FnOnce() -> CuResult) -> CuResult {
    let started = profile_enabled().then(std::time::Instant::now);
    let on_target = started.is_some() && sampler::is_target();
    if on_target {
        sampler::enter();
    }
    let result = action();
    if on_target {
        sampler::leave();
    }
    if let Some(started) = started {
        profile_record(api, started.elapsed().as_nanos() as u64);
    }
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

/// `CUDA_FOR_INTEL_PROFILE=1` also samples the thread that launches kernels (about every
/// millisecond) and reports where it spends its time outside this driver: the module and
/// offset of the first stack word that points into a module other than the system DLLs.
#[cfg(windows)]
pub(super) mod sampler {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU32, Ordering};

    static TARGET: AtomicU32 = AtomicU32::new(0);
    static DEPTH: AtomicU32 = AtomicU32::new(0);

    pub(in super::super) fn is_target() -> bool {
        let target = TARGET.load(Ordering::Relaxed);
        target != 0 && target == unsafe { GetCurrentThreadId() }
    }
    pub(in super::super) fn enter() {
        DEPTH.fetch_add(1, Ordering::Relaxed);
    }
    pub(in super::super) fn leave() {
        DEPTH.fetch_sub(1, Ordering::Relaxed);
    }

    unsafe extern "system" {
        fn GetCurrentThreadId() -> u32;
        fn GetCurrentProcess() -> isize;
        fn ReadProcessMemory(process: isize, address: *const u8, buffer: *mut u8, size: usize, read: *mut usize) -> i32;
        fn OpenThread(access: u32, inherit: i32, id: u32) -> isize;
        fn SuspendThread(thread: isize) -> u32;
        fn ResumeThread(thread: isize) -> u32;
        fn GetThreadContext(thread: isize, context: *mut u8) -> i32;
        fn CloseHandle(handle: isize) -> i32;
        fn GetModuleHandleExW(flags: u32, address: *const u16, module: *mut isize) -> i32;
        fn GetModuleFileNameW(module: isize, name: *mut u16, size: u32) -> u32;
    }

    pub(in super::super) fn note_launch_thread() {
        if TARGET.load(Ordering::Relaxed) == 0
            && TARGET.compare_exchange(0, unsafe { GetCurrentThreadId() }, Ordering::SeqCst, Ordering::SeqCst).is_ok()
        {
            std::thread::spawn(run);
        }
    }

    fn module_of(address: u64) -> Option<(String, u64)> {
        let mut module = 0isize;
        // FROM_ADDRESS | UNCHANGED_REFCOUNT
        if unsafe { GetModuleHandleExW(0x4 | 0x2, address as *const u16, &mut module) } == 0 {
            return None;
        }
        let mut name = [0u16; 260];
        let length = unsafe { GetModuleFileNameW(module, name.as_mut_ptr(), 260) } as usize;
        let name = String::from_utf16_lossy(&name[..length]);
        let base = name.rsplit('\\').next().unwrap_or("?").to_ascii_lowercase();
        Some((base, address - module as u64))
    }

    fn run() {
        #[repr(align(16))]
        struct Context([u8; 1232]);
        let mut context = Box::new(Context([0; 1232]));
        let thread = unsafe { OpenThread(0x2 | 0x8 | 0x40, 0, TARGET.load(Ordering::Relaxed)) }; // suspend|get context|query
        if thread == 0 {
            return;
        }
        let mut stack = [0u64; 512];
        let mut histogram: HashMap<String, u64> = HashMap::new();
        let mut samples = 0u64;
        let mut last = std::time::Instant::now();
        loop {
            std::thread::sleep(std::time::Duration::from_millis(1));
            if unsafe { SuspendThread(thread) } == u32::MAX {
                return;
            }
            context.0[0x30..0x34].copy_from_slice(&0x0010_000bu32.to_le_bytes()); // CONTEXT_CONTROL | CONTEXT_INTEGER
            let ok = unsafe { GetThreadContext(thread, context.0.as_mut_ptr()) } != 0;
            let rsp = u64::from_le_bytes(context.0[0x98..0xa0].try_into().unwrap());
            let rip = u64::from_le_bytes(context.0[0xf8..0x100].try_into().unwrap());
            // Copy the raw stack while the thread is suspended; allocate and look up modules only
            // after it resumes, because it may hold the heap or loader lock.
            stack.fill(0);
            if ok {
                let mut read = 0usize;
                for bytes in [4096usize, 1024, 256] {
                    if unsafe { ReadProcessMemory(GetCurrentProcess(), rsp as *const u8, stack.as_mut_ptr().cast(), bytes, &mut read) } != 0 {
                        break;
                    }
                }
            }
            let in_api = DEPTH.load(Ordering::Relaxed) > 0;
            unsafe { ResumeThread(thread) };
            let mut key = String::from(if in_api { "IN-API " } else { "OUTSIDE " });
            if ok {
                let top = module_of(rip).map(|(m, o)| format!("{m}+{:x}", o >> 4 << 4)).unwrap_or_else(|| "?".into());
                key = format!("rip={top}");
                for &word in stack.iter() {
                    if word < 0x10000 {
                        continue;
                    }
                    if let Some((module, offset)) = module_of(word)
                        && !["ntdll.dll", "kernelbase.dll", "kernel32.dll", "win32u.dll", "ucrtbase.dll", "msvcp140.dll"].contains(&module.as_str())
                    {
                        key.push_str(&format!(" from={module}+{:x}", offset >> 4 << 4));
                        break;
                    }
                }
            }
            *histogram.entry(key).or_default() += 1;
            samples += 1;
            if last.elapsed().as_secs() >= 4 {
                last = std::time::Instant::now();
                let mut rows: Vec<_> = histogram.iter().collect();
                rows.sort_by_key(|row| std::cmp::Reverse(*row.1));
                let mut text = format!("SAMPLE-BEGIN total={samples}");
                for (key, count) in rows.into_iter().take(14) {
                    text.push_str(&format!("\nSAMPLE {count} {key}"));
                }
                super::super::export_log_always(text);
                histogram.clear();
                samples = 0;
            }
        }
        #[allow(unreachable_code)]
        unsafe {
            CloseHandle(thread);
        }
    }
}

#[cfg(not(windows))]
pub(super) mod sampler {
    pub(in super::super) fn note_launch_thread() {}
    pub(in super::super) fn is_target() -> bool {
        false
    }
    pub(in super::super) fn enter() {}
    pub(in super::super) fn leave() {}
}
