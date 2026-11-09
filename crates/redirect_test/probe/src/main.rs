//! `cfi-redirect-probe <app_dir> <expected_dir|-> [--spawn-child]`
//!
//! Loads, from `app_dir`: `cfi_consumer.dll` (imports cublas64_12.dll by name), then
//! `cublas64_12.dll` by full path, then `cublas64_11.dll` by full path. With `expected_dir`, every
//! cuBLAS module must come from there (or its `aliases` copy), never from `app_dir`. With `-`
//! (control run without cfi-run), they must come from `app_dir`. `--spawn-child` reruns the same
//! check in a child process (std's CreateProcessW) and requires it to pass too. Exit 0 = pass.

#[cfg(not(windows))]
fn main() {
    eprintln!("cfi-redirect-probe: Windows only");
    std::process::exit(2);
}

#[cfg(windows)]
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: cfi-redirect-probe <app_dir> <expected_dir|-> [--spawn-child]");
        std::process::exit(2);
    }
    let code = match win::check(&args[0], &args[1]) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("FAIL (pid {}): {e}", std::process::id());
            1
        }
    };
    if code == 0 && args.get(2).map(String::as_str) == Some("--spawn-child") {
        let exe = std::env::current_exe().expect("current_exe");
        let status = std::process::Command::new(exe).args(&args[..2]).status().expect("spawn child");
        if !status.success() {
            eprintln!("FAIL: child probe exited with {status}");
            std::process::exit(1);
        }
        println!("PASS child");
    }
    if code == 0 {
        println!("PASS pid {}", std::process::id());
    }
    std::process::exit(code);
}

#[cfg(windows)]
mod win {
    use std::ffi::c_void;
    use std::os::windows::ffi::OsStrExt;
    use std::path::{Path, PathBuf};

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryW(name: *const u16) -> *mut c_void;
        fn LoadLibraryExW(name: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
        fn GetModuleHandleW(name: *const u16) -> *mut c_void;
        fn GetModuleFileNameW(module: *mut c_void, buf: *mut u16, n: u32) -> u32;
        fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
    }

    fn wide(p: &Path) -> Vec<u16> {
        p.as_os_str().encode_wide().chain([0]).collect()
    }

    fn path_of(module: *mut c_void) -> Result<PathBuf, String> {
        if module.is_null() {
            return Err(format!("load failed: {}", std::io::Error::last_os_error()));
        }
        let mut buf = vec![0u16; 32768];
        let n = unsafe { GetModuleFileNameW(module, buf.as_mut_ptr(), buf.len() as u32) } as usize;
        Ok(PathBuf::from(String::from_utf16_lossy(&buf[..n])))
    }

    fn norm(p: &Path) -> String {
        let s = p.to_string_lossy().to_ascii_lowercase().replace('/', "\\");
        s.strip_prefix(r"\\?\").unwrap_or(&s).trim_end_matches('\\').to_string()
    }

    pub fn check(app: &str, expected: &str) -> Result<(), String> {
        let app = std::path::absolute(app).map_err(|e| e.to_string())?;
        let redirected = expected != "-";
        let want = if redirected { std::path::absolute(expected).map_err(|e| e.to_string())? } else { app.clone() };
        let verify = |what: &str, got: &Path, alias: bool| -> Result<(), String> {
            println!("{what}: {}", got.display());
            let dir = norm(got.parent().unwrap_or(Path::new("")));
            let ok = if !redirected {
                dir == norm(&app)
            } else if alias {
                dir != norm(&app) && dir.contains("alias")
            } else {
                dir == norm(&want)
            };
            if ok { Ok(()) } else { Err(format!("{what} loaded from {}, expected {}", got.display(), if alias { "an alias dir" } else { "" }.to_owned() + &want.display().to_string())) }
        };

        // 1. Static import by name from a DLL in the app directory (torch_cuda.dll's case).
        let consumer = unsafe { LoadLibraryW(wide(&app.join("cfi_consumer.dll")).as_ptr()) };
        path_of(consumer).map_err(|e| format!("cfi_consumer.dll: {e}"))?;
        let by_name: Vec<u16> = std::ffi::OsStr::new("cublas64_12.dll").encode_wide().chain([0]).collect();
        verify("static import cublas64_12.dll", &path_of(unsafe { GetModuleHandleW(by_name.as_ptr()) })?, false)?;
        let call = unsafe { GetProcAddress(consumer, c"consumer_call".as_ptr().cast()) };
        if call.is_null() {
            return Err("consumer_call missing".into());
        }
        let call: extern "C" fn() -> u32 = unsafe { std::mem::transmute(call) };
        if call() != 42 {
            return Err("consumer_call returned the wrong value".into());
        }
        // 2. LoadLibraryW with a full path to the app directory's copy.
        let full = unsafe { LoadLibraryW(wide(&app.join("cublas64_12.dll")).as_ptr()) };
        verify("LoadLibraryW full path cublas64_12.dll", &path_of(full)?, false)?;
        // 3. Another version by full path: served by an alias copy of ours.
        let v11 = unsafe { LoadLibraryExW(wide(&app.join("cublas64_11.dll")).as_ptr(), std::ptr::null_mut(), 0x8) };
        verify("LoadLibraryExW full path cublas64_11.dll", &path_of(v11)?, true)?;
        Ok(())
    }
}
