//! `cfi-run [--install-dir DIR] [--redirect-dll PATH] [--verbose] -- PROGRAM [ARGS...]`
//!
//! Starts PROGRAM suspended, adds `cfi_redirect.dll` to its import table (Detours), hands it the
//! install directory as a payload, resumes it, waits, and exits with its exit code. The redirect
//! then maps NVIDIA CUDA library loads to the install directory in PROGRAM and in every process
//! PROGRAM starts. Nothing is copied into or changed in PROGRAM's directory.
//!
//! Install directory, first that applies: `--install-dir`, `CFI_REDIRECT_DIR`, the directory of
//! cfi-run.exe when it holds nvcuda.dll, `%LOCALAPPDATA%\cuda-for-intel\install\bin`
//! (the `scripts/cfi.ps1 install` default). The redirect DLL: `--redirect-dll`, next to
//! cfi-run.exe, then in the install directory.

#[path = "../../cfi_redirect/src/shared.rs"]
#[allow(dead_code)]
mod shared;

use std::path::PathBuf;

const USAGE: &str = "usage: cfi-run [--install-dir DIR] [--redirect-dll PATH] [--verbose] -- PROGRAM [ARGS...]";

#[cfg_attr(not(windows), allow(dead_code))]
struct Options {
    install_dir: Option<PathBuf>,
    redirect_dll: Option<PathBuf>,
    verbose: bool,
    program: Vec<String>,
}

fn parse(args: Vec<String>) -> Result<Options, String> {
    let mut o = Options { install_dir: None, redirect_dll: None, verbose: false, program: Vec::new() };
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--" => {
                o.program = it.collect();
                break;
            }
            "--install-dir" => o.install_dir = Some(it.next().ok_or("--install-dir needs a value")?.into()),
            "--redirect-dll" => o.redirect_dll = Some(it.next().ok_or("--redirect-dll needs a value")?.into()),
            "--verbose" | "-v" => o.verbose = true,
            "--help" | "-h" => return Err(USAGE.into()),
            _ if a.starts_with('-') => return Err(format!("unknown option {a}\n{USAGE}")),
            _ => {
                o.program = std::iter::once(a).chain(it).collect();
                break;
            }
        }
    }
    if o.program.is_empty() {
        return Err(USAGE.into());
    }
    Ok(o)
}

/// Quote one argument for the MSVC CRT / `CommandLineToArgvW` rules.
fn quote(arg: &str) -> String {
    if !arg.is_empty() && !arg.contains([' ', '\t', '\n', '\u{b}', '"']) {
        return arg.to_string();
    }
    let mut out = String::from('"');
    let mut slashes = 0usize;
    for c in arg.chars() {
        match c {
            '\\' => slashes += 1,
            '"' => {
                out.extend(std::iter::repeat_n('\\', slashes * 2 + 1));
                out.push('"');
                slashes = 0;
            }
            _ => {
                out.extend(std::iter::repeat_n('\\', slashes));
                out.push(c);
                slashes = 0;
            }
        }
    }
    out.extend(std::iter::repeat_n('\\', slashes * 2));
    out.push('"');
    out
}

fn command_line(program: &[String]) -> String {
    program.iter().map(|a| quote(a)).collect::<Vec<_>>().join(" ")
}

fn main() {
    let opts = match parse(std::env::args().skip(1).collect()) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    std::process::exit(run(opts));
}

#[cfg(not(windows))]
fn run(_: Options) -> i32 {
    eprintln!("cfi-run: Windows only");
    2
}

#[cfg(windows)]
fn run(opts: Options) -> i32 {
    match win::run(&opts) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("cfi-run: {e}");
            2
        }
    }
}

#[cfg(windows)]
mod win {
    use super::{Options, command_line, shared};
    use cfi_detours_sys as detours;
    use std::ffi::{OsStr, c_void};
    use std::os::windows::ffi::OsStrExt;
    use std::path::{Path, PathBuf};

    type Handle = *mut c_void;

    #[repr(C)]
    struct StartupInfoW {
        cb: u32,
        reserved: *mut u16,
        desktop: *mut u16,
        title: *mut u16,
        x: u32,
        y: u32,
        x_size: u32,
        y_size: u32,
        x_count_chars: u32,
        y_count_chars: u32,
        fill_attribute: u32,
        flags: u32,
        show_window: u16,
        reserved2_size: u16,
        reserved2: *mut u8,
        std_input: Handle,
        std_output: Handle,
        std_error: Handle,
    }

    #[repr(C)]
    struct ProcessInformation {
        process: Handle,
        thread: Handle,
        process_id: u32,
        thread_id: u32,
    }

    const CREATE_SUSPENDED: u32 = 0x4;
    const CREATE_UNICODE_ENVIRONMENT: u32 = 0x400;
    const INFINITE: u32 = 0xffff_ffff;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn CreateProcessW(
            app: *const u16, cmd: *mut u16, pa: *const c_void, ta: *const c_void, inherit: i32, flags: u32,
            env: *const c_void, cwd: *const u16, si: *const StartupInfoW, pi: *mut ProcessInformation,
        ) -> i32;
        fn ResumeThread(thread: Handle) -> u32;
        fn TerminateProcess(process: Handle, code: u32) -> i32;
        fn WaitForSingleObject(handle: Handle, ms: u32) -> u32;
        fn GetExitCodeProcess(process: Handle, code: *mut u32) -> i32;
        fn CloseHandle(handle: Handle) -> i32;
        fn SetConsoleCtrlHandler(handler: *const c_void, add: i32) -> i32;
    }

    fn wide(s: &OsStr) -> Vec<u16> {
        s.encode_wide().chain(std::iter::once(0)).collect()
    }

    fn install_dir(opts: &Options, exe_dir: &Path) -> Result<PathBuf, String> {
        let dir = opts
            .install_dir
            .clone()
            .or_else(|| std::env::var_os(shared::ENV_DIR).filter(|v| !v.is_empty()).map(PathBuf::from))
            .or_else(|| exe_dir.join("nvcuda.dll").is_file().then(|| exe_dir.to_path_buf()))
            .or_else(|| {
                std::env::var_os("LOCALAPPDATA").map(|l| PathBuf::from(l).join("cuda-for-intel").join("install").join("bin"))
            })
            .ok_or("no install directory: pass --install-dir")?;
        let dir = std::path::absolute(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        if !dir.is_dir() {
            return Err(format!("install directory {} does not exist (run scripts/cfi.ps1 install, or pass --install-dir)", dir.display()));
        }
        Ok(dir)
    }

    fn redirect_dll(opts: &Options, exe_dir: &Path, dir: &Path) -> Result<PathBuf, String> {
        let found = opts.redirect_dll.clone().or_else(|| {
            [exe_dir.join(shared::DLL_NAME), dir.join(shared::DLL_NAME)].into_iter().find(|p| p.is_file())
        });
        let dll = found.ok_or_else(|| format!("{} not found next to cfi-run.exe or in {}", shared::DLL_NAME, dir.display()))?;
        let dll = std::path::absolute(&dll).map_err(|e| e.to_string())?;
        if !dll.is_file() {
            return Err(format!("{} does not exist", dll.display()));
        }
        let s = dll.to_str().filter(|s| s.is_ascii()).ok_or_else(|| {
            format!("{}: Detours needs an ASCII path for the injected DLL; move it to an ASCII path", dll.display())
        })?;
        Ok(PathBuf::from(s))
    }

    /// The current environment plus CFI_REDIRECT_DIR and the install directory first on PATH (so
    /// an exe that imports nvcuda.dll by name finds ours when System32 has none).
    fn environment(dir: &Path) -> Vec<u16> {
        let mut vars: Vec<(String, String)> = std::env::vars_os()
            .map(|(k, v)| (k.to_string_lossy().into_owned(), v.to_string_lossy().into_owned()))
            .filter(|(k, _)| !k.eq_ignore_ascii_case(shared::ENV_DIR))
            .collect();
        let d = dir.to_string_lossy().into_owned();
        match vars.iter_mut().find(|(k, _)| k.eq_ignore_ascii_case("PATH")) {
            Some((_, v)) => *v = format!("{d};{v}"),
            None => vars.push(("PATH".into(), d.clone())),
        }
        vars.push((shared::ENV_DIR.into(), d));
        vars.sort_by_key(|(k, _)| k.to_ascii_uppercase());
        let mut block = Vec::new();
        for (k, v) in vars {
            block.extend(OsStr::new(&format!("{k}={v}")).encode_wide());
            block.push(0);
        }
        block.push(0);
        block
    }

    pub(super) fn run(opts: &Options) -> Result<i32, String> {
        let exe = std::env::current_exe().map_err(|e| e.to_string())?;
        let exe_dir = exe.parent().unwrap_or(Path::new("."));
        let dir = install_dir(opts, exe_dir)?;
        let dll = redirect_dll(opts, exe_dir, &dir)?;
        if opts.verbose {
            eprintln!("cfi-run: install dir {}, redirect {}", dir.display(), dll.display());
        }
        let mut cmd = wide(OsStr::new(&command_line(&opts.program)));
        let env = environment(&dir);
        let mut si: StartupInfoW = unsafe { std::mem::zeroed() };
        si.cb = std::mem::size_of::<StartupInfoW>() as u32;
        let mut pi = ProcessInformation { process: std::ptr::null_mut(), thread: std::ptr::null_mut(), process_id: 0, thread_id: 0 };
        let ok = unsafe {
            CreateProcessW(
                std::ptr::null(), cmd.as_mut_ptr(), std::ptr::null(), std::ptr::null(), 1,
                CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT, env.as_ptr().cast(), std::ptr::null(), &si, &mut pi,
            )
        };
        if ok == 0 {
            return Err(format!("cannot start {}: {}", opts.program[0], std::io::Error::last_os_error()));
        }
        let fail = |msg: String| {
            unsafe {
                TerminateProcess(pi.process, 1);
                CloseHandle(pi.thread);
                CloseHandle(pi.process);
            }
            Err(msg)
        };
        let ansi: Vec<u8> = dll.to_string_lossy().bytes().chain([0]).collect();
        let dlls = [ansi.as_ptr()];
        if unsafe { detours::DetourUpdateProcessWithDll(pi.process, dlls.as_ptr(), 1) } == 0 {
            return fail(format!(
                "injecting {} failed: {} (a 32-bit PROGRAM is not supported)",
                dll.display(),
                std::io::Error::last_os_error()
            ));
        }
        let payload: Vec<u16> = dir.as_os_str().encode_wide().collect();
        let (data1, data2, data3, data4) = shared::PAYLOAD_GUID;
        let guid = detours::Guid { data1, data2, data3, data4 };
        if unsafe { detours::DetourCopyPayloadToProcess(pi.process, &guid, payload.as_ptr().cast(), (payload.len() * 2) as u32) } == 0 {
            return fail(format!("copying the payload failed: {}", std::io::Error::last_os_error()));
        }
        // Ctrl+C reaches the child through the shared console; we only wait for it.
        unsafe { SetConsoleCtrlHandler(std::ptr::null(), 1) };
        if unsafe { ResumeThread(pi.thread) } == u32::MAX {
            return fail(format!("resuming {} failed", opts.program[0]));
        }
        let mut code = 1u32;
        unsafe {
            WaitForSingleObject(pi.process, INFINITE);
            GetExitCodeProcess(pi.process, &mut code);
            CloseHandle(pi.thread);
            CloseHandle(pi.process);
        }
        Ok(code as i32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quoting_follows_commandlinetoargvw() {
        assert_eq!(quote("plain"), "plain");
        assert_eq!(quote(""), "\"\"");
        assert_eq!(quote("a b"), "\"a b\"");
        assert_eq!(quote(r#"say "hi""#), r#""say \"hi\"""#);
        assert_eq!(quote(r"dir with space\"), r#""dir with space\\""#);
        assert_eq!(quote(r"C:\no\space"), r"C:\no\space");
    }

    #[test]
    fn options_split_at_double_dash() {
        let o = parse(["--install-dir", "D", "--", "python", "-c", "x"].map(String::from).to_vec()).unwrap();
        assert_eq!(o.install_dir.as_deref(), Some(std::path::Path::new("D")));
        assert_eq!(o.program, ["python", "-c", "x"]);
        let o = parse(["app.exe", "--flag"].map(String::from).to_vec()).unwrap();
        assert_eq!(o.program, ["app.exe", "--flag"]);
        assert!(parse(vec![]).is_err());
        assert!(parse(["--bogus"].map(String::from).to_vec()).is_err());
    }
}
