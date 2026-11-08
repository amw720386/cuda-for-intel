//! Locating a clang with the NVPTX backend. Pure lookup logic (environment and probing are
//! injected) so it can be unit-tested without a GPU or an LLVM install.
//!
//! Lookup order, first usable wins (see `docs/nvrtc.md`):
//! 1. `NVRTC_SHIM_CLANG`: a clang binary, or a directory holding one (`<dir>/clang`, `<dir>/bin/clang`).
//!    Explicit: if it is set but unusable the lookup fails instead of falling through.
//! 2. `NVRTC_SHIM_CLANG_DIR`: an LLVM directory (`<dir>/clang`, `<dir>/bin/clang`). Explicit as above.
//! 3. A `clang/` directory beside the shim library and beside the host executable
//!    (`<d>/clang/clang`, `<d>/clang/bin/clang`).
//! 4. `clang` / `clang-NN` on `PATH`.
//! 5. Standard install locations.
use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// Everything the lookup reads from the process, so tests can fake it.
#[derive(Debug, Clone, Default)]
pub struct Env {
    pub clang: Option<OsString>,
    pub clang_dir: Option<OsString>,
    pub path: Option<OsString>,
    /// Directory containing the shim library (DLL / .so).
    pub module_dir: Option<PathBuf>,
    /// Directory containing the host executable.
    pub exe_dir: Option<PathBuf>,
}

impl Env {
    pub fn from_process() -> Self {
        Self {
            clang: std::env::var_os("NVRTC_SHIM_CLANG").filter(|v| !v.is_empty()),
            clang_dir: std::env::var_os("NVRTC_SHIM_CLANG_DIR").filter(|v| !v.is_empty()),
            path: std::env::var_os("PATH"),
            module_dir: module_dir(),
            exe_dir: std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)),
        }
    }
}

/// Result of probing one candidate binary.
#[derive(Debug, Clone, PartialEq)]
pub enum Probe {
    Missing,
    /// Exists but `--print-targets` could not be run or failed.
    NotRunnable(String),
    /// Runs, but `--print-targets` has no `nvptx64`.
    NoNvptx,
    Ok,
}

const EXE: &str = if cfg!(windows) { ".exe" } else { "" };

fn clang_in(dir: &Path) -> [PathBuf; 2] {
    [dir.join(format!("clang{EXE}")), dir.join("bin").join(format!("clang{EXE}"))]
}

/// Ordered list of `(origin, candidate path)` for the implicit part of the lookup (steps 3 to 5).
pub fn implicit_candidates(env: &Env) -> Vec<(&'static str, PathBuf)> {
    let mut c: Vec<(&'static str, PathBuf)> = Vec::new();
    let mut beside: Vec<PathBuf> = Vec::new();
    for d in [&env.module_dir, &env.exe_dir].into_iter().flatten() {
        if !beside.contains(d) {
            beside.push(d.clone());
        }
    }
    for d in &beside {
        for p in clang_in(&d.join("clang")) {
            c.push(("a clang/ directory beside the shim", p));
        }
    }
    let mut names = vec!["clang".to_owned()];
    names.extend((15..=25).rev().map(|v| format!("clang-{v}")));
    if let Some(path) = &env.path {
        for dir in std::env::split_paths(path) {
            for n in &names {
                c.push(("PATH", dir.join(format!("{n}{EXE}"))));
            }
        }
    }
    for dir in ["/opt/homebrew/opt/llvm/bin", "/usr/local/opt/llvm/bin", "C:\\Program Files\\LLVM\\bin"] {
        c.push(("a standard install location", Path::new(dir).join(format!("clang{EXE}"))));
    }
    for v in (15..=25).rev() {
        c.push(("a standard install location", PathBuf::from(format!("/usr/lib/llvm-{v}/bin/clang"))));
    }
    c
}

fn describe(origin: &str, p: &Path, probe: &Probe) -> String {
    match probe {
        Probe::Missing => format!("{origin}: {} does not exist", p.display()),
        Probe::NotRunnable(why) => format!("{origin}: {} could not be run ({why})", p.display()),
        Probe::NoNvptx => format!(
            "{origin}: {} is a clang built WITHOUT the NVPTX backend (`--print-targets` has no nvptx64)",
            p.display()
        ),
        Probe::Ok => String::new(),
    }
}

const HINT: &str = "The NVRTC shim needs an LLVM clang with the NVPTX backend (the official LLVM release archives \
and distro/Homebrew packages have it; some vendor clangs, e.g. the Intel oneAPI icx-based ones, do not). \
Install LLVM and either put its bin directory on PATH, set NVRTC_SHIM_CLANG to the clang binary, \
set NVRTC_SHIM_CLANG_DIR to the LLVM directory, or place the LLVM tree in a clang/ directory next to the shim library.";

/// Resolve the clang to use. `probe` checks one binary.
pub fn resolve(env: &Env, probe: &dyn Fn(&Path) -> Probe) -> Result<PathBuf, String> {
    let mut notes: Vec<String> = Vec::new();
    // Explicit settings: a failure here is final, so a typo is not silently masked by PATH.
    for (var, val) in [("NVRTC_SHIM_CLANG", &env.clang), ("NVRTC_SHIM_CLANG_DIR", &env.clang_dir)] {
        let Some(val) = val else { continue };
        let given = PathBuf::from(val);
        let mut cands = vec![given.clone()];
        if var == "NVRTC_SHIM_CLANG_DIR" {
            cands = clang_in(&given).to_vec();
        } else if !given.is_file() {
            cands.extend(clang_in(&given)); // a directory was given
        }
        let mut local = Vec::new();
        for c in &cands {
            match probe(c) {
                Probe::Ok => return Ok(c.clone()),
                Probe::Missing => {}
                other => local.push(describe(var, c, &other)),
            }
        }
        if local.is_empty() {
            local.push(format!("{var}={} does not point to a clang binary (looked for {})", given.display(),
                cands.iter().map(|c| c.display().to_string()).collect::<Vec<_>>().join(", ")));
        }
        return Err(format!("{}\n{HINT}", local.join("\n")));
    }
    for (origin, p) in implicit_candidates(env) {
        match probe(&p) {
            Probe::Ok => return Ok(p),
            Probe::Missing => {}
            other => notes.push(describe(origin, &p, &other)),
        }
    }
    let head = if notes.is_empty() { "no clang found".to_owned() } else { notes.join("\n") };
    Err(format!("{head}\n{HINT}"))
}

/// Real probe: runs `<clang> --print-targets`.
pub fn probe_binary(p: &Path) -> Probe {
    if !p.is_file() {
        return Probe::Missing;
    }
    match std::process::Command::new(p).arg("--print-targets").output() {
        Err(e) => Probe::NotRunnable(e.to_string()),
        Ok(o) if !o.status.success() => Probe::NotRunnable(format!("exit {}", o.status)),
        Ok(o) => {
            let all = format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr));
            if all.contains("nvptx64") { Probe::Ok } else { Probe::NoNvptx }
        }
    }
}

/// True when clang's stderr from a compile says the NVPTX target is not built in.
pub fn indicates_missing_nvptx(log: &str) -> bool {
    ["unknown target triple 'nvptx", "No available targets are compatible", "unable to create target", "unknown target 'nvptx", "unsupported option '--cuda-gpu-arch"]
        .iter()
        .any(|s| log.contains(s))
}

pub const MISSING_NVPTX_MESSAGE: &str =
    "error: the clang used by the NVRTC shim has no NVPTX backend, so it cannot generate PTX. Use an LLVM clang whose `clang --print-targets` lists nvptx64 \
     (see NVRTC_SHIM_CLANG / NVRTC_SHIM_CLANG_DIR in docs/nvrtc.md).";

#[cfg(windows)]
fn module_dir() -> Option<PathBuf> {
    use std::os::windows::ffi::OsStringExt;
    unsafe extern "system" {
        fn GetModuleHandleExW(flags: u32, addr: *const core::ffi::c_void, out: *mut *mut core::ffi::c_void) -> i32;
        fn GetModuleFileNameW(module: *mut core::ffi::c_void, buf: *mut u16, len: u32) -> u32;
    }
    let mut h = std::ptr::null_mut();
    let mut buf = [0u16; 1024];
    unsafe {
        // FROM_ADDRESS | UNCHANGED_REFCOUNT
        if GetModuleHandleExW(0x4 | 0x2, module_dir as *const core::ffi::c_void, &mut h) == 0 {
            return None;
        }
        let n = GetModuleFileNameW(h, buf.as_mut_ptr(), buf.len() as u32) as usize;
        if n == 0 || n >= buf.len() {
            return None;
        }
        PathBuf::from(OsString::from_wide(&buf[..n])).parent().map(Path::to_path_buf)
    }
}

#[cfg(unix)]
fn module_dir() -> Option<PathBuf> {
    use std::os::unix::ffi::OsStringExt;
    #[repr(C)]
    struct DlInfo {
        fname: *const core::ffi::c_char,
        fbase: *mut core::ffi::c_void,
        sname: *const core::ffi::c_char,
        saddr: *mut core::ffi::c_void,
    }
    unsafe extern "C" {
        fn dladdr(addr: *const core::ffi::c_void, info: *mut DlInfo) -> i32;
    }
    let mut info = DlInfo { fname: std::ptr::null(), fbase: std::ptr::null_mut(), sname: std::ptr::null(), saddr: std::ptr::null_mut() };
    unsafe {
        if dladdr(module_dir as *const core::ffi::c_void, &mut info) == 0 || info.fname.is_null() {
            return None;
        }
        let bytes = std::ffi::CStr::from_ptr(info.fname).to_bytes().to_vec();
        PathBuf::from(OsString::from_vec(bytes)).parent().map(Path::to_path_buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn probe_map(m: HashMap<PathBuf, Probe>) -> impl Fn(&Path) -> Probe {
        move |p| m.get(p).cloned().unwrap_or(Probe::Missing)
    }
    fn exe(p: &str) -> PathBuf {
        PathBuf::from(format!("{p}{EXE}"))
    }

    #[test]
    fn explicit_file_wins_over_everything() {
        let env = Env { clang: Some("/x/clang".into()), path: Some("/p".into()), ..Default::default() };
        let m = HashMap::from([(PathBuf::from("/x/clang"), Probe::Ok), (exe("/p/clang"), Probe::Ok)]);
        assert_eq!(resolve(&env, &probe_map(m)).unwrap(), PathBuf::from("/x/clang"));
    }

    #[test]
    fn explicit_without_nvptx_is_final_and_names_the_problem() {
        let env = Env { clang: Some("/x/clang".into()), path: Some("/p".into()), ..Default::default() };
        let m = HashMap::from([(PathBuf::from("/x/clang"), Probe::NoNvptx), (exe("/p/clang"), Probe::Ok)]);
        let e = resolve(&env, &probe_map(m)).unwrap_err();
        assert!(e.contains("WITHOUT the NVPTX backend") && e.contains("NVRTC_SHIM_CLANG"), "{e}");
    }

    #[test]
    fn explicit_missing_is_final() {
        let env = Env { clang: Some("/nope/clang".into()), path: Some("/p".into()), ..Default::default() };
        let m = HashMap::from([(exe("/p/clang"), Probe::Ok)]);
        let e = resolve(&env, &probe_map(m)).unwrap_err();
        assert!(e.contains("does not point to a clang binary"), "{e}");
    }

    #[test]
    fn clang_env_may_name_a_directory() {
        let env = Env { clang: Some("/llvm".into()), ..Default::default() };
        let m = HashMap::from([(PathBuf::from("/llvm").join("bin").join(format!("clang{EXE}")), Probe::Ok)]);
        assert!(resolve(&env, &probe_map(m)).unwrap().ends_with(format!("clang{EXE}")));
    }

    #[test]
    fn clang_dir_checks_dir_then_bin() {
        let env = Env { clang_dir: Some("/llvm".into()), ..Default::default() };
        let want = PathBuf::from("/llvm").join("bin").join(format!("clang{EXE}"));
        let m = HashMap::from([(want.clone(), Probe::Ok)]);
        assert_eq!(resolve(&env, &probe_map(m)).unwrap(), want);
        let e = resolve(&env, &probe_map(HashMap::new())).unwrap_err();
        assert!(e.contains("NVRTC_SHIM_CLANG_DIR"), "{e}");
    }

    #[test]
    fn bundled_clang_beside_the_shim_beats_path() {
        let env = Env { module_dir: Some("/app".into()), path: Some("/p".into()), ..Default::default() };
        let bundled = PathBuf::from("/app").join("clang").join("bin").join(format!("clang{EXE}"));
        let m = HashMap::from([(bundled.clone(), Probe::Ok), (exe("/p/clang"), Probe::Ok)]);
        assert_eq!(resolve(&env, &probe_map(m)).unwrap(), bundled);
    }

    #[test]
    fn exe_dir_is_searched_too_and_deduplicated() {
        let env = Env { module_dir: Some("/app".into()), exe_dir: Some("/app".into()), ..Default::default() };
        assert_eq!(implicit_candidates(&env).iter().filter(|(o, _)| o.contains("beside")).count(), 2);
        let env = Env { module_dir: Some("/lib".into()), exe_dir: Some("/bin".into()), ..Default::default() };
        let bundled = PathBuf::from("/bin").join("clang").join(format!("clang{EXE}"));
        let m = HashMap::from([(bundled.clone(), Probe::Ok)]);
        assert_eq!(resolve(&env, &probe_map(m)).unwrap(), bundled);
    }

    #[test]
    fn path_skips_clang_without_nvptx_and_takes_next() {
        let env = Env { path: Some(std::env::join_paths(["/a", "/b"]).unwrap()), ..Default::default() };
        let m = HashMap::from([(exe("/a/clang"), Probe::NoNvptx), (exe("/b/clang"), Probe::Ok)]);
        assert_eq!(resolve(&env, &probe_map(m)).unwrap(), exe("/b/clang"));
    }

    #[test]
    fn nothing_found_gives_actionable_message() {
        let e = resolve(&Env::default(), &probe_map(HashMap::new())).unwrap_err();
        assert!(e.contains("no clang found") && e.contains("NVRTC_SHIM_CLANG_DIR") && e.contains("PATH"), "{e}");
    }

    #[test]
    fn rejected_candidates_are_reported_when_nothing_works() {
        let env = Env { path: Some("/a".into()), ..Default::default() };
        let m = HashMap::from([(exe("/a/clang"), Probe::NoNvptx)]);
        let e = resolve(&env, &probe_map(m)).unwrap_err();
        assert!(e.contains("WITHOUT the NVPTX backend") && e.contains("/a"), "{e}");
    }

    #[test]
    fn detects_missing_backend_text() {
        assert!(indicates_missing_nvptx("error: unknown target triple 'nvptx64-nvidia-cuda'"));
        assert!(indicates_missing_nvptx("error: unable to create target: 'No available targets are compatible with triple \"nvptx64\"'"));
        assert!(!indicates_missing_nvptx("hw.cu:3:1: error: expected ';'"));
    }

    #[test]
    fn real_probe_handles_missing_and_non_clang() {
        assert_eq!(probe_binary(Path::new("/definitely/not/here")), Probe::Missing);
        assert!(module_dir().is_some());
    }
}
