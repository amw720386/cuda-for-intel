//! CUDA C++ to PTX through an external clang with the NVPTX backend.
//!
//! The strategy is documented in `docs/nvrtc.md`. This module is pure Rust (no FFI) so it can be
//! unit-tested and used from other crates.
use crate::{cache, headers, options};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, OnceLock};

/// Newest virtual architecture the generated PTX is allowed to target unless
/// `NVRTC_SHIM_MAX_ARCH` raises it. The translator only implements sm_75-era instructions, so a
/// request for compute_86 is compiled as sm_75 (and `__CUDA_ARCH__` reflects that).
pub const DEFAULT_ARCH_CAP: u32 = 75;
/// Architectures reported by `nvrtcGetSupportedArchs`.
pub const SUPPORTED_ARCHS: &[i32] = &[50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 87, 89, 90];

#[derive(Debug, Clone, Default)]
pub struct Request {
    pub source: String,
    pub name: String,
    /// (include name, contents)
    pub headers: Vec<(String, String)>,
    pub options: Vec<String>,
    pub name_expressions: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct Output {
    pub ptx: String,
    pub log: String,
    /// name expression -> mangled (lowered) name
    pub lowered: BTreeMap<String, String>,
    /// Effective `sm_XX` the PTX was generated for (0 for pass-through).
    pub arch: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Failure {
    InvalidOption(String),
    /// Compilation diagnostics (log) or tool problems.
    Compilation(String),
    /// No usable clang found.
    NoToolchain(String),
}

impl Failure {
    pub fn log(&self) -> &str {
        match self {
            Failure::InvalidOption(s) | Failure::Compilation(s) | Failure::NoToolchain(s) => s,
        }
    }
}

/// True when `source` is already PTX (first token after comments is `.version`).
pub fn is_ptx(source: &str) -> bool {
    let mut rest = source.trim_start();
    loop {
        if let Some(r) = rest.strip_prefix("//") {
            rest = r.split_once('\n').map_or("", |x| x.1).trim_start();
        } else if let Some(r) = rest.strip_prefix("/*") {
            rest = r.split_once("*/").map_or("", |x| x.1).trim_start();
        } else {
            break;
        }
    }
    rest.starts_with(".version")
}

fn llvm_has_nvptx(clang: &Path) -> bool {
    Command::new(clang)
        .arg("--print-targets")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).contains("nvptx64"))
        .unwrap_or(false)
}

/// Locates a clang with NVPTX support: `NVRTC_SHIM_CLANG`, then PATH, then standard install dirs.
pub fn find_clang() -> Result<PathBuf, String> {
    static FOUND: OnceLock<Result<PathBuf, String>> = OnceLock::new();
    FOUND
        .get_or_init(|| {
            if let Some(p) = std::env::var_os("NVRTC_SHIM_CLANG") {
                let p = PathBuf::from(p);
                return if llvm_has_nvptx(&p) {
                    Ok(p)
                } else {
                    Err(format!("NVRTC_SHIM_CLANG={} is not a clang with the NVPTX backend", p.display()))
                };
            }
            let exe = if cfg!(windows) { ".exe" } else { "" };
            let mut cands: Vec<PathBuf> = Vec::new();
            let mut names = vec!["clang".to_owned()];
            names.extend((15..=25).rev().map(|v| format!("clang-{v}")));
            if let Some(path) = std::env::var_os("PATH") {
                for dir in std::env::split_paths(&path) {
                    for n in &names {
                        cands.push(dir.join(format!("{n}{exe}")));
                    }
                }
            }
            for dir in [
                "/opt/homebrew/opt/llvm/bin",
                "/usr/local/opt/llvm/bin",
                "C:\\Program Files\\LLVM\\bin",
            ] {
                cands.push(Path::new(dir).join(format!("clang{exe}")));
            }
            for v in (15..=25).rev() {
                cands.push(PathBuf::from(format!("/usr/lib/llvm-{v}/bin/clang")));
            }
            for c in cands {
                if c.is_file() && llvm_has_nvptx(&c) {
                    return Ok(c);
                }
            }
            Err("no clang with the NVPTX backend found; install LLVM (e.g. `brew install llvm`, `apt install clang`, \
                 or the Windows LLVM installer) or set NVRTC_SHIM_CLANG to its clang binary"
                .into())
        })
        .clone()
}

fn clang_version(clang: &Path) -> String {
    Command::new(clang)
        .arg("--version")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).lines().next().unwrap_or("").to_owned())
        .unwrap_or_default()
}

fn arch_cap() -> u32 {
    std::env::var("NVRTC_SHIM_MAX_ARCH").ok().and_then(|v| v.trim_start_matches("sm_").parse().ok()).unwrap_or(DEFAULT_ARCH_CAP)
}

fn ptx_isa_for(arch: u32) -> &'static str {
    match arch {
        0..=89 => "+ptx80",
        90..=99 => "+ptx83",
        _ => "+ptx86",
    }
}

/// Writes the built-in headers once per content hash and returns the directory.
fn builtin_dir() -> Result<PathBuf, String> {
    static DIR: OnceLock<Result<PathBuf, String>> = OnceLock::new();
    DIR.get_or_init(|| {
        let files = headers::files();
        let mut h = crate::hash::Hasher::default();
        for (n, c) in &files {
            h.str(n).str(c);
        }
        let dir = std::env::temp_dir().join(format!("nvrtc_shim_inc_{}", h.finish()));
        for (name, contents) in &files {
            let path = dir.join(name);
            if path.is_file() && std::fs::read_to_string(&path).is_ok_and(|c| c == *contents) {
                continue;
            }
            std::fs::create_dir_all(path.parent().unwrap()).map_err(|e| e.to_string())?;
            let tmp = path.with_extension(format!("tmp{}", std::process::id()));
            std::fs::write(&tmp, contents).map_err(|e| e.to_string())?;
            std::fs::rename(&tmp, &path).map_err(|e| e.to_string())?;
        }
        Ok(dir)
    })
    .clone()
}

fn safe_relative(name: &str) -> Option<PathBuf> {
    let p = Path::new(name);
    if name.is_empty() || p.is_absolute() || p.components().any(|c| matches!(c, std::path::Component::ParentDir | std::path::Component::Prefix(_))) {
        return None;
    }
    Some(p.to_path_buf())
}

struct TempDir(PathBuf);
impl TempDir {
    fn new() -> std::io::Result<Self> {
        static N: Mutex<u64> = Mutex::new(0);
        let n = {
            let mut g = N.lock().unwrap();
            *g += 1;
            *g
        };
        let p = std::env::temp_dir().join(format!("nvrtc_shim_{}_{n}", std::process::id()));
        std::fs::create_dir_all(&p)?;
        Ok(Self(p))
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn cache_key(req: &Request, parsed: &options::Parsed, clang_id: &str, arch: u32) -> String {
    let mut h = crate::hash::Hasher::default();
    h.str(env!("CARGO_PKG_VERSION")).str(clang_id).str(&arch.to_string());
    h.str(&req.source).str(&req.name);
    for (n, c) in &req.headers {
        h.str(n).str(c);
    }
    for o in &req.options {
        h.str(o);
    }
    h.str(&format!("{parsed:?}"));
    h.str("names");
    for n in &req.name_expressions {
        h.str(n);
    }
    for (_, c) in headers::files() {
        h.str(c);
    }
    h.finish()
}

/// Compile CUDA C++ (or pass PTX through) as `nvrtcCompileProgram` does.
pub fn compile(req: &Request) -> Result<Output, Failure> {
    let parsed = options::parse(&req.options).map_err(Failure::InvalidOption)?;
    if is_ptx(&req.source) {
        return Ok(Output { ptx: req.source.clone(), log: String::new(), lowered: BTreeMap::new(), arch: 0 });
    }
    let clang = find_clang().map_err(Failure::NoToolchain)?;
    let arch = parsed.arch.clamp(50, arch_cap());
    let clang_id = format!("{}|{}", clang.display(), clang_version(&clang));
    let key = cache_key(req, &parsed, &clang_id, arch);
    if let Some(hit) = cache::get(&key) {
        return hit;
    }
    let result = run_clang(req, &parsed, &clang, arch);
    cache::put(&key, &result);
    result
}

fn run_clang(req: &Request, parsed: &options::Parsed, clang: &Path, arch: u32) -> Result<Output, Failure> {
    let builtin = builtin_dir().map_err(|e| Failure::Compilation(format!("cannot stage built-in headers: {e}")))?;
    let tmp = TempDir::new().map_err(|e| Failure::Compilation(format!("cannot create temp dir: {e}")))?;
    let inc = tmp.0.join("inc");
    std::fs::create_dir_all(&inc).map_err(|e| Failure::Compilation(e.to_string()))?;
    for (name, contents) in &req.headers {
        let rel = safe_relative(name).ok_or_else(|| Failure::Compilation(format!("invalid include name '{name}'")))?;
        let path = inc.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).map_err(|e| Failure::Compilation(e.to_string()))?;
        std::fs::write(&path, contents).map_err(|e| Failure::Compilation(e.to_string()))?;
    }
    // Main source: line directive keeps diagnostics pointing at the user's program name.
    let display = req.name.replace(['\\', '"', '\n'], "_");
    let mut src = String::new();
    if parsed.default_device {
        src.push_str("#pragma clang force_cuda_host_device begin\n");
    }
    src.push_str(&format!("#line 1 \"{display}\"\n"));
    src.push_str(&req.source);
    if !src.ends_with('\n') {
        src.push('\n');
    }
    if parsed.default_device {
        // Closing pragma after the user's code, before the name-expression probes.
        src.push_str("#pragma clang force_cuda_host_device end\n");
    }
    let name_exprs: Vec<&String> = req.name_expressions.iter().collect();
    for (i, e) in name_exprs.iter().enumerate() {
        let expr = e.trim();
        let addr = if expr.starts_with('&') { expr.to_owned() } else { format!("&({expr})") };
        src.push_str(&format!("__attribute__((device)) void* __nvrtc_ne_{i} = reinterpret_cast<void*>({addr});\n"));
    }
    let main = tmp.0.join("program.cu");
    std::fs::write(&main, &src).map_err(|e| Failure::Compilation(e.to_string()))?;
    let out = tmp.0.join("out.ptx");

    let mut cmd = Command::new(clang);
    cmd.args(["-x", "cuda", "--cuda-device-only"])
        .arg(format!("--cuda-gpu-arch=sm_{arch}"))
        .args(["-nocudainc", "-nocudalib", "-nostdinc", "-nostdinc++", "-O2", "-fno-jump-tables"])
        .args(["-Xclang", "-target-feature", "-Xclang", ptx_isa_for(arch)])
        .arg(format!("-std={}", parsed.std))
        .args(["-fno-exceptions", "-fno-rtti", "-fno-color-diagnostics", "-fno-diagnostics-show-note-include-stack"])
        .args(["-Wno-unknown-cuda-version", "-Wno-unused-command-line-argument", "-Wno-c++17-extensions"])
        .arg(format!("-I{}", inc.display()))
        .arg("-isystem")
        .arg(&builtin)
        .arg("-include")
        .arg(builtin.join(headers::BUILTINS));
    cmd.arg(if parsed.fmad { "-ffp-contract=fast" } else { "-ffp-contract=off" });
    if parsed.ftz {
        cmd.arg("-fcuda-flush-denormals-to-zero");
    }
    if parsed.fast_math {
        cmd.args(["-D__USE_FAST_MATH__=1", "-fno-math-errno"]);
    }
    if parsed.no_warnings {
        cmd.arg("-w");
    }
    cmd.args(&parsed.clang_args).arg("-S").arg("-o").arg(&out).arg(&main);

    let output = cmd.output().map_err(|e| Failure::NoToolchain(format!("cannot run {}: {e}", clang.display())))?;
    let mut log = String::from_utf8_lossy(&output.stderr).into_owned();
    // Hide staging paths so logs read like NVRTC's.
    for (needle, repl) in [
        (tmp.0.join("program.cu").display().to_string(), display.clone()),
        (format!("{}/", builtin.display()), String::new()),
        (format!("{}/", inc.display()), String::new()),
    ] {
        log = log.replace(&needle, &repl);
    }
    if !output.status.success() {
        if log.trim().is_empty() {
            log = format!("clang exited with {}\n", output.status);
        }
        return Err(Failure::Compilation(log));
    }
    let mut ptx = std::fs::read_to_string(&out).map_err(|e| Failure::Compilation(format!("{log}\nno PTX produced: {e}")))?;
    let lowered = extract_lowered(&mut ptx, &name_exprs);
    for (i, e) in name_exprs.iter().enumerate() {
        if !lowered.contains_key(e.as_str()) {
            return Err(Failure::Compilation(format!("{log}\nerror: could not lower name expression #{i} '{e}'\n")));
        }
    }
    Ok(Output { ptx, log, lowered, arch })
}

/// Reads `__nvrtc_ne_N = <symbol>;` initializers and removes the probe variables and the
/// forward declarations clang emitted for them.
fn extract_lowered(ptx: &mut String, exprs: &[&String]) -> BTreeMap<String, String> {
    let mut lowered = BTreeMap::new();
    let mut keep: Vec<&str> = Vec::new();
    for line in ptx.lines() {
        if let Some(pos) = line.find("__nvrtc_ne_") {
            let rest = &line[pos + "__nvrtc_ne_".len()..];
            let idx: String = rest.chars().take_while(char::is_ascii_digit).collect();
            if let (Ok(i), Some((_, init))) = (idx.parse::<usize>(), line.split_once('=')) {
                let sym = init.trim().trim_end_matches(';').trim();
                if let Some(e) = exprs.get(i) {
                    lowered.insert((*e).clone(), sym.to_owned());
                }
            }
            continue;
        }
        keep.push(line);
    }
    let mut text = keep.join("\n");
    text.push('\n');
    *ptx = strip_entry_forward_decls(&text);
    lowered
}

/// Removes `.visible .entry NAME ( ... ) ;` prototypes whose definition appears later.
fn strip_entry_forward_decls(ptx: &str) -> String {
    let lines: Vec<&str> = ptx.lines().collect();
    let mut out = Vec::with_capacity(lines.len());
    let mut i = 0;
    while i < lines.len() {
        let l = lines[i];
        if l.starts_with(".visible .entry ") || l.starts_with(".entry ") {
            // Prototype form: header line, "(", params, ")", ";"
            let mut j = i + 1;
            while j < lines.len() && lines[j] != ";" && !lines[j].starts_with('{') && !lines[j].starts_with(".visible") && j < i + 64 {
                j += 1;
            }
            if j < lines.len() && lines[j] == ";" {
                i = j + 1;
                continue;
            }
        }
        out.push(l);
        i += 1;
    }
    let mut s = out.join("\n");
    s.push('\n');
    s
}
