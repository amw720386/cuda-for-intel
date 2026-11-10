//! `cfi-precompile [--install-dir DIR] [options] <file-or-dir>...`
//!
//! Loads cuda-for-intel's `nvcuda` library and runs its exported `cfi_precompile_main`, so the
//! translator that fills the caches is exactly the one the application will load (its version
//! is part of every cache key). The library is looked up in `--install-dir`, then
//! `CFI_REDIRECT_DIR`, then next to this executable. All other options are the library's; run
//! with `--help` for the list.

use std::ffi::{CString, c_char, c_int};
use std::path::PathBuf;

#[cfg(windows)]
const LIB: &str = "nvcuda.dll";
#[cfg(target_os = "macos")]
const LIB: &str = "libnvcuda.dylib";
#[cfg(all(unix, not(target_os = "macos")))]
const LIB: &str = "libnvcuda.so";

/// The cuBLAS shim next to `nvcuda` generates its XMX GEMM kernels at run time; ask it to build
/// the common ones now (the primary context opened by the precompile run is still current).
fn xmx_kernels(nvcuda: &std::path::Path) {
    #[cfg(windows)]
    let name = "cublas64_12.dll";
    #[cfg(not(windows))]
    let name = "libcublas_shim.so";
    let Some(path) = nvcuda.parent().map(|d| d.join(name)).filter(|p| p.is_file()) else { return };
    // SAFETY: our own cuBLAS shim; it binds to the nvcuda already loaded in this process.
    let Ok(lib) = (unsafe { libloading::Library::new(&path) }) else { return };
    type Warm = unsafe extern "system" fn() -> u32;
    // SAFETY: signature as exported in crates/cublas_shim/src/xmx_host.rs.
    if let Ok(warm) = unsafe { lib.get::<Warm>(b"cfiXmxPrecompile\0") } {
        let started = std::time::Instant::now();
        let n = unsafe { warm() };
        println!("cuBLAS XMX kernels: {n} built or cached in {:.1}s", started.elapsed().as_secs_f64());
    }
    std::mem::forget(lib);
}

fn main() {
    let mut args: Vec<String> = std::env::args().collect();
    let mut dir: Option<PathBuf> = None;
    if let Some(i) = args.iter().position(|a| a == "--install-dir") {
        if i + 1 >= args.len() {
            eprintln!("cfi-precompile: --install-dir needs a value");
            std::process::exit(1);
        }
        dir = Some(PathBuf::from(args.remove(i + 1)));
        args.remove(i);
    }
    let candidates: Vec<PathBuf> = dir
        .into_iter()
        .chain(std::env::var_os("CFI_REDIRECT_DIR").map(PathBuf::from))
        .chain(std::env::current_exe().ok().and_then(|e| e.parent().map(PathBuf::from)))
        .map(|d| d.join(LIB))
        .collect();
    let Some(lib_path) = candidates.iter().find(|p| p.is_file()) else {
        eprintln!(
            "cfi-precompile: {LIB} not found (tried {}); pass --install-dir",
            candidates.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
        );
        std::process::exit(1);
    };
    // SAFETY: loading our own library runs no initialisers beyond the Rust runtime's.
    let lib = match unsafe { libloading::Library::new(lib_path) } {
        Ok(l) => l,
        Err(e) => {
            eprintln!("cfi-precompile: cannot load {}: {e}", lib_path.display());
            std::process::exit(1);
        }
    };
    type Main = unsafe extern "C" fn(c_int, *const *const c_char) -> c_int;
    // SAFETY: the symbol has this signature in src/cuda/precompile.rs.
    let main: libloading::Symbol<Main> = match unsafe { lib.get(b"cfi_precompile_main\0") } {
        Ok(f) => f,
        Err(_) => {
            eprintln!(
                "cfi-precompile: {} has no cfi_precompile_main (a build older than the precompile tool)",
                lib_path.display()
            );
            std::process::exit(1);
        }
    };
    let native = !args.iter().any(|a| matches!(a.as_str(), "--translate-only" | "--no-native" | "--list" | "-h" | "--help"));
    let owned: Vec<CString> = args.into_iter().map(|a| CString::new(a).unwrap_or_default()).collect();
    let ptrs: Vec<*const c_char> = owned.iter().map(|a| a.as_ptr()).collect();
    // SAFETY: argv holds argc valid NUL-terminated strings that outlive the call.
    let code = unsafe { main(ptrs.len() as c_int, ptrs.as_ptr()) };
    if code == 0 && native {
        xmx_kernels(lib_path);
    }
    // Exit without unloading: the library's device threads may still hold driver state.
    std::mem::forget(lib);
    std::process::exit(code);
}
