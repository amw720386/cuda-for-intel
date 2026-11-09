//! Which DLL names are redirected, and to which file in the install directory.
//!
//! Families (case-insensitive base names, `N` = digits and `_`):
//! `nvcuda.dll`, `nvml.dll`, `cublas64_N.dll`, `cublasLt64_N.dll`, `cudnn*64_N.dll`,
//! `nvrtc64_N.dll`, `cufft64_N.dll`, `cusparse64_N.dll`, `cusolver64_N.dll`.
//! A request is redirected only when one of its candidates exists in the install directory, so a
//! library we do not ship (or an opt-in one the user did not install) still loads NVIDIA's.

/// Install-directory candidates for `base` (a file name without directory), best first. The
/// first candidate is always `base` itself; later ones are our stand-in under another version
/// (for example cu118's `cublas64_11.dll` is served by our `cublas64_12.dll`, which exports the
/// same `_v2` ABI). `None` when the name is not a redirected family.
pub fn candidates(base: &str) -> Option<Vec<String>> {
    let lower = base.to_ascii_lowercase();
    let versioned = |prefix: &str| {
        lower
            .strip_prefix(prefix)
            .and_then(|rest| rest.strip_suffix(".dll"))
            .is_some_and(|v| !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit() || b == b'_'))
    };
    let fallbacks: &[&str] = if lower == "nvcuda.dll" || lower == "nvml.dll" {
        &[]
    } else if versioned("cublaslt64_") {
        &["cublasLt64_12.dll", "cublas64_12.dll"]
    } else if versioned("cublas64_") {
        &["cublas64_12.dll"]
    } else if versioned("nvrtc64_") {
        &["nvrtc64_120_0.dll"]
    } else if versioned("cufft64_") {
        &["cufft64_11.dll"]
    } else if versioned("cusparse64_") {
        &["cusparse64_12.dll"]
    } else if versioned("cusolver64_") {
        &["cusolver64_11.dll"]
    } else if is_cudnn(&lower) {
        // cudnn64_9 and its sub-libraries (cudnn_ops64_9, cudnn_graph64_9, ...): same name only.
        &[]
    } else {
        return None;
    };
    let mut out = vec![base.to_string()];
    for f in fallbacks {
        if !f.eq_ignore_ascii_case(base) {
            out.push((*f).to_string());
        }
    }
    Some(out)
}

fn is_cudnn(lower: &str) -> bool {
    let Some(stem) = lower.strip_prefix("cudnn").and_then(|s| s.strip_suffix(".dll")) else {
        return false;
    };
    let Some(at) = stem.rfind("64_") else { return false };
    let (kind, version) = (&stem[..at], &stem[at + 3..]);
    (kind.is_empty() || (kind.starts_with('_') && kind[1..].bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')))
        && !version.is_empty()
        && version.bytes().all(|b| b.is_ascii_digit())
}

/// File-name component of a Windows path (after the last `\` or `/`).
pub fn base_name(path: &str) -> &str {
    path.rsplit(['\\', '/']).next().unwrap_or(path)
}

/// Where a request for `requested` should load from, given which files exist in the install
/// directory. Returns `(install_file, alias)`: `install_file` is the candidate found; `alias` is
/// true when it has another name than the request, so the caller must load a copy named like the
/// request (the loader binds later by-name imports to the module's own file name).
pub fn resolve(requested: &str, exists: impl Fn(&str) -> bool) -> Option<(String, bool)> {
    let base = base_name(requested);
    let list = candidates(base)?;
    list.into_iter()
        .find(|c| exists(c))
        .map(|c| {
            let alias = !c.eq_ignore_ascii_case(base);
            (c, alias)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn families_match_and_others_do_not() {
        for yes in [
            "nvcuda.dll",
            "NVCUDA.DLL",
            "nvml.dll",
            "cublas64_12.dll",
            "cublas64_11.dll",
            "cublasLt64_12.dll",
            "cudnn64_9.dll",
            "cudnn64_8.dll",
            "cudnn_ops64_9.dll",
            "cudnn_engines_precompiled64_9.dll",
            "nvrtc64_120_0.dll",
            "nvrtc64_112_0.dll",
            "cufft64_11.dll",
            "cusparse64_12.dll",
            "cusolver64_11.dll",
        ] {
            assert!(candidates(yes).is_some(), "{yes}");
        }
        for no in [
            "kernel32.dll",
            "cudart64_12.dll",
            "nvrtc-builtins64_120.dll",
            "cufftw64_11.dll",
            "cusolverMg64_11.dll",
            "cublas64_.dll",
            "cublas64_12.dll.bak",
            "torch_cuda.dll",
            "cudnnx.dll",
            "nvcuda_hybrid.dll",
        ] {
            assert!(candidates(no).is_none(), "{no}");
        }
    }

    #[test]
    fn absolute_path_resolves_by_base_name() {
        let have = |n: &str| n.eq_ignore_ascii_case("cublas64_12.dll");
        let p = r"C:\app\venv\Lib\site-packages\torch\lib\cublas64_12.dll"; // leak-ok: placeholder path
        assert_eq!(resolve(p, have), Some(("cublas64_12.dll".into(), false)));
        assert_eq!(resolve("D:/x/cuBLAS64_11.DLL", have), Some(("cublas64_12.dll".into(), true)));
        assert_eq!(resolve(r"x\cublasLt64_12.dll", have), Some(("cublas64_12.dll".into(), true)));
        let have_lt = |n: &str| n == "cublasLt64_12.dll" || n == "cublas64_12.dll";
        assert_eq!(resolve("cublasLt64_11.dll", have_lt), Some(("cublasLt64_12.dll".into(), true)));
    }

    #[test]
    fn missing_from_install_dir_is_not_redirected() {
        assert_eq!(resolve(r"C:\t\cudnn64_9.dll", |_| false), None); // leak-ok
        assert_eq!(resolve("cudnn_ops64_9.dll", |n| n == "cudnn64_9.dll"), None);
        assert_eq!(resolve("kernel32.dll", |_| true), None);
    }
}
