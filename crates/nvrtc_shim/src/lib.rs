//! NVRTC compatibility shim (`nvrtc64_120_0.dll` / `libnvrtc.so.12`).
//!
//! CUDA C++ is compiled to PTX by an external LLVM clang (NVPTX backend) with a small built-in
//! CUDA runtime header subset; PTX source is passed through unchanged. See `docs/nvrtc.md`.
#![allow(non_snake_case, clippy::missing_safety_doc)]

pub mod cache;
pub mod compiler;
mod hash;
mod headers;
pub mod options;

use compiler::{Failure, Output, Request};
use std::ffi::{CStr, c_char, c_int, c_void};

pub type NvrtcResult = c_int;
pub const NVRTC_SUCCESS: NvrtcResult = 0;
pub const NVRTC_ERROR_OUT_OF_MEMORY: NvrtcResult = 1;
pub const NVRTC_ERROR_PROGRAM_CREATION_FAILURE: NvrtcResult = 2;
pub const NVRTC_ERROR_INVALID_INPUT: NvrtcResult = 3;
pub const NVRTC_ERROR_INVALID_PROGRAM: NvrtcResult = 4;
pub const NVRTC_ERROR_INVALID_OPTION: NvrtcResult = 5;
pub const NVRTC_ERROR_COMPILATION: NvrtcResult = 6;
pub const NVRTC_ERROR_BUILTIN_OPERATION_FAILURE: NvrtcResult = 7;
pub const NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION: NvrtcResult = 8;
pub const NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION: NvrtcResult = 9;
pub const NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID: NvrtcResult = 10;
pub const NVRTC_ERROR_INTERNAL_ERROR: NvrtcResult = 11;

/// Opaque `nvrtcProgram` handle.
pub struct Program {
    magic: u64,
    request: Request,
    compiled: bool,
    output: Output,
    /// Log of a failed compile (kept so `nvrtcGetProgramLog` works after an error).
    failure_log: String,
    /// C-string copies for `nvrtcGetLoweredName` (stable pointers).
    lowered_c: Vec<(String, std::ffi::CString)>,
}

const MAGIC: u64 = 0x4e56_5254_4350_5247;

pub type NvrtcProgram = *mut Program;

unsafe fn prog<'a>(p: NvrtcProgram) -> Option<&'a mut Program> {
    if p.is_null() {
        return None;
    }
    let r = unsafe { &mut *p };
    (r.magic == MAGIC).then_some(r)
}

unsafe fn cstr(p: *const c_char) -> Option<String> {
    if p.is_null() { None } else { Some(unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()) }
}

#[unsafe(no_mangle)]
pub extern "C" fn nvrtcGetErrorString(result: NvrtcResult) -> *const c_char {
    let s: &'static [u8] = match result {
        0 => b"NVRTC_SUCCESS\0",
        1 => b"NVRTC_ERROR_OUT_OF_MEMORY\0",
        2 => b"NVRTC_ERROR_PROGRAM_CREATION_FAILURE\0",
        3 => b"NVRTC_ERROR_INVALID_INPUT\0",
        4 => b"NVRTC_ERROR_INVALID_PROGRAM\0",
        5 => b"NVRTC_ERROR_INVALID_OPTION\0",
        6 => b"NVRTC_ERROR_COMPILATION\0",
        7 => b"NVRTC_ERROR_BUILTIN_OPERATION_FAILURE\0",
        8 => b"NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION\0",
        9 => b"NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION\0",
        10 => b"NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID\0",
        11 => b"NVRTC_ERROR_INTERNAL_ERROR\0",
        _ => b"NVRTC_ERROR unknown\0",
    };
    s.as_ptr().cast()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nvrtcVersion(major: *mut c_int, minor: *mut c_int) -> NvrtcResult {
    if major.is_null() || minor.is_null() {
        return NVRTC_ERROR_INVALID_INPUT;
    }
    unsafe {
        *major = 12;
        *minor = 4;
    }
    NVRTC_SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nvrtcGetNumSupportedArchs(n: *mut c_int) -> NvrtcResult {
    if n.is_null() {
        return NVRTC_ERROR_INVALID_INPUT;
    }
    unsafe { *n = compiler::SUPPORTED_ARCHS.len() as c_int };
    NVRTC_SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nvrtcGetSupportedArchs(out: *mut c_int) -> NvrtcResult {
    if out.is_null() {
        return NVRTC_ERROR_INVALID_INPUT;
    }
    for (i, a) in compiler::SUPPORTED_ARCHS.iter().enumerate() {
        unsafe { *out.add(i) = *a };
    }
    NVRTC_SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nvrtcCreateProgram(
    out: *mut NvrtcProgram,
    src: *const c_char,
    name: *const c_char,
    num_headers: c_int,
    headers: *const *const c_char,
    include_names: *const *const c_char,
) -> NvrtcResult {
    if out.is_null() {
        return NVRTC_ERROR_INVALID_INPUT;
    }
    let Some(source) = (unsafe { cstr(src) }) else { return NVRTC_ERROR_INVALID_INPUT };
    if num_headers < 0 || (num_headers > 0 && (headers.is_null() || include_names.is_null())) {
        return NVRTC_ERROR_INVALID_INPUT;
    }
    let mut hs = Vec::new();
    for i in 0..num_headers as usize {
        let (Some(c), Some(n)) = (unsafe { cstr(*headers.add(i)) }, unsafe { cstr(*include_names.add(i)) }) else {
            return NVRTC_ERROR_INVALID_INPUT;
        };
        hs.push((n, c));
    }
    let name = unsafe { cstr(name) }.filter(|n| !n.is_empty()).unwrap_or_else(|| "default_program".into());
    let p = Box::new(Program {
        magic: MAGIC,
        request: Request { source, name, headers: hs, ..Default::default() },
        compiled: false,
        output: Output::default(),
        failure_log: String::new(),
        lowered_c: Vec::new(),
    });
    unsafe { *out = Box::into_raw(p) };
    NVRTC_SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nvrtcDestroyProgram(p: *mut NvrtcProgram) -> NvrtcResult {
    if p.is_null() {
        return NVRTC_ERROR_INVALID_INPUT;
    }
    let handle = unsafe { *p };
    let Some(r) = (unsafe { prog(handle) }) else { return NVRTC_ERROR_INVALID_PROGRAM };
    r.magic = 0;
    drop(unsafe { Box::from_raw(handle) });
    unsafe { *p = std::ptr::null_mut() };
    NVRTC_SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nvrtcAddNameExpression(p: NvrtcProgram, expr: *const c_char) -> NvrtcResult {
    let Some(r) = (unsafe { prog(p) }) else { return NVRTC_ERROR_INVALID_PROGRAM };
    let Some(e) = (unsafe { cstr(expr) }).filter(|e| !e.trim().is_empty()) else { return NVRTC_ERROR_INVALID_INPUT };
    if r.compiled {
        return NVRTC_ERROR_INVALID_PROGRAM;
    }
    r.request.name_expressions.push(e);
    NVRTC_SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nvrtcCompileProgram(p: NvrtcProgram, num_options: c_int, options: *const *const c_char) -> NvrtcResult {
    let Some(r) = (unsafe { prog(p) }) else { return NVRTC_ERROR_INVALID_PROGRAM };
    if num_options < 0 || (num_options > 0 && options.is_null()) {
        return NVRTC_ERROR_INVALID_INPUT;
    }
    let mut opts = Vec::new();
    for i in 0..num_options as usize {
        match unsafe { cstr(*options.add(i)) } {
            Some(o) => opts.push(o),
            None => return NVRTC_ERROR_INVALID_OPTION,
        }
    }
    r.request.options = opts;
    r.lowered_c.clear();
    match compiler::compile(&r.request) {
        Ok(out) => {
            r.lowered_c = out
                .lowered
                .iter()
                .filter_map(|(k, v)| Some((k.clone(), std::ffi::CString::new(v.as_str()).ok()?)))
                .collect();
            r.output = out;
            r.failure_log.clear();
            r.compiled = true;
            NVRTC_SUCCESS
        }
        Err(f) => {
            r.failure_log = f.log().to_owned();
            r.output = Output::default();
            r.compiled = false;
            match f {
                Failure::InvalidOption(_) => NVRTC_ERROR_INVALID_OPTION,
                Failure::Compilation(_) => NVRTC_ERROR_COMPILATION,
                Failure::NoToolchain(_) => NVRTC_ERROR_BUILTIN_OPERATION_FAILURE,
            }
        }
    }
}

fn log_of(r: &Program) -> &str {
    if r.compiled { &r.output.log } else { &r.failure_log }
}

unsafe fn copy_text(text: &str, size: *mut usize, dst: *mut c_char) -> NvrtcResult {
    if dst.is_null() {
        if size.is_null() {
            return NVRTC_ERROR_INVALID_INPUT;
        }
        unsafe { *size = text.len() + 1 };
        return NVRTC_SUCCESS;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(text.as_ptr(), dst.cast::<u8>(), text.len());
        *dst.add(text.len()) = 0;
    }
    NVRTC_SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nvrtcGetProgramLogSize(p: NvrtcProgram, size: *mut usize) -> NvrtcResult {
    let Some(r) = (unsafe { prog(p) }) else { return NVRTC_ERROR_INVALID_PROGRAM };
    if size.is_null() {
        return NVRTC_ERROR_INVALID_INPUT;
    }
    unsafe { copy_text(log_of(r), size, std::ptr::null_mut()) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nvrtcGetProgramLog(p: NvrtcProgram, dst: *mut c_char) -> NvrtcResult {
    let Some(r) = (unsafe { prog(p) }) else { return NVRTC_ERROR_INVALID_PROGRAM };
    if dst.is_null() {
        return NVRTC_ERROR_INVALID_INPUT;
    }
    unsafe { copy_text(log_of(r), std::ptr::null_mut(), dst) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nvrtcGetPTXSize(p: NvrtcProgram, size: *mut usize) -> NvrtcResult {
    let Some(r) = (unsafe { prog(p) }) else { return NVRTC_ERROR_INVALID_PROGRAM };
    if size.is_null() || !r.compiled {
        return NVRTC_ERROR_INVALID_INPUT;
    }
    unsafe { copy_text(&r.output.ptx, size, std::ptr::null_mut()) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nvrtcGetPTX(p: NvrtcProgram, dst: *mut c_char) -> NvrtcResult {
    let Some(r) = (unsafe { prog(p) }) else { return NVRTC_ERROR_INVALID_PROGRAM };
    if dst.is_null() || !r.compiled {
        return NVRTC_ERROR_INVALID_INPUT;
    }
    unsafe { copy_text(&r.output.ptx, std::ptr::null_mut(), dst) }
}

/// There is no SASS assembler here. Like real NVRTC for `compute_XX` targets the CUBIN is empty,
/// except that for `sm_XX` targets the PTX text is returned (cuModuleLoadData in this project
/// accepts PTX text), so apps that prefer CUBIN for real architectures still work.
fn cubin_text(r: &Program, requested_sm: bool) -> &str {
    if requested_sm || std::env::var_os("NVRTC_SHIM_CUBIN_IS_PTX").is_some() { &r.output.ptx } else { "" }
}

fn wants_sm(r: &Program) -> bool {
    r.request.options.iter().any(|o| o.contains("sm_") && (o.contains("arch")))
        || r.request.options.windows(2).any(|w| (w[0] == "-arch" || w[0] == "--gpu-architecture") && w[1].starts_with("sm_"))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nvrtcGetCUBINSize(p: NvrtcProgram, size: *mut usize) -> NvrtcResult {
    let Some(r) = (unsafe { prog(p) }) else { return NVRTC_ERROR_INVALID_PROGRAM };
    if size.is_null() || !r.compiled {
        return NVRTC_ERROR_INVALID_INPUT;
    }
    let t = cubin_text(r, wants_sm(r));
    unsafe { *size = if t.is_empty() { 0 } else { t.len() + 1 } };
    NVRTC_SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nvrtcGetCUBIN(p: NvrtcProgram, dst: *mut c_char) -> NvrtcResult {
    let Some(r) = (unsafe { prog(p) }) else { return NVRTC_ERROR_INVALID_PROGRAM };
    if dst.is_null() || !r.compiled {
        return NVRTC_ERROR_INVALID_INPUT;
    }
    let t = cubin_text(r, wants_sm(r));
    if t.is_empty() { NVRTC_SUCCESS } else { unsafe { copy_text(t, std::ptr::null_mut(), dst) } }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nvrtcGetLTOIRSize(_p: NvrtcProgram, size: *mut usize) -> NvrtcResult {
    if !size.is_null() {
        unsafe { *size = 0 };
    }
    NVRTC_ERROR_INVALID_PROGRAM
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nvrtcGetLTOIR(_p: NvrtcProgram, _dst: *mut c_char) -> NvrtcResult {
    NVRTC_ERROR_INVALID_PROGRAM
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nvrtcGetLoweredName(p: NvrtcProgram, expr: *const c_char, lowered: *mut *const c_char) -> NvrtcResult {
    let Some(r) = (unsafe { prog(p) }) else { return NVRTC_ERROR_INVALID_PROGRAM };
    let Some(e) = (unsafe { cstr(expr) }) else { return NVRTC_ERROR_INVALID_INPUT };
    if lowered.is_null() {
        return NVRTC_ERROR_INVALID_INPUT;
    }
    if r.request.name_expressions.is_empty() {
        return NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION;
    }
    if !r.compiled {
        return NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION;
    }
    match r.lowered_c.iter().find(|(k, _)| *k == e) {
        Some((_, c)) => {
            unsafe { *lowered = c.as_ptr() };
            NVRTC_SUCCESS
        }
        None => NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID,
    }
}

// Deprecated / rarely used entry points kept link-compatible.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nvrtcSetFlowCallback(_p: NvrtcProgram, _cb: *const c_void, _payload: *mut c_void) -> NvrtcResult {
    NVRTC_SUCCESS
}

#[unsafe(no_mangle)]
pub extern "C" fn nvrtcSetPCHHeapSize(_size: usize) -> NvrtcResult {
    NVRTC_SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nvrtcGetPCHHeapSize(size: *mut usize) -> NvrtcResult {
    if size.is_null() {
        return NVRTC_ERROR_INVALID_INPUT;
    }
    unsafe { *size = 0 };
    NVRTC_SUCCESS
}
