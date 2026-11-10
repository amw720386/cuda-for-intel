//! A CUDA Driver API library backed by the SPIR-V simulator, so that vosen/ptx_tests (which
//! loads "a CUDA library" by path) runs our PTX translator on a machine without an Intel GPU.
//!
//! `cuModuleLoadData` translates the PTX with `src/ptx.rs` exactly as `nvcuda.dll` does
//! (`ptx::translate`), and `cuLaunchKernel` runs the SPIR-V on `src/ptx/sim.rs`. Device memory
//! is host memory: every launch copies the buffers its pointer arguments point into into a fresh
//! simulator heap and copies them back afterwards.
//!
//! `SIMCUDA_VERBOSE=1` prints every translation or simulation error to stderr; `2` also prints
//! each loaded PTX module.
#![allow(dead_code, clippy::missing_safety_doc)]

mod ptx;
mod sim_host;

use sim_host::sim::{Gpu, Launch, Sim};
use std::collections::{BTreeMap, HashMap};
use std::ffi::{CStr, c_char, c_uint, c_void};
use std::sync::Mutex;

const SUCCESS: u32 = 0;
const INVALID_VALUE: u32 = 1;
const OUT_OF_MEMORY: u32 = 2;
const INVALID_PTX: u32 = 218;
const INVALID_HANDLE: u32 = 400;
const NOT_FOUND: u32 = 500;
const LAUNCH_FAILED: u32 = 719;

/// Fake device addresses start here, far from the simulator's own address ranges.
const DEVICE_BASE: u64 = 0x7000_0000_0000;
const DEVICE_BYTES: usize = 8 << 30;

struct Function {
    module: usize,
    name: String,
    params: Vec<ptx::Param>,
}

#[derive(Default)]
struct State {
    next_handle: usize,
    modules: HashMap<usize, rspirv::dr::Module>,
    functions: HashMap<usize, Function>,
    /// base address -> bytes
    memory: BTreeMap<u64, Vec<u8>>,
    next_address: u64,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

fn verbose() -> bool {
    std::env::var_os("SIMCUDA_VERBOSE").is_some_and(|v| v != "0")
}

fn with_state<F: FnOnce(&mut State) -> u32>(f: F) -> u32 {
    let mut guard = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let state = guard.get_or_insert_with(|| State { next_handle: 1, next_address: DEVICE_BASE, ..Default::default() });
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(state))) {
        Ok(code) => code,
        Err(p) => {
            let msg = p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
            eprintln!("simcuda: panic: {msg}");
            LAUNCH_FAILED
        }
    }
}

impl State {
    fn handle(&mut self) -> usize {
        let h = self.next_handle;
        self.next_handle += 1;
        h
    }
    /// The allocation containing `addr`: (base, offset).
    fn find(&self, addr: u64) -> Option<(u64, u64)> {
        let (&base, bytes) = self.memory.range(..=addr).next_back()?;
        (addr < base + bytes.len().max(1) as u64).then_some((base, addr - base))
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn cuInit(_flags: c_uint) -> u32 {
    SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuCtxCreate_v2(pctx: *mut *mut c_void, _flags: c_uint, _dev: i32) -> u32 {
    if pctx.is_null() {
        return INVALID_VALUE;
    }
    unsafe { *pctx = 1 as *mut c_void };
    SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> u32 {
    let used: usize = with_state(|s| s.memory.values().map(Vec::len).sum::<usize>() as u32) as usize;
    unsafe {
        if !free.is_null() {
            *free = DEVICE_BYTES.saturating_sub(used);
        }
        if !total.is_null() {
            *total = DEVICE_BYTES;
        }
    }
    SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuModuleLoadData(module: *mut *mut c_void, image: *const c_void) -> u32 {
    if module.is_null() || image.is_null() {
        return INVALID_VALUE;
    }
    let text = match unsafe { CStr::from_ptr(image as *const c_char) }.to_str() {
        Ok(t) => t.to_owned(),
        Err(_) => return INVALID_PTX,
    };
    if std::env::var("SIMCUDA_VERBOSE").is_ok_and(|v| v == "2") {
        eprintln!("simcuda: module:\n{text}");
        if let Ok(prepared) = ptx::prepare_module(&text) {
            eprintln!("simcuda: prepared:\n{prepared}");
        }
    }
    with_state(|s| {
        let translated = match ptx::translate(&text) {
            Ok(t) => t,
            Err(e) => {
                if verbose() {
                    eprintln!("simcuda: translate: {e}");
                }
                return INVALID_PTX;
            }
        };
        let words = match rspirv::dr::load_words(&translated.words) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("simcuda: load SPIR-V: {e:?}");
                return INVALID_PTX;
            }
        };
        let h = s.handle();
        s.modules.insert(h, words);
        for k in translated.kernels {
            let f = s.handle();
            s.functions.insert(f, Function { module: h, name: k.name, params: k.params });
        }
        unsafe { *module = h as *mut c_void };
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuModuleGetFunction(hfunc: *mut *mut c_void, hmod: *mut c_void, name: *const c_char) -> u32 {
    if hfunc.is_null() || name.is_null() {
        return INVALID_VALUE;
    }
    let name = unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned();
    with_state(|s| {
        let m = hmod as usize;
        if !s.modules.contains_key(&m) {
            return INVALID_HANDLE;
        }
        match s.functions.iter().find(|(_, f)| f.module == m && f.name == name) {
            Some((&h, _)) => {
                unsafe { *hfunc = h as *mut c_void };
                SUCCESS
            }
            None => NOT_FOUND,
        }
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuModuleUnload(hmod: *mut c_void) -> u32 {
    with_state(|s| {
        let m = hmod as usize;
        if s.modules.remove(&m).is_none() {
            return INVALID_HANDLE;
        }
        s.functions.retain(|_, f| f.module != m);
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemAlloc_v2(dptr: *mut u64, bytes: usize) -> u32 {
    if dptr.is_null() {
        return INVALID_VALUE;
    }
    with_state(|s| {
        let used: usize = s.memory.values().map(Vec::len).sum();
        if used + bytes > DEVICE_BYTES {
            return OUT_OF_MEMORY;
        }
        let base = s.next_address;
        s.next_address = (base + bytes as u64 + 4096 + 255) & !255;
        s.memory.insert(base, vec![0; bytes]);
        unsafe { *dptr = base };
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuMemFree_v2(dptr: u64) -> u32 {
    with_state(|s| if s.memory.remove(&dptr).is_some() { SUCCESS } else { INVALID_VALUE })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemcpyHtoD_v2(dst: u64, src: *const c_void, bytes: usize) -> u32 {
    with_state(|s| {
        let Some((base, off)) = s.find(dst) else { return INVALID_VALUE };
        let buf = s.memory.get_mut(&base).unwrap();
        let off = off as usize;
        if off + bytes > buf.len() {
            return INVALID_VALUE;
        }
        let src = unsafe { std::slice::from_raw_parts(src as *const u8, bytes) };
        buf[off..off + bytes].copy_from_slice(src);
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemcpyDtoH_v2(dst: *mut c_void, src: u64, bytes: usize) -> u32 {
    with_state(|s| {
        let Some((base, off)) = s.find(src) else { return INVALID_VALUE };
        let buf = &s.memory[&base];
        let off = off as usize;
        if off + bytes > buf.len() {
            return INVALID_VALUE;
        }
        let dst = unsafe { std::slice::from_raw_parts_mut(dst as *mut u8, bytes) };
        dst.copy_from_slice(&buf[off..off + bytes]);
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuStreamSynchronize(_stream: *mut c_void) -> u32 {
    SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuLaunchKernel(
    f: *mut c_void,
    gx: c_uint,
    gy: c_uint,
    gz: c_uint,
    bx: c_uint,
    by: c_uint,
    bz: c_uint,
    _shared: c_uint,
    _stream: *mut c_void,
    params: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> u32 {
    if !extra.is_null() {
        return INVALID_VALUE;
    }
    with_state(|s| {
        let Some(func) = s.functions.get(&(f as usize)) else { return INVALID_HANDLE };
        let Some(module) = s.modules.get(&func.module) else { return INVALID_HANDLE };
        // Pointer arguments are rebased into a fresh simulator heap.
        let mut gpu = Gpu::default();
        let mut placed: BTreeMap<u64, u64> = BTreeMap::new();
        let mut args = Vec::with_capacity(func.params.len());
        for (i, p) in func.params.iter().enumerate() {
            match p {
                ptx::Param::U64 | ptx::Param::GlobalPtr => {}
                other => {
                    eprintln!("simcuda: unsupported parameter kind {other:?}");
                    return INVALID_VALUE;
                }
            }
            let value = unsafe { (*params.add(i) as *const u64).read_unaligned() };
            let arg = match s.find(value) {
                Some((base, off)) => {
                    let sim_base = *placed.entry(base).or_insert_with(|| gpu.upload(&s.memory[&base]));
                    sim_base + off
                }
                None => value,
            };
            args.push(arg);
        }
        let mut sim = Sim::new(module);
        let mut launch = Launch::new([gx, gy, gz], [bx, by, bz], args);
        launch.max_steps = u64::MAX;
        if let Err(e) = sim.run(&mut gpu, &launch) {
            if verbose() {
                eprintln!("simcuda: simulate {}: {e}", func.name);
            }
            return LAUNCH_FAILED;
        }
        for (base, sim_base) in placed {
            let len = s.memory[&base].len();
            let data = gpu.download(sim_base, len);
            s.memory.insert(base, data);
        }
        SUCCESS
    })
}
