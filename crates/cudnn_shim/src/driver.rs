//! Minimal Driver API access (nvcuda.dll / libcuda.so.1 = cuda-for-intel) and
//! the kernel cache used to launch the generated PTX.
use crate::ptxgen::{self, ConvKind, Elem};
use libloading::Library;
use std::collections::HashMap;
use std::ffi::{CString, c_char, c_int, c_void};
use std::sync::{Mutex, OnceLock};

type CuResult = c_int;
pub type CUstream = *mut c_void;
type CUcontext = *mut c_void;
type CUmodule = *mut c_void;
type CUfunction = *mut c_void;

#[allow(non_snake_case)]
struct Driver {
    _lib: Library,
    cuCtxGetCurrent: unsafe extern "system" fn(*mut CUcontext) -> CuResult,
    cuModuleLoadData: unsafe extern "system" fn(*mut CUmodule, *const c_void) -> CuResult,
    cuModuleGetFunction: unsafe extern "system" fn(*mut CUfunction, CUmodule, *const c_char) -> CuResult,
    cuLaunchKernel: unsafe extern "system" fn(
        CUfunction, u32, u32, u32, u32, u32, u32, u32, CUstream, *mut *mut c_void, *mut *mut c_void,
    ) -> CuResult,
}

// Load the shim's driver dependency with the DLL on Windows so the same driver is shared.
#[cfg(windows)]
#[link(name = "nvcuda", kind = "raw-dylib")]
unsafe extern "system" {
    #[allow(non_snake_case)]
    fn cuCtxGetCurrent(context: *mut CUcontext) -> CuResult;
}

fn driver() -> Option<&'static Driver> {
    static D: OnceLock<Option<Driver>> = OnceLock::new();
    D.get_or_init(|| unsafe {
        #[cfg(windows)]
        let lib = Library::new("nvcuda.dll").ok()?;
        #[cfg(not(windows))]
        let lib = Library::new("libcuda.so.1").ok()?;
        macro_rules! sym {
            ($n:literal) => {
                *lib.get(concat!($n, "\0").as_bytes()).ok()?
            };
        }
        Some(Driver {
            #[cfg(windows)]
            cuCtxGetCurrent,
            #[cfg(not(windows))]
            cuCtxGetCurrent: sym!("cuCtxGetCurrent"),
            cuModuleLoadData: sym!("cuModuleLoadData"),
            cuModuleGetFunction: sym!("cuModuleGetFunction"),
            cuLaunchKernel: sym!("cuLaunchKernel"),
            _lib: lib,
        })
    })
    .as_ref()
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Kid {
    Conv(ConvKind, Elem),
    Add(Elem),
    Relu(Elem),
    BnTrain(Elem),
    BnInfer(Elem),
    BnBwd(Elem),
}

impl Kid {
    pub fn name(self) -> String {
        match self {
            Kid::Conv(k, e) => ptxgen::conv_entry(k, e),
            Kid::Add(e) => ptxgen::add_tensor_entry(e),
            Kid::Relu(e) => ptxgen::relu_entry(e),
            Kid::BnTrain(e) => ptxgen::bn_train_entry(e),
            Kid::BnInfer(e) => ptxgen::bn_infer_entry(e),
            Kid::BnBwd(e) => ptxgen::bn_bwd_entry(e),
        }
    }
    pub fn ptx(self) -> String {
        match self {
            Kid::Conv(k, e) => ptxgen::generate_conv(k, e),
            Kid::Add(e) => ptxgen::generate_add_tensor(e),
            Kid::Relu(e) => ptxgen::generate_relu(e),
            Kid::BnTrain(e) => ptxgen::generate_bn_train(e),
            Kid::BnInfer(e) => ptxgen::generate_bn_infer(e),
            Kid::BnBwd(e) => ptxgen::generate_bn_bwd(e),
        }
    }
}

#[derive(Clone, Copy)]
struct Kernel(CUfunction);
unsafe impl Send for Kernel {}
unsafe impl Sync for Kernel {}

/// One kernel argument, 8 bytes of storage, passed by value.
#[derive(Clone, Copy, Debug)]
pub enum Arg {
    P(u64),
    F(f32),
    U(u32),
}

pub enum LaunchError {
    NoDriver,
    NoContext,
    Module(i32),
    Launch(i32),
}

fn kernel_for(drv: &Driver, kid: Kid) -> Result<Kernel, LaunchError> {
    static CACHE: OnceLock<Mutex<HashMap<(usize, Kid), Kernel>>> = OnceLock::new();
    let mut ctx: CUcontext = std::ptr::null_mut();
    if unsafe { (drv.cuCtxGetCurrent)(&mut ctx) } != 0 || ctx.is_null() {
        return Err(LaunchError::NoContext);
    }
    let mut cache = CACHE.get_or_init(Default::default).lock().unwrap();
    if let Some(k) = cache.get(&(ctx as usize, kid)) {
        return Ok(*k);
    }
    let ptx = format!("{}\0", kid.ptx());
    let mut module: CUmodule = std::ptr::null_mut();
    let err = unsafe { (drv.cuModuleLoadData)(&mut module, ptx.as_ptr().cast()) };
    if err != 0 {
        return Err(LaunchError::Module(err));
    }
    let cname = CString::new(kid.name()).unwrap();
    let mut f: CUfunction = std::ptr::null_mut();
    let err = unsafe { (drv.cuModuleGetFunction)(&mut f, module, cname.as_ptr()) };
    if err != 0 || f.is_null() {
        return Err(LaunchError::Module(err));
    }
    let k = Kernel(f);
    cache.insert((ctx as usize, kid), k);
    Ok(k)
}

/// Launch a 1-D grid kernel on `stream`.
pub fn launch(kid: Kid, grid: u32, block: u32, stream: CUstream, args: &[Arg]) -> Result<(), LaunchError> {
    let drv = driver().ok_or(LaunchError::NoDriver)?;
    let k = kernel_for(drv, kid)?;
    let mut store: Vec<u64> = args
        .iter()
        .map(|a| match a {
            Arg::P(p) => *p,
            Arg::F(f) => f.to_bits() as u64,
            Arg::U(u) => *u as u64,
        })
        .collect();
    // little-endian: the low 4 bytes of the u64 hold the 32-bit value
    let mut params: Vec<*mut c_void> = store.iter_mut().map(|v| v as *mut u64 as *mut c_void).collect();
    let err = unsafe {
        (drv.cuLaunchKernel)(k.0, grid, 1, 1, block, 1, 1, 0, stream, params.as_mut_ptr(), std::ptr::null_mut())
    };
    if err != 0 { Err(LaunchError::Launch(err)) } else { Ok(()) }
}
