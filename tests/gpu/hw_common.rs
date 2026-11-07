//! Shared helpers for the hardware kernel validation tests (hw_*.rs).
use super::{Handle, library, ok, symbols, test_device};
use std::ffi::{c_char, c_void};
use std::ptr::{from_mut, null_mut};

pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn u32(&mut self) -> u32 {
        (self.next() >> 32) as u32
    }
    pub fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.u32() % (hi - lo + 1) as u32) as i32
    }
}

pub fn bytes_of_u32(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub fn u32s_of(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Exact binary16 conversion for values that are exactly representable
/// (round-to-nearest-even otherwise, subnormals flushed to the nearest multiple).
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127;
    let mant = bits & 0x7f_ffff;
    if value == 0.0 {
        return sign;
    }
    if exp > 15 {
        return sign | 0x7c00;
    }
    if exp >= -14 {
        let mut half = (((exp + 15) as u32) << 10) | (mant >> 13);
        let rest = mant & 0x1fff;
        if rest > 0x1000 || (rest == 0x1000 && half & 1 == 1) {
            half += 1;
        }
        return sign | half as u16;
    }
    let shift = (-14 - exp) as u32 + 13;
    if shift > 24 {
        return sign;
    }
    let full = mant | 0x80_0000;
    let mut half = full >> shift;
    let rest = full & ((1 << shift) - 1);
    let halfway = 1 << (shift - 1);
    if rest > halfway || (rest == halfway && half & 1 == 1) {
        half += 1;
    }
    sign | half as u16
}

pub fn f16_to_f32(half: u16) -> f32 {
    let sign = if half & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((half >> 10) & 0x1f) as i32;
    let mant = (half & 0x3ff) as f32;
    match exp {
        0 => sign * mant * 2f32.powi(-24),
        31 => {
            if mant == 0.0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => sign * (1.0 + mant / 1024.0) * 2f32.powi(exp - 15),
    }
}

/// Compile `ptx` and launch `entry` with `bufs` (device buffers, copied in and
/// back out) followed by u32 `scalars`. Returns Err(message) when module load fails.
#[allow(non_snake_case)]
pub fn launch(
    ptx: &str,
    entry: &str,
    grid: u32,
    block: u32,
    bufs: &mut [Vec<u8>],
    scalars: &[u32],
) -> Result<(), String> {
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuCtxDestroy_v2(Handle),
        cuModuleLoadData(*mut Handle, *const c_void),
        cuModuleGetFunction(*mut Handle, Handle, *const c_char),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemFree_v2(u64),
        cuMemcpyHtoD_v2(u64, *const c_void, usize),
        cuMemcpyDtoH_v2(*mut c_void, u64, usize),
        cuCtxSynchronize(),
        cuLaunchKernel(Handle, u32, u32, u32, u32, u32, u32, u32, Handle, *mut Handle, *mut Handle),
    );
    ok!(cuInit(0));
    let mut context = null_mut();
    ok!(cuCtxCreate_v2(&mut context, 0, test_device()));
    let source = std::ffi::CString::new(ptx).unwrap();
    let mut module = null_mut();
    let status = unsafe { cuModuleLoadData(&mut module, source.as_ptr().cast()) };
    if status != 0 {
        unsafe { cuCtxDestroy_v2(context) };
        return Err(format!("cuModuleLoadData({entry}) = {status}"));
    }
    let name = std::ffi::CString::new(entry).unwrap();
    let mut kernel = null_mut();
    let status = unsafe { cuModuleGetFunction(&mut kernel, module, name.as_ptr()) };
    if status != 0 {
        unsafe { cuCtxDestroy_v2(context) };
        return Err(format!("cuModuleGetFunction({entry}) = {status}"));
    }
    let mut devices = Vec::new();
    for buf in bufs.iter() {
        let mut dev = 0u64;
        ok!(cuMemAlloc_v2(&mut dev, buf.len().max(16)));
        ok!(cuMemcpyHtoD_v2(dev, buf.as_ptr().cast(), buf.len()));
        devices.push(dev);
    }
    let mut scalars = scalars.to_vec();
    let mut params: Vec<*mut c_void> = devices
        .iter_mut()
        .map(|d| from_mut(d).cast::<c_void>())
        .collect();
    params.extend(scalars.iter_mut().map(|s| from_mut(s).cast::<c_void>()));
    let status = unsafe {
        cuLaunchKernel(kernel, grid, 1, 1, block, 1, 1, 0, null_mut(), params.as_mut_ptr(), null_mut())
    };
    let sync = unsafe { cuCtxSynchronize() };
    if status != 0 || sync != 0 {
        unsafe { cuCtxDestroy_v2(context) };
        return Err(format!("launch({entry}) = {status}, sync = {sync}"));
    }
    for (buf, dev) in bufs.iter_mut().zip(&devices) {
        ok!(cuMemcpyDtoH_v2(buf.as_mut_ptr().cast(), *dev, buf.len()));
        ok!(cuMemFree_v2(*dev));
    }
    ok!(cuCtxDestroy_v2(context));
    Ok(())
}

/// PTX prologue declaring `n` u64 pointer params p0.. and loading them as global pointers %rd1..%rdn.
pub fn prologue(entry: &str, buffers: usize, scalars: usize, extra: &str) -> String {
    let mut s = format!(
        ".version 8.0\n.target sm_80\n.address_size 64\n.visible .entry {entry}(\n"
    );
    let mut params = Vec::new();
    for i in 0..buffers {
        params.push(format!("\t.param .u64 {entry}_p{i}"));
    }
    for i in 0..scalars {
        params.push(format!("\t.param .u32 {entry}_s{i}"));
    }
    s += &params.join(",\n");
    s += "\n)\n{\n";
    s += "\t.reg .pred %p<16>;\n\t.reg .b32 %r<128>;\n\t.reg .b64 %rd<32>;\n\t.reg .f32 %f<32>;\n";
    s += extra;
    for i in 0..buffers {
        s += &format!(
            "\tld.param.u64 %rd{1}, [{entry}_p{i}];\n\tcvta.to.global.u64 %rd{0}, %rd{1};\n",
            i + 1,
            i + 8
        );
    }
    for i in 0..scalars {
        s += &format!("\tld.param.u32 %r{}, [{entry}_s{i}];\n", 100 + i);
    }
    s
}
