//! CUDA arrays, texture objects and surface objects, emulated on linear device memory.
//!
//! A `CUarray` is a dense allocation (`pitch = width * element bytes`, no padding) whose handle is
//! its device address. A `CUtexObject` / `CUsurfObject` is the device address of a 128-byte
//! descriptor (`ptx/texdesc.rs`) that the translated `tex`/`suld`/`sust`/`txq`/`suq` code reads
//! (`ptx/tier3_tex.rs`). Supported: 1D, 2D and 3D arrays of 1, 2 or 4 channels of
//! u8/u16/u32/s8/s16/s32/f16/f32; texture objects over arrays, linear memory and pitched 2D
//! memory with every address mode, nearest and linear filtering, normalised coordinates, read
//! as integer and border colours; surface objects over arrays. Not supported (NOT_SUPPORTED):
//! mipmapped, layered, cubemap and sparse arrays, resource views that change the format or
//! shape, sRGB conversion, and the texture-reference API.

use super::*;
use super::gaps::{cuCtxPopCurrent_v2, cuCtxPushCurrent_v2};
use crate::ptx::texdesc::{self, Desc, Kind};

const LAYERED: u32 = 0x01;
const CUBEMAP: u32 = 0x04;
const SPARSE: u32 = 0x40;
const DEFERRED_MAPPING: u32 = 0x80;

#[repr(C)]
pub(crate) struct ArrayDescriptor {
    width: usize,
    height: usize,
    format: c_uint,
    channels: c_uint,
}

#[repr(C)]
pub(crate) struct Array3DDescriptor {
    width: usize,
    height: usize,
    depth: usize,
    format: c_uint,
    channels: c_uint,
    flags: c_uint,
}

/// `CUDA_RESOURCE_DESC`: the type, a 128-byte union and flags.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct ResourceDesc {
    res_type: c_uint,
    res: [u64; 16],
    flags: c_uint,
}

/// `CUDA_TEXTURE_DESC`
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct TextureDesc {
    address_mode: [c_uint; 3],
    filter_mode: c_uint,
    flags: c_uint,
    max_anisotropy: c_uint,
    mipmap_filter_mode: c_uint,
    mipmap_level_bias: f32,
    min_mipmap_level_clamp: f32,
    max_mipmap_level_clamp: f32,
    border_color: [f32; 4],
    reserved: [c_int; 12],
}

/// `CUDA_RESOURCE_VIEW_DESC`
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct ResourceViewDesc {
    format: c_uint,
    width: usize,
    height: usize,
    depth: usize,
    first_mipmap_level: c_uint,
    last_mipmap_level: c_uint,
    first_layer: c_uint,
    last_layer: c_uint,
    reserved: [c_uint; 16],
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ArrayInfo {
    pub base: u64,
    pub width: usize,
    /// As created: 0 for 1D, and depth 0 for 1D/2D.
    pub height: usize,
    pub depth: usize,
    pub format: c_uint,
    pub channels: c_uint,
    pub flags: c_uint,
    pub kind: Kind,
}

impl ArrayInfo {
    pub(crate) fn element_bytes(&self) -> usize {
        self.kind.bytes() as usize * self.channels as usize
    }
    pub(crate) fn pitch(&self) -> usize {
        self.width * self.element_bytes()
    }
    pub(crate) fn rows(&self) -> usize {
        self.height.max(1)
    }
    pub(crate) fn bytes(&self) -> usize {
        self.pitch() * self.rows() * self.depth.max(1)
    }
}

struct Object {
    resource: ResourceDesc,
    texture: Option<TextureDesc>,
    view: Option<ResourceViewDesc>,
    context: usize,
}

// SAFETY: plain data (handles are integers).
unsafe impl Send for Object {}

static ARRAYS: Mutex<BTreeMap<u64, ArrayInfo>> = Mutex::new(BTreeMap::new());
static OBJECTS: Mutex<BTreeMap<u64, Object>> = Mutex::new(BTreeMap::new());

fn arrays() -> std::sync::MutexGuard<'static, BTreeMap<u64, ArrayInfo>> {
    ARRAYS.lock().unwrap_or_else(|p| p.into_inner())
}

fn objects() -> std::sync::MutexGuard<'static, BTreeMap<u64, Object>> {
    OBJECTS.lock().unwrap_or_else(|p| p.into_inner())
}

/// Array behind a handle.
pub(crate) fn array(handle: Opaque) -> Option<ArrayInfo> {
    arrays().get(&(handle.addr() as u64)).copied()
}

fn current_context() -> Result<usize, CuResult> {
    let mut ctx: Opaque = std::ptr::null_mut();
    let rc = unsafe { cuCtxGetCurrent(&mut ctx) };
    if rc != Success {
        return Err(rc);
    }
    if ctx.is_null() { Err(InvalidContext) } else { Ok(ctx.addr()) }
}

/// Frees `address` with `context` current (the creating context owns the allocation).
fn free_in(context: usize, address: u64) -> CuResult {
    let rc = cuCtxPushCurrent_v2(std::ptr::without_provenance_mut(context));
    if rc != Success {
        return rc;
    }
    let freed = cuMemFree_v2(address);
    let mut popped: Opaque = std::ptr::null_mut();
    let _ = unsafe { cuCtxPopCurrent_v2(&mut popped) };
    freed
}

fn create_array(width: usize, height: usize, depth: usize, format: c_uint, channels: c_uint, flags: c_uint) -> Result<u64, CuResult> {
    let kind = Kind::from_cu(format).ok_or(InvalidValue)?;
    if width == 0 || !matches!(channels, 1 | 2 | 4) || (depth > 0 && height == 0) {
        return Err(InvalidValue);
    }
    if flags & (LAYERED | CUBEMAP | SPARSE | DEFERRED_MAPPING) != 0 {
        return Err(NotSupported);
    }
    let info = ArrayInfo { base: 0, width, height, depth, format, channels, flags, kind };
    let bytes = info.width.checked_mul(info.element_bytes()).and_then(|p| p.checked_mul(info.rows())).and_then(|p| p.checked_mul(depth.max(1))).ok_or(OutOfMemory)?;
    if u32::try_from(width).is_err() || u32::try_from(info.rows()).is_err() || u32::try_from(depth.max(1)).is_err() {
        return Err(InvalidValue);
    }
    let mut base = 0u64;
    let rc = unsafe { cuMemAlloc_v2(&mut base, bytes) };
    if rc != Success {
        return Err(rc);
    }
    arrays().insert(base, ArrayInfo { base, ..info });
    Ok(base)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuArrayCreate_v2(out: *mut Opaque, desc: *const ArrayDescriptor) -> CuResult {
    trace::call("cuArrayCreate_v2", || {
        let (Some(out), Some(d)) = (unsafe { out.as_mut() }, unsafe { desc.as_ref() }) else { return InvalidValue };
        match create_array(d.width, d.height, 0, d.format, d.channels, 0) {
            Ok(h) => {
                *out = std::ptr::without_provenance_mut(h as usize);
                Success
            }
            Err(e) => e,
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuArray3DCreate_v2(out: *mut Opaque, desc: *const Array3DDescriptor) -> CuResult {
    trace::call("cuArray3DCreate_v2", || {
        let (Some(out), Some(d)) = (unsafe { out.as_mut() }, unsafe { desc.as_ref() }) else { return InvalidValue };
        match create_array(d.width, d.height, d.depth, d.format, d.channels, d.flags) {
            Ok(h) => {
                *out = std::ptr::without_provenance_mut(h as usize);
                Success
            }
            Err(e) => e,
        }
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuArrayDestroy(handle: Opaque) -> CuResult {
    trace::call("cuArrayDestroy", || {
        let Some(info) = arrays().remove(&(handle.addr() as u64)) else { return InvalidHandle };
        cuMemFree_v2(info.base)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuArrayGetDescriptor_v2(out: *mut ArrayDescriptor, handle: Opaque) -> CuResult {
    trace::call("cuArrayGetDescriptor_v2", || {
        let Some(out) = (unsafe { out.as_mut() }) else { return InvalidValue };
        let Some(a) = array(handle) else { return InvalidHandle };
        *out = ArrayDescriptor { width: a.width, height: a.height, format: a.format, channels: a.channels };
        Success
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuArray3DGetDescriptor_v2(out: *mut Array3DDescriptor, handle: Opaque) -> CuResult {
    trace::call("cuArray3DGetDescriptor_v2", || {
        let Some(out) = (unsafe { out.as_mut() }) else { return InvalidValue };
        let Some(a) = array(handle) else { return InvalidHandle };
        *out = Array3DDescriptor { width: a.width, height: a.height, depth: a.depth, format: a.format, channels: a.channels, flags: a.flags };
        Success
    })
}

/// Device address of `[offset, offset + bytes)` inside an array (row-major, dense).
fn array_range(handle: Opaque, offset: usize, bytes: usize) -> Result<u64, CuResult> {
    let a = array(handle).ok_or(InvalidHandle)?;
    let end = offset.checked_add(bytes).ok_or(InvalidValue)?;
    if end > a.bytes() {
        return Err(InvalidValue);
    }
    Ok(a.base + offset as u64)
}

fn status_of(r: Result<u64, CuResult>, then: impl FnOnce(u64) -> CuResult) -> CuResult {
    match r {
        Ok(v) => then(v),
        Err(e) => e,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemcpyHtoA_v2(dst: Opaque, offset: usize, src: *const c_void, bytes: usize) -> CuResult {
    trace::call("cuMemcpyHtoA_v2", || status_of(array_range(dst, offset, bytes), |d| unsafe { cuMemcpyHtoD_v2(d, src, bytes) }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemcpyAtoH_v2(dst: Opaque, src: Opaque, offset: usize, bytes: usize) -> CuResult {
    trace::call("cuMemcpyAtoH_v2", || status_of(array_range(src, offset, bytes), |s| unsafe { cuMemcpyDtoH_v2(dst, s, bytes) }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemcpyHtoAAsync_v2(dst: Opaque, offset: usize, src: *const c_void, bytes: usize, stream: Opaque) -> CuResult {
    trace::call("cuMemcpyHtoAAsync_v2", || {
        status_of(array_range(dst, offset, bytes), |d| unsafe { cuMemcpyHtoDAsync_v2(d, src, bytes, stream) })
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemcpyAtoHAsync_v2(dst: Opaque, src: Opaque, offset: usize, bytes: usize, stream: Opaque) -> CuResult {
    trace::call("cuMemcpyAtoHAsync_v2", || {
        status_of(array_range(src, offset, bytes), |s| unsafe { cuMemcpyDtoHAsync_v2(dst, s, bytes, stream) })
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemcpyDtoA_v2(dst: Opaque, offset: usize, src: u64, bytes: usize) -> CuResult {
    trace::call("cuMemcpyDtoA_v2", || status_of(array_range(dst, offset, bytes), |d| unsafe { cuMemcpyDtoD_v2(d, src, bytes) }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemcpyAtoD_v2(dst: u64, src: Opaque, offset: usize, bytes: usize) -> CuResult {
    trace::call("cuMemcpyAtoD_v2", || status_of(array_range(src, offset, bytes), |s| unsafe { cuMemcpyDtoD_v2(dst, s, bytes) }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemcpyAtoA_v2(dst: Opaque, dst_offset: usize, src: Opaque, src_offset: usize, bytes: usize) -> CuResult {
    trace::call("cuMemcpyAtoA_v2", || {
        status_of(array_range(dst, dst_offset, bytes), |d| status_of(array_range(src, src_offset, bytes), |s| unsafe { cuMemcpyDtoD_v2(d, s, bytes) }))
    })
}

const RES_ARRAY: c_uint = 0;
const RES_MIPMAPPED: c_uint = 1;
const RES_LINEAR: c_uint = 2;
const RES_PITCH2D: c_uint = 3;

/// The descriptor for a resource (texture fields left at their defaults).
fn resource_desc(r: &ResourceDesc) -> Result<Desc, CuResult> {
    let lo = |i: usize| r.res[i] as u32;
    let hi = |i: usize| (r.res[i] >> 32) as u32;
    let mut d = Desc { resource_type: r.res_type, ..Desc::default() };
    match r.res_type {
        RES_ARRAY => {
            let a = array(std::ptr::without_provenance_mut(r.res[0] as usize)).ok_or(InvalidHandle)?;
            d.base = a.base;
            d.width = a.width as u32;
            d.height = a.rows() as u32;
            d.depth = a.depth.max(1) as u32;
            d.dims = if a.depth > 0 { 3 } else if a.height > 0 { 2 } else { 1 };
            d.kind = a.kind as u32;
            d.channels = a.channels;
            d.cu_format = a.format;
            d.pitch = a.pitch() as u64;
            d.slice = (a.pitch() * a.rows()) as u64;
        }
        RES_LINEAR | RES_PITCH2D => {
            let format = lo(1);
            let channels = hi(1);
            let kind = Kind::from_cu(format).ok_or(InvalidValue)?;
            if r.res[0] == 0 || !matches!(channels, 1 | 2 | 4) {
                return Err(InvalidValue);
            }
            d.base = r.res[0];
            d.kind = kind as u32;
            d.channels = channels;
            d.cu_format = format;
            let esz = u64::from(d.element_bytes());
            if r.res_type == RES_LINEAR {
                let elements = r.res[2] / esz;
                d.width = u32::try_from(elements).map_err(|_| InvalidValue)?;
                d.height = 1;
                d.depth = 1;
                d.dims = 1;
                d.linear = true;
                d.pitch = r.res[2];
                d.slice = r.res[2];
            } else {
                let (width, height, pitch) = (r.res[2], r.res[3], r.res[4]);
                if width == 0 || height == 0 || pitch < width * esz {
                    return Err(InvalidValue);
                }
                d.width = u32::try_from(width).map_err(|_| InvalidValue)?;
                d.height = u32::try_from(height).map_err(|_| InvalidValue)?;
                d.depth = 1;
                d.dims = 2;
                d.pitch = pitch;
                d.slice = pitch * height;
            }
            if d.width == 0 {
                return Err(InvalidValue);
            }
        }
        RES_MIPMAPPED => return Err(NotSupported),
        _ => return Err(InvalidValue),
    }
    Ok(d)
}

/// Uploads `desc` into a fresh 128-byte allocation and registers the object.
fn publish(desc: &Desc, object: Object) -> Result<u64, CuResult> {
    let mut address = 0u64;
    let rc = unsafe { cuMemAlloc_v2(&mut address, texdesc::SIZE) };
    if rc != Success {
        return Err(rc);
    }
    let bytes = desc.encode();
    let rc = unsafe { cuMemcpyHtoD_v2(address, bytes.as_ptr().cast(), bytes.len()) };
    if rc != Success {
        let _ = cuMemFree_v2(address);
        return Err(rc);
    }
    objects().insert(address, object);
    Ok(address)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuTexObjectCreate(
    out: *mut u64,
    resource: *const ResourceDesc,
    texture: *const TextureDesc,
    view: *const ResourceViewDesc,
) -> CuResult {
    trace::call("cuTexObjectCreate", || {
        let (Some(out), Some(r), Some(t)) = (unsafe { out.as_mut() }, unsafe { resource.as_ref() }, unsafe { texture.as_ref() }) else {
            return InvalidValue;
        };
        let view = unsafe { view.as_ref() }.copied();
        // A view may only restate the resource (no format change, no sub-range, no mip levels).
        if let Some(v) = &view
            && (v.format != 0 || v.first_mipmap_level != 0 || v.last_mipmap_level != 0 || v.first_layer != 0 || v.last_layer != 0)
        {
            return NotSupported;
        }
        let result = (|| {
            let mut d = resource_desc(r)?;
            if t.address_mode.iter().any(|m| *m > 3) || t.filter_mode > 1 {
                return Err(InvalidValue);
            }
            // sRGB (0x10) is not converted; the other flags do not change 2D/3D sampling here.
            if t.flags & 0x10 != 0 {
                return Err(NotSupported);
            }
            d.address_mode = t.address_mode;
            d.filter = t.filter_mode;
            d.flags = t.flags & (texdesc::READ_AS_INTEGER | texdesc::NORMALIZED_COORDINATES);
            d.border = t.border_color;
            // CUDA: 32-bit integer channels and READ_AS_INTEGER cannot be linearly filtered.
            let integer = matches!(Kind::from_cu(d.cu_format), Some(Kind::U32 | Kind::S32))
                || (d.flags & texdesc::READ_AS_INTEGER != 0 && !matches!(Kind::from_cu(d.cu_format), Some(Kind::F16 | Kind::F32)));
            if integer && d.filter == 1 {
                return Err(InvalidValue);
            }
            let context = current_context()?;
            publish(&d, Object { resource: *r, texture: Some(*t), view, context })
        })();
        match result {
            Ok(h) => {
                *out = h;
                Success
            }
            Err(e) => e,
        }
    })
}

fn destroy_object(handle: u64, surface: bool) -> CuResult {
    let mut map = objects();
    match map.get(&handle) {
        Some(o) if o.texture.is_some() != surface => {}
        _ => return InvalidValue,
    }
    let o = map.remove(&handle).expect("checked above");
    drop(map);
    free_in(o.context, handle)
}

#[unsafe(no_mangle)]
pub extern "system" fn cuTexObjectDestroy(handle: u64) -> CuResult {
    trace::call("cuTexObjectDestroy", || destroy_object(handle, false))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuTexObjectGetResourceDesc(out: *mut ResourceDesc, handle: u64) -> CuResult {
    trace::call("cuTexObjectGetResourceDesc", || {
        let Some(out) = (unsafe { out.as_mut() }) else { return InvalidValue };
        match objects().get(&handle) {
            Some(o) if o.texture.is_some() => {
                *out = o.resource;
                Success
            }
            _ => InvalidValue,
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuTexObjectGetTextureDesc(out: *mut TextureDesc, handle: u64) -> CuResult {
    trace::call("cuTexObjectGetTextureDesc", || {
        let Some(out) = (unsafe { out.as_mut() }) else { return InvalidValue };
        match objects().get(&handle).and_then(|o| o.texture) {
            Some(t) => {
                *out = t;
                Success
            }
            None => InvalidValue,
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuTexObjectGetResourceViewDesc(out: *mut ResourceViewDesc, handle: u64) -> CuResult {
    trace::call("cuTexObjectGetResourceViewDesc", || {
        let Some(out) = (unsafe { out.as_mut() }) else { return InvalidValue };
        let map = objects();
        let Some(o) = map.get(&handle).filter(|o| o.texture.is_some()) else { return InvalidValue };
        // No view given: the view is the whole resource in its own format.
        *out = o.view.unwrap_or(ResourceViewDesc {
            format: 0,
            width: 0,
            height: 0,
            depth: 0,
            first_mipmap_level: 0,
            last_mipmap_level: 0,
            first_layer: 0,
            last_layer: 0,
            reserved: [0; 16],
        });
        Success
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuSurfObjectCreate(out: *mut u64, resource: *const ResourceDesc) -> CuResult {
    trace::call("cuSurfObjectCreate", || {
        let (Some(out), Some(r)) = (unsafe { out.as_mut() }, unsafe { resource.as_ref() }) else { return InvalidValue };
        if r.res_type != RES_ARRAY {
            return InvalidValue;
        }
        let result = resource_desc(r).and_then(|d| {
            let context = current_context()?;
            publish(&d, Object { resource: *r, texture: None, view: None, context })
        });
        match result {
            Ok(h) => {
                *out = h;
                Success
            }
            Err(e) => e,
        }
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuSurfObjectDestroy(handle: u64) -> CuResult {
    trace::call("cuSurfObjectDestroy", || destroy_object(handle, true))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuSurfObjectGetResourceDesc(out: *mut ResourceDesc, handle: u64) -> CuResult {
    trace::call("cuSurfObjectGetResourceDesc", || {
        let Some(out) = (unsafe { out.as_mut() }) else { return InvalidValue };
        match objects().get(&handle) {
            Some(o) if o.texture.is_none() => {
                *out = o.resource;
                Success
            }
            _ => InvalidValue,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_sizes_match_cuda_headers() {
        assert_eq!(std::mem::size_of::<ArrayDescriptor>(), 24);
        assert_eq!(std::mem::size_of::<Array3DDescriptor>(), 40);
        assert_eq!(std::mem::size_of::<ResourceDesc>(), 144);
        assert_eq!(std::mem::offset_of!(ResourceDesc, res), 8);
        assert_eq!(std::mem::offset_of!(ResourceDesc, flags), 136);
        assert_eq!(std::mem::size_of::<TextureDesc>(), 104);
        assert_eq!(std::mem::offset_of!(TextureDesc, border_color), 40);
        assert_eq!(std::mem::size_of::<ResourceViewDesc>(), 112);
    }

    #[test]
    fn linear_and_pitch_resources_describe_their_memory() {
        let mut r = ResourceDesc { res_type: RES_LINEAR, res: [0; 16], flags: 0 };
        r.res[0] = 0x1000;
        r.res[1] = 0x20 | (4u64 << 32); // float4
        r.res[2] = 160;
        let d = resource_desc(&r).unwrap();
        assert_eq!((d.width, d.dims, d.linear, d.element_bytes()), (10, 1, true, 16));
        r.res_type = RES_PITCH2D;
        r.res[1] = 0x01 | (1u64 << 32); // uchar1
        r.res[2] = 30;
        r.res[3] = 7;
        r.res[4] = 64;
        let d = resource_desc(&r).unwrap();
        assert_eq!((d.width, d.height, d.pitch, d.slice, d.dims), (30, 7, 64, 448, 2));
        r.res[4] = 16;
        assert_eq!(resource_desc(&r).unwrap_err(), InvalidValue, "pitch below the row size");
        r.res_type = RES_MIPMAPPED;
        assert_eq!(resource_desc(&r).unwrap_err(), NotSupported);
    }
}
