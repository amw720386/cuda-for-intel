//! Resolve the WDDM adapter LUID so CUDA Runtime can match our device to DXGI.
#![cfg(windows)]

use libloading::os::windows::{LOAD_LIBRARY_SEARCH_SYSTEM32, Library};
use std::ffi::c_void;
use std::ptr;

type HRESULT = i32;
const DXGI_ERROR_NOT_FOUND: HRESULT = 0x887A0002u32 as HRESULT;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Luid {
    low_part: u32,
    high_part: i32,
}

#[repr(C)]
struct Guid {
    data1: u32,
    data2: u16,
    data3: u16,
    data4: [u8; 8],
}

#[repr(C)]
struct DxgiAdapterDesc {
    description: [u16; 128],
    vendor_id: u32,
    device_id: u32,
    sub_sys_id: u32,
    revision: u32,
    dedicated_video_memory: usize,
    dedicated_system_memory: usize,
    shared_system_memory: usize,
    adapter_luid: Luid,
}

#[repr(C)]
struct DxgiAdapterVtbl {
    _qi: *mut c_void,
    _add_ref: *mut c_void,
    release: unsafe extern "system" fn(*mut c_void) -> u32,
    _set_private: *mut c_void,
    _set_private_iface: *mut c_void,
    _get_private: *mut c_void,
    _get_parent: *mut c_void,
    _enum_outputs: *mut c_void,
    get_desc: unsafe extern "system" fn(*mut c_void, *mut DxgiAdapterDesc) -> HRESULT,
}

#[repr(C)]
struct DxgiFactoryVtbl {
    _qi: *mut c_void,
    _add_ref: *mut c_void,
    release: unsafe extern "system" fn(*mut c_void) -> u32,
    _set_private: *mut c_void,
    _set_private_iface: *mut c_void,
    _get_private: *mut c_void,
    _get_parent: *mut c_void,
    enum_adapters: unsafe extern "system" fn(*mut c_void, u32, *mut *mut c_void) -> HRESULT,
}

#[repr(C)]
struct DxgiObject {
    vtbl: *const c_void,
}

unsafe fn vtbl<T>(obj: *mut c_void) -> *const T {
    unsafe { (*(obj as *mut DxgiObject)).vtbl as *const T }
}

fn luid_bytes(luid: Luid) -> [u8; 8] {
    let mut bytes = [0u8; 8];
    bytes[0..4].copy_from_slice(&luid.low_part.to_le_bytes());
    bytes[4..8].copy_from_slice(&luid.high_part.to_le_bytes());
    bytes
}

/// Return the adapter's LUID and, for an exact PCI ID match, its subsystem ID.
pub fn adapter_identity(vendor_id: u32, device_id: u32) -> Option<([u8; 8], Option<u32>)> {
    type CreateFactory = unsafe extern "system" fn(*const Guid, *mut *mut c_void) -> HRESULT;
    // IID_IDXGIFactory = {7b7166ec-21c7-44ae-b21a-c9ae321ae369}
    const IID_FACTORY: Guid = Guid {
        data1: 0x7b7166ec,
        data2: 0x21c7,
        data3: 0x44ae,
        data4: [0xb2, 0x1a, 0xc9, 0xae, 0x32, 0x1a, 0xe3, 0x69],
    };

    unsafe {
        let module = Library::load_with_flags("dxgi.dll", LOAD_LIBRARY_SEARCH_SYSTEM32).ok()?;
        let create = module.get::<CreateFactory>(b"CreateDXGIFactory").ok()?;
        let mut factory = ptr::null_mut();
        if create(&IID_FACTORY, &mut factory) < 0 || factory.is_null() {
            return None;
        }

        let factory_vtbl = &*vtbl::<DxgiFactoryVtbl>(factory);
        let mut index = 0u32;
        let mut exact = None;
        let mut fallback = None;
        loop {
            let mut adapter = ptr::null_mut();
            let hr = (factory_vtbl.enum_adapters)(factory, index, &mut adapter);
            if hr == DXGI_ERROR_NOT_FOUND || adapter.is_null() {
                break;
            }
            let adapter_vtbl = &*vtbl::<DxgiAdapterVtbl>(adapter);
            let mut desc = std::mem::zeroed::<DxgiAdapterDesc>();
            if (adapter_vtbl.get_desc)(adapter, &mut desc) >= 0 {
                let bytes = luid_bytes(desc.adapter_luid);
                // Microsoft Basic Render Driver is vendor 0x1414.
                let software = desc.vendor_id == 0x1414;
                if desc.vendor_id == vendor_id && desc.device_id == device_id {
                    exact = Some((bytes, Some(desc.sub_sys_id)));
                } else if !software && desc.vendor_id == vendor_id && fallback.is_none() {
                    fallback = Some(bytes);
                } else if !software && fallback.is_none() {
                    fallback = Some(bytes);
                }
            }
            (adapter_vtbl.release)(adapter);
            if exact.is_some() {
                break;
            }
            index = index.saturating_add(1);
            if index > 64 {
                break;
            }
        }
        (factory_vtbl.release)(factory);
        exact.or_else(|| fallback.map(|luid| (luid, None)))
    }
}
