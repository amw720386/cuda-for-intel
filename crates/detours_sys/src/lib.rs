//! The handful of Microsoft Detours entry points that `cfi_redirect` and `cfi-run` use.
//! Declarations follow `vendor/src/detours.h` (all `WINAPI`, which is the C ABI on x64).
#![cfg(windows)]
#![allow(non_snake_case)]

use std::ffi::c_void;

/// Windows `GUID`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Guid {
    pub data1: u32,
    pub data2: u16,
    pub data3: u16,
    pub data4: [u8; 8],
}

pub type Handle = *mut c_void;

// Linked by build.rs (cc emits `rustc-link-lib=static=detours`).
unsafe extern "system" {
    pub fn DetourTransactionBegin() -> i32;
    pub fn DetourTransactionAbort() -> i32;
    pub fn DetourTransactionCommit() -> i32;
    pub fn DetourUpdateThread(thread: Handle) -> i32;
    pub fn DetourAttach(pointer: *mut *mut c_void, detour: *mut c_void) -> i32;
    pub fn DetourRestoreAfterWith() -> i32;
    pub fn DetourUpdateProcessWithDll(process: Handle, dlls: *const *const u8, count: u32) -> i32;
    pub fn DetourCopyPayloadToProcess(process: Handle, guid: *const Guid, data: *const c_void, size: u32) -> i32;
    pub fn DetourFindPayloadEx(guid: *const Guid, size: *mut u32) -> *mut c_void;
}
