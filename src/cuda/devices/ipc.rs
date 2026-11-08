//! CUDA IPC. Level Zero can export a memory handle (`zeMemGetIpcHandle`), but only to
//! a process that shares the driver and a context, which a CUDA application cannot
//! arrange through this API, so every entry point answers `CUDA_ERROR_NOT_SUPPORTED`
//! after validating its arguments the way CUDA does.

use super::super::*;

/// `CUipcMemHandle` / `CUipcEventHandle`: 64 opaque bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IpcHandle {
    reserved: [u8; 64],
}

impl IpcHandle {
    #[cfg(test)]
    pub(crate) fn zeroed() -> Self {
        Self { reserved: [0; 64] }
    }
}

fn device_pointer(pointer: u64) -> Result<(), CuResult> {
    state(|state| {
        let address = pointer as usize;
        state
            .contexts
            .values()
            .any(|c| contains_range(&c.allocations, address, 1))
            .then_some(())
            .ok_or(InvalidValue)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuIpcGetMemHandle(handle: *mut IpcHandle, pointer: u64) -> CuResult {
    trace::call("cuIpcGetMemHandle", || {
        if handle.is_null() {
            return InvalidValue;
        }
        status(device_pointer(pointer).and(Err(NotSupported)))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuIpcOpenMemHandle_v2(
    pointer: *mut u64,
    _handle: IpcHandle,
    flags: c_uint,
) -> CuResult {
    trace::call("cuIpcOpenMemHandle_v2", || {
        // CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS is the only defined flag.
        if pointer.is_null() || flags & !1 != 0 {
            return InvalidValue;
        }
        status(state(|_| Err(NotSupported)))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuIpcCloseMemHandle(_pointer: u64) -> CuResult {
    trace::call("cuIpcCloseMemHandle", || {
        status(state(|_| Err(NotSupported)))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuIpcGetEventHandle(handle: *mut IpcHandle, event: Opaque) -> CuResult {
    trace::call("cuIpcGetEventHandle", || {
        if handle.is_null() || event.is_null() {
            return InvalidValue;
        }
        status(state(|_| Err(NotSupported)))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuIpcOpenEventHandle(event: *mut Opaque, _handle: IpcHandle) -> CuResult {
    trace::call("cuIpcOpenEventHandle", || {
        if event.is_null() {
            return InvalidValue;
        }
        status(state(|_| Err(NotSupported)))
    })
}
