use super::*;

#[derive(Default)]
pub(super) struct Primary {
    pub(super) key: Option<usize>,
    retains: usize,
}

impl State {
    fn primary(&mut self, ordinal: c_int) -> Result<&mut Primary, CuResult> {
        self.runtime
            .devices
            .get(ordinal as usize)
            .ok_or(InvalidDevice)?;
        Ok(self.primary.entry(ordinal).or_default())
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuDevicePrimaryCtxRetain(
    out: *mut Opaque,
    ordinal: c_int,
) -> CuResult {
    super::export_log(format!("cuDevicePrimaryCtxRetain ordinal={ordinal} enter"));
    let result = trace::call("cuDevicePrimaryCtxRetain", || {
        if out.is_null() {
            return InvalidValue;
        }
        status(state(|state| {
            let primary = state.primary(ordinal)?;
            let retains = primary.retains.checked_add(1).ok_or(OutOfMemory)?;
            let key = match primary.key {
                Some(key) => {
                    state.contexts[&key].check_live()?;
                    key
                }
                None => state.create_context(ordinal)?,
            };
            state.primary.insert(
                ordinal,
                Primary {
                    key: Some(key),
                    retains,
                },
            );
            unsafe { out.write(std::ptr::without_provenance_mut(key)) };
            Ok(())
        }))
    });
    super::export_log(format!(
        "cuDevicePrimaryCtxRetain ordinal={ordinal} -> {}",
        result as i32
    ));
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuDevicePrimaryCtxGetState(
    ordinal: c_int,
    flags: *mut c_uint,
    active: *mut c_int,
) -> CuResult {
    trace::call("cuDevicePrimaryCtxGetState", || {
        if flags.is_null() || active.is_null() {
            return InvalidValue;
        }
        status(state(|state| {
            let primary = state.primary(ordinal)?;
            unsafe {
                flags.write(0);
                active.write(c_int::from(primary.key.is_some()));
            }
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuDevicePrimaryCtxSetFlags_v2(ordinal: c_int, flags: c_uint) -> CuResult {
    trace::call("cuDevicePrimaryCtxSetFlags_v2", || {
        set_flags(ordinal, flags, false)
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuDevicePrimaryCtxSetFlags(ordinal: c_int, flags: c_uint) -> CuResult {
    trace::call("cuDevicePrimaryCtxSetFlags", || {
        set_flags(ordinal, flags, true)
    })
}

fn set_flags(ordinal: c_int, flags: c_uint, legacy: bool) -> CuResult {
    status(state(|state| {
        let primary = state.primary(ordinal)?;
        if flags & !0x1ff != 0 || !matches!(flags & 7, 0 | 1 | 2 | 4) {
            return Err(InvalidValue);
        }
        if legacy && primary.key.is_some() {
            return Err(PrimaryContextActive);
        }
        // Only the automatic scheduling policy with no optional flags is implemented.
        if flags != 0 {
            return Err(NotSupported);
        }
        Ok(())
    }))
}

fn retire(ordinal: c_int, reset: bool) -> CuResult {
    finish_destroy(state(|state| {
        let primary = state.primary(ordinal)?;
        if !reset && primary.retains == 0 {
            return Err(InvalidContext);
        }
        if !reset && primary.retains > 1 {
            primary.retains -= 1;
            return Ok((Ok(()), None, None));
        }

        let teardown = if let Some(key) = primary.key {
            let teardown = state.destroy_context(key)?;
            // Failed partial teardown keeps ownership and permits retry.
            if state.contexts.contains_key(&key) {
                return Ok(teardown);
            }
            teardown
        } else {
            (Ok(()), None, None)
        };
        let primary = state.primary.get_mut(&ordinal).unwrap();
        primary.key = None;
        if !reset {
            primary.retains -= 1;
        }
        Ok(teardown)
    }))
}

#[unsafe(no_mangle)]
pub extern "system" fn cuDevicePrimaryCtxRelease(ordinal: c_int) -> CuResult {
    trace::call("cuDevicePrimaryCtxRelease", || retire(ordinal, false))
}

#[unsafe(no_mangle)]
pub extern "system" fn cuDevicePrimaryCtxRelease_v2(ordinal: c_int) -> CuResult {
    trace::call("cuDevicePrimaryCtxRelease_v2", || retire(ordinal, false))
}

#[unsafe(no_mangle)]
pub extern "system" fn cuDevicePrimaryCtxReset(ordinal: c_int) -> CuResult {
    trace::call("cuDevicePrimaryCtxReset", || retire(ordinal, true))
}

#[unsafe(no_mangle)]
pub extern "system" fn cuDevicePrimaryCtxReset_v2(ordinal: c_int) -> CuResult {
    trace::call("cuDevicePrimaryCtxReset_v2", || retire(ordinal, true))
}
