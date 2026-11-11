//! Last group size and argument bytes set on each Level Zero kernel, so a launch only calls
//! `zeKernelSetGroupSize` / `zeKernelSetArgumentValue` for values that changed.
//!
//! Level Zero kernel arguments are state of the kernel object: they persist across appends and
//! `zeCommandListAppendLaunchKernel` captures the values current at append time. Re-setting an
//! identical value is therefore redundant, and ggml relaunches the same kernels with mostly
//! identical scalars every layer and every token. The driver resolves pointer arguments to their
//! allocation when they are set, so every free bumps a memory epoch that drops all cached
//! values (a new allocation at a freed address must be re-set). Only 1, 2, 4 and 8-byte values
//! are cached; ggml passes nothing wider. Entries are dropped when the
//! kernel or its context is destroyed, before the handle can be reused.
//! Module destruction counts as a free (module globals can be arguments of other modules'
//! kernels).
//!
//! This relies on the driver resolving pointer arguments when they are set and capturing every
//! argument at append time, as the Level Zero specification and Intel's driver do. Only group
//! size and argument values are tracked: other kernel state (`zeKernelSetIndirectAccess`,
//! `zeKernelSetCacheConfig`) is set at kernel creation and never cached. Every argument and
//! group-size call must go through `apply`: a direct `zeKernelSetArgumentValue` or
//! `zeKernelSetGroupSize` elsewhere would desync the cache.
//! `CUDA_FOR_INTEL_ARG_CACHE=0` turns the cache off.
use super::{Error, Raw};
use crate::ptx::fxhash::FxHashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Default)]
struct Kernel {
    epoch: u64,
    group: Option<[u32; 3]>,
    args: Vec<Option<(u8, u64)>>,
}

static KERNELS: Mutex<Option<FxHashMap<usize, Kernel>>> = Mutex::new(None);
static MEMORY_EPOCH: AtomicU64 = AtomicU64::new(0);

fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("CUDA_FOR_INTEL_ARG_CACHE").as_deref() != Some(std::ffi::OsStr::new("0"))
    })
}

/// A device or host allocation was (or may have been) freed.
pub(super) fn memory_freed() {
    MEMORY_EPOCH.fetch_add(1, Ordering::SeqCst);
}

/// `kernel` is about to be destroyed.
pub(super) fn forget(kernel: Raw) {
    if let Ok(mut guard) = KERNELS.lock()
        && let Some(map) = guard.as_mut()
    {
        map.remove(&kernel.addr());
    }
}

/// A context is about to be destroyed: its kernels die with it.
pub(super) fn forget_all() {
    if let Ok(mut guard) = KERNELS.lock() {
        *guard = None;
    }
}

/// The value of a 1, 2, 4 or 8-byte argument with its size; `None` (never cached) otherwise.
fn bytes(value: *const std::ffi::c_void, size: usize) -> Option<(u8, u64)> {
    if value.is_null() {
        return None;
    }
    let bits = unsafe {
        match size {
            1 => u64::from(value.cast::<u8>().read()),
            2 => u64::from(value.cast::<u16>().read_unaligned()),
            4 => u64::from(value.cast::<u32>().read_unaligned()),
            8 => value.cast::<u64>().read_unaligned(),
            _ => return None,
        }
    };
    Some((size as u8, bits))
}

/// Sets the group size and arguments of `kernel`, skipping values it already holds.
/// `set_group(x, y, z)` and `set_arg(index, size, value)` make the native calls.
pub(super) fn apply(
    kernel: Raw,
    group: [u32; 3],
    args: &[super::streams::KernelArg],
    mut set_group: impl FnMut([u32; 3]) -> Result<(), Error>,
    mut set_arg: impl FnMut(u32, usize, *const std::ffi::c_void) -> Result<(), Error>,
) -> Result<(), Error> {
    let mut guard = match enabled().then(|| KERNELS.lock()) {
        Some(Ok(guard)) => guard,
        _ => {
            set_group(group)?;
            for (index, arg) in args.iter().enumerate() {
                set_arg(index as u32, arg.size, arg.value)?;
            }
            return Ok(());
        }
    };
    let state = guard.get_or_insert_with(FxHashMap::default).entry(kernel.addr()).or_default();
    apply_to(state, MEMORY_EPOCH.load(Ordering::SeqCst), group, args, set_group, set_arg)
}

fn apply_to(
    state: &mut Kernel,
    epoch: u64,
    group: [u32; 3],
    args: &[super::streams::KernelArg],
    mut set_group: impl FnMut([u32; 3]) -> Result<(), Error>,
    mut set_arg: impl FnMut(u32, usize, *const std::ffi::c_void) -> Result<(), Error>,
) -> Result<(), Error> {
    if state.epoch != epoch {
        state.epoch = epoch;
        state.args.clear();
    }
    if state.group != Some(group) {
        state.group = None;
        set_group(group)?;
        state.group = Some(group);
    }
    if state.args.len() < args.len() {
        state.args.resize(args.len(), None);
    }
    for ((index, arg), cached) in args.iter().enumerate().zip(&mut state.args) {
        let value = bytes(arg.value, arg.size);
        if value.is_some() && *cached == value {
            continue;
        }
        // Unknown until the call succeeds.
        *cached = None;
        set_arg(index as u32, arg.size, arg.value)?;
        *cached = value;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::level_zero::KernelArg;

    fn arg(value: &u64, size: usize) -> KernelArg {
        KernelArg { value: std::ptr::from_ref(value).cast(), size }
    }

    /// Runs one launch's argument setting against `state` and returns the native calls made.
    fn run(state: &mut Kernel, epoch: u64, group: [u32; 3], args: &[KernelArg]) -> Vec<String> {
        let calls = std::cell::RefCell::new(Vec::new());
        apply_to(
            state,
            epoch,
            group,
            args,
            |g| {
                calls.borrow_mut().push(format!("group {g:?}"));
                Ok(())
            },
            |index, size, _| {
                calls.borrow_mut().push(format!("arg {index} {size}"));
                Ok(())
            },
        )
        .unwrap();
        calls.into_inner()
    }

    #[test]
    fn only_changed_values_reach_the_driver_and_frees_reset_the_cache() {
        let (a, b, c) = (1u64, 2u64, 0xdead_beefu64);
        let mut state = Kernel::default();
        assert_eq!(run(&mut state, 0, [32, 1, 1], &[arg(&a, 4), arg(&c, 8)]), ["group [32, 1, 1]", "arg 0 4", "arg 1 8"]);
        assert!(run(&mut state, 0, [32, 1, 1], &[arg(&a, 4), arg(&c, 8)]).is_empty());
        assert_eq!(run(&mut state, 0, [64, 1, 1], &[arg(&b, 4), arg(&c, 8)]), ["group [64, 1, 1]", "arg 0 4"]);
        // Same low bytes, different size: set again.
        assert_eq!(run(&mut state, 0, [64, 1, 1], &[arg(&b, 8), arg(&c, 8)]), ["arg 0 8"]);
        // A free (new epoch) re-sets every argument.
        assert_eq!(run(&mut state, 1, [64, 1, 1], &[arg(&b, 8), arg(&c, 8)]), ["arg 0 8", "arg 1 8"]);
        // Null and wide values are never cached.
        let wide = KernelArg { value: std::ptr::from_ref(&[0u64; 2]).cast(), size: 12 };
        let null = KernelArg { value: std::ptr::null(), size: 8 };
        assert_eq!(run(&mut state, 1, [64, 1, 1], &[wide, null]), ["arg 0 12", "arg 1 8"]);
        assert_eq!(run(&mut state, 1, [64, 1, 1], &[wide, null]), ["arg 0 12", "arg 1 8"]);
    }

    #[test]
    fn a_failed_set_is_retried_on_the_next_launch() {
        let a = 7u64;
        let mut state = Kernel::default();
        let failed = apply_to(&mut state, 0, [1, 1, 1], &[arg(&a, 4)], |_| Ok(()), |_, _, _| {
            Err(Error::Driver("zeKernelSetArgumentValue", 0x78000004))
        });
        assert!(failed.is_err());
        assert_eq!(run(&mut state, 0, [1, 1, 1], &[arg(&a, 4)]), ["arg 0 4"]);
        let failed = apply_to(&mut state, 0, [2, 1, 1], &[arg(&a, 4)], |_| {
            Err(Error::Driver("zeKernelSetGroupSize", 0x78000004))
        }, |_, _, _| Ok(()));
        assert!(failed.is_err());
        assert_eq!(run(&mut state, 0, [2, 1, 1], &[arg(&a, 4)]), ["group [2, 1, 1]"]);
    }
}
