//! Peer access and cross-device copies.
//!
//! Every CUDA context owns its own Level Zero context, and a USM pointer is only valid
//! inside the Level Zero context that allocated it. A kernel therefore cannot
//! dereference another device's memory, even when `zeDeviceCanAccessPeer` says the
//! hardware link exists, so `cuDeviceCanAccessPeer` reports 0 and peer access cannot be
//! enabled. Frameworks react by copying with `cuMemcpyPeer`, which always works: the
//! data is staged through a host buffer, read with the source context and written with
//! the destination context. `zeDeviceCanAccessPeer` is used for the link ranking
//! (`cuDeviceGetP2PAttribute`) and decides whether direct kernel access could be
//! offered once allocations share one Level Zero context.

use super::super::*;
use std::sync::atomic::{AtomicBool, Ordering};

/// Staging chunk. Large enough to saturate PCIe, small enough to bound memory use.
pub(crate) const CHUNK: usize = 4 << 20;

/// Whether kernels may dereference peer allocations once the link is native. Off until
/// allocations of linked devices share a Level Zero context.
static DIRECT_KERNEL_ACCESS: AtomicBool = AtomicBool::new(false);

#[cfg(test)]
pub(crate) fn set_direct_kernel_access(enabled: bool) {
    DIRECT_KERNEL_ACCESS.store(enabled, Ordering::SeqCst);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Link {
    /// Level Zero reports peer access in both directions.
    Native,
    /// Copies are staged through host memory.
    Staged,
}

pub(crate) fn link(runtime: &Runtime, a: usize, b: usize) -> Link {
    let (x, y) = (&runtime.devices[a], &runtime.devices[b]);
    match (
        x.can_access_peer(&runtime.api, y),
        y.can_access_peer(&runtime.api, x),
    ) {
        (Some(true), Some(true)) => Link::Native,
        _ => Link::Staged,
    }
}

fn kernel_access(link: Link) -> bool {
    link == Link::Native && DIRECT_KERNEL_ACCESS.load(Ordering::SeqCst)
}

/// Ranges of `size` bytes cut out of a transfer, as (offset, length).
pub(crate) fn chunks(size: usize, chunk: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..size)
        .step_by(chunk.max(1))
        .map(move |offset| (offset, chunk.min(size - offset)))
}

fn device_pair(state: &State, a: c_int, b: c_int) -> Result<(usize, usize), CuResult> {
    let count = state.runtime.devices.len();
    let index = |ordinal: c_int| {
        usize::try_from(ordinal)
            .ok()
            .filter(|&i| i < count)
            .ok_or(InvalidDevice)
    };
    Ok((index(a)?, index(b)?))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuDeviceCanAccessPeer(
    out: *mut c_int,
    device: c_int,
    peer: c_int,
) -> CuResult {
    trace::call("cuDeviceCanAccessPeer", || {
        if out.is_null() {
            return InvalidValue;
        }
        status(state(|state| {
            let (a, b) = device_pair(state, device, peer)?;
            let can = a != b && kernel_access(link(&state.runtime, a, b));
            unsafe { out.write(c_int::from(can)) };
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuDeviceGetP2PAttribute(
    out: *mut c_int,
    attr: c_int,
    src: c_int,
    dst: c_int,
) -> CuResult {
    trace::call("cuDeviceGetP2PAttribute", || {
        // 1 PERFORMANCE_RANK, 2 ACCESS_SUPPORTED, 3 NATIVE_ATOMIC_SUPPORTED,
        // 4 CUDA_ARRAY_ACCESS_SUPPORTED, 5 ONLY_PARTIAL_NATIVE_ATOMIC_SUPPORTED.
        if out.is_null() || !(1..=5).contains(&attr) {
            return InvalidValue;
        }
        status(state(|state| {
            let (a, b) = device_pair(state, src, dst)?;
            if a == b {
                return Err(InvalidDevice);
            }
            let link = link(&state.runtime, a, b);
            let value = match attr {
                // Lower is faster: a native link beats a host-staged copy.
                1 => match link {
                    Link::Native => 1,
                    Link::Staged => 2,
                },
                2 => c_int::from(kernel_access(link)),
                _ => 0,
            };
            unsafe { out.write(value) };
            Ok(())
        }))
    })
}

/// The peer context and the current one, validated for (en|dis)abling access.
fn peer_pair(state: &State, peer: Opaque) -> Result<(usize, usize), CuResult> {
    let current_key = current().ok_or(InvalidContext)?;
    let current = state.contexts.get(&current_key).ok_or(ContextIsDestroyed)?;
    let other = state.contexts.get(&peer.addr()).ok_or(InvalidContext)?;
    other.check_live()?;
    if other.device == current.device {
        return Err(InvalidDevice);
    }
    Ok((current_key, peer.addr()))
}

#[unsafe(no_mangle)]
pub extern "system" fn cuCtxEnablePeerAccess(peer: Opaque, flags: c_uint) -> CuResult {
    trace::call("cuCtxEnablePeerAccess", || {
        if flags != 0 {
            return InvalidValue;
        }
        status(state(|state| {
            let (current_key, peer_key) = peer_pair(state, peer)?;
            let (a, b) = (
                state.contexts[&current_key].device as usize,
                state.contexts[&peer_key].device as usize,
            );
            if !kernel_access(link(&state.runtime, a, b)) {
                return Err(PeerAccessUnsupported);
            }
            let context = state.contexts.get_mut(&current_key).ok_or(InvalidContext)?;
            if !context.peer_access.insert(peer_key) {
                return Err(PeerAccessAlreadyEnabled);
            }
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuCtxDisablePeerAccess(peer: Opaque) -> CuResult {
    trace::call("cuCtxDisablePeerAccess", || {
        status(state(|state| {
            let (current_key, peer_key) = peer_pair(state, peer)?;
            let context = state.contexts.get_mut(&current_key).ok_or(InvalidContext)?;
            if context.peer_access.remove(&peer_key) {
                Ok(())
            } else {
                Err(PeerAccessNotEnabled)
            }
        }))
    })
}

/// Copy `size` bytes between device allocations of two live contexts. Same-context
/// copies run natively; cross-context copies are staged through host memory in
/// `CHUNK`-sized pieces and complete before returning, which satisfies the stream
/// ordering of the asynchronous forms.
pub(crate) fn copy_between(
    state: &mut State,
    dst_key: usize,
    dst: u64,
    src_key: usize,
    src: u64,
    size: usize,
) -> Result<(), CuResult> {
    for key in [dst_key, src_key] {
        state
            .contexts
            .get(&key)
            .ok_or(InvalidContext)?
            .check_live()?;
    }
    if size == 0 {
        return Ok(());
    }
    let (dst, src) = (dst as usize, src as usize);
    let api = &state.runtime.api;
    let source = state.contexts.get_mut(&src_key).ok_or(InvalidContext)?;
    if !contains_range(&source.allocations, src, size) {
        return Err(InvalidValue);
    }
    source.streams.wait_all(api)?;
    if dst_key == src_key {
        if !contains_range(&source.allocations, dst, size) {
            return Err(InvalidValue);
        }
        let (dst_ptr, src_ptr) = (
            std::ptr::with_exposed_provenance_mut(dst),
            std::ptr::with_exposed_provenance(src),
        );
        return unsafe { source.backend.copy(api, dst_ptr, src_ptr, size) }
            .inspect_err(|error| source.backend.health.observe(error))
            .map_err(backend_error);
    }
    let target = state.contexts.get_mut(&dst_key).ok_or(InvalidContext)?;
    if !contains_range(&target.allocations, dst, size) {
        return Err(InvalidValue);
    }
    target.streams.wait_all(api)?;
    let mut staging = vec![0u8; size.min(CHUNK)];
    for (offset, length) in chunks(size, CHUNK) {
        let source = state.contexts.get(&src_key).ok_or(InvalidContext)?;
        let from = std::ptr::with_exposed_provenance(src + offset);
        unsafe {
            source
                .backend
                .copy(api, staging.as_mut_ptr().cast(), from, length)
        }
        .inspect_err(|error| source.backend.health.observe(error))
        .map_err(backend_error)?;
        let target = state.contexts.get(&dst_key).ok_or(InvalidContext)?;
        let to = std::ptr::with_exposed_provenance_mut(dst + offset);
        unsafe { target.backend.copy(api, to, staging.as_ptr().cast(), length) }
            .inspect_err(|error| target.backend.health.observe(error))
            .map_err(backend_error)?;
    }
    Ok(())
}

fn peer_copy(
    dst: u64,
    dst_context: Opaque,
    src: u64,
    src_context: Opaque,
    size: usize,
    stream: Opaque,
) -> CuResult {
    status(state(|state| {
        // A stream handle must name a live stream (any context); 0 is the legacy stream.
        let stream_key = stream.addr();
        if stream_key > 2 && !state.stream_owners.contains_key(&stream_key) {
            return Err(InvalidHandle);
        }
        copy_between(state, dst_context.addr(), dst, src_context.addr(), src, size)
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemcpyPeer(
    dst: u64,
    dst_context: Opaque,
    src: u64,
    src_context: Opaque,
    size: usize,
) -> CuResult {
    trace::call("cuMemcpyPeer", || {
        peer_copy(dst, dst_context, src, src_context, size, std::ptr::null_mut())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemcpyPeerAsync(
    dst: u64,
    dst_context: Opaque,
    src: u64,
    src_context: Opaque,
    size: usize,
    stream: Opaque,
) -> CuResult {
    trace::call("cuMemcpyPeerAsync", || {
        peer_copy(dst, dst_context, src, src_context, size, stream)
    })
}

/// The context that owns the device allocation covering `size` bytes at `address`,
/// preferring `first` so that a clash of address ranges resolves to the caller's own.
pub(crate) fn owner_of(
    state: &State,
    first: Option<usize>,
    address: usize,
    size: usize,
) -> Option<usize> {
    let owns = |key: usize| {
        state
            .contexts
            .get(&key)
            .is_some_and(|c| contains_range(&c.allocations, address, size.max(1)))
    };
    first
        .filter(|&key| owns(key))
        .or_else(|| state.contexts.keys().copied().find(|&key| owns(key)))
}

/// `cuMemcpyDtoD` with unified addressing: when an operand belongs to another device's
/// context the copy becomes a peer copy. `None` means the normal path applies, which is
/// always the case on a single device.
pub(crate) fn foreign_dtod(dst: u64, src: u64, size: usize) -> Option<CuResult> {
    if size == 0 {
        return None;
    }
    let result = state(|state| {
        if state.runtime.devices.len() < 2 {
            return Ok(None);
        }
        let Some(me) = current() else { return Ok(None) };
        let d = owner_of(state, Some(me), dst as usize, size);
        let s = owner_of(state, Some(me), src as usize, size);
        match (d, s) {
            (Some(d), Some(s)) if d != me || s != me => {
                copy_between(state, d, dst, s, src, size).map(|()| Some(()))
            }
            _ => Ok(None),
        }
    });
    match result {
        Ok(Some(())) => Some(Success),
        Ok(None) => None,
        Err(error) => Some(error),
    }
}

/// `cuMemFree` of an allocation made in another context (CUDA frees by pointer, whatever
/// context is current). `None` when the current context owns it or nothing does.
pub(crate) fn free_foreign(address: u64) -> Option<CuResult> {
    let result = state(|state| {
        if state.runtime.devices.len() < 2 || address == 0 {
            return Ok(None);
        }
        let me = current();
        let address = address as usize;
        let Some(owner) = state
            .contexts
            .iter()
            .find(|&(key, c)| Some(*key) != me && c.allocations.contains_key(&address))
            .map(|(&key, _)| key)
        else {
            return Ok(None);
        };
        if me.is_some_and(|me| {
            state
                .contexts
                .get(&me)
                .is_some_and(|c| c.allocations.contains_key(&address))
        }) {
            return Ok(None);
        }
        let api = &state.runtime.api;
        let context = state.contexts.get_mut(&owner).ok_or(InvalidContext)?;
        context.check_live()?;
        context.streams.wait_all(api)?;
        context.free(api, address).map(|()| Some(()))
    });
    match result {
        Ok(Some(())) => Some(Success),
        Ok(None) => None,
        Err(error) => Some(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_cover_the_transfer_exactly() {
        assert_eq!(chunks(0, 8).count(), 0);
        assert_eq!(chunks(5, 8).collect::<Vec<_>>(), [(0, 5)]);
        assert_eq!(chunks(8, 8).collect::<Vec<_>>(), [(0, 8)]);
        assert_eq!(chunks(20, 8).collect::<Vec<_>>(), [(0, 8), (8, 8), (16, 4)]);
        let total: usize = chunks(CHUNK * 3 + 17, CHUNK).map(|(_, n)| n).sum();
        assert_eq!(total, CHUNK * 3 + 17);
    }
}
