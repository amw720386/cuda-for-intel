//! `cuStreamWaitEvent` where the event was recorded in another context (typically
//! another device). A Level Zero event belongs to one Level Zero context and cannot be
//! waited on from a command list of another, so the producer's work is flushed and the
//! host waits for it before the wait call returns. Everything the waiting stream
//! receives afterwards is then correctly ordered after the event.

use super::{Fence, stream_owner};
use crate::level_zero::DeviceHealth;
use crate::level_zero::WaitFn;
use crate::cuda::{
    CURRENT, CuResult, CuResult::*, Opaque, State, backend_error, current, state, status,
};
use std::sync::Arc;

pub(super) enum Route {
    /// The wait is finished (cross-context waits complete on the host).
    Done(CuResult),
    /// Stream and event share the current context: the usual stream-ordered wait.
    Ordinary,
    /// Stream and event share a context that is not the current one.
    OrdinaryIn(usize),
}

/// Decide how a wait is carried out. With one device it is always `Ordinary`.
pub(super) fn route(stream: Opaque, event: Opaque, flags: u32) -> Route {
    match state(|state| plan(state, stream, event, flags)) {
        Ok(Plan::Ordinary) => Route::Ordinary,
        Ok(Plan::SameContext(owner)) => Route::OrdinaryIn(owner),
        Ok(Plan::Never) => Route::Done(Success),
        Ok(Plan::Host(_guard, fence, wait, health)) => Route::Done(status(
            fence
                .synchronize(wait)
                .inspect_err(|error| health.observe(error))
                .map_err(backend_error),
        )),
        Err(error) => Route::Done(error),
    }
}

/// Run `action` with `owner` as the thread's current context, restoring it afterwards.
pub(super) fn with_current<T>(owner: usize, action: impl FnOnce() -> T) -> T {
    let saved = CURRENT.with_borrow_mut(|stack| {
        let saved = stack.last().copied();
        crate::cuda::current_stack::replace_top(stack, owner);
        saved
    });
    let result = action();
    CURRENT.with_borrow_mut(|stack| match saved {
        Some(key) => crate::cuda::current_stack::replace_top(stack, key),
        None => {
            stack.pop();
        }
    });
    result
}

enum Plan {
    Ordinary,
    SameContext(usize),
    /// The event was never recorded: nothing to wait for.
    Never,
    Host(Arc<()>, Arc<Fence>, WaitFn, Arc<DeviceHealth>),
}

fn plan(
    state: &mut State,
    stream: Opaque,
    event: Opaque,
    flags: u32,
) -> Result<Plan, CuResult> {
    if state.runtime.devices.len() < 2 {
        return Ok(Plan::Ordinary);
    }
    let (waiter, key) = stream_owner(state, stream)?;
    let own = state.contexts.get(&waiter).ok_or(crate::cuda::CuResult::ContextIsDestroyed)?;
    if own.streams.events.contains_key(&event.addr()) {
        return Ok(if current() == Some(waiter) {
            Plan::Ordinary
        } else {
            Plan::SameContext(waiter)
        });
    }
    let Some(owner) = state
        .contexts
        .iter()
        .find(|(_, c)| c.streams.events.contains_key(&event.addr()))
        .map(|(&k, _)| k)
    else {
        return Ok(Plan::Ordinary); // unknown event: the ordinary path reports the error
    };
    if flags != 0 {
        return Err(InvalidValue);
    }
    own.check_live()?;
    if own.streams.get(key)?.capture.is_some() {
        return Err(StreamCaptureUnsupported);
    }
    let api = &state.runtime.api;
    let context = state.contexts.get_mut(&owner).ok_or(InvalidContext)?;
    context.check_live()?;
    let streams = &mut context.streams;
    if streams.captured.contains_key(&event.addr()) {
        return Err(CapturedEvent);
    }
    // Never recorded: nothing to wait for, as in CUDA.
    let Some(recording) = streams.events.get(&event.addr()).and_then(Option::as_ref) else {
        return Ok(Plan::Never);
    };
    let (producer, fence) = (recording.producer, Arc::clone(&recording.fence));
    streams.flush_producer(api, producer, &fence)?;
    Ok(Plan::Host(
        Arc::clone(&streams.event_waiters),
        fence,
        api.event_wait_fn(),
        Arc::clone(&context.backend.health),
    ))
}
