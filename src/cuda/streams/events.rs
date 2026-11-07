use super::{Fence, Recording, Streams};
use crate::cuda::trace;
use crate::cuda::{
    Api, Context, CuResult, Opaque, backend_error, reserve_handle, status, with_context,
    with_context_handles,
};
use CuResult::*;
use std::sync::Arc;

impl Streams {
    /// Capture-aware part of a stream wait. `None` means the wait is an ordinary
    /// stream-ordered wait, `Some` is the final outcome.
    fn capture_wait(&mut self, key: usize, event: usize) -> Option<Result<(), CuResult>> {
        let own = self.capture(key).ok().flatten().cloned();
        let recorded = self.captured.get(&event).cloned();
        match (own, recorded) {
            (None, None) => None,
            (Some(own), None) => {
                // A capturing stream cannot depend on work outside the capture.
                if matches!(self.events.get(&event), Some(None)) {
                    Some(Ok(()))
                } else {
                    Some(Err(own.invalidate_as(StreamCaptureIsolation)))
                }
            }
            (own, Some((capture, producer, sequence))) => {
                if capture.is_invalidated() {
                    return Some(Err(StreamCaptureInvalidated));
                }
                match own {
                    Some(own) if !own.same(&capture) => {
                        Some(Err(own.invalidate_as(StreamCaptureMerge)))
                    }
                    Some(_) => {
                        capture.note_wait(producer, key, sequence);
                        capture.apply_wait(event, key);
                        Some(Ok(()))
                    }
                    None => {
                        // Fork: the waiting stream joins the capture.
                        if key == 0 {
                            return Some(Err(capture.invalidate()));
                        }
                        capture.join(key);
                        *self.capture_slot(key).ok()? = Some(capture.clone());
                        capture.note_wait(producer, key, sequence);
                        capture.apply_wait(event, key);
                        Some(Ok(()))
                    }
                }
            }
        }
    }

    /// Record `event` on `key` when a graph event-record node is replayed.
    pub(in crate::cuda) fn replay_event_record(
        &mut self,
        api: &Api,
        context: &Context,
        event: usize,
        key: usize,
    ) -> Result<(), CuResult> {
        if !self.events.contains_key(&event) {
            return Err(InvalidHandle);
        }
        self.captured.remove(&event);
        let completion = self.event_marker(api, context, key, None)?;
        self.events.insert(
            event,
            Some(Recording {
                producer: key,
                fence: completion,
            }),
        );
        Ok(())
    }

    /// Make `key` wait for `event` when a graph event-wait node is replayed.
    pub(in crate::cuda) fn replay_event_wait(
        &mut self,
        api: &Api,
        context: &Context,
        event: usize,
        key: usize,
    ) -> Result<(), CuResult> {
        let recording = self
            .events
            .get(&event)
            .ok_or(InvalidHandle)?
            .as_ref()
            .map(|recording| (recording.producer, Arc::clone(&recording.fence)));
        if let Some((producer, fence)) = recording {
            if key != producer {
                self.flush_producer(api, producer, &fence)?;
            }
            self.event_marker(api, context, key, Some(fence))?;
        }
        Ok(())
    }

    pub(super) fn event(&self, handle: Opaque) -> Result<&Option<Recording>, CuResult> {
        self.events.get(&handle.addr()).ok_or(InvalidHandle)
    }

    pub(super) fn event_marker(
        &mut self,
        api: &Api,
        context: &Context,
        key: usize,
        dependency: Option<Arc<Fence>>,
    ) -> Result<Arc<Fence>, CuResult> {
        let blocking = self.get(key)?.blocking;
        self.collect_retired(api)?;
        let completion = self.next_fence(api, context)?;
        let mut waits = self.legacy_waits(api, key, blocking)?;
        if let Some(fence) = dependency
            && !waits.iter().any(|wait| Arc::ptr_eq(wait, &fence))
        {
            waits.push(fence);
        }
        self.get_mut(key)?.record(
            api,
            context,
            Some(Arc::clone(&completion)),
            waits,
            |list, signal, waits| list.barrier(api, signal.unwrap(), waits),
        )?;
        Ok(completion)
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuEventCreate(out: *mut Opaque, flags: u32) -> CuResult {
    trace::call("cuEventCreate", || {
        status(with_context_handles(|_, context, next| {
            if out.is_null() {
                return Err(InvalidValue);
            }
            match flags {
                // DEFAULT, BLOCKING_SYNC and DISABLE_TIMING. BLOCKING_SYNC only picks the
                // host wait strategy and host waits here always block in the driver.
                0..=3 => {}
                6 | 7 => return Err(NotSupported), // INTERPROCESS
                _ => return Err(InvalidValue),
            }
            let key = reserve_handle(next)?;
            context.streams.events.insert(key, None);
            context.streams.times.register(key, flags);
            unsafe { out.write(std::ptr::without_provenance_mut(key)) };
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuEventRecord(event: Opaque, stream: Opaque) -> CuResult {
    trace::call("cuEventRecord", || {
        status(with_context(|runtime, context| {
            let streams = &mut context.streams;
            streams.event(event)?;
            let key = streams.launch_key(stream)?;
            if let Some(capture) = streams.capture(key)?.cloned() {
                // Captured events only mark a dependency; they create no node.
                if capture.is_invalidated() {
                    return Err(StreamCaptureInvalidated);
                }
                let sequence = capture.record_event(key);
                capture.snapshot_event(event.addr(), key);
                streams
                    .captured
                    .insert(event.addr(), (capture, key, sequence));
                return Ok(());
            }
            streams.captured.remove(&event.addr());
            // Publish only after append succeeds. Older consumers keep their own
            // reference to the previous recording, even after this handle changes.
            let completion = streams.record_event(&runtime.api, &context.backend, key, event.addr())?;
            streams.events.insert(
                event.addr(),
                Some(Recording {
                    producer: key,
                    fence: completion,
                }),
            );
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuEventQuery(event: Opaque) -> CuResult {
    trace::call("cuEventQuery", || {
        status(with_context(|runtime, context| {
            let streams = &context.streams;
            if streams.captured.contains_key(&event.addr()) {
                return Err(CapturedEvent);
            }
            if let Some(recording) = streams.event(event)? {
                if let Ok(producer) = streams.get(recording.producer) {
                    producer.check()?;
                }
                if !recording
                    .fence
                    .ready(&runtime.api)
                    .inspect_err(|error| context.backend.health.observe(error))
                    .map_err(backend_error)?
                {
                    return Err(NotReady);
                }
            }
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuEventSynchronize(event: Opaque) -> CuResult {
    trace::call("cuEventSynchronize", || {
        let snapshot = with_context(|runtime, context| {
            let streams = &mut context.streams;
            if streams.captured.contains_key(&event.addr()) {
                return Err(CapturedEvent);
            }
            let Some(recording) = streams.event(event)? else {
                return Ok(None);
            };
            let producer = recording.producer;
            let fence = Arc::clone(&recording.fence);
            streams.flush_producer(&runtime.api, producer, &fence)?;
            Ok(Some((
                Arc::clone(&streams.event_waiters),
                fence,
                runtime.api.event_wait_fn(),
                Arc::clone(&context.backend.health),
            )))
        });
        status(snapshot.and_then(|snapshot| {
            // Keep the fence and its context alive with the state mutex released.
            if let Some((_context_guard, fence, wait, health)) = snapshot {
                fence
                    .synchronize(wait)
                    .inspect_err(|error| health.observe(error))
                    .map_err(backend_error)?;
            }
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuEventDestroy_v2(event: Opaque) -> CuResult {
    trace::call("cuEventDestroy_v2", || {
        status(with_context(|_, context| {
            let recording = context
                .streams
                .events
                .remove(&event.addr())
                .ok_or(InvalidHandle)?;
            context
                .streams
                .times
                .release(event.addr(), recording.map(|recording| recording.fence));
            context.streams.captured.remove(&event.addr());
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuEventRecordWithFlags(event: Opaque, stream: Opaque, flags: u32) -> CuResult {
    match flags {
        0 => cuEventRecord(event, stream),
        // External record nodes have no Level Zero equivalent here.
        1 => NotSupported,
        _ => InvalidValue,
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn cuStreamWaitEvent(stream: Opaque, event: Opaque, flags: u32) -> CuResult {
    trace::call("cuStreamWaitEvent", || {
        status(with_context(|runtime, context| {
            if flags != 0 {
                return Err(InvalidValue);
            }
            let streams = &mut context.streams;
            let key = streams.launch_key(stream)?;
            streams.event(event)?;
            if let Some(result) = streams.capture_wait(key, event.addr()) {
                return result;
            }
            if let Some(recording) = streams.event(event)? {
                let producer = recording.producer;
                let fence = Arc::clone(&recording.fence);
                if key != producer {
                    streams.flush_producer(&runtime.api, producer, &fence)?;
                }
                streams.event_marker(&runtime.api, &context.backend, key, Some(fence))?;
            }
            Ok(())
        }))
    })
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;
    use crate::cuda::streams::{cuStreamCreate, cuStreamDestroy_v2, cuStreamQuery};
    use crate::cuda::{
        cuCtxCreate_v2, cuCtxDestroy_v2, cuCtxSetCurrent, cuCtxSynchronize, cuInit, test_device,
        with_context,
    };
    use std::ptr::{null_mut, without_provenance_mut};
    use std::time::{Duration, Instant};

    #[test]
    #[ignore = "requires an Intel Level Zero GPU"]
    fn gated_event_generations_and_concurrent_teardown() {
        assert_eq!(cuInit(0), Success);
        let (mut context, mut producer, mut consumer) = (null_mut(), null_mut(), null_mut());
        let (mut event, mut later, mut prefix) = (null_mut(), null_mut(), null_mut());
        assert_eq!(
            unsafe { cuCtxCreate_v2(&mut context, 0, test_device()) },
            Success
        );
        assert_eq!(unsafe { cuStreamCreate(&mut producer, 1) }, Success);
        assert_eq!(unsafe { cuStreamCreate(&mut consumer, 0) }, Success);
        assert_eq!(unsafe { cuEventCreate(&mut event, 2) }, Success);
        assert_eq!(unsafe { cuEventCreate(&mut later, 0) }, Success);
        assert_eq!(unsafe { cuEventCreate(&mut prefix, 2) }, Success);
        assert_eq!(cuEventQuery(event), Success);
        assert_eq!(cuEventSynchronize(event), Success);
        assert_eq!(cuEventRecord(prefix, producer), Success);

        let gate = with_context(|runtime, context| {
            let api = &runtime.api;
            let streams = &mut context.streams;
            let gate = streams.next_fence(api, &context.backend)?;
            let stream = streams.get_mut(producer.addr())?;
            stream.record(api, &context.backend, None, Vec::new(), |list, _, _| {
                list.gate(api, &gate)
            })?;
            Ok(gate)
        })
        .unwrap();

        assert_eq!(cuStreamWaitEvent(consumer, event, 0), Success);
        assert_eq!(cuStreamQuery(consumer), Success);
        assert_eq!(cuEventRecord(event, producer), Success);
        assert_eq!(cuEventRecord(later, producer), Success);
        assert_eq!(cuEventQuery(event), NotReady);
        assert_eq!(cuEventQuery(later), NotReady);
        assert_eq!(cuEventSynchronize(prefix), Success);
        assert_eq!(cuStreamQuery(producer), NotReady);
        assert_eq!(cuEventDestroy_v2(prefix), Success);
        assert_eq!(cuStreamWaitEvent(null_mut(), event, 0), Success);
        // A blocking-stream marker takes the implicit edge from the legacy wait.
        assert_eq!(cuEventRecord(later, consumer), Success);
        assert_eq!(cuEventQuery(later), NotReady);
        // Keep two explicit consumers of the original generation as well.
        assert_eq!(cuStreamWaitEvent(consumer, event, 0), Success);
        assert_eq!(cuStreamQuery(consumer), NotReady);
        assert_eq!(cuStreamQuery(null_mut()), NotReady);

        let (context_key, event_key) = (context.addr(), event.addr());
        std::thread::scope(|scope| {
            let waiter = scope.spawn(move || {
                assert_eq!(
                    cuCtxSetCurrent(without_provenance_mut(context_key)),
                    Success
                );
                cuEventSynchronize(without_provenance_mut(event_key))
            });
            let deadline = Instant::now() + Duration::from_secs(10);
            while !with_context(|_, context| Ok(context.streams.has_waiters())).unwrap() {
                if waiter.is_finished() || Instant::now() >= deadline {
                    with_context(|runtime, _| {
                        gate.host_signal(&runtime.api).map_err(backend_error)
                    })
                    .unwrap();
                    panic!("host waiter did not snapshot the pending recording");
                }
                std::thread::yield_now();
            }

            // Re-record downstream of an existing wait. Redirecting that wait
            // to the new recording would create a cycle in the consumer list.
            let rerecord = cuEventRecord(event, consumer);
            let pending = cuEventQuery(event);
            let destroyed = cuEventDestroy_v2(event);
            let stale = cuEventQuery(event);
            // Force a refill while the old producer, consumers and host waiter
            // are still pending. A reset of any of their fences is unsafe.
            let refill = (0..64).try_for_each(|_| match cuEventRecord(later, producer) {
                Success => Ok(()),
                error => Err(error),
            });
            let pending_pools =
                with_context(|_, context| Ok(context.streams.fences.pool_count())).unwrap();
            let retired = cuStreamDestroy_v2(producer);
            let teardown = cuCtxDestroy_v2(context);
            with_context(|runtime, _| gate.host_signal(&runtime.api).map_err(backend_error))
                .unwrap();
            assert_eq!(waiter.join().unwrap(), Success);
            assert_eq!(rerecord, Success);
            assert_eq!(pending, NotReady);
            assert_eq!(destroyed, Success);
            assert_eq!(stale, InvalidHandle);
            assert_eq!(refill, Ok(()));
            assert_eq!(pending_pools, 2);
            assert_eq!(retired, Success);
            assert_eq!(teardown, NotReady);
        });
        assert_eq!(cuEventSynchronize(later), Success);
        assert_eq!(cuCtxSynchronize(), Success);
        drop(gate);
        assert_eq!(cuEventDestroy_v2(later), Success);

        assert_eq!(unsafe { cuEventCreate(&mut event, 2) }, Success);
        for _ in 0..8 {
            for _ in 0..32 {
                assert_eq!(cuEventRecord(event, consumer), Success);
            }
            assert_eq!(cuEventSynchronize(event), Success);
            assert_eq!(cuCtxSynchronize(), Success);
        }
        with_context(|_, context| {
            assert_eq!(context.streams.fences.pool_count(), 2);
            assert!(context.streams.retired.is_empty());
            Ok(())
        })
        .unwrap();
        assert_eq!(cuCtxDestroy_v2(context), Success);
    }
}
