use super::CuResult::*;
use super::graphs::Capture;
use super::{
    Api, Context, CuResult, Opaque, State, backend_error, context_key, current, reserve_handle,
    state, status,
};
use crate::cuda::trace;
use crate::level_zero::KernelArg;
use crate::level_zero::{CommandList, Fence, Fences, Handle};
use crate::level_zero::{DeviceHealth, QueueFence};
use batch::{Batch, Policy};
use batch_wake::Wake;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

mod batch;
pub(super) mod extra;
#[cfg(all(test, windows))]
mod batch_tests;
mod batch_wake;
mod cross_device;
pub(super) mod events;
pub(super) mod timing;
#[cfg(test)]
mod fake_tests;
#[cfg(test)]
mod launch_bench;
#[cfg(test)]
mod graph_api_tests;
#[cfg(test)]
mod globals_tests;
#[cfg(test)]
mod graph_tests;
#[cfg(test)]
mod timing_tests;
#[cfg(test)]
mod copy_validation_tests;
#[cfg(test)]
mod subst_tests;
#[cfg(all(test, windows))]
mod profiling;
#[cfg(all(test, windows))]
mod recovery_tests;

const LEGACY: usize = 1;
const PER_THREAD: usize = 2;

pub(super) fn stream_key(handle: Opaque) -> Result<usize, CuResult> {
    match handle.addr() {
        // CUDA special stream handles:
        //   0 / cudaStreamLegacy (1) → legacy default stream
        //   cudaStreamPerThread (2) → per-thread default stream
        // True per-thread isolation is not implemented yet; map PerThread onto
        // the legacy stream so Runtime clients (ggml-cuda) can memcpy/launch.
        0 | LEGACY | PER_THREAD => Ok(0),
        key => Ok(key),
    }
}

#[derive(Clone)]
struct Submission {
    completion: Arc<Fence>,
    _waits: Vec<Arc<Fence>>,
}

struct Recording {
    producer: usize,
    fence: Arc<Fence>,
}

enum Completion {
    Immediate(Arc<CommandList>, u64),
    Regular(Arc<QueueFence>),
}

struct Stream {
    immediate: Option<Arc<CommandList>>,
    batch: Option<Box<Batch>>,
    blocking: bool,
    dirty: bool,
    generation: u64,
    /// Work-groups appended since the stream last drained (see `inflight_limit`).
    queued_groups: u64,
    pending: VecDeque<Submission>,
    append_error: Option<CuResult>,
    capture: Option<Capture>,
    health: Arc<DeviceHealth>,
}

impl Stream {
    fn new(
        api: &Api,
        context: &Context,
        blocking: bool,
        policy: Option<Policy>,
        wake: &Arc<Wake>,
    ) -> Result<Self, CuResult> {
        let (immediate, batch) = match policy {
            Some(policy) => (
                None,
                Some(Box::new(Batch::new(
                    api,
                    context,
                    policy,
                    Arc::clone(wake),
                )?)),
            ),
            None => (
                Some(Arc::new(
                    context
                        .create_stream(api)
                        .inspect_err(|error| context.health.observe(error))
                        .map_err(backend_error)?,
                )),
                None,
            ),
        };
        Ok(Self {
            immediate,
            batch,
            blocking,
            dirty: false,
            generation: 0,
            queued_groups: 0,
            pending: VecDeque::new(),
            append_error: None,
            capture: None,
            health: Arc::clone(&context.health),
        })
    }

    fn tail(&self) -> Option<Arc<Fence>> {
        self.pending
            .back()
            .map(|pending| Arc::clone(&pending.completion))
    }

    fn reap(&mut self, api: &Api) -> Result<(), CuResult> {
        while let Some(pending) = self.pending.front() {
            if !pending
                .completion
                .ready(api)
                .inspect_err(|error| self.health.observe(error))
                .map_err(backend_error)?
            {
                break;
            }
            self.pending.pop_front();
        }
        Ok(())
    }

    fn check(&self) -> Result<(), CuResult> {
        if self.health.lost() {
            return Err(LaunchFailed);
        }
        self.append_error.map_or(Ok(()), Err)?;
        if let Some(batch) = &self.batch {
            batch.check()?;
        }
        Ok(())
    }

    fn flush(&mut self, api: &Api) -> Result<(), CuResult> {
        self.check()?;
        if let Some(batch) = &mut self.batch {
            batch.flush(api)?;
        }
        Ok(())
    }

    fn snapshot(&self) -> Option<Completion> {
        if let Some(batch) = &self.batch {
            batch
                .tail()
                .map(|fence| Completion::Regular(Arc::clone(fence)))
        } else {
            self.dirty.then(|| {
                Completion::Immediate(
                    Arc::clone(self.immediate.as_ref().unwrap()),
                    self.generation,
                )
            })
        }
    }

    fn wait(&mut self, api: &Api, timeout: u64) -> Result<(), CuResult> {
        self.check()?;
        if let Some(batch) = &mut self.batch {
            if batch.buffered() {
                return Err(NotReady);
            }
            batch.wait_tail(api, timeout)?;
            batch.reap(api)?;
            self.complete();
        } else if self.dirty {
            self.immediate
                .as_ref()
                .unwrap()
                .wait(api, timeout)
                .inspect_err(|error| self.health.observe(error))
                .map_err(backend_error)?;
            self.complete();
        }
        Ok(())
    }

    fn drain(&mut self, api: &Api, timeout: u64) -> Result<(), CuResult> {
        if let Some(batch) = &mut self.batch {
            batch.drain(api, timeout)?;
        } else if self.dirty {
            self.immediate
                .as_ref()
                .unwrap()
                .wait(api, timeout)
                .inspect_err(|error| self.health.observe(error))
                .map_err(backend_error)?;
        }
        self.complete();
        Ok(())
    }

    fn destroy(&mut self, api: &Api) -> Result<(), CuResult> {
        if let Some(batch) = &mut self.batch {
            batch.destroy(api)
        } else {
            self.immediate
                .as_ref()
                .unwrap()
                .destroy(api)
                .map_err(backend_error)
        }
    }

    fn record(
        &mut self,
        api: &Api,
        context: &Context,
        signal: Option<Arc<Fence>>,
        waits: Vec<Arc<Fence>>,
        append: impl FnOnce(
            &CommandList,
            Option<&Fence>,
            &[Arc<Fence>],
        ) -> Result<(), crate::level_zero::Error>,
    ) -> Result<(), CuResult> {
        if let Some(capture) = &mut self.capture {
            return Err(capture.invalidate());
        }
        self.check()?;
        let list = match &mut self.batch {
            Some(batch) => batch.recording(api, context)?,
            None => self.immediate.as_ref().unwrap(),
        };
        let result = append(list, signal.as_deref(), &waits);
        let submission = signal.map(|completion| Submission {
            completion,
            _waits: waits,
        });
        if let Err(error) = result {
            let attempted_append = matches!(&error, crate::level_zero::Error::Driver(call, _)
                if call.starts_with("zeCommandListAppend"));
            let fatal = error.device_lost();
            self.health.observe(&error);
            let error = backend_error(error);
            if let Some(batch) = &mut self.batch {
                batch.record_failed(error, submission);
            } else if attempted_append || fatal {
                // An immediate append may already have reached the driver.
                self.submitted();
                self.pending.extend(submission);
                self.append_error = Some(error);
            }
            return Err(error);
        }
        self.submitted();
        let result = match &mut self.batch {
            Some(batch) => batch.appended(api, submission.as_ref()),
            None => Ok(()),
        };
        self.pending.extend(submission);
        result
    }

    fn submitted(&mut self) {
        self.dirty = true;
        self.generation = self.generation.wrapping_add(1);
    }

    fn complete(&mut self) {
        self.queued_groups = 0;
        self.pending.clear();
        self.dirty = false;
    }

    fn has_waiters(&self) -> bool {
        self.immediate
            .as_ref()
            .is_some_and(|list| Arc::strong_count(list) > 1)
            || self.batch.as_deref().is_some_and(Batch::has_waiters)
    }
}

pub(super) struct Streams {
    legacy: Option<Stream>,
    live: HashMap<usize, Stream>,
    retired: Vec<Stream>,
    fences: Fences,
    events: HashMap<usize, Option<Recording>>,
    times: timing::EventTimes,
    /// Events recorded while their stream was capturing: event -> (capture, stream, op sequence).
    pub(super) captured: HashMap<usize, (Capture, usize, u64)>,
    policy: Option<Policy>,
    wake: Arc<Wake>,
    worker: Option<JoinHandle<Result<(), CuResult>>>,
    event_waiters: Arc<()>,
    track_legacy: bool,
    execution_error: Option<CuResult>,
}

impl Streams {
    pub(super) fn new(api: &Api, context: &Context, owner: usize) -> Result<Self, CuResult> {
        let policy = Policy::configured()?;
        let wake = Arc::new(Wake::new(
            policy.is_some_and(|policy| policy.high_resolution),
        )?);
        // The worker cannot access driver handles until the complete context
        // is published under the runtime lock. It exits if initialization fails.
        let worker = policy
            .and_then(|policy| policy.delay)
            .map(|_| batch::spawn(owner, Arc::clone(&wake)))
            .transpose()?;
        let legacy = Stream::new(api, context, true, policy, &wake)?;
        Ok(Self {
            legacy: Some(legacy),
            policy,
            wake,
            worker,
            live: HashMap::new(),
            retired: Vec::new(),
            fences: Fences::default(),
            events: HashMap::new(),
            times: timing::EventTimes::default(),
            captured: HashMap::new(),
            event_waiters: Arc::new(()),
            track_legacy: false,
            execution_error: None,
        })
    }

    fn get(&self, key: usize) -> Result<&Stream, CuResult> {
        if key == 0 {
            self.legacy.as_ref().ok_or(InvalidContext)
        } else {
            self.live.get(&key).ok_or(InvalidHandle)
        }
    }

    fn get_mut(&mut self, key: usize) -> Result<&mut Stream, CuResult> {
        if key == 0 {
            self.legacy.as_mut().ok_or(InvalidContext)
        } else {
            self.live.get_mut(&key).ok_or(InvalidHandle)
        }
    }

    fn explicit(&self) -> impl Iterator<Item = &Stream> {
        self.live.values().chain(&self.retired)
    }

    fn all_mut(&mut self) -> impl Iterator<Item = &mut Stream> {
        self.legacy
            .iter_mut()
            .chain(self.live.values_mut())
            .chain(&mut self.retired)
    }

    pub(super) fn has_waiters(&self) -> bool {
        Arc::strong_count(&self.event_waiters) > 1
            || self
                .legacy
                .iter()
                .chain(self.explicit())
                .any(Stream::has_waiters)
    }

    fn selected(&mut self, key: Option<usize>) -> Result<Vec<&mut Stream>, CuResult> {
        if let Some(key) = key.filter(|&key| key != 0) {
            return Ok(vec![self.get_mut(key)?]);
        }
        Ok(self
            .all_mut()
            .filter(|stream| key.is_none() || stream.blocking)
            .collect())
    }

    fn snapshot(
        &mut self,
        api: &Api,
        key: Option<usize>,
        skip_capture: bool,
    ) -> Result<Vec<Completion>, CuResult> {
        self.selected(key)?
            .into_iter()
            .filter_map(|stream| {
                if let Some(capture) = &mut stream.capture {
                    // Best-effort pre-drains must not invalidate a capture.
                    return (!skip_capture).then(|| Err(capture.invalidate()));
                }
                match stream.flush(api) {
                    Ok(()) => stream.snapshot().map(Ok),
                    Err(error) => Some(Err(error)),
                }
            })
            .collect()
    }

    pub(super) fn stop(&mut self) -> Option<JoinHandle<Result<(), CuResult>>> {
        let worker = self.worker.take()?;
        self.wake.notify();
        Some(worker)
    }

    fn collect_retired(&mut self, api: &Api) -> Result<(), CuResult> {
        let mut index = 0;
        while index < self.retired.len() {
            let stream = &mut self.retired[index];
            if stream.has_waiters() {
                index += 1;
                continue;
            }
            match stream.wait(api, 0) {
                Ok(()) => {
                    stream.destroy(api)?;
                    self.retired.swap_remove(index);
                }
                Err(NotReady) => index += 1,
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }

    fn next_fence(&mut self, api: &Api, context: &Context) -> Result<Arc<Fence>, CuResult> {
        if self.fences.needs_refill() {
            for stream in self.all_mut() {
                stream.reap(api)?;
            }
        }
        self.fences
            .acquire(api, context)
            .inspect_err(|error| context.health.observe(error))
            .map_err(backend_error)
    }

    fn track_legacy(&mut self, api: &Api, context: &Context) -> Result<(), CuResult> {
        if !self.track_legacy {
            if self.get(0)?.dirty {
                // Capture any default work submitted before the first blocking
                // stream existed. The marker is itself an asynchronous append.
                let completion = self.next_fence(api, context)?;
                let legacy = self.get_mut(0)?;
                legacy.record(
                    api,
                    context,
                    Some(completion),
                    Vec::new(),
                    |list, signal, _| list.signal(api, signal.unwrap()),
                )?;
            }
            self.track_legacy = true;
        }
        Ok(())
    }

    pub(super) fn launch(
        &mut self,
        api: &Api,
        context: &Context,
        kernel: Handle,
        key: usize,
        geometry: ([u32; 3], [u32; 3]),
        args: &[KernelArg],
    ) -> Result<(), CuResult> {
        // Level Zero rejects further appends with OUT_OF_DEVICE_MEMORY when too much large-grid
        // work is queued and unfinished, so bound the outstanding work-groups per stream.
        let groups: u64 = geometry.0.iter().map(|&g| u64::from(g)).product();
        let limit = inflight_group_limit();
        let queued = self.get(key)?.queued_groups;
        if limit != 0 && queued != 0 && should_drain(queued, groups, limit) {
            self.wait_stream(api, key, u64::MAX)?;
        }
        let result = self.record_ordered(api, context, key, |list, signal, waits| {
            list.launch(api, kernel, geometry, args, signal, waits)
        });
        if result.is_ok() {
            let stream = self.get_mut(key)?;
            stream.queued_groups = stream.queued_groups.saturating_add(groups);
        }
        result
    }

    pub(super) unsafe fn copy(
        &mut self,
        api: &Api,
        context: &Context,
        key: usize,
        pointers: (Opaque, *const std::ffi::c_void),
        size: usize,
    ) -> Result<(), CuResult> {
        self.record_ordered(api, context, key, |list, signal, waits| unsafe {
            list.copy(api, pointers, size, signal, waits)
        })
    }

    pub(super) fn fill(
        &mut self,
        api: &Api,
        context: &Context,
        key: usize,
        fill: (Opaque, u8, usize),
    ) -> Result<(), CuResult> {
        self.record_ordered(api, context, key, |list, signal, waits| {
            list.fill(api, fill, signal, waits)
        })
    }

    fn record_ordered(
        &mut self,
        api: &Api,
        context: &Context,
        key: usize,
        append: impl FnOnce(
            &CommandList,
            Option<&Fence>,
            &[Arc<Fence>],
        ) -> Result<(), crate::level_zero::Error>,
    ) -> Result<(), CuResult> {
        let blocking = self.get(key)?.blocking;
        self.collect_retired(api)?;
        let signal = if blocking && self.track_legacy {
            Some(self.next_fence(api, context)?)
        } else {
            None
        };
        let waits = self.legacy_waits(api, key, blocking)?;
        self.get_mut(key)?.record(api, context, signal, waits, append)
    }

    fn legacy_waits(
        &mut self,
        api: &Api,
        key: usize,
        blocking: bool,
    ) -> Result<Vec<Arc<Fence>>, CuResult> {
        if key == 0 {
            let mut waits = Vec::new();
            for stream in self
                .live
                .values_mut()
                .chain(&mut self.retired)
                .filter(|stream| stream.blocking)
            {
                stream.flush(api)?;
                waits.extend(stream.tail());
            }
            Ok(waits)
        } else if blocking {
            let legacy = self.get_mut(0)?;
            legacy.flush(api)?;
            Ok(legacy.tail().into_iter().collect())
        } else {
            Ok(Vec::new())
        }
    }

    fn flush_producer(
        &mut self,
        api: &Api,
        producer: usize,
        fence: &Arc<Fence>,
    ) -> Result<(), CuResult> {
        // A removed producer submitted its last list before retirement.
        if let Ok(stream) = self.get_mut(producer) {
            stream.check()?;
            if let Some(batch) = &mut stream.batch
                && batch.contains_recording(fence)
            {
                batch.flush(api)?;
            }
        }
        Ok(())
    }

    pub(super) fn launch_key(&self, handle: Opaque) -> Result<usize, CuResult> {
        let key = stream_key(handle)?;
        self.get(key)?;
        Ok(key)
    }

    pub(super) fn capture_slot(&mut self, key: usize) -> Result<&mut Option<Capture>, CuResult> {
        Ok(&mut self.get_mut(key)?.capture)
    }

    pub(super) fn capture(&mut self, key: usize) -> Result<Option<&mut Capture>, CuResult> {
        Ok(self.capture_slot(key)?.as_mut())
    }

    pub(super) fn wait_legacy(&mut self, api: &Api, timeout: u64) -> Result<(), CuResult> {
        if timeout != 0 {
            for stream in self.all_mut().filter(|stream| stream.blocking) {
                stream.flush(api)?;
            }
        }
        for stream in self.all_mut().filter(|stream| stream.blocking) {
            stream.wait(api, timeout)?;
        }
        Ok(())
    }

    /// Flush and wait a specific stream (including non-blocking). Used when a
    /// pageable `cudaMemcpyAsync` falls back to a host-synchronous copy and must
    /// still observe prior work queued on that stream.
    pub(super) fn wait_stream(
        &mut self,
        api: &Api,
        key: usize,
        timeout: u64,
    ) -> Result<(), CuResult> {
        let stream = self.get_mut(key)?;
        stream.flush(api)?;
        stream.wait(api, timeout)?;
        Ok(())
    }

    pub(super) fn stream_is_blocking(&self, key: usize) -> Result<bool, CuResult> {
        Ok(self.get(key)?.blocking)
    }

    pub(super) fn wait_all(&mut self, api: &Api) -> Result<(), CuResult> {
        let mut flushed = Ok(());
        for stream in self.all_mut() {
            flushed = flushed.and(stream.flush(api));
        }
        flushed?;
        for stream in self.all_mut() {
            stream.wait(api, u64::MAX)?;
        }
        self.collect_retired(api)
    }

    pub(super) fn drain_for_teardown(&mut self, api: &Api) -> Result<(), CuResult> {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut execution_error = self.execution_error;
        for stream in self.all_mut() {
            if let Err(error) = stream.flush(api) {
                execution_error.get_or_insert(error);
            }
        }
        self.execution_error = execution_error;
        // Flush every producer before waiting on any queue. Failed open lists
        // stay owned and inaccessible to consumers until teardown destroys them.
        for stream in self.all_mut() {
            let left = deadline.saturating_duration_since(Instant::now());
            stream.drain(api, left.as_nanos() as u64)?;
        }
        Ok(())
    }

    pub(super) fn execution_error(&self) -> Option<CuResult> {
        self.execution_error
    }

    pub(super) fn destroy(&mut self, api: &Api) -> Result<(), CuResult> {
        if let Some(legacy) = &mut self.legacy {
            legacy.destroy(api)?;
            self.legacy = None;
        }
        while let Some((&key, stream)) = self.live.iter_mut().next() {
            stream.destroy(api)?;
            self.live.remove(&key);
        }
        while let Some(stream) = self.retired.last_mut() {
            stream.destroy(api)?;
            self.retired.pop();
        }
        self.events.clear();
        self.times.clear();
        self.captured.clear();
        self.fences.destroy(api).map_err(backend_error)
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuStreamCreate(out: *mut Opaque, flags: u32) -> CuResult {
    trace::call("cuStreamCreate", || {
        status(state(|state| {
            let owner = current().ok_or(InvalidContext)?;
            let runtime = &state.runtime;
            let next = &mut state.next_handle;
            let context = state.contexts.get_mut(&owner).ok_or(ContextIsDestroyed)?;
            context.check_live()?;
            if out.is_null() || flags > 1 {
                return Err(InvalidValue);
            }
            context.streams.collect_retired(&runtime.api)?;
            let key = reserve_handle(next)?;
            if flags == 0 {
                context
                    .streams
                    .track_legacy(&runtime.api, &context.backend)?;
            }
            let streams = &mut context.streams;
            let stream = Stream::new(
                &runtime.api,
                &context.backend,
                flags == 0,
                streams.policy,
                &streams.wake,
            )?;
            streams.live.insert(key, stream);
            state.stream_owners.insert(key, owner);
            extra::note_stream(key, flags);
            unsafe { out.write(std::ptr::without_provenance_mut(key)) };
            Ok(())
        }))
    })
}

// Explicit stream handles identify their owning context independently of the calling
// thread's current context. Launch additionally requires the function's context.
fn stream_owner(state: &State, handle: Opaque) -> Result<(usize, usize), CuResult> {
    let key = stream_key(handle)?;
    let owner = if key == 0 {
        current().ok_or(InvalidContext)?
    } else {
        match state.stream_owners.get(&key) {
            Some(&owner)
                if state
                    .contexts
                    .get(&owner)
                    .is_some_and(|context| context.streams.live.contains_key(&key)) =>
            {
                owner
            }
            _ => return Err(InvalidHandle),
        }
    };
    Ok((owner, key))
}

fn with_stream(
    handle: Opaque,
    action: impl FnOnce(&Api, &mut Streams, usize) -> Result<(), CuResult>,
) -> Result<(), CuResult> {
    state(|state| {
        let (owner, key) = stream_owner(state, handle)?;
        let context = state.contexts.get_mut(&owner).ok_or(ContextIsDestroyed)?;
        context.check_live()?;
        action(&state.runtime.api, &mut context.streams, key)
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuStreamDestroy_v2(handle: Opaque) -> CuResult {
    trace::call("cuStreamDestroy_v2", || {
        let result = status(with_stream(handle, |api, streams, key| {
            if key == 0 {
                return Err(InvalidHandle);
            }
            streams.collect_retired(api)?;
            let stream = streams.get_mut(key)?;
            stream.flush(api)?;
            let ready = if stream.has_waiters() {
                Err(NotReady)
            } else {
                stream.wait(api, 0)
            };
            let cleanup = match ready {
                Ok(()) => stream.destroy(api),
                Err(NotReady) => Err(NotReady),
                Err(error) => return Err(error),
            };
            let stream = streams.live.remove(&key).ok_or(InvalidHandle)?;
            if cleanup.is_err() {
                streams.retired.push(stream);
            }
            if ready == Err(NotReady) {
                Ok(())
            } else {
                cleanup
            }
        }));
        // Drop the owner entry once the stream has left its context's live set.
        if let Ok(key) = stream_key(handle) {
            let _ = state(|state| {
                if let Some(&owner) = state.stream_owners.get(&key)
                    && !state
                        .contexts
                        .get(&owner)
                        .is_some_and(|context| context.streams.live.contains_key(&key))
                {
                    state.stream_owners.remove(&key);
                }
                Ok(())
            });
        }
        result
    })
}

fn synchronize(
    target: impl FnOnce(&State) -> Result<(usize, Option<usize>), CuResult>,
) -> Result<(), CuResult> {
    synchronize_with(target, false)
}

fn synchronize_with(
    target: impl FnOnce(&State) -> Result<(usize, Option<usize>), CuResult>,
    skip_capture: bool,
) -> Result<(), CuResult> {
    let (owner, lists, immediate_wait, fence_wait, health) = state(|state| {
        let (owner, key) = target(state)?;
        let context = state.contexts.get_mut(&owner).ok_or(ContextIsDestroyed)?;
        context.check_live()?;
        context.streams.collect_retired(&state.runtime.api)?;
        Ok((
            owner,
            context.streams.snapshot(&state.runtime.api, key, skip_capture)?,
            state.runtime.api.stream_wait_fn(),
            state.runtime.api.fence_wait_fn(),
            Arc::clone(&context.backend.health),
        ))
    })?;
    if lists.is_empty() {
        return Ok(());
    }
    // Each snapshot keeps its list alive across destruction of the public stream
    // handle. The runtime and its Level Zero loader outlive all API calls.
    for completion in &lists {
        match completion {
            Completion::Immediate(list, _) => list.wait_with(immediate_wait, u64::MAX),
            Completion::Regular(fence) => fence.wait(fence_wait, u64::MAX),
        }
        .inspect_err(|error| health.observe(error))
        .map_err(backend_error)?;
    }
    state(move |state| {
        let context = state.contexts.get_mut(&owner).ok_or(ContextIsDestroyed)?;
        for stream in context.streams.all_mut() {
            if let Some(immediate) = &stream.immediate
                && lists.iter().any(|completion| {
                    matches!(completion,
                        Completion::Immediate(list, generation)
                            if Arc::ptr_eq(list, immediate) && *generation == stream.generation
                    )
                })
            {
                stream.complete();
            }
        }
        drop(lists);
        context.streams.collect_retired(&state.runtime.api)?;
        Ok(())
    })
}

/// Wait for the current context's streams (`None` = all, `Some(0)` = blocking
/// streams, `Some(key)` = one stream) with the runtime lock released, so other
/// threads are not stalled behind a GPU wait. Best effort and capture-safe:
/// callers still perform their own in-lock wait afterwards, which is then a
/// no-op for work this call drained and still covers work enqueued meanwhile.
pub(super) fn drain_unlocked(key: Option<usize>) {
    let _ = synchronize_with(
        |_| Ok((current().ok_or(InvalidContext)?, key)),
        true,
    );
}

pub(super) fn synchronize_context(handle: Opaque) -> CuResult {
    status(synchronize(|state| {
        let key = context_key(handle)?;
        if !handle.is_null() && !state.contexts.contains_key(&key) {
            return Err(InvalidContext);
        }
        Ok((key, None))
    }))
}

#[unsafe(no_mangle)]
pub extern "system" fn cuStreamSynchronize(handle: Opaque) -> CuResult {
    trace::call("cuStreamSynchronize", || {
        status(synchronize(|state| {
            let (owner, key) = stream_owner(state, handle)?;
            Ok((owner, Some(key)))
        }))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuStreamQuery(handle: Opaque) -> CuResult {
    trace::call("cuStreamQuery", || {
        status(with_stream(handle, |api, streams, key| {
            if let Some(capture) = streams.capture(key)? {
                return Err(capture.invalidate());
            }
            if key == 0 {
                streams.wait_legacy(api, 0)
            } else {
                streams.get_mut(key)?.wait(api, 0)
            }
        }))
    })
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;
    use crate::cuda::{
        cuCtxCreate_v2, cuCtxDestroy_v2, cuCtxSynchronize, cuInit, cuLaunchKernel, cuMemAlloc_v2,
        cuMemFreeHost, cuMemHostAlloc, cuMemcpyDtoHAsync_v2, cuMemcpyHtoDAsync_v2,
        cuModuleGetFunction, cuModuleLoadData, test_device,
    };
    use std::ptr::null_mut;

    #[test]
    #[ignore = "requires an Intel Level Zero GPU"]
    fn gated_stream_stays_pending_without_blocking_other_sync() {
        assert_eq!(cuInit(0), Success);
        let mut context = null_mut();
        assert_eq!(
            unsafe { cuCtxCreate_v2(&mut context, 0, test_device()) },
            Success
        );
        let (mut gated, mut idle) = (null_mut(), null_mut());
        assert_eq!(unsafe { cuStreamCreate(&mut gated, 1) }, Success);
        assert_eq!(unsafe { cuStreamCreate(&mut idle, 0) }, Success);
        let (mut src, mut dst, mut device) = (null_mut(), null_mut(), 0);
        assert_eq!(unsafe { cuMemHostAlloc(&mut src, 1, 0) }, Success);
        assert_eq!(unsafe { cuMemHostAlloc(&mut dst, 1, 0) }, Success);
        assert_eq!(unsafe { cuMemAlloc_v2(&mut device, 1) }, Success);
        unsafe {
            src.cast::<u8>().write(42);
            dst.cast::<u8>().write(0);
        }
        let ptx = c".version 4.1 .target sm_52 .address_size 64 .entry noop() { ret; }";
        let (mut module, mut kernel) = (null_mut(), null_mut());
        assert_eq!(
            unsafe { cuModuleLoadData(&mut module, ptx.as_ptr().cast()) },
            Success
        );
        assert_eq!(
            unsafe { cuModuleGetFunction(&mut kernel, module, c"noop".as_ptr()) },
            Success
        );

        let gate = super::super::with_context(|runtime, context| {
            let api = &runtime.api;
            let gate = context
                .streams
                .fences
                .acquire(api, &context.backend)
                .map_err(backend_error)?;
            let stream = context.streams.get_mut(gated.addr())?;
            stream.record(api, &context.backend, None, Vec::new(), |list, _, _| {
                list.gate(api, &gate)
            })?;
            Ok(gate)
        })
        .unwrap();

        // No kernel duration or elapsed-time assumptions. Only the host signal
        // below can satisfy the device wait.
        assert_eq!(
            unsafe { cuMemcpyHtoDAsync_v2(device, src, 1, gated) },
            Success
        );
        assert_eq!(
            unsafe { cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, gated, null_mut(), null_mut()) },
            Success
        );
        assert_eq!(
            unsafe { cuMemcpyDtoHAsync_v2(dst, device, 1, gated) },
            Success
        );
        assert_eq!(cuStreamQuery(gated), NotReady);
        assert_eq!(cuStreamQuery(null_mut()), Success);
        assert_eq!(cuStreamSynchronize(idle), Success);
        assert_eq!(cuStreamSynchronize(null_mut()), Success);
        let gated_key = gated.addr();
        std::thread::scope(|scope| {
            let waiter = scope
                .spawn(move || cuStreamSynchronize(std::ptr::without_provenance_mut(gated_key)));
            while !super::super::with_context(|_, context| Ok(context.streams.has_waiters()))
                .unwrap()
            {
                assert!(
                    !waiter.is_finished(),
                    "wait returned before the gate opened"
                );
                std::thread::yield_now();
            }
            assert_eq!(cuStreamQuery(null_mut()), Success);
            assert_eq!(cuStreamDestroy_v2(gated), Success);
            assert_eq!(cuStreamQuery(gated), InvalidHandle);
            super::super::with_context(|runtime, context| {
                assert_eq!(context.streams.retired.len(), 1);
                gate.host_signal(&runtime.api).map_err(backend_error)
            })
            .unwrap();
            assert_eq!(waiter.join().unwrap(), Success);
        });
        assert_eq!(cuCtxSynchronize(), Success);
        assert_eq!(unsafe { dst.cast::<u8>().read() }, 42);
        assert_eq!(cuMemFreeHost(src), Success);
        assert_eq!(cuMemFreeHost(dst), Success);
        drop(gate);
        super::super::with_context(|_, context| {
            assert!(context.streams.retired.is_empty());
            Ok(())
        })
        .unwrap();
        for _ in 0..16 {
            let mut stream = null_mut();
            assert_eq!(unsafe { cuStreamCreate(&mut stream, 0) }, Success);
            for _ in 0..16 {
                assert_eq!(
                    unsafe {
                        cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, stream, null_mut(), null_mut())
                    },
                    Success
                );
            }
            assert_eq!(cuStreamSynchronize(stream), Success);
            assert_eq!(cuStreamDestroy_v2(stream), Success);
        }
        super::super::with_context(|_, context| {
            assert_eq!(context.streams.fences.pool_count(), 1);
            assert_eq!(context.streams.live.len(), 1);
            assert!(context.streams.retired.is_empty());
            Ok(())
        })
        .unwrap();
        assert_eq!(cuStreamDestroy_v2(idle), Success);
        assert_eq!(cuCtxDestroy_v2(context), Success);
    }
}

/// Default cap on work-groups queued on one stream before waiting for it to drain.
const DEFAULT_MAX_INFLIGHT_GROUPS: u64 = 65_536;

/// `CUDA_FOR_INTEL_MAX_INFLIGHT_GROUPS`: 0 disables the window.
fn inflight_group_limit() -> u64 {
    static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("CUDA_FOR_INTEL_MAX_INFLIGHT_GROUPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_MAX_INFLIGHT_GROUPS)
    })
}

fn should_drain(queued: u64, next: u64, limit: u64) -> bool {
    queued.saturating_add(next) > limit
}

#[cfg(test)]
mod inflight_window_tests {
    use super::*;

    #[test]
    fn drains_only_when_outstanding_groups_would_exceed_the_limit() {
        assert!(!should_drain(0, 14_336, 65_536));
        assert!(!should_drain(14_336 * 3, 14_336, 65_536));
        assert!(should_drain(14_336 * 4, 14_336, 65_536));
        assert!(!should_drain(60_000, 1, 65_536));
        assert!(should_drain(u64::MAX, 1, 65_536));
        // many tiny launches still queue deeply
        assert!(!should_drain(50_000, 1, 65_536));
    }
}
