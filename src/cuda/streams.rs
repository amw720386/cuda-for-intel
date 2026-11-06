use super::CuResult::*;
use super::{
    Api, Context, CuResult, Opaque, State, backend_error, current, reserve_handle, state, status,
    with_context_handles,
};
use crate::level_zero::{CommandList, Fence, Fences, Module};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

mod events;

const LEGACY: usize = 1;
const PER_THREAD: usize = 2;

fn stream_key(handle: Opaque) -> Result<usize, CuResult> {
    match handle.addr() {
        0 | LEGACY => Ok(0),
        PER_THREAD => Err(NotSupported),
        key => Ok(key),
    }
}

struct Submission {
    completion: Arc<Fence>,
    _waits: Vec<Arc<Fence>>,
}

struct Stream {
    list: Arc<CommandList>,
    blocking: bool,
    dirty: bool,
    generation: u64,
    pending: VecDeque<Submission>,
}

impl Stream {
    fn new(list: CommandList, blocking: bool) -> Self {
        Self {
            list: Arc::new(list),
            blocking,
            dirty: false,
            generation: 0,
            pending: VecDeque::new(),
        }
    }

    fn tail(&self) -> Option<Arc<Fence>> {
        self.pending
            .back()
            .map(|pending| Arc::clone(&pending.completion))
    }

    fn reap(&mut self, api: &Api) -> Result<(), CuResult> {
        while let Some(pending) = self.pending.front() {
            if !pending.completion.ready(api).map_err(backend_error)? {
                break;
            }
            self.pending.pop_front();
        }
        Ok(())
    }

    fn wait(&mut self, api: &Api, timeout: u64) -> Result<(), CuResult> {
        if self.dirty {
            self.list.wait(api, timeout).map_err(backend_error)?;
            self.complete();
        }
        Ok(())
    }

    fn submitted(&mut self) {
        self.dirty = true;
        self.generation = self.generation.wrapping_add(1);
    }

    fn complete(&mut self) {
        self.pending.clear();
        self.dirty = false;
    }

    fn has_waiters(&self) -> bool {
        Arc::strong_count(&self.list) > 1
    }
}

pub(super) struct Streams {
    legacy: Option<Stream>,
    live: HashMap<usize, Stream>,
    retired: Vec<Stream>,
    fences: Fences,
    events: HashMap<usize, Option<Arc<Fence>>>,
    event_waiters: Arc<()>,
    track_legacy: bool,
}

impl Streams {
    pub(super) fn new(api: &Api, context: &Context) -> Result<Self, CuResult> {
        let list = context.create_stream(api).map_err(backend_error)?;
        Ok(Self {
            legacy: Some(Stream::new(list, true)),
            live: HashMap::new(),
            retired: Vec::new(),
            fences: Fences::default(),
            events: HashMap::new(),
            event_waiters: Arc::new(()),
            track_legacy: false,
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

    fn snapshot(&self, key: Option<usize>) -> Result<Vec<(Arc<CommandList>, u64)>, CuResult> {
        if let Some(key) = key.filter(|&key| key != 0) {
            let stream = self.get(key)?;
            return Ok(stream
                .dirty
                .then(|| (Arc::clone(&stream.list), stream.generation))
                .into_iter()
                .collect());
        }
        Ok(self
            .legacy
            .iter()
            .chain(self.explicit())
            .filter(|stream| stream.dirty && (key.is_none() || stream.blocking))
            .map(|stream| (Arc::clone(&stream.list), stream.generation))
            .collect())
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
                    stream.list.destroy(api).map_err(backend_error)?;
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
        self.fences.acquire(api, context).map_err(backend_error)
    }

    fn track_legacy(&mut self, api: &Api, context: &Context) -> Result<(), CuResult> {
        if !self.track_legacy {
            if self.get(0)?.dirty {
                // Capture any default work submitted before the first blocking
                // stream existed. The marker is itself an asynchronous append.
                let completion = self.next_fence(api, context)?;
                let legacy = self.get_mut(0)?;
                legacy
                    .list
                    .signal(api, &completion)
                    .map_err(backend_error)?;
                legacy.submitted();
                legacy.pending.push_back(Submission {
                    completion,
                    _waits: Vec::new(),
                });
            }
            self.track_legacy = true;
        }
        Ok(())
    }

    pub(super) fn launch(
        &mut self,
        api: &Api,
        context: &Context,
        module: &Module,
        key: usize,
        geometry: ([u32; 3], [u32; 3]),
        args: &[&[u8]],
    ) -> Result<(), CuResult> {
        let blocking = self.get(key)?.blocking;
        self.collect_retired(api)?;
        let signal = if blocking && self.track_legacy {
            Some(self.next_fence(api, context)?)
        } else {
            None
        };
        let waits = self.legacy_waits(key, blocking);
        let stream = self.get_mut(key)?;
        stream
            .list
            .launch(api, module, geometry, args, signal.as_deref(), &waits)
            .map_err(backend_error)?;
        stream.submitted();
        if let Some(completion) = signal {
            stream.pending.push_back(Submission {
                completion,
                _waits: waits,
            });
        }
        Ok(())
    }

    pub(super) unsafe fn copy(
        &mut self,
        api: &Api,
        context: &Context,
        key: usize,
        pointers: (Opaque, *const std::ffi::c_void),
        size: usize,
    ) -> Result<(), CuResult> {
        let blocking = self.get(key)?.blocking;
        self.collect_retired(api)?;
        let signal = if blocking && self.track_legacy {
            Some(self.next_fence(api, context)?)
        } else {
            None
        };
        let waits = self.legacy_waits(key, blocking);
        let stream = self.get_mut(key)?;
        unsafe {
            stream
                .list
                .copy(api, pointers, size, signal.as_deref(), &waits)
        }
        .map_err(backend_error)?;
        stream.submitted();
        if let Some(completion) = signal {
            stream.pending.push_back(Submission {
                completion,
                _waits: waits,
            });
        }
        Ok(())
    }

    fn legacy_waits(&self, key: usize, blocking: bool) -> Vec<Arc<Fence>> {
        if key == 0 {
            self.explicit()
                .filter(|stream| stream.blocking)
                .filter_map(Stream::tail)
                .collect()
        } else if blocking {
            self.legacy
                .as_ref()
                .and_then(Stream::tail)
                .into_iter()
                .collect()
        } else {
            Vec::new()
        }
    }

    pub(super) fn launch_key(&self, handle: Opaque) -> Result<usize, CuResult> {
        let key = stream_key(handle)?;
        self.get(key)?;
        Ok(key)
    }

    pub(super) fn wait_legacy(&mut self, api: &Api, timeout: u64) -> Result<(), CuResult> {
        for stream in self.all_mut().filter(|stream| stream.blocking) {
            stream.wait(api, timeout)?;
        }
        Ok(())
    }

    pub(super) fn wait_all(&mut self, api: &Api) -> Result<(), CuResult> {
        for stream in self.all_mut() {
            stream.wait(api, u64::MAX)?;
        }
        self.collect_retired(api)
    }

    pub(super) fn destroy(&mut self, api: &Api) -> Result<(), CuResult> {
        if let Some(legacy) = &self.legacy {
            legacy.list.destroy(api).map_err(backend_error)?;
            self.legacy = None;
        }
        while let Some((&key, stream)) = self.live.iter().next() {
            stream.list.destroy(api).map_err(backend_error)?;
            self.live.remove(&key);
        }
        self.events.clear();
        self.fences.destroy(api).map_err(backend_error)
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuStreamCreate(out: *mut Opaque, flags: u32) -> CuResult {
    status(with_context_handles(|runtime, context, next| {
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
        let list = context
            .backend
            .create_stream(&runtime.api)
            .map_err(backend_error)?;
        context
            .streams
            .live
            .insert(key, Stream::new(list, flags == 0));
        unsafe { out.write(std::ptr::without_provenance_mut(key)) };
        Ok(())
    }))
}

// Explicit stream handles identify their owning context independently of the calling
// thread's current context. Launch additionally requires the function's context.
fn stream_owner(state: &State, handle: Opaque) -> Result<(usize, usize), CuResult> {
    let key = stream_key(handle)?;
    let owner = if key == 0 {
        current().ok_or(InvalidContext)?
    } else {
        state
            .contexts
            .iter()
            .find(|(_, context)| context.streams.live.contains_key(&key))
            .map(|(&owner, _)| owner)
            .ok_or(InvalidHandle)?
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
        action(&state.runtime.api, &mut context.streams, key)
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuStreamDestroy_v2(handle: Opaque) -> CuResult {
    status(with_stream(handle, |api, streams, key| {
        if key == 0 {
            return Err(InvalidHandle);
        }
        streams.collect_retired(api)?;
        let stream = streams.get_mut(key)?;
        let ready = if stream.has_waiters() {
            Err(NotReady)
        } else {
            stream.wait(api, 0)
        };
        match ready {
            Ok(()) => {
                stream.list.destroy(api).map_err(backend_error)?;
                streams.live.remove(&key);
            }
            Err(NotReady) => {
                let stream = streams.live.remove(&key).ok_or(InvalidHandle)?;
                streams.retired.push(stream);
            }
            Err(error) => return Err(error),
        }
        Ok(())
    }))
}

fn synchronize(
    target: impl FnOnce(&State) -> Result<(usize, Option<usize>), CuResult>,
) -> Result<(), CuResult> {
    let (owner, lists, wait) = state(|state| {
        let (owner, key) = target(state)?;
        let context = state.contexts.get_mut(&owner).ok_or(ContextIsDestroyed)?;
        context.streams.collect_retired(&state.runtime.api)?;
        Ok((
            owner,
            context.streams.snapshot(key)?,
            state.runtime.api.stream_wait_fn(),
        ))
    })?;
    if lists.is_empty() {
        return Ok(());
    }
    // Each snapshot keeps its list alive across destruction of the public stream
    // handle. The runtime and its Level Zero loader outlive all API calls.
    for (list, _) in &lists {
        list.wait_with(wait, u64::MAX).map_err(backend_error)?;
    }
    state(move |state| {
        let context = state.contexts.get_mut(&owner).ok_or(ContextIsDestroyed)?;
        for stream in context.streams.all_mut() {
            if lists.iter().any(|(list, generation)| {
                Arc::ptr_eq(list, &stream.list) && *generation == stream.generation
            }) {
                stream.complete();
            }
        }
        drop(lists);
        context.streams.collect_retired(&state.runtime.api)?;
        Ok(())
    })
}

pub(super) fn synchronize_context() -> CuResult {
    status(synchronize(|_| {
        Ok((current().ok_or(InvalidContext)?, None))
    }))
}

#[unsafe(no_mangle)]
pub extern "system" fn cuStreamSynchronize(handle: Opaque) -> CuResult {
    status(synchronize(|state| {
        let (owner, key) = stream_owner(state, handle)?;
        Ok((owner, Some(key)))
    }))
}

#[unsafe(no_mangle)]
pub extern "system" fn cuStreamQuery(handle: Opaque) -> CuResult {
    status(with_stream(handle, |api, streams, key| {
        if key == 0 {
            streams.wait_legacy(api, 0)
        } else {
            streams.get_mut(key)?.wait(api, 0)
        }
    }))
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;
    use crate::cuda::{
        cuCtxCreate_v2, cuCtxDestroy_v2, cuCtxSynchronize, cuInit, cuLaunchKernel, cuMemAlloc_v2,
        cuMemFreeHost, cuMemHostAlloc, cuMemcpyDtoHAsync_v2, cuMemcpyHtoDAsync_v2,
        cuModuleGetFunction, cuModuleLoadData,
    };
    use std::ptr::null_mut;

    #[test]
    #[ignore = "requires an Intel Level Zero GPU"]
    fn gated_stream_stays_pending_without_blocking_other_sync() {
        assert_eq!(cuInit(0), Success);
        let mut context = null_mut();
        assert_eq!(unsafe { cuCtxCreate_v2(&mut context, 0, 0) }, Success);
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
            stream.list.gate(api, &gate).map_err(backend_error)?;
            stream.submitted();
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
