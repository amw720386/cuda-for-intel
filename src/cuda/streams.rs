use super::CuResult::*;
use super::{
    Api, Context, CuResult, Opaque, backend_error, reserve_handle, state, status,
    with_context_handles,
};
use crate::level_zero::{CommandList, Fence, Fences, Module};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

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
    list: CommandList,
    blocking: bool,
    dirty: bool,
    pending: VecDeque<Submission>,
}

impl Stream {
    fn new(list: CommandList, blocking: bool) -> Self {
        Self {
            list,
            blocking,
            dirty: false,
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
            self.pending.clear();
            self.dirty = false;
        }
        Ok(())
    }
}

pub(super) struct Streams {
    legacy: Option<Stream>,
    live: HashMap<usize, Stream>,
    retired: Vec<Stream>,
    fences: Fences,
}

impl Streams {
    pub(super) fn new(api: &Api, context: &Context) -> Result<Self, CuResult> {
        let list = context.create_stream(api).map_err(backend_error)?;
        Ok(Self {
            legacy: Some(Stream::new(list, true)),
            live: HashMap::new(),
            retired: Vec::new(),
            fences: Fences::default(),
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

    fn collect_retired(&mut self, api: &Api) -> Result<(), CuResult> {
        let mut index = 0;
        while index < self.retired.len() {
            let stream = &mut self.retired[index];
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
        let signal = if blocking {
            if self.fences.needs_refill() {
                for stream in self.all_mut() {
                    stream.reap(api)?;
                }
            }
            Some(self.fences.acquire(api, context).map_err(backend_error)?)
        } else {
            None
        };
        let waits = if key == 0 {
            self.explicit()
                .filter(|stream| stream.blocking)
                .filter_map(Stream::tail)
                .collect()
        } else if blocking {
            self.get(0)?.tail().into_iter().collect()
        } else {
            Vec::new()
        };
        let stream = self.get_mut(key)?;
        stream
            .list
            .launch(api, module, geometry, args, signal.as_deref(), &waits)
            .map_err(backend_error)?;
        stream.dirty = true;
        if let Some(completion) = signal {
            stream.pending.push_back(Submission {
                completion,
                _waits: waits,
            });
        }
        Ok(())
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
fn with_stream(
    handle: Opaque,
    action: impl FnOnce(&Api, &mut Streams, usize) -> Result<(), CuResult>,
) -> Result<(), CuResult> {
    state(|state| {
        let key = stream_key(handle)?;
        let context = if key == 0 {
            let key = super::current().ok_or(InvalidContext)?;
            state.contexts.get_mut(&key).ok_or(ContextIsDestroyed)?
        } else {
            state
                .contexts
                .values_mut()
                .find(|context| context.streams.live.contains_key(&key))
                .ok_or(InvalidHandle)?
        };
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
        match stream.wait(api, 0) {
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

fn stream_wait(handle: Opaque, timeout: u64) -> CuResult {
    status(with_stream(handle, |api, streams, key| {
        if key == 0 {
            streams.wait_legacy(api, timeout)?;
        } else {
            streams.get_mut(key)?.wait(api, timeout)?;
        }
        Ok(())
    }))
}

#[unsafe(no_mangle)]
pub extern "system" fn cuStreamSynchronize(handle: Opaque) -> CuResult {
    stream_wait(handle, u64::MAX)
}

#[unsafe(no_mangle)]
pub extern "system" fn cuStreamQuery(handle: Opaque) -> CuResult {
    stream_wait(handle, 0)
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;
    use crate::cuda::{cuCtxCreate_v2, cuCtxDestroy_v2, cuCtxSynchronize, cuInit};
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

        let gate = super::super::with_context(|runtime, context| {
            let api = &runtime.api;
            let gate = context
                .streams
                .fences
                .acquire(api, &context.backend)
                .map_err(backend_error)?;
            let stream = context.streams.get_mut(gated.addr())?;
            stream.list.gate(api, &gate).map_err(backend_error)?;
            stream.dirty = true;
            Ok(gate)
        })
        .unwrap();

        // No kernel duration or elapsed-time assumptions. Only the host signal
        // below can satisfy the device wait.
        assert_eq!(cuStreamQuery(gated), NotReady);
        assert_eq!(cuStreamQuery(null_mut()), Success);
        assert_eq!(cuStreamSynchronize(idle), Success);
        assert_eq!(cuStreamSynchronize(null_mut()), Success);
        assert_eq!(cuStreamDestroy_v2(gated), Success);
        assert_eq!(cuStreamQuery(gated), InvalidHandle);
        super::super::with_context(|runtime, context| {
            assert_eq!(context.streams.retired.len(), 1);
            gate.host_signal(&runtime.api).map_err(backend_error)
        })
        .unwrap();
        drop(gate);
        assert_eq!(cuCtxSynchronize(), Success);
        super::super::with_context(|_, context| {
            assert!(context.streams.retired.is_empty());
            Ok(())
        })
        .unwrap();
        assert_eq!(cuStreamDestroy_v2(idle), Success);
        assert_eq!(cuCtxDestroy_v2(context), Success);
    }
}
