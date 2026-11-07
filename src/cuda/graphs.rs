//! Single-stream relaxed capture and replay of supported CUDA operations.

use super::*;
use crate::level_zero::KernelArg;
use std::mem::MaybeUninit;
use std::sync::Arc;

#[derive(Default)]
pub(super) struct Graphs {
    graphs: HashMap<usize, Graph>,
    executable: HashMap<usize, Graph>,
}

#[derive(Clone)]
struct Graph {
    context: usize,
    kernels: Arc<[Kernel]>,
}

#[derive(Default)]
pub(super) struct Capture {
    invalidated: bool,
    kernels: Vec<Kernel>,
}

impl Capture {
    pub(super) fn push(&mut self, kernel: Kernel) -> Result<(), CuResult> {
        if self.invalidated {
            return Err(StreamCaptureInvalidated);
        }
        self.kernels.push(kernel);
        Ok(())
    }

    pub(super) fn invalidate(&mut self) -> CuResult {
        self.invalidated = true;
        StreamCaptureUnsupported
    }
}

pub(super) struct Kernel {
    function: usize,
    geometry: ([u32; 3], [u32; 3]),
    args: Vec<Box<[MaybeUninit<u8>]>>,
}

impl Kernel {
    pub(super) unsafe fn capture(
        function: usize,
        geometry: ([u32; 3], [u32; 3]),
        args: &[KernelArg],
    ) -> Self {
        let args = args
            .iter()
            .map(|arg| {
                // Aggregate parameters can contain uninitialized padding.
                let mut bytes = Box::<[u8]>::new_uninit_slice(arg.size);
                unsafe {
                    std::ptr::copy_nonoverlapping(arg.value.cast(), bytes.as_mut_ptr(), arg.size)
                };
                bytes
            })
            .collect();
        Self {
            function,
            geometry,
            args,
        }
    }

    fn entry<'a>(&self, context: &'a CudaContext) -> Result<&'a CudaEntry, CuResult> {
        let (module, name) = context.functions.get(&self.function).ok_or(InvalidHandle)?;
        let module = context.modules.get(module).ok_or(InvalidHandle)?;
        module.entries.get(name).ok_or(InvalidHandle)
    }

    fn validate(&self, context: &CudaContext) -> Result<(), CuResult> {
        let entry = self.entry(context)?;
        entry.function.ok_or(InvalidHandle)?;
        let mut index = 0;
        for param in &entry.params {
            if *param == Param::GlobalPtr {
                let address = unsafe { self.args[index].as_ptr().cast::<u64>().read_unaligned() };
                if address != 0 && !contains_range(&context.allocations, address as usize, 1) {
                    return Err(InvalidValue);
                }
            }
            index += match param {
                Param::Words { words, .. } => usize::from(*words),
                _ => 1,
            };
        }
        Ok(())
    }

    fn replay(&self, api: &Api, context: &mut CudaContext, stream: usize) -> Result<(), CuResult> {
        let (_, kernel) = self.entry(context)?.function.ok_or(InvalidHandle)?;
        let args: Vec<_> = self
            .args
            .iter()
            .map(|arg| KernelArg {
                value: arg.as_ptr().cast(),
                size: arg.len(),
            })
            .collect();
        context
            .streams
            .launch(api, &context.backend, kernel, stream, self.geometry, &args)
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn cuStreamBeginCapture_v2(stream: Opaque, mode: c_int) -> CuResult {
    trace::call("cuStreamBeginCapture_v2", || {
        status(with_context(|_, context| {
            let key = context.streams.launch_key(stream)?;
            if key == 0 || !matches!(mode, 0..=2) {
                return Err(InvalidValue);
            }
            if mode != 2 {
                return Err(NotSupported);
            }
            let capture = context.streams.capture_slot(key)?;
            if capture.is_some() {
                return Err(IllegalState);
            }
            *capture = Some(Capture::default());
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuStreamEndCapture(stream: Opaque, out: *mut Opaque) -> CuResult {
    trace::call("cuStreamEndCapture", || {
        if out.is_null() {
            return InvalidValue;
        }
        unsafe { out.write(std::ptr::null_mut()) };
        status(state(|state| {
            let owner = current().ok_or(InvalidContext)?;
            let context = state.contexts.get_mut(&owner).ok_or(ContextIsDestroyed)?;
            context.check_live()?;
            let key = context.streams.launch_key(stream)?;
            let slot = context.streams.capture_slot(key)?;
            let capture = slot.take().ok_or(IllegalState)?;
            if capture.invalidated {
                return Err(StreamCaptureInvalidated);
            }
            let graph = reserve_handle(&mut state.next_handle)?;
            state.graphs.graphs.insert(
                graph,
                Graph {
                    context: owner,
                    kernels: capture.kernels.into(),
                },
            );
            unsafe { out.write(std::ptr::without_provenance_mut(graph)) };
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuStreamIsCapturing(stream: Opaque, out: *mut c_int) -> CuResult {
    trace::call("cuStreamIsCapturing", || {
        if out.is_null() {
            return InvalidValue;
        }
        status(with_context(|_, context| {
            let key = context.streams.launch_key(stream)?;
            let mode = context
                .streams
                .capture(key)?
                .map_or(0, |capture| if capture.invalidated { 2 } else { 1 });
            unsafe { out.write(mode) };
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuGraphDestroy(graph: Opaque) -> CuResult {
    trace::call("cuGraphDestroy", || {
        status(state(|state| {
            state
                .graphs
                .graphs
                .remove(&graph.addr())
                .map(|_| ())
                .ok_or(InvalidHandle)
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphInstantiateWithFlags(
    out: *mut Opaque,
    graph: Opaque,
    flags: u64,
) -> CuResult {
    trace::call("cuGraphInstantiateWithFlags", || {
        if out.is_null() {
            return InvalidValue;
        }
        if flags != 0 {
            return NotSupported;
        }
        status(state(|state| {
            let graph = state
                .graphs
                .graphs
                .get(&graph.addr())
                .ok_or(InvalidHandle)?
                .clone();
            let handle = reserve_handle(&mut state.next_handle)?;
            state.graphs.executable.insert(handle, graph);
            unsafe { out.write(std::ptr::without_provenance_mut(handle)) };
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuGraphExecDestroy(graph: Opaque) -> CuResult {
    trace::call("cuGraphExecDestroy", || {
        status(state(|state| {
            state
                .graphs
                .executable
                .remove(&graph.addr())
                .map(|_| ())
                .ok_or(InvalidHandle)
        }))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuGraphLaunch(graph: Opaque, stream: Opaque) -> CuResult {
    trace::call("cuGraphLaunch", || {
        status(state(|state| {
            let graph = state
                .graphs
                .executable
                .get(&graph.addr())
                .ok_or(InvalidHandle)?;
            if current() != Some(graph.context) {
                return Err(InvalidContext);
            }
            let context = state
                .contexts
                .get_mut(&graph.context)
                .ok_or(ContextIsDestroyed)?;
            context.check_live()?;
            let key = context.streams.launch_key(stream)?;
            if let Some(capture) = context.streams.capture(key)? {
                return Err(capture.invalidate());
            }
            for kernel in &*graph.kernels {
                kernel.validate(context)?;
            }
            for kernel in &*graph.kernels {
                kernel.replay(&state.runtime.api, context, key)?;
            }
            Ok(())
        }))
    })
}
