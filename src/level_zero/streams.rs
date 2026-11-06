use super::{
    Api, Context, Error, EventDesc, EventPoolDesc, GroupCount, Handle, Header, QueueDesc, Raw,
    call, check, invoke,
};
use std::ptr::null_mut;
use std::sync::Arc;

pub(crate) struct CommandList(pub(super) Handle);

pub(crate) struct KernelArg {
    pub(crate) value: *const std::ffi::c_void,
    pub(crate) size: usize,
}

// Submissions remain serialized by the frontend. Level Zero permits immediate-list
// host synchronization from simultaneous threads.
unsafe impl Sync for CommandList {}

pub(crate) type WaitFn = unsafe extern "system" fn(Raw, u64) -> u32;

impl Api {
    pub(crate) fn stream_wait_fn(&self) -> WaitFn {
        self.zeCommandListHostSynchronize
    }

    pub(crate) fn event_wait_fn(&self) -> WaitFn {
        self.zeEventHostSynchronize
    }
}

impl Context {
    pub(crate) fn create_stream(&self, api: &Api) -> Result<CommandList, Error> {
        self.create_stream_at(api, self.ordinal, 1 << 1) // ZE_COMMAND_QUEUE_FLAG_IN_ORDER
    }

    #[cfg(all(test, windows))]
    pub(crate) fn create_copy_stream(&self, api: &Api, ordinal: u32) -> Result<CommandList, Error> {
        // Request the selected queue explicitly and preserve in-order execution.
        self.create_stream_at(api, ordinal, (1 << 1) | 1)
    }

    fn create_stream_at(&self, api: &Api, ordinal: u32, flags: u32) -> Result<CommandList, Error> {
        let desc = QueueDesc {
            header: Header::new(0xe),
            ordinal,
            index: 0,
            flags,
            mode: 2, // ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS
            priority: 0,
        };
        let mut list = null_mut();
        call("zeCommandListCreateImmediate (asynchronous)", || unsafe {
            (api.zeCommandListCreateImmediate)(self.handle.0, self.device.0, &desc, &mut list)
        })?;
        Ok(CommandList(Handle(list)))
    }
}

impl CommandList {
    #[cfg(all(test, windows))]
    pub(crate) unsafe fn append_timestamp(
        &self,
        clock: &super::profiling::Clock,
        dst: *mut u64,
    ) -> Result<(), Error> {
        unsafe { clock.append(self.0.0, dst) }
    }

    pub(crate) unsafe fn copy(
        &self,
        api: &Api,
        (dst, src): (Raw, *const std::ffi::c_void),
        size: usize,
        signal: Option<&Fence>,
        waits: &[Arc<Fence>],
    ) -> Result<(), Error> {
        let mut events = WaitEvents::new(waits)?;
        // Keep waits on the copy; move any completion signal onto the following
        // ranges barrier so host-scope fences observe destination coherence.
        call("zeCommandListAppendMemoryCopy", || unsafe {
            (api.zeCommandListAppendMemoryCopy)(
                self.0.0,
                dst,
                src,
                size,
                null_mut(),
                events.count,
                events.ptr(),
            )
        })?;
        let signal = {
            let signal = signal.map_or(null_mut(), |fence| fence.handle.0);
            #[cfg(all(test, windows))]
            {
                super::profiling::operation_signal(signal)
            }
            #[cfg(not(all(test, windows)))]
            {
                signal
            }
        };
        let range = dst.cast_const();
        call("zeCommandListAppendMemoryRangesBarrier", || unsafe {
            (api.zeCommandListAppendMemoryRangesBarrier)(
                self.0.0,
                1,
                &size,
                &range,
                signal,
                0,
                null_mut(),
            )
        })
    }

    pub(crate) fn signal(&self, api: &Api, fence: &Fence) -> Result<(), Error> {
        call("zeCommandListAppendSignalEvent", || unsafe {
            (api.zeCommandListAppendSignalEvent)(self.0.0, fence.handle.0)
        })
    }

    pub(crate) fn barrier(
        &self,
        api: &Api,
        signal: &Fence,
        waits: &[Arc<Fence>],
    ) -> Result<(), Error> {
        let mut events = WaitEvents::new(waits)?;
        call("zeCommandListAppendBarrier", || unsafe {
            (api.zeCommandListAppendBarrier)(self.0.0, signal.handle.0, events.count, events.ptr())
        })
    }

    pub(crate) fn queue_barrier(&self, api: &Api) -> Result<(), Error> {
        call("zeCommandListAppendBarrier", || unsafe {
            (api.zeCommandListAppendBarrier)(self.0.0, null_mut(), 0, null_mut())
        })
    }

    pub(crate) fn destroy(&self, api: &Api) -> Result<(), Error> {
        call("zeCommandListDestroy", || unsafe {
            (api.zeCommandListDestroy)(self.0.0)
        })
    }

    pub(crate) fn wait(&self, api: &Api, timeout: u64) -> Result<(), Error> {
        self.wait_with(api.zeCommandListHostSynchronize, timeout)
    }

    pub(crate) fn wait_with(&self, wait: WaitFn, timeout: u64) -> Result<(), Error> {
        call("zeCommandListHostSynchronize", || unsafe {
            wait(self.0.0, timeout)
        })
    }

    pub(crate) fn launch(
        &self,
        api: &Api,
        kernel: Handle,
        geometry: ([u32; 3], [u32; 3]),
        args: &[KernelArg],
        signal: Option<&Fence>,
        waits: &[Arc<Fence>],
    ) -> Result<(), Error> {
        let kernel = kernel.0;
        let ([x, y, z], [bx, by, bz]) = geometry;
        call("zeKernelSetGroupSize", || unsafe {
            (api.zeKernelSetGroupSize)(kernel, bx, by, bz)
        })?;
        for (index, value) in args.iter().enumerate() {
            call("zeKernelSetArgumentValue", || unsafe {
                (api.zeKernelSetArgumentValue)(kernel, index as u32, value.size, value.value)
            })?;
        }
        let mut events = WaitEvents::new(waits)?;
        let groups = GroupCount { x, y, z };
        let signal = signal.map_or(null_mut(), |fence| fence.handle.0);
        #[cfg(all(test, windows))]
        let signal = super::profiling::operation_signal(signal);
        call("zeCommandListAppendLaunchKernel", || unsafe {
            (api.zeCommandListAppendLaunchKernel)(
                self.0.0,
                kernel,
                &groups,
                signal,
                events.count,
                events.ptr(),
            )
        })
    }
}

struct WaitEvents {
    handles: Vec<Raw>,
    count: u32,
}

impl WaitEvents {
    fn new(waits: &[Arc<Fence>]) -> Result<Self, Error> {
        let count = u32::try_from(waits.len())
            .map_err(|_| Error::Driver("wait event count", 0x78000008))?;
        Ok(Self {
            handles: waits.iter().map(|fence| fence.handle.0).collect(),
            count,
        })
    }

    fn ptr(&mut self) -> *mut Raw {
        if self.count == 0 {
            null_mut()
        } else {
            self.handles.as_mut_ptr()
        }
    }
}

pub(crate) struct Fence {
    handle: Handle,
}

// Level Zero allows simultaneous host waits and queries. The frontend
// serializes appends and retains references to prevent reset or destruction.
unsafe impl Sync for Fence {}

impl Fence {
    pub(crate) fn synchronize(&self, wait: WaitFn) -> Result<(), Error> {
        self.wait(wait, u64::MAX)
    }

    pub(crate) fn wait(&self, wait: WaitFn, timeout: u64) -> Result<(), Error> {
        call("zeEventHostSynchronize", || unsafe {
            wait(self.handle.0, timeout)
        })
    }

    pub(crate) fn ready(&self, api: &Api) -> Result<bool, Error> {
        match invoke("zeEventQueryStatus", || unsafe {
            (api.zeEventQueryStatus)(self.handle.0)
        }) {
            1 => Ok(false), // ZE_RESULT_NOT_READY
            status => check("zeEventQueryStatus", status).map(|()| true),
        }
    }

    #[cfg(all(test, windows))]
    pub(crate) fn host_signal(&self, api: &Api) -> Result<(), Error> {
        let call = unsafe {
            api.library()
                .get::<unsafe extern "system" fn(Raw) -> u32>(b"zeEventHostSignal\0")
        }
        .map_err(Error::Loader)?;
        check("zeEventHostSignal", unsafe { call(self.handle.0) })
    }
}

#[cfg(all(test, windows))]
impl CommandList {
    pub(crate) fn gate(&self, api: &Api, gate: &Fence) -> Result<(), Error> {
        let append_wait = unsafe {
            api.library()
                .get::<unsafe extern "system" fn(Raw, u32, *mut Raw) -> u32>(
                    b"zeCommandListAppendWaitOnEvents\0",
                )
        }
        .map_err(Error::Loader)?;
        let mut wait = gate.handle.0;
        call("zeCommandListAppendWaitOnEvents", || unsafe {
            append_wait(self.0.0, 1, &mut wait)
        })
    }
}

const POOL_SIZE: u32 = 64;

#[derive(Default)]
pub(crate) struct Fences {
    pools: Vec<Handle>,
    live: Vec<Arc<Fence>>,
    free: Vec<Arc<Fence>>,
    next_index: u32,
}

impl Fences {
    #[cfg(all(test, windows))]
    pub(crate) fn pool_count(&self) -> usize {
        self.pools.len()
    }

    pub(crate) fn needs_refill(&self) -> bool {
        self.free.is_empty() && (self.pools.is_empty() || self.next_index == POOL_SIZE)
    }

    pub(crate) fn acquire(&mut self, api: &Api, context: &Context) -> Result<Arc<Fence>, Error> {
        if self.needs_refill() {
            let mut index = 0;
            while index < self.live.len() {
                let fence = &self.live[index];
                if Arc::strong_count(fence) == 1 {
                    call("zeEventHostReset", || unsafe {
                        (api.zeEventHostReset)(fence.handle.0)
                    })?;
                    self.free.push(self.live.swap_remove(index));
                } else {
                    index += 1;
                }
            }
            if self.free.is_empty() {
                let desc = EventPoolDesc {
                    header: Header::new(0x10),
                    flags: 1,
                    count: POOL_SIZE,
                };
                let mut pool = null_mut();
                let mut device = context.device.0;
                call("zeEventPoolCreate", || unsafe {
                    (api.zeEventPoolCreate)(context.handle.0, &desc, 1, &mut device, &mut pool)
                })?;
                self.pools.push(Handle(pool));
                self.next_index = 0;
            }
        }
        let fence = if let Some(fence) = self.free.pop() {
            fence
        } else {
            let pool = self.pools.last().ok_or(Error::InvalidProperties)?;
            let desc = EventDesc {
                header: Header::new(0x11),
                index: self.next_index,
                signal: 1 << 2, // ZE_EVENT_SCOPE_FLAG_HOST also covers device visibility.
                wait: 1 << 1,   // ZE_EVENT_SCOPE_FLAG_DEVICE
            };
            let mut event = null_mut();
            call("zeEventCreate", || unsafe {
                (api.zeEventCreate)(pool.0, &desc, &mut event)
            })?;
            self.next_index += 1;
            Arc::new(Fence {
                handle: Handle(event),
            })
        };
        self.live.push(Arc::clone(&fence));
        Ok(fence)
    }

    pub(crate) fn destroy(&mut self, api: &Api) -> Result<(), Error> {
        for events in [&mut self.free, &mut self.live] {
            while let Some(fence) = events.last() {
                call("zeEventDestroy", || unsafe {
                    (api.zeEventDestroy)(fence.handle.0)
                })?;
                events.pop();
            }
        }
        while let Some(pool) = self.pools.last() {
            call("zeEventPoolDestroy", || unsafe {
                (api.zeEventPoolDestroy)(pool.0)
            })?;
            self.pools.pop();
        }
        Ok(())
    }
}
