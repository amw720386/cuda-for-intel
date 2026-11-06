use super::{
    Api, Context, Error, EventDesc, EventPoolDesc, GroupCount, Handle, Header, Module, QueueDesc,
    Raw, check,
};
use std::ptr::null_mut;
use std::sync::Arc;

pub(crate) struct CommandList(Handle);

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
        let desc = QueueDesc {
            header: Header::new(0xe),
            ordinal: self.ordinal,
            index: 0,
            flags: 1 << 1, // ZE_COMMAND_QUEUE_FLAG_IN_ORDER
            mode: 2,       // ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS
            priority: 0,
        };
        let mut list = null_mut();
        check("zeCommandListCreateImmediate (asynchronous)", unsafe {
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
        let signal = signal.map_or(null_mut(), |fence| fence.handle.0);
        #[cfg(all(test, windows))]
        let signal = super::profiling::operation_signal(signal);
        check("zeCommandListAppendMemoryCopy", unsafe {
            (api.zeCommandListAppendMemoryCopy)(
                self.0.0,
                dst,
                src,
                size,
                signal,
                events.count,
                events.ptr(),
            )
        })
    }

    pub(crate) fn signal(&self, api: &Api, fence: &Fence) -> Result<(), Error> {
        check("zeCommandListAppendSignalEvent", unsafe {
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
        check("zeCommandListAppendBarrier", unsafe {
            (api.zeCommandListAppendBarrier)(self.0.0, signal.handle.0, events.count, events.ptr())
        })
    }

    pub(crate) fn destroy(&self, api: &Api) -> Result<(), Error> {
        check("zeCommandListDestroy", unsafe {
            (api.zeCommandListDestroy)(self.0.0)
        })
    }

    pub(crate) fn wait(&self, api: &Api, timeout: u64) -> Result<(), Error> {
        self.wait_with(api.zeCommandListHostSynchronize, timeout)
    }

    pub(crate) fn wait_with(&self, wait: WaitFn, timeout: u64) -> Result<(), Error> {
        check("zeCommandListHostSynchronize", unsafe {
            wait(self.0.0, timeout)
        })
    }

    pub(crate) fn launch(
        &self,
        api: &Api,
        module: &Module,
        geometry: ([u32; 3], [u32; 3]),
        args: &[&[u8]],
        signal: Option<&Fence>,
        waits: &[Arc<Fence>],
    ) -> Result<(), Error> {
        let kernel = module
            .kernel
            .ok_or(Error::Driver("launch without a kernel", 0x7800001c))?
            .0;
        let ([x, y, z], [bx, by, bz]) = geometry;
        check("zeKernelSetGroupSize", unsafe {
            (api.zeKernelSetGroupSize)(kernel, bx, by, bz)
        })?;
        for (index, value) in args.iter().enumerate() {
            check("zeKernelSetArgumentValue", unsafe {
                (api.zeKernelSetArgumentValue)(
                    kernel,
                    index as u32,
                    value.len(),
                    value.as_ptr().cast(),
                )
            })?;
        }
        let mut events = WaitEvents::new(waits)?;
        let groups = GroupCount { x, y, z };
        let signal = signal.map_or(null_mut(), |fence| fence.handle.0);
        #[cfg(all(test, windows))]
        let signal = super::profiling::operation_signal(signal);
        check("zeCommandListAppendLaunchKernel", unsafe {
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
        check("zeEventHostSynchronize", unsafe {
            wait(self.handle.0, u64::MAX)
        })
    }

    pub(crate) fn ready(&self, api: &Api) -> Result<bool, Error> {
        match unsafe { (api.zeEventQueryStatus)(self.handle.0) } {
            1 => Ok(false), // ZE_RESULT_NOT_READY
            status => check("zeEventQueryStatus", status).map(|()| true),
        }
    }

    #[cfg(all(test, windows))]
    pub(crate) fn host_signal(&self, api: &Api) -> Result<(), Error> {
        let call = unsafe {
            api._library
                .get::<unsafe extern "system" fn(Raw) -> u32>(b"zeEventHostSignal\0")
        }
        .map_err(Error::Loader)?;
        check("zeEventHostSignal", unsafe { call(self.handle.0) })
    }
}

#[cfg(all(test, windows))]
impl CommandList {
    pub(crate) fn gate(&self, api: &Api, gate: &Fence) -> Result<(), Error> {
        let call = unsafe {
            api._library
                .get::<unsafe extern "system" fn(Raw, u32, *mut Raw) -> u32>(
                    b"zeCommandListAppendWaitOnEvents\0",
                )
        }
        .map_err(Error::Loader)?;
        let mut wait = gate.handle.0;
        check("zeCommandListAppendWaitOnEvents", unsafe {
            call(self.0.0, 1, &mut wait)
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
                    check("zeEventHostReset", unsafe {
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
                check("zeEventPoolCreate", unsafe {
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
            check("zeEventCreate", unsafe {
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
                check("zeEventDestroy", unsafe {
                    (api.zeEventDestroy)(fence.handle.0)
                })?;
                events.pop();
            }
        }
        while let Some(pool) = self.pools.last() {
            check("zeEventPoolDestroy", unsafe {
                (api.zeEventPoolDestroy)(pool.0)
            })?;
            self.pools.pop();
        }
        Ok(())
    }
}
