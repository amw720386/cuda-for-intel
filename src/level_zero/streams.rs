use super::{
    Api, Context, Error, EventDesc, EventPoolDesc, GroupCount, Handle, Header, Module, QueueDesc,
    Raw, check,
};
use std::ptr::null_mut;
use std::sync::Arc;

pub(crate) struct CommandList(Handle);

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
    pub(crate) fn destroy(&self, api: &Api) -> Result<(), Error> {
        check("zeCommandListDestroy", unsafe {
            (api.zeCommandListDestroy)(self.0.0)
        })
    }

    pub(crate) fn wait(&self, api: &Api, timeout: u64) -> Result<(), Error> {
        check("zeCommandListHostSynchronize", unsafe {
            (api.zeCommandListHostSynchronize)(self.0.0, timeout)
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
        let mut events: Vec<Raw> = waits.iter().map(|fence| fence.handle.0).collect();
        let count = u32::try_from(events.len())
            .map_err(|_| Error::Driver("wait event count", 0x78000008))?;
        let wait = if events.is_empty() {
            null_mut()
        } else {
            events.as_mut_ptr()
        };
        let groups = GroupCount { x, y, z };
        check("zeCommandListAppendLaunchKernel", unsafe {
            (api.zeCommandListAppendLaunchKernel)(
                self.0.0,
                kernel,
                &groups,
                signal.map_or(null_mut(), |fence| fence.handle.0),
                count,
                wait,
            )
        })
    }
}

pub(crate) struct Fence {
    handle: Handle,
}

// A fence is an opaque handle. The frontend serializes operations, while
// shared references keep the slot from being recycled during a device wait.
unsafe impl Sync for Fence {}

impl Fence {
    pub(crate) fn ready(&self, api: &Api) -> Result<bool, Error> {
        match unsafe { (api.zeEventQueryStatus)(self.handle.0) } {
            1 => Ok(false), // ZE_RESULT_NOT_READY
            status => check("zeEventQueryStatus", status).map(|()| true),
        }
    }

    #[cfg(test)]
    pub(crate) fn host_signal(&self, api: &Api) -> Result<(), Error> {
        let call = unsafe {
            api._library
                .get::<unsafe extern "system" fn(Raw) -> u32>(b"zeEventHostSignal\0")
        }
        .map_err(Error::Loader)?;
        check("zeEventHostSignal", unsafe { call(self.handle.0) })
    }
}

#[cfg(test)]
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
