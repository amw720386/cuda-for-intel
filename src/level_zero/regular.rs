use super::streams::WaitFn;
use super::{Api, CommandList, Context, Error, Handle, Header, QueueDesc, check};
use std::ptr::null_mut;

#[repr(C)]
pub(super) struct ListDesc {
    header: Header,
    ordinal: u32,
    flags: u32,
}

#[repr(C)]
pub(super) struct FenceDesc {
    header: Header,
    flags: u32,
}

pub(crate) struct Queue(Handle);

pub(crate) struct QueueFence(Handle);

impl Api {
    pub(crate) fn fence_wait_fn(&self) -> WaitFn {
        self.zeFenceHostSynchronize
    }
}

// Level Zero permits simultaneous fence queries and waits. The frontend
// ensures that reset and destruction cannot overlap with either.
unsafe impl Sync for QueueFence {}

impl Context {
    pub(crate) fn create_queue(&self, api: &Api) -> Result<Queue, Error> {
        let desc = QueueDesc {
            header: Header::new(0xe),
            ordinal: self.ordinal,
            index: 0, // Required when physical queue selection is implicit.
            flags: 0,
            mode: 2, // ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS
            priority: 0,
        };
        let mut queue = null_mut();
        check("zeCommandQueueCreate", unsafe {
            (api.zeCommandQueueCreate)(self.handle.0, self.device.0, &desc, &mut queue)
        })?;
        Ok(Queue(Handle(queue)))
    }

    pub(crate) fn create_list(&self, api: &Api) -> Result<CommandList, Error> {
        let desc = ListDesc {
            header: Header::new(0xf),
            ordinal: self.ordinal,
            flags: 1 << 3, // ZE_COMMAND_LIST_FLAG_IN_ORDER
        };
        let mut list = null_mut();
        check("zeCommandListCreate", unsafe {
            (api.zeCommandListCreate)(self.handle.0, self.device.0, &desc, &mut list)
        })?;
        Ok(CommandList(Handle(list)))
    }
}

impl Queue {
    pub(crate) fn create_fence(&self, api: &Api) -> Result<QueueFence, Error> {
        let desc = FenceDesc {
            header: Header::new(0x12),
            flags: 0,
        };
        let mut fence = null_mut();
        check("zeFenceCreate", unsafe {
            (api.zeFenceCreate)(self.0.0, &desc, &mut fence)
        })?;
        Ok(QueueFence(Handle(fence)))
    }

    pub(crate) fn execute(
        &self,
        api: &Api,
        list: &CommandList,
        fence: &QueueFence,
    ) -> Result<(), Error> {
        let mut handle = list.0.0;
        check("zeCommandListClose", unsafe {
            (api.zeCommandListClose)(handle)
        })?;
        check("zeCommandQueueExecuteCommandLists", unsafe {
            (api.zeCommandQueueExecuteCommandLists)(self.0.0, 1, &mut handle, fence.0.0)
        })
    }

    pub(crate) fn destroy(&self, api: &Api) -> Result<(), Error> {
        check("zeCommandQueueDestroy", unsafe {
            (api.zeCommandQueueDestroy)(self.0.0)
        })
    }
}

impl CommandList {
    pub(crate) fn reset(&self, api: &Api) -> Result<(), Error> {
        check("zeCommandListReset", unsafe {
            (api.zeCommandListReset)(self.0.0)
        })
    }
}

impl QueueFence {
    pub(crate) fn ready(&self, api: &Api) -> Result<bool, Error> {
        match unsafe { (api.zeFenceQueryStatus)(self.0.0) } {
            1 => Ok(false),
            status => check("zeFenceQueryStatus", status).map(|()| true),
        }
    }

    pub(crate) fn wait(&self, wait: WaitFn, timeout: u64) -> Result<(), Error> {
        check("zeFenceHostSynchronize", unsafe { wait(self.0.0, timeout) })
    }

    pub(crate) fn reset(&self, api: &Api) -> Result<(), Error> {
        check("zeFenceReset", unsafe { (api.zeFenceReset)(self.0.0) })
    }

    pub(crate) fn destroy(&self, api: &Api) -> Result<(), Error> {
        check("zeFenceDestroy", unsafe { (api.zeFenceDestroy)(self.0.0) })
    }
}
