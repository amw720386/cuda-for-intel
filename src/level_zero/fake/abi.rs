use super::*;

// Raw outputs are written only by the fake functions with the same ABI as the
// driver. The real backend supplies valid output storage to each function.
fn create(parent: Raw, parent_kind: Kind, kind: Kind, out: *mut Raw) -> Status {
    call_driver(|driver| {
        driver.get(parent.addr(), parent_kind)?;
        unsafe { out.write(without_provenance_mut(driver.create(parent.addr(), kind))) };
        Ok(0)
    })
}

unsafe extern "system" fn context_create(
    parent: Raw,
    _: *const ContextDesc,
    out: *mut Raw,
) -> Status {
    call_driver(|driver| {
        let parent = parent.addr();
        if !DEVICES.iter().any(|device| device.0 == parent) {
            return Err("invalid native driver");
        }
        unsafe { out.write(without_provenance_mut(driver.create(parent, Kind::Context))) };
        Ok(0)
    })
}

unsafe extern "system" fn queue_create(
    ctx: Raw,
    device: Raw,
    desc: *const QueueDesc,
    out: *mut Raw,
) -> Status {
    call_driver(|driver| {
        if unsafe { (*desc).ordinal } != driver.device_ordinal(ctx, device)? {
            return Err("selected wrong compute queue ordinal");
        }
        unsafe {
            out.write(without_provenance_mut(
                driver.create(ctx.addr(), Kind::Queue),
            ))
        };
        Ok(0)
    })
}

fn list_create(ctx: Raw, device: Raw, ordinal: u32, mode: u32, out: *mut Raw) -> Status {
    call_driver(|driver| {
        if ordinal != driver.device_ordinal(ctx, device)? {
            return Err("selected wrong list queue ordinal");
        }
        let id = driver.create(ctx.addr(), Kind::List);
        driver.lists.insert(
            id,
            List {
                mode,
                closed: false,
                commands: VecDeque::new(),
            },
        );
        unsafe { out.write(without_provenance_mut(id)) };
        Ok(0)
    })
}

unsafe extern "system" fn list_immediate(
    ctx: Raw,
    device: Raw,
    desc: *const QueueDesc,
    out: *mut Raw,
) -> Status {
    let desc = unsafe { &*desc };
    list_create(ctx, device, desc.ordinal, desc.mode, out)
}

unsafe extern "system" fn list_regular(
    ctx: Raw,
    device: Raw,
    desc: *const regular::ListDesc,
    out: *mut Raw,
) -> Status {
    list_create(ctx, device, unsafe { (*desc).ordinal }, 0, out)
}

unsafe extern "system" fn list_close(list: Raw) -> Status {
    call_driver(|driver| {
        let list = driver
            .lists
            .get_mut(&list.addr())
            .ok_or("closed missing list")?;
        if list.closed || list.mode != 0 {
            return Err("closed an already closed or immediate list");
        }
        list.closed = true;
        Ok(0)
    })
}

unsafe extern "system" fn list_reset(list: Raw) -> Status {
    call_driver(|driver| {
        if driver.pending_on(list.addr()) {
            return Err("recycled an in-flight list");
        }
        let list = driver
            .lists
            .get_mut(&list.addr())
            .ok_or("reset missing list")?;
        if !list.closed || !list.commands.is_empty() {
            return Err("reset a list without executing it");
        }
        list.closed = false;
        Ok(0)
    })
}

unsafe extern "system" fn execute(queue: Raw, count: u32, lists: *mut Raw, fence: Raw) -> Status {
    call_driver(|driver| {
        let context = driver.get(queue.addr(), Kind::Queue)?.parent;
        let completion = driver.get(fence.addr(), Kind::Fence)?;
        if count != 1
            || completion.parent != queue.addr()
            || completion.signaled
            || driver.pending_on(fence.addr())
        {
            return Err("invalid execute count or fence reuse");
        }
        let id = unsafe { lists.read() }.addr();
        if driver.get(id, Kind::List)?.parent != context || driver.pending_on(id) {
            return Err("executed a foreign or in-flight list");
        }
        let list = driver.lists.get_mut(&id).ok_or("missing execute list")?;
        if !list.closed || list.commands.is_empty() {
            return Err("executed an open or empty list");
        }
        let commands = std::mem::take(&mut list.commands);
        driver.submit(Submission {
            list: id,
            queue: queue.addr(),
            fence: fence.addr(),
            commands,
        });
        Ok(driver.execute_result.take().unwrap_or(0))
    })
}

unsafe extern "system" fn module_create(
    ctx: Raw,
    device: Raw,
    _: *const ModuleDesc,
    out: *mut Raw,
    log: *mut Raw,
) -> Status {
    call_driver(|driver| {
        driver.device_ordinal(ctx, device)?;
        unsafe {
            out.write(without_provenance_mut(
                driver.create(ctx.addr(), Kind::Module),
            ));
            log.write(without_provenance_mut(
                driver.create(ctx.addr(), Kind::BuildLog),
            ));
        }
        Ok(0)
    })
}

fn allocate(ctx: Raw, device: Option<Raw>, bytes: usize, kind: Kind, out: *mut Raw) -> Status {
    call_driver(|driver| {
        if let Some(device) = device {
            driver.device_ordinal(ctx, device)?;
        } else {
            driver.get(ctx.addr(), Kind::Context)?;
        }
        let id = driver.create(ctx.addr(), kind);
        driver
            .objects
            .get_mut(&id)
            .ok_or("missing allocation")?
            .size = bytes;
        unsafe { out.write(without_provenance_mut(id)) };
        Ok(0)
    })
}

unsafe extern "system" fn alloc_device(
    ctx: Raw,
    _: *const AllocDesc,
    bytes: usize,
    _: usize,
    device: Raw,
    out: *mut Raw,
) -> Status {
    allocate(ctx, Some(device), bytes, Kind::DeviceAllocation, out)
}

unsafe extern "system" fn alloc_host(
    ctx: Raw,
    _: *const HostAllocDesc,
    bytes: usize,
    _: usize,
    out: *mut Raw,
) -> Status {
    allocate(ctx, None, bytes, Kind::HostAllocation, out)
}

unsafe extern "system" fn free(ctx: Raw, ptr: Raw) -> Status {
    call_driver(|driver| {
        let object = driver.objects.get(&ptr.addr()).ok_or("double free")?;
        if object.parent != ctx.addr()
            || !matches!(object.kind, Kind::HostAllocation | Kind::DeviceAllocation)
        {
            return Err("free of foreign or non-allocation handle");
        }
        driver.release(ptr.addr(), object.kind)?;
        Ok(0)
    })
}

fn wait_condition(driver: &Driver, id: Id, kind: Kind) -> DriverResult<bool> {
    let object = driver.get(id, kind)?;
    Ok(match kind {
        Kind::Fence | Kind::Event => object.signaled,
        _ => !driver.pending_on(id),
    })
}

fn wait(id: Raw, kind: Kind, timeout: u64) -> Status {
    let id = id.addr();
    let mut guard = lock();
    let Some(driver) = guard.as_mut() else {
        return UNKNOWN;
    };
    match wait_condition(driver, id, kind) {
        Ok(true) => return 0,
        Ok(false) if timeout == 0 => return NOT_READY,
        Ok(false) => (),
        Err(issue) => {
            driver.ledger.violations.push(issue);
            return UNKNOWN;
        }
    }
    driver.waiters += 1;
    CHANGED.notify_all();
    let limit = Duration::from_nanos(timeout).min(WAIT_LIMIT);
    let deadline = Instant::now() + limit;
    loop {
        let Some(driver) = guard.as_mut() else {
            return UNKNOWN;
        };
        let left = deadline.saturating_duration_since(Instant::now());
        match wait_condition(driver, id, kind) {
            Ok(false) if !left.is_zero() => {
                guard = CHANGED
                    .wait_timeout(guard, left)
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .0;
            }
            result => {
                driver.waiters -= 1;
                return match result {
                    Ok(true) => 0,
                    Ok(false) => NOT_READY,
                    Err(issue) => {
                        driver.ledger.violations.push(issue);
                        UNKNOWN
                    }
                };
            }
        }
    }
}

fn marker(signal: Raw, count: u32, waits: *mut Raw) -> Command {
    let waits = if count == 0 {
        vec![]
    } else {
        unsafe { std::slice::from_raw_parts(waits, count as usize) }
            .iter()
            .map(|ptr| ptr.addr())
            .collect()
    };
    Command {
        waits,
        signal: signal.addr(),
        ..Command::default()
    }
}

unsafe extern "system" fn append_barrier(
    list: Raw,
    signal: Raw,
    count: u32,
    waits: *mut Raw,
) -> Status {
    call_driver(|driver| driver.append(list.addr(), marker(signal, count, waits)))
}

unsafe extern "system" fn append_signal(list: Raw, signal: Raw) -> Status {
    unsafe { append_barrier(list, signal, 0, null_mut()) }
}

unsafe extern "system" fn append_ranges_barrier(
    list: Raw,
    count: u32,
    sizes: *const usize,
    ranges: *const *const c_void,
    signal: Raw,
    waits: u32,
    wait_events: *mut Raw,
) -> Status {
    if count != 1 || sizes.is_null() || ranges.is_null() {
        return fake::unexpected("zeCommandListAppendMemoryRangesBarrier");
    }
    let _ = (unsafe { sizes.read() }, unsafe { ranges.read() });
    unsafe { append_barrier(list, signal, waits, wait_events) }
}

unsafe extern "system" fn append_copy(
    list: Raw,
    dst: Raw,
    src: *const c_void,
    _: usize,
    signal: Raw,
    count: u32,
    waits: *mut Raw,
) -> Status {
    call_driver(|driver| {
        let mut command = marker(signal, count, waits);
        command.resources.extend(
            [dst.addr(), src.addr()]
                .into_iter()
                .filter_map(|ptr| driver.allocation(ptr)),
        );
        driver.append(list.addr(), command)
    })
}

unsafe extern "system" fn group_size(kernel: Raw, _: u32, _: u32, _: u32) -> Status {
    call_driver(|driver| {
        driver.get(kernel.addr(), Kind::Kernel)?;
        Ok(0)
    })
}

unsafe extern "system" fn kernel_arg(
    kernel: Raw,
    index: u32,
    size: usize,
    data: *const c_void,
) -> Status {
    call_driver(|driver| {
        driver.get(kernel.addr(), Kind::Kernel)?;
        let address = (size == size_of::<usize>())
            .then(|| unsafe { data.cast::<usize>().read_unaligned() })
            .and_then(|address| driver.allocation(address));
        let args = driver.args.entry(kernel.addr()).or_default();
        if let Some(address) = address {
            args.insert(index, address);
        } else {
            args.remove(&index);
        }
        Ok(0)
    })
}

unsafe extern "system" fn append_kernel(
    list: Raw,
    kernel: Raw,
    _: *const GroupCount,
    signal: Raw,
    count: u32,
    waits: *mut Raw,
) -> Status {
    call_driver(|driver| {
        let parent = driver.get(kernel.addr(), Kind::Kernel)?.parent;
        let mut command = marker(signal, count, waits);
        command.resources.extend([kernel.addr(), parent]);
        command.resources.extend(
            driver
                .args
                .get(&kernel.addr())
                .into_iter()
                .flat_map(|args| args.values())
                .copied(),
        );
        driver.append(list.addr(), command)
    })
}

unsafe extern "system" fn pool_create(
    ctx: Raw,
    _: *const EventPoolDesc,
    _: u32,
    _: *mut Raw,
    out: *mut Raw,
) -> Status {
    create(ctx, Kind::Context, Kind::EventPool, out)
}

macro_rules! make_creators {
    ($($name:ident($desc:ty, $parent:ident, $kind:ident)),+ $(,)?) => {$(
        unsafe extern "system" fn $name(parent: Raw, _: *const $desc, out: *mut Raw) -> Status {
            create(parent, Kind::$parent, Kind::$kind, out)
        }
    )+};
}

make_creators! {
    kernel_create(KernelDesc, Module, Kernel),
    event_create(EventDesc, EventPool, Event),
    fence_create(regular::FenceDesc, Queue, Fence),
}

macro_rules! lifecycle {
    (destroy $($name:ident($kind:ident)),+ $(,)?) => {$(
        unsafe extern "system" fn $name(id: Raw) -> Status {
            call_driver(|driver| { driver.release(id.addr(), Kind::$kind)?; Ok(0) })
        }
    )+};
    (reset $($name:ident($kind:ident)),+ $(,)?) => {$(
        unsafe extern "system" fn $name(id: Raw) -> Status {
            call_driver(|driver| driver.reset_signal(id.addr(), Kind::$kind))
        }
    )+};
    (wait $($name:ident($kind:ident)),+ $(,)?) => {$(
        unsafe extern "system" fn $name(id: Raw, timeout: u64) -> Status {
            wait(id, Kind::$kind, timeout)
        }
    )+};
}

lifecycle!(destroy context_destroy(Context), queue_destroy(Queue), list_destroy(List),
    module_destroy(Module), log_destroy(BuildLog), kernel_destroy(Kernel),
    pool_destroy(EventPool), event_destroy(Event), fence_destroy(Fence));
lifecycle!(reset event_reset(Event), fence_reset(Fence));
lifecycle!(wait queue_wait(Queue), list_wait(List), event_wait(Event), fence_wait(Fence));

unsafe extern "system" fn event_query(id: Raw) -> Status {
    wait(id, Kind::Event, 0)
}
unsafe extern "system" fn fence_query(id: Raw) -> Status {
    wait(id, Kind::Fence, 0)
}

pub(super) fn api() -> Api {
    Api {
        zeContextCreate: context_create,
        zeContextDestroy: context_destroy,
        zeCommandListCreateImmediate: list_immediate,
        zeCommandListCreate: list_regular,
        zeCommandListClose: list_close,
        zeCommandListReset: list_reset,
        zeCommandQueueCreate: queue_create,
        zeCommandQueueDestroy: queue_destroy,
        zeCommandQueueSynchronize: queue_wait,
        zeCommandQueueExecuteCommandLists: execute,
        zeFenceCreate: fence_create,
        zeFenceDestroy: fence_destroy,
        zeFenceReset: fence_reset,
        zeFenceQueryStatus: fence_query,
        zeFenceHostSynchronize: fence_wait,
        zeCommandListDestroy: list_destroy,
        zeCommandListHostSynchronize: list_wait,
        zeCommandListAppendSignalEvent: append_signal,
        zeCommandListAppendBarrier: append_barrier,
        zeCommandListAppendMemoryRangesBarrier: append_ranges_barrier,
        zeEventPoolCreate: pool_create,
        zeEventPoolDestroy: pool_destroy,
        zeEventCreate: event_create,
        zeEventDestroy: event_destroy,
        zeEventHostReset: event_reset,
        zeEventQueryStatus: event_query,
        zeEventHostSynchronize: event_wait,
        zeMemAllocDevice: alloc_device,
        zeMemAllocHost: alloc_host,
        zeMemFree: free,
        zeCommandListAppendMemoryCopy: append_copy,
        zeModuleCreate: module_create,
        zeModuleDestroy: module_destroy,
        zeModuleBuildLogDestroy: log_destroy,
        zeKernelCreate: kernel_create,
        zeKernelDestroy: kernel_destroy,
        zeKernelSetGroupSize: group_size,
        zeKernelSetArgumentValue: kernel_arg,
        zeCommandListAppendLaunchKernel: append_kernel,
        ..Api::stub()
    }
}
