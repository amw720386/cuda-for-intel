use super::{Api, CommandList, Context, CuResult, Fence, Submission, Wake, backend_error};
use crate::cuda::RUNTIME;
use crate::level_zero::{DeviceHealth, Queue, QueueFence};
use std::collections::VecDeque;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
pub(super) struct Policy {
    pub cap: usize,
    pub delay: Option<Duration>,
    pub high_resolution: bool,
    #[cfg(test)]
    pub eager_after_idle: bool,
}

#[cfg(test)]
thread_local! {
    pub(super) static TEST_POLICY: std::cell::Cell<Option<Option<Policy>>> = const { std::cell::Cell::new(None) };
}

impl Policy {
    pub fn configured() -> Result<Option<Self>, CuResult> {
        #[cfg(test)]
        if let Some(policy) = TEST_POLICY.get() {
            return Ok(policy);
        }
        if !cfg!(feature = "experimental-batching") {
            return Ok(None);
        }
        let Some(cap) = std::env::var_os("CUDA_FOR_INTEL_BATCH_CAP") else {
            return Ok(None);
        };
        let cap = cap
            .to_str()
            .and_then(|cap| cap.parse::<usize>().ok())
            .filter(|cap| (1..=1024).contains(cap))
            .ok_or(CuResult::InvalidValue)?;
        let micros = match std::env::var("CUDA_FOR_INTEL_BATCH_US") {
            Ok(value) => value
                .parse::<u64>()
                .ok()
                .filter(|value| (1..=1_000_000).contains(value))
                .ok_or(CuResult::InvalidValue)?,
            Err(std::env::VarError::NotPresent) => 250,
            Err(_) => return Err(CuResult::InvalidValue),
        };
        Ok(Some(Self {
            cap,
            #[cfg(test)]
            eager_after_idle: false,
            delay: Some(Duration::from_micros(micros)),
            high_resolution: match std::env::var("CUDA_FOR_INTEL_BATCH_TIMER") {
                Ok(timer) if timer == "high-resolution" => true,
                Ok(timer) if timer == "condvar" => false,
                Err(std::env::VarError::NotPresent) => false,
                _ => return Err(CuResult::InvalidValue),
            },
        }))
    }
}

struct Slot {
    list: Option<CommandList>,
    fence: Option<Arc<QueueFence>>,
    events: Vec<Submission>,
}

impl Slot {
    fn destroy(&mut self, api: &Api) -> Result<(), CuResult> {
        if let Some(list) = &self.list {
            list.destroy(api).map_err(backend_error)?;
            self.list = None;
        }
        if let Some(fence) = &self.fence {
            fence.destroy(api).map_err(backend_error)?;
            self.fence = None;
        }
        Ok(())
    }
}

pub(super) struct Batch {
    queue: Queue,
    policy: Policy,
    wake: Arc<Wake>,
    open: Option<Slot>,
    first: Option<Instant>,
    count: usize,
    #[cfg(test)]
    last_append: Option<Instant>,
    pending: VecDeque<Slot>,
    free: Vec<Slot>,
    error: Option<CuResult>,
    uncertain_submission: bool,
    health: Arc<DeviceHealth>,
    #[cfg(test)]
    pub stats: Stats,
}

#[cfg(test)]
#[derive(Default)]
pub(super) struct Stats {
    pub background: usize,
    pub last_delay: Duration,
    pub wake_late: Duration,
    pub lock_time: Duration,
    pub scan_late: Duration,
    pub close_time: Duration,
    pub execute_time: Duration,
    pub recycled: usize,
    pub recycle_time: Duration,
    pub submitted: usize,
}

impl Batch {
    pub fn new(
        api: &Api,
        context: &Context,
        policy: Policy,
        wake: Arc<Wake>,
    ) -> Result<Self, CuResult> {
        Ok(Self {
            queue: context
                .create_queue(api)
                .inspect_err(|error| context.health.observe(error))
                .map_err(backend_error)?,
            policy,
            wake,
            open: None,
            first: None,
            count: 0,
            #[cfg(test)]
            last_append: None,
            pending: VecDeque::new(),
            free: Vec::new(),
            error: None,
            uncertain_submission: false,
            health: Arc::clone(&context.health),
            #[cfg(test)]
            stats: Stats::default(),
        })
    }

    pub fn check(&self) -> Result<(), CuResult> {
        if self.health.lost() {
            return Err(CuResult::LaunchFailed);
        }
        self.error.map_or(Ok(()), Err)
    }

    pub fn buffered(&self) -> bool {
        self.first.is_some()
    }

    pub fn deadline(&self) -> Option<Instant> {
        self.first
            .zip(self.policy.delay)
            .map(|(first, delay)| first + delay)
    }

    pub fn contains_recording(&self, fence: &Arc<Fence>) -> bool {
        self.open.as_ref().is_some_and(|slot| {
            slot.events
                .iter()
                .any(|event| Arc::ptr_eq(&event.completion, fence))
        })
    }

    pub fn recording(&mut self, api: &Api, context: &Context) -> Result<&CommandList, CuResult> {
        self.check()?;
        if self.open.is_none() {
            self.reap(api)?;
            let slot = if let Some(slot) = self.free.pop() {
                slot
            } else {
                let list = context
                    .create_list(api)
                    .inspect_err(|error| self.health.observe(error))
                    .map_err(backend_error)?;
                let fence = match self.queue.create_fence(api) {
                    Ok(fence) => fence,
                    Err(error) => {
                        self.health.observe(&error);
                        let mut slot = Slot {
                            list: Some(list),
                            fence: None,
                            events: Vec::new(),
                        };
                        if let Err(cleanup) = slot.destroy(api) {
                            self.open = Some(slot);
                            self.error = Some(cleanup);
                        }
                        return Err(backend_error(error));
                    }
                };
                Slot {
                    list: Some(list),
                    fence: Some(Arc::new(fence)),
                    events: Vec::new(),
                }
            };
            // The regular-list IN_ORDER flag applies within a list. A
            // queue-scoped barrier orders this list after earlier submissions.
            let ordering = if self.pending.is_empty() {
                Ok(())
            } else {
                slot.list.as_ref().unwrap().queue_barrier(api)
            };
            self.open = Some(slot);
            if let Err(error) = ordering {
                self.health.observe(&error);
                let error = backend_error(error);
                self.record_failed(error, None);
                return Err(error);
            }
        }
        Ok(self.open.as_ref().unwrap().list.as_ref().unwrap())
    }

    pub fn record_failed(&mut self, error: CuResult, events: Option<Submission>) {
        // A failed append may have mutated the open list. Keep its events alive
        // and do not submit it or recycle the slot.
        self.open.as_mut().unwrap().events.extend(events);
        self.error = Some(error);
    }

    pub fn appended(&mut self, api: &Api, events: Option<&Submission>) -> Result<(), CuResult> {
        if let Some(events) = events {
            self.open.as_mut().unwrap().events.push(events.clone());
        }
        self.count += 1;
        if self.count == 1 {
            self.first = Some(Instant::now());
        }
        #[cfg(test)]
        let eager = self.policy.eager_after_idle && {
            let now = Instant::now();
            let last = self.last_append.replace(now);
            self.count == 1
                && last.is_none_or(|last| now.duration_since(last) >= Duration::from_millis(1))
        };
        #[cfg(not(test))]
        let eager = false;
        if eager
            || self.count == self.policy.cap
            || self.count > 1
                && self
                    .deadline()
                    .is_some_and(|deadline| Instant::now() >= deadline)
        {
            self.flush(api)?;
        } else if self.count == 1 && self.policy.delay.is_some() {
            self.wake.notify();
        }
        Ok(())
    }

    pub fn flush(&mut self, api: &Api) -> Result<(), CuResult> {
        self.check()?;
        if self.buffered() {
            let slot = self.open.as_ref().unwrap();
            #[cfg(test)]
            let close_start = Instant::now();
            let close = slot.list.as_ref().unwrap().close(api);
            #[cfg(test)]
            {
                self.stats.close_time = close_start.elapsed();
            }
            if let Err(error) = close {
                self.health.observe(&error);
                let error = backend_error(error);
                self.error = Some(error);
                return Err(error);
            }
            #[cfg(test)]
            let execute_start = Instant::now();
            let execute = self.queue.execute(
                api,
                slot.list.as_ref().unwrap(),
                slot.fence.as_ref().unwrap(),
            );
            #[cfg(test)]
            {
                self.stats.execute_time = execute_start.elapsed();
            }
            if let Err(error) = execute {
                // The driver may have accepted work before reporting failure.
                self.uncertain_submission = true;
                self.health.observe(&error);
                let error = backend_error(error);
                self.error = Some(error);
                return Err(error);
            }
            #[cfg(test)]
            {
                self.stats.submitted += 1;
            }
            self.pending.push_back(self.open.take().unwrap());
            self.first = None;
            self.count = 0;
        }
        Ok(())
    }

    fn background_flush(&mut self, api: &Api) {
        #[cfg(test)]
        let first = self.first.unwrap();
        if self.flush(api).is_ok() {
            #[cfg(test)]
            {
                self.stats.background += 1;
                self.stats.last_delay = first.elapsed();
            }
        }
    }

    pub fn wait_tail(&mut self, api: &Api, timeout: u64) -> Result<(), CuResult> {
        if let Some(fence) = self.tail() {
            fence
                .wait(api.fence_wait_fn(), timeout)
                .inspect_err(|error| self.health.observe(error))
                .map_err(backend_error)?;
        }
        Ok(())
    }

    pub fn tail(&self) -> Option<&Arc<QueueFence>> {
        self.pending.back().and_then(|slot| slot.fence.as_ref())
    }

    pub fn has_waiters(&self) -> bool {
        self.pending
            .iter()
            .any(|slot| Arc::strong_count(slot.fence.as_ref().unwrap()) > 1)
    }

    pub fn reap(&mut self, api: &Api) -> Result<(), CuResult> {
        self.check()?;
        while let Some(slot) = self.pending.front_mut() {
            let fence = slot.fence.as_ref().unwrap();
            if Arc::strong_count(fence) > 1
                || !fence
                    .ready(api)
                    .inspect_err(|error| self.health.observe(error))
                    .map_err(backend_error)?
            {
                break;
            }
            // Keep memory bounded after bursts. Outstanding lists are limited
            // by driver allocations, without blocking the appending thread.
            let reset = if self.free.len() < 32 {
                #[cfg(test)]
                let start = Instant::now();
                let result = slot
                    .list
                    .as_ref()
                    .unwrap()
                    .reset(api)
                    .and_then(|()| fence.reset(api))
                    .inspect_err(|error| self.health.observe(error))
                    .map_err(backend_error);
                #[cfg(test)]
                {
                    self.stats.recycled += 1;
                    self.stats.recycle_time += start.elapsed();
                }
                result
            } else {
                slot.destroy(api)
            };
            if let Err(error) = reset {
                self.error = Some(error);
                return Err(error);
            }
            let mut slot = self.pending.pop_front().unwrap();
            slot.events.clear();
            if slot.list.is_some() {
                self.free.push(slot);
            }
        }
        Ok(())
    }

    pub fn drain(&mut self, api: &Api, timeout: u64) -> Result<(), CuResult> {
        if self.uncertain_submission || !self.pending.is_empty() {
            self.queue
                .wait(api, timeout)
                .inspect_err(|error| self.health.observe(error))
                .map_err(backend_error)?;
            self.uncertain_submission = false;
        }
        Ok(())
    }

    // Only called after the queue has drained. Keep each still-owned handle
    // in its slot until its native destroy succeeds, including on retries.
    pub fn destroy(&mut self, api: &Api) -> Result<(), CuResult> {
        while let Some(slot) = self.pending.front_mut() {
            slot.destroy(api)?;
            self.pending.pop_front();
        }
        if let Some(slot) = &mut self.open {
            slot.destroy(api)?;
            self.open = None;
            self.first = None;
        }
        while let Some(slot) = self.free.last_mut() {
            slot.destroy(api)?;
            self.free.pop();
        }
        self.queue.destroy(api).map_err(backend_error)
    }
}

pub(super) fn spawn(
    owner: usize,
    wake: Arc<Wake>,
) -> Result<JoinHandle<Result<(), CuResult>>, CuResult> {
    std::thread::Builder::new()
        .name("cuda-batch".into())
        .spawn(move || {
            let mut next = Some(Instant::now());
            loop {
                let result = wake.wait(next);
                #[cfg(test)]
                let woke = Instant::now();
                let Ok(mut guard) = RUNTIME.lock() else {
                    return Err(CuResult::Unknown);
                };
                #[cfg(test)]
                let lock_time = woke.elapsed();
                let Some(state) = guard.as_mut() else {
                    return Ok(());
                };
                let Some(context) = state.contexts.get_mut(&owner) else {
                    return shutdown();
                };
                if context.streams.worker.is_none() {
                    return shutdown();
                }
                next = None;
                for batch in context
                    .streams
                    .all_mut()
                    .filter_map(|stream| stream.batch.as_mut())
                {
                    if let Err(error) = result {
                        batch.error.get_or_insert(error);
                    }
                    if batch.check().is_err() {
                        continue;
                    }
                    if let Some(deadline) = batch.deadline() {
                        if deadline <= Instant::now() {
                            #[cfg(test)]
                            {
                                // A deadline can also pass during a work notification
                                // or while acquiring the runtime lock.
                                batch.stats.wake_late = woke.saturating_duration_since(deadline);
                                batch.stats.lock_time = lock_time;
                                batch.stats.scan_late = deadline.elapsed();
                            }
                            batch.background_flush(&state.runtime.api);
                        } else {
                            next = Some(
                                next.map_or(deadline, |earliest: Instant| earliest.min(deadline)),
                            );
                        }
                    }
                }
                if let Err(error) = result {
                    context.teardown_error = Some(error);
                    return Err(error);
                }
            }
        })
        .map_err(|_| CuResult::OperatingSystem)
}

fn shutdown() -> Result<(), CuResult> {
    #[cfg(test)]
    if crate::level_zero::fault::status("batch worker shutdown").is_some() {
        return Err(CuResult::OperatingSystem);
    }
    Ok(())
}
