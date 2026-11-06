use super::{Api, CommandList, Context, CuResult, Fence, Submission, backend_error};
use crate::cuda::RUNTIME;
use crate::level_zero::{Queue, QueueFence};
use std::collections::VecDeque;
use std::sync::{Arc, Condvar};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
pub(super) struct Policy {
    pub cap: usize,
    pub delay: Option<Duration>,
}

impl Policy {
    pub fn configured() -> Result<Option<Self>, CuResult> {
        #[cfg(all(test, windows))]
        if let Some(policy) = super::batch_tests::POLICY.get() {
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
            delay: Some(Duration::from_micros(micros)),
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
    wake: Arc<Condvar>,
    open: Option<Slot>,
    first: Option<Instant>,
    count: usize,
    pending: VecDeque<Slot>,
    free: Vec<Slot>,
    error: Option<CuResult>,
    #[cfg(test)]
    pub stats: Stats,
}

#[cfg(test)]
#[derive(Default)]
pub(super) struct Stats {
    pub background: usize,
    pub last_delay: Duration,
    pub recycled: usize,
    pub recycle_time: Duration,
    pub submitted: usize,
}

impl Batch {
    pub fn new(
        api: &Api,
        context: &Context,
        policy: Policy,
        wake: Arc<Condvar>,
    ) -> Result<Self, CuResult> {
        Ok(Self {
            queue: context.create_queue(api).map_err(backend_error)?,
            policy,
            wake,
            open: None,
            first: None,
            count: 0,
            pending: VecDeque::new(),
            free: Vec::new(),
            error: None,
            #[cfg(test)]
            stats: Stats::default(),
        })
    }

    pub fn check(&self) -> Result<(), CuResult> {
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
                let list = context.create_list(api).map_err(backend_error)?;
                let fence = match self.queue.create_fence(api) {
                    Ok(fence) => fence,
                    Err(error) => {
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
        if self.count == self.policy.cap
            || self.count > 1
                && self
                    .deadline()
                    .is_some_and(|deadline| Instant::now() >= deadline)
        {
            self.flush(api)?;
        } else if self.count == 1 && self.policy.delay.is_some() {
            self.wake.notify_one();
        }
        Ok(())
    }

    pub fn flush(&mut self, api: &Api) -> Result<(), CuResult> {
        self.check()?;
        if self.buffered() {
            let slot = self.open.as_ref().unwrap();
            if let Err(error) = self.queue.execute(
                api,
                slot.list.as_ref().unwrap(),
                slot.fence.as_ref().unwrap(),
            ) {
                let error = backend_error(error);
                // A failed execute may have handed references to the driver.
                // Keep the slot and all of its references in quarantine.
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
            if Arc::strong_count(fence) > 1 || !fence.ready(api).map_err(backend_error)? {
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

    pub fn destroy(&mut self, api: &Api) -> Result<(), CuResult> {
        self.check()?;
        if self.buffered() || !self.pending.is_empty() {
            return Err(CuResult::NotReady);
        }
        if let Some(slot) = &mut self.open {
            slot.destroy(api)?;
            self.open = None;
        }
        while let Some(slot) = self.free.last_mut() {
            slot.destroy(api)?;
            self.free.pop();
        }
        self.queue.destroy(api).map_err(backend_error)
    }
}

pub(super) fn spawn(owner: usize, wake: Arc<Condvar>) -> Result<JoinHandle<()>, CuResult> {
    std::thread::Builder::new()
        .name("cuda-batch".into())
        .spawn(move || {
            let Ok(mut guard) = RUNTIME.lock() else {
                return;
            };
            loop {
                let Some(state) = guard.as_mut() else { return };
                let Some(context) = state.contexts.get_mut(&owner) else {
                    return;
                };
                if context.streams.worker.is_none() {
                    return;
                }
                let mut next = None;
                for batch in context
                    .streams
                    .all_mut()
                    .filter_map(|stream| stream.batch.as_mut())
                {
                    if batch.check().is_err() {
                        continue;
                    }
                    if let Some(deadline) = batch.deadline() {
                        if deadline <= Instant::now() {
                            batch.background_flush(&state.runtime.api);
                        } else {
                            next = Some(
                                next.map_or(deadline, |earliest: Instant| earliest.min(deadline)),
                            );
                        }
                    }
                }
                let result = if let Some(deadline) = next {
                    wake.wait_timeout(guard, deadline.saturating_duration_since(Instant::now()))
                        .map(|(guard, _)| guard)
                        .map_err(|_| ())
                } else {
                    wake.wait(guard).map_err(|_| ())
                };
                let Ok(new_guard) = result else { return };
                guard = new_guard;
            }
        })
        .map_err(|_| CuResult::OperatingSystem)
}
