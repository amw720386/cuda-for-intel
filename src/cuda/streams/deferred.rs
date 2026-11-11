//! Deferred kernel submission on an immediate stream (`CUDA_FOR_INTEL_SUBMIT=deferred`).
//!
//! Launches without stream-ordering events are recorded into an in-order regular command list
//! instead of the stream's immediate list. The regular list is closed and appended to the immediate
//! list (`zeCommandListImmediateAppendCommandListsExp`) when it holds `threshold` launches, and
//! before any other operation on the stream (copy, fill, event record or wait, timestamp) and any
//! host synchronization or query, so CUDA ordering is unchanged: everything still executes in
//! order on the stream's one immediate list. `threshold` starts at 4 after the stream goes idle and
//! doubles per flush up to `CUDA_FOR_INTEL_SUBMIT_CAP` (default 64), so the GPU starts early and
//! later chunks amortise the per-submission cost (the shape of ggml-vulkan's submit cadence).
//!
//! Appended lists are reset for reuse once the stream has completed (every host synchronization
//! that observes the whole stream); more than `MAX_SUBMITTED` outstanding lists force a wait.
use super::{Api, CommandList, Context, CuResult, backend_error};
use crate::level_zero::{DeviceHealth, Error};
use std::sync::Arc;

const FIRST_THRESHOLD: usize = 4;
const DEFAULT_CAP: usize = 64;
const MAX_SUBMITTED: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Config {
    pub cap: usize,
}

#[cfg(test)]
thread_local! {
    pub(super) static TEST_CONFIG: std::cell::Cell<Option<Option<Config>>> = const { std::cell::Cell::new(None) };
}

impl Config {
    /// Read when a context is created; `None` keeps direct immediate appends.
    pub fn configured(api: &Api) -> Option<Self> {
        #[cfg(test)]
        if let Some(config) = TEST_CONFIG.get() {
            return config;
        }
        let config = Self::from(|name| std::env::var(name).ok())?;
        if api.append_lists.is_none() {
            eprintln!(
                "cuda-for-intel: CUDA_FOR_INTEL_SUBMIT=deferred needs zeCommandListImmediateAppendCommandListsExp; using immediate submission"
            );
            return None;
        }
        Some(config)
    }

    fn from(get: impl Fn(&str) -> Option<String>) -> Option<Self> {
        let mode = get("CUDA_FOR_INTEL_SUBMIT")?;
        if !mode.trim().eq_ignore_ascii_case("deferred") {
            return None;
        }
        let cap = get("CUDA_FOR_INTEL_SUBMIT_CAP")
            .and_then(|cap| cap.trim().parse::<usize>().ok())
            .filter(|cap| (1..=4096).contains(cap))
            .unwrap_or(DEFAULT_CAP);
        Some(Self { cap })
    }
}

pub(super) struct Deferred {
    config: Config,
    threshold: usize,
    open: Option<CommandList>,
    count: usize,
    /// Closed lists appended to the immediate list and possibly still executing.
    submitted: Vec<CommandList>,
    /// Lists known complete; reset before reuse.
    idle: Vec<CommandList>,
    health: Arc<DeviceHealth>,
    #[cfg(test)]
    pub flushes: usize,
}

impl Deferred {
    pub fn new(config: Config, health: Arc<DeviceHealth>) -> Self {
        Self {
            config,
            threshold: FIRST_THRESHOLD.min(config.cap),
            open: None,
            count: 0,
            submitted: Vec::new(),
            idle: Vec::new(),
            health,
            #[cfg(test)]
            flushes: 0,
        }
    }

    fn observe(&self, error: Error) -> CuResult {
        self.health.observe(&error);
        backend_error(error)
    }

    #[cfg(test)]
    pub fn buffered(&self) -> usize {
        self.count
    }

    /// The regular list the next deferred launch goes to.
    pub fn recording(&mut self, api: &Api, context: &Context) -> Result<&CommandList, CuResult> {
        if self.open.is_none() {
            let list = match self.idle.pop() {
                Some(list) => {
                    if let Err(error) = list.reset(api) {
                        // Keep ownership so teardown still destroys it.
                        self.idle.push(list);
                        return Err(self.observe(error));
                    }
                    list
                }
                None => context.create_list(api).map_err(|error| self.observe(error))?,
            };
            self.open = Some(list);
        }
        Ok(self.open.as_ref().unwrap())
    }

    /// A launch was appended to the open list. Returns whether the list should be flushed now.
    pub fn appended(&mut self) -> bool {
        self.count += 1;
        self.count >= self.threshold
    }

    /// Close the open list and append it to `immediate`. With nothing buffered this is a no-op.
    /// On failure the buffered launches are lost and the caller marks the stream failed.
    pub fn flush(&mut self, api: &Api, immediate: &CommandList) -> Result<(), CuResult> {
        if self.count == 0 {
            return Ok(());
        }
        if self.submitted.len() >= MAX_SUBMITTED {
            // Unsynchronised for a long time: wait so the lists can be recycled.
            immediate
                .wait(api, u64::MAX)
                .map_err(|error| self.observe(error))?;
            self.complete();
        }
        let list = self.open.take().unwrap();
        self.count = 0;
        let result = list.close(api).and_then(|()| immediate.append_lists(api, &list, None, &[]));
        // Even on failure the driver may reference the list; treat it as in flight.
        self.submitted.push(list);
        result.map_err(|error| self.observe(error))?;
        self.threshold = (self.threshold * 2).min(self.config.cap);
        #[cfg(test)]
        {
            self.flushes += 1;
        }
        Ok(())
    }

    /// The stream's immediate list has completed everything appended so far.
    pub fn complete(&mut self) {
        self.idle.append(&mut self.submitted);
        self.threshold = FIRST_THRESHOLD.min(self.config.cap);
    }

    /// Only after the stream has drained.
    pub fn destroy(&mut self, api: &Api) -> Result<(), CuResult> {
        for lists in [&mut self.idle, &mut self.submitted] {
            while let Some(list) = lists.last() {
                list.destroy(api).map_err(backend_error)?;
                lists.pop();
            }
        }
        if let Some(list) = &self.open {
            list.destroy(api).map_err(backend_error)?;
            self.open = None;
            self.count = 0;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configuration() {
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            move |name: &str| pairs.iter().find(|(key, _)| *key == name).map(|(_, value)| value.to_string())
        };
        assert_eq!(Config::from(env(&[])), None);
        assert_eq!(Config::from(env(&[("CUDA_FOR_INTEL_SUBMIT", "immediate")])), None);
        assert_eq!(Config::from(env(&[("CUDA_FOR_INTEL_SUBMIT", "Deferred")])), Some(Config { cap: 64 }));
        assert_eq!(
            Config::from(env(&[("CUDA_FOR_INTEL_SUBMIT", "deferred"), ("CUDA_FOR_INTEL_SUBMIT_CAP", "256")])),
            Some(Config { cap: 256 })
        );
        assert_eq!(
            Config::from(env(&[("CUDA_FOR_INTEL_SUBMIT", "deferred"), ("CUDA_FOR_INTEL_SUBMIT_CAP", "0")])),
            Some(Config { cap: 64 })
        );
    }
}

#[cfg(test)]
mod fake_tests {
    use super::super::events::{cuEventRecord, cuEventSynchronize, cuStreamWaitEvent};
    use super::super::fake_tests::{
        BATCH, Fixture, allocations, assert_baseline, context, event, function, module, stream,
    };
    use super::super::{cuStreamQuery, cuStreamSynchronize, stream_key};
    use super::*;
    use crate::cuda::*;
    use crate::level_zero::fake::{self, Kind};
    use crate::level_zero::fault::arm;
    use std::ptr::null_mut;

    fn deferred_context(cap: usize) -> Opaque {
        TEST_CONFIG.set(Some(Some(Config { cap })));
        let ctx = context(None);
        TEST_CONFIG.set(None);
        ctx
    }

    fn launch(function: Opaque, stream: Opaque, address: u64) -> CuResult {
        let mut arg = std::ptr::from_ref(&address).cast_mut().cast();
        unsafe { cuLaunchKernel(function, 2, 1, 1, 1, 1, 1, 0, stream, &mut arg, null_mut()) }
    }

    /// (launches buffered in the open list, lists appended so far) on `stream`.
    fn state(stream: Opaque) -> (usize, usize) {
        let key = stream_key(stream).unwrap();
        with_context(|_, context| {
            let deferred = context.streams.get(key)?.deferred.as_ref().unwrap();
            Ok((deferred.buffered(), deferred.flushes))
        })
        .unwrap()
    }

    #[test]
    fn fake_deferred_submission_ramps_and_flushes_at_sync_points() {
        let _fixture = Fixture::new();
        let ctx = deferred_context(16);
        let (s, f) = (stream(), function(module()));
        let (device, host) = allocations();
        // Thresholds 4, 8, 16, 16, ...: 4 + 8 + 16 = 28 launches make three appends.
        for _ in 0..30 {
            assert_eq!(launch(f, s, device), Success);
        }
        assert_eq!(state(s), (2, 3));
        // Copies run after the buffered launches.
        assert_eq!(unsafe { cuMemcpyDtoHAsync_v2(host, device, 4, s) }, Success);
        assert_eq!(state(s), (0, 4));
        // Event record, cross-stream wait and query flush as well.
        let (other, e) = (stream(), event());
        assert_eq!(launch(f, s, device), Success);
        assert_eq!(cuEventRecord(e, s), Success);
        assert_eq!(state(s), (0, 5));
        assert_eq!(cuStreamWaitEvent(other, e, 0), Success);
        assert_eq!(launch(f, other, device), Success);
        assert_eq!(state(other), (1, 0));
        assert_eq!(cuStreamQuery(other), Success);
        assert_eq!(state(other), (0, 1));
        assert_eq!(cuEventSynchronize(e), Success);
        // A host synchronize submits, waits and recycles the lists; the ramp restarts at 4.
        for _ in 0..3 {
            assert_eq!(launch(f, s, device), Success);
        }
        assert_eq!(cuStreamSynchronize(s), Success);
        assert_eq!(state(s), (0, 6));
        for _ in 0..4 {
            assert_eq!(launch(f, s, device), Success);
        }
        assert_eq!(state(s), (0, 7));
        // The legacy stream defers too, and blocking work orders after it.
        assert_eq!(launch(f, null_mut(), device), Success);
        assert_eq!(state(null_mut()), (1, 0));
        assert_eq!(cuCtxSynchronize(), Success);
        assert_eq!(state(null_mut()), (0, 1));
        assert_eq!(cuCtxDestroy_v2(ctx), Success);
        assert_baseline();
    }

    #[test]
    fn fake_deferred_lists_are_recycled_across_tokens() {
        let _fixture = Fixture::new();
        let ctx = deferred_context(64);
        let (s, f) = (stream(), function(module()));
        let (device, host) = allocations();
        for _ in 0..50 {
            for _ in 0..700 {
                assert_eq!(launch(f, s, device), Success);
            }
            assert_eq!(unsafe { cuMemcpyDtoHAsync_v2(host, device, 4, s) }, Success);
            assert_eq!(cuStreamSynchronize(s), Success);
        }
        // 4 + 8 + 16 + 32 + 64 * 10 = 700: 14 regular lists per token, reused every token, plus
        // the two streams' immediate lists and the context's synchronous list.
        let lists = fake::snapshot().live.get(&Kind::List).copied().unwrap_or(0);
        assert_eq!(lists, 14 + 3, "lists alive");
        assert_eq!(cuCtxDestroy_v2(ctx), Success);
        assert_baseline();
    }

    #[test]
    fn fake_deferred_append_failure_fails_the_stream_and_tears_down() {
        let _fixture = Fixture::new();
        let ctx = deferred_context(8);
        let (s, f) = (stream(), function(module()));
        let (device, _) = allocations();
        for _ in 0..3 {
            assert_eq!(launch(f, s, device), Success);
        }
        arm("zeCommandListImmediateAppendCommandListsExp", 0x7fff_fffe);
        assert_eq!(launch(f, s, device), Unknown);
        assert_eq!(launch(f, s, device), Unknown);
        assert_eq!(cuStreamSynchronize(s), Unknown);
        // Teardown reports the stream's error once and still releases everything.
        assert_eq!(cuCtxDestroy_v2(ctx), Unknown);
        assert_eq!(cuCtxDestroy_v2(ctx), InvalidContext);
        assert_baseline();
    }

    #[test]
    fn fake_deferred_is_off_for_batching_and_by_default() {
        let _fixture = Fixture::new();
        for (submit, policy) in [(None, None), (Some(Config { cap: 8 }), BATCH)] {
            TEST_CONFIG.set(Some(submit));
            let ctx = context(policy);
            TEST_CONFIG.set(None);
            let s = stream();
            with_context(|_, context| {
                assert!(context.streams.get(stream_key(s)?)?.deferred.is_none());
                Ok(())
            })
            .unwrap();
            assert_eq!(cuCtxDestroy_v2(ctx), Success);
            assert_baseline();
        }
    }
}
