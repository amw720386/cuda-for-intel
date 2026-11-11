//! Bound on the work-groups queued on one stream (`CUDA_FOR_INTEL_MAX_INFLIGHT_GROUPS`).
//!
//! Level Zero rejected appends with OUT_OF_DEVICE_MEMORY when about a thousand unsynchronised
//! 14336-group launches piled up (`hw_big_dispatch_loop_unsynchronised`), so a stream waits once
//! too many work-groups are outstanding. `CUDA_FOR_INTEL_INFLIGHT=drain` waits for the whole
//! stream, which leaves the GPU idle until the next append arrives; with a 65536-group limit a
//! Qwen2.5-0.5B decode token (about 450k work-groups) drains about seven times.
//!
//! The default `window` mode keeps a similar bound without emptying the queue: every
//! `limit / 4` counted work-groups one launch carries a completion event (a checkpoint), and a
//! stream over the limit waits for its oldest checkpoints only, so later work keeps the GPU busy
//! meanwhile. A launch counts as at most `limit / 2` work-groups, so a launch larger than the limit
//! (the 151936-row output head) neither forces a drain itself nor makes the next launch wait for it.
//!
//! Bound: the counted outstanding work stays within `limit`, so at most two launches larger than
//! `limit / 2` are outstanding at once plus the one being appended, each at most `16 * limit`
//! work-groups (larger launches count in full and wait for an empty stream); everything smaller
//! counts in full. With the default limit that is a few hundred thousand work-groups for decode-shaped work,
//! far below the roughly 14 million queued work-groups that hit OUT_OF_DEVICE_MEMORY; drain mode
//! keeps at most `max(limit, one launch)`.
use super::Fence;
use std::collections::VecDeque;
use std::sync::Arc;

/// Default cap on work-groups queued on one stream before the host waits.
const DEFAULT_LIMIT: u64 = 65_536;
/// Window mode counts a launch above this many limits in full.
const HUGE_FACTOR: u64 = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Config {
    /// Outstanding work-groups per stream; 0 disables the bound.
    pub limit: u64,
    /// Wait for the oldest checkpoints instead of draining the stream.
    pub window: bool,
}

#[cfg(test)]
thread_local! {
    pub(super) static TEST_CONFIG: std::cell::Cell<Option<Config>> = const { std::cell::Cell::new(None) };
}

impl Config {
    /// Read when a context is created.
    pub fn configured() -> Self {
        #[cfg(test)]
        if let Some(config) = TEST_CONFIG.get() {
            return config;
        }
        Self::from(|name| std::env::var(name).ok())
    }

    fn from(get: impl Fn(&str) -> Option<String>) -> Self {
        Self {
            limit: get("CUDA_FOR_INTEL_MAX_INFLIGHT_GROUPS")
                .and_then(|value| value.trim().parse().ok())
                .unwrap_or(DEFAULT_LIMIT),
            // Window unless `drain` is asked for (Arc B390 decode p50 55.6 -> 61.4 tok/s FA off).
            window: !get("CUDA_FOR_INTEL_INFLIGHT").is_some_and(|mode| mode.trim().eq_ignore_ascii_case("drain")),
        }
    }

    /// Work-groups between two checkpoints in window mode.
    fn interval(self) -> u64 {
        (self.limit / 4).max(1)
    }

    /// Work-groups a launch of `groups` counts for: all of them in drain mode, at most half the
    /// limit in window mode, except that a launch above `HUGE_FACTOR * limit` counts in full (and so
    /// waits for the stream to empty), so a few huge launches cannot approach the OOM threshold.
    pub fn counted(self, groups: u64) -> u64 {
        if self.window && groups <= self.limit.saturating_mul(HUGE_FACTOR) {
            groups.min((self.limit / 2).max(1))
        } else {
            groups
        }
    }

    /// Whether `next` more counted work-groups must wait: outstanding plus the next launch would
    /// exceed the limit (the original drain rule; window mode passes counted sizes).
    pub fn needs_room(self, queued: u64, next: u64) -> bool {
        if self.limit == 0 || queued == 0 {
            return false;
        }
        queued.saturating_add(next) > self.limit
    }
}

/// Per-stream accounting of queued work-groups.
#[derive(Default)]
pub(super) struct Window {
    issued: u64,
    retired: u64,
    last_checkpoint: u64,
    /// `(issued count after the launch, its completion event)`, oldest first.
    checkpoints: VecDeque<(u64, Arc<Fence>)>,
    #[cfg(test)]
    drains: usize,
}

impl Window {
    pub fn queued(&self) -> u64 {
        self.issued - self.retired
    }

    pub fn wants_checkpoint(&self, config: Config, next: u64) -> bool {
        config.window
            && config.limit != 0
            && self.issued.saturating_add(next) - self.last_checkpoint >= config.interval()
    }

    pub fn launched(&mut self, groups: u64, checkpoint: Option<Arc<Fence>>) {
        self.issued = self.issued.saturating_add(groups);
        if let Some(fence) = checkpoint {
            self.last_checkpoint = self.issued;
            self.checkpoints.push_back((self.issued, fence));
        }
    }

    pub fn oldest(&mut self) -> Option<(u64, Arc<Fence>)> {
        self.checkpoints.pop_front()
    }

    /// Everything up to issued count `at` has completed.
    pub fn retire(&mut self, at: u64) {
        self.retired = self.retired.max(at);
    }

    /// The whole stream has completed.
    pub fn complete(&mut self) {
        #[cfg(test)]
        {
            self.drains += 1;
        }
        self.retired = self.issued;
        self.last_checkpoint = self.issued;
        self.checkpoints.clear();
    }

    #[cfg(test)]
    pub fn checkpoints(&self) -> usize {
        self.checkpoints.len()
    }

    #[cfg(test)]
    pub fn retired(&self) -> u64 {
        self.retired
    }

    #[cfg(test)]
    pub fn drains(&self) -> usize {
        self.drains
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drain(limit: u64) -> Config {
        Config { limit, window: false }
    }

    #[test]
    fn drain_mode_keeps_the_original_rule() {
        let config = drain(65_536);
        assert!(!config.needs_room(0, 14_336));
        assert!(!config.needs_room(0, 1 << 20));
        assert!(!config.needs_room(14_336 * 3, 14_336));
        assert!(config.needs_room(14_336 * 4, 14_336));
        assert!(!config.needs_room(60_000, 1));
        assert!(config.needs_room(u64::MAX, 1));
        assert!(!config.needs_room(50_000, 1));
        assert!(!drain(0).needs_room(u64::MAX, u64::MAX));
    }

    #[test]
    fn window_mode_caps_the_next_launch_at_half_the_limit() {
        let config = Config { limit: 65_536, window: true };
        assert_eq!(config.counted(151_936), 32_768);
        assert_eq!(config.counted(14_336), 14_336);
        assert_eq!(drain(65_536).counted(151_936), 151_936);
        assert_eq!(config.counted(16 * 65_536), 32_768);
        assert_eq!(config.counted(16 * 65_536 + 1), 16 * 65_536 + 1);
        assert!(!config.needs_room(32_768, config.counted(151_936)));
        assert!(config.needs_room(32_769, config.counted(151_936)));
        assert!(config.needs_room(60_000, 14_336));
        assert!(!config.needs_room(0, config.counted(151_936)));
    }

    #[test]
    fn configuration_parses_both_variables() {
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            move |name: &str| pairs.iter().find(|(key, _)| *key == name).map(|(_, value)| value.to_string())
        };
        let window = |limit| Config { limit, window: true };
        assert_eq!(Config::from(env(&[])), window(DEFAULT_LIMIT));
        assert_eq!(Config::from(env(&[("CUDA_FOR_INTEL_MAX_INFLIGHT_GROUPS", "0")])), window(0));
        assert_eq!(Config::from(env(&[("CUDA_FOR_INTEL_MAX_INFLIGHT_GROUPS", "x")])), window(DEFAULT_LIMIT));
        assert_eq!(Config::from(env(&[("CUDA_FOR_INTEL_INFLIGHT", "")])), window(DEFAULT_LIMIT));
        assert_eq!(Config::from(env(&[("CUDA_FOR_INTEL_INFLIGHT", " DRAIN ")])), drain(DEFAULT_LIMIT));
        assert_eq!(
            Config::from(env(&[("CUDA_FOR_INTEL_INFLIGHT", "drain"), ("CUDA_FOR_INTEL_MAX_INFLIGHT_GROUPS", "1000")])),
            drain(1000)
        );
        assert_eq!(
            Config::from(env(&[("CUDA_FOR_INTEL_INFLIGHT", "Window"), ("CUDA_FOR_INTEL_MAX_INFLIGHT_GROUPS", "1000")])),
            Config { limit: 1000, window: true }
        );
        assert_eq!(Config::from(env(&[("CUDA_FOR_INTEL_INFLIGHT", "drain")])), drain(DEFAULT_LIMIT));
    }

    #[test]
    fn checkpoints_every_quarter_limit() {
        let config = Config { limit: 400, window: true };
        let mut window = Window::default();
        let mut marked = Vec::new();
        for launch in 0..20 {
            let mark = window.wants_checkpoint(config, 30);
            window.launched(30, None);
            if mark {
                window.last_checkpoint = window.issued;
                marked.push(launch);
            }
        }
        // 100-group interval with 30-group launches: every fourth launch.
        assert_eq!(marked, [3, 7, 11, 15, 19]);
        assert!(!window.wants_checkpoint(drain(400), 1000));
        window.complete();
        assert_eq!(window.queued(), 0);
        assert!(!window.wants_checkpoint(config, 99));
        assert!(window.wants_checkpoint(config, 100));
    }
}

#[cfg(test)]
mod fake_tests {
    use super::super::fake_tests::{BATCH, Fixture, allocations, assert_baseline, context, function, module, stream};
    use super::*;
    use super::super::cuStreamSynchronize;
    use crate::cuda::*;
    use std::ptr::null_mut;

    fn launch(function: Opaque, stream: Opaque, address: u64, groups: u32) -> CuResult {
        let mut arg = std::ptr::from_ref(&address).cast_mut().cast();
        unsafe { cuLaunchKernel(function, groups, 1, 1, 1, 1, 1, 0, stream, &mut arg, null_mut()) }
    }

    #[test]
    fn fake_window_waits_for_checkpoints_instead_of_draining() {
        let _fixture = Fixture::new();
        for policy in [None, BATCH] {
            for window in [false, true] {
                TEST_CONFIG.set(Some(Config { limit: 40, window }));
                let ctx = context(policy);
                TEST_CONFIG.set(None);
                let (s, f) = (stream(), function(module()));
                let (device, _) = allocations();
                // Blocking legacy stream as well: its launches already carry a fence each.
                for target in [s, null_mut()] {
                    let key = super::super::stream_key(target).unwrap();
                    let drains = |context: &mut CudaContext| context.streams.get(key).map(|stream| stream.window.drains());
                    let before = with_context(|_, context| drains(context)).unwrap();
                    for _ in 0..200 {
                        assert_eq!(launch(f, target, device, 3), Success);
                    }
                    // One launch larger than the limit.
                    assert_eq!(launch(f, target, device, 100), Success);
                    assert_eq!(launch(f, target, device, 3), Success);
                    with_context(|_, context| {
                        let drained = drains(context)? - before;
                        let window_state = &context.streams.get(key)?.window;
                        assert!(window_state.queued() <= 40 + 100, "{}", window_state.queued());
                        if window {
                            assert_eq!(drained, 0, "window mode drained: batch {} legacy {}", policy.is_some(), target.is_null());
                            assert!(window_state.retired() > 0, "no checkpoint was waited for");
                            assert!(window_state.checkpoints() <= 6, "{}", window_state.checkpoints());
                        } else {
                            assert!(drained >= 10, "{drained}");
                            assert_eq!(window_state.checkpoints(), 0);
                        }
                        Ok(())
                    })
                    .unwrap();
                    assert_eq!(cuStreamSynchronize(target), Success);
                    with_context(|_, context| {
                        assert_eq!(context.streams.get(key)?.window.checkpoints(), 0, "batch {} window {window} legacy {}", policy.is_some(), target.is_null());
                        Ok(())
                    })
                    .unwrap();
                }
                assert_eq!(cuCtxDestroy_v2(ctx), Success);
                assert_baseline();
            }
        }
    }
}
