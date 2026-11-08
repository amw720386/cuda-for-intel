//! Per-device translator probes.
//!
//! The PTX translator is configured through two process-wide switches that depend on
//! what the target GPU can do: the dynamic shared-memory window and the integer
//! dot-product lowering of `dp4a`. Devices of one machine may differ, so each device
//! is probed once, remembers its answer, and the answer is applied to the translator
//! right before a module for that device is translated (translation runs under the
//! runtime lock, so no other device's module can interleave).

use crate::level_zero::{Device, DeviceInfo};

/// Default window when `CUDA_FOR_INTEL_DYN_SHARED` is unset: GGML's larger MMQ tiles
/// launch with more than 48 KiB of dynamic shared memory.
pub(crate) const DEFAULT_WINDOW: u32 = 65536;
/// What CUDA guarantees without opt-in.
const BASELINE_WINDOW: u32 = 49152;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DeviceProbe {
    /// Bytes reserved per workgroup for `extern __shared__` blocks.
    pub dynamic_shared: u32,
    /// Whether `dp4a` may lower to the SPIR-V integer dot product.
    pub dot_product: bool,
}

/// Environment inputs of a probe, passed explicitly so tests need no process state.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ProbeEnv {
    /// `CUDA_FOR_INTEL_DYN_SHARED`
    pub wanted_window: Option<u32>,
    /// `CUDA_FOR_INTEL_DP4A=1`: the integer dot product is opt-in.
    pub dp4a_requested: bool,
}

impl ProbeEnv {
    pub(crate) fn from_process() -> Self {
        Self {
            wanted_window: std::env::var("CUDA_FOR_INTEL_DYN_SHARED")
                .ok()
                .and_then(|v| v.parse().ok()),
            dp4a_requested: std::env::var_os("CUDA_FOR_INTEL_DP4A").as_deref()
                == Some(std::ffi::OsStr::new("1")),
        }
    }
}

/// Dynamic shared window for a device with `device_max` bytes of shared local memory.
pub(crate) fn window(device_max: u32, wanted: Option<u32>) -> u32 {
    wanted
        .unwrap_or(DEFAULT_WINDOW)
        .min(device_max)
        .max(BASELINE_WINDOW.min(device_max))
}

/// Probe one device. `dot_product_supported` runs only when the opt-in is set, and
/// compiles the dp4a probe module on that very device.
pub(crate) fn compute(
    info: &DeviceInfo,
    env: ProbeEnv,
    dot_product_supported: impl FnOnce() -> bool,
) -> DeviceProbe {
    DeviceProbe {
        dynamic_shared: window(info.max_shared_local_memory, env.wanted_window),
        dot_product: env.dp4a_requested && dot_product_supported(),
    }
}

/// The device's probe, computed on first use.
pub(crate) fn ensure(device: &Device, compute: impl FnOnce() -> DeviceProbe) -> DeviceProbe {
    *device.probe.get_or_init(compute)
}

impl DeviceProbe {
    /// Make the translator produce code for this device.
    pub(crate) fn apply(&self) {
        crate::ptx::set_dynamic_shared_bytes(self.dynamic_shared);
        crate::ptx::set_dot_product(self.dot_product);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_follows_device_capacity() {
        assert_eq!(window(131072, None), 65536);
        assert_eq!(window(65536, None), 65536);
        assert_eq!(window(32768, None), 32768, "never above the device limit");
        assert_eq!(window(131072, Some(1024)), 49152, "never below the CUDA baseline");
        assert_eq!(window(131072, Some(98304)), 98304);
        assert_eq!(window(20000, Some(1024)), 20000);
    }
}
