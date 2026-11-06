use super::Handle;
use std::time::Duration;

pub(super) const GUARD: usize = 8;
pub(super) const SENTINEL: f32 = -99_999.0;
pub(super) const EPSILON: f32 = 1e-5;
pub(super) const BATCH: usize = 32;
pub(super) const KERNELS: [(&std::ffi::CStr, &str); 3] = [
    (c"rmsnorm", include_str!("../../examples/rmsnorm.ptx")),
    (
        c"forge_residual",
        include_str!("../../examples/forge_residual.ptx"),
    ),
    (
        c"forge_reduce",
        include_str!("../../examples/forge_reduce.ptx"),
    ),
];

pub(crate) fn arg<T>(value: &T) -> Handle {
    std::ptr::from_ref(value).cast_mut().cast()
}

pub(super) fn report(label: &str, samples: &mut [Duration]) {
    samples.sort_unstable();
    let micros = |duration: Duration| duration.as_secs_f64() * 1e6;
    let p95 = (95 * samples.len()).div_ceil(100) - 1;
    println!(
        "{label}: n={} median={:.3} us p95={:.3} us",
        samples.len(),
        micros(samples[samples.len() / 2]),
        micros(samples[p95]),
    );
}
