//! Background translation and native compilation of the kernels a library is likely to launch
//! next, so a cold start does not pay translation plus Intel-compiler time serially in the
//! launch path.
//!
//! When the first kernel of a library is resolved, every other kernel of that library is queued
//! (most likely first) on a small thread pool. Each job translates PTX -> SPIR-V into the
//! on-disk translation cache (`xcache`) and compiles the SPIR-V into the on-disk native-binary
//! cache (`native_cache`). A later launch then finds both caches warm. The launch path never
//! waits for prefetch work unless it needs the very kernel a worker is already producing.
//!
//! The SPIR-V is exactly what the foreground path would produce (same translator, same keys).
//!
//! Environment:
//! * `CUDA_FOR_INTEL_PREFETCH=0` disables everything here.
//! * `CUDA_FOR_INTEL_PREFETCH_THREADS=N` pool size (default `min(8, cores - 2)`, at least 1).
//! * `CUDA_FOR_INTEL_PREFETCH_COMPILE=N` per-library cap on background Intel-compiler jobs
//!   (default 48; 0 translates only).
//! * `CUDA_FOR_INTEL_WARMUP=1` removes the caps: every kernel of every library is translated
//!   and compiled (a one-off cache fill; see docs/first-run-latency.md).

use super::xcache;
use crate::level_zero::NativeCompiler;
use std::collections::{HashMap, VecDeque};
use std::ffi::CString;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

/// Source text of one library plus the work derived from it once.
pub(crate) struct LibShared {
    pub(crate) text: String,
    pub(crate) hash: u64,
    prepared: OnceLock<Option<Arc<String>>>,
}

impl LibShared {
    pub(crate) fn new(text: String) -> Self {
        let hash = xcache::module_hash(&text);
        Self { text, hash, prepared: OnceLock::new() }
    }

    /// Normalised PTX for per-kernel translation; computed by whichever thread asks first.
    pub(crate) fn prepared(&self) -> Option<Arc<String>> {
        self.prepared
            .get_or_init(|| {
                let started = std::time::Instant::now();
                let prepared = crate::ptx::prepare_source(&self.text).ok().map(Arc::new);
                super::export_log(format_args!(
                    "lazy library prepared: ptx_bytes={} ok={} prepare_ms={}",
                    self.text.len(),
                    prepared.is_some(),
                    started.elapsed().as_millis()
                ));
                prepared
            })
            .clone()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Slot {
    Queued,
    Working,
}

struct Job {
    key: u64,
    shared: Arc<LibShared>,
    name: String,
    compiler: Option<Arc<NativeCompiler>>,
    /// Remaining background Intel-compiler jobs for this library.
    compile_budget: Arc<AtomicUsize>,
}

struct Pool {
    queue: Mutex<VecDeque<Job>>,
    wake: Condvar,
    slots: Mutex<HashMap<u64, Slot>>,
    done: Condvar,
}

fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("CUDA_FOR_INTEL_PREFETCH").as_deref() != Some(std::ffi::OsStr::new("0")))
}

fn warmup() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("CUDA_FOR_INTEL_WARMUP").is_some_and(|v| v != "0"))
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.trim().parse().ok()
}

static POOL: OnceLock<&'static Pool> = OnceLock::new();

fn pool() -> &'static Pool {
    POOL.get_or_init(|| {
        let pool: &'static Pool = Box::leak(Box::new(Pool {
            queue: Mutex::new(VecDeque::new()),
            wake: Condvar::new(),
            slots: Mutex::new(HashMap::new()),
            done: Condvar::new(),
        }));
        let cores = std::thread::available_parallelism().map_or(2, usize::from);
        let threads = env_usize("CUDA_FOR_INTEL_PREFETCH_THREADS").unwrap_or_else(|| cores.saturating_sub(2).min(8)).max(1);
        for n in 0..threads {
            let _ = std::thread::Builder::new()
                .name(format!("cfi-prefetch-{n}"))
                .stack_size(16 << 20)
                .spawn(move || worker(pool));
        }
        pool
    })
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Lower is more likely to be launched soon: the kernel families an LLM step touches first,
/// then kernels sharing a name prefix with the one that was just requested.
fn rank(name: &str, requested: &str) -> u8 {
    const HOT: [&str; 12] = [
        "mul_mat_vec", "mul_mat_q", "mmq", "mmvq", "quantize", "norm", "soft_max", "rope", "k_get_rows", "dequantize", "scale", "silu",
    ];
    let family = name.bytes().zip(requested.bytes()).take_while(|(a, b)| a == b).count() >= 16;
    let hot = HOT.iter().any(|h| name.contains(h));
    match (hot, family) {
        (true, true) => 0,
        (true, false) => 1,
        (false, true) => 2,
        (false, false) => 3,
    }
}

/// Queue the other kernels of a library. Call once per library, after the kernel the caller
/// needs has been resolved.
pub(crate) fn start_library(shared: &Arc<LibShared>, names: &[CString], requested: &str, compiler: Option<NativeCompiler>) {
    if !enabled() || cfg!(test) {
        return;
    }
    let pool = pool();
    let compiler = compiler.map(Arc::new);
    let budget = if warmup() { usize::MAX } else { env_usize("CUDA_FOR_INTEL_PREFETCH_COMPILE").unwrap_or(48) };
    let compile_budget = Arc::new(AtomicUsize::new(budget));
    let mut order: Vec<String> = names.iter().map(|n| n.to_string_lossy().into_owned()).filter(|n| n != requested).collect();
    order.sort_by_key(|n| rank(n, requested));
    if !warmup() {
        order.truncate(env_usize("CUDA_FOR_INTEL_PREFETCH_ENTRIES").unwrap_or(1024));
    }
    let mut slots = lock(&pool.slots);
    let mut queue = lock(&pool.queue);
    for name in order {
        let key = xcache::key_for_hash(shared.hash, &name);
        if slots.contains_key(&key) {
            continue;
        }
        slots.insert(key, Slot::Queued);
        queue.push_back(Job { key, shared: Arc::clone(shared), name, compiler: compiler.clone(), compile_budget: Arc::clone(&compile_budget) });
    }
    drop(queue);
    drop(slots);
    pool.wake.notify_all();
}

/// Called by the launch path before it translates `key` itself. A queued job is cancelled (the
/// caller does the work now); a job already running is awaited so the work is not done twice.
pub(crate) fn claim_or_wait(key: u64) {
    if !enabled() {
        return;
    }
    let Some(&pool) = POOL.get() else { return };
    let mut slots = lock(&pool.slots);
    loop {
        match slots.get(&key) {
            None => return,
            Some(Slot::Queued) => {
                slots.remove(&key);
                return;
            }
            Some(Slot::Working) => {
                slots = pool.done.wait(slots).unwrap_or_else(|poisoned| poisoned.into_inner());
            }
        }
    }
}

fn worker(pool: &'static Pool) {
    loop {
        let job = {
            let mut queue = lock(&pool.queue);
            loop {
                if let Some(job) = queue.pop_front() {
                    break job;
                }
                queue = pool.wake.wait(queue).unwrap_or_else(|poisoned| poisoned.into_inner());
            }
        };
        {
            // Skip jobs the launch path already took over.
            let mut slots = lock(&pool.slots);
            match slots.get(&job.key) {
                Some(Slot::Queued) => {
                    slots.insert(job.key, Slot::Working);
                }
                _ => continue,
            }
        }
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&job)));
        lock(&pool.slots).remove(&job.key);
        pool.done.notify_all();
    }
}

fn run(job: &Job) {
    let started = std::time::Instant::now();
    let cached = xcache::get(job.key);
    let translated = match cached {
        Some(hit) => hit,
        None => {
            let Some(prepared) = job.shared.prepared() else { return };
            let Ok(t) = crate::ptx::translate_prepared_entry(&prepared, &job.name) else { return };
            xcache::put(job.key, &t);
            t
        }
    };
    // Kernels with module variables keep the whole-library path.
    if !translated.globals.is_empty() {
        return;
    }
    let translate_ms = started.elapsed().as_millis();
    let mut compile_ms = 0;
    if let Some(compiler) = &job.compiler
        && job
            .compile_budget
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| if n == usize::MAX { Some(n) } else { n.checked_sub(1) })
            .is_ok()
    {
        let compile_started = std::time::Instant::now();
        compiler.precompile(&translated.words);
        compile_ms = compile_started.elapsed().as_millis();
    }
    super::export_log(format_args!(
        "prefetched {}: translate_ms={translate_ms} compile_ms={compile_ms}",
        job.name.chars().take(80).collect::<String>()
    ));
}

/// Smoke test hooks for unit tests.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_prefers_hot_family() {
        assert!(rank("_Z12mul_mat_vec_qIx", "_Z12mul_mat_vec_qIy") < rank("_Z7unknown", "_Z12mul_mat_vec_qIy"));
        assert_eq!(rank("zzz", "aaa"), 3);
    }
}
