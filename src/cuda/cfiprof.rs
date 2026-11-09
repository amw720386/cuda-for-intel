//! `CFI_PROFILE=1`: per-kernel, per-API and memcpy profile for decode/prefill analysis.
//!
//! Disabled (the default) every hook is one cached boolean load. When enabled, each kernel launch is
//! followed by a wait on its stream, so the reported kernel time is GPU time plus a constant submission
//! latency (host-side timing; Level Zero timestamp events are not used because they would need a
//! per-launch event pool in the hot path). Use it to rank kernels, not to read absolute speed.
//! The report goes to stderr and to the file named by `CFI_PROFILE_OUT` at process exit and on
//! context destruction, and every `CFI_PROFILE_FLUSH_MS` (default 2 s) while kernels run (written to
//! a temp file and renamed, so a reader never sees a partial file). Format: tab-separated lines,
//! parsed by `scripts/profile/summarize.py`.
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

pub(super) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("CFI_PROFILE").is_some_and(|v| v == "1"))
}

#[derive(Default, Clone, Debug, PartialEq)]
pub(super) struct KernelStat {
    pub count: u64,
    /// Wall time from before submission until the stream drained (serialised launch).
    pub total_ns: u64,
    /// Host time spent inside the submit call.
    pub launch_ns: u64,
    /// Host time spent waiting for completion after submit.
    pub sync_ns: u64,
}

#[derive(Default, Clone, Debug, PartialEq)]
pub(super) struct ApiStat {
    pub count: u64,
    pub total_ns: u64,
}

#[derive(Default, Clone, Debug, PartialEq)]
pub(super) struct CopyStat {
    pub count: u64,
    pub bytes: u64,
}

#[derive(Default)]
pub(super) struct Profile {
    pub kernels: HashMap<String, KernelStat>,
    pub apis: HashMap<&'static str, ApiStat>,
    pub copies: HashMap<&'static str, CopyStat>,
}

impl Profile {
    pub fn kernel(&mut self, name: &str, launch_ns: u64, sync_ns: u64) {
        let stat = self.kernels.entry(name.to_owned()).or_default();
        stat.count += 1;
        stat.launch_ns += launch_ns;
        stat.sync_ns += sync_ns;
        stat.total_ns += launch_ns + sync_ns;
    }

    pub fn api(&mut self, name: &'static str, ns: u64) {
        let stat = self.apis.entry(name).or_default();
        stat.count += 1;
        stat.total_ns += ns;
    }

    pub fn copy(&mut self, kind: &'static str, bytes: usize) {
        let stat = self.copies.entry(kind).or_default();
        stat.count += 1;
        stat.bytes += bytes as u64;
    }

    /// Sorted, tab-separated report. Kernels by total time (ties by name), APIs by total time,
    /// copies by bytes. Kernel names have tabs and newlines replaced by spaces.
    pub fn render(&self) -> String {
        let mut out = String::from("CFIPROF\t1\n");
        let mut kernels: Vec<_> = self.kernels.iter().collect();
        kernels.sort_by(|a, b| b.1.total_ns.cmp(&a.1.total_ns).then_with(|| a.0.cmp(b.0)));
        for (name, s) in kernels {
            let name: String = name.chars().map(|c| if c.is_control() { ' ' } else { c }).collect();
            out.push_str(&format!("K\t{}\t{}\t{}\t{}\t{name}\n", s.count, s.total_ns, s.launch_ns, s.sync_ns));
        }
        let mut apis: Vec<_> = self.apis.iter().collect();
        apis.sort_by(|a, b| b.1.total_ns.cmp(&a.1.total_ns).then_with(|| a.0.cmp(b.0)));
        for (name, s) in apis {
            out.push_str(&format!("A\t{}\t{}\t{name}\n", s.count, s.total_ns));
        }
        let mut copies: Vec<_> = self.copies.iter().collect();
        copies.sort_by(|a, b| b.1.bytes.cmp(&a.1.bytes).then_with(|| a.0.cmp(b.0)));
        for (kind, s) in copies {
            out.push_str(&format!("M\t{}\t{}\t{kind}\n", s.count, s.bytes));
        }
        out
    }
}

static PROFILE: Mutex<Option<Profile>> = Mutex::new(None);

unsafe extern "C" {
    fn atexit(callback: extern "C" fn()) -> i32;
}

extern "C" fn at_exit() {
    dump(true);
}

fn with(action: impl FnOnce(&mut Profile)) {
    static REGISTER: OnceLock<()> = OnceLock::new();
    REGISTER.get_or_init(|| {
        unsafe { atexit(at_exit) };
    });
    if let Ok(mut guard) = PROFILE.lock() {
        action(guard.get_or_insert_with(Profile::default));
    }
}

pub(super) fn record_kernel(name: &str, launch_ns: u64, sync_ns: u64) {
    with(|p| p.kernel(name, launch_ns, sync_ns));
    flush_periodically();
}

/// Rewrite `CFI_PROFILE_OUT` at most every `CFI_PROFILE_FLUSH_MS` (default 2000, 0 = only at exit),
/// checked after every kernel launch and traced API call.
/// Hosts such as Ollama terminate their runner process, so neither `atexit` nor context destruction
/// runs; the periodic rewrite keeps a cumulative file on disk that is at most one interval old.
fn flush_periodically() {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;
    static START: OnceLock<Instant> = OnceLock::new();
    static INTERVAL_MS: OnceLock<u64> = OnceLock::new();
    static LAST_MS: AtomicU64 = AtomicU64::new(0);
    let interval = *INTERVAL_MS.get_or_init(|| {
        if std::env::var_os("CFI_PROFILE_OUT").is_none() {
            return 0;
        }
        std::env::var("CFI_PROFILE_FLUSH_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(2000)
    });
    if interval == 0 {
        return;
    }
    let now = START.get_or_init(Instant::now).elapsed().as_millis() as u64;
    let last = LAST_MS.load(Ordering::Relaxed);
    if now.saturating_sub(last) >= interval
        && LAST_MS.compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed).is_ok()
    {
        dump(false);
    }
}

pub(super) fn record_api(name: &'static str, ns: u64) {
    with(|p| p.api(name, ns));
    flush_periodically();
}

pub(super) fn record_copy(kind: &'static str, bytes: usize) {
    if enabled() {
        with(|p| p.copy(kind, bytes));
    }
}

/// Write the report to `CFI_PROFILE_OUT` (overwrites, so it is always cumulative) and, when
/// `to_stderr`, to stderr.
pub(super) fn dump(to_stderr: bool) {
    if !enabled() {
        return;
    }
    let Ok(guard) = PROFILE.lock() else { return };
    let Some(profile) = guard.as_ref() else { return };
    let text = profile.render();
    if to_stderr {
        eprintln!("CUDA for Intel CFI_PROFILE report\n{text}");
    }
    if let Some(path) = std::env::var_os("CFI_PROFILE_OUT") {
        let mut tmp = path.clone();
        tmp.push(".tmp");
        if std::fs::write(&tmp, text).is_ok() {
            let _ = std::fs::rename(&tmp, &path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregates_and_sorts() {
        let mut p = Profile::default();
        p.kernel("small", 10, 5);
        p.kernel("big", 100, 900);
        p.kernel("big", 100, 800);
        p.kernel("tie_b", 1, 1);
        p.kernel("tie_a", 1, 1);
        p.api("cuCtxSynchronize", 7);
        p.api("cuLaunchKernel", 70);
        p.copy("HtoD", 4096);
        p.copy("HtoD", 4096);
        p.copy("DtoH", 8);
        assert_eq!(
            p.kernels["big"],
            KernelStat { count: 2, total_ns: 1900, launch_ns: 200, sync_ns: 1700 }
        );
        let text = p.render();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "CFIPROF\t1");
        assert_eq!(lines[1], "K\t2\t1900\t200\t1700\tbig");
        assert_eq!(lines[2], "K\t1\t15\t10\t5\tsmall");
        assert_eq!(lines[3], "K\t1\t2\t1\t1\ttie_a");
        assert_eq!(lines[4], "K\t1\t2\t1\t1\ttie_b");
        assert_eq!(lines[5], "A\t1\t70\tcuLaunchKernel");
        assert_eq!(lines[6], "A\t1\t7\tcuCtxSynchronize");
        assert_eq!(lines[7], "M\t2\t8192\tHtoD");
        assert_eq!(lines[8], "M\t1\t8\tDtoH");
    }

    #[test]
    fn sanitizes_names() {
        let mut p = Profile::default();
        p.kernel("a\tb\nc", 1, 1);
        assert!(p.render().contains("\ta b c\n"));
    }
}
