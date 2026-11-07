use super::CuResult;
use std::fmt;
use std::io::Write;
use std::sync::OnceLock;

pub(super) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var_os("CUDA_FOR_INTEL_TRACE").is_some_and(|value| value == "1"))
}

pub(super) fn record(message: fmt::Arguments<'_>) {
    let _ = writeln!(std::io::stderr().lock(), "CUDA for Intel trace {message}");
}

/// `CUDA_FOR_INTEL_PROFILE=1`: per-API call counts and wall time, appended to the export log
/// as cumulative `PROF` lines about every two seconds.
pub(super) fn profile_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("CUDA_FOR_INTEL_PROFILE").as_deref() == Some(std::ffi::OsStr::new("1")))
}

type ProfileTable = std::collections::HashMap<&'static str, (u64, u64)>;

pub(super) fn profile_record(api: &'static str, nanos: u64) {
    static TABLE: std::sync::Mutex<Option<(ProfileTable, std::time::Instant)>> = std::sync::Mutex::new(None);
    let Ok(mut guard) = TABLE.lock() else { return };
    let (table, last) = guard.get_or_insert_with(|| (ProfileTable::new(), std::time::Instant::now()));
    let entry = table.entry(api).or_default();
    entry.0 += 1;
    entry.1 += nanos;
    if last.elapsed().as_secs() >= 2 {
        *last = std::time::Instant::now();
        let mut rows: Vec<_> = table.iter().collect();
        rows.sort();
        let mut text = String::from("PROF-BEGIN");
        for (name, (count, total)) in rows {
            text.push_str(&format!("\nPROF {name} {count} {}", total / 1000));
        }
        super::export_log(text);
    }
}

pub(super) fn call(api: &'static str, action: impl FnOnce() -> CuResult) -> CuResult {
    let started = profile_enabled().then(std::time::Instant::now);
    let result = action();
    if let Some(started) = started {
        profile_record(api, started.elapsed().as_nanos() as u64);
    }
    if enabled() {
        record(format_args!("{api} -> {}", result as i32));
    }
    result
}

pub(super) fn symbol(value: &[u8]) -> &str {
    if value.starts_with(b"cu")
        && value.len() <= 128
        && value
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
    {
        std::str::from_utf8(value).unwrap()
    } else {
        "<redacted>"
    }
}
