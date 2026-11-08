//! Bring-up tooling: per-kernel translate status for large ATen modules.
use super::{parse, spirv, Error};
use std::cell::RefCell;

thread_local! {
    static SKIPPED: RefCell<Vec<(String, String)>> = const { RefCell::new(Vec::new()) };
}

pub(super) fn note_skip(name: &str, error: &Error) {
    SKIPPED.with(|s| s.borrow_mut().push((name.to_string(), error.to_string())));
}

pub(super) fn skipped() -> Vec<(String, String)> {
    SKIPPED.with(|s| s.borrow().clone())
}

pub(super) fn probe_module(prepared: &str) -> Result<Vec<(String, Result<(), String>)>, Error> {
    SKIPPED.with(|s| s.borrow_mut().clear());
    let (entries, vars) = parse::parse_module(prepared)?;
    let mut out = Vec::new();
    for entry in entries {
        let name = entry.name.to_string();
        let device = vars.device.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| spirv::emit(vec![entry], device)))
            .map_err(|_| "panic".to_string())
            .and_then(|r| {
                r.map(|t| {
                    // PROBE_SPV=<dir>: keep each translated kernel for `spirv-val`.
                    if let Some(dir) = std::env::var_os("PROBE_SPV") {
                        let bytes: Vec<u8> = t.words.iter().flat_map(|w| w.to_le_bytes()).collect();
                        let _ = std::fs::write(std::path::Path::new(&dir).join(format!("k{}.spv", out.len())), bytes);
                    }
                })
                .map_err(|e| e.to_string())
            });
        out.push((name, result));
    }
    Ok(out)
}

/// One entry's outcome for the perf harness: FNV-1a hash of the SPIR-V words, or the error.
pub(super) type TimedEntry = (String, Result<u64, String>, std::time::Duration);

fn hash_words(words: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for w in words {
        for b in w.to_le_bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// Translate every entry of a prepared module (optionally on `threads` threads), hashing the
/// SPIR-V. Bring-up and regression tooling only.
pub(super) fn probe_module_hashed(prepared: &str, threads: usize) -> Result<Vec<TimedEntry>, Error> {
    let (entries, vars) = parse::parse_module(prepared)?;
    let n = entries.len();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let slots: Vec<std::sync::Mutex<Option<TimedEntry>>> = (0..n).map(|_| std::sync::Mutex::new(None)).collect();
    let entries: Vec<_> = entries.into_iter().map(std::sync::Mutex::new).collect();
    let lenient = super::lenient_undefined();
    std::thread::scope(|scope| {
        for _ in 0..threads.max(1) {
            scope.spawn(|| {
                super::set_lenient_undefined(lenient);
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    let entry = entries[i].lock().unwrap().clone();
                    let name = entry.name.to_string();
                    let start = std::time::Instant::now();
                    let device = vars.device.clone();
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| spirv::emit(vec![entry], device)))
                        .map_err(|_| "panic".to_string())
                        .and_then(|r| r.map(|t| hash_words(&t.words)).map_err(|e| e.to_string()));
                    *slots[i].lock().unwrap() = Some((name, result, start.elapsed()));
                }
            });
        }
    });
    Ok(slots.into_iter().map(|s| s.into_inner().unwrap().unwrap()).collect())
}

/// Like [`probe_module_hashed`], but through the lazy per-entry path the runtime uses
/// (`translate_entry` on a prepared library), so parse-once costs are visible.
pub(super) fn probe_lazy_hashed(prepared: &str, threads: usize) -> Result<Vec<TimedEntry>, Error> {
    let names: Vec<String> = super::entry_names(prepared)?.into_iter().map(str::to_string).collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let slots: Vec<std::sync::Mutex<Option<TimedEntry>>> = (0..names.len()).map(|_| std::sync::Mutex::new(None)).collect();
    let lenient = super::lenient_undefined();
    std::thread::scope(|scope| {
        for _ in 0..threads.max(1) {
            scope.spawn(|| {
                super::set_lenient_undefined(lenient);
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= names.len() {
                        break;
                    }
                    let start = std::time::Instant::now();
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| super::translate_entry(prepared, &names[i])))
                        .map_err(|_| "panic".to_string())
                        .and_then(|r| r.map(|t| hash_words(&t.words)).map_err(|e| e.to_string()));
                    *slots[i].lock().unwrap() = Some((names[i].clone(), result, start.elapsed()));
                }
            });
        }
    });
    Ok(slots.into_iter().map(|s| s.into_inner().unwrap().unwrap()).collect())
}
