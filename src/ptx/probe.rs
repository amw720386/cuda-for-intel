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
            .and_then(|r| r.map(|_| ()).map_err(|e| e.to_string()));
        out.push((name, result));
    }
    Ok(out)
}
