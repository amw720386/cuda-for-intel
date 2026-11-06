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

pub(super) fn call(api: &'static str, action: impl FnOnce() -> CuResult) -> CuResult {
    let result = action();
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
