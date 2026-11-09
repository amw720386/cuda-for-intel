//! Names shared by the redirect DLL and the `cfi-run` launcher.

/// Payload GUID carrying the install directory (UTF-16, no terminator) into a child process.
/// {5c1f0e9a-3d2b-4c71-9a0e-cf1d0a7e2b41}
pub const PAYLOAD_GUID: (u32, u16, u16, [u8; 8]) =
    (0x5c1f_0e9a, 0x3d2b, 0x4c71, [0x9a, 0x0e, 0xcf, 0x1d, 0x0a, 0x7e, 0x2b, 0x41]);

/// Directory holding our replacement DLLs. Read when no payload is present (manual injection).
pub const ENV_DIR: &str = "CFI_REDIRECT_DIR";
/// `1` prints every redirect decision to stderr.
pub const ENV_LOG: &str = "CFI_REDIRECT_LOG";
/// File name of the redirect DLL.
pub const DLL_NAME: &str = "cfi_redirect.dll";
