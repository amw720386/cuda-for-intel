//! A clang without NVPTX must produce an actionable compile failure, not a panic. Own test binary
//! because the clang lookup is cached per process.
#![cfg(unix)]
use nvrtc64_120_0::compiler::{self, Failure, Request};
use std::os::unix::fs::PermissionsExt;

#[test]
fn clang_without_nvptx_gives_actionable_log() {
    let dir = std::env::temp_dir().join(format!("nvrtc_fake_clang_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let fake = dir.join("clang");
    std::fs::write(&fake, "#!/bin/sh\necho '    x86-64 - 64-bit X86'\n").unwrap();
    std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
    // SAFETY: single test in this binary, set before any shim call.
    unsafe { std::env::set_var("NVRTC_SHIM_CLANG", &fake) };
    let req = Request { source: "extern \"C\" __global__ void k(){}".into(), name: "a.cu".into(), ..Default::default() };
    let err = compiler::compile(&req).unwrap_err();
    assert!(matches!(err, Failure::NoToolchain(_)));
    assert!(err.log().contains("WITHOUT the NVPTX backend") && err.log().contains("NVRTC_SHIM_CLANG_DIR"), "{}", err.log());
    let _ = std::fs::remove_dir_all(&dir);
}
