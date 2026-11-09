// Compiles the vendored Microsoft Detours (x64 Windows only). Other targets build an empty crate.
// CFI_DETOURS_SKIP_BUILD=1 skips the C++ compile, so `cargo check --target x86_64-pc-windows-msvc`
// works from a non-Windows host without MSVC (the result type-checks but cannot link).
const FILES: [&str; 5] = ["creatwth.cpp", "detours.cpp", "disasm.cpp", "image.cpp", "modules.cpp"];

fn main() {
    println!("cargo:rerun-if-changed=vendor/src");
    println!("cargo:rerun-if-env-changed=CFI_DETOURS_SKIP_BUILD");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    if std::env::var_os("CFI_DETOURS_SKIP_BUILD").is_some() {
        println!("cargo:warning=CFI_DETOURS_SKIP_BUILD set: Detours not compiled (check only)");
        return;
    }
    let mut build = cc::Build::new();
    build.include("vendor/src").cpp(true).warnings(false);
    for f in FILES {
        build.file(format!("vendor/src/{f}"));
    }
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() != Ok("msvc") {
        build.flag("-fms-extensions");
    }
    build.compile("detours");
}
