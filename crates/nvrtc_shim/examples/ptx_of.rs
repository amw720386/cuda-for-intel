//! `cargo run -p nvrtc_shim --example ptx_of -- file.cu [name_expr...]` prints the PTX the shim
//! produces (needs clang with NVPTX; set NVRTC_SHIM_CLANG).
use nvrtc64_120_0::compiler::{self, Request};
fn main() {
    let mut args = std::env::args().skip(1);
    let file = args.next().expect("source file");
    let req = Request {
        source: std::fs::read_to_string(&file).unwrap(),
        name: file,
        name_expressions: args.collect(),
        ..Default::default()
    };
    match compiler::compile(&req) {
        Ok(out) => print!("{}", out.ptx),
        Err(e) => {
            eprintln!("{}", e.log());
            std::process::exit(1);
        }
    }
}
