//! Standalone probe for ZLUDA f4df215edbc3b6698a1c61ef7a8688fdff3872be.
//! Build as main.rs with a path dependency on that checkout's ptx_parser.
use ptx_parser::{Directive, parse_module_checked};
use std::time::Instant;

fn main() {
    for path in std::env::args_os().skip(1) {
        let source = std::fs::read_to_string(&path).expect("read PTX");
        let parsed = parse_module_checked(&source).expect("parse PTX");
        let mut kernels = 0;
        let mut globals = 0;
        for directive in &parsed.directives {
            match directive {
                Directive::Method(_, method) if method.func_directive.name.is_kernel() => {
                    kernels += 1;
                    if kernels > 2 {
                        continue;
                    }
                    let mut end = 0usize;
                    let fields: Vec<_> = method
                        .func_directive
                        .input_arguments
                        .iter()
                        .map(|parameter| {
                            let layout = parameter.info.v_type.layout().expect("argument layout");
                            let align = parameter.info.align.map_or(layout.align(), |n| n as usize);
                            let offset = end.next_multiple_of(align);
                            end = offset + layout.size();
                            (offset, layout.size(), align)
                        })
                        .collect();
                    println!("{} fields={fields:?} bytes={end}", method.func_directive.name());
                }
                Directive::Variable(..) => globals += 1,
                _ => {}
            }
        }
        let mut times = Vec::new();
        for i in 0..36 {
            let start = Instant::now();
            std::hint::black_box(parse_module_checked(&source).expect("parse PTX"));
            if i >= 5 {
                times.push(start.elapsed());
            }
        }
        times.sort_unstable();
        println!(
            "{path:?}: PTX {:?}, {} kernel entries, {globals} globals, parse median={:.2} us",
            parsed.ptx_version,
            kernels,
            times[15].as_secs_f64() * 1e6,
        );
    }
}
