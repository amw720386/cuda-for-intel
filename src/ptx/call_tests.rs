//! `.func` calls beyond the nvcc `.param` ABI (roadmap item 8): `.reg` parameters and return
//! values, function-scope statics, function pointers and indirect calls (a switch over the
//! address-taken functions), and the precise errors for calls that cannot be inlined.
use super::sim::{Gpu, Launch, Sim};
use super::translate;

const HEADER: &str = ".version 8.8\n.target sm_80\n.address_size 64\n";

/// Runs the module with one `out` buffer of `threads` words and returns the words.
fn run(src: &str, threads: u32) -> Vec<u32> {
    let t = translate(src).unwrap_or_else(|e| panic!("{e}\n{src}"));
    let module = rspirv::dr::load_words(&t.words).unwrap();
    let mut gpu = Gpu::default();
    let out = gpu.alloc(4 * threads as usize);
    Sim::new(&module).run(&mut gpu, &Launch::new([1, 1, 1], [threads, 1, 1], vec![out])).unwrap();
    assert!(gpu.violations.borrow().is_empty(), "{:?}", gpu.violations.borrow());
    (0..u64::from(threads)).map(|i| gpu.mem.read(out + 4 * i, 4) as u32).collect()
}

/// An entry that computes `%r3` from `%r1 = %tid.x` with `body` and stores it at `out[tid]`.
fn entry(body: &str) -> String {
    format!(
        ".visible .entry k(.param .u64 pout)\n{{\n\
         .reg .pred %p<4>;\n.reg .b32 %r<16>;\n.reg .b64 %rd<16>;\n\
         ld.param.u64 %rd1, [pout];\ncvta.to.global.u64 %rd1, %rd1;\nmov.u32 %r1, %tid.x;\n\
         {body}\n\
         mul.wide.u32 %rd14, %r1, 4;\nadd.s64 %rd15, %rd1, %rd14;\nst.global.u32 [%rd15], %r3;\nret;\n}}\n"
    )
}

/// nvcc-style `.param` function `name(x) = x <op> k`.
fn param_func(name: &str, op: &str, k: u32) -> String {
    format!(
        ".func (.param .b32 func_retval0) {name}(.param .b32 {name}_param_0)\n{{\n\
         .reg .b32 %r<3>;\nld.param.u32 %r1, [{name}_param_0];\n{op} %r2, %r1, {k};\n\
         st.param.b32 [func_retval0+0], %r2;\nret;\n}}\n"
    )
}

/// `call (retval0), %rd5, (param0), <label>;` with the argument `%r1`; the result lands in `%r3`.
fn indirect_call(guard: &str, label_decl: &str, label: &str) -> String {
    format!(
        "{{\n.param .b32 param0;\nst.param.b32 [param0+0], %r1;\n.param .b32 retval0;\n{label_decl}\n\
         {guard} call (retval0), %rd5, (param0), {label};\nld.param.b32 %r3, [retval0+0];\n}}"
    )
}

/// Selects `f0`, `f1` or `f2` by `tid % 3` into `%rd5`.
const SELECT_POINTER: &str = "rem.u32 %r2, %r1, 3;\n\
    mov.u64 %rd2, f0;\nmov.u64 %rd3, f1;\nmov.u64 %rd4, f2;\n\
    setp.eq.u32 %p1, %r2, 1;\nselp.b64 %rd5, %rd3, %rd2, %p1;\n\
    setp.eq.u32 %p2, %r2, 2;\nselp.b64 %rd5, %rd4, %rd5, %p2;";

fn three_funcs() -> String {
    [param_func("f0", "add.s32", 1), param_func("f1", "mul.lo.s32", 2), param_func("f2", "sub.s32", 3)].concat()
}

#[test]
fn indirect_call_switches_over_address_taken_functions() {
    let call = indirect_call("", "prototype_0 : .callprototype (.param .b32 _) _ (.param .b32 _);", "prototype_0");
    let src = format!("{HEADER}{}{}", three_funcs(), entry(&format!("{SELECT_POINTER}\n{call}")));
    let got = run(&src, 9);
    let want: Vec<u32> = (0..9u32).map(|t| [t + 1, t * 2, t.wrapping_sub(3)][(t % 3) as usize]).collect();
    assert_eq!(got, want);
}

#[test]
fn indirect_call_honours_calltargets_and_guards() {
    // Only f0 and f2 are listed; f1's address is taken but it is not a candidate. The guarded
    // call is skipped on odd threads, which keep %r3 = 100.
    let select = "rem.u32 %r2, %r1, 2;\nmov.u64 %rd2, f0;\nmov.u64 %rd3, f1;\nmov.u64 %rd4, f2;\n\
                  setp.eq.u32 %p1, %r2, 0;\nsetp.lt.u32 %p2, %r1, 4;\nselp.b64 %rd5, %rd2, %rd4, %p2;\nmov.u32 %r3, 100;";
    let call = indirect_call("@%p1", "targets_0 : .calltargets f0, f2;", "targets_0");
    let src = format!("{HEADER}{}{}", three_funcs(), entry(&format!("{select}\n{call}")));
    let got = run(&src, 8);
    let want: Vec<u32> =
        (0..8u32).map(|t| if t % 2 == 1 { 100 } else if t < 4 { t + 1 } else { t.wrapping_sub(3) }).collect();
    assert_eq!(got, want);
}

#[test]
fn function_pointer_tokens_are_distinct_and_non_null() {
    let src = format!(
        "{HEADER}{}{}",
        three_funcs(),
        entry("mov.u64 %rd2, f0;\nmov.u64 %rd3, f1;\nmov.u64 %rd4, f2;\n\
               setp.eq.u32 %p1, %r1, 1;\nselp.b64 %rd5, %rd3, %rd2, %p1;\n\
               setp.eq.u32 %p2, %r1, 2;\nselp.b64 %rd5, %rd4, %rd5, %p2;\ncvt.u32.u64 %r3, %rd5;")
    );
    let got = run(&src, 3);
    assert!(got.iter().all(|&t| t != 0), "{got:?}");
    assert!(got[0] != got[1] && got[1] != got[2] && got[0] != got[2], "{got:?}");
}

#[test]
fn reg_parameters_and_predicate_returns() {
    // ZLUDA-style `.reg` ABI: two results, one a predicate, and an immediate argument.
    let src = format!(
        "{HEADER}.func (.reg .u32 %a, .reg .pred %b) f(.reg .u32 %x, .reg .u32 %y)\n{{\n\
         add.u32 %a, %x, %y;\nsetp.gt.u32 %b, %x, 2;\nret;\n}}\n{}",
        entry("call (%r2, %p3), f, (%r1, 10);\nselp.b32 %r3, %r2, 7, %p3;")
    );
    assert_eq!(run(&src, 5), vec![7, 7, 7, 13, 14]);
}

#[test]
fn function_scope_shared_is_one_variable() {
    // Each call site of `bump` adds to the same `.shared` word, as PTX's static storage does.
    let src = format!(
        "{HEADER}.func (.reg .u32 %o) bump(.reg .u32 %v)\n{{\n\
         .shared .align 4 .u32 acc;\n.reg .u32 %t;\n\
         ld.shared.u32 %t, [acc];\nadd.u32 %t, %t, %v;\nst.shared.u32 [acc], %t;\nmov.u32 %o, %t;\nret;\n}}\n{}",
        entry("mov.u32 %r5, 0;\nst.shared.u32 [bump__acc], %r5;\ncall (%r2), bump, (5);\ncall (%r3), bump, (%r2);")
    );
    // 5, then 5 + 5.
    assert_eq!(run(&src, 1), vec![10]);
}

fn rejection(src: &str) -> String {
    translate(src).expect_err("expected a rejection").to_string()
}

#[test]
fn calls_that_cannot_be_inlined_fail_precisely() {
    let extern_call = format!(
        "{HEADER}.extern .func (.param .b32 func_retval0) ext(.param .b32 p);\n{}",
        entry("{\n.param .b32 param0;\nst.param.b32 [param0+0], %r1;\n.param .b32 retval0;\n\
               call (retval0), ext, (param0);\nld.param.b32 %r3, [retval0+0];\n}")
    );
    let err = rejection(&extern_call);
    assert!(err.contains("unsupported PTX call: ext is an .extern .func with no definition"), "{err}");

    // An indirect call whose pointer did not come from any function of this module.
    let call = indirect_call("", "prototype_0 : .callprototype (.param .b32 _) _ (.param .b32 _);", "prototype_0");
    let opaque = format!("{HEADER}{}{}", param_func("f0", "add.s32", 1), entry(&format!("ld.global.u64 %rd5, [%rd1];\n{call}")));
    let err = rejection(&opaque);
    assert!(err.contains("no address-taken .func in this module matches the prototype of the indirect call through %rd5"), "{err}");

    // The prototype excludes the only address-taken function (two parameters instead of one).
    let call = indirect_call("", "prototype_0 : .callprototype (.param .b32 _) _ (.param .b32 _, .param .b32 _);", "prototype_0");
    let mismatch =
        format!("{HEADER}{}{}", param_func("f0", "add.s32", 1), entry(&format!("mov.u64 %rd5, f0;\n{call}")));
    assert!(rejection(&mismatch).contains("no address-taken .func"));

    let recursive = format!(
        "{HEADER}.func (.reg .u32 %o) r(.reg .u32 %v)\n{{\ncall (%o), r, (%v);\nret;\n}}\n{}",
        entry("call (%r3), r, (%r1);")
    );
    let err = rejection(&recursive);
    assert!(err.contains("unsupported PTX call: r is recursive"), "{err}");

    let undefined = format!("{HEADER}{}", entry("call (%r3), nowhere, (%r1);"));
    assert!(rejection(&undefined).contains("nowhere has no .func definition"));
}

/// Calls inside an inlined callee are resolved through the same tokens.
#[test]
fn indirect_call_inside_a_callee() {
    let call = indirect_call("", "prototype_1 : .callprototype (.param .b32 _) _ (.param .b32 _);", "prototype_1")
        .replace("%rd5", "%rd9")
        .replace("%r1", "%r7")
        .replace("%r3", "%r8");
    let apply = format!(
        ".func (.reg .u32 %res) apply(.reg .b64 %rd9, .reg .u32 %r7)\n{{\n.reg .b32 %r8;\n{call}\nmov.u32 %res, %r8;\nret;\n}}\n"
    );
    let src = format!(
        "{HEADER}{}{apply}{}",
        three_funcs(),
        entry(&format!("{SELECT_POINTER}\ncall (%r3), apply, (%rd5, %r1);"))
    );
    let got = run(&src, 6);
    let want: Vec<u32> = (0..6u32).map(|t| [t + 1, t * 2, t.wrapping_sub(3)][(t % 3) as usize]).collect();
    assert_eq!(got, want);
}

/// A device function table in a module-scope initializer (`__device__ fn_t table[] = {...}`).
#[test]
fn function_table_in_global_initializer() {
    let call = indirect_call("", "prototype_0 : .callprototype (.param .b32 _) _ (.param .b32 _);", "prototype_0");
    let src = format!(
        "{HEADER}{}.global .align 8 .u64 table[3] = {{f0, f1, f2}};\n{}",
        three_funcs(),
        entry(&format!(
            "rem.u32 %r2, %r1, 3;\nmov.u64 %rd2, table;\nmul.wide.u32 %rd3, %r2, 8;\nadd.s64 %rd4, %rd2, %rd3;\n\
             ld.global.u64 %rd5, [%rd4];\n{call}"
        ))
    );
    let got = run(&src, 6);
    let want: Vec<u32> = (0..6u32).map(|t| [t + 1, t * 2, t.wrapping_sub(3)][(t % 3) as usize]).collect();
    assert_eq!(got, want);
}
