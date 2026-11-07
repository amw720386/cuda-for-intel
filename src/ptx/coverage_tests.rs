//! Instruction-coverage tests. Every snippet is translated and, when a
//! `spirv-val` executable is available (`SPIRV_VAL` or `PATH`), validated.
use super::*;
use rspirv::binary::Assemble;
use rspirv::dr::Operand;
use rspirv::spirv::{ExecutionMode, Op};

/// Wraps `body` in a kernel with one global output pointer (`%rdo`, already
/// converted to a global address) and the given extra register declarations.
pub(super) fn kernel(regs: &str, body: &str) -> String {
    format!(
        r#".version 9.0
.target sm_75
.address_size 64
.visible .entry probe(.param .u64 out_param, .param .u32 n_param)
{{
    .reg .pred %p<8>;
    .reg .b32 %r<32>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<16>;
    {regs}
    ld.param.u64 %rd0, [out_param];
    cvta.to.global.u64 %rdo, %rd0;
    ld.param.u32 %r0, [n_param];
{body}
    ret;
}}
"#
    )
    .replace("%rdo", "%rd15")
}

pub(super) fn validate(words: &[u32]) {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let validator = std::env::var_os("SPIRV_VAL").unwrap_or_else(|| "spirv-val".into());
    let mut module = rspirv::dr::load_words(words).unwrap();
    // spirv-val does not know the Intel SubgroupSize execution mode.
    module.execution_modes.retain(|inst| {
        !matches!(
            inst.operands.get(1),
            Some(Operand::ExecutionMode(ExecutionMode::SubgroupSize))
        )
    });
    let bytes: Vec<u8> = module
        .assemble()
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    let Ok(mut process) = Command::new(&validator)
        .args(["--target-env", "spv1.3", "-"])
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    else {
        return; // validator not installed
    };
    let mut stdin = process.stdin.take().unwrap();
    let writer = std::thread::spawn(move || stdin.write_all(&bytes));
    let output = process.wait_with_output().unwrap();
    let _ = writer.join();
    assert!(
        output.status.success(),
        "spirv-val rejected the module: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Translates and validates; returns the decoded module.
pub(super) fn lower(source: &str) -> rspirv::dr::Module {
    let translated = translate(source).unwrap_or_else(|e| panic!("{e}\n{source}"));
    validate(&translated.words);
    rspirv::dr::load_words(&translated.words).unwrap()
}

pub(super) fn has_op(module: &rspirv::dr::Module, op: Op) -> bool {
    module.all_inst_iter().any(|inst| inst.class.opcode == op)
}

#[test]
fn flattens_register_scopes_from_inline_asm_helpers() {
    let source = kernel(
        "",
        r#"    mov.u32 %r1, 7;
    {
        .reg .b64 u64addr;
        .reg .b32 t<2>;
        cvt.u64.u32 u64addr, %r1;
        add.s32 t0, %r1, 1;
        st.global.u32 [%rd15], t0;
        st.global.u64 [%rd15+8], u64addr;
    }
    {
        .reg .b64 u64addr;
        cvt.u64.u32 u64addr, %r0;
        st.global.u64 [%rd15+16], u64addr;
    }"#,
    );
    lower(&source);
}

#[test]
fn executes_guarded_instructions_through_conditional_blocks() {
    let source = kernel(
        "",
        r#"    mov.u32 %r1, 0;
    setp.lt.u32 %p1, %r0, 4;
    @%p1 mov.u32 %r1, 5;
    @!%p1 add.s32 %r1, %r1, 9;
    st.global.u32 [%rd15], %r1;
    @%p1 ret;
    st.global.u32 [%rd15+4], %r1;"#,
    );
    let module = lower(&source);
    assert!(has_op(&module, Op::BranchConditional));
}
