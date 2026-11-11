use super::*;
use rspirv::dr::Operand;
use rspirv::spirv::{Decoration, Op, StorageClass};

const MODULE: &str = r#"
.version 7.0
.target sm_75
.address_size 64

.global .align 4 .u32 counter = 7;
.const .align 4 .u32 table[4] = {10, 20, 30, 40};
.visible .global .align 8 .b8 scratch[24];
.extern .global .b8 elsewhere[];
.shared .align 4 .b8 tile[64];

.visible .entry reader(.param .u64 out)
{
    .reg .b32 %r<6>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u64 %rd3, counter;
    ld.global.u32 %r1, [%rd3];
    ld.const.u32 %r2, [table+8];
    add.u32 %r3, %r1, %r2;
    st.global.u32 [scratch+4], %r3;
    mov.u32 %r5, tile;
    st.shared.u32 [%r5+4], %r3;
    bar.sync 0;
    ld.shared.u32 %r4, [%r5+4];
    st.global.u32 [%rd2], %r4;
    ret;
}
"#;

fn module(words: &[u32]) -> rspirv::dr::Module {
    rspirv::dr::load_words(words).unwrap()
}

#[test]
fn exports_global_and_const_variables_with_initializers() {
    let translated = translate(MODULE).unwrap();
    let names: Vec<_> = translated.globals.iter().map(|g| (g.name.as_str(), g.size, g.constant)).collect();
    assert_eq!(names, [("counter", 4, false), ("table", 16, true), ("scratch", 24, false)]);
    let module = module(&translated.words);
    let variables: Vec<_> = module
        .types_global_values
        .iter()
        .filter(|inst| inst.class.opcode == Op::Variable && inst.operands.first() == Some(&Operand::StorageClass(StorageClass::CrossWorkgroup)))
        .collect();
    assert_eq!(variables.len(), 3);
    for variable in &variables {
        assert!(variable.operands.contains(&Operand::StorageClass(StorageClass::CrossWorkgroup)));
        // Every variable carries an initializer operand after the storage class.
        assert_eq!(variable.operands.len(), 2);
    }
    let exported: Vec<String> = module
        .annotations
        .iter()
        .filter(|inst| {
            inst.class.opcode == Op::Decorate
                && inst.operands.contains(&Operand::Decoration(Decoration::LinkageAttributes))
        })
        .filter_map(|inst| match &inst.operands[2] {
            Operand::LiteralString(name) => Some(name.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(exported, ["counter", "table", "scratch"]);
    let aligns: Vec<u32> = module
        .annotations
        .iter()
        .filter(|inst| inst.operands.contains(&Operand::Decoration(Decoration::Alignment)))
        .filter_map(|inst| match inst.operands[2] {
            Operand::LiteralBit32(value) => Some(value),
            _ => None,
        })
        .collect();
    assert_eq!(aligns, [4, 4, 8]);
    assert!(module.capabilities.iter().any(|c| c.operands[0] == Operand::Capability(rspirv::spirv::Capability::Linkage)));
    // The shared tile is accounted to the kernel that uses it.
    assert_eq!(translated.kernels[0].shared_bytes, 64);
}

#[test]
fn initializer_words_are_embedded() {
    let translated = translate(MODULE).unwrap();
    let module = module(&translated.words);
    let constant = |id| {
        module.types_global_values.iter().find(|inst| inst.result_id == Some(id))
    };
    let table = module
        .types_global_values
        .iter()
        .filter(|inst| inst.class.opcode == Op::Variable && inst.operands.first() == Some(&Operand::StorageClass(StorageClass::CrossWorkgroup)))
        .nth(1)
        .unwrap();
    let Operand::IdRef(initializer) = table.operands[1] else { panic!() };
    let composite = constant(initializer).unwrap();
    assert_eq!(composite.class.opcode, Op::ConstantComposite);
    let values: Vec<u32> = composite
        .operands
        .iter()
        .map(|operand| {
            let Operand::IdRef(id) = operand else { panic!() };
            match constant(*id).unwrap().operands[0] {
                Operand::LiteralBit32(value) => value,
                _ => panic!(),
            }
        })
        .collect();
    assert_eq!(values, [10, 20, 30, 40]);
    // The scalar zero-initialized variable uses OpConstantNull.
    let scratch = module
        .types_global_values
        .iter()
        .filter(|inst| inst.class.opcode == Op::Variable && inst.operands.first() == Some(&Operand::StorageClass(StorageClass::CrossWorkgroup)))
        .nth(2)
        .unwrap();
    let Operand::IdRef(initializer) = scratch.operands[1] else { panic!() };
    assert_eq!(constant(initializer).unwrap().class.opcode, Op::ConstantNull);
}

#[test]
fn unreferenced_shared_and_unknown_symbols() {
    // A kernel that never names the module `.shared` array does not reserve it.
    let source = MODULE.replace(
        "mov.u32 %r5, tile;
    st.shared.u32 [%r5+4], %r3;\n    bar.sync 0;\n    ld.shared.u32 %r4, [%r5+4];",
        "mov.u32 %r4, %r3;",
    );
    assert_eq!(translate(&source).unwrap().kernels[0].shared_bytes, 0);
    // An address of a symbol that was never declared is rejected, not mistaken for shared.
    let missing = MODULE.replace("mov.u64 %rd3, counter;", "mov.u64 %rd3, not_declared;");
    assert!(translate(&missing).is_err());
}

#[test]
fn modules_without_variables_emit_none() {
    let translated = translate(include_str!("../../examples/vector_add.ptx")).unwrap();
    assert!(translated.globals.is_empty());
    let module = module(&translated.words);
    assert!(module.annotations.iter().all(|inst| !inst.operands.contains(&Operand::Decoration(Decoration::LinkageAttributes))));
}

#[test]
#[ignore = "requires the Khronos spirv-val executable"]
fn globals_module_validates_as_opencl_spirv() {
    // Universal SPIR-V 1.3 rules: see `coverage_tests::validate_named` for why not `opencl2.2`.
    let translated = translate(MODULE).unwrap();
    assert!(super::coverage_tests::validate_named(&translated.words, "the globals module"), "spirv-val is not installed");
}

#[test]
fn direct_symbol_stores_and_hardware_probe_module_translate() {
    let source = r#"
.version 7.0
.target sm_75
.address_size 64
.global .align 4 .u32 counter = 7;
.const .align 4 .u32 table[4] = {10, 20, 30, 40};
.global .align 4 .u32 result;
.entry combine(.param .u64 .ptr .global out)
{
    .reg .b32 %r<6>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u64 %rd3, counter;
    ld.global.u32 %r1, [%rd3];
    ld.const.u32 %r2, [table+8];
    add.u32 %r3, %r1, %r2;
    st.global.u32 [result], %r3;
    st.global.u32 [%rd2], %r3;
    ret;
}
"#;
    let translated = translate(source).unwrap();
    assert_eq!(translated.globals.len(), 3);
    assert_eq!(translated.kernels[0].name, "combine");
}
