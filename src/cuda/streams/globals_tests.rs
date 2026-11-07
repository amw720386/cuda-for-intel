//! Module-scope variables on the fake Level Zero backend.
use super::fake_tests::{Fixture, assert_baseline, context};
use crate::cuda::globals::{cuLibraryGetGlobal, cuModuleGetGlobal_v2, cuModuleGetSurfRef};
use crate::cuda::library::{cuLibraryLoadData, cuLibraryUnload};
use crate::cuda::*;
use crate::level_zero::fake;
use std::ffi::CStr;
use std::ptr::{null, null_mut};

const PTX: &CStr = c"
.version 7.0
.target sm_75
.address_size 64
.global .align 4 .u32 counter = 7;
.const .align 4 .u32 table[4] = {10, 20, 30, 40};
.global .align 8 .b8 scratch[24];
.global .u32 missing_export;
.entry k(.param .u64 .ptr .global out)
{
    .reg .b32 %r<4>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u64 %rd3, counter;
    ld.global.u32 %r1, [%rd3];
    ld.const.u32 %r2, [table+4];
    add.u32 %r3, %r1, %r2;
    st.global.u32 [%rd2], %r3;
    ret;
}
";

fn load(ptx: &CStr) -> Opaque {
    let mut module = null_mut();
    assert_eq!(unsafe { cuModuleLoadData(&mut module, ptx.as_ptr().cast()) }, Success);
    module
}

fn global(module: Opaque, name: &CStr) -> Result<(u64, usize), CuResult> {
    let (mut address, mut bytes) = (0u64, 0usize);
    match unsafe { cuModuleGetGlobal_v2(&mut address, &mut bytes, module, name.as_ptr()) } {
        Success => Ok((address, bytes)),
        error => Err(error),
    }
}

#[test]
fn module_globals_resolve_report_size_and_accept_memcpy() {
    let _fixture = Fixture::new();
    let ctx = context(None);
    let module = load(PTX);

    let (counter, bytes) = global(module, c"counter").unwrap();
    assert_eq!((bytes, counter != 0), (4, true));
    // The address is stable and distinct per variable.
    assert_eq!(global(module, c"counter").unwrap(), (counter, 4));
    let (table, bytes) = global(module, c"table").unwrap();
    assert_eq!(bytes, 16);
    let (scratch, bytes) = global(module, c"scratch").unwrap();
    assert_eq!(bytes, 24);
    assert!(table != counter && scratch != table);

    // Either output may be omitted, like the CUDA driver.
    let mut only_size = 0usize;
    assert_eq!(
        unsafe { cuModuleGetGlobal_v2(null_mut(), &mut only_size, module, c"table".as_ptr()) },
        Success
    );
    assert_eq!(only_size, 16);
    let mut only_address = 0u64;
    assert_eq!(
        unsafe { cuModuleGetGlobal_v2(&mut only_address, null_mut(), module, c"table".as_ptr()) },
        Success
    );
    assert_eq!(only_address, table);

    assert_eq!(global(module, c"nope"), Err(NotFound));
    assert_eq!(global(module, c"k"), Err(NotFound), "kernels are not variables");
    assert_eq!(
        unsafe { cuModuleGetGlobal_v2(null_mut(), null_mut(), module, null()) },
        InvalidValue
    );
    assert_eq!(global(std::ptr::without_provenance_mut(0xdead), c"counter"), Err(InvalidHandle));

    // Host copies work on the whole object, including .const data, but not past it.
    let data = [1u32, 2, 3, 4, 5];
    assert_eq!(unsafe { cuMemcpyHtoD_v2(table, data.as_ptr().cast(), 16) }, Success);
    assert_eq!(unsafe { cuMemcpyHtoD_v2(table + 8, data.as_ptr().cast(), 8) }, Success);
    assert_eq!(unsafe { cuMemcpyHtoD_v2(table, data.as_ptr().cast(), 20) }, InvalidValue);
    let mut back = [0u32; 4];
    assert_eq!(unsafe { cuMemcpyDtoH_v2(back.as_mut_ptr().cast(), table, 16) }, Success);

    // A module variable is not an allocation the application may free.
    assert_eq!(cuMemFree_v2(counter), InvalidValue);
    assert_eq!(global(module, c"counter").unwrap().0, counter);

    // Unloading retires the range.
    assert_eq!(cuModuleUnload(module), Success);
    assert_eq!(unsafe { cuMemcpyHtoD_v2(table, data.as_ptr().cast(), 4) }, InvalidValue);
    assert_eq!(global(module, c"counter"), Err(InvalidHandle));
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();
}

#[test]
fn backend_that_does_not_export_the_variable_fails_cleanly() {
    let _fixture = Fixture::new();
    let ctx = context(None);
    let module = load(PTX);
    // Declared by PTX but absent from the Level Zero module.
    assert!(global(module, c"missing_export").is_err());
    // A backend that reports a smaller object than the translator declared is rejected.
    fake::set_global_size(Some(8));
    assert_eq!(global(module, c"table"), Err(InvalidImage));
    assert_eq!(global(module, c"counter").unwrap().1, 4);
    fake::set_global_size(None);
    assert_eq!(cuModuleUnload(module), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert!(fake::snapshot().ledger.violations.iter().all(|v| v.contains("not exported")));
}

#[test]
fn library_globals_and_unsupported_references() {
    let _fixture = Fixture::new();
    let ctx = context(None);
    let mut library = null_mut();
    assert_eq!(
        unsafe {
            cuLibraryLoadData(&mut library, PTX.as_ptr().cast(), null(), null(), 0, null(), null(), 0)
        },
        Success
    );
    let (mut address, mut bytes) = (0u64, 0usize);
    assert_eq!(
        unsafe { cuLibraryGetGlobal(&mut address, &mut bytes, library, c"scratch".as_ptr()) },
        Success
    );
    assert_eq!((address != 0, bytes), (true, 24));
    assert_eq!(unsafe { cuMemcpyHtoD_v2(address, [0u8; 24].as_ptr().cast(), 24) }, Success);
    assert_eq!(
        unsafe { cuLibraryGetGlobal(&mut address, &mut bytes, library, c"zzz".as_ptr()) },
        NotFound
    );
    assert_eq!(
        unsafe { cuLibraryGetGlobal(&mut address, &mut bytes, library, null()) },
        InvalidValue
    );
    let mut out = null_mut();
    assert_eq!(unsafe { cuModuleGetSurfRef(&mut out, null_mut(), c"s".as_ptr()) }, NotSupported);
    assert_eq!(cuLibraryUnload(library), Success);
    assert_eq!(unsafe { cuMemcpyHtoD_v2(address, [0u8; 4].as_ptr().cast(), 4) }, InvalidValue);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();
}
