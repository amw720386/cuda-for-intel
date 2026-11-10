//! CUDA VMM on hardware: the GGML VMM-pool pattern (one reservation, physical chunks mapped
//! back to back, per-chunk cuMemSetAccess), a kernel writing across the chunk boundary,
//! copies across it, a second mapping of the same physical pages (aliasing), teardown, and
//! the cost of growing the pool. Prints one `VMM:` summary line per run for the log.
use super::{Handle, library, ok, symbols, test_device};
use std::ffi::{c_char, c_int, c_void};
use std::ptr::{from_mut, null_mut};
use std::time::Instant;

const PTX: &std::ffi::CStr = c"
.version 7.0
.target sm_75
.address_size 64
.entry stamp(.param .u64 .ptr .global out, .param .u32 n)
{
    .reg .pred %p;
    .reg .b32 %r<6>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [out];
    ld.param.u32 %r1, [n];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r2, %r2, %r3, %r4;
    setp.ge.u32 %p, %r2, %r1;
    @%p ret;
    xor.b32 %r5, %r2, 0xA5A5A5A5;
    mul.wide.u32 %rd2, %r2, 4;
    add.u64 %rd3, %rd1, %rd2;
    st.global.u32 [%rd3], %r5;
    ret;
}
";

#[repr(C)]
#[derive(Clone, Copy)]
struct Location {
    kind: c_int,
    id: c_int,
}

#[repr(C)]
struct Prop {
    kind: c_int,
    handle_types: c_int,
    location: Location,
    win32: *mut c_void,
    flags: [u8; 8],
}

#[repr(C)]
struct Access {
    location: Location,
    flags: c_int,
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn vmm_pool_growth_kernel_copies_and_aliasing() {
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuDeviceGetName(*mut c_char, i32, i32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuCtxDestroy_v2(Handle),
        cuCtxSynchronize(),
        cuModuleLoadData(*mut Handle, *const c_void),
        cuModuleUnload(Handle),
        cuModuleGetFunction(*mut Handle, Handle, *const c_char),
        cuLaunchKernel(Handle, u32, u32, u32, u32, u32, u32, u32, Handle, *mut Handle, *mut Handle),
        cuMemcpyHtoD_v2(u64, *const c_void, usize),
        cuMemcpyDtoH_v2(*mut c_void, u64, usize),
        cuMemcpyDtoD_v2(u64, u64, usize),
        cuMemsetD8_v2(u64, u8, usize),
        cuMemFree_v2(u64),
        cuMemGetAllocationGranularity(*mut usize, *const Prop, c_int),
        cuMemCreate(*mut u64, usize, *const Prop, u64),
        cuMemRelease(u64),
        cuMemAddressReserve(*mut u64, usize, usize, u64, u64),
        cuMemAddressFree(u64, usize),
        cuMemMap(u64, usize, usize, u64, u64),
        cuMemUnmap(u64, usize),
        cuMemSetAccess(u64, usize, *const Access, usize),
    );
    ok!(cuInit(0));
    let device = test_device();
    let mut name = [0 as c_char; 256];
    ok!(cuDeviceGetName(name.as_mut_ptr(), 256, device));
    let name = unsafe { std::ffi::CStr::from_ptr(name.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    let mut context = null_mut();
    ok!(cuCtxCreate_v2(&mut context, 0, device));

    let prop = Prop {
        kind: 1,
        handle_types: 0,
        location: Location {
            kind: 1,
            id: device,
        },
        win32: null_mut(),
        flags: [0; 8],
    };
    let access = Access {
        location: prop.location,
        flags: 3,
    };
    let mut granularity = 0usize;
    let status = unsafe { cuMemGetAllocationGranularity(&mut granularity, &prop, 0) };
    if status == 801 {
        println!("VMM: device={name} NOT_SUPPORTED (loader lacks zeVirtualMem entry points)");
        ok!(cuCtxDestroy_v2(context));
        return;
    }
    assert_eq!(status, 0, "cuMemGetAllocationGranularity");
    assert!(granularity >= 2 << 20 && granularity.is_power_of_two());

    // GGML reserves a large range once and maps chunks as the pool grows.
    let chunk = 2 * granularity;
    let reserve = 64 * granularity;
    let mut base = 0u64;
    ok!(cuMemAddressReserve(&mut base, reserve, 0, 0, 0));
    let mut handles = [0u64; 3];
    let started = Instant::now();
    for (i, handle) in handles.iter_mut().enumerate() {
        ok!(cuMemCreate(handle, chunk, &prop, 0));
        let at = base + (i * chunk) as u64;
        ok!(cuMemMap(at, chunk, 0, *handle, 0));
        ok!(cuMemSetAccess(at, chunk, &access, 1));
    }
    let grow_us = started.elapsed().as_secs_f64() * 1e6 / handles.len() as f64;
    let total = handles.len() * chunk;

    // A kernel writes every word of all three chunks.
    let mut module = null_mut();
    ok!(cuModuleLoadData(&mut module, PTX.as_ptr().cast()));
    let mut kernel = null_mut();
    ok!(cuModuleGetFunction(&mut kernel, module, c"stamp".as_ptr()));
    let words = (total / 4) as u32;
    let (mut out, mut n) = (base, words);
    let mut params = [from_mut(&mut out).cast::<c_void>(), from_mut(&mut n).cast()];
    let started = Instant::now();
    ok!(cuLaunchKernel(
        kernel,
        words.div_ceil(256),
        1,
        1,
        256,
        1,
        1,
        0,
        null_mut(),
        params.as_mut_ptr(),
        null_mut()
    ));
    ok!(cuCtxSynchronize());
    let kernel_us = started.elapsed().as_secs_f64() * 1e6;
    let mut host = vec![0u32; total / 4];
    ok!(cuMemcpyDtoH_v2(host.as_mut_ptr().cast(), base, total));
    let bad = host
        .iter()
        .enumerate()
        .filter(|&(i, &v)| v != i as u32 ^ 0xA5A5_A5A5)
        .count();
    assert_eq!(bad, 0, "kernel output across VMM chunks");

    // A device copy that straddles the first chunk boundary, then a host copy across it.
    let straddle = (chunk - 4096) as u64;
    ok!(cuMemcpyDtoD_v2(
        base + straddle,
        base + 2 * chunk as u64,
        8192
    ));
    let mut window = vec![0u32; 2048];
    ok!(cuMemcpyDtoH_v2(
        window.as_mut_ptr().cast(),
        base + straddle,
        8192
    ));
    let first = 2 * chunk / 4;
    assert!(
        window
            .iter()
            .enumerate()
            .all(|(i, &v)| v == (first + i) as u32 ^ 0xA5A5_A5A5)
    );
    let pattern: Vec<u8> = (0..8192u32).map(|i| (i * 7) as u8).collect();
    ok!(cuMemcpyHtoD_v2(
        base + straddle,
        pattern.as_ptr().cast(),
        pattern.len()
    ));
    let mut back = vec![0u8; 8192];
    ok!(cuMemcpyDtoH_v2(
        back.as_mut_ptr().cast(),
        base + straddle,
        back.len()
    ));
    assert_eq!(back, pattern);

    // The same physical pages mapped a second time read what the first mapping wrote.
    let mut alias_base = 0u64;
    ok!(cuMemAddressReserve(&mut alias_base, chunk, 0, 0, 0));
    ok!(cuMemMap(alias_base, chunk, 0, handles[1], 0));
    ok!(cuMemSetAccess(alias_base, chunk, &access, 1));
    ok!(cuMemsetD8_v2(base + chunk as u64, 0x5c, 4096));
    let mut seen = vec![0u8; 4096];
    ok!(cuMemcpyDtoH_v2(seen.as_mut_ptr().cast(), alias_base, 4096));
    let aliased = seen.iter().all(|&b| b == 0x5c);
    assert!(aliased, "second mapping does not alias the physical memory");

    assert_eq!(
        unsafe { cuMemFree_v2(base) },
        1,
        "VMM ranges are not cuMemAlloc memory"
    );
    ok!(cuMemUnmap(alias_base, chunk));
    ok!(cuMemAddressFree(alias_base, chunk));
    let started = Instant::now();
    ok!(cuMemUnmap(base, total));
    let unmap_us = started.elapsed().as_secs_f64() * 1e6;
    for handle in handles {
        ok!(cuMemRelease(handle));
    }
    ok!(cuMemAddressFree(base, reserve));
    ok!(cuModuleUnload(module));
    ok!(cuCtxDestroy_v2(context));
    println!(
        "VMM: device={name} granularity={granularity} chunk={chunk} chunks=3 \
         grow_per_chunk_us={grow_us:.0} kernel_us={kernel_us:.0} unmap_all_us={unmap_us:.0} \
         kernel=PASS copies=PASS alias=PASS"
    );
}
