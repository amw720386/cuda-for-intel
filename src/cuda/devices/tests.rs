//! Multi-device behaviour on the fake Level Zero backend with 2-4 devices.
use super::peer::{self, *};
use super::*;
use crate::cuda::discovery::cuDeviceGetPCIBusId;
use crate::cuda::gaps::{cuCtxPopCurrent_v2, cuCtxPushCurrent_v2, cuPointerGetAttribute};
use crate::cuda::primary::*;
use crate::cuda::streams::events::{
    cuEventCreate, cuEventDestroy_v2, cuEventRecord, cuEventSynchronize, cuStreamWaitEvent,
};
use crate::cuda::streams::{cuStreamCreate, cuStreamDestroy_v2};
use crate::level_zero::fake::{self, DeviceSpec};
use std::ptr::{null_mut, without_provenance_mut};
use std::sync::MutexGuard;

struct Fixture {
    _serial: MutexGuard<'static, ()>,
    devices: usize,
}

impl Fixture {
    fn new(specs: &[DeviceSpec]) -> Self {
        Self::with(specs, &Selection::default())
    }

    fn with(specs: &[DeviceSpec], selection: &Selection) -> Self {
        let serial = TEST_RUNTIME
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut runtime = fake::start_with(specs);
        arrange(&mut runtime.devices, selection);
        let devices = runtime.devices.len();
        assert!(
            RUNTIME
                .lock()
                .unwrap()
                .replace(State {
                    runtime,
                    next_handle: 3,
                    stream_owners: HashMap::new(),
                    contexts: HandleMap::default(),
                    incomplete_contexts: Vec::new(),
                    primary: HashMap::new(),
                    libraries: library::Libraries::default(),
                    graphs: graphs::Graphs::default(),
                    lazy_modules: HashMap::new(),
                })
                .is_none()
        );
        peer::set_direct_kernel_access(false);
        assert_eq!(cuInit(0), Success);
        Self {
            _serial: serial,
            devices,
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let quiet = std::thread::panicking();
        for ordinal in 0..self.devices as c_int {
            let _ = cuDevicePrimaryCtxReset(ordinal);
        }
        let keys: Vec<usize> = RUNTIME
            .lock()
            .unwrap()
            .as_ref()
            .map(|s| s.contexts.keys().copied().collect())
            .unwrap_or_default();
        for key in keys {
            let _ = cuCtxDestroy_v2(without_provenance_mut(key));
        }
        let state = RUNTIME.lock().unwrap().take().unwrap();
        CURRENT.with_borrow_mut(Vec::clear);
        peer::set_direct_kernel_access(false);
        let violations = fake::snapshot().ledger.violations;
        fake::finish();
        assert!(quiet || state.contexts.is_empty());
        assert!(quiet || violations.is_empty(), "{violations:?}");
    }
}

const GIB: u64 = 1 << 30;

fn spec(index: usize, driver: usize, name: &str) -> DeviceSpec {
    let mut spec = DeviceSpec::new(index, driver);
    spec.name = name.into();
    spec.total_memory = 8 * GIB;
    spec.max_allocation = 1 << 28;
    spec.uuid = [index as u8 + 1; 16];
    spec.pci = Some([0, index as u32 + 1, 0, 0]);
    spec
}

/// iGPU on driver 0, two dGPUs on driver 1 (a linked pair) and one on driver 2 that
/// has fp64, a larger shared memory and a different subgroup size.
fn mixed() -> Vec<DeviceSpec> {
    let mut igpu = spec(0, 0, "iGPU");
    igpu.integrated = true;
    igpu.total_memory = 4 * GIB;
    let mut a = spec(1, 1, "dGPU-A");
    a.total_memory = 16 * GIB;
    a.p2p_group = Some(1);
    let mut b = spec(2, 1, "dGPU-B");
    b.p2p_group = Some(1);
    b.subgroup_sizes = vec![8, 16, 32];
    let mut c = spec(3, 2, "dGPU-C");
    c.total_memory = 16 * GIB;
    c.fp64 = true;
    c.shared_local_memory = 131072;
    c.max_workgroup_size = 1024;
    c.subgroup_sizes = vec![32];
    vec![igpu, a, b, c]
}

fn pair() -> Vec<DeviceSpec> {
    vec![spec(0, 0, "Alpha"), spec(1, 1, "Beta")]
}

fn count() -> c_int {
    let mut n = -1;
    assert_eq!(unsafe { cuDeviceGetCount(&mut n) }, Success);
    n
}

fn name(ordinal: c_int) -> String {
    let mut buffer = [0 as c_char; 64];
    assert_eq!(
        unsafe { cuDeviceGetName(buffer.as_mut_ptr(), 64, ordinal) },
        Success
    );
    unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

fn names() -> Vec<String> {
    (0..count()).map(name).collect()
}

fn attr(ordinal: c_int, attribute: c_int) -> c_int {
    let mut value = -1;
    assert_eq!(
        unsafe { cuDeviceGetAttribute(&mut value, attribute, ordinal) },
        Success
    );
    value
}

fn uuid(ordinal: c_int) -> [u8; 16] {
    let mut out = [0u8; 16];
    assert_eq!(unsafe { cuDeviceGetUuid(&mut out, ordinal) }, Success);
    out
}

fn context(ordinal: c_int) -> Opaque {
    let mut out = null_mut();
    assert_eq!(unsafe { cuCtxCreate_v2(&mut out, 0, ordinal) }, Success);
    out
}

fn alloc(size: usize) -> u64 {
    let mut out = 0;
    assert_eq!(unsafe { cuMemAlloc_v2(&mut out, size) }, Success);
    out
}

fn current_device() -> c_int {
    let mut out = -1;
    assert_eq!(unsafe { cuCtxGetDevice(&mut out) }, Success);
    out
}

fn current_ctx() -> Opaque {
    let mut out = without_provenance_mut(99);
    assert_eq!(unsafe { cuCtxGetCurrent(&mut out) }, Success);
    out
}

fn selection(order: Option<&str>, visible: Option<&str>) -> Selection {
    Selection {
        order: order.map(Into::into),
        visible: visible.map(Into::into),
    }
}

// ---------------------------------------------------------------------------
// Enumeration, ordering, filtering
// ---------------------------------------------------------------------------

#[test]
fn devices_of_several_drivers_are_all_enumerated_discrete_first() {
    let _f = Fixture::new(&mixed());
    assert_eq!(count(), 4);
    // 16 GiB dGPUs first (driver order breaks the tie), then 8 GiB, integrated last.
    assert_eq!(names(), ["dGPU-A", "dGPU-C", "dGPU-B", "iGPU"]);
    assert_eq!(attr(3, 18), 1, "only the iGPU is integrated");
    assert_eq!(attr(0, 18), 0);
}

#[test]
fn pci_bus_id_order_and_lookup_follow_the_hardware() {
    let mut specs = mixed();
    specs[3].pci = Some([0, 0x01, 0, 0]);
    specs[0].pci = Some([0, 0x40, 0, 0]);
    specs[1].pci = Some([0, 0x20, 0, 0]);
    specs[2].pci = Some([1, 0x03, 0, 0]);
    let _f = Fixture::with(&specs, &selection(Some("PCI_BUS_ID"), None));
    assert_eq!(names(), ["dGPU-C", "dGPU-A", "iGPU", "dGPU-B"]);
    let mut out = [0 as c_char; 32];
    assert_eq!(unsafe { cuDeviceGetPCIBusId(out.as_mut_ptr(), 32, 3) }, Success);
    assert_eq!(unsafe { CStr::from_ptr(out.as_ptr()) }, c"0001:03:00.0");
    let mut found = -1;
    assert_eq!(
        unsafe { cuDeviceGetByPCIBusId(&mut found, c"0000:20:00.0".as_ptr()) },
        Success
    );
    assert_eq!(found, 1);
    assert_eq!(
        unsafe { cuDeviceGetByPCIBusId(&mut found, c"0000:77:00.0".as_ptr()) },
        InvalidDevice
    );
    assert_eq!(
        unsafe { cuDeviceGetByPCIBusId(&mut found, c"nonsense".as_ptr()) },
        InvalidValue
    );
    assert_eq!(unsafe { cuDeviceGetByPCIBusId(null_mut(), c"0:0.0".as_ptr()) }, InvalidValue);
    // Per-device PCI attributes: bus (33), device (34) and domain (50).
    assert_eq!((attr(3, 50), attr(3, 33), attr(3, 34)), (1, 3, 0));
}

#[test]
fn visible_devices_select_reorder_and_renumber() {
    let _f = Fixture::with(&mixed(), &selection(None, Some("3,0")));
    // CUDA order is [A, C, B, iGPU]; "3,0" keeps iGPU then A.
    assert_eq!(names(), ["iGPU", "dGPU-A"]);
    assert_eq!(count(), 2);
    let mut out = 0;
    assert_eq!(unsafe { cuDeviceGet(&mut out, 1) }, Success);
    assert_eq!(unsafe { cuDeviceGet(&mut out, 2) }, InvalidDevice);
    assert_eq!(unsafe { cuDeviceGet(&mut out, -1) }, InvalidDevice);
    let mut n = 0;
    assert_eq!(unsafe { cuDeviceGetName(null_mut(), 8, 0) }, InvalidValue);
    assert_eq!(unsafe { cuDeviceTotalMem_v2(&mut n, 5) }, InvalidDevice);
}

#[test]
fn visible_devices_by_uuid_survive_reordering() {
    let wanted = order::format_uuid(&[3; 16]); // dGPU-B
    let _f = Fixture::with(&mixed(), &selection(None, Some(&format!("{wanted},GPU-0404"))));
    assert_eq!(names(), ["dGPU-B", "dGPU-C"]);
    // "GPU-0404" is a unique prefix of dGPU-C's all-0x04 UUID.
    assert_eq!(uuid(1), [4; 16]);
}

#[test]
fn invalid_visible_entries_cut_the_list_and_empty_hides_all() {
    let _f = Fixture::with(&mixed(), &selection(None, Some("1,9,0")));
    assert_eq!(names(), ["dGPU-C"]);
    drop(_f);
    let _f = Fixture::with(&mixed(), &selection(None, Some("")));
    assert_eq!(count(), 0, "no visible device");
    let mut out = 0;
    assert_eq!(unsafe { cuDeviceGet(&mut out, 0) }, InvalidDevice);
}

#[test]
fn a_single_device_is_left_untouched() {
    let mut only = spec(0, 0, "Solo");
    only.uuid = [0; 16];
    let _f = Fixture::new(&[only]);
    assert_eq!(names(), ["Solo"]);
    assert_eq!(uuid(0), [0; 16], "a lone device keeps the driver's UUID");
}

#[test]
fn device_uuids_are_distinct_and_stable() {
    let mut specs = mixed();
    for s in &mut specs {
        s.uuid = [0; 16]; // a driver that reports no UUID
        s.pci = None;
    }
    let first = {
        let _f = Fixture::new(&specs);
        (0..4).map(uuid).collect::<Vec<_>>()
    };
    let second = {
        let _f = Fixture::new(&specs);
        (0..4).map(uuid).collect::<Vec<_>>()
    };
    assert_eq!(first, second, "synthesized UUIDs are deterministic");
    for (i, a) in first.iter().enumerate() {
        assert_ne!(*a, [0; 16]);
        for b in &first[i + 1..] {
            assert_ne!(a, b);
        }
    }
    // The v2 entry point answers identically.
    let _f = Fixture::new(&specs);
    let mut out = [0u8; 16];
    assert_eq!(unsafe { cuDeviceGetUuid_v2(&mut out, 2) }, Success);
    assert_eq!(out, uuid(2));
    assert_eq!(unsafe { cuDeviceGetUuid_v2(&mut out, 4) }, InvalidDevice);
    assert_eq!(unsafe { cuDeviceGetUuid_v2(null_mut(), 0) }, InvalidValue);
}

#[test]
fn reported_uuids_come_from_the_level_zero_device() {
    let _f = Fixture::new(&mixed());
    // Specs carry [index + 1; 16]; CUDA order is A(2), C(4), B(3), iGPU(1).
    assert_eq!([uuid(0), uuid(1), uuid(2), uuid(3)], [[2; 16], [4; 16], [3; 16], [1; 16]]);
}

#[test]
fn attributes_describe_each_device_not_the_first() {
    let _f = Fixture::new(&mixed());
    // CUDA order: 0 A (32 KiB SLM), 1 C (128 KiB, 1024 threads), 2 B, 3 iGPU.
    assert_eq!(attr(0, 8), 32768);
    assert_eq!(attr(1, 8), 131072);
    assert_eq!(attr(1, 1), 1024);
    assert_eq!(attr(0, 1), 256);
    // SM stand-in derives from threads / widest subgroup: 256/16, 1024/32, 256/32.
    assert_eq!((attr(0, 16), attr(1, 16), attr(2, 16)), (16, 32, 8));
    let mut total = 0usize;
    assert_eq!(unsafe { cuDeviceTotalMem_v2(&mut total, 0) }, Success);
    assert_eq!(total as u64, 16 * GIB);
    assert_eq!(unsafe { cuDeviceTotalMem_v2(&mut total, 3) }, Success);
    assert_eq!(total as u64, 4 * GIB);
}

#[test]
fn tiles_of_one_card_form_a_multi_gpu_board() {
    // Flat hierarchy: two tiles of a card share a PCI address.
    let mut tile0 = spec(0, 0, "Max tile");
    let mut tile1 = spec(1, 0, "Max tile");
    let alone = spec(2, 1, "Arc");
    tile0.pci = Some([0, 0x9a, 0, 0]);
    tile1.pci = Some([0, 0x9a, 0, 0]);
    tile0.total_memory = 32 * GIB;
    tile1.total_memory = 32 * GIB;
    let _f = Fixture::new(&[alone, tile0, tile1]);
    assert_eq!(names(), ["Max tile", "Max tile", "Arc"]);
    assert_eq!((attr(0, 84), attr(1, 84), attr(2, 84)), (1, 1, 0));
    assert_eq!((attr(0, 85), attr(1, 85)), (0, 0), "one board group");
    assert_eq!(attr(2, 85), 2);
    // Tiles of one driver keep distinct indices inside it.
    let state = RUNTIME.lock().unwrap();
    let devices = &state.as_ref().unwrap().runtime.devices;
    assert_eq!((devices[0].device_index, devices[1].device_index), (0, 1));
}

#[test]
fn cache_tags_separate_different_gpus_but_not_identical_ones() {
    let mut twin_a = spec(0, 0, "Arc A770");
    let mut twin_b = spec(1, 0, "Arc A770");
    twin_a.device_id = 0x56a0;
    twin_b.device_id = 0x56a0;
    let mut other = spec(2, 1, "Arc B580");
    other.device_id = 0xe20b;
    let _f = Fixture::new(&[twin_a, twin_b, other]);
    let state = RUNTIME.lock().unwrap();
    let devices = &state.as_ref().unwrap().runtime.devices;
    let tag = |i: usize| crate::level_zero::cache_tag(&devices[i].info);
    let tags: Vec<_> = (0..3).map(tag).collect();
    assert_eq!(tags.iter().filter(|t| **t == tag(0)).count(), 2);
    assert_eq!(tags.iter().filter(|t| **t == tag(2)).count(), 1);
}

// ---------------------------------------------------------------------------
// Contexts and the current context
// ---------------------------------------------------------------------------

#[test]
fn current_context_follows_the_device_of_the_last_context_set() {
    let _f = Fixture::new(&mixed());
    assert!(current_ctx().is_null());
    let (c0, c1, c2) = (context(0), context(1), context(2));
    assert_eq!(current_device(), 2);
    assert_eq!(cuCtxSetCurrent(c0), Success);
    assert_eq!((current_device(), current_ctx()), (0, c0));
    assert_eq!(cuCtxSetCurrent(c1), Success);
    assert_eq!(current_device(), 1);
    let mut device = -1;
    assert_eq!(unsafe { cuCtxGetDevice_v2(&mut device, c2) }, Success);
    assert_eq!(device, 2, "explicit handle ignores the current context");
    assert_eq!(cuCtxSetCurrent(null_mut()), Success);
    assert_eq!(unsafe { cuCtxGetDevice(&mut device) }, InvalidContext);
    for c in [c0, c1, c2] {
        assert_eq!(cuCtxDestroy_v2(c), Success);
    }
}

#[test]
fn push_and_pop_switch_devices_like_a_stack() {
    let _f = Fixture::new(&pair());
    let c0 = context(0);
    let c1 = context(1);
    assert_eq!(current_device(), 1);
    assert_eq!(cuCtxPushCurrent_v2(c0), Success);
    assert_eq!(current_device(), 0);
    let mut popped = null_mut();
    assert_eq!(unsafe { cuCtxPopCurrent_v2(&mut popped) }, Success);
    assert_eq!(popped, c0);
    assert_eq!((current_device(), current_ctx()), (1, c1));
    assert_eq!(unsafe { cuCtxPopCurrent_v2(&mut popped) }, Success);
    assert_eq!(popped, c1);
    assert_eq!(unsafe { cuCtxPopCurrent_v2(&mut popped) }, Success, "c0's creation entry");
    assert_eq!(popped, c0);
    assert_eq!(unsafe { cuCtxPopCurrent_v2(&mut popped) }, InvalidContext);
    assert_eq!(cuCtxPushCurrent_v2(without_provenance_mut(0xdead)), InvalidContext);
    assert_eq!(cuCtxDestroy_v2(c0), Success);
    assert_eq!(cuCtxDestroy_v2(c1), Success);
}

#[test]
fn the_current_context_is_per_thread() {
    let _f = Fixture::new(&pair());
    let c0 = context(0);
    let key = c0.addr();
    let other = std::thread::spawn(move || {
        assert!(current_ctx().is_null());
        let mut device = 0;
        assert_eq!(unsafe { cuCtxGetDevice(&mut device) }, InvalidContext);
        let c1 = context(1);
        assert_eq!(current_device(), 1);
        // Another thread may use the first thread's context by handle.
        assert_eq!(cuCtxSetCurrent(without_provenance_mut(key)), Success);
        assert_eq!(current_device(), 0);
        assert_eq!(cuCtxSetCurrent(c1), Success);
        assert_eq!(cuCtxDestroy_v2(c1), Success);
    });
    other.join().unwrap();
    assert_eq!(current_device(), 0, "this thread was unaffected");
    assert_eq!(cuCtxDestroy_v2(c0), Success);
}

#[test]
fn contexts_cannot_be_created_on_missing_devices() {
    let _f = Fixture::new(&pair());
    let mut out = null_mut();
    assert_eq!(unsafe { cuCtxCreate_v2(&mut out, 0, 2) }, InvalidDevice);
    assert_eq!(unsafe { cuCtxCreate_v2(&mut out, 0, -1) }, InvalidDevice);
}

#[test]
fn primary_contexts_are_independent_per_device() {
    let _f = Fixture::new(&mixed());
    let mut a = null_mut();
    let mut b = null_mut();
    let (mut flags, mut active) = (0, -1);
    assert_eq!(unsafe { cuDevicePrimaryCtxRetain(&mut a, 0) }, Success);
    assert_eq!(unsafe { cuDevicePrimaryCtxGetState(1, &mut flags, &mut active) }, Success);
    assert_eq!(active, 0, "device 1 stays inactive");
    assert_eq!(unsafe { cuDevicePrimaryCtxRetain(&mut b, 1) }, Success);
    assert_ne!(a, b);
    let mut again = null_mut();
    assert_eq!(unsafe { cuDevicePrimaryCtxRetain(&mut again, 0) }, Success);
    assert_eq!(again, a, "same device, same primary context");
    let mut device = -1;
    assert_eq!(unsafe { cuCtxGetDevice_v2(&mut device, b) }, Success);
    assert_eq!(device, 1);
    assert_eq!(cuDevicePrimaryCtxRelease_v2(0), Success);
    assert_eq!(unsafe { cuDevicePrimaryCtxGetState(0, &mut flags, &mut active) }, Success);
    assert_eq!(active, 1, "one retain remains on device 0");
    assert_eq!(cuDevicePrimaryCtxRelease_v2(0), Success);
    assert_eq!(unsafe { cuDevicePrimaryCtxGetState(0, &mut flags, &mut active) }, Success);
    assert_eq!(active, 0);
    assert_eq!(unsafe { cuDevicePrimaryCtxGetState(1, &mut flags, &mut active) }, Success);
    assert_eq!(active, 1, "device 1 is unaffected");
    assert_eq!(cuDevicePrimaryCtxRelease_v2(1), Success);
    assert_eq!(cuDevicePrimaryCtxRelease_v2(1), InvalidContext);
    assert_eq!(cuDevicePrimaryCtxRelease_v2(9), InvalidDevice);
}

// ---------------------------------------------------------------------------
// Memory ownership
// ---------------------------------------------------------------------------

#[test]
fn pointer_attributes_name_the_owning_device_whatever_is_current() {
    let _f = Fixture::new(&mixed());
    let c0 = context(0);
    let p0 = alloc(256);
    let c3 = context(3);
    let p3 = alloc(256);
    let ordinal_of = |pointer: u64| {
        let mut value = u32::MAX;
        assert_eq!(
            unsafe { cuPointerGetAttribute((&raw mut value).cast(), 9, pointer) },
            Success
        );
        value
    };
    let context_of = |pointer: u64| {
        let mut value = 0u64;
        assert_eq!(
            unsafe { cuPointerGetAttribute((&raw mut value).cast(), 1, pointer) },
            Success
        );
        value as usize
    };
    // Current is device 3 now.
    assert_eq!((ordinal_of(p0), ordinal_of(p3)), (0, 3));
    assert_eq!((context_of(p0), context_of(p3)), (c0.addr(), c3.addr()));
    assert_eq!(cuCtxSetCurrent(c0), Success);
    assert_eq!((ordinal_of(p0), ordinal_of(p3)), (0, 3));
    let mut value = 0u32;
    assert_eq!(
        unsafe { cuPointerGetAttribute((&raw mut value).cast(), 9, 0x1234) },
        InvalidValue,
        "unowned pointer"
    );
    assert_eq!(cuMemFree_v2(p3), Success, "freed from another device's context");
    assert_eq!(
        unsafe { cuPointerGetAttribute((&raw mut value).cast(), 9, p3) },
        InvalidValue
    );
    assert_eq!(cuMemFree_v2(p3), InvalidValue, "double free");
    assert_eq!(cuMemFree_v2(p0), Success);
    assert_eq!(cuCtxDestroy_v2(c3), Success);
    assert_eq!(cuCtxDestroy_v2(c0), Success);
}

#[test]
fn allocation_limits_and_free_memory_are_per_device() {
    let mut small = spec(0, 0, "Small");
    small.max_allocation = 1024;
    small.total_memory = 1 << 20;
    let mut big = spec(1, 1, "Big");
    big.max_allocation = 1 << 20;
    big.total_memory = 8 << 20;
    let _f = Fixture::new(&[small, big]);
    // Discrete GPUs tie on kind; "Big" has more memory and so is device 0.
    assert_eq!(names(), ["Big", "Small"]);
    let c_big = context(0);
    let c_small = context(1);
    let mut p = 0;
    assert_eq!(unsafe { cuMemAlloc_v2(&mut p, 4096) }, OutOfMemory, "Small's cap");
    assert_eq!(cuCtxSetCurrent(c_big), Success);
    let kept = alloc(4096);
    let (mut free, mut total) = (0, 0);
    assert_eq!(unsafe { cuMemGetInfo_v2(&mut free, &mut total) }, Success);
    assert_eq!((total, free), (8 << 20, (8 << 20) - 4096));
    assert_eq!(cuCtxSetCurrent(c_small), Success);
    assert_eq!(unsafe { cuMemGetInfo_v2(&mut free, &mut total) }, Success);
    assert_eq!((total, free), (1 << 20, 1 << 20), "other device's allocation not counted");
    assert_eq!(cuCtxSetCurrent(c_big), Success);
    assert_eq!(cuMemFree_v2(kept), Success);
    assert_eq!(cuCtxDestroy_v2(c_small), Success);
    assert_eq!(cuCtxDestroy_v2(c_big), Success);
}

// ---------------------------------------------------------------------------
// Peer access and copies
// ---------------------------------------------------------------------------

#[test]
fn peer_access_is_honestly_unsupported_but_ranks_links() {
    let _f = Fixture::new(&mixed());
    // CUDA order: 0 A, 1 C, 2 B, 3 iGPU; A and B are linked (same driver and group).
    let can = |a, b| {
        let mut out = -1;
        assert_eq!(unsafe { cuDeviceCanAccessPeer(&mut out, a, b) }, Success);
        out
    };
    assert_eq!((can(0, 2), can(0, 1), can(0, 0), can(1, 3)), (0, 0, 0, 0));
    let mut out = -1;
    assert_eq!(unsafe { cuDeviceCanAccessPeer(&mut out, 0, 4) }, InvalidDevice);
    assert_eq!(unsafe { cuDeviceCanAccessPeer(&mut out, -1, 0) }, InvalidDevice);
    let rank = |a, b, attribute| {
        let mut out = -1;
        assert_eq!(unsafe { cuDeviceGetP2PAttribute(&mut out, attribute, a, b) }, Success);
        out
    };
    assert_eq!((rank(0, 2, 1), rank(2, 0, 1)), (1, 1), "native link is symmetric");
    assert_eq!(rank(0, 1, 1), 2, "different drivers stage through the host");
    assert_eq!(rank(0, 2, 2), 0, "no direct kernel access");
    assert_eq!((rank(0, 2, 3), rank(0, 2, 4), rank(0, 2, 5)), (0, 0, 0));
    assert_eq!(unsafe { cuDeviceGetP2PAttribute(&mut out, 1, 0, 0) }, InvalidDevice);
    assert_eq!(unsafe { cuDeviceGetP2PAttribute(&mut out, 9, 0, 1) }, InvalidValue);
    assert_eq!(unsafe { cuDeviceGetP2PAttribute(&mut out, 1, 0, 7) }, InvalidDevice);
}

#[test]
fn enabling_peer_access_reports_unsupported_like_cuda() {
    let _f = Fixture::new(&mixed());
    assert_eq!(cuCtxEnablePeerAccess(null_mut(), 0), InvalidContext, "nothing current");
    let c0 = context(0);
    let c2 = context(2);
    let c0b = context(0);
    assert_eq!(cuCtxEnablePeerAccess(c2, 1), InvalidValue, "flags must be 0");
    assert_eq!(cuCtxEnablePeerAccess(c2, 0), PeerAccessUnsupported);
    assert_eq!(cuCtxEnablePeerAccess(c0, 0), InvalidDevice, "same device");
    assert_eq!(cuCtxEnablePeerAccess(without_provenance_mut(777), 0), InvalidContext);
    assert_eq!(cuCtxDisablePeerAccess(c2), PeerAccessNotEnabled);
    for c in [c0b, c2, c0] {
        assert_eq!(cuCtxDestroy_v2(c), Success);
    }
}

#[test]
fn peer_access_state_machine_when_the_link_allows_direct_access() {
    let _f = Fixture::new(&mixed());
    peer::set_direct_kernel_access(true);
    let can = |a, b| {
        let mut out = -1;
        assert_eq!(unsafe { cuDeviceCanAccessPeer(&mut out, a, b) }, Success);
        out
    };
    // Only the linked A/B pair qualifies; B/iGPU and cross-driver pairs do not.
    assert_eq!((can(0, 2), can(2, 0), can(0, 1), can(2, 3)), (1, 1, 0, 0));
    let mut access = -1;
    assert_eq!(unsafe { cuDeviceGetP2PAttribute(&mut access, 2, 0, 2) }, Success);
    assert_eq!(access, 1);
    let c_a = context(0);
    let c_b = context(2);
    let c_c = context(1);
    assert_eq!(cuCtxSetCurrent(c_a), Success);
    assert_eq!(cuCtxEnablePeerAccess(c_b, 0), Success);
    assert_eq!(cuCtxEnablePeerAccess(c_b, 0), PeerAccessAlreadyEnabled);
    assert_eq!(cuCtxEnablePeerAccess(c_c, 0), PeerAccessUnsupported);
    assert_eq!(cuCtxSetCurrent(c_b), Success);
    assert_eq!(cuCtxDisablePeerAccess(c_a), PeerAccessNotEnabled, "directional");
    assert_eq!(cuCtxEnablePeerAccess(c_a, 0), Success);
    assert_eq!(cuCtxSetCurrent(c_a), Success);
    assert_eq!(cuCtxDisablePeerAccess(c_b), Success);
    assert_eq!(cuCtxDisablePeerAccess(c_b), PeerAccessNotEnabled);
    assert_eq!(cuCtxEnablePeerAccess(c_b, 0), Success, "re-enable after disable");
    for c in [c_c, c_b, c_a] {
        assert_eq!(cuCtxDestroy_v2(c), Success);
    }
}

/// Copies issued on the devices given by their position in the fixture's spec list.
fn copies_of<const N: usize>(specs: [usize; N]) -> Vec<(usize, usize, usize, usize)> {
    fake::copy_log()
        .into_iter()
        .filter(|(device, ..)| specs.iter().any(|&i| fake::device_handle(i) == *device))
        .collect()
}

#[test]
fn peer_copies_are_staged_through_host_memory_in_chunks() {
    let mut specs = pair();
    for s in &mut specs {
        s.max_allocation = 64 << 20;
    }
    let _f = Fixture::new(&specs);
    let c0 = context(0);
    let src = alloc(10 << 20);
    let c1 = context(1);
    let dst = alloc(10 << 20);
    let before = copies_of([0, 1]).len();
    let size = 9 << 20; // 4 MiB + 4 MiB + 1 MiB
    assert_eq!(unsafe { cuMemcpyPeer(dst, c1, src, c0, size) }, Success);
    let log = &copies_of([0, 1])[before..];
    assert_eq!(log.len(), 6, "three chunks, a read and a write each");
    let host = log[0].1;
    for (i, (offset, length)) in [(0, CHUNK), (CHUNK, CHUNK), (2 * CHUNK, 1 << 20)]
        .into_iter()
        .enumerate()
    {
        let (read, write) = (log[2 * i], log[2 * i + 1]);
        assert_eq!(read, (fake::device_handle(0), host, src as usize + offset, length), "read by the source context");
        assert_eq!(write, (fake::device_handle(1), dst as usize + offset, host, length), "write by the destination context");
    }
    assert_ne!(host, dst as usize);
    // The async form with the null/legacy stream does the same.
    let before = copies_of([0, 1]).len();
    assert_eq!(
        unsafe { cuMemcpyPeerAsync(dst, c1, src, c0, 100, null_mut()) },
        Success
    );
    assert_eq!(copies_of([0, 1]).len() - before, 2);
    // Bogus stream handles are rejected, real ones accepted.
    assert_eq!(
        unsafe { cuMemcpyPeerAsync(dst, c1, src, c0, 100, without_provenance_mut(0x4242)) },
        InvalidHandle
    );
    let mut stream = null_mut();
    assert_eq!(unsafe { cuStreamCreate(&mut stream, 1) }, Success);
    assert_eq!(unsafe { cuMemcpyPeerAsync(dst, c1, src, c0, 100, stream) }, Success);
    assert_eq!(cuStreamDestroy_v2(stream), Success);
    assert_eq!(cuMemFree_v2(dst), Success);
    assert_eq!(cuCtxSetCurrent(c0), Success);
    assert_eq!(cuMemFree_v2(src), Success);
    assert_eq!(cuCtxDestroy_v2(c1), Success);
    assert_eq!(cuCtxDestroy_v2(c0), Success);
}

#[test]
fn peer_copy_validates_ranges_and_contexts() {
    let _f = Fixture::new(&pair());
    let c0 = context(0);
    let src = alloc(128);
    let c1 = context(1);
    let dst = alloc(64);
    assert_eq!(unsafe { cuMemcpyPeer(dst, c1, src, c0, 65) }, InvalidValue, "dst too small");
    assert_eq!(unsafe { cuMemcpyPeer(dst, c1, src + 100, c0, 64) }, InvalidValue, "src overruns");
    assert_eq!(unsafe { cuMemcpyPeer(dst, c0, src, c1, 8) }, InvalidValue, "swapped owners");
    assert_eq!(unsafe { cuMemcpyPeer(dst, c1, src, c0, 0) }, Success, "empty copy");
    assert_eq!(
        unsafe { cuMemcpyPeer(dst, without_provenance_mut(5), src, c0, 8) },
        InvalidContext
    );
    let before = copies_of([0, 1]).len();
    assert_eq!(unsafe { cuMemcpyPeer(dst, c1, src, c0, 64) }, Success);
    assert_eq!(copies_of([0, 1]).len() - before, 2);
    // Same context: a plain device copy, no staging.
    let second = alloc(64);
    let before = copies_of([1]).len();
    assert_eq!(unsafe { cuMemcpyPeer(second, c1, dst, c1, 64) }, Success);
    assert_eq!(copies_of([1]).len() - before, 1);
    assert_eq!(cuMemFree_v2(second), Success);
    assert_eq!(cuMemFree_v2(dst), Success);
    assert_eq!(cuCtxDestroy_v2(c1), Success);
    assert_eq!(unsafe { cuMemcpyPeer(dst, c1, src, c0, 8) }, InvalidContext, "destroyed");
    assert_eq!(cuCtxSetCurrent(c0), Success);
    assert_eq!(cuMemFree_v2(src), Success);
    assert_eq!(cuCtxDestroy_v2(c0), Success);
}

#[test]
fn device_to_device_copies_across_devices_use_the_peer_path() {
    let _f = Fixture::new(&pair());
    let c0 = context(0);
    let a = alloc(256);
    let a2 = alloc(256);
    let c1 = context(1);
    let b = alloc(256);
    // Current is device 1, the source lives on device 0.
    let before = fake::copy_log().len();
    assert_eq!(unsafe { cuMemcpyDtoD_v2(b, a, 128) }, Success);
    let log = &fake::copy_log()[before..];
    assert_eq!(log.len(), 2, "staged read + write");
    assert_eq!((log[0].0, log[1].0), (fake::device_handle(0), fake::device_handle(1)));
    let before = fake::copy_log().len();
    assert_eq!(unsafe { cuMemcpyDtoDAsync_v2(a, b, 128, null_mut()) }, Success);
    assert_eq!(fake::copy_log().len() - before, 2);
    // Both operands local: ordinary single copy.
    assert_eq!(cuCtxSetCurrent(c0), Success);
    let before = fake::copy_log().len();
    assert_eq!(unsafe { cuMemcpyDtoD_v2(a2, a, 64) }, Success);
    assert_eq!(fake::copy_log().len() - before, 1);
    // An address nobody owns is still rejected.
    assert_eq!(unsafe { cuMemcpyDtoD_v2(0x10, a, 64) }, InvalidValue);
    for p in [a, a2] {
        assert_eq!(cuMemFree_v2(p), Success);
    }
    assert_eq!(cuMemFree_v2(b), Success, "device 1 allocation freed from device 0");
    assert_eq!(cuCtxDestroy_v2(c1), Success);
    assert_eq!(cuCtxDestroy_v2(c0), Success);
}

// ---------------------------------------------------------------------------
// Streams and events across devices
// ---------------------------------------------------------------------------

#[test]
fn a_stream_may_wait_for_an_event_recorded_on_another_device() {
    let _f = Fixture::new(&pair());
    let c0 = context(0);
    let (mut s0, mut e0) = (null_mut(), null_mut());
    assert_eq!(unsafe { cuStreamCreate(&mut s0, 1) }, Success);
    assert_eq!(unsafe { cuEventCreate(&mut e0, 0) }, Success);
    let c1 = context(1);
    let mut s1 = null_mut();
    assert_eq!(unsafe { cuStreamCreate(&mut s1, 1) }, Success);
    // Never recorded: waiting is a no-op.
    assert_eq!(cuStreamWaitEvent(s1, e0, 0), Success);
    assert_eq!(cuStreamWaitEvent(s1, e0, 1), InvalidValue);
    // Record on device 0, wait on device 1.
    assert_eq!(cuCtxSetCurrent(c0), Success);
    assert_eq!(cuEventRecord(e0, s0), Success);
    assert_eq!(cuCtxSetCurrent(c1), Success);
    assert_eq!(cuStreamWaitEvent(s1, e0, 0), Success);
    // The default stream waits too, and an unknown event is still an error.
    assert_eq!(cuStreamWaitEvent(null_mut(), e0, 0), Success);
    assert_eq!(
        cuStreamWaitEvent(s1, without_provenance_mut(0x9999), 0),
        InvalidHandle
    );
    // A stream of another device waits for an event while this device is current.
    assert_eq!(cuStreamWaitEvent(s0, e0, 0), Success, "same-context wait unchanged");
    assert_eq!(cuEventSynchronize(e0), InvalidHandle, "events stay bound to their device");
    assert_eq!(cuCtxSetCurrent(c0), Success);
    assert_eq!(cuEventSynchronize(e0), Success);
    assert_eq!(cuEventDestroy_v2(e0), Success);
    assert_eq!(cuCtxSetCurrent(c1), Success);
    assert_eq!(cuStreamWaitEvent(s1, e0, 0), InvalidHandle, "destroyed event");
    assert_eq!(cuStreamDestroy_v2(s1), Success);
    assert_eq!(cuCtxSetCurrent(c0), Success);
    assert_eq!(cuStreamDestroy_v2(s0), Success);
    assert_eq!(cuCtxDestroy_v2(c1), Success);
    assert_eq!(cuCtxDestroy_v2(c0), Success);
}

#[test]
fn streams_belong_to_their_device_even_when_another_is_current() {
    let _f = Fixture::new(&pair());
    let c0 = context(0);
    let mut s0 = null_mut();
    assert_eq!(unsafe { cuStreamCreate(&mut s0, 1) }, Success);
    let c1 = context(1);
    // Synchronize and destroy resolve the stream's owner, not the current context.
    assert_eq!(crate::cuda::streams::cuStreamSynchronize(s0), Success);
    assert_eq!(cuStreamDestroy_v2(s0), Success);
    assert_eq!(cuStreamDestroy_v2(s0), InvalidHandle);
    assert_eq!(cuCtxDestroy_v2(c1), Success);
    assert_eq!(cuCtxDestroy_v2(c0), Success);
}

// ---------------------------------------------------------------------------
// Modules, kernels and probes per device
// ---------------------------------------------------------------------------

const SMALL_KERNEL: &CStr = c".version 4.1 .target sm_52 .address_size 64 \
.entry k(.param .u64 .ptr .global dst) { .reg .b64 %rd<2>; .reg .b32 %r<2>; \
ld.param.u64 %rd1, [dst]; mov.u32 %r1, 7; st.global.u32 [%rd1], %r1; ret; }";

const BIG_SHARED_KERNEL: &CStr = c".version 4.1 .target sm_52 .address_size 64 \
.entry big(.param .u64 .ptr .global dst) { .reg .b64 %rd<2>; .reg .b32 %r<2>; .shared .align 4 .b8 buf[65536]; \
ld.param.u64 %rd1, [dst]; mov.u32 %r1, 7; st.shared.u32 [buf], %r1; st.global.u32 [%rd1], %r1; ret; }";

fn load(image: &CStr) -> (CuResult, Opaque) {
    let mut module = null_mut();
    let result = unsafe { cuModuleLoadData(&mut module, image.as_ptr().cast()) };
    (result, module)
}

#[test]
fn modules_are_built_for_and_launched_on_their_own_device() {
    let _f = Fixture::new(&mixed());
    let c_a = context(0);
    let (result, m_a) = load(SMALL_KERNEL);
    assert_eq!(result, Success);
    let c_c = context(1);
    let (result, m_c) = load(SMALL_KERNEL);
    assert_eq!(result, Success, "built again for the second device");
    let mut f_a = null_mut();
    let mut f_c = null_mut();
    assert_eq!(unsafe { cuModuleGetFunction(&mut f_c, m_c, c"k".as_ptr()) }, Success);
    let p_c = alloc(64);
    let launch = |function: Opaque, pointer: u64| {
        let mut arg = std::ptr::from_ref(&pointer).cast_mut().cast();
        unsafe { cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, null_mut(), &mut arg, null_mut()) }
    };
    assert_eq!(launch(f_c, p_c), Success);
    assert_eq!(cuCtxSetCurrent(c_a), Success);
    // A module handle is only usable in its own context.
    assert_eq!(unsafe { cuModuleGetFunction(&mut f_a, m_c, c"k".as_ptr()) }, InvalidHandle);
    assert_eq!(unsafe { cuModuleGetFunction(&mut f_a, m_a, c"k".as_ptr()) }, Success);
    assert_ne!(f_a, f_c);
    assert_ne!(launch(f_c, p_c), Success, "function and pointer of another device");
    assert_eq!(launch(f_a, 0x1), InvalidValue, "device pointer owned by nobody here");
    // The module was created in the context that was current: unloading from the other fails.
    assert_eq!(cuModuleUnload(m_c), InvalidHandle);
    assert_eq!(cuModuleUnload(m_a), Success);
    assert_eq!(cuCtxSetCurrent(c_c), Success);
    assert_eq!(cuModuleUnload(m_c), Success);
    assert_eq!(cuMemFree_v2(p_c), Success);
    assert_eq!(cuCtxDestroy_v2(c_c), Success);
    assert_eq!(cuCtxDestroy_v2(c_a), Success);
}

#[test]
fn shared_memory_limits_decide_per_device_whether_a_module_loads() {
    let _f = Fixture::new(&mixed());
    // Device 0 (A) has 32 KiB of shared memory, device 1 (C) has 128 KiB.
    let c_a = context(0);
    assert_eq!(load(BIG_SHARED_KERNEL).0, NotSupported);
    let c_c = context(1);
    let (result, module) = load(BIG_SHARED_KERNEL);
    assert_eq!(result, Success);
    assert_eq!(cuModuleUnload(module), Success);
    assert_eq!(cuCtxDestroy_v2(c_c), Success);
    assert_eq!(cuCtxDestroy_v2(c_a), Success);
}

#[test]
fn translator_probes_are_remembered_per_device() {
    let _f = Fixture::new(&mixed());
    let c_a = context(0);
    let (result, module) = load(SMALL_KERNEL);
    assert_eq!(result, Success);
    assert_eq!(cuModuleUnload(module), Success);
    {
        let state = RUNTIME.lock().unwrap();
        let devices = &state.as_ref().unwrap().runtime.devices;
        assert!(devices[0].probe.get().is_some(), "probed by its first module");
        assert!(devices[1].probe.get().is_none(), "other devices are not probed");
        // 32 KiB device: the window never exceeds what the device offers.
        assert_eq!(devices[0].probe.get().unwrap().dynamic_shared, 32768);
    }
    let c_c = context(1);
    let (result, module) = load(SMALL_KERNEL);
    assert_eq!(result, Success);
    assert_eq!(cuModuleUnload(module), Success);
    let state = RUNTIME.lock().unwrap();
    let devices = &state.as_ref().unwrap().runtime.devices;
    assert_eq!(devices[1].probe.get().unwrap().dynamic_shared, 65536);
    assert_eq!(devices[0].probe.get().unwrap().dynamic_shared, 32768, "not overwritten");
    drop(state);
    assert_eq!(cuCtxDestroy_v2(c_c), Success);
    assert_eq!(cuCtxDestroy_v2(c_a), Success);
}

#[test]
fn probe_results_are_computed_once_per_device() {
    let _f = Fixture::new(&pair());
    let state = RUNTIME.lock().unwrap();
    let devices = &state.as_ref().unwrap().runtime.devices;
    let env = probe::ProbeEnv {
        wanted_window: Some(40000),
        dp4a_requested: true,
    };
    let mut runs = 0;
    let mut make = |device: &Device, supported: bool| {
        probe::ensure(device, || {
            probe::compute(&device.info, env, || {
                runs += 1;
                supported
            })
        })
    };
    let first = make(&devices[0], true);
    let again = make(&devices[0], false);
    let second = make(&devices[1], false);
    assert!(first.dot_product && again.dot_product, "first answer sticks");
    assert!(!second.dot_product, "each device decides for itself");
    assert_eq!(runs, 2);
    // fake devices expose 32768 bytes of shared memory.
    assert_eq!(first.dynamic_shared, 32768);
}

// ---------------------------------------------------------------------------
// IPC
// ---------------------------------------------------------------------------

#[test]
fn ipc_entry_points_validate_then_report_not_supported() {
    let _f = Fixture::new(&pair());
    let c0 = context(0);
    let p = alloc(64);
    let mut handle = IpcHandle::zeroed();
    assert_eq!(unsafe { cuIpcGetMemHandle(&mut handle, p) }, NotSupported);
    assert_eq!(unsafe { cuIpcGetMemHandle(&mut handle, 0x77) }, InvalidValue);
    assert_eq!(unsafe { cuIpcGetMemHandle(null_mut(), p) }, InvalidValue);
    let mut opened = 0u64;
    assert_eq!(unsafe { cuIpcOpenMemHandle_v2(&mut opened, handle, 1) }, NotSupported);
    assert_eq!(unsafe { cuIpcOpenMemHandle_v2(&mut opened, handle, 2) }, InvalidValue);
    assert_eq!(unsafe { cuIpcOpenMemHandle_v2(null_mut(), handle, 1) }, InvalidValue);
    assert_eq!(cuIpcCloseMemHandle(p), NotSupported);
    let mut event = null_mut();
    assert_eq!(unsafe { cuEventCreate(&mut event, 0) }, Success);
    assert_eq!(unsafe { cuIpcGetEventHandle(&mut handle, event) }, NotSupported);
    assert_eq!(unsafe { cuIpcGetEventHandle(&mut handle, null_mut()) }, InvalidValue);
    let mut opened_event = null_mut();
    assert_eq!(unsafe { cuIpcOpenEventHandle(&mut opened_event, handle) }, NotSupported);
    assert_eq!(unsafe { cuIpcOpenEventHandle(null_mut(), handle) }, InvalidValue);
    assert_eq!(cuEventDestroy_v2(event), Success);
    assert_eq!(cuMemFree_v2(p), Success);
    assert_eq!(cuCtxDestroy_v2(c0), Success);
}

#[test]
fn driver_entry_points_for_multi_device_apis_are_discoverable() {
    use crate::cuda::discovery::cuGetProcAddress;
    for name in [
        c"cuDeviceGetUuid_v2",
        c"cuDeviceGetByPCIBusId",
        c"cuIpcGetMemHandle",
        c"cuIpcOpenMemHandle",
        c"cuIpcCloseMemHandle",
        c"cuIpcGetEventHandle",
        c"cuIpcOpenEventHandle",
        c"cuDeviceCanAccessPeer",
        c"cuMemcpyPeerAsync",
    ] {
        let mut function = null_mut();
        assert_eq!(
            unsafe { cuGetProcAddress(name.as_ptr(), &mut function, 12000, 0) },
            Success,
            "{name:?}"
        );
        assert!(!function.is_null());
    }
}
