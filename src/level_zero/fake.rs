//! Test-only Level Zero lifecycle model. Commands carry dependencies and
//! completion signals, with no kernel execution or memory emulation.
use super::*;
mod abi;
use std::collections::{BTreeMap, VecDeque};
use std::ptr::{null_mut, without_provenance_mut};
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

const UNKNOWN: Status = 0x7fff_fffe;
const NOT_READY: Status = 1;
/// Description of one fake device. `start()` uses two distinct-driver defaults;
/// `start_with()` builds topologies of any size for multi-device tests.
#[derive(Clone, Debug)]
pub(crate) struct DeviceSpec {
    pub name: String,
    /// Devices with equal `driver` values share one fake Level Zero driver.
    pub driver: usize,
    pub integrated: bool,
    pub total_memory: u64,
    pub max_allocation: u64,
    pub uuid: [u8; 16],
    pub pci: Option<[u32; 4]>,
    pub fp64: bool,
    pub subgroup_sizes: Vec<u32>,
    pub max_workgroup_size: u32,
    pub shared_local_memory: u32,
    pub device_id: u32,
    /// Devices of one driver that share a group report `zeDeviceCanAccessPeer`.
    pub p2p_group: Option<u32>,
}

impl DeviceSpec {
    pub(crate) fn new(index: usize, driver: usize) -> Self {
        Self {
            name: "Lifecycle fixture".into(),
            driver,
            integrated: false,
            total_memory: 1 << 20,
            max_allocation: if index == 0 { 4096 } else { 128 },
            uuid: [0; 16],
            pci: Some([0, 2, 3, 1]),
            fp64: false,
            subgroup_sizes: vec![16],
            max_workgroup_size: 256,
            shared_local_memory: 32768,
            device_id: index as u32,
            p2p_group: None,
        }
    }
}

/// Distinct drivers, device handles and compute queue ordinals for device `index`
/// in driver group `driver`: drivers get odd ids, devices even ones.
fn ids(index: usize, driver: usize) -> (Id, Id, u32) {
    (1 + 2 * driver, 2 + 2 * index, 7 + 4 * index as u32)
}
/// Device timer frequency advertised by the fake devices.
pub(crate) const FAKE_TIMER_HZ: u64 = 10_000_000;
const WAIT_LIMIT: Duration = Duration::from_secs(3);
static DRIVER: Mutex<Option<Driver>> = Mutex::new(None);
static CHANGED: Condvar = Condvar::new();

type Id = usize;
type DriverResult<T> = Result<T, &'static str>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Kind {
    Context,
    Queue,
    List,
    Module,
    BuildLog,
    Kernel,
    EventPool,
    Event,
    Fence,
    HostAllocation,
    DeviceAllocation,
}

#[derive(Default)]
struct Command {
    resources: Vec<Id>,
    waits: Vec<Id>,
    signal: Id,
    /// Host address that receives the fake device clock when the command completes.
    stamp: usize,
}

impl Command {
    fn references(&self, id: Id) -> bool {
        self.signal == id || self.resources.contains(&id) || self.waits.contains(&id)
    }
}

struct Submission {
    list: Id,
    queue: Id,
    fence: Id,
    commands: VecDeque<Command>,
}

struct List {
    mode: u32,
    closed: bool,
    commands: VecDeque<Command>,
}

struct Object {
    kind: Kind,
    parent: Id,
    signaled: bool,
    size: usize,
}

#[derive(Debug, Default, Clone)]
pub(crate) struct Ledger {
    pub created: BTreeMap<Kind, usize>,
    pub released: BTreeMap<Kind, usize>,
    pub submitted: usize,
    pub completed: usize,
    pub discarded_recorded: usize,
    pub violations: Vec<&'static str>,
}

#[derive(Debug)]
pub(crate) struct Snapshot {
    pub ledger: Ledger,
    pub live: BTreeMap<Kind, usize>,
    pub pending: usize,
    pub recorded: usize,
    pub waiters: usize,
}

#[derive(Default)]
struct Driver {
    next: Id,
    objects: BTreeMap<Id, Object>,
    lists: BTreeMap<Id, List>,
    args: BTreeMap<Id, BTreeMap<u32, Id>>,
    pending: Vec<Submission>,
    ledger: Ledger,
    paused: bool,
    waiters: usize,
    execute_result: Option<Status>,
    /// Fake device timestamp counter and its increment per timestamp write.
    clock: u64,
    clock_step: u64,
    /// Module variables handed out by `zeModuleGetGlobalPointer`.
    globals: BTreeMap<(Id, String), Id>,
    global_size: Option<usize>,
    /// (driver, device handle, compute queue ordinal) per fake device.
    devices: Vec<(Id, Id, u32)>,
    /// Device-handle pairs that `zeDeviceCanAccessPeer` answers true for.
    peers: Vec<(Id, Id)>,
    /// Every memory copy appended: (device handle, destination, source, bytes).
    copies: Vec<(Id, usize, usize, usize)>,
    /// Device each command list was created for.
    list_device: BTreeMap<Id, Id>,
}

impl Driver {
    fn device_ordinal(&self, context: Raw, device: Raw) -> DriverResult<u32> {
        let parent = self.get(context.addr(), Kind::Context)?.parent;
        self.devices
            .iter()
            .find_map(|&(driver, handle, ordinal)| {
                (driver == parent && handle == device.addr()).then_some(ordinal)
            })
            .ok_or("native operation selected a foreign device")
    }

    fn create(&mut self, parent: Id, kind: Kind) -> Id {
        self.next += 0x10000;
        self.objects.insert(
            self.next,
            Object {
                kind,
                parent,
                signaled: false,
                size: 0,
            },
        );
        *self.ledger.created.entry(kind).or_default() += 1;
        self.next
    }

    fn get(&self, id: Id, kind: Kind) -> DriverResult<&Object> {
        self.objects
            .get(&id)
            .filter(|o| o.kind == kind)
            .ok_or("missing handle or wrong native resource kind")
    }

    fn pending_on(&self, id: Id) -> bool {
        self.pending.iter().any(|job| {
            job.list == id
                || job.queue == id
                || job.fence == id
                || job.commands.iter().any(|cmd| cmd.references(id))
        })
    }

    fn release(&mut self, id: Id, kind: Kind) -> DriverResult<()> {
        self.get(id, kind)?;
        if self.pending_on(id) {
            return Err("destroyed an in-flight resource");
        }
        if self.objects.values().any(|object| object.parent == id) {
            return Err("destroyed a native parent before its child");
        }
        if self
            .lists
            .values()
            .any(|list| list.commands.iter().any(|cmd| cmd.references(id)))
        {
            return Err("destroyed a resource referenced by a recorded command");
        }
        if let Some(list) = self.lists.remove(&id) {
            self.ledger.discarded_recorded += list.commands.len();
        }
        self.args.remove(&id);
        self.objects.remove(&id);
        *self.ledger.released.entry(kind).or_default() += 1;
        Ok(())
    }

    fn reset_signal(&mut self, id: Id, kind: Kind) -> DriverResult<Status> {
        self.get(id, kind)?;
        if self.pending_on(id)
            || self
                .lists
                .values()
                .any(|list| list.commands.iter().any(|cmd| cmd.references(id)))
        {
            return Err("reset a referenced event or fence");
        }
        self.objects.get_mut(&id).ok_or("lost signal")?.signaled = false;
        Ok(0)
    }

    fn allocation(&self, address: Id) -> Option<Id> {
        self.objects
            .range(..=address)
            .next_back()
            .and_then(|(&base, object)| (address - base < object.size).then_some(base))
    }

    fn append(&mut self, list: Id, command: Command) -> DriverResult<Status> {
        for id in command
            .waits
            .iter()
            .copied()
            .chain((command.signal != 0).then_some(command.signal))
        {
            self.get(id, Kind::Event)?;
        }
        let target = self.lists.get_mut(&list).ok_or("append to missing list")?;
        if target.closed {
            return Err("append to closed list");
        }
        if target.mode == 0 {
            target.commands.push_back(command);
        } else {
            let sync = target.mode == 1;
            self.submit(Submission {
                list,
                queue: 0,
                fence: 0,
                commands: [command].into(),
            });
            if sync && self.pending_on(list) {
                return Err("synchronous list could not complete");
            }
        }
        Ok(0)
    }

    fn submit(&mut self, job: Submission) {
        self.ledger.submitted += job.commands.len();
        self.pending.push(job);
        if !self.paused {
            self.complete_ready();
        }
    }

    fn signal(&mut self, id: Id) {
        if let Some(object) = self.objects.get_mut(&id) {
            object.signaled = true;
        }
    }

    fn complete_ready(&mut self) {
        while let Some(index) = self.pending.iter().enumerate().position(|(index, job)| {
            !self.pending[..index]
                .iter()
                .any(|prior| prior.list == job.list || (job.queue != 0 && prior.queue == job.queue))
                && job.commands.front().is_none_or(|cmd| {
                    cmd.waits
                        .iter()
                        .all(|wait| self.objects.get(wait).is_some_and(|obj| obj.signaled))
                })
        }) {
            if let Some(command) = self.pending[index].commands.pop_front() {
                if command.stamp != 0 {
                    // The destination is host memory owned by the caller of the append.
                    unsafe { (command.stamp as *mut u64).write_volatile(self.clock) };
                    self.clock = self.clock.wrapping_add(self.clock_step);
                }
                self.signal(command.signal);
                self.ledger.completed += 1;
            }
            if self.pending[index].commands.is_empty() {
                let job = self.pending.remove(index);
                self.signal(job.fence);
            }
        }
    }

    fn snapshot(&self) -> Snapshot {
        let mut live = BTreeMap::new();
        for object in self.objects.values() {
            *live.entry(object.kind).or_default() += 1;
        }
        Snapshot {
            ledger: self.ledger.clone(),
            live,
            pending: self.pending.len(),
            recorded: self.lists.values().map(|list| list.commands.len()).sum(),
            waiters: self.waiters,
        }
    }
}

fn lock() -> MutexGuard<'static, Option<Driver>> {
    DRIVER
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn call_driver(action: impl FnOnce(&mut Driver) -> DriverResult<Status>) -> Status {
    let mut guard = lock();
    let Some(driver) = guard.as_mut() else {
        return UNKNOWN;
    };
    let status = match action(driver) {
        Ok(status) => status,
        Err(issue) => {
            driver.ledger.violations.push(issue);
            UNKNOWN
        }
    };
    CHANGED.notify_all();
    status
}

pub(super) fn unexpected(call: &'static str) -> Status {
    call_driver(|_| Err(call))
}

pub(crate) fn start() -> Runtime {
    start_with(&[DeviceSpec::new(0, 0), DeviceSpec::new(1, 1)])
}

pub(crate) fn start_with(specs: &[DeviceSpec]) -> Runtime {
    // Fake handle ids restart at every start: drop argument-cache entries a previous test left
    // behind (for example by not destroying its context) so a reused id starts clean.
    super::arg_cache::forget_all();
    super::arg_cache::memory_freed();
    let table: Vec<(Id, Id, u32)> = specs
        .iter()
        .enumerate()
        .map(|(index, spec)| ids(index, spec.driver))
        .collect();
    let mut peers = Vec::new();
    for (a, x) in specs.iter().enumerate() {
        for (b, y) in specs.iter().enumerate() {
            if a != b && x.driver == y.driver && x.p2p_group.is_some() && x.p2p_group == y.p2p_group
            {
                peers.push((table[a].1, table[b].1));
            }
        }
    }
    assert!(
        lock()
            .replace(Driver {
                devices: table.clone(),
                peers,
                ..Driver::default()
            })
            .is_none()
    );
    let devices = specs
        .iter()
        .zip(table)
        .enumerate()
        .map(|(index, (spec, (driver, handle, ordinal)))| Device {
            info: DeviceInfo {
                name: spec.name.clone(),
                vendor_id: 0x8086,
                device_id: spec.device_id,
                device_flags: u32::from(spec.integrated),
                pci_address: spec.pci,
                driver_version: 0,
                api_version: 0x10000,
                uuid: spec.uuid,
                total_memory: spec.total_memory,
                max_allocation: spec.max_allocation,
                max_workgroup_size: spec.max_workgroup_size,
                max_workgroup_dimensions: [spec.max_workgroup_size, 16, 16],
                max_grid_dimensions: [65535; 3],
                max_shared_local_memory: spec.shared_local_memory,
                subgroup_sizes: spec.subgroup_sizes.clone(),
                spirv_version: 0x10000,
                fp32_denorms: true,
                fp32_round_to_nearest: true,
                fp32_inf_nan: true,
                fp32_fma: true,
                device_memory_atomics: true,
                module_flags: 0,
                floating_point_flags: [0, 0x27, if spec.fp64 { 0x3f } else { 0 }],
                float_atomic_flags: None,
                memory_access_flags: [1, 3, 0, 0, 0],
                memory_heaps: vec![],
                max_kernel_arguments_size: 1024,
                driver_extensions: vec![],
                queue_groups: vec![
                    QueueGroup {
                        ordinal: 3,
                        flags: 2,
                        queues: 1,
                    },
                    QueueGroup {
                        ordinal,
                        flags: 3,
                        queues: 2,
                    },
                ],
                // 10 MHz device timer with the 36-bit global counter of Intel GPUs.
                timestamp_frequency_hz: Some(FAKE_TIMER_HZ),
                timestamp_valid_bits: 36,
                kernel_timestamp_valid_bits: 32,
            },
            driver: Handle(without_provenance_mut(driver)),
            handle: Handle(without_provenance_mut(handle)),
            driver_index: spec.driver as u32,
            device_index: specs[..index]
                .iter()
                .filter(|other| other.driver == spec.driver)
                .count() as u32,
            probe: Default::default(),
        })
        .collect();
    Runtime {
        api: abi::api(),
        devices,
    }
}

/// True when `zeDeviceCanAccessPeer(device, peer)` is true in the fake topology.
fn peer_access(driver: &Driver, device: Id, peer: Id) -> bool {
    driver.peers.contains(&(device, peer))
}

/// See `abi::set_null_launch`.
pub(crate) fn set_null_launch(api: &mut Api, on: bool) {
    abi::set_null_launch(api, on);
}

/// Fake device handle of the device described by `specs[index]` in `start_with`.
pub(crate) fn device_handle(index: usize) -> usize {
    ids(index, 0).1
}

/// Memory copies appended so far: (device handle, destination, source, bytes).
pub(crate) fn copy_log() -> Vec<(usize, usize, usize, usize)> {
    lock().as_ref().map(|d| d.copies.clone()).unwrap_or_default()
}

pub(crate) fn finish() {
    lock().take();
}

pub(crate) fn snapshot() -> Snapshot {
    lock().as_ref().unwrap().snapshot()
}

pub(crate) fn pause() {
    lock().as_mut().unwrap().paused = true;
}

pub(crate) fn complete() {
    call_driver(|driver| {
        driver.paused = false;
        driver.complete_ready();
        Ok(0)
    });
}

/// Set the fake device clock and its advance (in ticks) per timestamp write.
pub(crate) fn set_clock(start: u64, step: u64) {
    let mut guard = lock();
    let driver = guard.as_mut().unwrap();
    driver.clock = start;
    driver.clock_step = step;
}

/// Override the size reported for module variables (`None` restores the default).
pub(crate) fn set_global_size(size: Option<usize>) {
    lock().as_mut().unwrap().global_size = size;
}

pub(crate) fn fail_after_execute(status: Status) {
    lock().as_mut().unwrap().execute_result = Some(status);
}

pub(crate) fn until(predicate: impl Fn(&Snapshot) -> bool) -> bool {
    let guard = lock();
    let (guard, _) = CHANGED
        .wait_timeout_while(guard, WAIT_LIMIT, |driver| {
            driver
                .as_ref()
                .is_some_and(|driver| !predicate(&driver.snapshot()))
        })
        .unwrap();
    guard
        .as_ref()
        .is_some_and(|driver| predicate(&driver.snapshot()))
}
