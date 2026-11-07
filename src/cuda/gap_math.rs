//! Hardware-independent logic behind the Driver API gap entry points.
//!
//! This file depends only on `std` so it can be unit tested anywhere.
use std::collections::BTreeMap;

/// Device limits that bound occupancy, as CUDA reports them per multiprocessor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Limits {
    pub max_threads_per_block: u32,
    pub max_threads_per_mp: u32,
    pub max_blocks_per_mp: u32,
    pub shared_per_mp: u32,
    pub shared_per_block: u32,
    pub warp_size: u32,
    pub multiprocessors: u32,
}

/// Resident blocks per multiprocessor for one launch shape. Register pressure
/// is not modelled because the backend does not expose it.
pub(crate) fn active_blocks(limits: Limits, block: u32, shared: u64) -> u32 {
    if block == 0 || block > limits.max_threads_per_block || shared > limits.shared_per_block as u64
    {
        return 0;
    }
    let by_threads = limits.max_threads_per_mp / block;
    let by_shared = if shared == 0 {
        u32::MAX
    } else {
        (limits.shared_per_mp as u64 / shared).min(u32::MAX as u64) as u32
    };
    by_threads.min(limits.max_blocks_per_mp).min(by_shared)
}

/// Block size maximizing resident threads, and the grid that fills the device.
/// `shared_for` returns dynamic shared bytes for a candidate block size.
/// A zero `limit` means no limit. Ties favor the larger block.
pub(crate) fn potential_block_size(
    limits: Limits,
    limit: u32,
    shared_for: impl Fn(u32) -> u64,
) -> (u32, u32) {
    let cap = if limit == 0 {
        limits.max_threads_per_block
    } else {
        limit.min(limits.max_threads_per_block)
    };
    let step = limits.warp_size.max(1);
    let mut best = (0, 0, 0); // (resident threads, block, blocks per mp)
    let mut candidate = cap / step * step;
    // Blocks smaller than one warp are only considered when the cap is smaller.
    if candidate == 0 {
        candidate = cap;
    }
    loop {
        if candidate == 0 {
            break;
        }
        let blocks = active_blocks(limits, candidate, shared_for(candidate));
        let threads = blocks * candidate;
        if threads > best.0 {
            best = (threads, candidate, blocks);
        }
        if candidate <= step {
            break;
        }
        candidate -= step;
    }
    (best.2 * limits.multiprocessors, best.1)
}

/// Base and size of the allocation containing `address`.
pub(crate) fn range_of(map: &BTreeMap<usize, usize>, address: usize) -> Option<(usize, usize)> {
    let (&base, &size) = map.range(..=address).next_back()?;
    (address - base < size).then_some((base, size))
}

/// Whether `[start, start + size)` overlaps any range in `map`.
pub(crate) fn overlaps(map: &BTreeMap<usize, usize>, start: usize, size: usize) -> bool {
    let Some(end) = start.checked_add(size) else {
        return true;
    };
    if let Some((&base, &len)) = map.range(..start).next_back()
        && base.saturating_add(len) > start
    {
        return true;
    }
    map.range(start..end).next().is_some()
}

/// CUDA_MEMCPY2D/3D memory types.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MemoryType {
    Host,
    Device,
    Array,
    Unified,
}

impl MemoryType {
    pub(crate) fn from_raw(raw: u32) -> Option<Self> {
        match raw {
            1 => Some(Self::Host),
            2 => Some(Self::Device),
            3 => Some(Self::Array),
            4 => Some(Self::Unified),
            _ => None,
        }
    }
}

/// One side of a strided copy. `offset` is the byte offset of element (x, y, z)
/// from the base pointer of that side.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Side {
    pub x_bytes: usize,
    pub y: usize,
    pub z: usize,
    pub pitch: usize,
    /// Rows per plane. Ignored for 2D copies.
    pub height: usize,
}

impl Side {
    fn row_offset(&self, row: usize, plane: usize) -> Option<usize> {
        let rows = plane.checked_mul(self.height)?.checked_add(self.y)?;
        let rows = rows.checked_add(row)?;
        rows.checked_mul(self.pitch)?
            .checked_add(self.z.checked_mul(self.height)?.checked_mul(self.pitch)?)?
            .checked_add(self.x_bytes)
    }
}

/// A validated strided copy, iterated one contiguous row at a time.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Plan {
    pub src: Side,
    pub dst: Side,
    pub width: usize,
    pub rows: usize,
    pub planes: usize,
}

impl Plan {
    /// Rejects pitches narrower than the row, and offsets that overflow.
    pub(crate) fn new(
        src: Side,
        dst: Side,
        width: usize,
        rows: usize,
        planes: usize,
    ) -> Option<Self> {
        let plan = Self {
            src,
            dst,
            width,
            rows,
            planes,
        };
        if width == 0 || rows == 0 || planes == 0 {
            return Some(plan);
        }
        for side in [src, dst] {
            // A single row needs no pitch; more rows must not overlap.
            if (rows > 1 || planes > 1) && side.pitch < side.x_bytes.checked_add(width)? {
                return None;
            }
            if planes > 1 && side.height < side.y.checked_add(rows)? {
                return None;
            }
        }
        plan.row(rows - 1, planes - 1)?;
        Some(plan)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.width == 0 || self.rows == 0 || self.planes == 0
    }

    /// (source offset, destination offset) of a row, from each side's base.
    pub(crate) fn row(&self, row: usize, plane: usize) -> Option<(usize, usize)> {
        Some((
            self.src.row_offset(row, plane)?,
            self.dst.row_offset(row, plane)?,
        ))
    }

    /// Byte span `[start, start + len)` touched on one side (true for source).
    pub(crate) fn span(&self, source: bool) -> Option<(usize, usize)> {
        let side = if source { &self.src } else { &self.dst };
        let first = side.row_offset(0, 0)?;
        let (last, _) = if source {
            self.row(self.rows - 1, self.planes - 1)?
        } else {
            let (_, last) = self.row(self.rows - 1, self.planes - 1)?;
            (last, 0)
        };
        Some((first, last.checked_add(self.width)? - first))
    }
}

/// Bytes `pattern` repeats to, if every byte of the expanded value is equal.
/// Lets 16/32-bit memsets use the byte fill the backend provides.
pub(crate) fn uniform_byte(value: u32, width: usize) -> Option<u8> {
    let bytes = value.to_le_bytes();
    let used = &bytes[..width.min(4)];
    used.iter().all(|b| *b == used[0]).then_some(used[0])
}

/// Expand a 16/32-bit pattern across `count` elements (little endian).
pub(crate) fn expand_pattern(value: u32, width: usize, count: usize) -> Vec<u8> {
    let bytes = value.to_le_bytes();
    let mut out = Vec::with_capacity(width * count);
    for _ in 0..count {
        out.extend_from_slice(&bytes[..width]);
    }
    out
}

/// What cuFuncGetAttribute knows about a function.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FuncFacts {
    pub max_threads: u32,
    pub max_dynamic_shared: u32,
    pub device_max_shared: u32,
}

pub(crate) const PTX_AND_BINARY_VERSION: i32 = 75;
pub(crate) const ATTR_MAX_DYNAMIC_SHARED: i32 = 8;
pub(crate) const ATTR_CARVEOUT: i32 = 9;

pub(crate) fn function_attribute(attr: i32, facts: FuncFacts) -> Option<i32> {
    Some(match attr {
        0 => facts.max_threads.min(i32::MAX as u32) as i32,
        // Static shared, constant, local sizes are not tracked by the translator.
        1..=3 => 0,
        // The translator does not allocate registers; report a stable stand-in.
        4 => 32,
        5 | 6 => PTX_AND_BINARY_VERSION,
        7 => 0,
        ATTR_MAX_DYNAMIC_SHARED => facts.max_dynamic_shared.min(i32::MAX as u32) as i32,
        ATTR_CARVEOUT => -1,
        // Cluster launch is unsupported: required/preferred cluster size 0.
        10..=15 => 0,
        _ => return None,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SetAttribute {
    Accepted,
    Invalid,
    Unsupported,
}

pub(crate) fn set_function_attribute(attr: i32, value: i32, facts: FuncFacts) -> SetAttribute {
    match attr {
        ATTR_MAX_DYNAMIC_SHARED
            if (0..=facts.device_max_shared as i64).contains(&(value as i64)) =>
        {
            SetAttribute::Accepted
        }
        ATTR_MAX_DYNAMIC_SHARED => SetAttribute::Invalid,
        ATTR_CARVEOUT if (-1..=100).contains(&value) => SetAttribute::Accepted,
        ATTR_CARVEOUT => SetAttribute::Invalid,
        10..=14 if value == 0 => SetAttribute::Accepted,
        // 15 is NON_PORTABLE_CLUSTER_SIZE_ALLOWED, a permission, not a request.
        15 => SetAttribute::Accepted,
        10..=14 => SetAttribute::Unsupported,
        _ => SetAttribute::Invalid,
    }
}

/// How cuLaunchKernelEx treats one launch attribute.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LaunchAttribute {
    /// Hint, or a constraint satisfied by ordinary in-order execution.
    Ignore,
    Unsupported,
}

/// `cluster` holds the first three words of the attribute value union.
pub(crate) fn launch_attribute(id: u32, value: [u32; 3]) -> LaunchAttribute {
    match id {
        // IGNORE, ACCESS_POLICY_WINDOW, SYNCHRONIZATION_POLICY.
        0 | 1 | 3 => LaunchAttribute::Ignore,
        // COOPERATIVE needs grid-wide forward-progress guarantees.
        2 => LaunchAttribute::Unsupported,
        4 if value == [1, 1, 1] => LaunchAttribute::Ignore,
        4 => LaunchAttribute::Unsupported,
        // Cluster scheduling hint, programmatic stream serialization (stricter
        // ordering is always legal), priority, memory sync domain hints.
        5 | 6 | 8..=10 => LaunchAttribute::Ignore,
        // Programmatic event, launch completion event, device-updatable
        // kernel node, and anything newer.
        _ => LaunchAttribute::Unsupported,
    }
}

/// CUpointer_attribute values the shim answers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Kind {
    Unknown,
    Device,
    Host,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PointerFacts {
    pub kind: Kind,
    pub pointer: usize,
    pub base: usize,
    pub size: usize,
    pub context: usize,
    pub ordinal: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Value {
    U32(u32),
    U64(u64),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PointerError {
    Invalid,
    Unsupported,
}

pub(crate) fn pointer_attribute(attr: i32, facts: &PointerFacts) -> Result<Value, PointerError> {
    use PointerError::*;
    use Value::*;
    let known = facts.kind != Kind::Unknown;
    Ok(match attr {
        1 if known => U64(facts.context as u64), // CONTEXT
        2 => U32(match facts.kind {
            // MEMORY_TYPE
            Kind::Host => 1,
            Kind::Device => 2,
            Kind::Unknown => return Err(Invalid),
        }),
        3 if facts.kind == Kind::Device => U64(facts.pointer as u64), // DEVICE_POINTER
        3 if known => return Err(Unsupported),
        4 if facts.kind == Kind::Host => U64(facts.pointer as u64), // HOST_POINTER
        4 if known => return Err(Invalid),
        6 => U32(0),                                             // SYNC_MEMOPS
        7 | 20 if known => U64(facts.base as u64),               // BUFFER_ID, MEMORY_BLOCK_ID
        8 | 10 | 15 if known => U32(0), // IS_MANAGED, legacy IPC, GPUDirect RDMA
        9 if known => U32(facts.ordinal as u32), // DEVICE_ORDINAL
        11 if known => U64(facts.base as u64), // RANGE_START_ADDR
        12 if known => U64(facts.size as u64), // RANGE_SIZE
        13 if known => U32((facts.kind == Kind::Device) as u32), // MAPPED
        14 if known => U32(0),          // ALLOWED_HANDLE_TYPES
        16 if known => U32(3),          // ACCESS_FLAGS: read and write
        17 if known => U64(0),          // MEMPOOL_HANDLE
        18 | 19 if known => return Err(Invalid), // VMM mapping only
        1..=20 if !known => return Err(Invalid),
        _ => return Err(Invalid),
    })
}

/// Default value for an unregistered pointer in cuPointerGetAttributes.
pub(crate) fn pointer_default(attr: i32) -> Option<Value> {
    Some(match attr {
        1 | 3 | 4 | 7 | 11 | 12 | 17 | 20 => Value::U64(0),
        2 | 6 | 8 | 9 | 10 | 13 | 14 | 15 | 16 => Value::U32(0),
        _ => return None,
    })
}

/// Pool handles are synthetic: one default pool per device ordinal.
pub(crate) const POOL_BASE: usize = 1 << 40;

pub(crate) fn pool_handle(ordinal: usize) -> usize {
    POOL_BASE + ordinal + 1
}

pub(crate) fn pool_ordinal(handle: usize) -> Option<usize> {
    handle
        .checked_sub(POOL_BASE + 1)
        .filter(|_| handle > POOL_BASE)
}

/// Mempool attribute defaults, `None` for attributes the shim cannot report.
pub(crate) fn pool_default(attr: i32) -> Option<u64> {
    match attr {
        1..=3 => Some(1),
        4 => Some(0),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const L: Limits = Limits {
        max_threads_per_block: 1024,
        max_threads_per_mp: 1024,
        max_blocks_per_mp: 16,
        shared_per_mp: 65536,
        shared_per_block: 49152,
        warp_size: 32,
        multiprocessors: 8,
    };

    #[test]
    fn occupancy_is_bounded_by_threads_blocks_and_shared() {
        assert_eq!(active_blocks(L, 256, 0), 4);
        assert_eq!(active_blocks(L, 32, 0), 16);
        assert_eq!(active_blocks(L, 256, 32768), 2);
        assert_eq!(active_blocks(L, 256, 49153), 0);
        assert_eq!(active_blocks(L, 0, 0), 0);
        assert_eq!(active_blocks(L, 2048, 0), 0);
    }

    #[test]
    fn potential_block_size_maximizes_resident_threads() {
        let (grid, block) = potential_block_size(L, 0, |_| 0);
        assert_eq!(block, 1024);
        assert_eq!(grid, 8);
        // Shared memory scaling with block size favors a smaller block.
        let (_, block) = potential_block_size(L, 0, |b| b as u64 * 128);
        assert_eq!(active_blocks(L, block, block as u64 * 128) * block, 512);
        let (grid, block) = potential_block_size(L, 100, |_| 0);
        assert_eq!(block, 64); // 96 only fits 10 blocks (960 threads); 64 fits 16
        assert_eq!(grid, 8 * 16);
        // A limit below one warp is honored.
        assert_eq!(potential_block_size(L, 8, |_| 0).1, 8);
        // Unlaunchable (too much shared) reports zero.
        assert_eq!(potential_block_size(L, 0, |_| 1 << 30), (0, 0));
    }

    #[test]
    fn address_ranges() {
        let map = BTreeMap::from([(100, 10), (200, 50)]);
        assert_eq!(range_of(&map, 105), Some((100, 10)));
        assert_eq!(range_of(&map, 110), None);
        assert_eq!(range_of(&map, 99), None);
        assert_eq!(range_of(&map, 249), Some((200, 50)));
        assert!(overlaps(&map, 105, 1));
        assert!(overlaps(&map, 90, 11));
        assert!(overlaps(&map, 199, 2));
        assert!(!overlaps(&map, 110, 90));
        assert!(!overlaps(&map, 250, 10));
        assert!(overlaps(&map, usize::MAX, 2));
    }

    fn side(pitch: usize) -> Side {
        Side {
            x_bytes: 0,
            y: 0,
            z: 0,
            pitch,
            height: 0,
        }
    }

    #[test]
    fn copy_2d_rows_follow_pitch() {
        let mut src = side(64);
        src.x_bytes = 4;
        src.y = 2;
        let plan = Plan::new(src, side(16), 16, 3, 1).unwrap();
        assert_eq!(plan.row(0, 0), Some((2 * 64 + 4, 0)));
        assert_eq!(plan.row(2, 0), Some((4 * 64 + 4, 32)));
        assert_eq!(plan.span(true), Some((132, 2 * 64 + 16)));
        assert_eq!(plan.span(false), Some((0, 2 * 16 + 16)));
        // Pitch narrower than the row would overlap rows.
        assert!(Plan::new(src, side(8), 16, 3, 1).is_none());
        // A single row needs no pitch.
        assert!(Plan::new(side(0), side(0), 16, 1, 1).is_some());
        // Overflow is rejected rather than wrapped.
        assert!(Plan::new(side(usize::MAX / 2), side(usize::MAX / 2), 16, 4, 1).is_none());
        assert!(Plan::new(side(8), side(8), 0, 5, 1).unwrap().is_empty());
    }

    #[test]
    fn copy_3d_planes_use_height() {
        let mk = |pitch, height| Side {
            x_bytes: 0,
            y: 0,
            z: 1,
            pitch,
            height,
        };
        let plan = Plan::new(mk(32, 4), mk(16, 2), 16, 2, 3).unwrap();
        // z = 1 skips one full plane (height * pitch) on each side.
        assert_eq!(plan.row(0, 0), Some((4 * 32, 2 * 16)));
        assert_eq!(
            plan.row(1, 2),
            Some(((2 * 4 + 1 + 4) * 32, (2 * 2 + 1 + 2) * 16))
        );
        // Height smaller than the rows per plane overlaps planes.
        assert!(Plan::new(mk(32, 1), mk(16, 2), 16, 2, 3).is_none());
    }

    #[test]
    fn memset_patterns() {
        assert_eq!(uniform_byte(0xabab, 2), Some(0xab));
        assert_eq!(uniform_byte(0xab01, 2), None);
        assert_eq!(uniform_byte(0x0101_0101, 4), Some(1));
        assert_eq!(uniform_byte(0x0001_0101, 4), None);
        assert_eq!(expand_pattern(0x0201, 2, 2), [1, 2, 1, 2]);
        assert_eq!(expand_pattern(0x04030201, 4, 1), [1, 2, 3, 4]);
    }

    #[test]
    fn function_attributes() {
        let f = FuncFacts {
            max_threads: 512,
            max_dynamic_shared: 1000,
            device_max_shared: 65536,
        };
        assert_eq!(function_attribute(0, f), Some(512));
        assert_eq!(function_attribute(8, f), Some(1000));
        assert_eq!(function_attribute(9, f), Some(-1));
        assert_eq!(function_attribute(11, f), Some(0));
        assert_eq!(function_attribute(99, f), None);
        assert_eq!(set_function_attribute(8, 65536, f), SetAttribute::Accepted);
        assert_eq!(set_function_attribute(8, 65537, f), SetAttribute::Invalid);
        assert_eq!(set_function_attribute(8, -1, f), SetAttribute::Invalid);
        assert_eq!(set_function_attribute(9, 100, f), SetAttribute::Accepted);
        assert_eq!(set_function_attribute(9, 101, f), SetAttribute::Invalid);
        assert_eq!(set_function_attribute(11, 2, f), SetAttribute::Unsupported);
        assert_eq!(set_function_attribute(0, 1, f), SetAttribute::Invalid);
    }

    #[test]
    fn launch_attributes() {
        use LaunchAttribute::*;
        assert_eq!(launch_attribute(0, [0; 3]), Ignore);
        assert_eq!(launch_attribute(2, [1; 3]), Unsupported);
        assert_eq!(launch_attribute(4, [1, 1, 1]), Ignore);
        assert_eq!(launch_attribute(4, [2, 1, 1]), Unsupported);
        assert_eq!(launch_attribute(6, [1, 0, 0]), Ignore);
        assert_eq!(launch_attribute(7, [0; 3]), Unsupported);
        assert_eq!(launch_attribute(9999, [0; 3]), Unsupported);
    }

    #[test]
    fn pointer_attributes() {
        let device = PointerFacts {
            kind: Kind::Device,
            pointer: 0x1010,
            base: 0x1000,
            size: 64,
            context: 7,
            ordinal: 1,
        };
        assert_eq!(pointer_attribute(2, &device), Ok(Value::U32(2)));
        assert_eq!(pointer_attribute(3, &device), Ok(Value::U64(0x1010)));
        assert_eq!(pointer_attribute(4, &device), Err(PointerError::Invalid));
        assert_eq!(pointer_attribute(11, &device), Ok(Value::U64(0x1000)));
        assert_eq!(pointer_attribute(12, &device), Ok(Value::U64(64)));
        assert_eq!(pointer_attribute(9, &device), Ok(Value::U32(1)));
        assert_eq!(pointer_attribute(1, &device), Ok(Value::U64(7)));
        assert_eq!(pointer_attribute(8, &device), Ok(Value::U32(0)));
        assert_eq!(pointer_attribute(18, &device), Err(PointerError::Invalid));
        let host = PointerFacts {
            kind: Kind::Host,
            ..device
        };
        assert_eq!(pointer_attribute(2, &host), Ok(Value::U32(1)));
        assert_eq!(pointer_attribute(4, &host), Ok(Value::U64(0x1010)));
        assert_eq!(pointer_attribute(3, &host), Err(PointerError::Unsupported));
        let unknown = PointerFacts {
            kind: Kind::Unknown,
            ..device
        };
        for attr in [1, 2, 3, 9, 11] {
            assert_eq!(
                pointer_attribute(attr, &unknown),
                Err(PointerError::Invalid)
            );
        }
        assert_eq!(pointer_default(2), Some(Value::U32(0)));
        assert_eq!(pointer_default(11), Some(Value::U64(0)));
        assert_eq!(pointer_default(99), None);
    }

    #[test]
    fn pool_handles_round_trip() {
        for ordinal in [0, 1, 7] {
            assert_eq!(pool_ordinal(pool_handle(ordinal)), Some(ordinal));
        }
        assert_eq!(pool_ordinal(0), None);
        assert_eq!(pool_ordinal(POOL_BASE), None);
        assert_eq!(pool_default(4), Some(0));
        assert_eq!(pool_default(7), None);
    }
}
