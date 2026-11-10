//! Fake `zeVirtualMem*` / `zePhysicalMem*`: address bookkeeping with the driver's
//! preconditions (page alignment, mappings inside a reservation, no overlap, no
//! destroying mapped memory), so the CUDA layer's misuse shows up as a violation.
use super::*;
use crate::level_zero::vmm::{PhysicalMemDesc, VmmApi};

/// Page size the fake reports for every query.
pub(crate) const FAKE_PAGE: usize = 64 * 1024;
/// First virtual address handed out; far above the object ids used as fake pointers.
const VA_BASE: usize = 1 << 40;

#[derive(Default)]
pub(super) struct VmmState {
    next_va: usize,
    /// Reservation base -> (context, size).
    reservations: BTreeMap<usize, (Id, usize)>,
    /// Mapping base -> (size, physical memory, offset, access).
    mappings: BTreeMap<usize, (usize, Id, usize, u32)>,
    /// Physical memory id -> size.
    physical: BTreeMap<Id, usize>,
}

/// What the fake driver holds, for assertions.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct VmmSnapshot {
    pub reservations: Vec<(usize, usize)>,
    /// (base, size, physical id, offset, access)
    pub mappings: Vec<(usize, usize, usize, usize, u32)>,
    pub physical: Vec<(usize, usize)>,
}

pub(crate) fn vmm_snapshot() -> VmmSnapshot {
    lock()
        .as_ref()
        .map(|driver| {
            let v = &driver.vmm;
            VmmSnapshot {
                reservations: v.reservations.iter().map(|(&b, &(_, s))| (b, s)).collect(),
                mappings: v
                    .mappings
                    .iter()
                    .map(|(&b, &(s, p, o, a))| (b, s, p, o, a))
                    .collect(),
                physical: v.physical.iter().map(|(&p, &s)| (p, s)).collect(),
            }
        })
        .unwrap_or_default()
}

fn aligned(value: usize) -> bool {
    value.is_multiple_of(FAKE_PAGE)
}

impl VmmState {
    fn reservation_of(&self, base: usize, size: usize) -> Option<(usize, Id)> {
        let (&start, &(context, length)) = self.reservations.range(..=base).next_back()?;
        (base - start < length && size <= length - (base - start)).then_some((start, context))
    }

    /// Mappings overlapping [base, base + size).
    fn overlapping(&self, base: usize, size: usize) -> Vec<(usize, usize)> {
        let end = base + size;
        self.mappings
            .range(..end)
            .filter(|&(&start, &(length, ..))| start + length > base)
            .map(|(&start, &(length, ..))| (start, length))
            .collect()
    }

    /// True when [base, base + size) is covered by mappings without gaps.
    fn fully_mapped(&self, base: usize, size: usize) -> bool {
        let mut cursor = base;
        for (start, length) in self.overlapping(base, size) {
            if start > cursor {
                return false;
            }
            cursor = cursor.max(start + length);
        }
        cursor >= base + size
    }
}

unsafe extern "system" fn reserve(
    ctx: Raw,
    start: *const c_void,
    size: usize,
    out: *mut Raw,
) -> Status {
    call_driver(|driver| {
        driver.get(ctx.addr(), Kind::Context)?;
        if size == 0 || !aligned(size) {
            return Err("zeVirtualMemReserve: unaligned size");
        }
        let v = &mut driver.vmm;
        if v.next_va == 0 {
            v.next_va = VA_BASE;
        }
        // Honour the hint when it is free, like the real driver may.
        let hint = start.addr();
        let free = |v: &VmmState, at: usize| {
            aligned(at)
                && !v
                    .reservations
                    .iter()
                    .any(|(&b, &(_, s))| at < b + s && b < at + size)
        };
        let base = if hint != 0 && free(v, hint) {
            hint
        } else {
            v.next_va
        };
        v.next_va = v.next_va.max(base + size) + FAKE_PAGE; // guard page between reservations
        v.reservations.insert(base, (ctx.addr(), size));
        unsafe { out.write(without_provenance_mut(base)) };
        Ok(0)
    })
}

unsafe extern "system" fn free(ctx: Raw, ptr: *const c_void, size: usize) -> Status {
    call_driver(|driver| {
        let v = &mut driver.vmm;
        match v.reservations.get(&ptr.addr()) {
            Some(&(context, length)) if context == ctx.addr() && length == size => {}
            _ => return Err("zeVirtualMemFree: not a reservation of this context"),
        }
        if !v.overlapping(ptr.addr(), size).is_empty() {
            return Err("zeVirtualMemFree: range still mapped");
        }
        v.reservations.remove(&ptr.addr());
        Ok(0)
    })
}

unsafe extern "system" fn page_size(ctx: Raw, device: Raw, _: usize, out: *mut usize) -> Status {
    call_driver(|driver| {
        driver.device_ordinal(ctx, device)?;
        unsafe { out.write(FAKE_PAGE) };
        Ok(0)
    })
}

unsafe extern "system" fn physical_create(
    ctx: Raw,
    device: Raw,
    desc: *const PhysicalMemDesc,
    out: *mut Raw,
) -> Status {
    call_driver(|driver| {
        driver.device_ordinal(ctx, device)?;
        let (flags, size) = unsafe { ((*desc).flags, (*desc).size) };
        if flags != 0 {
            return Err("zePhysicalMemCreate: expected device memory");
        }
        if size == 0 || !aligned(size) {
            return Err("zePhysicalMemCreate: unaligned size");
        }
        let id = driver.create(ctx.addr(), Kind::PhysicalMemory);
        driver.vmm.physical.insert(id, size);
        unsafe { out.write(without_provenance_mut(id)) };
        Ok(0)
    })
}

unsafe extern "system" fn physical_destroy(ctx: Raw, memory: Raw) -> Status {
    call_driver(|driver| {
        let id = memory.addr();
        if driver.get(id, Kind::PhysicalMemory)?.parent != ctx.addr() {
            return Err("zePhysicalMemDestroy: foreign context");
        }
        if driver.vmm.mappings.values().any(|&(_, p, ..)| p == id) {
            return Err("zePhysicalMemDestroy: memory still mapped");
        }
        driver.vmm.physical.remove(&id);
        driver.release(id, Kind::PhysicalMemory)?;
        Ok(0)
    })
}

unsafe extern "system" fn map(
    ctx: Raw,
    ptr: *const c_void,
    size: usize,
    memory: Raw,
    offset: usize,
    access: u32,
) -> Status {
    call_driver(|driver| {
        let (base, id) = (ptr.addr(), memory.addr());
        if size == 0 || !aligned(base) || !aligned(size) || !aligned(offset) || access > 2 {
            return Err("zeVirtualMemMap: bad alignment or access");
        }
        let v = &mut driver.vmm;
        let physical = *v
            .physical
            .get(&id)
            .ok_or("zeVirtualMemMap: missing physical memory")?;
        if offset > physical || size > physical - offset {
            return Err("zeVirtualMemMap: range beyond physical memory");
        }
        match v.reservation_of(base, size) {
            Some((_, context)) if context == ctx.addr() => {}
            _ => return Err("zeVirtualMemMap: outside this context's reservations"),
        }
        if !v.overlapping(base, size).is_empty() {
            return Err("zeVirtualMemMap: range already mapped");
        }
        v.mappings.insert(base, (size, id, offset, access));
        Ok(0)
    })
}

unsafe extern "system" fn unmap(ctx: Raw, ptr: *const c_void, size: usize) -> Status {
    call_driver(|driver| {
        let base = ptr.addr();
        let v = &mut driver.vmm;
        if v.reservation_of(base, size)
            .is_none_or(|(_, c)| c != ctx.addr())
        {
            return Err("zeVirtualMemUnmap: outside this context's reservations");
        }
        let hits = v.overlapping(base, size);
        if hits.iter().any(|&(s, l)| s < base || s + l > base + size) {
            return Err("zeVirtualMemUnmap: partial mapping");
        }
        for (start, _) in hits {
            v.mappings.remove(&start);
        }
        Ok(0)
    })
}

unsafe extern "system" fn set_access(
    ctx: Raw,
    ptr: *const c_void,
    size: usize,
    access: u32,
) -> Status {
    call_driver(|driver| {
        let base = ptr.addr();
        let v = &mut driver.vmm;
        if access > 2
            || v.reservation_of(base, size)
                .is_none_or(|(_, c)| c != ctx.addr())
        {
            return Err("zeVirtualMemSetAccessAttribute: bad range or access");
        }
        if !v.fully_mapped(base, size) {
            return Err("zeVirtualMemSetAccessAttribute: unmapped pages");
        }
        for (start, _) in v.overlapping(base, size) {
            if let Some(entry) = v.mappings.get_mut(&start) {
                entry.3 = access;
            }
        }
        Ok(0)
    })
}

pub(super) fn api() -> VmmApi {
    VmmApi {
        reserve,
        free,
        page_size,
        physical_create,
        physical_destroy,
        map,
        unmap,
        set_access,
    }
}
