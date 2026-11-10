//! Level Zero virtual memory (`zeVirtualMem*`, `zePhysicalMem*`), the backing for the
//! CUDA VMM family (`cuMemCreate`, `cuMemAddressReserve`, `cuMemMap`, `cuMemSetAccess`).
//!
//! The entry points are optional: a loader without them leaves `Api::vmm` empty and the
//! CUDA layer keeps reporting `CUDA_ERROR_NOT_SUPPORTED`.
use super::{Context, Error, Handle, Header, Raw, Status, call};
use libloading::Library;
use std::ffi::c_void;

/// `ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC`.
const PHYSICAL_MEM_DESC: u32 = 0x20;

/// `ze_memory_access_attribute_t`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub(crate) enum Access {
    None = 0,
    ReadWrite = 1,
    ReadOnly = 2,
}

#[repr(C)]
pub(crate) struct PhysicalMemDesc {
    header: Header,
    /// `ze_physical_mem_flags_t`; 0 is device memory (`ALLOCATE_ON_DEVICE` is the default).
    pub(crate) flags: u32,
    pub(crate) size: usize,
}

/// `zeVirtualMemReserve(ctx, pStart, size, *pptr)`
pub(crate) type Reserve = unsafe extern "system" fn(Raw, *const c_void, usize, *mut Raw) -> Status;
/// `zeVirtualMemFree(ctx, ptr, size)`, `zeVirtualMemUnmap(ctx, ptr, size)`
pub(crate) type Range = unsafe extern "system" fn(Raw, *const c_void, usize) -> Status;
/// `zeVirtualMemQueryPageSize(ctx, device, size, *pagesize)`
pub(crate) type PageSize = unsafe extern "system" fn(Raw, Raw, usize, *mut usize) -> Status;
/// `zePhysicalMemCreate(ctx, device, desc, *phys)`
pub(crate) type PhysicalCreate =
    unsafe extern "system" fn(Raw, Raw, *const PhysicalMemDesc, *mut Raw) -> Status;
/// `zePhysicalMemDestroy(ctx, phys)`
pub(crate) type PhysicalDestroy = unsafe extern "system" fn(Raw, Raw) -> Status;
/// `zeVirtualMemMap(ctx, ptr, size, phys, offset, access)`
pub(crate) type Map =
    unsafe extern "system" fn(Raw, *const c_void, usize, Raw, usize, u32) -> Status;
/// `zeVirtualMemSetAccessAttribute(ctx, ptr, size, access)`
pub(crate) type SetAccess = unsafe extern "system" fn(Raw, *const c_void, usize, u32) -> Status;

#[derive(Clone, Copy)]
pub(crate) struct VmmApi {
    pub(crate) reserve: Reserve,
    pub(crate) free: Range,
    pub(crate) page_size: PageSize,
    pub(crate) physical_create: PhysicalCreate,
    pub(crate) physical_destroy: PhysicalDestroy,
    pub(crate) map: Map,
    pub(crate) unmap: Range,
    pub(crate) set_access: SetAccess,
}

impl VmmApi {
    /// All eight entry points, or `None` if the loader lacks any of them.
    pub(crate) fn load(library: &Library) -> Option<Self> {
        unsafe {
            Some(Self {
                reserve: *library.get::<Reserve>(b"zeVirtualMemReserve\0").ok()?,
                free: *library.get::<Range>(b"zeVirtualMemFree\0").ok()?,
                page_size: *library
                    .get::<PageSize>(b"zeVirtualMemQueryPageSize\0")
                    .ok()?,
                physical_create: *library
                    .get::<PhysicalCreate>(b"zePhysicalMemCreate\0")
                    .ok()?,
                physical_destroy: *library
                    .get::<PhysicalDestroy>(b"zePhysicalMemDestroy\0")
                    .ok()?,
                map: *library.get::<Map>(b"zeVirtualMemMap\0").ok()?,
                unmap: *library.get::<Range>(b"zeVirtualMemUnmap\0").ok()?,
                set_access: *library
                    .get::<SetAccess>(b"zeVirtualMemSetAccessAttribute\0")
                    .ok()?,
            })
        }
    }
}

fn address(ptr: usize) -> *const c_void {
    std::ptr::with_exposed_provenance(ptr)
}

/// VMM operations against one Level Zero context and its device.
impl Context {
    fn vmm(api: &super::Api) -> Result<VmmApi, Error> {
        api.vmm
            .ok_or(Error::Driver("zeVirtualMem entry points", 0x78000001))
    }

    /// `zeVirtualMemQueryPageSize`: the mapping granularity for a range of `size` bytes.
    pub(crate) fn vmm_page_size(&self, api: &super::Api, size: usize) -> Result<usize, Error> {
        let vmm = Self::vmm(api)?;
        let mut page = 0usize;
        call("zeVirtualMemQueryPageSize", || unsafe {
            (vmm.page_size)(self.handle.0, self.device.0, size, &mut page)
        })?;
        if page == 0 || !page.is_power_of_two() {
            return Err(Error::InvalidProperties);
        }
        Ok(page)
    }

    /// Reserve `size` bytes of virtual address space, at `hint` when the driver can.
    pub(crate) fn vmm_reserve(
        &self,
        api: &super::Api,
        hint: usize,
        size: usize,
    ) -> Result<usize, Error> {
        let vmm = Self::vmm(api)?;
        let mut out = std::ptr::null_mut();
        call("zeVirtualMemReserve", || unsafe {
            (vmm.reserve)(self.handle.0, address(hint), size, &mut out)
        })?;
        Ok(out.expose_provenance())
    }

    pub(crate) fn vmm_free(&self, api: &super::Api, base: usize, size: usize) -> Result<(), Error> {
        let vmm = Self::vmm(api)?;
        call("zeVirtualMemFree", || unsafe {
            (vmm.free)(self.handle.0, address(base), size)
        })
    }

    /// Device physical memory of `size` bytes (a multiple of the page size).
    pub(crate) fn physical_create(&self, api: &super::Api, size: usize) -> Result<Handle, Error> {
        let vmm = Self::vmm(api)?;
        let desc = PhysicalMemDesc {
            header: Header::new(PHYSICAL_MEM_DESC),
            flags: 0,
            size,
        };
        let mut out = std::ptr::null_mut();
        call("zePhysicalMemCreate", || unsafe {
            (vmm.physical_create)(self.handle.0, self.device.0, &desc, &mut out)
        })?;
        Ok(Handle(out))
    }

    pub(crate) fn physical_destroy(&self, api: &super::Api, memory: Handle) -> Result<(), Error> {
        let vmm = Self::vmm(api)?;
        call("zePhysicalMemDestroy", || unsafe {
            (vmm.physical_destroy)(self.handle.0, memory.0)
        })
    }

    pub(crate) fn vmm_map(
        &self,
        api: &super::Api,
        base: usize,
        size: usize,
        memory: Handle,
        offset: usize,
        access: Access,
    ) -> Result<(), Error> {
        let vmm = Self::vmm(api)?;
        call("zeVirtualMemMap", || unsafe {
            (vmm.map)(
                self.handle.0,
                address(base),
                size,
                memory.0,
                offset,
                access as u32,
            )
        })
    }

    pub(crate) fn vmm_unmap(
        &self,
        api: &super::Api,
        base: usize,
        size: usize,
    ) -> Result<(), Error> {
        let vmm = Self::vmm(api)?;
        call("zeVirtualMemUnmap", || unsafe {
            (vmm.unmap)(self.handle.0, address(base), size)
        })
    }

    pub(crate) fn vmm_set_access(
        &self,
        api: &super::Api,
        base: usize,
        size: usize,
        access: Access,
    ) -> Result<(), Error> {
        let vmm = Self::vmm(api)?;
        call("zeVirtualMemSetAccessAttribute", || unsafe {
            (vmm.set_access)(self.handle.0, address(base), size, access as u32)
        })
    }
}
