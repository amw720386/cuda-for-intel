//! Module-scope variables: `cuModuleGetGlobal`, `cuLibraryGetGlobal` and friends.
//!
//! The PTX translator exports every `.global` and `.const` variable from the
//! SPIR-V module by name. The address is resolved lazily with
//! `zeModuleGetGlobalPointer` and then registered as a device range of the
//! context, so `cuMemcpyHtoD`/`DtoH`/`DtoD` and memsets accept it like any
//! allocation. The range is owned by the module: `cuMemFree` rejects it and
//! unloading the module retires it.
use super::*;

/// One exported variable of a loaded module.
pub(super) struct ModuleGlobal {
    /// Size of the PTX object in bytes (what CUDA reports).
    pub(super) size: usize,
    /// Device address once resolved.
    pub(super) address: Option<usize>,
}

pub(super) fn from_translation(
    globals: &[crate::ptx::globals::GlobalVar],
) -> Result<HashMap<CString, ModuleGlobal>, CuResult> {
    globals
        .iter()
        .map(|var| {
            let name = CString::new(var.name.as_str()).map_err(|_| InvalidPtx)?;
            Ok((
                name,
                ModuleGlobal {
                    size: var.size as usize,
                    address: None,
                },
            ))
        })
        .collect()
}

impl CudaContext {
    /// Device address and size of `name` in a loaded module.
    pub(super) fn module_global(
        &mut self,
        api: &Api,
        module_key: usize,
        name: &CStr,
    ) -> Result<(usize, usize), CuResult> {
        let module = self.modules.get_mut(&module_key).ok_or(InvalidHandle)?;
        if module.unloading {
            return Err(InvalidHandle);
        }
        let global = module.globals.get_mut(name).ok_or(NotFound)?;
        if let Some(address) = global.address {
            return Ok((address, global.size));
        }
        let (address, size) = module
            .backend
            .global_pointer(api, name)
            .inspect_err(|error| self.backend.health.observe(error))
            .map_err(backend_error)?;
        // A module that does not export what the translator declared is corrupt.
        if address == 0 || size < global.size {
            return Err(InvalidImage);
        }
        global.address = Some(address);
        self.allocations.insert(address, global.size);
        self.module_globals.insert(address);
        Ok((address, global.size))
    }

    /// Retire every global range of a module that is being unloaded.
    pub(super) fn retire_module_globals(&mut self, module_key: usize) {
        if let Some(module) = self.modules.get(&module_key) {
            let addresses: Vec<usize> = module.globals.values().filter_map(|global| global.address).collect();
            for address in addresses {
                if let Some(length) = self.allocations.remove(&address) {
                    self.note_free(address, length);
                }
                self.module_globals.remove(&address);
            }
        }
    }
}

fn write_outputs(dptr: *mut u64, bytes: *mut usize, (address, size): (usize, usize)) {
    if !dptr.is_null() {
        unsafe { dptr.write(address as u64) };
    }
    if !bytes.is_null() {
        unsafe { bytes.write(size) };
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuModuleGetGlobal_v2(
    dptr: *mut u64,
    bytes: *mut usize,
    module: Opaque,
    name: *const c_char,
) -> CuResult {
    trace::call("cuModuleGetGlobal_v2", || {
        if name.is_null() {
            return InvalidValue;
        }
        let name = unsafe { CStr::from_ptr(name) };
        status(with_context(|runtime, context| {
            let found = context.module_global(&runtime.api, module.addr(), name)?;
            write_outputs(dptr, bytes, found);
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuLibraryGetGlobal(
    dptr: *mut u64,
    bytes: *mut usize,
    library: Opaque,
    name: *const c_char,
) -> CuResult {
    trace::call("cuLibraryGetGlobal", || {
        if name.is_null() {
            return InvalidValue;
        }
        let name = unsafe { CStr::from_ptr(name) };
        status(state(|state| {
            let (context_key, module) = library::module_for_library(state, library.addr())?;
            let context = state
                .contexts
                .get_mut(&context_key)
                .ok_or(ContextIsDestroyed)?;
            let found = context.module_global(&state.runtime.api, module, name)?;
            write_outputs(dptr, bytes, found);
            Ok(())
        }))
    })
}

/// Texture and surface references are legacy objects with no Level Zero analogue.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuModuleGetSurfRef(
    _out: *mut Opaque,
    _module: Opaque,
    _name: *const c_char,
) -> CuResult {
    NotSupported
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuModuleGetTexRef(
    _out: *mut Opaque,
    _module: Opaque,
    _name: *const c_char,
) -> CuResult {
    NotSupported
}
