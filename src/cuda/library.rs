//! Context-independent CUDA libraries with a Level Zero module per CUDA context.

use super::*;

#[derive(Default)]
pub(super) struct Libraries {
    pub(in crate::cuda) images: HashMap<usize, Library>,
    kernels: HashMap<usize, (usize, CString)>,
}

impl Libraries {
    pub(super) fn contains_kernel(&self, key: usize) -> bool {
        self.kernels.contains_key(&key)
    }
}

pub(in crate::cuda) struct Library {
    ptx: Vec<u8>,
    /// Source text, content hash and normalised PTX shared with the background prefetcher.
    shared: Option<std::sync::Arc<super::prefetch::LibShared>>,
    /// Whether the other kernels of this library were queued for background translation.
    prefetched: bool,
    entries: HashMap<CString, Option<usize>>,
    unloading: bool,
}

/// Register the library as a lazy PTX module (keyed by the library handle).
pub(super) fn ensure_lazy_library(state: &mut State, library: usize) -> Result<(), CuResult> {
    if state.lazy_modules.contains_key(&library) {
        return Ok(());
    }
    let image = state.libraries.images.get(&library).ok_or(InvalidHandle)?;
    if image.unloading {
        return Err(InvalidHandle);
    }
    let context_key = current().ok_or(InvalidContext)?;
    state.contexts.get(&context_key).ok_or(ContextIsDestroyed)?.check_live()?;
    let source = std::str::from_utf8(&image.ptx).map_err(|_| InvalidPtx)?;
    let names = ptx::entry_names(source)
        .map_err(|_| InvalidPtx)?
        .into_iter()
        .filter_map(|n| CString::new(n).ok())
        .collect();
    state.lazy_modules.insert(
        library,
        super::LazyModule {
            context: context_key,
            raw: source.to_owned(),
            prepared: std::cell::OnceCell::new(),
            names,
            children: HashMap::new(),
        },
    );
    Ok(())
}

pub(super) fn module_for_library(state: &mut State, library: usize) -> Result<(usize, usize), CuResult> {
    let library_image = state.libraries.images.get(&library).ok_or(InvalidHandle)?;
    if library_image.unloading {
        return Err(InvalidHandle);
    }
    let context_key = current().ok_or(InvalidContext)?;
    let context = state
        .contexts
        .get_mut(&context_key)
        .ok_or(ContextIsDestroyed)?;
    context.check_live()?;
    if let Some(&module) = context.library_modules.get(&library) {
        return Ok((context_key, module));
    }
    let module = load_ptx(
        &state.runtime,
        context,
        &mut state.next_handle,
        &library_image.ptx,
        true,
    )?;
    context.library_modules.insert(library, module);
    Ok((context_key, module))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuLibraryLoadData(
    out: *mut Opaque,
    code: *const c_void,
    jit_options: *const c_int,
    jit_values: *const Opaque,
    jit_count: c_uint,
    library_options: *const c_int,
    library_values: *const Opaque,
    library_count: c_uint,
) -> CuResult {
    trace::call("cuLibraryLoadData", || {
        if out.is_null()
            || code.is_null()
            || (jit_count != 0 && (jit_options.is_null() || jit_values.is_null()))
            || (library_count != 0 && (library_options.is_null() || library_values.is_null()))
        {
            return InvalidValue;
        }
        // We take ownership of a copy, so BINARY_IS_PRESERVED is a safe hint.
        if jit_count != 0
            || library_count > 1
            || (library_count == 1 && unsafe { library_options.read() } != 1)
        {
            if trace::enabled() {
                trace::record(format_args!(
                    "library options unsupported: jit={jit_count}, library={library_count}"
                ));
            }
            return NotSupported;
        }
        let code = code.cast::<u8>();
        let bytes = match unsafe { code.read() } {
            0xb1 | 0x50 => match unsafe { cfi_fatbin::extract_ptx(code) } {
                Some(ptx) => ptx,
                // SASS-only image: nothing here can run on this device, which is
                // what a real driver reports for a cubin built for another arch.
                None => {
                    if trace::enabled() {
                        trace::record(format_args!(
                            "library image has no PTX (fatbin): {}",
                            unsafe { cfi_fatbin::describe(code) }
                        ));
                    }
                    return NoBinaryForGpu;
                }
            },
            0x7f => {
                if trace::enabled() {
                    trace::record(format_args!("library image is a raw ELF cubin"));
                }
                return NoBinaryForGpu;
            }
            _ => unsafe { CStr::from_ptr(code.cast()) }.to_bytes().to_vec(),
        };
        let Ok(source) = std::str::from_utf8(&bytes) else {
            return InvalidPtx;
        };
        let names = match ptx::entry_names(source) {
            Ok(names) => names,
            Err(error) => {
                diagnostic(&error);
                return InvalidPtx;
            }
        };
        let mut entries = HashMap::new();
        for name in names {
            let Ok(name) = CString::new(name) else {
                return InvalidPtx;
            };
            if entries.insert(name, None).is_some() {
                return InvalidPtx;
            }
        }
        status(state(|state| {
            let key = reserve_handle(&mut state.next_handle)?;
            state.libraries.images.insert(
                key,
                Library {
                    ptx: bytes,
                    shared: None,
                    prefetched: false,
                    entries,
                    unloading: false,
                },
            );
            unsafe { out.write(std::ptr::without_provenance_mut(key)) };
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuLibraryGetKernel(
    out: *mut Opaque,
    library: Opaque,
    name: *const c_char,
) -> CuResult {
    trace::call("cuLibraryGetKernel", || {
        if out.is_null() || name.is_null() {
            return InvalidValue;
        }
        status(state(|state| {
            let key = library.addr();
            let library = state.libraries.images.get_mut(&key).ok_or(InvalidHandle)?;
            if library.unloading {
                return Err(InvalidHandle);
            }
            let name = unsafe { CStr::from_ptr(name) };
            let slot = library.entries.get_mut(name).ok_or(NotFound)?;
            let kernel = if let Some(kernel) = *slot {
                kernel
            } else {
                let kernel = reserve_handle(&mut state.next_handle)?;
                state
                    .libraries
                    .kernels
                    .insert(kernel, (key, name.to_owned()));
                *slot = Some(kernel);
                kernel
            };
            unsafe { out.write(std::ptr::without_provenance_mut(kernel)) };
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuLibraryGetModule(out: *mut Opaque, library: Opaque) -> CuResult {
    trace::call("cuLibraryGetModule", || {
        if out.is_null() {
            return InvalidValue;
        }
        status(state(|state| {
            // Lazy module: kernels translate one at a time on cuModuleGetFunction.
            ensure_lazy_library(state, library.addr())?;
            unsafe { out.write(std::ptr::without_provenance_mut(library.addr())) };
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuKernelGetFunction(out: *mut Opaque, kernel: Opaque) -> CuResult {
    trace::call("cuKernelGetFunction", || {
        if out.is_null() {
            return InvalidValue;
        }
        status(state(|state| {
            let function = kernel_function(state, kernel.addr())?;
            unsafe { out.write(std::ptr::without_provenance_mut(function)) };
            Ok(())
        }))
    })
}

pub(super) fn kernel_function(state: &mut State, kernel: usize) -> Result<usize, CuResult> {
    let (library, name) = state.libraries.kernels.get(&kernel).ok_or(InvalidHandle)?;
    let (library, name) = (*library, name.clone());
    if trace::enabled() {
        trace::record(format_args!(
            "resolve library kernel {}",
            name.to_string_lossy()
        ));
    }
    let result = ensure_lazy_library(state, library)
        .and_then(|()| super::lazy_module_function(state, library, &name));
    if matches!(result, Err(NotFound | InvalidPtx)) {
        if trace::enabled()
            && let Ok(source) = std::str::from_utf8(&state.libraries.images[&library].ptx)
            && let Err(error) = ptx::diagnose_entry(source, &name.to_string_lossy())
        {
            trace::record(format_args!("library entry translation failed: {error}"));
        }
        return Err(InvalidPtx);
    }
    result
}

/// Libraries are translated per kernel on first use, so report lazy loading.
/// Without this entry cudart assumes eager loading and loads every registered
/// fatbin (including SASS-only ones) at context creation.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuModuleGetLoadingMode(mode: *mut c_int) -> CuResult {
    trace::call("cuModuleGetLoadingMode", || {
        if mode.is_null() {
            return InvalidValue;
        }
        // CU_MODULE_LAZY_LOADING
        unsafe { mode.write(2) };
        Success
    })
}

/// `CUDA_FOR_INTEL_LAZY=0` restores whole-library modules.
fn lazy_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("CUDA_FOR_INTEL_LAZY").as_deref() != Some(std::ffi::OsStr::new("0")))
}

/// Resolves a library kernel through a module holding only that kernel. `Ok(None)` means the
/// kernel needs the whole-library path (module variables, or a translation the entry path
/// cannot do), which then reports any real error itself.
fn lazy_kernel_function(state: &mut State, library: usize, name: &CStr) -> Result<Option<usize>, CuResult> {
    let context_key = current().ok_or(InvalidContext)?;
    // Translation settings (dp4a lowering, dynamic shared window) must be fixed before any
    // source is normalised.
    probe_dot_product(
        &state.runtime,
        state.contexts.get_mut(&context_key).ok_or(ContextIsDestroyed)?,
    );
    let image = state.libraries.images.get_mut(&library).ok_or(InvalidHandle)?;
    if image.unloading {
        return Err(InvalidHandle);
    }
    if let Some(&module) = state
        .contexts
        .get(&context_key)
        .ok_or(ContextIsDestroyed)?
        .entry_modules
        .get(&(library, name.to_owned()))
    {
        let context = state.contexts.get_mut(&context_key).ok_or(ContextIsDestroyed)?;
        return module_function(&state.runtime.api, context, &mut state.next_handle, module, name).map(Some);
    }
    if image.shared.is_none() {
        let Ok(text) = std::str::from_utf8(&image.ptx) else { return Ok(None) };
        image.shared = Some(std::sync::Arc::new(super::prefetch::LibShared::new(text.to_owned())));
    }
    let shared = image.shared.clone().expect("set above");
    let entry_name = name.to_string_lossy();
    let cache_key = xcache::key_for_hash(shared.hash, &entry_name);
    // A background worker may be producing this very kernel: wait for it instead of repeating it.
    super::prefetch::claim_or_wait(cache_key);
    let translated = match xcache::get(cache_key) {
        Some(hit) => hit,
        None => {
            let started = std::time::Instant::now();
            let Some(prepared) = shared.prepared() else { return Ok(None) };
            // CFI_DUMP_ENTRY=<substring> with CFI_DUMP_PTX_DIR keeps the normalised PTX of matching kernels.
            if let (Some(pattern), Some(dir)) = (std::env::var_os("CFI_DUMP_ENTRY"), std::env::var_os("CFI_DUMP_PTX_DIR"))
                && entry_name.contains(&*pattern.to_string_lossy())
                && let Some(start) = prepared.find(&format!(".entry {entry_name}"))
            {
                let end = prepared[start + 6..].find(".entry").map_or(prepared.len(), |e| start + 6 + e);
                let file = format!("entry-{}.ptx", entry_name.chars().take(80).collect::<String>());
                let _ = std::fs::write(std::path::Path::new(&dir).join(file), &prepared[start..end]);
            }
            let translated = match ptx::translate_prepared_entry(&prepared, &entry_name) {
                Ok(t) => t,
                Err(error) => {
                    export_log(format_args!("lazy entry translation fell back: {entry_name}: {error}"));
                    return Ok(None);
                }
            };
            xcache::put(cache_key, &translated);
            export_log(format_args!("lazy entry translated: {entry_name} translate_ms={}", started.elapsed().as_millis()));
            translated
        }
    };
    // Module variables are per-module storage, so kernels that use them keep the shared module.
    if !translated.globals.is_empty() {
        return Ok(None);
    }
    let ptx_len = shared.text.len();
    let others: Vec<CString> = if image.prefetched { Vec::new() } else { image.entries.keys().cloned().collect() };
    let first_in_library = !image.prefetched;
    image.prefetched = true;
    let context = state.contexts.get_mut(&context_key).ok_or(ContextIsDestroyed)?;
    context.check_live()?;
    let module = match install_translation(&state.runtime, context, &mut state.next_handle, translated, ptx_len, std::time::Duration::ZERO) {
        Ok(module) => module,
        Err(InvalidPtx) => return Ok(None),
        Err(error) => return Err(error),
    };
    if first_in_library {
        let compiler = context.backend.native_compiler(&state.runtime.api);
        super::prefetch::start_library(&shared, &others, &entry_name, compiler);
    }
    context.entry_modules.insert((library, name.to_owned()), module);
    module_function(&state.runtime.api, context, &mut state.next_handle, module, name).map(Some)
}

#[unsafe(no_mangle)]
pub extern "system" fn cuLibraryUnload(library: Opaque) -> CuResult {
    trace::call("cuLibraryUnload", || {
        status(state(|state| {
            let key = library.addr();
            if let Some(lazy) = state.lazy_modules.remove(&key) {
                for ((ctx, _), (child, _)) in lazy.children.iter() {
                    if let Some(context) = state.contexts.get_mut(ctx) {
                        context.streams.wait_all(&state.runtime.api)?;
                        context.unload(&state.runtime.api, *child)?;
                    }
                }
            }
            let library = state.libraries.images.get_mut(&key).ok_or(InvalidHandle)?;
            library.unloading = true;
            for context in state.contexts.values_mut() {
                if let Some(&module) = context.library_modules.get(&key) {
                    context.streams.wait_all(&state.runtime.api)?;
                    context.unload(&state.runtime.api, module)?;
                    context.library_modules.remove(&key);
                }
                let lazy: Vec<_> = context
                    .entry_modules
                    .iter()
                    .filter(|((owner, _), _)| *owner == key)
                    .map(|(entry, &module)| (entry.clone(), module))
                    .collect();
                for (entry, module) in lazy {
                    context.streams.wait_all(&state.runtime.api)?;
                    context.unload(&state.runtime.api, module)?;
                    context.entry_modules.remove(&entry);
                }
            }
            for kernel in library.entries.values().flatten() {
                state.libraries.kernels.remove(kernel);
            }
            state.libraries.images.remove(&key);
            Ok(())
        }))
    })
}
