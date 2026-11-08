//! Driver glue for kernel substitution (see `crate::subst`): chooses, lazily builds, validates and
//! launches native replacements for recognised ggml kernels, with the translated kernel as the
//! fallback for every failure.
use super::*;
use crate::level_zero::KernelArg;
use crate::subst::{self, Decision, Plan, Switch, probe};
use std::sync::Arc;

enum Native {
    Ready { module: Module, kernel: Handle },
    Failed,
}

#[derive(Default)]
pub(super) struct Natives {
    map: HashMap<String, Native>,
    /// (dst, tmp_fixup) of the last native `mul_mat_q`: its stream-k fixup kernel is skipped.
    pending_fixup: Option<(u64, u64)>,
}

impl Natives {
    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.map.len()
    }

    pub(super) fn destroy(&mut self, api: &Api) {
        for (_, native) in self.map.drain() {
            if let Native::Ready { mut module, .. } = native {
                let _ = module.destroy(api);
            }
        }
    }
}

/// Decides at kernel-resolution time (module load) whether `name` has a native replacement.
pub(super) fn plan_for(name: &str, params: &[Param]) -> Option<Arc<Plan>> {
    match subst::decide(name, params, &Switch::from_env()) {
        Decision::Substitute(plan) => {
            export_log_always(format_args!(
                "subst: {name} -> native {} (layout {:016x})",
                plan.key, plan.layout_hash
            ));
            Some(Arc::new(plan))
        }
        Decision::Mismatch(why) => {
            export_log_always(format_args!("subst: keeping translated PTX, {why}"));
            None
        }
        Decision::NotApplicable => None,
    }
}

fn probe_enabled() -> bool {
    std::env::var_os("CUDA_FOR_INTEL_SUBST_PROBE").as_deref() != Some(std::ffi::OsStr::new("0"))
}

/// Launches `translated` or, when a validated native kernel exists for `plan` and the launch
/// has the expected shape, the native replacement.
pub(super) fn launch(
    api: &Api,
    context: &mut CudaContext,
    plan: Option<&Arc<Plan>>,
    translated: Handle,
    stream: usize,
    geometry: ([u32; 3], [u32; 3]),
    args: &[KernelArg],
) -> Result<(), CuResult> {
    if let Some(plan) = plan {
        let needs_values = matches!(plan.kernel, subst::Kernel::MmqQ80 | subst::Kernel::MmqFixupQ80);
        let vals: Vec<u64> = if needs_values { arg_values(args) } else { Vec::new() };
        if plan.kernel == subst::Kernel::MmqQ80 {
            context.subst.pending_fixup = None;
        }
        if plan.kernel == subst::Kernel::MmqFixupQ80 {
            // [ids, bounds, dst, tmp, ...]: only fixups of a native mul_mat_q are no-ops
            if vals.len() == 13 && context.subst.pending_fixup == Some((vals[2], vals[3])) {
                context.subst.pending_fixup = None;
                return Ok(());
            }
        } else if let Some(mapped) = plan.map_launch(geometry.0, geometry.1, &vals)
            && let Some(kernel) = native(api, context, plan, stream)
        {
            match context.streams.launch(api, &context.backend, kernel, stream, mapped, args) {
                Ok(()) => {
                    if plan.kernel == subst::Kernel::MmqQ80 {
                        context.subst.pending_fixup = Some((vals[4], vals[5]));
                    }
                    return Ok(());
                }
                Err(error) => {
                    export_log_always(format_args!("subst: native {} launch failed ({error:?}); using PTX", plan.key));
                    context.subst.map.insert(plan.key.clone(), Native::Failed);
                }
            }
        }
    }
    context.streams.launch(api, &context.backend, translated, stream, geometry, args)
}

/// Launch words as integers (pointers as addresses).
fn arg_values(args: &[KernelArg]) -> Vec<u64> {
    args.iter()
        .map(|a| {
            let mut b = [0u8; 8];
            let n = a.size.min(8);
            unsafe { std::ptr::copy_nonoverlapping(a.value.cast::<u8>(), b.as_mut_ptr(), n) };
            u64::from_le_bytes(b)
        })
        .collect()
}

fn native(api: &Api, context: &mut CudaContext, plan: &Plan, stream: usize) -> Option<Handle> {
    match context.subst.map.get(&plan.key) {
        Some(Native::Ready { kernel, .. }) => return Some(*kernel),
        Some(Native::Failed) => return None,
        None => {}
    }
    let started = std::time::Instant::now();
    let entry = match build(api, context, plan, stream) {
        Ok((module, kernel)) => {
            export_log_always(format_args!(
                "subst: native {} ready (build+probe {} ms)",
                plan.key,
                started.elapsed().as_millis()
            ));
            context.subst.map.insert(plan.key.clone(), Native::Ready { module, kernel });
            Some(kernel)
        }
        Err(why) => {
            export_log_always(format_args!("subst: native {} rejected, using PTX: {why}", plan.key));
            context.subst.map.insert(plan.key.clone(), Native::Failed);
            None
        }
    };
    entry
}

fn build(api: &Api, context: &mut CudaContext, plan: &Plan, stream: usize) -> Result<(Module, Handle), String> {
    let words = plan.build(crate::ptx::dot_product_enabled())?;
    let mut module = context.backend.load_module(api, &words).map_err(|e| format!("module build: {e}"))?;
    let name = CString::new(plan.entry_name()).map_err(|e| e.to_string())?;
    let kernel = match module.create_kernel(api, &name) {
        Ok(kernel) => kernel,
        Err(error) => {
            let _ = module.destroy(api);
            return Err(format!("kernel create: {error}"));
        }
    };
    #[cfg(test)]
    let skip = api.is_fake();
    #[cfg(not(test))]
    let skip = false;
    if probe_enabled() && !skip {
        let mut device = Probe { api, context, kernel, stream, buffers: Vec::new() };
        let result = probe::validate(plan, &mut device);
        device.release();
        if let Err(why) = result {
            let _ = module.destroy(api);
            return Err(format!("first-use validation: {why}"));
        }
    }
    Ok((module, kernel))
}

/// The real device behind a probe case: USM buffers and a launch on the caller's stream.
struct Probe<'a> {
    api: &'a Api,
    context: &'a mut CudaContext,
    kernel: Handle,
    stream: usize,
    buffers: Vec<usize>,
}

impl Probe<'_> {
    fn release(&mut self) {
        for address in self.buffers.drain(..) {
            let _ = self.context.backend.free(self.api, address);
        }
    }
}

impl probe::Device for Probe<'_> {
    fn alloc(&mut self, bytes: usize) -> Result<u64, String> {
        let address = self.context.backend.alloc(self.api, bytes.max(16)).map_err(|e| e.to_string())?;
        self.buffers.push(address);
        Ok(address as u64)
    }

    fn upload(&mut self, addr: u64, data: &[u8]) -> Result<(), String> {
        unsafe {
            self.context.backend.copy(
                self.api,
                std::ptr::with_exposed_provenance_mut(addr as usize),
                data.as_ptr().cast(),
                data.len(),
            )
        }
        .map_err(|e| e.to_string())
    }

    fn download(&mut self, addr: u64, len: usize) -> Result<Vec<u8>, String> {
        let mut out = vec![0u8; len];
        unsafe {
            self.context.backend.copy(
                self.api,
                out.as_mut_ptr().cast(),
                std::ptr::with_exposed_provenance(addr as usize),
                len,
            )
        }
        .map_err(|e| e.to_string())?;
        Ok(out)
    }

    fn run(&mut self, grid: [u32; 3], block: [u32; 3], args: &[probe::PArg]) -> Result<(), String> {
        let storage: Vec<([u8; 8], usize)> = args
            .iter()
            .map(|a| match *a {
                probe::PArg::Ptr(v) | probe::PArg::U64(v) => (v.to_le_bytes(), 8),
                probe::PArg::U32(v) => {
                    let mut b = [0u8; 8];
                    b[..4].copy_from_slice(&v.to_le_bytes());
                    (b, 4)
                }
            })
            .collect();
        let kargs: Vec<KernelArg> = storage
            .iter()
            .map(|(bytes, size)| KernelArg { value: bytes.as_ptr().cast_mut().cast(), size: *size })
            .collect();
        self.context
            .streams
            .launch(self.api, &self.context.backend, self.kernel, self.stream, (grid, block), &kargs)
            .map_err(|e| format!("launch: {e:?}"))?;
        self.context
            .streams
            .wait_stream(self.api, self.stream, u64::MAX)
            .map_err(|e| format!("wait: {e:?}"))
    }
}
