//! Ahead-of-time kernel precompilation (`cfi-precompile`, `cfi.ps1 precompile`).
//!
//! The first launch of every kernel normally pays PTX -> SPIR-V translation plus the Intel
//! compiler. This walks executables, DLLs, fatbins and `.ptx` files, extracts the PTX record
//! the runtime would pick from every fatbin frame (`cfi_fatbin::extract_ptx`, the same call
//! the load paths use), translates every entry on all cores into the SPIR-V cache (`xcache`)
//! and builds each module into the Level Zero native-binary cache (`native_cache`).
//!
//! Cache correctness: nothing here has a key of its own. Each entry uses the key the lazy
//! launch path computes (`xcache::key_for_hash(module_hash(raw PTX), entry)`), after the
//! device probe has fixed the same translator switches, with the same thread-local leniency.
//! The native binary is keyed by the SPIR-V words, device and driver identity and build
//! flags, exactly as in `Context::load_module`. A stale or mismatched entry is therefore just
//! a miss at run time, never a wrong kernel. Run it with the same `CFI_*` and
//! `CUDA_FOR_INTEL_*` environment as the application (cache directories, translator switches).
//!
//! The entry point is the exported `cfi_precompile_main`, so the tool always uses the very
//! `nvcuda` build that will run the application (its translator version is part of the key).

use super::*;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

const USAGE: &str = "usage: cfi-precompile [options] <file-or-dir>...
  Translates the PTX of every fatbin in the given executables/DLLs (directories are walked)
  into the SPIR-V cache and compiles it into the native-binary cache.
options:
  --translate-only      no GPU: fill the SPIR-V cache only
  --dynamic-shared N    with --translate-only: the device's dynamic shared window (default 65536)
  --device N            device ordinal (default 0)
  --threads N           worker threads (default: logical cores)
  --filter TEXT         only entries whose name contains TEXT
  --all-files           also scan NVIDIA libraries that cfi-run replaces with ours
                        (cublas64_*, cublasLt64_*, cufft64_*, cusparse64_*, nvrtc64_*)
  --no-native           open the device (same translator switches) but fill the SPIR-V
                        cache only
  --native-skip TEXT    no native build for entries whose name contains TEXT (repeatable)
  --ggml-types LIST     native build only for ggml kernels instantiated for these tensor
                        types (comma list such as f16,q4_k,q6_k,q8_0; `all` = every type);
                        kernels without a ggml_type template argument are always built
  --gguf PATH            add the tensor types of a GGUF model, or of every GGUF file in a
                        directory such as Ollama's models/blobs, to --ggml-types (f32
                        and f16 are always added)
  --list                list modules and entries; translate nothing
  --report FILE         per-entry TSV: module, entry, result, translate_ms, compile_ms, detail
  --quiet               summary only";

#[derive(Default, Debug)]
pub(crate) struct Options {
    pub paths: Vec<std::path::PathBuf>,
    pub translate_only: bool,
    pub dynamic_shared: Option<u32>,
    pub device: c_int,
    pub threads: Option<usize>,
    pub filter: Option<String>,
    pub list: bool,
    pub all_files: bool,
    pub no_native: bool,
    pub native_skip: Vec<String>,
    /// ggml_type enum values to build natively (`--ggml-types`, `--gguf`); `None` = all.
    pub ggml_types: Option<Vec<u32>>,
    pub all_ggml_types: bool,
    pub gguf: Vec<std::path::PathBuf>,
    pub report: Option<std::path::PathBuf>,
    pub quiet: bool,
}

pub(crate) fn parse_args(args: &[String]) -> Result<Options, String> {
    let mut o = Options::default();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut value = |name: &str| it.next().cloned().ok_or_else(|| format!("{name} needs a value"));
        match a.as_str() {
            "--translate-only" => o.translate_only = true,
            "--list" => o.list = true,
            "--all-files" => o.all_files = true,
            "--no-native" => o.no_native = true,
            "--native-skip" => o.native_skip.push(value(a)?),
            "--ggml-types" => match parse_ggml_types(&value(a)?)? {
                Some(list) => o.ggml_types.get_or_insert_with(Vec::new).extend(list),
                None => o.all_ggml_types = true,
            },
            "--gguf" => o.gguf.push(value(a)?.into()),
            "--quiet" => o.quiet = true,
            "--dynamic-shared" => o.dynamic_shared = Some(value(a)?.parse().map_err(|_| "--dynamic-shared: not a number")?),
            "--device" => o.device = value(a)?.parse().map_err(|_| "--device: not a number")?,
            "--threads" => o.threads = Some(value(a)?.parse().map_err(|_| "--threads: not a number")?),
            "--filter" => o.filter = Some(value(a)?),
            "--report" => o.report = Some(value(a)?.into()),
            "-h" | "--help" => return Err(String::new()),
            s if s.starts_with("--") => return Err(format!("unknown option {s}")),
            s => o.paths.push(s.into()),
        }
    }
    if o.paths.is_empty() {
        return Err("no input files".into());
    }
    if !o.gguf.is_empty() {
        let mut types = o.ggml_types.take().unwrap_or_default();
        for p in &o.gguf {
            types.extend(gguf_types_in(p)?);
        }
        o.ggml_types = Some(types);
    }
    if let Some(types) = o.ggml_types.as_mut() {
        // Activations, norms and the KV cache are f32/f16 in every model.
        types.extend([0, 1]);
        types.sort_unstable();
        types.dedup();
    }
    if o.all_ggml_types {
        o.ggml_types = None;
    }
    Ok(o)
}

/// ggml's `enum ggml_type` values by the names ggml and Ollama print (lower case).
const GGML_TYPES: [(&str, u32); 31] = [
    ("f32", 0), ("f16", 1), ("q4_0", 2), ("q4_1", 3), ("q5_0", 6), ("q5_1", 7), ("q8_0", 8), ("q8_1", 9),
    ("q2_k", 10), ("q3_k", 11), ("q4_k", 12), ("q5_k", 13), ("q6_k", 14), ("q8_k", 15), ("iq2_xxs", 16),
    ("iq2_xs", 17), ("iq3_xxs", 18), ("iq1_s", 19), ("iq4_nl", 20), ("iq3_s", 21), ("iq2_s", 22), ("iq4_xs", 23),
    ("i8", 24), ("i16", 25), ("i32", 26), ("i64", 27), ("f64", 28), ("iq1_m", 29), ("bf16", 30), ("tq1_0", 34),
    ("mxfp4", 39),
];

fn parse_ggml_types(list: &str) -> Result<Option<Vec<u32>>, String> {
    if list.eq_ignore_ascii_case("all") {
        return Ok(None);
    }
    let mut out = Vec::new();
    for name in list.split(',').map(str::trim).filter(|n| !n.is_empty()) {
        let lower = name.to_ascii_lowercase();
        let Some(&(_, v)) = GGML_TYPES.iter().find(|(n, _)| *n == lower) else {
            return Err(format!("--ggml-types: unknown type {name}"));
        };
        out.push(v);
    }
    Ok(Some(out))
}

/// Tensor types of one GGUF file (header only), `None` when the file is not GGUF.
pub(crate) fn gguf_tensor_types(path: &std::path::Path) -> Result<Option<Vec<u32>>, String> {
    use std::io::Read;
    let err = |e: std::io::Error| format!("{}: {e}", path.display());
    let mut f = std::io::BufReader::new(std::fs::File::open(path).map_err(err)?);
    let mut magic = [0u8; 4];
    if f.read_exact(&mut magic).is_err() || &magic != b"GGUF" {
        return Ok(None);
    }
    fn u32_(f: &mut impl Read) -> Option<u32> {
        let mut b = [0u8; 4];
        f.read_exact(&mut b).ok()?;
        Some(u32::from_le_bytes(b))
    }
    fn u64_(f: &mut impl Read) -> Option<u64> {
        let mut b = [0u8; 8];
        f.read_exact(&mut b).ok()?;
        Some(u64::from_le_bytes(b))
    }
    fn skip(f: &mut impl Read, n: u64) -> Option<()> {
        (std::io::copy(&mut f.take(n), &mut std::io::sink()).ok()? == n).then_some(())
    }
    fn string(f: &mut impl Read) -> Option<()> {
        let n = u64_(f)?;
        (n < 1 << 24).then_some(())?;
        skip(f, n)
    }
    fn value(f: &mut impl Read, ty: u32, depth: u32) -> Option<()> {
        match ty {
            0 | 1 | 7 => skip(f, 1),
            2 | 3 => skip(f, 2),
            4..=6 => skip(f, 4),
            10..=12 => skip(f, 8),
            8 => string(f),
            9 if depth == 0 => {
                let inner = u32_(f)?;
                let n = u64_(f)?;
                (n < 1 << 32).then_some(())?;
                for _ in 0..n {
                    value(f, inner, 1)?;
                }
                Some(())
            }
            _ => None,
        }
    }
    let parse = |f: &mut std::io::BufReader<std::fs::File>| -> Option<Vec<u32>> {
        let version = u32_(f)?;
        if !(2..=3).contains(&version) {
            return None;
        }
        let tensors = u64_(f)?;
        let kvs = u64_(f)?;
        for _ in 0..kvs {
            string(f)?;
            let ty = u32_(f)?;
            value(f, ty, 0)?;
        }
        let mut types = Vec::new();
        for _ in 0..tensors.min(1 << 20) {
            string(f)?;
            let dims = u32_(f)?;
            skip(f, 8 * u64::from(dims.min(8)))?;
            types.push(u32_(f)?);
            u64_(f)?;
        }
        types.sort_unstable();
        types.dedup();
        Some(types)
    };
    parse(&mut f).map(Some).ok_or_else(|| format!("{}: malformed GGUF header", path.display()))
}

/// Tensor types of a GGUF file or of every GGUF file below a directory.
fn gguf_types_in(path: &std::path::Path) -> Result<Vec<u32>, String> {
    let mut files = Vec::new();
    if path.is_dir() {
        let rd = std::fs::read_dir(path).map_err(|e| format!("{}: {e}", path.display()))?;
        files.extend(rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.is_file()));
    } else {
        files.push(path.to_owned());
    }
    let mut types = Vec::new();
    let mut found = 0;
    for f in files {
        if let Some(t) = gguf_tensor_types(&f)? {
            found += 1;
            types.extend(t);
        }
    }
    if found == 0 {
        return Err(format!("{}: no GGUF model found", path.display()));
    }
    Ok(types)
}

/// ggml_type template arguments of a mangled kernel name (`L9ggml_type8E` is ggml_type 8; a
/// repeated enum argument is mangled as a substitution `LS0_8E`).
fn ggml_type_args(name: &str) -> Vec<u32> {
    let mut out = Vec::new();
    let has_enum = name.contains("L9ggml_type");
    for marker in ["L9ggml_type", "LS0_", "LS1_", "LS2_"] {
        if marker != "L9ggml_type" && !has_enum {
            continue;
        }
        let mut rest = name;
        while let Some(at) = rest.find(marker) {
            rest = &rest[at + marker.len()..];
            let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
            if !digits.is_empty() && rest[digits.len()..].starts_with('E') {
                out.push(digits.parse().unwrap_or(u32::MAX));
            }
        }
    }
    out
}

/// Whether an entry gets a native build under the selection options.
fn wants_native(o: &Options, entry: &str) -> bool {
    if o.native_skip.iter().any(|s| entry.contains(s.as_str())) {
        return false;
    }
    match &o.ggml_types {
        None => true,
        Some(types) => ggml_type_args(entry).iter().all(|t| types.contains(t)),
    }
}

/// Every PTX text the runtime would extract from the fatbin frames in `bytes` (an executable,
/// a DLL or a bare fatbin), in file order. Each frame header is bounds-checked against `bytes`
/// before `extract_ptx` reads it.
pub(crate) fn scan_image(bytes: &[u8]) -> (Vec<Vec<u8>>, usize) {
    const MAGIC: [u8; 4] = cfi_fatbin::FRAME_MAGIC.to_le_bytes();
    let mut found = Vec::new();
    let mut frames = 0;
    let mut at = 0usize;
    while at + 16 <= bytes.len() {
        if bytes[at..at + 4] != MAGIC {
            at += 4;
            continue;
        }
        let header = &bytes[at..at + 16];
        let version = u16::from_le_bytes([header[4], header[5]]);
        let header_size = usize::from(u16::from_le_bytes([header[6], header[7]]));
        let file_size = u64::from_le_bytes(header[8..16].try_into().unwrap());
        let total = usize::try_from(file_size).ok().and_then(|f| f.checked_add(header_size));
        let fits = total.is_some_and(|t| at.checked_add(t).is_some_and(|end| end <= bytes.len()));
        if version != 1 || header_size < 16 || file_size == 0 || file_size > 64 << 20 || !fits {
            at += 4;
            continue;
        }
        frames += 1;
        // SAFETY: the frame header and the `file_size` record bytes it declares lie inside `bytes`.
        if let Some(ptx) = unsafe { cfi_fatbin::extract_ptx(bytes[at..].as_ptr()) } {
            found.push(ptx);
        }
        at += total.unwrap().next_multiple_of(4).max(4);
    }
    (found, frames)
}

/// NVIDIA libraries whose kernels never run under cuda-for-intel: `cfi-run` (and an install)
/// loads our stand-in instead, so their PTX would only fill the caches with unused entries.
fn replaced_by_stand_in(path: &std::path::Path) -> bool {
    let name = path.file_name().map(|n| n.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
    name.ends_with(".dll") && ["cublas64_", "cublaslt64_", "cufft64_", "cusparse64_", "nvrtc64_"].iter().any(|p| name.starts_with(p))
}

fn walk(path: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    if path.is_dir() {
        let Ok(rd) = std::fs::read_dir(path) else { return };
        let mut v: Vec<_> = rd.filter_map(|e| e.ok()).map(|e| e.path()).collect();
        v.sort();
        for p in v {
            walk(&p, out);
        }
    } else if path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| ["dll", "exe", "so", "fatbin", "ptx", "pyd", "cubin"].contains(&e.to_ascii_lowercase().as_str()))
        || path.extension().is_none()
    {
        out.push(path.to_owned());
    }
}

struct ModuleSrc {
    origin: String,
    raw: String,
    hash: u64,
    prepared: OnceLock<Option<String>>,
}

struct Job {
    module: usize,
    entry: String,
    native: bool,
}

#[derive(Default)]
struct Counters {
    done: AtomicUsize,
    xcache_hit: AtomicUsize,
    translated: AtomicUsize,
    rejected: AtomicUsize,
    native_cached: AtomicUsize,
    native_built: AtomicUsize,
    native_failed: AtomicUsize,
    uncacheable: AtomicUsize,
    subst_built: AtomicUsize,
}

/// Opens the device the way an application does (primary context), applies the device probe
/// to the translator, and returns the native compiler. Leaves the primary context retained.
fn open_device(ordinal: c_int) -> Result<Option<crate::level_zero::NativeCompiler>, String> {
    let rc = cuInit(0);
    if rc != Success {
        return Err(format!("cuInit failed ({rc:?}); is the Intel GPU driver installed?"));
    }
    let mut ctx: Opaque = std::ptr::null_mut();
    let rc = unsafe { primary::cuDevicePrimaryCtxRetain(&mut ctx, ordinal) };
    if rc != Success {
        return Err(format!("device {ordinal}: cuDevicePrimaryCtxRetain failed ({rc:?})"));
    }
    let rc = cuCtxSetCurrent(ctx);
    if rc != Success {
        return Err(format!("cuCtxSetCurrent failed ({rc:?})"));
    }
    state(|state| {
        let key = current().ok_or(InvalidContext)?;
        let context = state.contexts.get_mut(&key).ok_or(ContextIsDestroyed)?;
        probe_dot_product(&state.runtime, context);
        Ok(context.backend.native_compiler(&state.runtime.api))
    })
    .map_err(|e| format!("device setup failed ({e:?})"))
}

pub(crate) struct Summary {
    pub modules: usize,
    pub entries: usize,
    pub xcache_hit: usize,
    pub translated: usize,
    pub rejected: usize,
    pub arch_stubs: usize,
    pub native_cached: usize,
    pub native_built: usize,
    pub native_failed: usize,
}

pub(crate) fn run(o: &Options, log: &mut (dyn std::io::Write + Send)) -> Result<Summary, String> {
    let started = std::time::Instant::now();
    let mut files = Vec::new();
    for p in &o.paths {
        if !p.exists() {
            return Err(format!("{}: not found", p.display()));
        }
        walk(p, &mut files);
    }
    // Collect unique modules (the same PTX in two DLLs is one cache entry).
    let mut modules: Vec<ModuleSrc> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let skipped = files.len();
    if !o.all_files {
        files.retain(|f| !replaced_by_stand_in(f));
    }
    let skipped = skipped - files.len();
    if skipped > 0 && !o.quiet {
        let _ = writeln!(log, "skipped {skipped} NVIDIA libraries that cfi-run replaces (--all-files scans them)");
    }
    for f in &files {
        let Ok(bytes) = std::fs::read(f) else { continue };
        let name = f.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let (texts, frames) = if f.extension().is_some_and(|e| e.eq_ignore_ascii_case("ptx")) {
            (vec![bytes], 0)
        } else {
            scan_image(&bytes)
        };
        let mut added = 0;
        for (i, t) in texts.into_iter().enumerate() {
            let Ok(raw) = String::from_utf8(t) else { continue };
            let hash = xcache::module_hash(&raw);
            if seen.insert(hash) {
                modules.push(ModuleSrc {
                    origin: format!("{name}#{i}"),
                    raw,
                    hash,
                    prepared: OnceLock::new(),
                });
                added += 1;
            }
        }
        if (frames > 0 || added > 0) && !o.quiet {
            let _ = writeln!(log, "{name}: {frames} fatbin frames, {added} new PTX modules");
        }
    }
    let mut jobs = Vec::new();
    let mut arch_stubs = 0;
    for (m, src) in modules.iter().enumerate() {
        let names = match ptx::entry_names(&src.raw) {
            Ok(n) => n,
            Err(e) => {
                let _ = writeln!(log, "{}: entry list failed: {e}", src.origin);
                continue;
            }
        };
        for n in names {
            if o.filter.as_ref().is_some_and(|f| !n.contains(f.as_str())) {
                continue;
            }
            // The runtime refuses architecture-guard stubs before translating them.
            if is_arch_guard_stub(&src.raw, n) {
                arch_stubs += 1;
                continue;
            }
            jobs.push(Job { module: m, entry: n.to_owned(), native: wants_native(o, n) });
        }
    }
    if o.list {
        for j in &jobs {
            let _ = writeln!(log, "{}\t{}", modules[j.module].origin, j.entry);
        }
    }
    let _ = writeln!(
        log,
        "{} files, {} PTX modules, {} entries ({} architecture-guard stubs skipped, {} selected for a native build)",
        files.len(),
        modules.len(),
        jobs.len(),
        arch_stubs,
        jobs.iter().filter(|j| j.native).count()
    );
    let mut summary = Summary {
        modules: modules.len(),
        entries: jobs.len(),
        xcache_hit: 0,
        translated: 0,
        rejected: 0,
        arch_stubs,
        native_cached: 0,
        native_built: 0,
        native_failed: 0,
    };
    if o.list || jobs.is_empty() {
        return Ok(summary);
    }
    let compiler = if o.translate_only {
        let window = devices::probe::window(
            o.dynamic_shared.unwrap_or(devices::probe::DEFAULT_WINDOW),
            devices::probe::ProbeEnv::from_process().wanted_window,
        );
        devices::probe::DeviceProbe {
            dynamic_shared: window,
            dot_product: false,
        }
        .apply();
        None
    } else {
        let c = open_device(o.device)?;
        let c = if o.no_native { None } else { c };
        if c.is_none() && !o.no_native {
            let _ = writeln!(log, "native-binary cache disabled (CUDA_FOR_INTEL_CACHE=0): SPIR-V cache only");
        }
        c
    };
    let _ = writeln!(
        log,
        "translator: dynamic shared window {} dot product {}; SPIR-V cache {}",
        ptx::dynamic_shared_bytes(),
        ptx::dot_product_enabled(),
        if xcache::dir_for_display().is_some() {
            "on"
        } else {
            "off (CFI_NO_CACHE)"
        }
    );
    schedule_jobs(&mut jobs, |m| modules[m].raw.len());
    let threads = o
        .threads
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(2, usize::from))
        .clamp(1, 64)
        .min(jobs.len());
    let counters = Counters::default();
    let next = AtomicUsize::new(0);
    let rows = std::sync::Mutex::new(Vec::<String>::new());
    let compiler = compiler.map(Arc::new);
    let total = jobs.len();
    let progress_every = (total / 20).max(1);
    let log_lock = std::sync::Mutex::new(&mut *log);
    std::thread::scope(|scope| {
        for t in 0..threads {
            let (jobs, modules, counters, next, rows, compiler, log_lock) =
                (&jobs, &modules, &counters, &next, &rows, compiler.clone(), &log_lock);
            let quiet = o.quiet;
            let _ = std::thread::Builder::new()
                .name(format!("cfi-precompile-{t}"))
                .stack_size(64 << 20)
                .spawn_scoped(scope, move || {
                    // As in the launch path: undefined register reads become zero.
                    ptx::set_lenient_undefined(true);
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(job) = jobs.get(i) else { break };
                        let src = &modules[job.module];
                        let row =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| one(src, &job.entry, compiler.as_deref().filter(|_| job.native), counters)))
                                .unwrap_or_else(|_| ("panic", 0, 0, String::new()));
                        if let Ok(mut r) = rows.lock() {
                            r.push(format!("{}\t{}\t{}\t{}\t{}\t{}", src.origin, job.entry, row.0, row.1, row.2, row.3));
                        }
                        let done = counters.done.fetch_add(1, Ordering::Relaxed) + 1;
                        if !quiet && done % progress_every == 0 {
                            if let Ok(mut l) = log_lock.lock() {
                                let _ = writeln!(l, "  {done}/{total}");
                            }
                        }
                    }
                });
        }
    });
    let log = log_lock.into_inner().unwrap_or_else(|p| p.into_inner());
    summary.xcache_hit = counters.xcache_hit.load(Ordering::Relaxed);
    summary.translated = counters.translated.load(Ordering::Relaxed);
    summary.rejected = counters.rejected.load(Ordering::Relaxed);
    summary.native_cached = counters.native_cached.load(Ordering::Relaxed);
    summary.native_built = counters.native_built.load(Ordering::Relaxed);
    summary.native_failed = counters.native_failed.load(Ordering::Relaxed);
    let rows = rows.into_inner().unwrap_or_else(|p| p.into_inner());
    if let Some(path) = &o.report {
        let mut text = String::from("module\tentry\tresult\ttranslate_ms\tcompile_ms\tdetail\n");
        for r in &rows {
            text.push_str(r);
            text.push('\n');
        }
        std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    if !o.quiet {
        let mut classes: BTreeMap<String, usize> = BTreeMap::new();
        for r in rows.iter().filter(|r| r.split('\t').nth(2) == Some("rejected")) {
            let detail = r.rsplit('\t').next().unwrap_or("");
            *classes.entry(detail.chars().take(100).collect()).or_default() += 1;
        }
        for (class, n) in &classes {
            let _ = writeln!(log, "  rejected x{n}: {class}");
        }
    }
    let _ = writeln!(
        log,
        "done in {:.1}s: {} translated, {} already in the SPIR-V cache, {} rejected by the translator, {} with module variables (not SPIR-V cached); native: {} built, {} already cached, {} failed, {} substitute kernels built",
        started.elapsed().as_secs_f64(),
        summary.translated,
        summary.xcache_hit,
        summary.rejected,
        counters.uncacheable.load(Ordering::Relaxed),
        summary.native_built,
        summary.native_cached,
        summary.native_failed,
        counters.subst_built.load(Ordering::Relaxed)
    );
    Ok(summary)
}

/// Orders the work queue. Big modules come first so the long Intel-compiler jobs do not end up
/// last. Within that order, the first entry of every module is moved to the front: its worker
/// runs the module's (single-threaded) PTX normalisation, so the modules are prepared
/// concurrently. Without this, all workers reach the same module together and wait for one
/// thread's prepare, module after module.
fn schedule_jobs(jobs: &mut Vec<Job>, module_len: impl Fn(usize) -> usize) {
    jobs.sort_by_key(|j| std::cmp::Reverse(module_len(j.module)));
    let mut seen = std::collections::HashSet::new();
    let (first, rest): (Vec<Job>, Vec<Job>) = std::mem::take(jobs).into_iter().partition(|j| seen.insert(j.module));
    jobs.extend(first);
    jobs.extend(rest);
}

/// Translates (or finds) one entry and builds it natively. Returns (result, translate_ms,
/// compile_ms, detail).
fn one(
    src: &ModuleSrc,
    entry: &str,
    compiler: Option<&crate::level_zero::NativeCompiler>,
    c: &Counters,
) -> (&'static str, u128, u128, String) {
    let t0 = std::time::Instant::now();
    let key = xcache::key_for_hash(src.hash, entry);
    let (translation, mut result) = match xcache::get(key) {
        Some(hit) => {
            c.xcache_hit.fetch_add(1, Ordering::Relaxed);
            (hit, "cached")
        }
        None => {
            let prepared = src.prepared.get_or_init(|| ptx::prepare_module(&src.raw).ok());
            let Some(prepared) = prepared else {
                c.rejected.fetch_add(1, Ordering::Relaxed);
                return ("rejected", 0, 0, "module normalisation failed".into());
            };
            match ptx::translate_prepared_entry(prepared, entry) {
                Ok(t) => {
                    if t.globals.is_empty() {
                        xcache::put(key, &t);
                    } else {
                        c.uncacheable.fetch_add(1, Ordering::Relaxed);
                    }
                    c.translated.fetch_add(1, Ordering::Relaxed);
                    (t, "translated")
                }
                Err(e) => {
                    c.rejected.fetch_add(1, Ordering::Relaxed);
                    let detail = e.to_string().replace(['\t', '\n'], " ");
                    return ("rejected", t0.elapsed().as_millis(), 0, detail);
                }
            }
        }
    };
    let translate_ms = t0.elapsed().as_millis();
    let mut compile_ms = 0;
    if let Some(compiler) = compiler.filter(|_| translation.words.first() == Some(&0x0723_0203)) {
        let t1 = std::time::Instant::now();
        if compiler.is_cached(&translation.words) {
            c.native_cached.fetch_add(1, Ordering::Relaxed);
        } else if compiler.precompile(&translation.words) {
            c.native_built.fetch_add(1, Ordering::Relaxed);
            result = if result == "cached" { "cached+built" } else { "translated+built" };
        } else {
            c.native_failed.fetch_add(1, Ordering::Relaxed);
            result = "native-failed";
        }
        // Native replacements (`crate::subst`) the launch path would build for this kernel.
        for k in &translation.kernels {
            if let crate::subst::Decision::Substitute(plan) = crate::subst::decide(&k.name, &k.params, &crate::subst::Switch::from_env())
                && let Ok(words) = plan.build(ptx::dot_product_enabled())
                // The stream-k fixup replacement has no module (it is skipped at launch); an empty
                // SPIR-V buffer must never reach the driver, which exits the process on it.
                && words.len() > 5
                && words[0] == 0x0723_0203
            {
                if compiler.is_cached(&words) {
                    c.native_cached.fetch_add(1, Ordering::Relaxed);
                } else if compiler.precompile(&words) {
                    c.subst_built.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        compile_ms = t1.elapsed().as_millis();
    }
    (result, translate_ms, compile_ms, String::new())
}

/// `cfi_precompile_main(argc, argv)`: the `cfi-precompile` command line, run inside this
/// library so the translator that fills the caches is the one that reads them. `argv[0]` is
/// the program name. Returns 0 on success, 1 on usage or I/O errors, 2 when the device could
/// not be opened.
///
/// # Safety
/// `argv` must point to `argc` valid NUL-terminated strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cfi_precompile_main(argc: c_int, argv: *const *const c_char) -> c_int {
    let args: Vec<String> = (1..argc.max(0) as usize)
        .filter_map(|i| {
            let p = unsafe { *argv.add(i) };
            (!p.is_null()).then(|| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
        })
        .collect();
    let options = match parse_args(&args) {
        Ok(o) => o,
        Err(e) => {
            if !e.is_empty() {
                eprintln!("cfi-precompile: {e}");
            }
            eprintln!("{USAGE}");
            return if e.is_empty() { 0 } else { 1 };
        }
    };
    let mut out = std::io::stdout();
    match run(&options, &mut out) {
        Ok(_) => 0,
        Err(e) => {
            eprintln!("cfi-precompile: {e}");
            if e.contains("cuInit") || e.contains("device") { 2 } else { 1 }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(ptx: &str) -> Vec<u8> {
        // One uncompressed PTX record (layout as read by crates/fatbin).
        let payload = ptx.as_bytes();
        let mut rec = vec![0u8; 64];
        rec[..2].copy_from_slice(&1u16.to_le_bytes()); // kind: PTX
        rec[2..4].copy_from_slice(&0x101u16.to_le_bytes());
        rec[4..8].copy_from_slice(&64u32.to_le_bytes()); // header size
        rec[8..12].copy_from_slice(&(payload.len() as u32).to_le_bytes());
        rec.extend_from_slice(payload);
        let mut f = Vec::new();
        f.extend_from_slice(&cfi_fatbin::FRAME_MAGIC.to_le_bytes());
        f.extend_from_slice(&1u16.to_le_bytes());
        f.extend_from_slice(&16u16.to_le_bytes());
        f.extend_from_slice(&(rec.len() as u64).to_le_bytes());
        f.extend_from_slice(&rec);
        // nvcc aligns frames in `.nv_fatb` to 8 bytes.
        f.resize(f.len().next_multiple_of(8), 0);
        f
    }

    #[test]
    fn schedule_starts_every_module_before_the_rest() {
        let job = |module: usize, entry: &str| Job { module, entry: entry.into(), native: false };
        // Module sizes: 0 is small, 1 is big, 2 is medium.
        let sizes = [10usize, 300, 200];
        let mut jobs = vec![job(0, "a0"), job(0, "a1"), job(1, "b0"), job(1, "b1"), job(1, "b2"), job(2, "c0"), job(2, "c1")];
        schedule_jobs(&mut jobs, |m| sizes[m]);
        let order: Vec<&str> = jobs.iter().map(|j| j.entry.as_str()).collect();
        // First one entry per module, biggest module first, then the others in the same big-first order.
        assert_eq!(order, ["b0", "c0", "a0", "b1", "b2", "c1", "a1"]);
        let mut one = vec![job(0, "x")];
        schedule_jobs(&mut one, |_| 1);
        assert_eq!(one.len(), 1);
        schedule_jobs(&mut Vec::new(), |_| 1);
    }

    #[test]
    fn args() {
        let o = parse_args(&["--translate-only".into(), "--threads".into(), "3".into(), "x.dll".into()]).unwrap();
        assert!(o.translate_only);
        assert_eq!(o.threads, Some(3));
        assert_eq!(o.paths.len(), 1);
        assert!(parse_args(&[]).is_err());
        assert!(parse_args(&["--bogus".into(), "a".into()]).is_err());
        assert!(replaced_by_stand_in(std::path::Path::new("x/cublasLt64_13.dll")));
        assert!(!replaced_by_stand_in(std::path::Path::new("x/ggml-cuda.dll")));
        assert!(!replaced_by_stand_in(std::path::Path::new("x/cusolver64_11.dll")), "NVIDIA cuSOLVER runs on our cuBLAS");
    }

    #[test]
    fn ggml_type_selection() {
        let mmq = "_Z9mul_mat_qIL9ggml_type8ELi128ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii";
        let fa = "_Z18flash_attn_ext_vecILi256ELi2EL9ggml_type1ELS0_8ELb1EEvPKc";
        assert_eq!(ggml_type_args(mmq), vec![8]);
        assert_eq!(ggml_type_args(fa), vec![1, 8]);
        assert!(ggml_type_args("_Z12rms_norm_f32ILi256ELb1ELb0EEvPKfPf").is_empty());
        let o = parse_args(&["--ggml-types".into(), "q8_0".into(), "x".into()]).unwrap();
        assert_eq!(o.ggml_types, Some(vec![0, 1, 8]));
        assert_eq!(parse_args(&["--ggml-types".into(), "all".into(), "x".into()]).unwrap().ggml_types, None);
        assert!(wants_native(&o, mmq) && wants_native(&o, fa) && wants_native(&o, "_Z12rms_norm_f32"));
        assert!(!wants_native(&o, "_Z9mul_mat_qIL9ggml_type12ELi128ELb0EEv"));
        let o = parse_args(&["--native-skip".into(), "flash_attn".into(), "x".into()]).unwrap();
        assert!(!wants_native(&o, fa) && wants_native(&o, mmq));
        assert!(parse_args(&["--ggml-types".into(), "q9_z".into(), "x".into()]).is_err());
    }

    #[test]
    fn gguf_header_types() {
        // GGUF v3: one string KV, one u32-array KV, two tensors (q8_0 and f32).
        let mut g = b"GGUF".to_vec();
        g.extend(3u32.to_le_bytes());
        g.extend(2u64.to_le_bytes());
        g.extend(2u64.to_le_bytes());
        let s = |g: &mut Vec<u8>, t: &str| {
            g.extend((t.len() as u64).to_le_bytes());
            g.extend(t.as_bytes());
        };
        s(&mut g, "general.name");
        g.extend(8u32.to_le_bytes());
        s(&mut g, "tiny");
        s(&mut g, "x.arr");
        g.extend(9u32.to_le_bytes());
        g.extend(4u32.to_le_bytes());
        g.extend(3u64.to_le_bytes());
        g.extend([0u8; 12]);
        for (name, ty) in [("blk.0.w", 8u32), ("norm", 0)] {
            s(&mut g, name);
            g.extend(2u32.to_le_bytes());
            g.extend(32u64.to_le_bytes());
            g.extend(4u64.to_le_bytes());
            g.extend(ty.to_le_bytes());
            g.extend(0u64.to_le_bytes());
        }
        let dir = std::env::temp_dir().join(format!("cfi-gguf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("m.gguf"), &g).unwrap();
        std::fs::write(dir.join("not-a-model"), b"hello").unwrap();
        assert_eq!(gguf_tensor_types(&dir.join("m.gguf")).unwrap(), Some(vec![0, 8]));
        assert_eq!(gguf_tensor_types(&dir.join("not-a-model")).unwrap(), None);
        let o = parse_args(&["--gguf".into(), dir.to_string_lossy().into_owned(), "x".into()]).unwrap();
        assert_eq!(o.ggml_types, Some(vec![0, 1, 8]));
        std::fs::write(dir.join("m.gguf"), &g[..g.len() - 5]).unwrap();
        assert!(gguf_tensor_types(&dir.join("m.gguf")).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn scan_finds_frames_and_ignores_truncated_ones() {
        let ptx = ".version 7.0\n.target sm_80\n.address_size 64\n.visible .entry k()\n{\nret;\n}\n";
        let mut image = vec![0u8; 100];
        image.extend(frame(ptx));
        image.extend([0u8; 12]);
        image.extend(frame(ptx));
        let mut truncated = frame(ptx);
        truncated.truncate(40);
        image.extend(truncated);
        let (found, frames) = scan_image(&image);
        assert_eq!(frames, 2, "the truncated frame is not counted");
        assert_eq!(found.len(), 2);
        assert!(found.iter().all(|p| std::str::from_utf8(p).unwrap().starts_with(".version 7.0")));
    }

    /// `CFI_PRECOMPILE_INPUT=<dll or dir> cargo test --release --lib precompile_input -- --ignored --nocapture`
    /// translates everything without a GPU (SPIR-V cache in `CFI_CACHE_DIR`, report in
    /// `CFI_PRECOMPILE_REPORT`).
    #[test]
    #[ignore]
    fn precompile_input() {
        let Some(input) = std::env::var_os("CFI_PRECOMPILE_INPUT") else {
            return;
        };
        let mut args = vec!["--translate-only".to_string(), input.to_string_lossy().into_owned()];
        if let Some(r) = std::env::var_os("CFI_PRECOMPILE_REPORT") {
            args.extend(["--report".into(), r.to_string_lossy().into_owned()]);
        }
        let o = parse_args(&args).unwrap();
        let mut out = std::io::stdout();
        let s = run(&o, &mut out).unwrap();
        assert!(s.entries > 0);
        assert_eq!(s.translated + s.xcache_hit + s.rejected, s.entries);
    }

    /// A precompiled entry is found under the key the launch path computes, holds the same
    /// SPIR-V as a fresh translation, and a translator switch in the environment makes it a miss.
    #[test]
    fn precompiled_entry_matches_launch_path_key() {
        let _guard = xcache::TEST_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let dir = std::env::temp_dir().join(format!("cfi-precompile-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // SAFETY: serialised with the other cache-directory tests by TEST_ENV_LOCK.
        unsafe { std::env::set_var("CFI_CACHE_DIR", dir.join("spv")) };
        let raw = ".version 7.0\n.target sm_80\n.address_size 64\n\
.visible .entry add1(.param .u64 p)\n{\n.reg .b32 %r<3>;\n.reg .b64 %rd<3>;\n\
ld.param.u64 %rd1, [p];\ncvta.to.global.u64 %rd2, %rd1;\nld.global.u32 %r1, [%rd2];\n\
add.s32 %r2, %r1, 1;\nst.global.u32 [%rd2], %r2;\nret;\n}\n";
        let file = dir.join("m.ptx");
        std::fs::write(&file, raw).unwrap();
        // 49152 is the translator's default window, so other tests see no switch change.
        let args = ["--translate-only", "--quiet", "--dynamic-shared", "49152", file.to_str().unwrap()].map(String::from);
        let mut sink = std::io::sink();
        let s = run(&parse_args(&args).unwrap(), &mut sink).unwrap();
        assert_eq!((s.entries, s.translated, s.rejected), (1, 1, 0));
        let hit = xcache::get(xcache::key(raw, "add1")).expect("launch-path key hits");
        ptx::set_lenient_undefined(true);
        let fresh = ptx::translate_prepared_entry(&ptx::prepare_module(raw).unwrap(), "add1").unwrap();
        ptx::set_lenient_undefined(false);
        assert_eq!(hit.words, fresh.words);
        // Second run: everything is already cached.
        let s = run(&parse_args(&args).unwrap(), &mut sink).unwrap();
        assert_eq!((s.translated, s.xcache_hit), (0, 1));
        // A translator switch changes the key; a launcher variable does not.
        let before = xcache::key(raw, "add1");
        unsafe { std::env::set_var("CFI_REDIRECT_DIR", "x") };
        assert_eq!(xcache::key(raw, "add1"), before);
        unsafe { std::env::set_var("CFI_OPT_TEST_SWITCH", "1") };
        assert_ne!(xcache::key(raw, "add1"), before);
        unsafe {
            std::env::remove_var("CFI_OPT_TEST_SWITCH");
            std::env::remove_var("CFI_REDIRECT_DIR");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
