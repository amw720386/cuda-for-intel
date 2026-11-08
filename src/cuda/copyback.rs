//! Debug hook: `CFI_DEBUG_COPYBACK=<selector>[,<selector>...]`.
//!
//! After each selected kernel launch the stream is synchronised and every pointer argument
//! that falls inside a device allocation is copied back to the host and summarised on stderr
//! and in the export log: non-finite counts and finite min/max for the bytes read as f32 and
//! as f16, plus an FNV-1a checksum. Selectors: `N` (launch index, 0-based, counted over every
//! non-captured launch), `A-B` (inclusive range), `n:<text>` (kernel-name substring, `|` joins
//! alternatives) or `all`. Nothing but kernel names, sizes and numbers is logged.
//!
//! Companions:
//! * `CFI_DEBUG_COPYBACK_MAX=<bytes>` caps the bytes read per buffer (default 16 MiB; the read
//!   starts at the pointer and stops at the end of its allocation).
//! * `CFI_DEBUG_COPYBACK_DUMP=<dir>` also writes `L<idx>.txt` (grid, block, scalar arguments),
//!   and `L<idx>_a<arg>_pre.bin` / `_post.bin` for replay on the CPU simulator or a second run
//!   with other translator switches. Pre snapshots are taken before the launch.
//! * Variant diffing: run the same workload twice with different switches
//!   (`CFI_OPT_LD16=0`, `CUDA_FOR_INTEL_DP4A=0`, `CUDA_FOR_INTEL_SUBST=none`, ...) and compare
//!   the per-launch checksum lines (`scripts/copyback_diff.py`).

use super::{CudaContext, contains_range};
use crate::level_zero::{Api, KernelArg};
use crate::ptx::Param;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

enum Selector {
    All,
    Index(u64),
    Range(u64, u64),
    Names(Vec<String>),
}

struct Config {
    selectors: Vec<Selector>,
    max_bytes: usize,
    dump: Option<std::path::PathBuf>,
}

fn config() -> Option<&'static Config> {
    static CONFIG: OnceLock<Option<Config>> = OnceLock::new();
    CONFIG
        .get_or_init(|| {
            let spec = std::env::var("CFI_DEBUG_COPYBACK").ok()?;
            let selectors = spec
                .split(',')
                .filter_map(|part| {
                    let part = part.trim();
                    if part == "all" {
                        Some(Selector::All)
                    } else if let Some(names) = part.strip_prefix("n:") {
                        Some(Selector::Names(names.split('|').map(str::to_owned).collect()))
                    } else if let Some((a, b)) = part.split_once('-') {
                        Some(Selector::Range(a.parse().ok()?, b.parse().ok()?))
                    } else {
                        Some(Selector::Index(part.parse().ok()?))
                    }
                })
                .collect();
            let max_bytes = std::env::var("CFI_DEBUG_COPYBACK_MAX")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(16 << 20);
            let dump = std::env::var_os("CFI_DEBUG_COPYBACK_DUMP").map(std::path::PathBuf::from);
            if let Some(dir) = &dump {
                let _ = std::fs::create_dir_all(dir);
            }
            Some(Config { selectors, max_bytes, dump })
        })
        .as_ref()
}

pub(super) fn enabled() -> bool {
    config().is_some()
}

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// One selected launch: its index and the pointer arguments with the bytes read before it.
pub(super) struct Pending {
    index: u64,
    buffers: Vec<Buffer>,
}

struct Buffer {
    arg: usize,
    ptr: usize,
    len: usize,
    pre: Option<Vec<u8>>,
}

fn pointer_value(arg: &KernelArg) -> u64 {
    let mut b = [0u8; 8];
    unsafe { std::ptr::copy_nonoverlapping(arg.value.cast::<u8>(), b.as_mut_ptr(), arg.size.min(8)) };
    u64::from_le_bytes(b)
}

fn read_back(api: &Api, context: &CudaContext, ptr: usize, len: usize) -> Option<Vec<u8>> {
    let mut out = vec![0u8; len];
    unsafe { context.backend.copy(api, out.as_mut_ptr().cast(), std::ptr::with_exposed_provenance(ptr), len) }
        .ok()?;
    Some(out)
}

fn wait(api: &Api, context: &mut CudaContext, stream: usize) {
    let _ = context.streams.wait_stream(api, stream, u64::MAX);
}

/// Called before the launch is submitted. Returns `Some` when this launch is selected.
pub(super) fn before(
    api: &Api,
    context: &mut CudaContext,
    stream: usize,
    name: &str,
    params: &[Param],
    args: &[KernelArg],
) -> Option<Pending> {
    let config = config()?;
    let index = COUNTER.fetch_add(1, Ordering::Relaxed);
    let hit = config.selectors.iter().any(|s| match s {
        Selector::All => true,
        Selector::Index(i) => *i == index,
        Selector::Range(a, b) => (*a..=*b).contains(&index),
        Selector::Names(names) => names.iter().any(|n| name.contains(n.as_str())),
    });
    if !hit {
        return None;
    }
    let mut buffers = Vec::new();
    for (arg, (param, value)) in params.iter().zip(args).enumerate() {
        if !matches!(param, Param::GlobalPtr) {
            continue;
        }
        let ptr = pointer_value(value) as usize;
        let Some((&base, &length)) = context.allocations.range(..=ptr).next_back() else { continue };
        if ptr == 0 || ptr - base >= length || !contains_range(&context.allocations, ptr, 1) {
            continue;
        }
        let len = (length - (ptr - base)).min(config.max_bytes);
        buffers.push(Buffer { arg, ptr, len, pre: None });
    }
    if config.dump.is_some() {
        wait(api, context, stream);
        for buffer in &mut buffers {
            buffer.pre = read_back(api, context, buffer.ptr, buffer.len);
        }
    }
    Some(Pending { index, buffers })
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let man = (h & 0x3ff) as f32;
    match exp {
        0 => sign * man * 2f32.powi(-24),
        31 => if man == 0.0 { sign * f32::INFINITY } else { f32::NAN },
        _ => sign * (1.0 + man / 1024.0) * 2f32.powi(exp - 15),
    }
}

#[derive(Default)]
struct Stats {
    nan: usize,
    inf: usize,
    min: f32,
    max: f32,
    finite: usize,
}

impl Stats {
    fn add(&mut self, v: f32) {
        if v.is_nan() {
            self.nan += 1;
        } else if v.is_infinite() {
            self.inf += 1;
        } else {
            if self.finite == 0 || v < self.min { self.min = v; }
            if self.finite == 0 || v > self.max { self.max = v; }
            self.finite += 1;
        }
    }
}

fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x100000001b3))
}

fn summarize(bytes: &[u8]) -> String {
    let (mut f32s, mut f16s) = (Stats::default(), Stats::default());
    for c in bytes.chunks_exact(4) {
        f32s.add(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
    for c in bytes.chunks_exact(2) {
        f16s.add(f16_to_f32(u16::from_le_bytes([c[0], c[1]])));
    }
    format!(
        "f32[nan={} inf={} min={:.6e} max={:.6e}] f16[nan={} inf={} min={:.4e} max={:.4e}] fnv={:016x}",
        f32s.nan, f32s.inf, f32s.min, f32s.max, f16s.nan, f16s.inf, f16s.min, f16s.max, fnv(bytes)
    )
}

/// Called after the launch was submitted: synchronise, read back, log.
pub(super) fn after(
    api: &Api,
    context: &mut CudaContext,
    stream: usize,
    pending: Pending,
    name: &str,
    geometry: ([u32; 3], [u32; 3]),
    args: &[KernelArg],
) {
    let Some(config) = config() else { return };
    wait(api, context, stream);
    let index = pending.index;
    let short: String = name.chars().take(90).collect();
    let line = format!("CFI copyback L{index} {short} grid={:?} block={:?}", geometry.0, geometry.1);
    log(&line);
    let mut scalars = String::new();
    for (i, arg) in args.iter().enumerate() {
        scalars.push_str(&format!("a{i} size={} value={:#x}\n", arg.size, pointer_value(arg)));
    }
    if let Some(dir) = &config.dump {
        let _ = std::fs::write(dir.join(format!("L{index}.txt")), format!("{name}\ngrid={:?} block={:?}\n{scalars}", geometry.0, geometry.1));
    }
    for buffer in pending.buffers {
        let Some(post) = read_back(api, context, buffer.ptr, buffer.len) else {
            log(&format!("CFI copyback L{index} a{} read failed", buffer.arg));
            continue;
        };
        let changed = buffer.pre.as_ref().map(|pre| pre != &post);
        log(&format!(
            "CFI copyback L{index} a{} bytes={} {}{}",
            buffer.arg,
            buffer.len,
            summarize(&post),
            changed.map_or(String::new(), |c| format!(" written={c}"))
        ));
        if let Some(dir) = &config.dump {
            if let Some(pre) = &buffer.pre {
                let _ = std::fs::write(dir.join(format!("L{index}_a{}_pre.bin", buffer.arg)), pre);
            }
            let _ = std::fs::write(dir.join(format!("L{index}_a{}_post.bin", buffer.arg)), &post);
        }
    }
}

fn log(line: &str) {
    eprintln!("{line}");
    super::export_log_always(line);
}
