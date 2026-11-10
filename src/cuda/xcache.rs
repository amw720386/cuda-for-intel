//! On-disk cache of translated kernels: (prepared PTX hash, entry name) -> SPIR-V words
//! plus kernel metadata. Skips the PTX parse/lowering of multi-megabyte ATen modules on
//! every process start. Independent of the Level Zero native-binary cache (keyed by the
//! SPIR-V hash), which sits behind it.
//!
//! Disabled by `CFI_NO_CACHE=1`; directory `CFI_CACHE_DIR` or
//! `%LOCALAPPDATA%\cuda-for-intel\spvcache` (`$XDG_CACHE_HOME`/`~/.cache` elsewhere).
//! Entries are written atomically (temp file + rename) and carry a checksum; any
//! mismatch is treated as a miss. Only translations without module-scope variables are
//! cached.

use crate::ptx::{KernelMetadata, Param, Translation};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

/// Bump when the translator's output for the same PTX can change.
const TRANSLATOR_VERSION: u32 = 7;

/// NUL-terminated build description read by `scripts/cfi-doctor.ps1`. The translator number is
/// spelled out literally; `build_info_matches_translator_version` keeps it in sync.
static BUILD_INFO: &[u8] = concat!("cfi-version=", env!("CARGO_PKG_VERSION"), ";translator=7\0").as_bytes();

/// Returns a static NUL-terminated string such as `cfi-version=0.1.0;translator=7`.
/// No allocation, no device access: safe to call right after `LoadLibrary`.
#[unsafe(no_mangle)]
pub extern "C" fn cfi_build_info() -> *const std::ffi::c_char {
    BUILD_INFO.as_ptr().cast()
}

#[cfg(test)]
mod build_info_tests {
    #[test]
    fn build_info_matches_translator_version() {
        let text = std::str::from_utf8(super::BUILD_INFO).unwrap();
        assert!(text.ends_with('\0'));
        assert!(text.contains(&format!("translator={};", super::TRANSLATOR_VERSION).replace(';', "\0")));
        assert!(text.contains(env!("CARGO_PKG_VERSION")));
    }
}
const MAGIC: u32 = 0x5850_4331; // "XPC1"

fn enabled() -> bool {
    std::env::var_os("CFI_NO_CACHE").is_none()
}

fn dir() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("CFI_CACHE_DIR") {
        return Some(PathBuf::from(d));
    }
    #[cfg(windows)]
    let base = std::env::var_os("LOCALAPPDATA").map(PathBuf::from)?;
    #[cfg(not(windows))]
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    Some(base.join("cuda-for-intel").join("spvcache"))
}

/// Hash of everything that influences the translation of an entry: the module text, the
/// entry, the crate and translator versions, and every process-wide `CFI_*` translator
/// switch (peepholes, dp4a lowering, ...), excluding tracing and cache controls.
pub(crate) fn key(prepared: &str, entry: &str) -> u64 {
    key_for_hash(module_hash(prepared), entry)
}

/// Fast content hash of a module's text (computed once per module, not once per kernel).
pub(crate) fn module_hash(text: &str) -> u64 {
    let bytes = text.as_bytes();
    let (mut a, mut b) = (0x243f_6a88_85a3_08d3u64, 0x1319_8a2e_0370_7344u64);
    let mut word = |w: u64| {
        a = (a ^ w).wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(29);
        b = b.wrapping_add(w).wrapping_mul(0xc2b2_ae3d_27d4_eb4f).rotate_left(31) ^ a;
    };
    let mut chunks = bytes.chunks_exact(8);
    for c in &mut chunks {
        word(u64::from_le_bytes(c.try_into().unwrap()));
    }
    let rest = chunks.remainder();
    let mut tail = [0u8; 8];
    tail[..rest.len()].copy_from_slice(rest);
    word(u64::from_le_bytes(tail) ^ ((rest.len() as u64) << 56));
    word(bytes.len() as u64);
    a ^ b.rotate_left(17)
}

pub(crate) fn key_for_hash(module: u64, entry: &str) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    module.hash(&mut h);
    entry.hash(&mut h);
    env!("CARGO_PKG_VERSION").hash(&mut h);
    TRANSLATOR_VERSION.hash(&mut h);
    // Process-wide translator switches that change the SPIR-V for the same PTX.
    crate::ptx::dot_product_enabled().hash(&mut h);
    crate::ptx::dynamic_shared_bytes().hash(&mut h);
    crate::ptx::load16_merge_enabled().hash(&mut h);
    crate::ptx::xor_shuffle_enabled().hash(&mut h);
    let mut env: Vec<(String, String)> = std::env::vars()
        .filter(|(k, _)| {
            k.starts_with("CFI_")
                && !matches!(
                    k.as_str(),
                    "CFI_NO_CACHE" | "CFI_CACHE_DIR" | "CFI_PTX_DUMP_DIR" | "CFI_CUBLAS_TRACE" | "CFI_ALLOW_EMPTY_KERNELS" | "CFI_REPORT_CC" | "CFI_NO_ALLOC_TOUCH"
                )
        })
        .collect();
    env.sort();
    env.hash(&mut h);
    h.finish()
}

fn path(key: u64) -> Option<PathBuf> {
    Some(dir()?.join(format!("{key:016x}.xt")))
}

fn checksum(bytes: &[u8]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut h);
    h.finish()
}

struct Writer(Vec<u8>);
impl Writer {
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn str(&mut self, s: &str) {
        self.u32(s.len() as u32);
        self.0.extend_from_slice(s.as_bytes());
    }
}
struct Reader<'a>(&'a [u8]);
impl Reader<'_> {
    fn take(&mut self, n: usize) -> Option<&[u8]> {
        if self.0.len() < n {
            return None;
        }
        let (a, b) = self.0.split_at(n);
        self.0 = b;
        Some(a)
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn str(&mut self) -> Option<String> {
        let n = self.u32()? as usize;
        String::from_utf8(self.take(n)?.to_vec()).ok()
    }
}

fn put_param(w: &mut Writer, p: &Param) {
    match *p {
        Param::U8 => w.u32(0),
        Param::U16 => w.u32(1),
        Param::U32 => w.u32(2),
        Param::F32 => w.u32(3),
        Param::U64 => w.u32(4),
        Param::GlobalPtr => w.u32(5),
        Param::Words { words, align, trim } => {
            w.u32(6);
            w.u32(u32::from(words));
            w.u32(u32::from(align));
            w.u32(u32::from(trim));
        }
        Param::Staged { bytes, align } => {
            w.u32(7);
            w.u32(u32::from(bytes));
            w.u32(u32::from(align));
        }
    }
}

fn get_param(r: &mut Reader) -> Option<Param> {
    Some(match r.u32()? {
        0 => Param::U8,
        1 => Param::U16,
        2 => Param::U32,
        3 => Param::F32,
        4 => Param::U64,
        5 => Param::GlobalPtr,
        6 => Param::Words {
            words: r.u32()? as u16,
            align: r.u32()? as u16,
            trim: r.u32()? as u8,
        },
        7 => Param::Staged { bytes: r.u32()? as u16, align: r.u32()? as u16 },
        _ => return None,
    })
}

pub(crate) fn get(key: u64) -> Option<Translation> {
    if !enabled() {
        return None;
    }
    let data = std::fs::read(path(key)?).ok()?;
    if data.len() < 12 {
        return None;
    }
    let (body, tail) = data.split_at(data.len() - 8);
    if checksum(body) != u64::from_le_bytes(tail.try_into().ok()?) {
        return None;
    }
    let mut r = Reader(body);
    if r.u32()? != MAGIC {
        return None;
    }
    let nwords = r.u32()? as usize;
    let raw = r.take(nwords.checked_mul(4)?)?;
    let words: Vec<u32> = raw.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let nk = r.u32()?;
    let mut kernels = Vec::new();
    for _ in 0..nk {
        let name = r.str()?;
        let np = r.u32()?;
        let params = (0..np).map(|_| get_param(&mut r)).collect::<Option<Vec<_>>>()?;
        let shared_bytes = r.u32()?;
        let max_threads = match r.u32()? {
            0 => None,
            n => Some(n),
        };
        let flags = r.u32()?;
        kernels.push(KernelMetadata {
            name,
            params,
            shared_bytes,
            max_threads,
            device_atomics: flags & 1 != 0,
            fp32: flags & 2 != 0,
            fp32_fma: flags & 4 != 0,
        });
    }
    Some(Translation { words, kernels, globals: Vec::new() })
}

pub(crate) fn put(key: u64, t: &Translation) {
    if !enabled() || !t.globals.is_empty() {
        return;
    }
    let Some(path) = path(key) else { return };
    let mut w = Writer(Vec::with_capacity(t.words.len() * 4 + 256));
    w.u32(MAGIC);
    w.u32(t.words.len() as u32);
    for word in &t.words {
        w.u32(*word);
    }
    w.u32(t.kernels.len() as u32);
    for k in &t.kernels {
        w.str(&k.name);
        w.u32(k.params.len() as u32);
        for p in &k.params {
            put_param(&mut w, p);
        }
        w.u32(k.shared_bytes);
        w.u32(k.max_threads.unwrap_or(0));
        w.u32(u32::from(k.device_atomics) | u32::from(k.fp32) << 1 | u32::from(k.fp32_fma) << 2);
    }
    let sum = checksum(&w.0);
    w.0.extend_from_slice(&sum.to_le_bytes());
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    if std::fs::write(&tmp, &w.0).is_ok() && std::fs::rename(&tmp, &path).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_corruption() {
        let dir = std::env::temp_dir().join(format!("cfi-xcache-test-{}", std::process::id()));
        // SAFETY: tests in this module run single-threaded with respect to this variable.
        unsafe { std::env::set_var("CFI_CACHE_DIR", &dir) };
        let t = Translation {
            words: vec![0x0723_0203, 1, 2, 3],
            kernels: vec![KernelMetadata {
                name: "k".into(),
                params: vec![Param::U32, Param::GlobalPtr, Param::Words { words: 3, align: 8, trim: 1 }, Param::Staged { bytes: 3144, align: 8 }],
                shared_bytes: 64,
                max_threads: Some(128),
                device_atomics: true,
                fp32: false,
                fp32_fma: true,
            }],
            globals: Vec::new(),
        };
        let k = key("module text", "k");
        assert!(get(k).is_none());
        put(k, &t);
        let back = get(k).expect("hit");
        assert_eq!(back.words, t.words);
        assert_eq!(back.kernels[0].params, t.kernels[0].params);
        assert_eq!(back.kernels[0].max_threads, Some(128));
        assert!(back.kernels[0].device_atomics && back.kernels[0].fp32_fma && !back.kernels[0].fp32);
        assert_ne!(key("module text", "k2"), k);
        // Corrupt the file: must be a miss, not a wrong answer.
        let p = path(k).unwrap();
        let mut bytes = std::fs::read(&p).unwrap();
        bytes[10] ^= 0xff;
        std::fs::write(&p, bytes).unwrap();
        assert!(get(k).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
