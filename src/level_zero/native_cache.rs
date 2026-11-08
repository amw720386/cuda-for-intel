//! Persistent on-disk cache of Level Zero native module binaries.
//!
//! The Intel compiler spends minutes on large kernels. After a successful `zeModuleCreate`
//! from SPIR-V the driver's native binary is stored here, keyed by a hash of the SPIR-V,
//! the device and driver identity, the build flags and the cache format version. Later
//! runs create the module with `ZE_MODULE_FORMAT_NATIVE` instead. Every failure (missing
//! file, corrupt entry, driver rejection) falls back to compiling the SPIR-V.
//!
//! Environment: `CUDA_FOR_INTEL_CACHE=0` disables the cache, `CUDA_FOR_INTEL_CACHE_DIR`
//! selects the directory (default `%LOCALAPPDATA%\cuda-for-intel\cache`, or
//! `$XDG_CACHE_HOME`/`~/.cache` elsewhere) and `CUDA_FOR_INTEL_CACHE_MAX_MB` caps its size
//! (default 4096; oldest entries are evicted first).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const FORMAT: u32 = 1;
const MAGIC: &[u8; 8] = b"CFINAT01";
const DEFAULT_MAX_MB: u64 = 4096;

pub(crate) fn directory() -> Option<&'static Path> {
    static DIR: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    DIR.get_or_init(|| {
        if std::env::var_os("CUDA_FOR_INTEL_CACHE").as_deref() == Some(std::ffi::OsStr::new("0")) {
            return None;
        }
        let base = if let Some(dir) = std::env::var_os("CUDA_FOR_INTEL_CACHE_DIR") {
            PathBuf::from(dir)
        } else if cfg!(windows) {
            PathBuf::from(std::env::var_os("LOCALAPPDATA")?)
                .join("cuda-for-intel")
                .join("cache")
        } else if let Some(dir) = std::env::var_os("XDG_CACHE_HOME") {
            PathBuf::from(dir).join("cuda-for-intel")
        } else {
            PathBuf::from(std::env::var_os("HOME")?)
                .join(".cache")
                .join("cuda-for-intel")
        };
        let dir = base.join(format!("v{FORMAT}"));
        std::fs::create_dir_all(&dir).ok()?;
        Some(dir)
    })
    .as_deref()
}

/// 128-bit non-cryptographic content hash (two multiply-rotate lanes over 8-byte words).
pub(crate) struct Hasher {
    a: u64,
    b: u64,
    len: u64,
}

impl Hasher {
    pub(crate) fn new() -> Self {
        Self { a: 0x243f_6a88_85a3_08d3, b: 0x1319_8a2e_0370_7344, len: 0 }
    }

    fn word(&mut self, w: u64) {
        self.a = (self.a ^ w).wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(29);
        self.b = (self.b.wrapping_add(w)).wrapping_mul(0xc2b2_ae3d_27d4_eb4f).rotate_left(31) ^ self.a;
    }

    pub(crate) fn bytes(&mut self, data: &[u8]) {
        self.len += data.len() as u64;
        let mut chunks = data.chunks_exact(8);
        for chunk in &mut chunks {
            self.word(u64::from_le_bytes(chunk.try_into().unwrap()));
        }
        let rest = chunks.remainder();
        let mut tail = [0u8; 8];
        tail[..rest.len()].copy_from_slice(rest);
        self.word(u64::from_le_bytes(tail) ^ ((rest.len() as u64) << 56));
    }

    pub(crate) fn finish(mut self) -> String {
        self.word(self.len);
        self.word(!self.len);
        format!("{:016x}{:016x}", self.a, self.b)
    }
}

pub(crate) fn key(words: &[u32], tag: &str, flags: &str) -> String {
    let mut hasher = Hasher::new();
    hasher.bytes(&FORMAT.to_le_bytes());
    hasher.bytes(env!("CARGO_PKG_VERSION").as_bytes());
    hasher.bytes(tag.as_bytes());
    hasher.bytes(&[0]);
    hasher.bytes(flags.as_bytes());
    hasher.bytes(&[0]);
    let bytes = unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), std::mem::size_of_val(words)) };
    hasher.bytes(bytes);
    hasher.finish()
}

fn path(dir: &Path, key: &str) -> PathBuf {
    dir.join(format!("{key}.bin"))
}

fn checksum(payload: &[u8]) -> u64 {
    let mut hasher = Hasher::new();
    hasher.bytes(payload);
    u64::from_str_radix(&hasher.finish()[..16], 16).unwrap_or(0)
}

/// Returns the stored native binary, verifying its header and checksum.
pub(crate) fn load(dir: &Path, key: &str) -> Option<Vec<u8>> {
    let file = path(dir, key);
    let data = std::fs::read(&file).ok()?;
    let valid = data.len() > 24
        && &data[..8] == MAGIC
        && u64::from_le_bytes(data[8..16].try_into().ok()?) == (data.len() - 24) as u64
        && u64::from_le_bytes(data[16..24].try_into().ok()?) == checksum(&data[24..]);
    if !valid {
        let _ = std::fs::remove_file(&file);
        return None;
    }
    // Refresh the modification time so eviction is least-recently-used.
    if let Ok(handle) = std::fs::OpenOptions::new().write(true).open(&file) {
        let _ = handle.set_modified(std::time::SystemTime::now());
    }
    Some(data[24..].to_vec())
}

/// Whether a (not yet verified) entry exists; `load` verifies it on use.
pub(crate) fn contains(dir: &Path, key: &str) -> bool {
    path(dir, key).is_file()
}

pub(crate) fn discard(dir: &Path, key: &str) {
    let _ = std::fs::remove_file(path(dir, key));
}

/// Writes atomically (temporary file, then rename) and trims the cache to its size cap.
pub(crate) fn store(dir: &Path, key: &str, payload: &[u8]) {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let temporary = dir.join(format!(
        "{key}.{}.{}.tmp",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let mut data = Vec::with_capacity(payload.len() + 24);
    data.extend_from_slice(MAGIC);
    data.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    data.extend_from_slice(&checksum(payload).to_le_bytes());
    data.extend_from_slice(payload);
    if std::fs::write(&temporary, &data).is_err() || std::fs::rename(&temporary, path(dir, key)).is_err() {
        let _ = std::fs::remove_file(&temporary);
        return;
    }
    trim(dir);
}

fn trim(dir: &Path) {
    let cap = std::env::var("CUDA_FOR_INTEL_CACHE_MAX_MB")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(DEFAULT_MAX_MB)
        .saturating_mul(1 << 20);
    let Ok(read) = std::fs::read_dir(dir) else { return };
    let mut files: Vec<_> = read
        .flatten()
        .filter_map(|entry| {
            let meta = entry.metadata().ok()?;
            let name = entry.file_name();
            let name = name.to_str()?;
            let stale_temporary = name.ends_with(".tmp");
            let modified = meta.modified().ok()?;
            if stale_temporary && modified.elapsed().is_ok_and(|age| age.as_secs() > 3600) {
                let _ = std::fs::remove_file(entry.path());
                return None;
            }
            name.ends_with(".bin").then(|| (modified, meta.len(), entry.path()))
        })
        .collect();
    let mut total: u64 = files.iter().map(|file| file.1).sum();
    if total <= cap {
        return;
    }
    files.sort();
    for (_, size, file) in files {
        if total <= cap {
            break;
        }
        if std::fs::remove_file(file).is_ok() {
            total -= size;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_detects_corruption_and_trims() {
        let dir = std::env::temp_dir().join(format!("cfi-cache-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let k = key(&[1, 2, 3], "dev", "");
        assert_ne!(k, key(&[1, 2, 4], "dev", ""));
        assert_ne!(k, key(&[1, 2, 3], "dev2", ""));
        assert_ne!(k, key(&[1, 2, 3], "dev", "-x"));
        assert!(load(&dir, &k).is_none());
        store(&dir, &k, b"native binary");
        assert_eq!(load(&dir, &k).unwrap(), b"native binary");
        let mut data = std::fs::read(path(&dir, &k)).unwrap();
        *data.last_mut().unwrap() ^= 1;
        std::fs::write(path(&dir, &k), data).unwrap();
        assert!(load(&dir, &k).is_none());
        assert!(!path(&dir, &k).exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
