//! Compile-result cache: process-wide memory map, plus an optional on-disk store selected by
//! `NVRTC_SHIM_CACHE_DIR` (shared across processes, e.g. PyTorch worker pools).
use crate::compiler::{Failure, Output};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Mutex, OnceLock};

type Entry = Result<Output, Failure>;

fn mem() -> &'static Mutex<HashMap<String, Entry>> {
    static M: OnceLock<Mutex<HashMap<String, Entry>>> = OnceLock::new();
    M.get_or_init(Default::default)
}

static HITS: Mutex<u64> = Mutex::new(0);

pub fn hits() -> u64 {
    *HITS.lock().unwrap()
}

fn disk_dir() -> Option<std::path::PathBuf> {
    if std::env::var_os("NVRTC_SHIM_NO_CACHE").is_some() {
        return None;
    }
    std::env::var_os("NVRTC_SHIM_CACHE_DIR").map(Into::into)
}

pub fn enabled() -> bool {
    std::env::var_os("NVRTC_SHIM_NO_CACHE").is_none()
}

pub fn get(key: &str) -> Option<Entry> {
    if !enabled() {
        return None;
    }
    if let Some(e) = mem().lock().unwrap().get(key) {
        *HITS.lock().unwrap() += 1;
        return Some(e.clone());
    }
    let dir = disk_dir()?;
    let text = std::fs::read_to_string(dir.join(format!("{key}.cache"))).ok()?;
    let out = decode(&text)?;
    *HITS.lock().unwrap() += 1;
    mem().lock().unwrap().insert(key.to_owned(), Ok(out.clone()));
    Some(Ok(out))
}

/// Only successful compilations are cached; failures are cheap to retry and may depend on
/// transient environment.
pub fn put(key: &str, entry: &Entry) {
    if !enabled() {
        return;
    }
    let Ok(out) = entry else { return };
    mem().lock().unwrap().insert(key.to_owned(), entry.clone());
    if let Some(dir) = disk_dir() {
        let _ = std::fs::create_dir_all(&dir);
        let tmp = dir.join(format!("{key}.{}.tmp", std::process::id()));
        if std::fs::write(&tmp, encode(out)).is_ok() {
            let _ = std::fs::rename(&tmp, dir.join(format!("{key}.cache")));
        }
    }
}

// Format: "NVRTCSHIM1\n" arch\n nnames\n (len\nname len\nsym)* loglen\nlog ptx-rest
fn encode(o: &Output) -> String {
    let mut s = format!("NVRTCSHIM1\n{}\n{}\n", o.arch, o.lowered.len());
    for (k, v) in &o.lowered {
        s.push_str(&format!("{}\n{k}{}\n{v}", k.len(), v.len()));
    }
    s.push_str(&format!("\n{}\n{}", o.log.len(), o.log));
    s.push_str(&o.ptx);
    s
}

fn decode(t: &str) -> Option<Output> {
    let mut rest = t.strip_prefix("NVRTCSHIM1\n")?;
    let line = |r: &mut &str| -> Option<String> {
        let (a, b) = r.split_once('\n')?;
        let a = a.to_owned();
        *r = b;
        Some(a)
    };
    let arch = line(&mut rest)?.parse().ok()?;
    let n: usize = line(&mut rest)?.parse().ok()?;
    let mut lowered = BTreeMap::new();
    for _ in 0..n {
        let kl: usize = line(&mut rest)?.parse().ok()?;
        let k = rest.get(..kl)?.to_owned();
        rest = &rest[kl..];
        let vl: usize = line(&mut rest)?.parse().ok()?;
        let v = rest.get(..vl)?.to_owned();
        rest = &rest[vl..];
    lowered.insert(k, v);
    }
    rest = rest.strip_prefix('\n')?;
    let ll: usize = line(&mut rest)?.parse().ok()?;
    let log = rest.get(..ll)?.to_owned();
    let ptx = rest.get(ll..)?.to_owned();
    Some(Output { ptx, log, lowered, arch })
}
