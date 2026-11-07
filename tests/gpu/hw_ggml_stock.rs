//! Compile coverage of every ggml CUDA kernel found in the stock Ollama plug-in's fatbins.
//! Set CFI_STOCK_GGML_DLL to the stock ggml-cuda.dll. Entries are grouped by kernel family;
//! up to CFI_STOCK_PER_FAMILY (default 3) instantiations per family are compiled on the GPU.
use super::{Handle, library, ok, symbols, test_device};
use std::collections::BTreeMap;
use std::ffi::{c_char, c_void};
use std::ptr::null_mut;

pub fn extract_all(path: &str) -> Vec<String> {
    let data = std::fs::read(path).expect("read dll");
    let magic = cfi_fatbin::FRAME_MAGIC.to_le_bytes();
    let mut blobs = Vec::new();
    let mut at = 0;
    while let Some(pos) = data[at..].windows(4).position(|w| w == magic) {
        let start = at + pos;
        if let Some(ptx) = unsafe { cfi_fatbin::extract_ptx(data[start..].as_ptr()) } {
            if let Ok(text) = String::from_utf8(ptx) {
                if text.contains(".entry") {
                    blobs.push(text);
                }
            }
        }
        at = start + 4;
    }
    blobs
}

fn family(name: &str) -> String {
    // _Z<len><ident>... -> ident
    let digits: String = name.trim_start_matches("_Z").chars().take_while(|c| c.is_ascii_digit()).collect();
    let skip = 2 + digits.len();
    match digits.parse::<usize>() {
        Ok(n) if name.len() >= skip + n => name[skip..skip + n].to_string(),
        _ => name.to_string(),
    }
}

#[test]
#[ignore = "requires CFI_STOCK_GGML_DLL, a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn hw_stock_ggml_kernels_compile() {
    let path = std::env::var("CFI_STOCK_GGML_DLL").expect("CFI_STOCK_GGML_DLL");
    let per_family: usize = std::env::var("CFI_STOCK_PER_FAMILY").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    let blobs = extract_all(&path);
    println!("fatbin PTX blobs: {}", blobs.len());
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuModuleLoadData(*mut Handle, *const c_void),
        cuModuleGetFunction(*mut Handle, Handle, *const c_char),
    );
    ok!(cuInit(0));
    let mut context = null_mut();
    ok!(cuCtxCreate_v2(&mut context, 0, test_device()));
    let mut seen: BTreeMap<String, (usize, usize, usize)> = BTreeMap::new(); // total, tried, ok
    let mut failures = Vec::new();
    let mut total_entries = 0;
    let only: Option<Vec<usize>> = std::env::var("CFI_STOCK_ONLY").ok().map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect());
    for (b, blob) in blobs.iter().enumerate() {
        if only.as_ref().is_some_and(|o| !o.contains(&b)) {
            continue;
        }
        let entries: Vec<String> = blob
            .lines()
            .filter_map(|l| {
                let l = l.trim();
                let l = l.strip_prefix(".visible ").unwrap_or(l);
                l.strip_prefix(".entry ").map(|r| r.trim_end_matches('(').trim().to_string())
            })
            .collect();
        total_entries += entries.len();
        println!("BLOB {b}: {} bytes, {} entries, first {}", blob.len(), entries.len(), entries.first().map(String::as_str).unwrap_or(""));
        if let Ok(dir) = std::env::var("CFI_DUMP_DIR") {
            let _ = std::fs::write(format!("{dir}\\blob{b}.ptx"), blob);
        }
        let src = std::ffi::CString::new(blob.as_str()).unwrap();
        let mut module = null_mut();
        let status = unsafe { cuModuleLoadData(&mut module, src.as_ptr().cast()) };
        if status != 0 {
            println!("blob {b}: cuModuleLoadData = {status}");
            continue;
        }
        for name in entries {
            if name.starts_with("_ZN") {
                continue; // CUB library kernels are not ggml kernels (and one hangs translation)
            }
            let fam = family(&name);
            let e = seen.entry(fam.clone()).or_default();
            e.0 += 1;
            if e.1 >= per_family {
                continue;
            }
            e.1 += 1;
            let cname = std::ffi::CString::new(name.clone()).unwrap();
            let mut kernel = null_mut();
            eprintln!("TRY {name}");
            let status = unsafe { cuModuleGetFunction(&mut kernel, module, cname.as_ptr()) };
            if status == 0 {
                e.2 += 1;
            } else {
                failures.push((fam, name, status));
            }
        }
    }
    println!("entries: {total_entries}, families: {}", seen.len());
    for (fam, (total, tried, good)) in &seen {
        println!("FAMILY {fam}: {good}/{tried} compiled ({total} instantiations)");
    }
    for (fam, name, status) in &failures {
        println!("FAIL {fam} status {status}: {name}");
    }
}
