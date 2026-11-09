//! Bounded PTX extraction from nvcc fatbins. Header layouts follow ZLUDA.

use std::io::Read;

pub const FRAME_MAGIC: u32 = 0xBA55_ED50;
pub const WRAPPER_MAGIC: u32 = 0x4662_43B1;
const MAX_SIZE: usize = 64 * 1024 * 1024;
const COMPRESSION_MASK: u64 = 0xb000;

/// Return the first extractable 64-bit PTX entry in an nvcc fatbin.
///
/// # Safety
/// The pointer must address a readable fatbin frame or version-one wrapper
/// and its referenced frame. The declared frame bytes must be readable.
pub unsafe fn extract_ptx(image: *const u8) -> Option<Vec<u8>> {
    if image.is_null() {
        return None;
    }
    let frame = match unsafe { image.cast::<u32>().read_unaligned() } {
        WRAPPER_MAGIC => {
            if unsafe { image.add(4).cast::<u32>().read_unaligned() } != 1 {
                return None;
            }
            unsafe { image.add(8).cast::<*const u8>().read_unaligned() }
        }
        FRAME_MAGIC => image,
        _ => return None,
    };
    if frame.is_null() {
        return None;
    }
    let header = unsafe { std::slice::from_raw_parts(frame, 16) };
    if u32::from_le_bytes(field(header, 0)?) != FRAME_MAGIC
        || u16::from_le_bytes(field(header, 4)?) != 1
    {
        return None;
    }
    let header_size = usize::from(u16::from_le_bytes(field(header, 6)?));
    let file_size = usize::try_from(u64::from_le_bytes(field(header, 8)?)).ok()?;
    if header_size < header.len() || !(1..=MAX_SIZE).contains(&file_size) {
        return None;
    }
    let records = unsafe { std::slice::from_raw_parts(frame.add(header_size), file_size) };
    extract_records(records)
}

/// One-line description of the records in a fatbin, for diagnostics.
///
/// # Safety
/// Same contract as [`extract_ptx`].
pub unsafe fn describe(image: *const u8) -> String {
    if image.is_null() {
        return "null".into();
    }
    let frame = match unsafe { image.cast::<u32>().read_unaligned() } {
        WRAPPER_MAGIC => unsafe { image.add(8).cast::<*const u8>().read_unaligned() },
        FRAME_MAGIC => image,
        magic => return format!("magic {magic:#x}"),
    };
    if frame.is_null() {
        return "null frame".into();
    }
    let header = unsafe { std::slice::from_raw_parts(frame, 16) };
    let (Some(hs), Some(fs)) = (field::<2>(header, 6), field::<8>(header, 8)) else {
        return "short".into();
    };
    let header_size = usize::from(u16::from_le_bytes(hs));
    let file_size = (u64::from_le_bytes(fs) as usize).min(MAX_SIZE);
    let mut records = unsafe { std::slice::from_raw_parts(frame.add(header_size), file_size) };
    let mut out = format!("frame hdr={header_size} size={file_size}:");
    while records.len() >= 64 {
        let kind = u16::from_le_bytes(field(records, 0).unwrap());
        let rev = u16::from_le_bytes(field(records, 2).unwrap());
        let hdr = u32::from_le_bytes(field(records, 4).unwrap()) as usize;
        let pay = u32::from_le_bytes(field(records, 8).unwrap()) as usize;
        let arch = u32::from_le_bytes(field(records, 28).unwrap());
        let flags = u64::from_le_bytes(field(records, 40).unwrap());
        out += &format!(" [kind={kind} rev={rev:#x} hdr={hdr} pay={pay} arch={arch} flags={flags:#x}]");
        match hdr.checked_add(pay).and_then(|e| records.get(e..)) {
            Some(rest) if hdr >= 64 => records = rest,
            _ => break,
        }
    }
    out
}


/// Architectures found in a fatbin: `ptx` entries (translatable) and `sass` entries (cubin ELF).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Summary {
    pub ptx: Vec<u32>,
    pub sass: Vec<u32>,
}

/// Documentation page the diagnostics point to.
pub const SASS_DOC: &str = "docs/sass-support.md";

/// Header-only walk of the record kinds and architectures (no decompression).
///
/// # Safety
/// Same contract as [`extract_ptx`].
pub unsafe fn summarize(image: *const u8) -> Option<Summary> {
    if image.is_null() {
        return None;
    }
    let frame = match unsafe { image.cast::<u32>().read_unaligned() } {
        WRAPPER_MAGIC => unsafe { image.add(8).cast::<*const u8>().read_unaligned() },
        FRAME_MAGIC => image,
        _ => return None,
    };
    if frame.is_null() {
        return None;
    }
    let header = unsafe { std::slice::from_raw_parts(frame, 16) };
    let header_size = usize::from(u16::from_le_bytes(field(header, 6)?));
    let file_size = usize::try_from(u64::from_le_bytes(field(header, 8)?)).ok()?.min(MAX_SIZE);
    let records = unsafe { std::slice::from_raw_parts(frame.add(header_size), file_size) };
    Some(summarize_records(records))
}

fn summarize_records(mut records: &[u8]) -> Summary {
    let mut out = Summary::default();
    while records.len() >= 64 {
        let (Some(kind), Some(hdr), Some(pay), Some(arch)) = (
            field(records, 0).map(u16::from_le_bytes),
            field(records, 4).map(u32::from_le_bytes),
            field(records, 8).map(u32::from_le_bytes),
            field(records, 28).map(u32::from_le_bytes),
        ) else {
            break;
        };
        match kind {
            1 if !out.ptx.contains(&arch) => out.ptx.push(arch),
            2 if !out.sass.contains(&arch) => out.sass.push(arch),
            _ => {}
        }
        let (hdr, pay) = (hdr as usize, pay as usize);
        match hdr.checked_add(pay).and_then(|e| records.get(e..)) {
            Some(rest) if hdr >= 64 => records = rest,
            _ => break,
        }
    }
    out.ptx.sort_unstable();
    out.sass.sort_unstable();
    out
}

/// `e_machine` value of a CUDA cubin (EM_CUDA).
pub const EM_CUDA: u16 = 190;

/// Architecture (`sm_XX` number) of a standalone cubin ELF image, `None` when the bytes are not a
/// 64-bit little-endian ELF with `e_machine == 190`. The sm number is the low byte of `e_flags`
/// (0 when unrecorded).
pub fn cubin_arch(image: &[u8]) -> Option<u32> {
    if image.get(..4)? != b"\x7fELF" || *image.get(4)? != 2 {
        return None;
    }
    let machine = u16::from_le_bytes(field(image, 18)?);
    if machine != EM_CUDA {
        return None;
    }
    Some(u32::from_le_bytes(field(image, 48)?) & 0xff)
}

fn arch_list(archs: &[u32]) -> String {
    if archs.is_empty() {
        return "none".into();
    }
    archs.iter().map(|a| format!("sm_{a}")).collect::<Vec<_>>().join(", ")
}

/// NVIDIA libraries cuda-for-intel ships a stand-in for (file-name prefixes, lower case). A
/// SASS-only image inside one of these means the process loaded NVIDIA's copy instead of ours.
const OWN_STAND_INS: [&str; 7] = ["cublaslt64_", "cublas64_", "cudnn", "cufft64_", "cusparse64_", "cusolver64_", "nvrtc64_"];

fn is_stand_in(module: &str) -> bool {
    let lower = module.to_ascii_lowercase();
    lower.ends_with(".dll") && OWN_STAND_INS.iter().any(|p| lower.starts_with(p))
}

/// Sentence naming the module that carries the SASS-only image, or "this module" when unknown.
fn carrier(module: Option<&str>, archs: &str) -> String {
    match module {
        Some(m) if is_stand_in(m) => format!(
            "{m} carries SASS only ({archs}) and no PTX; use cuda-for-intel's stand-in for it (scripts/cfi.ps1 \
             install) or start the program with cfi-run, which loads ours even when the program asks for NVIDIA's \
             copy by full path."
        ),
        Some(m) => format!("{m} carries SASS only ({archs}) and no PTX."),
        None => format!("this module has SASS (cubin) for {archs} but no PTX."),
    }
}

/// The actionable message for a module that has SASS but no PTX. `module` is the file name of
/// the DLL or executable that carries the image (see [`offending_module`]); it appears once.
pub fn no_ptx_message(origin: &str, summary: &Summary, module: Option<&str>) -> String {
    let at = module.map(|m| format!(" (in {m})")).unwrap_or_default();
    if summary.sass.is_empty() {
        return format!(
            "CUDA for Intel: {origin}: the fatbin{at} contains neither PTX nor SASS entries we can read \
             (PTX: {}); see {SASS_DOC}.",
            arch_list(&summary.ptx)
        );
    }
    let archs = arch_list(&summary.sass);
    if module.is_some_and(is_stand_in) {
        return format!(
            "CUDA for Intel: {origin}: {} This runtime translates PTX and cannot execute SASS \
             (CUDA_ERROR_NO_BINARY_FOR_GPU). See {SASS_DOC}.",
            carrier(module, &archs)
        );
    }
    format!(
        "CUDA for Intel: {origin}: {} This runtime translates PTX and cannot execute SASS, so the kernels cannot \
         run on this GPU (CUDA_ERROR_NO_BINARY_FOR_GPU). Use a build that embeds PTX (PyTorch Windows cu118 wheel, \
         conda-forge builds, or rebuild with TORCH_CUDA_ARCH_LIST=\"8.0+PTX\" / \
         -gencode arch=compute_80,code=compute_80). See {SASS_DOC}; scripts/ptxscan.py checks a binary.",
        carrier(module, &archs)
    )
}

/// Message for a standalone cubin ELF image.
pub fn cubin_message(origin: &str, arch: u32, module: Option<&str>) -> String {
    let what = match module {
        Some(m) if is_stand_in(m) => format!(
            "{m} carries SASS only (a standalone cubin ELF, e_machine 190, sm_{arch}); use cuda-for-intel's \
             stand-in for it (scripts/cfi.ps1 install) or start the program with cfi-run."
        ),
        Some(m) => format!("{m} passed a standalone cubin ELF (e_machine 190, sm_{arch}) with no PTX."),
        None => format!("image is a standalone cubin ELF (e_machine 190, sm_{arch}) with no PTX."),
    };
    format!(
        "CUDA for Intel: {origin}: {what} This runtime translates PTX and cannot execute SASS \
         (CUDA_ERROR_NO_BINARY_FOR_GPU). Rebuild with PTX embedded (-gencode \
         arch=compute_80,code=compute_80) or load a PTX/fatbin-with-PTX image. See {SASS_DOC}."
    )
}

/// Modules already reported ("" stands for "module unknown").
static REPORTED: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// Print `message` to stderr the first time `key` is seen in this process. Returns whether it printed.
pub fn report_once_for(key: &str, message: &str) -> bool {
    let mut seen = REPORTED.lock().unwrap_or_else(|e| e.into_inner());
    let key = key.to_ascii_lowercase();
    if seen.contains(&key) {
        return false;
    }
    seen.push(key);
    drop(seen);
    eprintln!("{message}");
    true
}

/// Print `message` once per process for images whose module is unknown.
pub fn report_once(message: &str) -> bool {
    report_once_for("", message)
}

/// Log a one-time actionable error for a fatbin without extractable PTX, naming the module that
/// carries it. Always on (not only under tracing); once per offending module. Returns the
/// message when it was printed.
///
/// # Safety
/// Same contract as [`extract_ptx`].
pub unsafe fn report_no_ptx(origin: &str, image: *const u8) -> Option<String> {
    let summary = unsafe { summarize(image) }.unwrap_or_default();
    let module = offending_module(image);
    let message = no_ptx_message(origin, &summary, module.as_deref());
    report_once_for(module.as_deref().unwrap_or(""), &message).then_some(message)
}

/// Same for a raw image whose first byte is ELF.
pub fn report_cubin(origin: &str, image: &[u8]) -> Option<String> {
    let arch = cubin_arch(image)?;
    let module = offending_module(image.as_ptr());
    let message = cubin_message(origin, arch, module.as_deref());
    report_once_for(module.as_deref().unwrap_or(""), &message).then_some(message)
}

/// File name of the DLL or executable that handed us `image`: the loaded module whose mapped
/// range contains the image (fatbins live in the carrier's `.nv_fatb` section), or, when the
/// image is a heap copy, the first module on the call stack that is neither this one nor a
/// Windows system DLL (the caller of `__cudaRegisterFatBinary` / `cuModuleLoadData`).
pub fn offending_module(image: *const u8) -> Option<String> {
    if let Some(m) = module::containing(image) {
        return Some(m.name);
    }
    let own = module::containing(offending_module as *const u8);
    module::first_foreign_caller(own.as_ref()).map(|m| m.name)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Module {
    /// Full path, compared to recognise our own module.
    path: String,
    /// File name, printed.
    name: String,
}

impl Module {
    fn from_path(path: String) -> Option<Self> {
        let name = path.rsplit(['\\', '/']).next().unwrap_or(&path).to_string();
        (!name.is_empty()).then_some(Self { path, name })
    }
}

#[cfg(windows)]
mod module {
    use super::Module;
    use std::ffi::c_void;

    const FROM_ADDRESS: u32 = 0x4;
    const UNCHANGED_REFCOUNT: u32 = 0x2;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetModuleHandleExW(flags: u32, address: *const u16, module: *mut *mut c_void) -> i32;
        fn GetModuleFileNameW(module: *mut c_void, buffer: *mut u16, size: u32) -> u32;
        fn RtlCaptureStackBackTrace(skip: u32, count: u32, frames: *mut *mut c_void, hash: *mut u32) -> u16;
    }

    pub(super) fn containing(address: *const u8) -> Option<Module> {
        if address.is_null() {
            return None;
        }
        let mut handle = std::ptr::null_mut();
        if unsafe { GetModuleHandleExW(FROM_ADDRESS | UNCHANGED_REFCOUNT, address.cast(), &mut handle) } == 0 {
            return None;
        }
        let mut buf = vec![0u16; 32768];
        let n = unsafe { GetModuleFileNameW(handle, buf.as_mut_ptr(), buf.len() as u32) } as usize;
        (n > 0 && n < buf.len()).then(|| Module::from_path(String::from_utf16_lossy(&buf[..n]))).flatten()
    }

    pub(super) fn first_foreign_caller(own: Option<&Module>) -> Option<Module> {
        let mut frames = [std::ptr::null_mut(); 48];
        let n = unsafe { RtlCaptureStackBackTrace(1, frames.len() as u32, frames.as_mut_ptr(), std::ptr::null_mut()) };
        let system = std::env::var("SystemRoot").map(|s| s.to_ascii_lowercase()).unwrap_or_default();
        frames[..usize::from(n)].iter().filter_map(|&f| containing(f.cast())).find(|m| {
            Some(m) != own && (system.is_empty() || !m.path.to_ascii_lowercase().starts_with(&system))
        })
    }
}

#[cfg(unix)]
mod module {
    use super::Module;
    use std::ffi::{CStr, c_char, c_int, c_void};

    #[repr(C)]
    struct DlInfo {
        fname: *const c_char,
        fbase: *mut c_void,
        sname: *const c_char,
        saddr: *mut c_void,
    }

    unsafe extern "C" {
        fn dladdr(address: *const c_void, info: *mut DlInfo) -> c_int;
    }

    pub(super) fn containing(address: *const u8) -> Option<Module> {
        if address.is_null() {
            return None;
        }
        let mut info = DlInfo { fname: std::ptr::null(), fbase: std::ptr::null_mut(), sname: std::ptr::null(), saddr: std::ptr::null_mut() };
        if unsafe { dladdr(address.cast(), &mut info) } == 0 {
            return None;
        }
        let mut path = if info.fname.is_null() { String::new() } else { unsafe { CStr::from_ptr(info.fname) }.to_string_lossy().into_owned() };
        if path.is_empty() {
            // glibc may leave the main program's name empty.
            path = std::env::current_exe().ok()?.to_string_lossy().into_owned();
        }
        Module::from_path(path)
    }

    /// Return-address walking is only wired up on Windows (where the DLLs run).
    pub(super) fn first_foreign_caller(_own: Option<&Module>) -> Option<Module> {
        None
    }
}

#[cfg(not(any(windows, unix)))]
mod module {
    use super::Module;
    pub(super) fn containing(_: *const u8) -> Option<Module> {
        None
    }
    pub(super) fn first_foreign_caller(_: Option<&Module>) -> Option<Module> {
        None
    }
}

fn field<const N: usize>(bytes: &[u8], offset: usize) -> Option<[u8; N]> {
    bytes.get(offset..offset + N)?.try_into().ok()
}

fn extract_records(mut records: &[u8]) -> Option<Vec<u8>> {
    while records.len() >= 64 {
        let kind = u16::from_le_bytes(field(records, 0)?);
        let revision = u16::from_le_bytes(field(records, 2)?);
        let header = u32::from_le_bytes(field(records, 4)?) as usize;
        let payload = u32::from_le_bytes(field(records, 8)?) as usize;
        if header < 64 || revision != 0x101 {
            return None;
        }
        let end = header.checked_add(payload)?;
        let contents = records.get(header..end)?;
        if kind == 1
            && let Some(ptx) = decode_ptx(records, contents)
        {
            return Some(ptx);
        }
        records = records.get(end..)?;
    }
    None
}

fn decode_ptx(header: &[u8], payload: &[u8]) -> Option<Vec<u8>> {
    let compressed = u32::from_le_bytes(field(header, 16)?) as usize;
    let flags = u64::from_le_bytes(field(header, 40)?);
    let decoded = usize::try_from(u64::from_le_bytes(field(header, 56)?)).ok()?;
    let source = if compressed == 0 {
        payload
    } else {
        payload.get(..compressed)?
    };
    if decoded > MAX_SIZE {
        return None;
    }
    let mut ptx = match flags & COMPRESSION_MASK {
        0 => payload
            .get(..if decoded == 0 { payload.len() } else { decoded })?
            .to_vec(),
        0x1000 => {
            let mut output = Vec::new();
            flate2::read::ZlibDecoder::new(source)
                .take(MAX_SIZE as u64 + 1)
                .read_to_end(&mut output)
                .ok()?;
            output
        }
        0x2000 => lz4_flex::block::decompress(source, decoded).ok()?,
        0x8000 => zstd::bulk::decompress(source, decoded.max(1)).ok()?,
        _ => return None,
    };
    if ptx.len() > MAX_SIZE || (decoded != 0 && ptx.len() != decoded) {
        return None;
    }
    ptx.truncate(ptx.iter().rposition(|&byte| byte != 0)? + 1);
    ptx.windows(8)
        .any(|text| text == b".version")
        .then_some(ptx)
}

#[cfg(test)]
mod tests {
    use super::*;


    fn record(kind: u16, arch: u32) -> Vec<u8> {
        let mut r = vec![0u8; 64 + 8];
        r[..2].copy_from_slice(&kind.to_le_bytes());
        r[2..4].copy_from_slice(&0x101u16.to_le_bytes());
        r[4..8].copy_from_slice(&64u32.to_le_bytes());
        r[8..12].copy_from_slice(&8u32.to_le_bytes());
        r[28..32].copy_from_slice(&arch.to_le_bytes());
        r
    }

    #[test]
    fn sass_only_frame_is_summarized_and_message_names_arches() {
        let mut records = record(2, 80);
        records.extend(record(2, 75));
        records.extend(record(2, 80));
        assert!(extract_records(&records).is_none());
        let s = summarize_records(&records);
        assert_eq!(s, Summary { ptx: vec![], sass: vec![75, 80] });
        let m = no_ptx_message("unit", &s, None);
        assert!(m.contains("sm_75, sm_80") && m.contains("docs/sass-support.md") && m.contains("+PTX"));
    }

    /// A SASS-only fatbin frame (sm_80 and sm_86 cubin records, no PTX) in this test binary's
    /// read-only data, like `.nv_fatb` in a real DLL.
    const fn sass_only_frame() -> [u8; 16 + 2 * 72] {
        let mut f = [0u8; 16 + 2 * 72];
        let magic = FRAME_MAGIC.to_le_bytes();
        let mut i = 0;
        while i < 4 {
            f[i] = magic[i];
            i += 1;
        }
        f[4] = 1; // version
        f[6] = 16; // header size
        f[8] = 2 * 72; // records size
        let mut r = 0;
        while r < 2 {
            let o = 16 + r * 72;
            f[o] = 2; // kind: SASS
            f[o + 2] = 0x01;
            f[o + 3] = 0x01; // revision 0x101
            f[o + 4] = 64; // record header size
            f[o + 8] = 8; // payload size
            f[o + 28] = if r == 0 { 80 } else { 86 };
            r += 1;
        }
        f
    }
    static SASS_ONLY: [u8; 16 + 2 * 72] = sass_only_frame();

    #[test]
    fn sass_only_diagnostic_names_the_carrying_module_once() {
        let image = std::hint::black_box(SASS_ONLY.as_ptr());
        assert!(unsafe { extract_ptx(image) }.is_none());
        let exe = std::env::current_exe().unwrap();
        let name = exe.file_name().unwrap().to_string_lossy().into_owned();
        assert_eq!(offending_module(image).as_deref(), Some(name.as_str()));
        let message = unsafe { report_no_ptx("unit", image) }.expect("first report prints");
        assert_eq!(message.matches(name.as_str()).count(), 1, "{message}");
        assert!(message.contains("sm_80, sm_86") && message.contains("carries SASS only"), "{message}");
        // Same module again: not printed a second time.
        assert!(unsafe { report_no_ptx("unit", image) }.is_none());
    }

    #[test]
    fn stand_in_library_message_points_at_ours_and_cfi_run() {
        let s = Summary { ptx: vec![], sass: vec![90] };
        let m = no_ptx_message("cuModuleLoadData", &s, Some("cublas64_12.dll"));
        assert_eq!(m.matches("cublas64_12.dll").count(), 1, "{m}");
        assert!(m.contains("cfi-run") && m.contains("sm_90"), "{m}");
        let m = no_ptx_message("x", &s, Some("torch_cuda.dll"));
        assert_eq!(m.matches("torch_cuda.dll").count(), 1, "{m}");
        assert!(m.contains("+PTX") && !m.contains("cfi-run"), "{m}");
        let c = cubin_message("x", 80, Some("cudnn_ops64_9.dll"));
        assert!(c.contains("cudnn_ops64_9.dll carries SASS only") && c.contains("cfi-run"), "{c}");
    }

    #[test]
    fn mixed_frame_lists_both_kinds() {
        let mut records = record(1, 37);
        records.extend(record(2, 50));
        let s = summarize_records(&records);
        assert_eq!((s.ptx, s.sass), (vec![37], vec![50]));
    }

    #[test]
    fn cubin_elf_machine_190_detected() {
        let mut elf = vec![0u8; 64];
        elf[..4].copy_from_slice(b"\x7fELF");
        elf[4] = 2;
        elf[18..20].copy_from_slice(&190u16.to_le_bytes());
        elf[48..52].copy_from_slice(&0x0050_0250u32.to_le_bytes());
        assert_eq!(cubin_arch(&elf), Some(0x50));
        elf[18] = 62;
        assert_eq!(cubin_arch(&elf), None);
        assert_eq!(cubin_arch(b"hello"), None);
    }

    #[test]
    fn extracts_lz4_and_zstd_records_with_padding() {
        let ptx = b".version 8.8\n.target sm_75\n.address_size 64\n\0";
        for (flags, compressed) in [
            (0x2000u64, lz4_flex::block::compress(ptx)),
            (0x8000u64, zstd::bulk::compress(ptx, 1).unwrap()),
        ] {
            let payload_size = compressed.len().next_multiple_of(8);
            let mut record = vec![0; 64 + payload_size];
            record[..2].copy_from_slice(&1u16.to_le_bytes());
            record[2..4].copy_from_slice(&0x101u16.to_le_bytes());
            record[4..8].copy_from_slice(&64u32.to_le_bytes());
            record[8..12].copy_from_slice(&(payload_size as u32).to_le_bytes());
            record[16..20].copy_from_slice(&(compressed.len() as u32).to_le_bytes());
            record[40..48].copy_from_slice(&flags.to_le_bytes());
            record[56..64].copy_from_slice(&(ptx.len() as u64).to_le_bytes());
            record[64..64 + compressed.len()].copy_from_slice(&compressed);
            assert_eq!(
                extract_records(&record).as_deref(),
                Some(&ptx[..ptx.len() - 1])
            );
            assert!(extract_records(&record[..record.len() - 1]).is_none());
        }
    }
}
