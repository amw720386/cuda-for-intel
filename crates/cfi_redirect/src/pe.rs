//! Reads the names in a PE file's import directory with a few small positioned reads, so a
//! 1 GB `torch_cuda.dll` costs a handful of kilobytes. Delay-load imports are not listed: they
//! go through `LoadLibrary` at run time, where the hook sees them anyway.

use std::io::{Read, Seek, SeekFrom};

const MAX_IMPORTS: usize = 4096;

fn read_at<R: Read + Seek>(r: &mut R, offset: u64, len: usize) -> Option<Vec<u8>> {
    r.seek(SeekFrom::Start(offset)).ok()?;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).ok()?;
    Some(buf)
}

fn u16_at(b: &[u8], o: usize) -> Option<u16> {
    Some(u16::from_le_bytes(b.get(o..o + 2)?.try_into().ok()?))
}

fn u32_at(b: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?))
}

struct Section {
    va: u32,
    size: u32,
    raw: u32,
}

/// DLL names imported by the PE image in `r` (as written in the file), or `None` when `r` is not
/// a PE image we can parse.
pub fn imports<R: Read + Seek>(r: &mut R) -> Option<Vec<String>> {
    let dos = read_at(r, 0, 64)?;
    if dos.get(..2)? != b"MZ" {
        return None;
    }
    let nt = u64::from(u32_at(&dos, 0x3c)?);
    let head = read_at(r, nt, 24)?;
    if head.get(..4)? != b"PE\0\0" {
        return None;
    }
    let sections = usize::from(u16_at(&head, 6)?).min(96);
    let opt_size = usize::from(u16_at(&head, 20)?);
    let opt = read_at(r, nt + 24, opt_size)?;
    let dirs = match u16_at(&opt, 0)? {
        0x10b => 96,
        0x20b => 112,
        _ => return None,
    };
    let import_rva = u32_at(&opt, dirs + 8)?;
    if import_rva == 0 {
        return Some(Vec::new());
    }
    let table = read_at(r, nt + 24 + opt_size as u64, sections * 40)?;
    let sections: Vec<Section> = table
        .chunks_exact(40)
        .filter_map(|s| {
            let vsize = u32_at(s, 8)?;
            let raw_size = u32_at(s, 16)?;
            Some(Section { va: u32_at(s, 12)?, size: vsize.max(raw_size), raw: u32_at(s, 20)? })
        })
        .collect();
    let offset = |rva: u32| {
        sections
            .iter()
            .find(|s| rva >= s.va && rva - s.va < s.size)
            .map(|s| u64::from(s.raw) + u64::from(rva - s.va))
    };
    let mut names = Vec::new();
    let mut at = offset(import_rva)?;
    for _ in 0..MAX_IMPORTS {
        let desc = read_at(r, at, 20)?;
        let name_rva = u32_at(&desc, 12)?;
        if name_rva == 0 {
            break;
        }
        if let Some(name) = offset(name_rva).and_then(|o| read_name(r, o)) {
            names.push(name);
        }
        at += 20;
    }
    Some(names)
}

fn read_name<R: Read + Seek>(r: &mut R, offset: u64) -> Option<String> {
    r.seek(SeekFrom::Start(offset)).ok()?;
    let mut buf = Vec::with_capacity(32);
    let mut byte = [0u8];
    while buf.len() < 260 {
        r.read_exact(&mut byte).ok()?;
        if byte[0] == 0 {
            return String::from_utf8(buf).ok();
        }
        buf.push(byte[0]);
    }
    None
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::Cursor;

    /// A minimal PE32+ image whose import directory lists `dlls`. One section at RVA 0x1000,
    /// file offset 0x200.
    pub(crate) fn synthetic_pe(dlls: &[&str]) -> Vec<u8> {
        let mut f = vec![0u8; 0x200];
        f[..2].copy_from_slice(b"MZ");
        f[0x3c..0x40].copy_from_slice(&0x80u32.to_le_bytes());
        f[0x80..0x84].copy_from_slice(b"PE\0\0");
        f[0x84..0x86].copy_from_slice(&0x8664u16.to_le_bytes());
        f[0x86..0x88].copy_from_slice(&1u16.to_le_bytes()); // sections
        f[0x94..0x96].copy_from_slice(&240u16.to_le_bytes()); // optional header size
        let opt = 0x98;
        f[opt..opt + 2].copy_from_slice(&0x20bu16.to_le_bytes());
        f[opt + 120..opt + 124].copy_from_slice(&0x1000u32.to_le_bytes()); // import dir rva
        let sec = opt + 240;
        f[sec..sec + 6].copy_from_slice(b".idata");
        f[sec + 8..sec + 12].copy_from_slice(&0x1000u32.to_le_bytes()); // vsize
        f[sec + 12..sec + 16].copy_from_slice(&0x1000u32.to_le_bytes()); // va
        f[sec + 16..sec + 20].copy_from_slice(&0x400u32.to_le_bytes()); // raw size
        f[sec + 20..sec + 24].copy_from_slice(&0x200u32.to_le_bytes()); // raw ptr
        let mut body = vec![0u8; 20 * (dlls.len() + 1)];
        for (i, d) in dlls.iter().enumerate() {
            let rva = 0x1000 + body.len() as u32;
            body[i * 20 + 12..i * 20 + 16].copy_from_slice(&rva.to_le_bytes());
            body.extend_from_slice(d.as_bytes());
            body.push(0);
        }
        body.resize(0x400, 0);
        f.extend(body);
        f
    }

    #[test]
    fn lists_import_names() {
        let pe = synthetic_pe(&["KERNEL32.dll", "cublas64_12.dll", "cublasLt64_12.dll"]);
        assert_eq!(
            imports(&mut Cursor::new(pe)).unwrap(),
            ["KERNEL32.dll", "cublas64_12.dll", "cublasLt64_12.dll"]
        );
    }

    #[test]
    fn rejects_non_pe_and_truncated() {
        assert!(imports(&mut Cursor::new(b"not a pe".to_vec())).is_none());
        let mut pe = synthetic_pe(&["a.dll"]);
        pe.truncate(0x100);
        assert!(imports(&mut Cursor::new(pe)).is_none());
    }
}
