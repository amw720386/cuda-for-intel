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
