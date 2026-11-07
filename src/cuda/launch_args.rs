use super::{CuResult, InvalidValue, Opaque};
use crate::level_zero::KernelArg;
use crate::ptx::Param;
use std::ops::Range;
use std::ptr::null_mut;

const BUFFER_POINTER: usize = 1;
const BUFFER_SIZE: usize = 2;

fn layout(params: &[Param]) -> impl Iterator<Item = (Param, Range<usize>)> {
    params.iter().copied().scan(0usize, |offset, param| {
        let start = offset.next_multiple_of(param.alignment());
        *offset = start + param.size();
        Some((param, start..*offset))
    })
}

pub(super) fn size(params: &[Param]) -> usize {
    layout(params).last().map_or(0, |(_, range)| range.end)
}

unsafe fn packed_buffer(extra: *const Opaque) -> Result<(Opaque, usize), CuResult> {
    let (mut buffer, mut size) = (None, None);
    // The supported option list contains at most two pairs and a terminator.
    for index in [0, 2, 4] {
        match unsafe { extra.add(index).read_unaligned() }.addr() {
            0 => {
                let size = size.unwrap_or(0);
                let buffer: Opaque = buffer.unwrap_or(null_mut());
                if size != 0 && buffer.is_null() {
                    return Err(InvalidValue);
                }
                return Ok((buffer, size));
            }
            BUFFER_POINTER if buffer.is_none() => {
                buffer = Some(unsafe { extra.add(index + 1).read_unaligned() });
            }
            BUFFER_SIZE if size.is_none() => {
                let value = unsafe { extra.add(index + 1).read_unaligned() }.cast::<usize>();
                if value.is_null() {
                    return Err(InvalidValue);
                }
                size = Some(unsafe { value.read_unaligned() });
            }
            _ => return Err(InvalidValue),
        }
    }
    Err(InvalidValue)
}

/// The caller must keep the buffers readable through the native argument-setting calls.
pub(super) unsafe fn read(
    params: &[Param],
    kernel_params: *const Opaque,
    extra: *const Opaque,
    valid_pointer: impl Fn(u64) -> bool,
) -> Result<Vec<KernelArg>, CuResult> {
    let packed = if extra.is_null() {
        if kernel_params.is_null() && !params.is_empty() {
            return Err(InvalidValue);
        }
        None
    } else {
        if !kernel_params.is_null() {
            return Err(InvalidValue);
        }
        Some(unsafe { packed_buffer(extra)? })
    };
    let mut args = Vec::with_capacity(params.len());
    for (index, (param, range)) in layout(params).enumerate() {
        let value = if let Some((buffer, size)) = packed {
            if range.end > size {
                return Err(InvalidValue);
            }
            unsafe { buffer.byte_add(range.start) }
        } else {
            unsafe { kernel_params.add(index).read_unaligned() }
        };
        // `args[i]` itself must be non-null (points at the host-side argument word).
        // The device pointer *value* may be null — CUDA kernels receive optional buffers
        // this way (GGML RoPE freq/ext tables), so only reject non-null unknown addresses.
        if value.is_null() {
            return Err(InvalidValue);
        }
        if param == Param::GlobalPtr {
            let address = unsafe { value.cast::<u64>().read_unaligned() };
            if address != 0 && !valid_pointer(address) {
                return Err(InvalidValue);
            }
        }
        // Pass raw byte ranges directly. A CUDA aggregate may contain padding
        // whose bytes were never initialized by the caller.
        match param {
            Param::Words { words, .. } => {
                args.extend((0..usize::from(words)).map(|index| KernelArg {
                    value: unsafe { value.byte_add(index * 4) },
                    size: 4,
                }));
            }
            _ => args.push(KernelArg {
                value,
                size: param.size(),
            }),
        }
    }
    Ok(args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr::{from_mut, without_provenance_mut};

    #[test]
    fn packed_options_and_aggregate_alignment() {
        let params = [
            Param::U32,
            Param::GlobalPtr,
            Param::Words { words: 3, align: 4, trim: 0 },
            Param::U64,
            Param::U8,
        ];
        let mut bytes = [0u8; 41];
        bytes[8..16].copy_from_slice(&42u64.to_ne_bytes());
        bytes[40] = 0xab;
        let mut count = bytes.len();
        let mut short_count = count - 1;
        let buffer = bytes.as_mut_ptr().cast();
        let count_ptr = from_mut(&mut count).cast();
        let pointer_tag = without_provenance_mut(BUFFER_POINTER);
        let size_tag = without_provenance_mut(BUFFER_SIZE);
        let extra = [size_tag, count_ptr, pointer_tag, buffer, null_mut()];
        let read_with = |options: &[Opaque]| unsafe {
            read(&params, null_mut(), options.as_ptr(), |address| {
                address == 42
            })
        };
        assert_eq!(size(&params), count);
        let args = read_with(&extra).ok().unwrap();
        assert_eq!(
            args.iter().map(|arg| arg.size).collect::<Vec<_>>(),
            [4, 8, 4, 4, 4, 8, 1]
        );
        assert_eq!(
            args.iter()
                .map(|arg| arg.value.addr() - buffer.addr())
                .collect::<Vec<_>>(),
            [0, 8, 16, 20, 24, 32, 40]
        );
        for malformed in [
            vec![pointer_tag, buffer, null_mut()],
            vec![size_tag, count_ptr, null_mut()],
            vec![size_tag, null_mut(), null_mut()],
            vec![size_tag, count_ptr, pointer_tag, null_mut(), null_mut()],
            vec![size_tag, count_ptr, size_tag],
            vec![without_provenance_mut(3)],
        ] {
            assert!(matches!(read_with(&malformed), Err(InvalidValue)));
        }
        let mut short = extra;
        short[1] = from_mut(&mut short_count).cast();
        assert!(matches!(read_with(&short), Err(InvalidValue)));
        assert!(matches!(
            unsafe { read(&params, extra.as_ptr(), extra.as_ptr(), |_| true) },
            Err(InvalidValue)
        ));
        let no_args = [null_mut()];
        assert!(
            unsafe { read(&[], null_mut(), no_args.as_ptr(), |_| false) }
                .ok()
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn null_device_pointer_kernel_args_are_accepted() {
        let mut null_ptr = 0u64;
        let mut live = 42u64;
        let params = [Param::GlobalPtr, Param::GlobalPtr];
        let mut kernel_params = [
            from_mut(&mut null_ptr).cast(),
            from_mut(&mut live).cast(),
        ];
        let args = unsafe {
            read(&params, kernel_params.as_ptr(), null_mut(), |address| {
                address == 42
            })
        }
        .expect("null device pointers are valid CUDA kernel args");
        assert_eq!(args.len(), 2);
        assert_eq!(args[0].size, 8);
        assert_eq!(args[1].size, 8);
        // Non-null unknown addresses remain rejected.
        live = 99;
        assert!(matches!(
            unsafe {
                read(&params, kernel_params.as_ptr(), null_mut(), |address| {
                    address == 42
                })
            },
            Err(InvalidValue)
        ));
    }
}
