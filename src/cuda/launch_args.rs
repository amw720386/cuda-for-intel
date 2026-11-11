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

/// Bytes of the native argument buffer (a staged aggregate is one pointer).
pub(super) fn abi_size(params: &[Param]) -> usize {
    params.iter().fold(0usize, |offset, param| {
        let alignment = if matches!(param, Param::Staged { .. }) { 8 } else { param.alignment() };
        offset.next_multiple_of(alignment) + param.abi_size()
    })
}

/// Replaces the per-word arguments of each staged aggregate by one pointer argument.
/// `stage(bytes)` copies the struct into device-visible staging memory and returns its address;
/// the returned boxes hold the pointer values and must outlive the argument-setting calls.
pub(super) fn collapse_staged(
    params: &[Param],
    args: Vec<KernelArg>,
    mut stage: impl FnMut(&[u8]) -> Result<u64, CuResult>,
) -> Result<(Vec<KernelArg>, Vec<Box<u64>>), CuResult> {
    let mut out = Vec::with_capacity(args.len());
    let mut keep = Vec::new();
    let mut index = 0;
    for param in params {
        let words = param.host_words();
        let group = args.get(index..index + words).ok_or(InvalidValue)?;
        index += words;
        if let Param::Staged { bytes, .. } = *param {
            // The words are consecutive ranges of one host struct.
            let first = group.first().ok_or(InvalidValue)?.value.cast::<u8>();
            let data = unsafe { std::slice::from_raw_parts(first, usize::from(bytes)) };
            let address = Box::new(stage(data)?);
            out.push(KernelArg { value: std::ptr::from_ref(&*address).cast_mut().cast(), size: 8 });
            keep.push(address);
        } else {
            out.extend_from_slice(group);
        }
    }
    Ok((out, keep))
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

/// One parameter's place in the CUDA argument layout, computed once per function.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Slot {
    /// Byte range of the parameter in a packed (`CU_LAUNCH_PARAM_BUFFER_POINTER`) buffer.
    start: usize,
    end: usize,
    /// Size of the native argument (a scalar or pointer).
    size: usize,
    /// Native 4-byte words of a `Words`/`Staged` aggregate, `None` for a scalar or pointer.
    words: Option<usize>,
    pointer: bool,
}

/// The argument layout of `params` (see `read_planned`).
pub(super) fn plan(params: &[Param]) -> Box<[Slot]> {
    layout(params)
        .map(|(param, range)| {
            let words = match param {
                Param::Words { .. } | Param::Staged { .. } => Some(param.host_words()),
                _ => None,
            };
            Slot {
                start: range.start,
                end: range.end,
                size: param.size(),
                words,
                pointer: param == Param::GlobalPtr,
            }
        })
        .collect()
}

/// The caller must keep the buffers readable through the native argument-setting calls.
pub(super) unsafe fn read(
    params: &[Param],
    kernel_params: *const Opaque,
    extra: *const Opaque,
    valid_pointer: impl Fn(u64) -> bool,
) -> Result<Vec<KernelArg>, CuResult> {
    unsafe { read_planned(&plan(params), kernel_params, extra, valid_pointer) }
}

/// `read` with the layout precomputed by `plan`.
pub(super) unsafe fn read_planned(
    slots: &[Slot],
    kernel_params: *const Opaque,
    extra: *const Opaque,
    valid_pointer: impl Fn(u64) -> bool,
) -> Result<Vec<KernelArg>, CuResult> {
    let packed = if extra.is_null() {
        if kernel_params.is_null() && !slots.is_empty() {
            return Err(InvalidValue);
        }
        None
    } else {
        if !kernel_params.is_null() {
            return Err(InvalidValue);
        }
        Some(unsafe { packed_buffer(extra)? })
    };
    let mut args = Vec::with_capacity(slots.len());
    for (index, slot) in slots.iter().enumerate() {
        let value = if let Some((buffer, size)) = packed {
            if slot.end > size {
                return Err(InvalidValue);
            }
            unsafe { buffer.byte_add(slot.start) }
        } else {
            unsafe { kernel_params.add(index).read_unaligned() }
        };
        // `args[i]` itself must be non-null (points at the host-side argument word).
        // The device pointer *value* may be null — CUDA kernels receive optional buffers
        // this way (GGML RoPE freq/ext tables), so only reject non-null unknown addresses.
        if value.is_null() {
            return Err(InvalidValue);
        }
        if slot.pointer {
            let address = unsafe { value.cast::<u64>().read_unaligned() };
            if address != 0 && !valid_pointer(address) {
                return Err(InvalidValue);
            }
        }
        // Pass raw byte ranges directly. A CUDA aggregate may contain padding
        // whose bytes were never initialized by the caller.
        match slot.words {
            None => args.push(KernelArg { value, size: slot.size }),
            Some(words) => args.extend((0..words).map(|index| KernelArg {
                value: unsafe { value.byte_add(index * 4) },
                size: 4,
            })),
        }
    }
    Ok(args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr::{from_mut, without_provenance_mut};

    #[test]
    fn staged_aggregates_collapse_to_one_pointer_argument() {
        let params = [Param::U32, Param::Staged { bytes: 2000, align: 8 }, Param::U8];
        assert_eq!(abi_size(&params), 8 + 8 + 1);
        assert_eq!(size(&params), 8 + 2000 + 1);
        let host: Vec<u8> = (0..2000u32).map(|i| i as u8).collect();
        let word = 5u32;
        let mut args = vec![KernelArg { value: std::ptr::from_ref(&word).cast(), size: 4 }];
        args.extend((0..500).map(|i| KernelArg { value: unsafe { host.as_ptr().add(i * 4) }.cast(), size: 4 }));
        let flag = 9u8;
        args.push(KernelArg { value: std::ptr::from_ref(&flag).cast(), size: 1 });
        let mut staged = Vec::new();
        let (out, keep) = collapse_staged(&params, args, |data| {
            staged.extend_from_slice(data);
            Ok(0xdead_0000)
        })
        .unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(staged, host);
        assert_eq!(unsafe { out[1].value.cast::<u64>().read() }, 0xdead_0000);
        assert_eq!((out[1].size, keep.len()), (8, 1));
    }

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
