//! Device-resident descriptor behind a texture or surface object handle.
//!
//! There is no image support in the OpenCL-environment SPIR-V target this project emits, so
//! texture and surface objects are emulated on linear device memory: a `CUtexObject` /
//! `CUsurfObject` is the device address of one of these 128-byte descriptors, and the
//! translator lowers `tex`, `txq`, `suld`, `sust` and `suq` to loads of descriptor fields,
//! software addressing and filtering, and ordinary global loads and stores of the texels
//! (`src/ptx/tier3_tex.rs`). The runtime side (`src/cuda/texture.rs`) fills the descriptor.

/// Byte offsets of the descriptor fields (all little-endian).
pub(crate) const BASE: u32 = 0; // u64 device address of texel (0, 0, 0)
pub(crate) const WIDTH: u32 = 8; // u32 elements
pub(crate) const HEIGHT: u32 = 12; // u32 rows (1 for 1D)
pub(crate) const DEPTH: u32 = 16; // u32 slices (1 for 1D and 2D)
pub(crate) const KIND: u32 = 20; // u32 channel kind, see `Kind`
pub(crate) const CHANNELS: u32 = 24; // u32 1, 2 or 4
pub(crate) const ELEMENT_BYTES: u32 = 28; // u32 channels * channel bytes
pub(crate) const PITCH: u32 = 32; // u64 bytes per row
pub(crate) const SLICE: u32 = 40; // u64 bytes per slice
pub(crate) const ADDRESS_MODE: u32 = 48; // 3 x u32 CUaddress_mode per axis
pub(crate) const FILTER: u32 = 60; // u32 CUfilter_mode
pub(crate) const FLAGS: u32 = 64; // u32 CU_TRSF_* flags
pub(crate) const DIMS: u32 = 68; // u32 1, 2 or 3
pub(crate) const BORDER: u32 = 72; // 4 x f32 border colour
pub(crate) const LINEAR: u32 = 88; // u32 1 for CU_RESOURCE_TYPE_LINEAR (out of range reads 0)
pub(crate) const CHANNEL_BYTES: u32 = 92; // u32 1, 2 or 4
pub(crate) const CU_FORMAT: u32 = 96; // u32 CUarray_format as given
pub(crate) const RESOURCE_TYPE: u32 = 100; // u32 CUresourcetype
pub(crate) const SIZE: usize = 128;

/// `CU_TRSF_READ_AS_INTEGER`
pub(crate) const READ_AS_INTEGER: u32 = 1;
/// `CU_TRSF_NORMALIZED_COORDINATES`
pub(crate) const NORMALIZED_COORDINATES: u32 = 2;

/// Channel kind codes stored at `KIND`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub(crate) enum Kind {
    U8 = 0,
    U16 = 1,
    U32 = 2,
    S8 = 3,
    S16 = 4,
    S32 = 5,
    F16 = 6,
    F32 = 7,
}

impl Kind {
    /// From `CUarray_format`.
    pub(crate) fn from_cu(format: u32) -> Option<Kind> {
        Some(match format {
            0x01 => Kind::U8,
            0x02 => Kind::U16,
            0x03 => Kind::U32,
            0x08 => Kind::S8,
            0x09 => Kind::S16,
            0x0a => Kind::S32,
            0x10 => Kind::F16,
            0x20 => Kind::F32,
            _ => return None,
        })
    }

    pub(crate) fn bytes(self) -> u32 {
        match self {
            Kind::U8 | Kind::S8 => 1,
            Kind::U16 | Kind::S16 | Kind::F16 => 2,
            Kind::U32 | Kind::S32 | Kind::F32 => 4,
        }
    }
}

/// Host-side view of one descriptor.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Desc {
    pub base: u64,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub kind: u32,
    pub channels: u32,
    pub pitch: u64,
    pub slice: u64,
    pub address_mode: [u32; 3],
    pub filter: u32,
    pub flags: u32,
    pub dims: u32,
    pub border: [f32; 4],
    pub linear: bool,
    pub cu_format: u32,
    pub resource_type: u32,
}

impl Desc {
    pub(crate) fn channel_bytes(&self) -> u32 {
        match self.kind {
            0 | 3 => 1,
            1 | 4 | 6 => 2,
            _ => 4,
        }
    }

    pub(crate) fn element_bytes(&self) -> u32 {
        self.channel_bytes() * self.channels
    }

    pub(crate) fn encode(&self) -> [u8; SIZE] {
        let mut b = [0u8; SIZE];
        let mut put32 = |off: u32, v: u32| b[off as usize..off as usize + 4].copy_from_slice(&v.to_le_bytes());
        put32(WIDTH, self.width);
        put32(HEIGHT, self.height.max(1));
        put32(DEPTH, self.depth.max(1));
        put32(KIND, self.kind);
        put32(CHANNELS, self.channels);
        put32(ELEMENT_BYTES, self.element_bytes());
        for (i, m) in self.address_mode.iter().enumerate() {
            put32(ADDRESS_MODE + 4 * i as u32, *m);
        }
        put32(FILTER, self.filter);
        put32(FLAGS, self.flags);
        put32(DIMS, self.dims);
        for (i, c) in self.border.iter().enumerate() {
            put32(BORDER + 4 * i as u32, c.to_bits());
        }
        put32(LINEAR, u32::from(self.linear));
        put32(CHANNEL_BYTES, self.channel_bytes());
        put32(CU_FORMAT, self.cu_format);
        put32(RESOURCE_TYPE, self.resource_type);
        b[BASE as usize..BASE as usize + 8].copy_from_slice(&self.base.to_le_bytes());
        b[PITCH as usize..PITCH as usize + 8].copy_from_slice(&self.pitch.to_le_bytes());
        b[SLICE as usize..SLICE as usize + 8].copy_from_slice(&self.slice.to_le_bytes());
        b
    }
}
