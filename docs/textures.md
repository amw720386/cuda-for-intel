# Textures, surfaces and CUDA arrays

Roadmap items 16 and 17 of [zluda-parity.md](zluda-parity.md). The OpenCL-environment SPIR-V that the translator emits has no
bindless images, so texture and surface objects are emulated on linear device memory. This page describes the design, what is
covered, and the limits.

## Design

- **CUDA arrays** (`cuArrayCreate_v2`, `cuArray3DCreate_v2`) are dense device allocations: rows are `width * element bytes`
  apart, with no padding, and slices are `rows * pitch` apart. The `CUarray` handle is the allocation's device address.
  `cuMemcpy2D_v2`, `cuMemcpy2DUnaligned_v2` and `cuMemcpy3D_v2` (and their async forms) accept `CU_MEMORYTYPE_ARRAY` on either
  side. The 1D copies `cuMemcpy{HtoA,AtoH,DtoA,AtoD,AtoA}_v2` (plus `HtoAAsync`/`AtoHAsync`) take byte offsets into that dense
  layout.
- **Texture and surface objects** (`cuTexObjectCreate`, `cuSurfObjectCreate`) upload a 128-byte descriptor
  (`src/ptx/texdesc.rs`) to device memory. The `CUtexObject`/`CUsurfObject` value is the descriptor's device address. The
  descriptor holds the base address, the extent, the pitches, the channel format and count, the address modes, the filter mode,
  the flags and the border colour.
- **Translator** (`src/ptx/tier3_tex.rs`). Each `tex`, `txq`, `suld.b`, `sust.b` or `suq` that takes a handle register is
  expanded into ordinary PTX. That code loads the descriptor fields and does CUDA's addressing per axis in arithmetic:
  normalised coordinates, wrap, clamp, mirror and border. It then filters, either nearest or linear with CUDA's 8-bit
  fixed-point weights (`frac` rounded to 1/256), and reads texels with global loads.
  - Every texel index is clamped into the resource before the load. A texel outside the resource is replaced by the border
    colour (or 0) after the load, so no access ever leaves the allocation. The simulator tests check this.
  - Wrap and mirror apply only with normalised coordinates. Without them, CUDA clamps, and so does this code.
  - A guarded instruction (`@%p tex ...`) branches around the whole expansion, because the handle may be invalid when the guard
    is false.
- **Channel formats**: u8, u16, u32, s8, s16, s32, f16 and f32, with 1, 2 or 4 channels.
  - 8- and 16-bit integers read as normalised floats (`/255`, `/65535`, `/127`, `/32767`, the signed ones clamped to -1),
    unless `CU_TRSF_READ_AS_INTEGER` is set.
  - 32-bit integers are converted numerically.
  - f16 is widened exactly.
  - Channels the format lacks read as 0.

## Coverage

| Call or instruction | Status |
| --- | --- |
| `cuArrayCreate_v2`, `cuArray3DCreate_v2`, `cuArrayDestroy`, `cuArrayGetDescriptor_v2`, `cuArray3DGetDescriptor_v2` | 1D, 2D and 3D. Layered, cubemap, sparse and deferred-mapping flags return `CUDA_ERROR_NOT_SUPPORTED` |
| `cuMemcpy2D*`, `cuMemcpy3D*` with array operands; `cuMemcpy{HtoA,AtoH,DtoA,AtoD,AtoA}_v2`, `cuMemcpy{HtoA,AtoH}Async_v2` | Yes |
| `cuTexObjectCreate`, `cuTexObjectDestroy`, `cuTexObjectGetResourceDesc`, `cuTexObjectGetTextureDesc`, `cuTexObjectGetResourceViewDesc` | Array, `LINEAR` and `PITCH2D` resources; every address mode; point and linear filtering; normalised coordinates; read as integer; border colour. Mipmapped arrays, sRGB, and resource views that change the format or select levels or layers return `CUDA_ERROR_NOT_SUPPORTED`. Linear filtering of 32-bit integer or read-as-integer textures is `CUDA_ERROR_INVALID_VALUE`, as in CUDA |
| `cuSurfObjectCreate`, `cuSurfObjectDestroy`, `cuSurfObjectGetResourceDesc` | Array resources |
| `tex.{1d,2d,3d}.v4.{f32,s32,u32}.{f32,s32}`, `tex.level.*` (level 0) | Yes, on texture-object handles |
| `txq.{width,height,depth,channel_data_type,channel_order,normalized_coords,filter_mode,addr_mode_N,array_size,num_mipmap_levels,num_samples}.b32` | Yes. `channel_data_type` returns the `CUarray_format` value, and `channel_order` returns the channel count |
| `suld.b.{1d,2d,3d}[.v2,.v4].{b8,b16,b32,b64}`, `sust.b.*` with `.zero`, `.clamp` or `.trap` | Yes. The x coordinate is in bytes, as in CUDA. `.trap` behaves like `.zero` (out-of-range loads read 0, out-of-range stores are dropped), because there is no trap |
| `suq.*.b32` | Yes |
| Texture references: `.texref`/`.surfref`/`.samplerref`, `cuTexRefSet*`, `cuModuleGetTexRef`/`GetSurfRef`, `cuSurfRefSetArray` | **No.** A precise translator error. The texture-reference API was removed in CUDA 12 |
| `tld4` (gather), `tex.grad`, layered (`a1d`/`a2d`), cube and multisample geometries, `sust.p` (formatted stores), `sured` | **No.** A precise translator error |

The 12 ZLUDA `test_ptx__tex_*_texobj` fixtures translate (`tests/ptx_corpus/expected_status.tsv`) and match host-computed
sampling on the B390 (`tests/gpu/hw_texture.rs`). The 12 texture-reference fixtures stay unsupported.

## Limits and cost

- No texture cache and no hardware filtering. Each `tex` loads up to 8 texels (3D linear) through the ordinary cache
  hierarchy, with about 150 to 1,000 instructions of addressing, conversion and filtering, depending on the geometry and the
  filter. The format and the modes are runtime descriptor fields, so every format path is present in the code.
- Linear filtering uses the descriptor's filter mode at run time. A point-sampled texture still computes the weights.
- The descriptor is read from global memory by every `tex`, and the kernel must have indirect access to it. Kernels get that
  for all allocations in this runtime.

## Tests

- `src/ptx/tex_tests.rs` (simulator, CI) checks every geometry, coordinate type, result type, format and channel count, address
  mode, filter mode and normalisation against an independent host model (`src/ptx/tex_model.rs`). That is over 1,000
  configurations at 16 coordinates each, bit-exact. It also covers linear-memory fetches, surface loads and stores in zero and
  clamp mode (with out-of-range threads), `txq`/`suq`, and the precise error for texture references. The simulator reports any
  access outside an allocation.
- `tests/gpu/hw_texture.rs` (B390) covers:
  - array creation, descriptors, 3D/2D/1D copies to and from arrays (host, device and array);
  - texture objects over arrays, pitched 2D memory and linear memory, in every mode, against the same model;
  - ZLUDA's 12 fixtures;
  - surface objects.
  See [hw-logs/textures-2026-11-09.md](hw-logs/textures-2026-11-09.md).
