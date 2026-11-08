# Versioning policy (DRAFT)

Semantic Versioning 2.0, in the 0.x phase: `0.MINOR.PATCH`.
- 0.x means the public contract below may change in a MINOR release, always with a CHANGELOG entry under "Changed" or "Removed" and a migration line.
- PATCH: bug fixes, translator coverage that only turns failures into successes, performance work, docs, no contract change.
- MINOR: new features, and any breaking change while 0.x.
- 1.0.0 only when the owner declares the contract stable and at least two device families are verified.

## What counts as breaking (the public contract)
1. **Environment switches**: the documented `CUDA_FOR_INTEL_*` variables (name, meaning, accepted values, default). Changing a default (for example SUBST auto, PREFETCH off) is breaking if it changes results or can break a working setup; it must be listed. Undocumented or `*_BENCH_*`/debug variables are not covered.
2. **DLL ABI and file layout**: exported symbols of the shipped DLLs (`nvcuda`, cudart/cublas shims, nvrtc shim), their names, and the staged directory layout the setup scripts produce.
3. **Translation cache**: on-disk format and the translator version stamp that keys it (today's code has a cache under `CUDA_FOR_INTEL_CACHE_DIR`; confirm the version constant name, the repo has no symbol literally named `TRANSLATOR_VERSION` at c065736). Any change to translator output or the cache format must bump that stamp so old caches are ignored. A bump is not breaking for users (the cache rebuilds) but is called out in the CHANGELOG because first runs are slower.
4. **Documented behaviour of scripts** in `scripts/` (parameters).
5. **Reported device properties** (compute capability reported to apps) when they change which code path an app takes.

Not covered: internal Rust APIs (`rlib`), test harness flags, log formats.

## Pre-releases and tags
Tags `vMAJOR.MINOR.PATCH`; release candidates `vX.Y.Z-rc.N`. Crate versions in all five `Cargo.toml` files move together.
