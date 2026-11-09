//! Device-independent half of the f64 and complex BLAS: `plan` (argument validation and
//! launch lists) and `ptxgen` (the kernels). Dependency-free, so `src/ptx.rs` includes this
//! module and runs the same launch lists in the SPIR-V simulator.
pub mod plan;
pub mod ptxgen;
