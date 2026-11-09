//! Device-independent half of the shim: `plan` (descriptors, launch lists) and `ptxgen`
//! (the kernels). `src/ptx.rs` includes this module to run both in the SPIR-V simulator.
pub mod plan;
pub mod ptxgen;
