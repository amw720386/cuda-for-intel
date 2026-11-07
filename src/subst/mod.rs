//! Kernel substitution: hand-written Intel-native SPIR-V replacements for known ggml-cuda kernels.
pub(crate) mod mmvq;
pub(crate) mod refs;
pub(crate) mod spv;
#[cfg(test)]
mod interp;
#[cfg(test)]
mod tests;
