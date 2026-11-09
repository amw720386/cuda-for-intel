//! `cfi_redirect.dll`: injected into a process by `cfi-run` (and from there into every child the
//! process creates). It detours `ntdll!LdrLoadDll`, which every `LoadLibrary{A,W,ExA,ExW}` call
//! reaches, and rewrites requests for NVIDIA CUDA libraries to the files in our install
//! directory, including requests by absolute path (torch's `torch\lib\cublas64_12.dll`).
//! Before a non-CUDA DLL loads, the hook reads its import table (and those of its same-directory
//! dependencies) and loads our stand-ins for any CUDA library it imports by name, so the
//! loader's static-import binding (which matches already-loaded modules by file name) picks ours.
//! Nothing in the application's directory is written; renamed copies ("aliases", such as our
//! `cublas64_12.dll` serving `cublas64_11.dll`) go to `<install>\aliases`.
//!
//! The mapping (`map`) and PE import reader (`pe`) are platform independent and unit tested on
//! every host; the hooks (`win`) build only on Windows.

pub mod map;
pub mod pe;
pub mod shared;

#[cfg(windows)]
mod win;
