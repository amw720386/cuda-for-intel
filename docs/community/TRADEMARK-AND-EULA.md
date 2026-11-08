# Trademark notice and CUDA EULA risk areas (DRAFT)

## A. Non-affiliation text (for README footer and release notes)

> CUDA, NVIDIA and related names are trademarks or registered trademarks of NVIDIA Corporation. Intel, Intel Arc, oneAPI and Level Zero are trademarks of Intel Corporation. Ollama, PyTorch, llama.cpp and other product names belong to their owners. CUDA for Intel is an independent open-source project. It is not affiliated with, sponsored by, or endorsed by NVIDIA or Intel. Product names are used only to describe compatibility.

Naming guidance (owner to decide): the project name contains "CUDA". Nominative use ("for Intel", compatibility with) is the common defence, but avoid logos, "CUDA-certified", "official", and avoid presenting the shim as "CUDA" itself in binary names beyond the required `nvcuda.dll`-style file names, which exist for loader compatibility.

## B. Plain-language CUDA EULA risk areas

**NOT LEGAL ADVICE. The EULA text changes between CUDA versions and applies to people who install the toolkit/driver. Get counsel before a public binary release.**

1. **Do not redistribute NVIDIA binaries.** Do not commit, attach to releases, or put in installers NVIDIA DLLs (for example `nvcudart_hybrid64.dll`, `cudart64_*.dll`, `cublas64_*.dll`), CUDA headers, libdevice bitcode or samples. The repo already ignores `*.dll`/`*.exe` and CONTRIBUTING forbids it; the release process (RELEASING.md) scans artifacts for them. Docs should tell users to obtain NVIDIA files from NVIDIA or from the app they already installed.
2. **Disassembly / reverse engineering of SASS.** The CUDA EULA restricts reverse engineering, decompiling or disassembling the SDK's output in some contexts. This project translates PTX (a documented ISA) and does not read SASS (no `cuobjdump`/`nvdisasm` use found in the source). Keep it that way: contributors must not paste SASS dumps, disassembly of NVIDIA libraries, or proprietary kernel source into issues, PRs or tests.
3. **Interoperability code derived from observation of NVIDIA's runtime**: `src/cuda/integrity.rs` (MD2 digest and key for CUDA 13's private integrity interface) and private export-table layouts are derived from ZLUDA. Interoperability is often treated more favourably in law, but this is the single area most worth showing counsel.
4. **Running unmodified third-party apps** that link NVIDIA libraries: the user's licence with NVIDIA governs those libraries; the project should not ship or auto-download them.
5. **Compatibility claims**: describe results factually ("tested with ... version ...") and only list verified results (see COMPAT-PROCESS.md).
6. **Test PTX/kernels from third parties** must keep their original licence headers (ZLUDA, ggml) and must not come from NVIDIA samples.

Contributor rules to add to CONTRIBUTING (owner to approve): no NVIDIA binaries or headers; no SASS or disassembly of NVIDIA code; no code copied from ZLUDA or other projects without a compatible licence and a notice.
