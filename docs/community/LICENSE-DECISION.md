# Licence decision (DRAFT for the repo owner)

Status: recommendation only. The `LICENSE` file and the `license` fields in the `Cargo.toml` files are unchanged by this branch. The choice belongs to the repo owner.
Not legal advice. Ask counsel before relying on any of this.

## 1. Audit of the tree at `release-cand` (c065736)

| Item | Finding |
| --- | --- |
| `LICENSE` | Present, MIT, "Copyright (c) 2026 CUDA for Intel contributors". Adds two attributions: Level Zero ABI declarations based on `ze_api.h` (Intel, MIT) and "private export table layouts / runtime integrity digest draw from ZLUDA, used under the MIT license". |
| `Cargo.toml` (root + 4 crates) | All `license = "MIT"`. |
| Dependencies (29 packages in `Cargo.lock`, 24 third-party) | All permissive: MIT, Apache-2.0, ISC, BSD-3-Clause, Zlib, 0BSD, or dual combinations. See `THIRD_PARTY_NOTICES.draft.md`. No GPL/AGPL/LGPL-only crate. `r-efi` offers `LGPL-2.1-or-later` only as one alternative of an OR expression (MIT/Apache can be chosen); it is Windows/UEFI-irrelevant and pulled in via `getrandom` on UEFI targets. |
| ZLUDA-derived code | (a) `src/cuda/integrity.rs` says "Based on the ZLUDA dark_api implementation at f4df215 (MIT, see LICENSE)". (b) `crates/fatbin/src/lib.rs`: "Header layouts follow ZLUDA". (c) `tests/ptx_corpus/zluda/` holds 232 PTX test kernels imported by `tests/ptx_corpus/import_zluda.py`, each with a header and `LICENSE-ZLUDA-MIT` / `LICENSE-ZLUDA-APACHE` copies. ZLUDA is `MIT OR Apache-2.0`. **Do not add further ZLUDA code**; the three items above already exist and should be reviewed (see open decisions). |
| llama.cpp / ggml (MIT) | `examples/ggml_add_f32.cu` is "Adapted from ggml/src/ggml-cuda/binbcast.cu at llama.cpp b12111"; `examples/ggml-LICENSE` carries the notice; several `examples/ggml_*.ptx` and `tests/ptx_corpus/ggml/*.ptx` are compiled from ggml-derived kernels. `scripts/build-ggml-ptx.py` stamps the ggml copyright header on generated files. Compatible with MIT or Apache-2.0 outbound as long as the notice stays. |
| SPIRV-Headers | Not vendored. SPIR-V enums come from the `spirv` / `rspirv` crates (Apache-2.0). |
| Level Zero | `ze_api.h` declarations re-expressed in Rust; attribution in `LICENSE`. Level Zero is MIT. |
| NVIDIA CUDA headers | **None vendored.** `crates/nvrtc_shim/include/*.h` are 864 lines of own minimal reimplementations (`cuda_fp16.h` etc.); no NVIDIA copyright text found (grep for "NVIDIA Corporation" is empty). Keep it that way: never copy `cuda_fp16.h`, `cuda_runtime.h`, `cooperative_groups.h` from a CUDA toolkit. |
| NVIDIA binaries | None committed (`*.dll`, `*.exe` are gitignored). `CONTRIBUTING.md` already forbids `nvcudart_hybrid64.dll`. |
| Generated PTX in `examples/` and the corpus | Produced by LLVM NVPTX (clang) or older NVVM ("Based on NVVM 7.0.1" header in some). Output of a compiler from ggml source (MIT); the NVVM banner is not an NVIDIA licence grant or restriction on the code, but counsel may want to confirm for those few files (see TRADEMARK-AND-EULA.md). |
| Contributors | git history has 3 author names. Copyright sits with "CUDA for Intel contributors" with no CLA; relicensing later needs consent of all of them. Decide before the contributor count grows. |
| Tools | `cargo-deny`, `cargo-license` and `cargo-about` are not installed on the audit machine; licences were read from `cargo metadata --locked`. `deny.toml.draft` is untested for that reason; run `cargo deny check` before adopting. |

## 2. Options

| Option | Pros | Cons for this project |
| --- | --- | --- |
| MIT (current) | Simplest; matches ZLUDA, ggml, Level Zero, so attribution is straightforward; maximal adoption; contributors understand it. | No express patent grant. A CUDA-compat layer sits in patent-dense territory (GPU ISAs, ABI). No defence against patent suits from contributors. |
| Apache-2.0 | Express patent licence from contributors plus patent-retaliation; clear NOTICE mechanism; can ingest MIT code. | Slightly heavier; one-way compatibility issue with GPLv2-only (not relevant here). Re-licensing from MIT to Apache needs agreement of existing contributors (small number now). |
| Dual MIT OR Apache-2.0 | Rust ecosystem norm; same terms as ZLUDA and most deps; users choose; patent grant available to those who want it; no friction with MIT-only or Apache-only consumers; inbound=outbound makes DCO simple. | Two licence files; marginally more text to maintain. |
| MPL-2.0 | File-level copyleft: modifications to our files stay open, linking is free. | Unfamiliar to many; friction for static/embedded users; a shim loaded as `nvcuda.dll` is a replaceable DLL so copyleft buys little; ZLUDA-derived and ggml code mix is fine but file-level tracking is a burden for a fast-moving small project. |
| GPL-3.0 | Strong reciprocity. | Incompatible with Apache-2.0 deps only in the GPLv2 direction (v3 is fine), but blocks use by vendors, Ollama-style bundlers and many users who ship the DLL next to proprietary apps; shim would be loaded into proprietary processes (legal grey area, chills adoption). Hurts the project goal. |
| AGPL-3.0 | Network copyleft. | Meaningless here (no network service) and the strongest adoption deterrent. |

## 3. Recommendation

**Dual license MIT OR Apache-2.0**, i.e. `license = "MIT OR Apache-2.0"`, with `LICENSE-MIT` and `LICENSE-APACHE` at the root, keep the existing attributions (Intel `ze_api.h`, ZLUDA, ggml) in a `NOTICE` / `THIRD_PARTY_NOTICES` file. Reasons:

1. All current inbound code and every dependency is permissive; nothing forces copyleft.
2. Dual licensing is the Rust ecosystem default and matches ZLUDA (the nearest relative), so derived/ported code can flow in either direction.
3. It adds the patent grant MIT lacks, which matters for a project that reimplements a vendor's API surface.
4. It costs nothing for existing MIT users: they can keep using it under MIT.
5. If the owner wants minimal change, staying MIT is acceptable; the remaining gaps (ZLUDA copyright line, notice file) are fixable without relicensing.

Mechanics if adopted (owner action, not done here): obtain consent of the 3 current authors (or confirm the "contributors" holder covers them), add the files, change the five `Cargo.toml` fields, add the SPDX header policy (optional), and start DCO sign-off from that commit onwards.

## 4. Open decisions for the owner

1. MIT vs MIT OR Apache-2.0 (above).
2. Who is the copyright holder name: "CUDA for Intel contributors" is not a legal entity; consider naming the individual maintainer(s) or "The cuda-for-intel authors".
3. The ZLUDA attribution in `LICENSE` names the project but not the copyright line. MIT requires keeping the copyright notice of the copied portion; check `tests/ptx_corpus/zluda/LICENSE-ZLUDA-MIT` and consider copying the exact ZLUDA notice into `THIRD_PARTY_NOTICES`.
4. Whether `src/cuda/integrity.rs` (an MD2 digest and 16-byte key used by CUDA 13's private integrity interface) and the fatbin header layouts are acceptable to keep; they are interoperability code derived from ZLUDA and from observation of NVIDIA's runtime. Counsel question, see TRADEMARK-AND-EULA.md.
5. Whether to fix the stale statement in README ("MIT licensed") after any change.
