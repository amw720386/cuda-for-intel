# DCO vs CLA (DRAFT recommendation)

## Recommendation: DCO sign-off

Use the Developer Certificate of Origin 1.1 (https://developercertificate.org/): contributors add `Signed-off-by: Name <email>` via `git commit -s`.

Why:
- Small early project; a CLA adds a signing step (and a CLA service, or a person to run one) that deters first-time contributors.
- Inbound = outbound licensing (contributions under the repo licence) is enough as long as the owner does not plan to relicense to a proprietary or dual commercial licence. The relicensing between permissive licences in LICENSE-DECISION.md is a one-time step handled now with 3 known authors.
- DCO is what the Linux kernel and many Rust/GPU projects use; no external service, only a check.
- The DCO also covers the main project-specific risk: certifying the contributor has the right to submit the code (no copied NVIDIA/proprietary code, third-party code carries its notice).

Choose a CLA only if the owner wants the right to relicense without asking each contributor, or a company will take ownership of the project.

## Drafts (not enabled)

- `drafts/dco.yml` : GitHub Action that fails a PR if a commit lacks a sign-off. Copy to `.github/workflows/dco.yml` to enable. (Alternative: install the "DCO" GitHub App, which needs no workflow.)
- `drafts/CONTRIBUTING-DCO-snippet.md` : text to paste into CONTRIBUTING.md (owned by the packaging branch).
- Appendix A below: CLA alternative.

Note for the DCO: the repo history so far has no sign-offs; apply the check to new PRs only (the workflow checks the PR commit range).

## Appendix A: CLA alternative (only if chosen instead)

> **Individual Contributor License Agreement (DRAFT, needs counsel)**
> You accept and agree to the following for your present and future contributions submitted to CUDA for Intel (the "Project").
> 1. **Copyright licence.** You grant the Project owner and recipients of the software a perpetual, worldwide, non-exclusive, royalty-free, irrevocable copyright licence to reproduce, prepare derivative works of, publicly display, publicly perform, sublicense and distribute your contributions and such derivative works.
> 2. **Patent licence.** You grant a perpetual, worldwide, non-exclusive, royalty-free, irrevocable patent licence to make, have made, use, sell, offer to sell, import and otherwise transfer the contribution, limited to patent claims you can license that are necessarily infringed by your contribution alone or with the Project. If you institute patent litigation alleging the contribution infringes a patent, the licences you received for that contribution end.
> 3. **Your rights.** You represent that each contribution is your original creation (or you have the right to submit it under these terms), that your employer, if any, has waived or authorised it, that it contains no code you are not allowed to submit (including NVIDIA proprietary code, headers, binaries or SASS disassembly), and that you will identify any third-party material and its licence.
> 4. **No warranty / support.** Contributions are provided "AS IS".
> 5. **Notice.** You will tell the Project if you become aware of facts that make any statement above inaccurate.
>
> Tooling if adopted: CLA Assistant (GitHub App) or a `cla.yml` workflow storing signatures in a separate branch.
