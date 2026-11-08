# Saved replies (copy into GitHub saved replies)

**Needs reproduction**
> Thanks for the report. To investigate we need: the environment-check output (`scripts\check-install.ps1` plus GPU, driver, OS and project commit/release), the app and version, the exact command, and a log with `CUDA_FOR_INTEL_LOG=1` for the failing part. Please remove user names, paths and addresses first. I have labelled this `status:needs-repro`.

**Unsupported hardware / app**
> Thanks. At the moment only the devices and apps listed in docs/compatibility.md are tested. This looks outside that, so I am closing it as `status:wontfix` for now. A compatibility report (docs/community/COMPAT-PROCESS.md) with your results would help us widen the list.

**NVIDIA files / SASS**
> Please remove the attachment. We cannot accept NVIDIA binaries, headers or disassembly (SASS) here. A PTX file you generated yourself, or a description of the kernel, is fine.

**Duplicate**
> This matches #N, so I am closing it as a duplicate. Please add anything new there.

**Stale**
> There has been no activity for 30 days and we are still missing the information above, so this will close in 14 days unless you reply.

**First PR**
> Thanks for your first PR. Please check the commits are signed off (`git commit -s`) and follow Conventional Commits, and that the PTX corpus (`tests/ptx_corpus/run.sh`) still passes if you touched the translator.

**Question redirect**
> This is a usage question rather than a bug. Please see the docs for your app (docs/ollama.md, docs/pytorch.md); if it turns out to be a bug, reopen with the information listed in the issue template.
