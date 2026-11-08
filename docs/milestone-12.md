# Milestone 12 — Cross-Intel-GPU validation and context isolation

Completed November 6, 2026. The CUDA subset passed the existing correctness matrix on the Intel Arc B390 (Panther Lake) and Intel Iris Xe (Alder Lake) with immediate lists and opt-in regular-list batching. Two independent contexts passed on each GPU. Each host exposed one Level Zero root GPU, so the physical multi-device regression was skipped on both. A two-device model verified bookkeeping and native resource ownership. The dedicated copy queue passed the Forge correctness workload. Its instrumented traces established no overlap for that run and the driver’s mapped operation timestamps failed clock validation.

## Access and setup

The local OpenSSH destination identity did not apply to the jump process. Separate `<gateway-host>` and `<arc-host>` host entries made the gateway accept the existing ED25519 key. The public file fingerprint matched the fingerprint derived from the private key. The gateway’s ED25519 fingerprint matched the existing known-host entry and the fingerprint supplied with the reservation. Intel’s public documentation did not publish an independent fingerprint. The Windows endpoint was pinned on first use over the authenticated gateway. Strict verification was used for the subsequent session.

Windows rejected the same public key and accepted password authentication. The interactive handoff timed out and one retry lost its temporary WSL directory. The successful connection used a local SSH prompt helper and a private WSL multiplex socket. Source and binary archives went through that connection and their SHA-256 checksums matched on both sides. [Intel Cloud SSH access guide](https://cloud.intel.com/docs/how_to_access.html)

The instance already had Visual Studio Community 18 with MSVC and Vulkan SDK 1.4.363.0. Rust 1.99.0 x86-64 MSVC with Clippy and rustfmt, and portable PowerShell 7.5.13, were installed under the validation directory in the remote user profile. Builds used that directory for Cargo and Rustup state. The selected graphics driver and Level Zero loader were the ones installed on each host. <!-- leak-ok: SDK version string -->

## Measured hardware and capabilities

| Property | Iris Xe | Panther Lake |
| --- | --- | --- |
| CPU | Core i5-1235U, 10 cores / 12 logical | Core Ultra X7 358H, 16 cores / 16 logical |
| GPU | Iris Xe Graphics, Alder Lake Xe-LP | Arc B390 GPU, Panther Lake Xe3 |
| PCI ID / revision | `8086:46a8` / `0c` | `8086:b080` / `04` |
| Level Zero UUID | `8680a8460c0000000002000000000000` | `868080b0040000000002000000000000` |
| Windows x64 | Windows 11 Enterprise, build 26200 | Windows 11 Pro, build 26300 |
| Graphics driver | `32.0.101.7085` | `32.0.101.9033` |
| System Level Zero loader | 1.24.0 | 1.32.0 |
| Level Zero API / raw driver version | 1.6 / 17008552 | 1.17 / 17013324 |
| Enumerated root GPUs | 1 | 1 |
| Compute group 0 | One queue, `0xf` (compute, copy, cooperative, metrics) | One queue, `0x7` (compute, copy, cooperative) |
| Copy-only group 1 | Absent | One queue, `0x2` |
| Subgroup sizes | 8, 16, 32 | 16, 32 |
| Workgroup limit | 512 work-items | 1,024 work-items |
| Grid limit | `u32::MAX` per axis | `u32::MAX` per axis |
| Shared local memory | 64 KiB | 128 KiB |
| Level Zero DDR heap / allocation limit | 7,447,306,240 B / 4,294,959,104 B | 36,055,617,536 B / 36,055,617,536 B |
| SPIR-V / max kernel argument bytes | 1.5 / 2,048 | 1.5 / 2,048 |
| FP16, FP32, FP64 property flags | `0x3f, 0x3f, 0x0` | `0x3f, 0x7f, 0x7f` |
| Timestamp frequency / widths (global, kernel) | 19.2 MHz / 36, 32 bits | 19.2 MHz / 64, 64 bits |

Both drivers advertised FP32 denormals, round-to-nearest, infinities and NaNs, FMA, and atomic access to device memory. Host, device and single-device shared allocation access flags were all `0x3`. Cross-device and system-wide shared access flags were zero. Both drivers advertised float atomics and mapped event timestamps. Panther Lake also advertised usable-memory properties, record/replay graphs and a queue copy offload hint. Extension flags describe driver reports. FP64 and float-atomic kernel execution are outside this CUDA subset. The complete extension inventories are in the raw logs. [Intel 358H product specifications](https://www.intel.com/content/www/us/en/products/sku/245527/intel-core-ultra-x7-processor-358h-18m-cache-up-to-4-80-ghz/specifications.html) · [Level Zero device properties](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/api/apis/device.html)

The Panther Lake host identified itself as a Mind Pro with 68,173,717,504 B of physical RAM, 66,575,896 KiB visible to Windows and 59,156,420 KiB available at inventory time. The C: volume reported 2,047,538,622,464 B capacity and 1,915,777,486,848 B free. These are measured values for the assigned machine. Both GPUs reported about 52.083 ns per timer tick.

## Correctness matrix

Each hardware cell covers immediate and C250 batching. The checked-in PTX fixture bytes, locked dependencies and numerical tolerances matched across hosts.

| Workload or check | Iris Xe | Arc B390 | Reference or assertion |
| --- | --- | --- | --- |
| PTX parser and lowering fixtures | Pass | Pass | Expected translation behavior and rejection cases |
| Independent SPIR-V validation | Pass | Pass | Khronos OpenCL validation of translated fixtures |
| Memory round trip and vector add | Pass | Pass | Host bytes, 1,003 CPU sums and guards |
| Block reduction, 6 widths | Pass | Pass | Exact dyadic inputs, FP32 tree-depth error bound and guards |
| Atomic tickets, 23 workgroups | Pass | Pass | Ticket coverage and counter wraparound |
| FP32 arithmetic, 4,096 cases | Pass | Pass | Bit-pattern contract for rounding, FMA, unfused operations and conversion |
| RMSNorm, 30 configurations | Pass | Pass | FP64 CPU reference, guards and repeatability |
| Forge chain and pinned pipeline | Pass | Pass | FP64 CPU reference, guards and event-linked ordering |
| Legacy/nonblocking streams and events | Pass | Pass | Ordering, event generations, forward progress and retirement |
| Pinned memory and asynchronous transfers | Pass | Pass | Copy direction, range checks and data integrity |
| Teardown, recovery and native lifetime | Pass | Pass | Pending work and fault-injection assertions |
| Two CUDA contexts on one root GPU | Pass | Pass | Ownership rejection, thread-local selection and survivor output |
| Two physical root GPUs in one process | Skipped | Skipped | Each host exposed one root |

Each host passed **13 DLL tests and 15 gated library tests per backend**. Format, locked debug and release builds, Clippy with warnings denied and **27 unit tests per feature configuration** passed on both hosts. The independent validator ran once per configuration on each host. Iris Xe used SPIRV-Tools v2025.1 through WSL. Panther Lake used SPIRV-Tools v2026.4 from the installed Vulkan SDK. The FP32 test requires bit equality only for its instruction-level numerical contract. RMSNorm and Forge use the existing tolerance checks against independent CPU results.

## Isolation and compatibility findings

- The same-device and capability-gated physical-device tests exercise the CUDA DLL. They reject foreign device and pinned pointers, modules, functions, events, and streams used for launches or copies. Explicit stream query and synchronization resolve the stream’s owning context. Event synchronization and dependencies require the current context under the supported subset. Two threads switch independently between contexts. Destroying one context leaves the other selected and able to complete work and produce the expected values. [CUDA context management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html) · [CUDA stream management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html)
- The native lifecycle fixture now exposes two simulated devices with distinct driver handles, device handles, compute queue ordinals and allocation limits. The new test checks routing, rejection, context destruction with the peer’s native commands pending, and subsequent peer progress. Across its immediate and batched cases, all 42 native objects were released, all eight submitted commands completed, and the ledger recorded zero violations. The fixture models ownership and completion without executing kernels or emulating memory.
- The prebuilt vector-add test initially failed on Panther Lake because its file-load case embedded the local build machine’s checkout path. The test now loads the fixture relative to the repository root and still exercises both file and in-memory module loading. The failed batched process also exited with a Windows access violation after the test panic left its context live. The rebuilt suite completed normally after the fixture correction. Abnormal process exit with live batching workers was outside this validation.
- Arc B390 passed with the existing PTX lowering and numerical tolerances. Production defaults to immediate lists and uses the selected compute group for copies and kernels. Physical multi-device execution, peer access and cross-device event dependencies remain outside the verified coverage.

## Dedicated copy queue experiment

On Arc B390 the test explicitly selected group 1 index 0 for event-linked upload and download and group 0 for compute. It ran the same 7×257 and 64×1024 Forge chains with one and two buffer slots. Blocking and single-stream controls used the original compute group. All configurations passed the CPU reference and guard checks. Each timestamp configuration used two passes of 31 samples after five warmups per pass.

| Timestamp evidence | Result on dedicated copy/compute lists |
| --- | --- |
| Device-side global timestamp brackets | All 248 event-linked samples passed clock-window, order, dependency and slot-reuse checks. Zero intersecting copy/kernel brackets. |
| Raw operation events | Per-operation durations passed the counter and run-duration checks. |
| Host-mapped operation events | 0 of 248 event-linked samples passed host anchor validation. The first operation typically mapped about 212 ms before its run’s host anchor. |

The brackets include marker and scheduling gaps. The instrumented workload showed no overlap within those brackets. Mapped events provided no valid cross-engine operation timeline, including in the compute-group controls. The analysis rejected samples outside the clock anchors. [Level Zero clock sampling](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/api/apis/device.html#zedevicegetglobaltimestamps) · [Mapped timestamp extension](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/api/extensions/event.html)

## Performance method

Both hosts ran the same x86-64 MSVC release test executable from baseline revision `19effe22fc81b771e6f7bd620d1fd964bb8a650f`. The SHA-256 was `5740df072df55a7d173add69496e35a1d515d8ab469a042d2cd8f55be630d52b`. There were two passes per host in A1, C250, A2 order. A1 and A2 use immediate lists. C250 uses a cap of 32 commands and a 250 µs high-resolution deadline. Cells report the range across the four immediate or two C250 distributions. OS, driver, CPU, GPU and host session differ across machines, so comparisons describe the measured systems.

The isolated-launch benchmark allowed the asynchronous deadline to submit work before observation. It recorded 300 completed samples after 20 warmups. Each burst was explicitly synchronized and recorded 35 samples after seven warmups. Forge measured 32 completed chains per sample with five warmups and 31 timed samples. Its p99 is the observed maximum. Forge used blocking copies from pinned buffers, one asynchronous stream, or three event-linked streams with one or two buffer sets. Timed runs used the production compute-group routing and excluded timestamp instrumentation.

The largest same-pass A1/A2 median difference relative to A1 was 21.5% on Iris Xe and 22.9% on Arc B390 for isolated launches, 17.3% and 25.0% for no-op bursts, and 38.7% and 2.0% for Forge. These controls inform comparisons of close results. C250 improved large no-op bursts on both systems and increased isolated latency on both. On Arc B390 the 7×257 single-stream Forge median improved by about 6%, while the 64×1024 single-stream and event-linked workloads were slower under C250.

### Launches and bursts

Completed time in µs, shown as median / p95 / p99.

| Workload | Iris Xe immediate | Iris Xe C250 | Arc B390 immediate | Arc B390 C250 |
| --- | ---: | ---: | ---: | ---: |
| Isolated no-op | 196.3–238.5 / 308.9–336.2 / 336.8–911.4 | 856.1–860.9 / 1045.4–1050.6 / 1275.0–1527.3 | 4.8–5.9 / 5.6–6.9 / 15.2–19.8 | 671.3–673.5 / 725.9–772.7 / 794.0–1198.2 |
| No-op ×1 | 84.6–122.3 / 166.5–249.7 / 179.6–276.6 | 89.6–91.2 / 114.5–117.4 / 114.7–125.1 | 4.9–7.5 / 6.3–14.4 / 6.4–45.1 | 11.6–11.7 / 12.0–13.9 / 12.1–14.0 |
| No-op ×10 | 540.0–730.4 / 813.8–1893.9 / 834.7–2417.7 | 108.9–119.9 / 220.2–298.5 / 280.2–416.1 | 17.4–21.6 / 18.9–25.1 / 19.1–225.0 | 19.8–20.6 / 21.4–21.9 / 22.1–29.4 |
| No-op ×100 | 5439.4–7010.3 / 6495.8–9167.9 / 6745.8–9877.8 | 389.0–520.9 / 692.1–730.2 / 706.6–974.5 | 140.6–142.1 / 143.1–144.6 / 144.5–146.5 | 100.5–105.7 / 107.4–117.6 / 112.5–119.5 |
| No-op ×1000 | 52008.7–73499.5 / 59832.2–83721.3 / 60265.0–100856.1 | 2965.3–4787.3 / 3822.8–12951.8 / 3898.6–13289.6 | 1378.9–1383.7 / 1381.2–2339.8 / 1386.7–2429.4 | 831.1–840.3 / 846.3–890.1 / 886.6–892.2 |
| RMSNorm ×1 | 108.2–165.0 / 134.0–312.1 / 153.9–1623.9 | 130.6–136.8 / 180.4–205.0 / 220.9–256.3 | 10.6–10.8 / 10.9–11.3 / 11.4–12.1 | 18.8 / 19.3–21.4 / 20.2–21.6 |
| RMSNorm ×32 | 2067.0–2796.3 / 2492.0–3718.3 / 3007.9–4057.8 | 769.5–787.8 / 885.2–1548.1 / 955.6–1742.8 | 153.7–154.1 / 154.0–154.7 / 156.9–158.0 | 193.6–193.8 / 197.3–199.6 / 200.5 |

### Forge

Completed time in ms per 32 chains, shown as median / p95 / p99.

| Workload | Iris Xe immediate | Iris Xe C250 | Arc B390 immediate | Arc B390 C250 |
| --- | ---: | ---: | ---: | ---: |
| 7×257 blocking | 17.196–20.370 / 20.103–29.767 / 21.588–32.732 | 16.437–16.611 / 20.038–24.325 / 21.397–36.578 | 1.423–1.442 / 1.435–1.470 / 1.444–2.504 | 1.833–1.868 / 2.104–2.260 / 2.337–2.707 |
| 7×257 single stream | 11.313–11.885 / 12.562–13.649 / 13.020–14.202 | 1.453–1.464 / 1.650–1.959 / 1.668–1.974 | 0.565–0.567 / 0.571–0.575 / 0.573–0.576 | 0.530–0.531 / 0.533–0.540 / 0.534–0.541 |
| 7×257 event-linked, 1 slot | 19.660–24.465 / 23.526–27.388 / 23.606–32.080 | 5.556–7.134 / 6.395–14.839 / 6.705–17.503 | 1.428–1.434 / 1.468–1.479 / 1.471–2.473 | 1.869–1.907 / 1.948–2.038 / 1.968–2.893 |
| 7×257 event-linked, 2 slots | 19.193–31.306 / 22.121–41.041 / 22.275–41.470 | 4.801–5.001 / 5.283–6.112 / 5.403–6.252 | 1.423–1.451 / 1.472–1.479 / 1.474–1.480 | 1.717–1.792 / 2.468–2.701 / 2.583–2.872 |
| 64×1024 blocking | 18.593–20.675 / 21.062–24.562 / 23.453–28.886 | 16.159–17.468 / 18.476–20.493 / 18.599–20.750 | 1.801–1.825 / 1.809–1.850 / 1.815–1.919 | 2.190–2.208 / 2.217–2.248 / 2.258–3.172 |
| 64×1024 single stream | 12.589–14.882 / 13.293–17.716 / 13.384–23.695 | 2.619–2.763 / 2.896–3.265 / 2.978–3.274 | 0.652–0.659 / 0.671–0.687 / 0.676–0.688 | 0.779–0.786 / 0.783–0.791 / 0.790–0.793 |
| 64×1024 event-linked, 1 slot | 21.460–25.827 / 24.465–31.651 / 24.879–35.511 | 6.373–6.693 / 7.224–7.286 / 7.374–7.375 | 1.725–1.754 / 1.777–1.841 / 1.786–2.746 | 2.152–2.157 / 2.284–2.353 / 3.048–3.327 |
| 64×1024 event-linked, 2 slots | 20.620–26.574 / 22.873–29.361 / 23.700–32.732 | 6.179–7.173 / 7.914–8.072 / 8.227–8.490 | 1.721–1.746 / 1.764–1.895 / 1.767–2.692 | 2.042–2.046 / 2.129–2.145 / 2.133–2.227 |

Throughput in thousands of completed chains per second, calculated from each completed median.

| Workload | Iris Xe immediate | Iris Xe C250 | Arc B390 immediate | Arc B390 C250 |
| --- | ---: | ---: | ---: | ---: |
| 7×257 blocking | 1.6–1.9 | 1.9 | 22.2–22.5 | 17.1–17.5 |
| 7×257 single stream | 2.7–2.8 | 21.9–22.0 | 56.4–56.7 | 60.2–60.4 |
| 7×257 event-linked, 1 slot | 1.3–1.6 | 4.5–5.8 | 22.3–22.4 | 16.8–17.1 |
| 7×257 event-linked, 2 slots | 1.0–1.7 | 6.4–6.7 | 22.0–22.5 | 17.9–18.6 |
| 64×1024 blocking | 1.5–1.7 | 1.8–2.0 | 17.5–17.8 | 14.5–14.6 |
| 64×1024 single stream | 2.2–2.5 | 11.6–12.2 | 48.6–49.1 | 40.7–41.1 |
| 64×1024 event-linked, 1 slot | 1.2–1.5 | 4.8–5.0 | 18.2–18.5 | 14.8–14.9 |
| 64×1024 event-linked, 2 slots | 1.2–1.6 | 4.5–5.2 | 18.3–18.6 | 15.6–15.7 |

### Standalone pinned transfers

Both hosts ran the same feature-enabled release binary from the baseline revision, SHA-256 `b5286638b91a2a83d0a70d58b4cfefbde8ebab2cbb40b309367b29a34c2265b1`. The transfer-only benchmark covered 4 KiB, 64 KiB, 1 MiB and 16 MiB in both directions, on the selected compute group. Each case ran 20 warmups and 101 timed samples per pass. Each asynchronous copy was followed by a stream wait. Values below are completed median / p95 in µs. The existing transfer harness reports these two percentiles. Raw logs include the intermediate sizes, direct Level Zero controls and submit/wait breakdowns.

| Workload | Iris Xe immediate | Iris Xe C250 | Arc B390 immediate | Arc B390 C250 |
| --- | ---: | ---: | ---: | ---: |
| 4 KiB HtoD, blocking | 69.20–82.30 / 79.30–147.40 | 70.10–88.00 / 76.80–128.90 | 6.00–6.10 / 6.30–6.70 | 6.20–6.70 / 6.60–7.20 |
| 4 KiB HtoD, async + wait | 80.60–98.70 / 86.80–159.50 | 70.50–89.60 / 86.30–144.80 | 6.20–6.50 / 6.70–7.00 | 12.70–13.00 / 13.40–14.90 |
| 4 KiB DtoH, blocking | 72.70–85.40 / 100.70–144.20 | 71.20–74.90 / 104.40–112.40 | 6.20–6.60 / 6.70–7.00 | 6.40–7.10 / 7.00–7.50 |
| 4 KiB DtoH, async + wait | 79.40–106.10 / 129.70–200.30 | 82.30–94.30 / 118.70–129.30 | 6.30–6.50 / 6.90–7.10 | 12.80–13.40 / 13.50–14.80 |
| 16 MiB HtoD, blocking | 1386.80–1584.90 / 1738.60–1958.60 | 1338.60–1440.00 / 1489.70–1682.10 | 185.10–195.20 / 186.20–198.20 | 198.60–368.20 / 211.00–375.50 |
| 16 MiB HtoD, async + wait | 1433.10–1707.20 / 1737.80–2002.80 | 1408.50–1747.90 / 1595.90–2399.00 | 194.20–200.20 / 200.40–209.80 | 203.40–203.80 / 207.70–217.50 |
| 16 MiB DtoH, blocking | 1393.60–1559.50 / 1630.20–1771.10 | 1393.90–1570.00 / 1573.80–1780.60 | 166.60–180.10 / 175.90–195.90 | 174.10–174.70 / 178.80–179.10 |
| 16 MiB DtoH, async + wait | 1380.20–1586.10 / 1603.40–1948.00 | 1350.70–1831.20 / 1582.40–2628.30 | 171.90–176.20 / 175.50–183.20 | 181.90–182.60 / 186.40–188.30 |

[Iris Xe transfer log](m12-results/iris-transfer.log) · [Arc B390 transfer log](m12-results/panther-transfer.log)

## Reproduce and audit

Run from the repository root with a Windows x64 Rust toolchain and Khronos `spirv-val` on PATH or selected through `SPIRV_VAL`.

```powershell
cargo fmt --all -- --check
foreach ($features in @(@(), @('--features', 'experimental-batching'))) {
    cargo build --locked @features
    cargo build --release --locked @features
    cargo clippy --locked --all-targets @features -- -D warnings
    cargo test --locked @features
    cargo test --locked --lib @features validate_opencl_spirv -- --ignored --nocapture
}
pwsh -File scripts/validate-gpu.ps1 -LogPath docs/m12-validation.log

# Separate experiment on an ordinal with a usable copy-only group
$env:CUDA_FOR_INTEL_DEVICE = '0'
$env:CUDA_FOR_INTEL_BENCH_PROFILING = '1'
cargo test --locked --lib gated_forge_copy_queue_timestamps -- --ignored --nocapture --test-threads=1
Remove-Item Env:CUDA_FOR_INTEL_BENCH_PROFILING

# Standalone pinned transfers. Run twice per setting in A1, C250, A2 order.
# Leave the three batch variables unset for A1/A2. For C250, set them as below.
$env:CUDA_FOR_INTEL_BATCH_CAP = '32'
$env:CUDA_FOR_INTEL_BATCH_US = '250'
$env:CUDA_FOR_INTEL_BATCH_TIMER = 'high-resolution'
cargo test --release --locked --features experimental-batching --lib pinned_transfer_benchmark -- --ignored --nocapture --test-threads=1
Remove-Item Env:CUDA_FOR_INTEL_BATCH_CAP, Env:CUDA_FOR_INTEL_BATCH_US, Env:CUDA_FOR_INTEL_BATCH_TIMER
```

The correctness source was code commit `7ec5cf8028aeebe74249ba4eff499d51b402ac9f`, Git tree `a181a08b818ec878bcbc578970b6690ca704ddc0`. The archive deployed to Panther Lake had SHA-256 `d623d4c0f19e95b7558d79ce219d037f2b4d564ecc6443dfbf70fcfaf36cae9f`. The runner records the checkout state when Git metadata is present, the toolchain, and hashes of `Cargo.lock`, PTX files, the loaded DLLs and the test executables.

| Fixture | SHA-256 |
| --- | --- |
| `atomic_tickets.ptx` | `13cc99d91c137ba87ce1d3e7cd4677100c97a695991c68048366ed086dc9a678` |
| `block_reduce.ptx` | `a5195e90e7332f7b9fa343f2f64374db1a8fa590aa770b49d9d5b958a081d4b3` |
| `forge_reduce.ptx` | `38dbddffc219ebe85e864b6a917983a4e5e2058f3f271c2274645fbaf54634fd` |
| `forge_residual.ptx` | `18968976e97e846e31479b9e888229547879c7fef5dee407c02290557c32ca2f` |
| `rmsnorm.ptx` | `8a3e23673023ac43e09738c35c6030f8068f409c6150d8a60429c87fbb01a2b3` |
| `vector_add.ptx` | `fcc93aae7b99b70ef2672f4f92e4b5750b4b474bb18cc964b3ef96dafbcfcd12` |

Local raw logs are under `docs/m12-results/`. [Iris Xe verification](m12-results/iris-verification.log) · [Arc B390 verification](m12-results/panther-verification.log) · [Panther Lake OS inventory](m12-results/panther-os.log) · [Panther Lake Level Zero inventory](m12-results/panther-inventory.log) · [Iris Xe performance](m12-results/iris-performance.log) · [Arc B390 performance](m12-results/panther-performance.log) · [Copy queue trace](m12-results/panther-copy-queue.log)

## Recommended Milestone 13

Run the physical-device regression on a host exposing two Intel root GPUs. Check the mapped event clock against device anchors on the copy and compute groups before testing production copy routing. Use a control workload with deliberately concurrent transfers and kernels to verify the measurement can detect overlap before relying on zero-overlap results.
