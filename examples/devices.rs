use nvcuda::DeviceInfo;

#[path = "../tests/gpu/isolation_capability.rs"]
mod isolation_capability;
use isolation_capability::supports_isolation;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let validation = match std::env::args().nth(1).as_deref() {
        None => false,
        Some("--validation") => true,
        Some(_) => return Err("usage: devices [--validation]".into()),
    };
    let devices = nvcuda::intel_devices()?;
    if devices.is_empty() {
        println!("SKIP: no Intel Level Zero root GPU found");
        return Ok(());
    }
    for (ordinal, device) in devices.iter().enumerate() {
        if validation {
            println!("DEVICE\t{ordinal}");
        }
        report(ordinal, device);
        if validation {
            validation_plan(ordinal, device);
            if !supports_isolation(device) {
                println!(
                    "SKIP\t{ordinal}\tcontext_isolation\trequires SPIR-V and pinned copies on the selected compute queue"
                );
            } else if !devices
                .iter()
                .enumerate()
                .any(|(peer, device)| peer != ordinal && supports_isolation(device))
            {
                println!(
                    "SKIP\t{ordinal}\tmulti_device_context_isolation\tno second root GPU supports the isolation workload"
                );
            }
        }
    }
    Ok(())
}

fn report(ordinal: usize, device: &DeviceInfo) {
    println!(
        "{ordinal}: {} ({:04x}:{:04x})",
        device.name, device.vendor_id, device.device_id
    );
    let uuid: String = device
        .uuid
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    println!("  UUID: {uuid}, device flags: {:#x}", device.device_flags);
    println!(
        "  Level Zero API: {}.{}, driver version (raw): {} ({:#x})",
        device.api_version >> 16,
        device.api_version & 0xffff,
        device.driver_version,
        device.driver_version
    );
    println!("  Subgroups: {:?}", device.subgroup_sizes);
    println!(
        "  Workgroup: {:?} ({} total), grid: {:?}, shared local: {} bytes",
        device.max_workgroup_dimensions,
        device.max_workgroup_size,
        device.max_grid_dimensions,
        device.max_shared_local_memory
    );
    println!(
        "  Memory: {} bytes ({} per allocation)",
        device.total_memory, device.max_allocation
    );
    for heap in &device.memory_heaps {
        println!(
            "  Heap {:?}: {} bytes, {} MHz, {}-bit bus, flags={:#x}",
            heap.name, heap.size, heap.clock_mhz, heap.bus_width, heap.flags
        );
    }
    println!(
        "  Access flags (host/device/shared single/shared cross/shared system): {:x?}",
        device.memory_access_flags
    );
    println!(
        "  SPIR-V: {}.{}, module flags={:#x}, max kernel argument bytes={}",
        device.spirv_version >> 16,
        device.spirv_version & 0xffff,
        device.module_flags,
        device.max_kernel_arguments_size
    );
    println!(
        "  Floating-point flags (FP16/FP32/FP64): {:x?}",
        device.floating_point_flags
    );
    println!(
        "  Runtime FP32 baseline: {}, FMA: {}, device allocation atomic access: {}",
        device.supports_fp32(),
        device.fp32_fma,
        device.device_memory_atomics
    );
    match device.float_atomic_flags {
        Some(flags) => {
            // Global operation bits are 0..2, local bits are 16..18.
            for ((width, mask), core) in [16, 32, 64]
                .into_iter()
                .zip(flags)
                .zip(device.floating_point_flags)
            {
                let operations = |bits: u32| {
                    ["load/store/exchange", "add/subtract", "min/max"]
                        .into_iter()
                        .enumerate()
                        .filter_map(|(index, name)| (bits & (1 << index) != 0).then_some(name))
                        .collect::<Vec<_>>()
                };
                println!(
                    "  FP{width} atomic extension flags={mask:#x}, core FP exposed={}, global={:?}, local={:?}",
                    core != 0,
                    operations(mask),
                    operations(mask >> 16)
                );
            }
        }
        None => println!("  FP atomic flags unavailable (extension absent or query unsuccessful)"),
    }
    println!(
        "  Selected compute group: {:?}",
        device.compute_queue().map(|group| group.ordinal)
    );
    for group in &device.queue_groups {
        println!(
            "  Queue group {}: flags={:#x}, queues={}, compute={}, copy={}, copy-only={}",
            group.ordinal,
            group.flags,
            group.queues,
            group.compute(),
            group.copy(),
            group.copy_only()
        );
    }
    println!(
        "  Timestamp frequency: {:?} Hz, global bits={}, kernel bits={}",
        device.timestamp_frequency_hz,
        device.timestamp_valid_bits,
        device.kernel_timestamp_valid_bits
    );
    for (name, version) in &device.driver_extensions {
        println!("  Extension {name}: {}.{}", version >> 16, version & 0xffff);
    }
}

fn validation_plan(ordinal: usize, device: &DeviceInfo) {
    let skip = |reason: &str, tests: &[&str]| {
        for test in tests {
            println!("SKIP\t{ordinal}\t{test}\t{reason}");
        }
    };
    let Some(group) = device.compute_queue() else {
        skip("no compute queue exposed", &["*"]);
        return;
    };
    let supported_kernel = |threads, groups, args, shared, allocation| {
        device.max_workgroup_size >= threads
            && device.max_workgroup_dimensions[0] >= threads
            && device.max_grid_dimensions[0] >= groups
            && device.max_kernel_arguments_size >= args
            && device.max_shared_local_memory >= shared
            && device.max_allocation >= allocation
    };
    let spirv = device.spirv_version >= 0x10000;
    let fp32 = spirv && device.supports_fp32();
    let fma = fp32 && device.fp32_fma;
    let copy = group.copy();
    let pinned = copy && device.memory_access_flags[0] & 1 != 0;

    if !copy {
        skip(
            "selected compute queue has no copy capability",
            &["device_memory_round_trip"],
        );
    }
    if !pinned {
        skip(
            "pinned copy needs host allocation access and copy on the selected compute queue",
            &[
                "pinned_async_transfers",
                "event_linked_transfers_make_progress",
                "gated_stream_stays_pending",
            ],
        );
    }
    for (test, flags, limits, reason) in [
        (
            "resolved_primary_context_runs_multiple_module_entries",
            spirv && copy,
            supported_kernel(1, 1, 12, 0, 4),
            "SPIR-V or copy capability",
        ),
        (
            "compiled_ptx_vector_add",
            fp32 && copy,
            supported_kernel(64, 32, 28, 0, 4096),
            "FP32 or copy capability",
        ),
        (
            "compiled_ptx_block_reduction",
            fp32 && copy,
            supported_kernel(128, 1003, 20, 512, 4096),
            "FP32 or copy capability",
        ),
        (
            "compiled_ptx_atomic_tickets",
            spirv && copy && device.device_memory_atomics,
            supported_kernel(64, 23, 16, 0, 11776),
            "SPIR-V, device atomic access or copy capability",
        ),
        (
            "compiled_ptx_rmsnorm",
            fma && copy,
            supported_kernel(128, 4, 32, 512, 16448),
            "FP32/FMA or copy capability",
        ),
        (
            "fp32_rounding_and_contraction",
            fma && copy,
            supported_kernel(64, 64, 16, 0, 114688),
            "FP32/FMA or copy capability",
        ),
        (
            "asynchronous_stream_order_and_lifetimes",
            spirv && pinned,
            supported_kernel(1, 1, 12, 0, 4),
            "SPIR-V or pinned copy capability",
        ),
        (
            "gated_m10_throughput",
            fma && copy,
            supported_kernel(128, 1, 32, 512, 4096),
            "FP32/FMA or copy capability",
        ),
    ] {
        if !flags {
            skip(reason, &[test]);
        }
        if !limits {
            skip(
                "workgroup, grid, argument, shared memory or allocation limit below workload requirement",
                &[test],
            );
        }
    }
    if !fp32
        || !copy
        || !supported_kernel(128, 3, 44, 0, 231736)
        || device.max_grid_dimensions[1] < 65
    {
        skip(
            "GGML add requires FP32, copies, and its launch and allocation limits",
            &["ggml_cuda_add_f32", "gated_ggml_add_device_time"],
        );
    }
    if !fma || !pinned || !supported_kernel(128, 512, 32, 512, 262144) {
        skip(
            "Forge requires FP32/FMA, pinned copies and its workload limits",
            &[
                "forge_transfer_pipeline",
                "dependent_forge_chain",
                "gated_forge_device_profile",
                "gated_forge_operation_timestamps",
                "gated_forge_copy_queue_timestamps",
                "gated_m10_forge",
            ],
        );
    }
    if !spirv {
        skip(
            "driver does not advertise SPIR-V 1.0",
            &[
                "gated_batch_",
                "gated_recovery_",
                "gated_m10_isolated",
                "gated_m10_device_execution",
                "gated_timestamp_storage_survives_stream_retirement",
            ],
        );
    }
    if device.timestamp_frequency_hz.is_none() || device.timestamp_valid_bits == 0 {
        skip(
            "global timestamp capability unavailable",
            &[
                "gated_forge_device_profile",
                "gated_forge_copy_queue_timestamps",
                "gated_timestamp_storage_survives_stream_retirement",
            ],
        );
    }
    if device.timestamp_frequency_hz.is_none() || device.kernel_timestamp_valid_bits == 0 {
        skip(
            "kernel timestamp capability unavailable",
            &[
                "gated_forge_operation_timestamps",
                "gated_forge_copy_queue_timestamps",
                "gated_m10_device_execution",
                "gated_ggml_add_device_time",
            ],
        );
    }
    if !device
        .queue_groups
        .iter()
        .any(|group| group.queues != 0 && group.copy_only())
    {
        skip(
            "no copy-only queue with a usable index",
            &["gated_forge_copy_queue_timestamps"],
        );
    }
}
