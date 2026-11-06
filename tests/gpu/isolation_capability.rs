pub fn supports_isolation(device: &nvcuda::DeviceInfo) -> bool {
    device.compute_queue().is_some_and(|queue| queue.copy())
        && device.memory_access_flags[0] & 1 != 0
        && device.spirv_version >= 0x10000
        && device.max_kernel_arguments_size >= 12
        && device.max_allocation >= 4
}
