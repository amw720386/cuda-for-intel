fn main() -> Result<(), Box<dyn std::error::Error>> {
    let devices = nvcuda::intel_devices()?;
    if devices.is_empty() {
        return Err("no Intel Level Zero GPU found".into());
    }
    for (ordinal, device) in devices.iter().enumerate() {
        println!("{ordinal}: {} ({:04x})", device.name, device.device_id);
        println!("  Subgroups: {:?}", device.subgroup_sizes);
        println!(
            "  Workgroup: {:?} ({} total)",
            device.max_workgroup_dimensions, device.max_workgroup_size
        );
        println!("  Grid: {:?}", device.max_grid_dimensions);
        println!(
            "  Memory: {} bytes ({} per allocation)",
            device.total_memory, device.max_allocation
        );
        println!(
            "  SPIR-V: {}.{}",
            device.spirv_version >> 16,
            device.spirv_version & 0xffff
        );
        println!(
            "  Level Zero driver extensions: {}",
            device.driver_extensions.len()
        );
    }
    Ok(())
}
