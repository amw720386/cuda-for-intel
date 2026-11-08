//! Multi-device behaviour: which Level Zero devices become CUDA devices and in which
//! order, device identity (UUID, PCI bus id), per-device translator probes, peer access
//! and CUDA IPC. Kept apart from `cuda.rs`, which only calls the small hooks below.
//!
//! One CUDA device is one device as returned by `zeDeviceGet` for every Level Zero
//! driver. Whether a multi-tile GPU appears as one device (composite hierarchy, the
//! default) or one per tile (`ZE_FLAT_DEVICE_HIERARCHY=FLAT`), and which devices exist
//! at all (`ZE_AFFINITY_MASK`), is decided by the Level Zero driver before this code runs.

mod ipc;
mod order;
pub(crate) mod peer;
pub(crate) mod probe;

use super::*;
use crate::level_zero::Device;
pub(crate) use ipc::*;

fn key_of(device: &Device) -> order::Key {
    order::Key {
        integrated: device.info.device_flags & 1 != 0,
        memory: device.info.total_memory,
        pci: device.info.pci_address,
        driver: device.driver_index,
        index: device.device_index,
        uuid: device.info.uuid,
    }
}

/// Environment that shapes the device list, read once by `cuInit`.
#[derive(Clone, Debug, Default)]
pub(crate) struct Selection {
    /// `CUDA_DEVICE_ORDER`
    pub order: Option<String>,
    /// `CUDA_VISIBLE_DEVICES`
    pub visible: Option<String>,
}

impl Selection {
    pub(crate) fn from_process() -> Self {
        Self {
            order: std::env::var("CUDA_DEVICE_ORDER").ok(),
            visible: std::env::var("CUDA_VISIBLE_DEVICES").ok(),
        }
    }
}

/// Give every device a distinct, stable UUID, order the list like CUDA and apply
/// `CUDA_VISIBLE_DEVICES`. With one device and no variables set nothing changes.
pub(crate) fn arrange(devices: &mut Vec<Device>, selection: &Selection) {
    // A lone device keeps whatever the driver reported.
    let devices_total = devices.len();
    for device in devices.iter_mut() {
        if devices_total > 1 && device.info.uuid == [0; 16] {
            let key = key_of(device);
            device.info.uuid = order::synthesize_uuid(&key, &device.info.name, device.info.device_id);
        }
    }
    let keys: Vec<_> = devices.iter().map(key_of).collect();
    let permutation = order::sorted(&keys, order::parse_order(selection.order.as_deref()));
    let mut slots: Vec<Option<Device>> = std::mem::take(devices).into_iter().map(Some).collect();
    *devices = permutation
        .into_iter()
        .filter_map(|index| slots[index].take())
        .collect();
    if let Some(visible) = &selection.visible {
        let uuids: Vec<_> = devices.iter().map(|d| d.info.uuid).collect();
        let chosen = order::parse_visible(visible, &uuids);
        let mut slots: Vec<Option<Device>> = std::mem::take(devices).into_iter().map(Some).collect();
        *devices = chosen
            .into_iter()
            .filter_map(|index| slots[index].take())
            .collect();
    }
}

/// Attributes that depend on the whole device list rather than on one device.
pub(crate) fn list_attribute(runtime: &Runtime, ordinal: c_int, attr: c_int) -> Option<c_int> {
    // 84 MULTI_GPU_BOARD, 85 MULTI_GPU_BOARD_GROUP_ID: tiles of one card (flat hierarchy)
    // share a PCI address and form one board.
    if !matches!(attr, 84 | 85) {
        return None;
    }
    let me = runtime.devices.get(ordinal as usize)?;
    let address = me.info.pci_address?;
    let mut members = runtime
        .devices
        .iter()
        .enumerate()
        .filter(|(_, d)| d.info.pci_address == Some(address));
    let first = members.next().map(|(i, _)| i)?;
    Some(match attr {
        84 => c_int::from(members.next().is_some()),
        _ => first as c_int,
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuDeviceGetUuid_v2(out: *mut [u8; 16], ordinal: c_int) -> CuResult {
    unsafe { cuDeviceGetUuid(out, ordinal) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuDeviceGetByPCIBusId(
    out: *mut c_int,
    bus_id: *const c_char,
) -> CuResult {
    trace::call("cuDeviceGetByPCIBusId", || {
        if out.is_null() || bus_id.is_null() {
            return InvalidValue;
        }
        let Some(address) = unsafe { CStr::from_ptr(bus_id) }
            .to_str()
            .ok()
            .and_then(order::parse_pci_bus_id)
        else {
            return InvalidValue;
        };
        status(runtime(|runtime| {
            let ordinal = runtime
                .devices
                .iter()
                .position(|d| d.info.pci_address == Some(address))
                .ok_or(InvalidDevice)?;
            unsafe { out.write(ordinal as c_int) };
            Ok(())
        }))
    })
}

#[cfg(test)]
mod tests;
