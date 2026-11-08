//! Hardware-independent device ordering, `CUDA_VISIBLE_DEVICES` parsing and UUID helpers.
//!
//! Level Zero hands out devices per driver in driver order. CUDA instead numbers them
//! fastest-first (`CUDA_DEVICE_ORDER=FASTEST_FIRST`, the default) or by PCI bus id, and
//! `CUDA_VISIBLE_DEVICES` then selects and renumbers a subset. Tile-level selection is
//! left to Level Zero itself (`ZE_AFFINITY_MASK`, `ZE_FLAT_DEVICE_HIERARCHY`), which the
//! driver applies before the shim sees any device.

/// What ordering and filtering need to know about one device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Key {
    pub integrated: bool,
    pub memory: u64,
    pub pci: Option<[u32; 4]>,
    pub driver: u32,
    pub index: u32,
    pub uuid: [u8; 16],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Order {
    Fastest,
    PciBusId,
}

/// `CUDA_DEVICE_ORDER`: `PCI_BUS_ID` or anything else (the default, `FASTEST_FIRST`).
pub(crate) fn parse_order(value: Option<&str>) -> Order {
    match value.map(str::trim) {
        Some(v) if v.eq_ignore_ascii_case("PCI_BUS_ID") => Order::PciBusId,
        _ => Order::Fastest,
    }
}

/// Indices of `keys` in CUDA order. Discrete GPUs come before integrated ones and
/// larger memory first as the performance proxy; ties keep Level Zero order.
pub(crate) fn sorted(keys: &[Key], order: Order) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..keys.len()).collect();
    indices.sort_by(|&a, &b| {
        let (x, y) = (&keys[a], &keys[b]);
        let natural = (x.driver, x.index).cmp(&(y.driver, y.index));
        match order {
            Order::Fastest => x
                .integrated
                .cmp(&y.integrated)
                .then(y.memory.cmp(&x.memory))
                .then(natural),
            Order::PciBusId => match (x.pci, y.pci) {
                (Some(p), Some(q)) => p.cmp(&q).then(natural),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => natural,
            },
        }
    });
    indices
}

/// `GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`, the form `nvidia-smi -L` and
/// `CUDA_VISIBLE_DEVICES` use.
pub(crate) fn format_uuid(uuid: &[u8; 16]) -> String {
    let hex: String = uuid.iter().map(|b| format!("{b:02x}")).collect();
    format!(
        "GPU-{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

/// Deterministic stand-in for a driver that reports an all-zero device UUID, so two
/// such devices stay distinguishable and the value is stable across runs.
pub(crate) fn synthesize_uuid(key: &Key, name: &str, device_id: u32) -> [u8; 16] {
    let mut out = [0u8; 16];
    for (lane, chunk) in out.chunks_mut(8).enumerate() {
        // FNV-1a, two lanes with different offsets.
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325 ^ (lane as u64).wrapping_mul(0x9e37_79b9);
        let mut feed = |bytes: &[u8]| {
            for &b in bytes {
                hash ^= u64::from(b);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        feed(name.as_bytes());
        feed(&device_id.to_le_bytes());
        feed(&key.driver.to_le_bytes());
        feed(&key.index.to_le_bytes());
        if let Some(pci) = key.pci {
            for part in pci {
                feed(&part.to_le_bytes());
            }
        }
        chunk.copy_from_slice(&hash.to_le_bytes());
    }
    out
}

/// Devices selected by a `CUDA_VISIBLE_DEVICES` value, as indices into `uuids` (the
/// CUDA-ordered device list), in the order listed.
///
/// Like CUDA, parsing stops at the first entry that is not a valid, unused index or a
/// unique `GPU-` UUID prefix; devices named before it stay visible. An empty value, or
/// one that starts with an invalid entry (for example `-1`), hides every device.
pub(crate) fn parse_visible(value: &str, uuids: &[[u8; 16]]) -> Vec<usize> {
    let mut chosen: Vec<usize> = Vec::new();
    if value.trim().is_empty() {
        return chosen;
    }
    for token in value.split(',') {
        let token = token.trim();
        let Some(index) = resolve(token, uuids) else {
            break;
        };
        if chosen.contains(&index) {
            break;
        }
        chosen.push(index);
    }
    chosen
}

fn resolve(token: &str, uuids: &[[u8; 16]]) -> Option<usize> {
    if token.is_empty() {
        return None;
    }
    if token.bytes().all(|b| b.is_ascii_digit()) {
        return token.parse::<usize>().ok().filter(|&i| i < uuids.len());
    }
    let prefix = token.get(..4).filter(|p| p.eq_ignore_ascii_case("GPU-"))?;
    let wanted: String = token[prefix.len()..]
        .chars()
        .filter(|&c| c != '-')
        .map(|c| c.to_ascii_lowercase())
        .collect();
    if wanted.is_empty() || !wanted.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let mut found = None;
    for (index, uuid) in uuids.iter().enumerate() {
        let hex: String = uuid.iter().map(|b| format!("{b:02x}")).collect();
        if hex.starts_with(&wanted) {
            if found.is_some() {
                return None; // ambiguous prefix
            }
            found = Some(index);
        }
    }
    found
}

/// Parse `[dddd:]bb:dd.f` (hex, case-insensitive) as used by `cuDeviceGetByPCIBusId`.
pub(crate) fn parse_pci_bus_id(text: &str) -> Option<[u32; 4]> {
    let (rest, function) = text.trim().rsplit_once('.')?;
    let mut parts = rest.split(':');
    let (domain, bus, device) = match (parts.next()?, parts.next()?, parts.next()) {
        (domain, bus, Some(device)) => (domain, bus, device),
        (bus, device, None) => ("0", bus, device),
    };
    if parts.next().is_some() {
        return None;
    }
    let hex = |s: &str, max: usize| {
        (!s.is_empty() && s.len() <= max && s.bytes().all(|b| b.is_ascii_hexdigit()))
            .then(|| u32::from_str_radix(s, 16).ok())
            .flatten()
    };
    let function = hex(function, 1).filter(|&f| f < 8)?;
    let device = hex(device, 2).filter(|&d| d < 32)?;
    Some([hex(domain, 8)?, hex(bus, 2)?, device, function])
}

/// `dddd:bb:dd.f`, the layout of `cuDeviceGetPCIBusId`.
pub(crate) fn format_pci_bus_id(address: [u32; 4]) -> String {
    let [domain, bus, slot, function] = address;
    format!("{domain:04x}:{bus:02x}:{slot:02x}.{function:x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(integrated: bool, memory: u64, driver: u32, index: u32) -> Key {
        Key {
            integrated,
            memory,
            pci: None,
            driver,
            index,
            uuid: [0; 16],
        }
    }

    fn uuid(first: u8, second: u8) -> [u8; 16] {
        let mut u = [0u8; 16];
        u[0] = first;
        u[1] = second;
        u
    }

    #[test]
    fn discrete_gpus_come_before_integrated_then_larger_memory() {
        let keys = [
            key(true, 8 << 30, 0, 0),
            key(false, 8 << 30, 0, 1),
            key(false, 16 << 30, 1, 0),
        ];
        assert_eq!(sorted(&keys, Order::Fastest), [2, 1, 0]);
    }

    #[test]
    fn equal_devices_keep_level_zero_order_across_drivers() {
        let keys = [
            key(false, 1, 1, 0),
            key(false, 1, 0, 1),
            key(false, 1, 0, 0),
        ];
        assert_eq!(sorted(&keys, Order::Fastest), [2, 1, 0]);
    }

    #[test]
    fn pci_order_sorts_by_bus_with_unknown_addresses_last() {
        let mut keys = [
            key(false, 1, 0, 0),
            key(false, 1, 0, 1),
            key(false, 1, 0, 2),
        ];
        keys[0].pci = Some([0, 0x3a, 0, 0]);
        keys[2].pci = Some([0, 0x0b, 0, 0]);
        assert_eq!(sorted(&keys, Order::PciBusId), [2, 0, 1]);
        keys[1].pci = Some([0, 0x3a, 0, 0]);
        // Same address: Level Zero order decides.
        assert_eq!(sorted(&keys, Order::PciBusId), [2, 0, 1]);
    }

    #[test]
    fn device_order_variable_is_case_insensitive() {
        assert_eq!(parse_order(Some("pci_bus_id")), Order::PciBusId);
        assert_eq!(parse_order(Some(" PCI_BUS_ID ")), Order::PciBusId);
        assert_eq!(parse_order(Some("FASTEST_FIRST")), Order::Fastest);
        assert_eq!(parse_order(Some("bogus")), Order::Fastest);
        assert_eq!(parse_order(None), Order::Fastest);
    }

    #[test]
    fn uuid_string_uses_the_nvidia_layout() {
        let u: [u8; 16] = std::array::from_fn(|i| (i as u8) * 0x11);
        assert_eq!(
            format_uuid(&u),
            "GPU-00112233-4455-6677-8899-aabbccddeeff"
        );
    }

    #[test]
    fn synthesized_uuids_are_stable_and_distinguish_devices() {
        let a = key(false, 1, 0, 0);
        let b = key(false, 1, 0, 1);
        assert_eq!(
            synthesize_uuid(&a, "Arc", 0x56a0),
            synthesize_uuid(&a, "Arc", 0x56a0)
        );
        assert_ne!(
            synthesize_uuid(&a, "Arc", 0x56a0),
            synthesize_uuid(&b, "Arc", 0x56a0)
        );
        assert_ne!(synthesize_uuid(&a, "Arc", 0x56a0), [0; 16]);
    }

    #[test]
    fn visible_devices_select_and_renumber_by_index() {
        let ids = [uuid(1, 0), uuid(2, 0), uuid(3, 0)];
        assert_eq!(parse_visible("2,0", &ids), [2, 0]);
        assert_eq!(parse_visible(" 1 , 2 ", &ids), [1, 2]);
        assert_eq!(parse_visible("0,1,2", &ids), [0, 1, 2]);
    }

    #[test]
    fn visible_devices_stop_at_the_first_invalid_entry() {
        let ids = [uuid(1, 0), uuid(2, 0)];
        assert_eq!(parse_visible("0,5,1", &ids), [0]);
        assert_eq!(parse_visible("5,0", &ids), Vec::<usize>::new());
        assert_eq!(parse_visible("-1", &ids), Vec::<usize>::new());
        assert_eq!(parse_visible("0,x,1", &ids), [0]);
        assert_eq!(parse_visible("1,1", &ids), [1], "duplicates end the list");
        assert_eq!(parse_visible("0,,1", &ids), [0]);
    }

    #[test]
    fn empty_visible_devices_hide_everything() {
        let ids = [uuid(1, 0)];
        assert!(parse_visible("", &ids).is_empty());
        assert!(parse_visible("   ", &ids).is_empty());
    }

    #[test]
    fn visible_devices_accept_uuids_and_unique_prefixes() {
        let ids = [uuid(0xab, 0xcd), uuid(0xab, 0xce), uuid(0x10, 0x00)];
        let full = format_uuid(&ids[1]);
        assert_eq!(parse_visible(&full, &ids), [1]);
        assert_eq!(parse_visible(&full.to_uppercase().replace("GPU-", "gpu-"), &ids), [1]);
        assert_eq!(parse_visible("GPU-abcd", &ids), [0]);
        assert_eq!(parse_visible("GPU-10", &ids), [2]);
        // "ab" matches two devices: ambiguous, so invalid.
        assert!(parse_visible("GPU-ab", &ids).is_empty());
        assert!(parse_visible("GPU-ffff", &ids).is_empty());
        assert!(parse_visible("GPU-", &ids).is_empty());
        assert!(parse_visible("GPU-xyz", &ids).is_empty());
        // A bare UUID without the prefix is not accepted.
        assert!(parse_visible("abcd", &ids).is_empty());
        assert_eq!(parse_visible(&format!("{full},0"), &ids), [1, 0]);
    }

    #[test]
    fn pci_bus_ids_round_trip() {
        assert_eq!(parse_pci_bus_id("0000:03:00.0"), Some([0, 3, 0, 0]));
        assert_eq!(parse_pci_bus_id("0001:AF:1f.7"), Some([1, 0xaf, 0x1f, 7]));
        assert_eq!(parse_pci_bus_id("03:00.0"), Some([0, 3, 0, 0]));
        assert_eq!(format_pci_bus_id([1, 0xaf, 0x1f, 7]), "0001:af:1f.7");
        for bad in ["", "0000:03:00", "0000:03:20.0", "0000:03:00.8", "zz:00.0", "0:0:0:0.0", "0000:100:00.0"] {
            assert_eq!(parse_pci_bus_id(bad), None, "{bad}");
        }
    }
}
