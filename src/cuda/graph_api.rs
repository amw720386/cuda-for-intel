//! Explicit CUDA graph construction and inspection: node creation, edges,
//! node parameters, cloning, capture dependency updates and host functions.
//!
//! Graphs are [`Dag`]s kept in creation order. Instantiation orders them
//! topologically (see `Dag::order`) and `cuGraphLaunch` replays that order.

use super::graphs::{Copy, CopyKind, Dag, Fill, HostFn, Kernel, Node, begin_capture, call_host};
use super::*;
use crate::ptx::Param;
use std::fmt::Write as _;

// CUgraphNodeType
const KERNEL: c_int = 0;
const MEMCPY: c_int = 1;
const MEMSET: c_int = 2;
const HOST: c_int = 3;
const GRAPH: c_int = 4;
const EMPTY: c_int = 5;
const WAIT_EVENT: c_int = 6;
const EVENT_RECORD: c_int = 7;

// CUmemorytype
const HOST_MEMORY: c_uint = 1;
const DEVICE_MEMORY: c_uint = 2;
const UNIFIED_MEMORY: c_uint = 4;

/// CUDA_KERNEL_NODE_PARAMS (v1 prefix, v2/v3 with `kernel` and `context`).
#[repr(C)]
pub struct KernelNodeParams {
    pub(crate) function: Opaque,
    pub(crate) grid: [u32; 3],
    pub(crate) block: [u32; 3],
    pub(crate) shared: u32,
    pub(crate) kernel_params: *mut Opaque,
    pub(crate) extra: *mut Opaque,
    pub(crate) kernel: Opaque,
    pub(crate) context: Opaque,
}

/// CUDA_KERNEL_NODE_PARAMS_v1 has no trailing kernel and context.
const KERNEL_PARAMS_V1_SIZE: usize = 56;

/// CUDA_MEMCPY3D.
#[repr(C)]
pub struct MemcpyNodeParams {
    pub(crate) src_x: usize,
    pub(crate) src_y: usize,
    pub(crate) src_z: usize,
    pub(crate) src_lod: usize,
    pub(crate) src_type: c_uint,
    pub(crate) src_host: *const c_void,
    pub(crate) src_device: u64,
    pub(crate) src_array: Opaque,
    pub(crate) reserved0: Opaque,
    pub(crate) src_pitch: usize,
    pub(crate) src_height: usize,
    pub(crate) dst_x: usize,
    pub(crate) dst_y: usize,
    pub(crate) dst_z: usize,
    pub(crate) dst_lod: usize,
    pub(crate) dst_type: c_uint,
    pub(crate) dst_host: *mut c_void,
    pub(crate) dst_device: u64,
    pub(crate) dst_array: Opaque,
    pub(crate) reserved1: Opaque,
    pub(crate) dst_pitch: usize,
    pub(crate) dst_height: usize,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) depth: usize,
}

/// CUDA_MEMSET_NODE_PARAMS.
#[repr(C)]
pub struct MemsetNodeParams {
    pub(crate) destination: u64,
    pub(crate) pitch: usize,
    pub(crate) value: u32,
    pub(crate) element_size: u32,
    pub(crate) width: usize,
    pub(crate) height: usize,
}

/// CUDA_HOST_NODE_PARAMS.
#[repr(C)]
pub struct HostNodeParams {
    pub(crate) function: Opaque,
    pub(crate) data: Opaque,
}

/// CUgraphEdgeData. Only default edges exist here.
#[repr(C)]
pub struct EdgeData {
    pub(crate) from_port: u8,
    pub(crate) to_port: u8,
    pub(crate) kind: u8,
    pub(crate) reserved: [u8; 5],
}

unsafe fn default_edges(edges: *const EdgeData, count: usize) -> Result<(), CuResult> {
    if edges.is_null() {
        return Ok(());
    }
    for index in 0..count {
        let edge = unsafe { &*edges.add(index) };
        if edge.from_port != 0 || edge.to_port != 0 || edge.kind != 0 {
            return Err(NotSupported);
        }
    }
    Ok(())
}

unsafe fn handles(list: *const Opaque, count: usize) -> Result<Vec<usize>, CuResult> {
    if count == 0 {
        return Ok(Vec::new());
    }
    if list.is_null() {
        return Err(InvalidValue);
    }
    Ok((0..count).map(|i| unsafe { list.add(i).read() }.addr()).collect())
}

fn node_ptr(id: usize) -> Opaque {
    std::ptr::without_provenance_mut(id)
}

/// Write `ids` with the CUDA in/out count convention: a null array only
/// reports the total, otherwise the first `capacity` handles are written, the
/// rest of the array is nulled and the count written back.
unsafe fn write_list(ids: &[usize], out: *mut Opaque, count: *mut usize) -> Result<(), CuResult> {
    if count.is_null() {
        return Err(InvalidValue);
    }
    if out.is_null() {
        unsafe { count.write(ids.len()) };
        return Ok(());
    }
    let capacity = unsafe { count.read() };
    for index in 0..capacity {
        unsafe { out.add(index).write(ids.get(index).map_or(std::ptr::null_mut(), |&id| node_ptr(id))) };
    }
    unsafe { count.write(ids.len().min(capacity)) };
    Ok(())
}

fn owning_graph(state: &State, node: Opaque) -> Result<usize, CuResult> {
    state.graphs.owner(node.addr()).ok_or(InvalidValue)
}

fn dag_of(state: &mut State, graph: Opaque) -> Result<&mut Dag, CuResult> {
    state.graphs.graphs.get_mut(&graph.addr()).ok_or(InvalidValue)
}

fn node_kind(node: &Node) -> c_int {
    match node {
        Node::Kernel(_) => KERNEL,
        Node::Copy(_) => MEMCPY,
        Node::Fill(_) => MEMSET,
        Node::Host(_) => HOST,
        Node::Child(_) => GRAPH,
        Node::Empty => EMPTY,
        Node::EventWait(_) => WAIT_EVENT,
        Node::EventRecord(_) => EVENT_RECORD,
    }
}

// ---------------------------------------------------------------------------
// Graph objects

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphCreate(out: *mut Opaque, flags: u32) -> CuResult {
    trace::call("cuGraphCreate", || {
        if out.is_null() || flags != 0 {
            return InvalidValue;
        }
        status(state(|state| {
            let owner = current().ok_or(InvalidContext)?;
            let handle = reserve_handle(&mut state.next_handle)?;
            state.graphs.graphs.insert(handle, Dag::new(owner));
            unsafe { out.write(node_ptr(handle)) };
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphClone(out: *mut Opaque, original: Opaque) -> CuResult {
    trace::call("cuGraphClone", || {
        if out.is_null() {
            return InvalidValue;
        }
        status(state(|state| {
            let dag = dag_of(state, original)?;
            if dag.capturing {
                return Err(IllegalState);
            }
            let copy = dag.duplicate();
            let handle = reserve_handle(&mut state.next_handle)?;
            state.graphs.graphs.insert(handle, copy);
            unsafe { out.write(node_ptr(handle)) };
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphNodeFindInClone(
    out: *mut Opaque,
    original: Opaque,
    clone: Opaque,
) -> CuResult {
    trace::call("cuGraphNodeFindInClone", || {
        if out.is_null() {
            return InvalidValue;
        }
        status(state(|state| {
            let found = *dag_of(state, clone)?
                .clone_map
                .get(&original.addr())
                .ok_or(InvalidValue)?;
            unsafe { out.write(node_ptr(found)) };
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphGetNodes(
    graph: Opaque,
    nodes: *mut Opaque,
    count: *mut usize,
) -> CuResult {
    trace::call("cuGraphGetNodes", || {
        status(state(|state| {
            let ids: Vec<usize> = dag_of(state, graph)?.entries.iter().map(|e| e.id).collect();
            unsafe { write_list(&ids, nodes, count) }
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphGetRootNodes(
    graph: Opaque,
    nodes: *mut Opaque,
    count: *mut usize,
) -> CuResult {
    trace::call("cuGraphGetRootNodes", || {
        status(state(|state| {
            let ids = dag_of(state, graph)?.roots();
            unsafe { write_list(&ids, nodes, count) }
        }))
    })
}

fn edges_of(dag: &Dag) -> Vec<(usize, usize)> {
    dag.entries
        .iter()
        .flat_map(|entry| entry.deps.iter().map(|&from| (from, entry.id)))
        .collect()
}

unsafe fn get_edges(
    graph: Opaque,
    from: *mut Opaque,
    to: *mut Opaque,
    data: *mut EdgeData,
    count: *mut usize,
) -> Result<(), CuResult> {
    if count.is_null() {
        return Err(InvalidValue);
    }
    state(|state| {
        let edges = edges_of(dag_of(state, graph)?);
        if from.is_null() || to.is_null() {
            unsafe { count.write(edges.len()) };
            return Ok(());
        }
        let capacity = unsafe { count.read() };
        for index in 0..capacity {
            let (a, b) = edges.get(index).copied().unwrap_or((0, 0));
            unsafe {
                from.add(index).write(node_ptr(a));
                to.add(index).write(node_ptr(b));
                if !data.is_null() {
                    data.add(index).write(EdgeData {
                        from_port: 0,
                        to_port: 0,
                        kind: 0,
                        reserved: [0; 5],
                    });
                }
            }
        }
        unsafe { count.write(edges.len().min(capacity)) };
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphGetEdges(
    graph: Opaque,
    from: *mut Opaque,
    to: *mut Opaque,
    count: *mut usize,
) -> CuResult {
    trace::call("cuGraphGetEdges", || {
        status(unsafe { get_edges(graph, from, to, std::ptr::null_mut(), count) })
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphGetEdges_v2(
    graph: Opaque,
    from: *mut Opaque,
    to: *mut Opaque,
    data: *mut EdgeData,
    count: *mut usize,
) -> CuResult {
    trace::call("cuGraphGetEdges_v2", || {
        status(unsafe { get_edges(graph, from, to, data, count) })
    })
}

unsafe fn node_relatives(
    node: Opaque,
    out: *mut Opaque,
    data: *mut EdgeData,
    count: *mut usize,
    parents: bool,
) -> Result<(), CuResult> {
    state(|state| {
        let graph = owning_graph(state, node)?;
        let dag = &state.graphs.graphs[&graph];
        let ids = if parents {
            dag.entry(node.addr())?.deps.clone()
        } else {
            dag.children(node.addr())
        };
        if !data.is_null() && !out.is_null() && !count.is_null() {
            for index in 0..unsafe { count.read() }.min(ids.len()) {
                unsafe {
                    data.add(index).write(EdgeData {
                        from_port: 0,
                        to_port: 0,
                        kind: 0,
                        reserved: [0; 5],
                    });
                }
            }
        }
        unsafe { write_list(&ids, out, count) }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphNodeGetDependencies(
    node: Opaque,
    out: *mut Opaque,
    count: *mut usize,
) -> CuResult {
    trace::call("cuGraphNodeGetDependencies", || {
        status(unsafe { node_relatives(node, out, std::ptr::null_mut(), count, true) })
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphNodeGetDependencies_v2(
    node: Opaque,
    out: *mut Opaque,
    data: *mut EdgeData,
    count: *mut usize,
) -> CuResult {
    trace::call("cuGraphNodeGetDependencies_v2", || {
        status(unsafe { node_relatives(node, out, data, count, true) })
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphNodeGetDependentNodes(
    node: Opaque,
    out: *mut Opaque,
    count: *mut usize,
) -> CuResult {
    trace::call("cuGraphNodeGetDependentNodes", || {
        status(unsafe { node_relatives(node, out, std::ptr::null_mut(), count, false) })
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphNodeGetDependentNodes_v2(
    node: Opaque,
    out: *mut Opaque,
    data: *mut EdgeData,
    count: *mut usize,
) -> CuResult {
    trace::call("cuGraphNodeGetDependentNodes_v2", || {
        status(unsafe { node_relatives(node, out, data, count, false) })
    })
}

unsafe fn change_edges(
    graph: Opaque,
    from: *const Opaque,
    to: *const Opaque,
    data: *const EdgeData,
    count: usize,
    add: bool,
) -> Result<(), CuResult> {
    unsafe { default_edges(data, count)? };
    let from = unsafe { handles(from, count)? };
    let to = unsafe { handles(to, count)? };
    state(|state| {
        let dag = dag_of(state, graph)?;
        // All or nothing.
        let mut staged = dag.clone();
        for (&a, &b) in from.iter().zip(&to) {
            if add {
                staged.add_edge(a, b)?;
            } else {
                staged.remove_edge(a, b)?;
            }
        }
        *dag = staged;
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphAddDependencies(
    graph: Opaque,
    from: *const Opaque,
    to: *const Opaque,
    count: usize,
) -> CuResult {
    trace::call("cuGraphAddDependencies", || {
        status(unsafe { change_edges(graph, from, to, std::ptr::null(), count, true) })
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphAddDependencies_v2(
    graph: Opaque,
    from: *const Opaque,
    to: *const Opaque,
    data: *const EdgeData,
    count: usize,
) -> CuResult {
    trace::call("cuGraphAddDependencies_v2", || {
        status(unsafe { change_edges(graph, from, to, data, count, true) })
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphRemoveDependencies(
    graph: Opaque,
    from: *const Opaque,
    to: *const Opaque,
    count: usize,
) -> CuResult {
    trace::call("cuGraphRemoveDependencies", || {
        status(unsafe { change_edges(graph, from, to, std::ptr::null(), count, false) })
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphRemoveDependencies_v2(
    graph: Opaque,
    from: *const Opaque,
    to: *const Opaque,
    data: *const EdgeData,
    count: usize,
) -> CuResult {
    trace::call("cuGraphRemoveDependencies_v2", || {
        status(unsafe { change_edges(graph, from, to, data, count, false) })
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuGraphDestroyNode(node: Opaque) -> CuResult {
    trace::call("cuGraphDestroyNode", || {
        status(state(|state| {
            let graph = owning_graph(state, node)?;
            state.graphs.param_storage.remove(&node.addr());
            state
                .graphs
                .graphs
                .get_mut(&graph)
                .ok_or(InvalidValue)?
                .remove_node(node.addr())
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphNodeGetType(node: Opaque, out: *mut c_int) -> CuResult {
    trace::call("cuGraphNodeGetType", || {
        if out.is_null() {
            return InvalidValue;
        }
        status(state(|state| {
            let graph = owning_graph(state, node)?;
            let kind = node_kind(&state.graphs.graphs[&graph].entry(node.addr())?.node);
            unsafe { out.write(kind) };
            Ok(())
        }))
    })
}

/// Write the graph in Graphviz format. Flags (verbosity) are accepted and ignored.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphDebugDotPrint(
    graph: Opaque,
    path: *const c_char,
    _flags: u32,
) -> CuResult {
    trace::call("cuGraphDebugDotPrint", || {
        if path.is_null() {
            return InvalidValue;
        }
        let path = unsafe { CStr::from_ptr(path) }.to_string_lossy().into_owned();
        let text = state(|state| {
            let dag = dag_of(state, graph)?;
            let mut text = String::from("digraph dot {\n");
            for entry in &dag.entries {
                let _ = writeln!(
                    text,
                    "  \"{:#x}\" [label=\"{}\"];",
                    entry.id,
                    match node_kind(&entry.node) {
                        KERNEL => "KERNEL",
                        MEMCPY => "MEMCPY",
                        MEMSET => "MEMSET",
                        HOST => "HOST",
                        GRAPH => "GRAPH",
                        EMPTY => "EMPTY",
                        WAIT_EVENT => "EVENT_WAIT",
                        _ => "EVENT_RECORD",
                    }
                );
            }
            for (from, to) in edges_of(dag) {
                let _ = writeln!(text, "  \"{from:#x}\" -> \"{to:#x}\";");
            }
            text.push_str("}\n");
            Ok(text)
        });
        match text {
            Ok(text) => match std::fs::write(path, text) {
                Ok(()) => Success,
                Err(_) => InvalidValue,
            },
            Err(error) => error,
        }
    })
}

// ---------------------------------------------------------------------------
// Node construction

fn add_node(
    out: *mut Opaque,
    graph: Opaque,
    dependencies: *const Opaque,
    count: usize,
    make: impl FnOnce(&mut State, usize) -> Result<Node, CuResult>,
) -> Result<(), CuResult> {
    if out.is_null() {
        return Err(InvalidValue);
    }
    let dependencies = unsafe { handles(dependencies, count)? };
    state(|state| {
        let owner = dag_of(state, graph)?.context;
        let node = make(state, owner)?;
        let id = dag_of(state, graph)?.add(node, &dependencies)?;
        unsafe { out.write(node_ptr(id)) };
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphAddEmptyNode(
    out: *mut Opaque,
    graph: Opaque,
    dependencies: *const Opaque,
    count: usize,
) -> CuResult {
    trace::call("cuGraphAddEmptyNode", || {
        status(add_node(out, graph, dependencies, count, |_, _| Ok(Node::Empty)))
    })
}

fn kernel_node(state: &mut State, owner: usize, params: &KernelNodeParams, v2: bool) -> Result<Kernel, CuResult> {
    let mut function = if params.function.is_null() && v2 {
        params.kernel.addr()
    } else {
        params.function.addr()
    };
    if state.libraries.contains_kernel(function) {
        function = library::kernel_function(state, function)?;
    }
    let runtime = &state.runtime;
    let context = state.contexts.get(&owner).ok_or(ContextIsDestroyed)?;
    let FunctionRef { module: module_key, name, .. } = context.functions.get(&function).ok_or(InvalidHandle)?;
    let module = context.modules.get(module_key).ok_or(InvalidHandle)?;
    let entry = module.entries.get(name).ok_or(InvalidHandle)?;
    entry.function.ok_or(InvalidHandle)?;
    if entry.shared_bytes != 0 && params.shared > entry.shared_bytes {
        return Err(InvalidValue);
    }
    validate_launch(
        &runtime.devices[context.device as usize].info,
        params.grid,
        params.block,
        entry.max_threads,
    )?;
    let args = unsafe {
        launch_args::read(&entry.params, params.kernel_params, params.extra, |address| {
            contains_range(&context.allocations, address as usize, 1)
                || contains_range(&context.host_allocations, address as usize, 1)
        })
    }?;
    let mut kernel = unsafe { Kernel::capture(function, (params.grid, params.block), &args) };
    kernel.shared = params.shared;
    Ok(kernel)
}

/// Read only the bytes the caller's layout defines: v1 callers pass a
/// 56-byte structure, so the trailing fields must not be touched.
unsafe fn load_kernel_params(raw: *const KernelNodeParams, v2: bool) -> Result<KernelNodeParams, CuResult> {
    if raw.is_null() {
        return Err(InvalidValue);
    }
    if v2 {
        return Ok(unsafe { raw.read_unaligned() });
    }
    let mut copy = KernelNodeParams {
        function: std::ptr::null_mut(),
        grid: [0; 3],
        block: [0; 3],
        shared: 0,
        kernel_params: std::ptr::null_mut(),
        extra: std::ptr::null_mut(),
        kernel: std::ptr::null_mut(),
        context: std::ptr::null_mut(),
    };
    unsafe {
        std::ptr::copy_nonoverlapping(raw.cast::<u8>(), (&raw mut copy).cast::<u8>(), KERNEL_PARAMS_V1_SIZE)
    };
    Ok(copy)
}

fn add_kernel(
    out: *mut Opaque,
    graph: Opaque,
    dependencies: *const Opaque,
    count: usize,
    params: *const KernelNodeParams,
    v2: bool,
) -> CuResult {
    status((|| {
        let params = unsafe { load_kernel_params(params, v2)? };
        add_node(out, graph, dependencies, count, |state, owner| {
            kernel_node(state, owner, &params, v2).map(Node::Kernel)
        })
    })())
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphAddKernelNode(
    out: *mut Opaque,
    graph: Opaque,
    dependencies: *const Opaque,
    count: usize,
    params: *const KernelNodeParams,
) -> CuResult {
    trace::call("cuGraphAddKernelNode", || {
        add_kernel(out, graph, dependencies, count, params, false)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphAddKernelNode_v2(
    out: *mut Opaque,
    graph: Opaque,
    dependencies: *const Opaque,
    count: usize,
    params: *const KernelNodeParams,
) -> CuResult {
    trace::call("cuGraphAddKernelNode_v2", || {
        add_kernel(out, graph, dependencies, count, params, true)
    })
}

fn copy_node(state: &State, owner: usize, params: &MemcpyNodeParams) -> Result<Copy, CuResult> {
    let context = state.contexts.get(&owner).ok_or(ContextIsDestroyed)?;
    let (height, depth) = (params.height.max(1), params.depth.max(1));
    if height != 1
        || depth != 1
        || params.src_y != 0
        || params.src_z != 0
        || params.dst_y != 0
        || params.dst_z != 0
        || params.src_lod != 0
        || params.dst_lod != 0
    {
        return Err(NotSupported);
    }
    let size = params.width;
    // Unified addresses name device memory when this context owns them.
    let device = |kind: c_uint, host: usize, device: u64, offset: usize| match kind {
        HOST_MEMORY => Ok((false, host + offset)),
        DEVICE_MEMORY => Ok((true, device as usize + offset)),
        UNIFIED_MEMORY => {
            let address = if device != 0 { device as usize } else { host } + offset;
            Ok((contains_range(&context.allocations, address, 1), address))
        }
        _ => Err(NotSupported),
    };
    let (src_device, src) = device(params.src_type, params.src_host.addr(), params.src_device, params.src_x)?;
    let (dst_device, dst) = device(params.dst_type, params.dst_host.addr(), params.dst_device, params.dst_x)?;
    let kind = match (src_device, dst_device) {
        (false, true) => CopyKind::HostToDevice,
        (true, false) => CopyKind::DeviceToHost,
        (true, true) => CopyKind::DeviceToDevice,
        (false, false) => return Err(NotSupported),
    };
    let copy = Copy { dst, src, size, kind };
    if size != 0 {
        let device_ok = |address| contains_range(&context.allocations, address, size);
        let valid = match kind {
            CopyKind::HostToDevice => device_ok(dst) && src != 0,
            CopyKind::DeviceToHost => device_ok(src) && dst != 0,
            CopyKind::DeviceToDevice => device_ok(dst) && device_ok(src),
        };
        if !valid {
            return Err(InvalidValue);
        }
    }
    Ok(copy)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphAddMemcpyNode(
    out: *mut Opaque,
    graph: Opaque,
    dependencies: *const Opaque,
    count: usize,
    params: *const MemcpyNodeParams,
    _context: Opaque,
) -> CuResult {
    trace::call("cuGraphAddMemcpyNode", || {
        if params.is_null() {
            return InvalidValue;
        }
        let params = unsafe { params.read_unaligned() };
        status(add_node(out, graph, dependencies, count, |state, owner| {
            copy_node(state, owner, &params).map(Node::Copy)
        }))
    })
}

fn fill_node(state: &State, owner: usize, params: &MemsetNodeParams) -> Result<Fill, CuResult> {
    let context = state.contexts.get(&owner).ok_or(ContextIsDestroyed)?;
    let width = params.element_size as usize;
    let bytes = params.value.to_le_bytes();
    let value = match width {
        1 => bytes[0],
        2 if bytes[0] == bytes[1] => bytes[0],
        4 if bytes.iter().all(|&b| b == bytes[0]) => bytes[0],
        2 | 4 => return Err(NotSupported),
        _ => return Err(InvalidValue),
    };
    let row = params.width.checked_mul(width).ok_or(InvalidValue)?;
    let height = params.height.max(1);
    if height != 1 && params.pitch != row {
        return Err(NotSupported);
    }
    let size = row.checked_mul(height).ok_or(InvalidValue)?;
    if size != 0 && !contains_range(&context.allocations, params.destination as usize, size) {
        return Err(InvalidValue);
    }
    Ok(Fill {
        dst: params.destination as usize,
        value,
        size,
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphAddMemsetNode(
    out: *mut Opaque,
    graph: Opaque,
    dependencies: *const Opaque,
    count: usize,
    params: *const MemsetNodeParams,
    _context: Opaque,
) -> CuResult {
    trace::call("cuGraphAddMemsetNode", || {
        if params.is_null() {
            return InvalidValue;
        }
        let params = unsafe { params.read_unaligned() };
        status(add_node(out, graph, dependencies, count, |state, owner| {
            fill_node(state, owner, &params).map(Node::Fill)
        }))
    })
}

fn host_node(params: *const HostNodeParams) -> Result<HostFn, CuResult> {
    if params.is_null() {
        return Err(InvalidValue);
    }
    let params = unsafe { params.read_unaligned() };
    if params.function.is_null() {
        return Err(InvalidValue);
    }
    Ok(HostFn {
        function: params.function.addr(),
        data: params.data.addr(),
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphAddHostNode(
    out: *mut Opaque,
    graph: Opaque,
    dependencies: *const Opaque,
    count: usize,
    params: *const HostNodeParams,
) -> CuResult {
    trace::call("cuGraphAddHostNode", || {
        status(host_node(params).and_then(|host| {
            add_node(out, graph, dependencies, count, |_, _| Ok(Node::Host(host)))
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphAddEventRecordNode(
    out: *mut Opaque,
    graph: Opaque,
    dependencies: *const Opaque,
    count: usize,
    event: Opaque,
) -> CuResult {
    trace::call("cuGraphAddEventRecordNode", || {
        if event.is_null() {
            return InvalidValue;
        }
        status(add_node(out, graph, dependencies, count, |_, _| {
            Ok(Node::EventRecord(event.addr()))
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphAddEventWaitNode(
    out: *mut Opaque,
    graph: Opaque,
    dependencies: *const Opaque,
    count: usize,
    event: Opaque,
) -> CuResult {
    trace::call("cuGraphAddEventWaitNode", || {
        if event.is_null() {
            return InvalidValue;
        }
        status(add_node(out, graph, dependencies, count, |_, _| {
            Ok(Node::EventWait(event.addr()))
        }))
    })
}

/// The child graph is copied into the node when it is added, as in CUDA.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphAddChildGraphNode(
    out: *mut Opaque,
    graph: Opaque,
    dependencies: *const Opaque,
    count: usize,
    child: Opaque,
) -> CuResult {
    trace::call("cuGraphAddChildGraphNode", || {
        let nodes = state(|state| {
            let child = dag_of(state, child)?;
            if child.capturing {
                return Err(IllegalState);
            }
            Ok((child.context, child.flatten()))
        });
        let (child_context, nodes) = match nodes {
            Ok(found) => found,
            Err(error) => return error,
        };
        status(add_node(out, graph, dependencies, count, |_, owner| {
            if owner != child_context {
                return Err(InvalidContext);
            }
            Ok(Node::Child(nodes))
        }))
    })
}

// ---------------------------------------------------------------------------
// Node parameters

fn with_node<T>(
    node: Opaque,
    action: impl FnOnce(&mut State, usize, usize) -> Result<T, CuResult>,
) -> Result<T, CuResult> {
    state(|state| {
        let graph = owning_graph(state, node)?;
        action(state, graph, node.addr())
    })
}

/// Contiguous argument blocks for each kernel parameter, rebuilt from the
/// stored 4-byte words and scalars.
fn kernel_param_blocks(context: &CudaContext, kernel: &Kernel) -> Result<Vec<Box<[u8]>>, CuResult> {
    let FunctionRef { module, name, .. } = context.functions.get(&kernel.function).ok_or(InvalidHandle)?;
    let entry = context
        .modules
        .get(module)
        .and_then(|module| module.entries.get(name))
        .ok_or(InvalidHandle)?;
    let mut blocks = Vec::new();
    let mut index = 0;
    for param in &entry.params {
        let words = match param {
            Param::Words { words, .. } => usize::from(*words),
            _ => 1,
        };
        let mut block = Vec::new();
        for arg in kernel.args.get(index..index + words).ok_or(InvalidValue)? {
            block.extend(arg.iter().map(|byte| unsafe { byte.assume_init() }));
        }
        index += words;
        blocks.push(block.into_boxed_slice());
    }
    Ok(blocks)
}

fn get_kernel_params(node: Opaque, out: *mut KernelNodeParams, v2: bool) -> Result<(), CuResult> {
    if out.is_null() {
        return Err(InvalidValue);
    }
    with_node(node, |state, graph, id| {
        let Node::Kernel(kernel) = &state.graphs.graphs[&graph].entry(id)?.node else {
            return Err(InvalidValue);
        };
        let owner = state.graphs.graphs[&graph].context;
        let context = state.contexts.get(&owner).ok_or(ContextIsDestroyed)?;
        let blocks = kernel_param_blocks(context, kernel)?;
        let pointers: Vec<usize> = blocks.iter().map(|block| block.as_ptr().addr()).collect();
        let params = KernelNodeParams {
            function: node_ptr(kernel.function),
            grid: kernel.geometry.0,
            block: kernel.geometry.1,
            shared: kernel.shared,
            kernel_params: pointers.as_ptr().cast_mut().cast(),
            extra: std::ptr::null_mut(),
            kernel: std::ptr::null_mut(),
            context: std::ptr::null_mut(),
        };
        state.graphs.param_storage.insert(id, (blocks, pointers));
        let size = if v2 { std::mem::size_of::<KernelNodeParams>() } else { KERNEL_PARAMS_V1_SIZE };
        unsafe { std::ptr::copy_nonoverlapping((&raw const params).cast::<u8>(), out.cast::<u8>(), size) };
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphKernelNodeGetParams(node: Opaque, out: *mut KernelNodeParams) -> CuResult {
    trace::call("cuGraphKernelNodeGetParams", || status(get_kernel_params(node, out, false)))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphKernelNodeGetParams_v2(node: Opaque, out: *mut KernelNodeParams) -> CuResult {
    trace::call("cuGraphKernelNodeGetParams_v2", || status(get_kernel_params(node, out, true)))
}

fn set_kernel_params(node: Opaque, params: *const KernelNodeParams, v2: bool) -> Result<(), CuResult> {
    let params = unsafe { load_kernel_params(params, v2)? };
    with_node(node, |state, graph, id| {
        let owner = state.graphs.graphs[&graph].context;
        if !matches!(state.graphs.graphs[&graph].entry(id)?.node, Node::Kernel(_)) {
            return Err(InvalidValue);
        }
        let kernel = kernel_node(state, owner, &params, v2)?;
        state.graphs.param_storage.remove(&id);
        state.graphs.graphs.get_mut(&graph).unwrap().entry_mut(id)?.node = Node::Kernel(kernel);
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphKernelNodeSetParams(node: Opaque, params: *const KernelNodeParams) -> CuResult {
    trace::call("cuGraphKernelNodeSetParams", || status(set_kernel_params(node, params, false)))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphKernelNodeSetParams_v2(node: Opaque, params: *const KernelNodeParams) -> CuResult {
    trace::call("cuGraphKernelNodeSetParams_v2", || status(set_kernel_params(node, params, true)))
}

fn memcpy_params(copy: &Copy) -> MemcpyNodeParams {
    let (src_type, dst_type) = match copy.kind {
        CopyKind::HostToDevice => (HOST_MEMORY, DEVICE_MEMORY),
        CopyKind::DeviceToHost => (DEVICE_MEMORY, HOST_MEMORY),
        CopyKind::DeviceToDevice => (DEVICE_MEMORY, DEVICE_MEMORY),
    };
    MemcpyNodeParams {
        src_x: 0,
        src_y: 0,
        src_z: 0,
        src_lod: 0,
        src_type,
        src_host: if src_type == HOST_MEMORY { std::ptr::with_exposed_provenance(copy.src) } else { std::ptr::null() },
        src_device: if src_type == DEVICE_MEMORY { copy.src as u64 } else { 0 },
        src_array: std::ptr::null_mut(),
        reserved0: std::ptr::null_mut(),
        src_pitch: copy.size,
        src_height: 1,
        dst_x: 0,
        dst_y: 0,
        dst_z: 0,
        dst_lod: 0,
        dst_type,
        dst_host: if dst_type == HOST_MEMORY { std::ptr::with_exposed_provenance_mut(copy.dst) } else { std::ptr::null_mut() },
        dst_device: if dst_type == DEVICE_MEMORY { copy.dst as u64 } else { 0 },
        dst_array: std::ptr::null_mut(),
        reserved1: std::ptr::null_mut(),
        dst_pitch: copy.size,
        dst_height: 1,
        width: copy.size,
        height: 1,
        depth: 1,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphMemcpyNodeGetParams(node: Opaque, out: *mut MemcpyNodeParams) -> CuResult {
    trace::call("cuGraphMemcpyNodeGetParams", || {
        if out.is_null() {
            return InvalidValue;
        }
        status(with_node(node, |state, graph, id| {
            let Node::Copy(copy) = &state.graphs.graphs[&graph].entry(id)?.node else {
                return Err(InvalidValue);
            };
            unsafe { out.write_unaligned(memcpy_params(copy)) };
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphMemcpyNodeSetParams(node: Opaque, params: *const MemcpyNodeParams) -> CuResult {
    trace::call("cuGraphMemcpyNodeSetParams", || {
        if params.is_null() {
            return InvalidValue;
        }
        let params = unsafe { params.read_unaligned() };
        status(with_node(node, |state, graph, id| {
            let owner = state.graphs.graphs[&graph].context;
            if !matches!(state.graphs.graphs[&graph].entry(id)?.node, Node::Copy(_)) {
                return Err(InvalidValue);
            }
            let copy = copy_node(state, owner, &params)?;
            state.graphs.graphs.get_mut(&graph).unwrap().entry_mut(id)?.node = Node::Copy(copy);
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphMemsetNodeGetParams(node: Opaque, out: *mut MemsetNodeParams) -> CuResult {
    trace::call("cuGraphMemsetNodeGetParams", || {
        if out.is_null() {
            return InvalidValue;
        }
        status(with_node(node, |state, graph, id| {
            let Node::Fill(fill) = &state.graphs.graphs[&graph].entry(id)?.node else {
                return Err(InvalidValue);
            };
            unsafe {
                out.write_unaligned(MemsetNodeParams {
                    destination: fill.dst as u64,
                    pitch: 0,
                    value: u32::from(fill.value),
                    element_size: 1,
                    width: fill.size,
                    height: 1,
                })
            };
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphMemsetNodeSetParams(node: Opaque, params: *const MemsetNodeParams) -> CuResult {
    trace::call("cuGraphMemsetNodeSetParams", || {
        if params.is_null() {
            return InvalidValue;
        }
        let params = unsafe { params.read_unaligned() };
        status(with_node(node, |state, graph, id| {
            let owner = state.graphs.graphs[&graph].context;
            if !matches!(state.graphs.graphs[&graph].entry(id)?.node, Node::Fill(_)) {
                return Err(InvalidValue);
            }
            let fill = fill_node(state, owner, &params)?;
            state.graphs.graphs.get_mut(&graph).unwrap().entry_mut(id)?.node = Node::Fill(fill);
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphHostNodeGetParams(node: Opaque, out: *mut HostNodeParams) -> CuResult {
    trace::call("cuGraphHostNodeGetParams", || {
        if out.is_null() {
            return InvalidValue;
        }
        status(with_node(node, |state, graph, id| {
            let Node::Host(host) = &state.graphs.graphs[&graph].entry(id)?.node else {
                return Err(InvalidValue);
            };
            unsafe {
                out.write_unaligned(HostNodeParams {
                    function: node_ptr(host.function),
                    data: node_ptr(host.data),
                })
            };
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphHostNodeSetParams(node: Opaque, params: *const HostNodeParams) -> CuResult {
    trace::call("cuGraphHostNodeSetParams", || {
        status(host_node(params).and_then(|host| {
            with_node(node, |state, graph, id| {
                let entry = state.graphs.graphs.get_mut(&graph).unwrap().entry_mut(id)?;
                if !matches!(entry.node, Node::Host(_)) {
                    return Err(InvalidValue);
                }
                entry.node = Node::Host(host);
                Ok(())
            })
        }))
    })
}

fn get_event(node: Opaque, out: *mut Opaque, record: bool) -> Result<(), CuResult> {
    if out.is_null() {
        return Err(InvalidValue);
    }
    with_node(node, |state, graph, id| {
        let event = match (&state.graphs.graphs[&graph].entry(id)?.node, record) {
            (Node::EventRecord(event), true) | (Node::EventWait(event), false) => *event,
            _ => return Err(InvalidValue),
        };
        unsafe { out.write(node_ptr(event)) };
        Ok(())
    })
}

fn set_event(node: Opaque, event: Opaque, record: bool) -> Result<(), CuResult> {
    if event.is_null() {
        return Err(InvalidValue);
    }
    with_node(node, |state, graph, id| {
        let entry = state.graphs.graphs.get_mut(&graph).unwrap().entry_mut(id)?;
        entry.node = match (&entry.node, record) {
            (Node::EventRecord(_), true) => Node::EventRecord(event.addr()),
            (Node::EventWait(_), false) => Node::EventWait(event.addr()),
            _ => return Err(InvalidValue),
        };
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphEventRecordNodeGetEvent(node: Opaque, out: *mut Opaque) -> CuResult {
    trace::call("cuGraphEventRecordNodeGetEvent", || status(get_event(node, out, true)))
}

#[unsafe(no_mangle)]
pub extern "system" fn cuGraphEventRecordNodeSetEvent(node: Opaque, event: Opaque) -> CuResult {
    trace::call("cuGraphEventRecordNodeSetEvent", || status(set_event(node, event, true)))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphEventWaitNodeGetEvent(node: Opaque, out: *mut Opaque) -> CuResult {
    trace::call("cuGraphEventWaitNodeGetEvent", || status(get_event(node, out, false)))
}

#[unsafe(no_mangle)]
pub extern "system" fn cuGraphEventWaitNodeSetEvent(node: Opaque, event: Opaque) -> CuResult {
    trace::call("cuGraphEventWaitNodeSetEvent", || status(set_event(node, event, false)))
}

// ---------------------------------------------------------------------------
// Executable graph updates (the node is a handle of the instantiated graph)

fn exec_update_node(
    exec: Opaque,
    node: Opaque,
    make: impl FnOnce(&mut State, usize, &Node) -> Result<Node, CuResult>,
) -> Result<(), CuResult> {
    state(|state| {
        let (owner, old) = {
            let exec = state.graphs.executable.get(&exec.addr()).ok_or(InvalidValue)?;
            (exec.context, exec.node(node.addr())?.clone())
        };
        let new = make(state, owner, &old)?;
        state
            .graphs
            .executable
            .get_mut(&exec.addr())
            .ok_or(InvalidValue)?
            .replace(node.addr(), new)
    })
}

fn exec_kernel(exec: Opaque, node: Opaque, params: *const KernelNodeParams, v2: bool) -> Result<(), CuResult> {
    let params = unsafe { load_kernel_params(params, v2)? };
    exec_update_node(exec, node, |state, owner, old| {
        if !matches!(old, Node::Kernel(_)) {
            return Err(InvalidValue);
        }
        kernel_node(state, owner, &params, v2).map(Node::Kernel)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphExecKernelNodeSetParams(exec: Opaque, node: Opaque, params: *const KernelNodeParams) -> CuResult {
    trace::call("cuGraphExecKernelNodeSetParams", || status(exec_kernel(exec, node, params, false)))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphExecKernelNodeSetParams_v2(exec: Opaque, node: Opaque, params: *const KernelNodeParams) -> CuResult {
    trace::call("cuGraphExecKernelNodeSetParams_v2", || status(exec_kernel(exec, node, params, true)))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphExecMemcpyNodeSetParams(
    exec: Opaque,
    node: Opaque,
    params: *const MemcpyNodeParams,
    _context: Opaque,
) -> CuResult {
    trace::call("cuGraphExecMemcpyNodeSetParams", || {
        if params.is_null() {
            return InvalidValue;
        }
        let params = unsafe { params.read_unaligned() };
        status(exec_update_node(exec, node, |state, owner, old| match old {
            Node::Copy(old) => {
                let copy = copy_node(state, owner, &params)?;
                // The transfer direction of an instantiated node cannot change.
                if copy.kind == old.kind {
                    Ok(Node::Copy(copy))
                } else {
                    Err(InvalidValue)
                }
            }
            _ => Err(InvalidValue),
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphExecMemsetNodeSetParams(
    exec: Opaque,
    node: Opaque,
    params: *const MemsetNodeParams,
    _context: Opaque,
) -> CuResult {
    trace::call("cuGraphExecMemsetNodeSetParams", || {
        if params.is_null() {
            return InvalidValue;
        }
        let params = unsafe { params.read_unaligned() };
        status(exec_update_node(exec, node, |state, owner, old| match old {
            Node::Fill(_) => fill_node(state, owner, &params).map(Node::Fill),
            _ => Err(InvalidValue),
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphExecHostNodeSetParams(exec: Opaque, node: Opaque, params: *const HostNodeParams) -> CuResult {
    trace::call("cuGraphExecHostNodeSetParams", || {
        status(host_node(params).and_then(|host| {
            exec_update_node(exec, node, |_, _, old| match old {
                Node::Host(_) => Ok(Node::Host(host)),
                _ => Err(InvalidValue),
            })
        }))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuGraphExecEventRecordNodeSetEvent(exec: Opaque, node: Opaque, event: Opaque) -> CuResult {
    trace::call("cuGraphExecEventRecordNodeSetEvent", || {
        if event.is_null() {
            return InvalidValue;
        }
        status(exec_update_node(exec, node, |_, _, old| match old {
            Node::EventRecord(_) => Ok(Node::EventRecord(event.addr())),
            _ => Err(InvalidValue),
        }))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuGraphExecEventWaitNodeSetEvent(exec: Opaque, node: Opaque, event: Opaque) -> CuResult {
    trace::call("cuGraphExecEventWaitNodeSetEvent", || {
        if event.is_null() {
            return InvalidValue;
        }
        status(exec_update_node(exec, node, |_, _, old| match old {
            Node::EventWait(_) => Ok(Node::EventWait(event.addr())),
            _ => Err(InvalidValue),
        }))
    })
}

// ---------------------------------------------------------------------------
// Capture control and host functions

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuStreamBeginCaptureToGraph(
    stream: Opaque,
    graph: Opaque,
    dependencies: *const Opaque,
    edges: *const EdgeData,
    count: usize,
    mode: c_int,
) -> CuResult {
    trace::call("cuStreamBeginCaptureToGraph", || {
        let dependencies = unsafe { default_edges(edges, count).and_then(|()| handles(dependencies, count)) };
        match dependencies {
            Ok(dependencies) => begin_capture(stream, mode, Some((graph.addr(), dependencies))),
            Err(error) => error,
        }
    })
}

fn update_capture_dependencies(
    stream: Opaque,
    dependencies: *const Opaque,
    edges: *const EdgeData,
    count: usize,
    flags: u32,
) -> Result<(), CuResult> {
    if flags > 1 {
        return Err(InvalidValue);
    }
    unsafe { default_edges(edges, count)? };
    let nodes = unsafe { handles(dependencies, count)? };
    state(|state| {
        let owner = current().ok_or(InvalidContext)?;
        let context = state.contexts.get_mut(&owner).ok_or(ContextIsDestroyed)?;
        context.check_live()?;
        let key = context.streams.launch_key(stream)?;
        let capture = context
            .streams
            .capture(key)?
            .cloned()
            .ok_or(IllegalState)?;
        if capture.is_invalidated() {
            return Err(StreamCaptureInvalidated);
        }
        let known = |node: usize| {
            capture.contains(node)
                || state
                    .graphs
                    .graphs
                    .get(&capture.graph())
                    .is_some_and(|dag| dag.find(node).is_some())
        };
        if nodes.iter().any(|&node| !known(node)) {
            return Err(InvalidValue);
        }
        capture.update_dependencies(key, &nodes, flags == 1);
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuStreamUpdateCaptureDependencies(
    stream: Opaque,
    dependencies: *const Opaque,
    count: usize,
    flags: u32,
) -> CuResult {
    trace::call("cuStreamUpdateCaptureDependencies", || {
        status(update_capture_dependencies(stream, dependencies, std::ptr::null(), count, flags))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuStreamUpdateCaptureDependencies_v2(
    stream: Opaque,
    dependencies: *const Opaque,
    edges: *const EdgeData,
    count: usize,
    flags: u32,
) -> CuResult {
    trace::call("cuStreamUpdateCaptureDependencies_v2", || {
        status(update_capture_dependencies(stream, dependencies, edges, count, flags))
    })
}

/// Run `function(data)` once the work already issued on `stream` is done. In a
/// capture it becomes a host node. Otherwise the call waits for the stream and
/// runs the function on the calling thread, outside the runtime lock.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuLaunchHostFunc(stream: Opaque, function: Opaque, data: Opaque) -> CuResult {
    trace::call("cuLaunchHostFunc", || {
        if function.is_null() {
            return InvalidValue;
        }
        let host = HostFn {
            function: function.addr(),
            data: data.addr(),
        };
        let captured = with_context(|_, context| {
            let key = context.streams.launch_key(stream)?;
            match context.streams.capture(key)? {
                Some(capture) => capture.push(key, Node::Host(host)).map(|()| true),
                None => Ok(false),
            }
        });
        match captured {
            Ok(true) => Success,
            Ok(false) => call_host(stream, host),
            Err(error) => error,
        }
    })
}
