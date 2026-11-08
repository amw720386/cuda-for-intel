//! cuDNN v8 backend API (`cudnnBackend*`) as a state machine over generic
//! attribute storage. Only the single-convolution graph (forward, backward
//! data, backward filter) is accepted; every other graph pattern finalizes
//! with a NOT_SUPPORTED status so cudnn_frontend reports "no engine".
use crate::abi::{self, Status};
use crate::desc::{self, ConvDesc, ConvPlan, Dir, FilterDesc, TensorDesc};
use std::collections::HashMap;

/// OPERATIONGRAPH has no deviceprop attribute in v8/v9 headers we know; kept for forward compatibility.
const A_DEVICEPROP_ANY: i32 = -1;
const A_PLAN_DEVICEPROP: i32 = 407;

#[derive(Clone, Debug, Default)]
pub struct Attr {
    pub ty: i32,
    pub count: i64,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct BDesc {
    pub ty: i32,
    pub finalized: bool,
    pub attrs: HashMap<i32, Attr>,
    /// Filled at finalize for operation graphs (engine count) and execution plans.
    pub engine_count: i64,
    pub exec: Option<PlanExec>,
}

/// What an execution plan runs and which variant-pack UIDs feed it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanExec {
    pub plan: ConvPlan,
    /// UIDs of the kernel operands (a, b, out); see `ptxgen::generate_conv`.
    pub uids: [i64; 3],
    pub alpha: f32,
    pub beta: f32,
}

pub type BPtr = *mut BDesc;

pub fn create(ty: i32) -> Result<BPtr, Status> {
    if !(0..=abi::BD_MAX).contains(&ty) {
        return Err(abi::BAD_PARAM);
    }
    Ok(Box::into_raw(Box::new(BDesc { ty, finalized: false, attrs: HashMap::new(), engine_count: 0, exec: None })))
}

/// # Safety
/// `p` must come from [`create`] and not be used afterwards.
pub unsafe fn destroy(p: BPtr) {
    drop(unsafe { Box::from_raw(p) });
}

fn elem_size(ty: i32) -> usize {
    match ty {
        abi::AT_HANDLE | abi::AT_VOID_PTR | abi::AT_BACKEND_DESCRIPTOR => 8,
        abi::AT_INT64 | abi::AT_DOUBLE => 8,
        abi::AT_BOOLEAN => 1,
        24 => 1, // char
        26 => 16, // fraction {int64,int64}
        _ => 4,
    }
}

/// Attribute types accepted per attribute name; `None` = unknown name (stored opaquely).
fn expected_types(name: i32) -> Option<&'static [i32]> {
    use abi::*;
    Some(match name {
        A_CONV_COMP_TYPE | A_TENSOR_DTYPE => &[AT_DATA_TYPE],
        A_CONV_MODE => &[AT_CONVOLUTION_MODE],
        A_CONV_DILATIONS | A_CONV_FILTER_STRIDES | A_CONV_POST_PADDINGS | A_CONV_PRE_PADDINGS | A_CONV_SPATIAL_DIMS
        | A_TENSOR_DIMS | A_TENSOR_STRIDES | A_TENSOR_UID | A_TENSOR_ALIGN | A_ENGINE_INDEX | A_VP_UIDS => &[AT_INT64],
        A_TENSOR_VIRTUAL => &[AT_BOOLEAN],
        A_FWD_ALPHA | A_FWD_BETA | A_BWDD_ALPHA | A_BWDD_BETA | A_BWDF_ALPHA | A_BWDF_BETA => &[AT_FLOAT, AT_DOUBLE],
        A_FWD_CONV | A_FWD_W | A_FWD_X | A_FWD_Y | A_BWDD_CONV | A_BWDD_W | A_BWDD_DX | A_BWDD_DY | A_BWDF_CONV
        | A_BWDF_DW | A_BWDF_X | A_BWDF_DY | A_GRAPH_OPS | A_HEUR_GRAPH | A_HEUR_RESULTS | A_CFG_ENGINE | A_CFG_KNOBS
        | A_PLAN_CFG | A_ENGINE_GRAPH => &[AT_BACKEND_DESCRIPTOR],
        A_GRAPH_HANDLE | A_PLAN_HANDLE => &[AT_HANDLE],
        A_VP_PTRS | A_VP_WORKSPACE => &[AT_VOID_PTR],
        _ => return None,
    })
}

/// # Safety
/// `data` must point at `count` elements of type `ty` (or be null when `count == 0`).
pub unsafe fn set_attribute(d: &mut BDesc, name: i32, ty: i32, count: i64, data: *const u8) -> Status {
    if d.finalized {
        return abi::BAD_PARAM;
    }
    if count < 0 || (count > 0 && data.is_null()) {
        return abi::BAD_PARAM;
    }
    if let Some(types) = expected_types(name) {
        if !types.contains(&ty) {
            return abi::BAD_PARAM_ATTRIBUTE_TYPE;
        }
        // single-valued attributes take exactly one element
        let single = !matches!(
            name,
            abi::A_CONV_DILATIONS | abi::A_CONV_FILTER_STRIDES | abi::A_CONV_POST_PADDINGS | abi::A_CONV_PRE_PADDINGS
                | abi::A_TENSOR_DIMS | abi::A_TENSOR_STRIDES | abi::A_GRAPH_OPS | abi::A_HEUR_RESULTS | abi::A_CFG_KNOBS
                | abi::A_VP_UIDS | abi::A_VP_PTRS
        );
        if single && count != 1 {
            return abi::BAD_PARAM;
        }
    }
    let n = count as usize * elem_size(ty);
    let bytes = if n == 0 { Vec::new() } else { unsafe { std::slice::from_raw_parts(data, n) }.to_vec() };
    d.attrs.insert(name, Attr { ty, count, bytes });
    abi::SUCCESS
}

fn i64s(a: &Attr) -> Vec<i64> {
    a.bytes.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()
}
fn ptrs(a: &Attr) -> Vec<usize> {
    a.bytes.chunks_exact(8).map(|c| usize::from_le_bytes(c.try_into().unwrap())).collect()
}
fn i32_of(a: &Attr) -> Option<i32> {
    a.bytes.get(..4).map(|b| i32::from_le_bytes(b.try_into().unwrap()))
}
fn scalar_f32(a: &Attr) -> Option<f32> {
    match a.ty {
        abi::AT_FLOAT => a.bytes.get(..4).map(|b| f32::from_le_bytes(b.try_into().unwrap())),
        abi::AT_DOUBLE => a.bytes.get(..8).map(|b| f64::from_le_bytes(b.try_into().unwrap()) as f32),
        _ => None,
    }
}

/// Follow a BACKEND_DESCRIPTOR attribute to its (finalized) target.
unsafe fn target<'a>(d: &BDesc, name: i32, ty: i32) -> Result<&'a BDesc, Status> {
    let a = d.attrs.get(&name).ok_or(abi::BAD_PARAM)?;
    let p = *ptrs(a).first().ok_or(abi::BAD_PARAM)? as BPtr;
    if p.is_null() {
        return Err(abi::BAD_PARAM);
    }
    let t = unsafe { &*p };
    if t.ty != ty {
        return Err(abi::BAD_PARAM_DESCRIPTOR_TYPE);
    }
    if !t.finalized {
        return Err(abi::BAD_PARAM_NOT_FINALIZED);
    }
    Ok(t)
}

fn tensor_of(t: &BDesc) -> Result<(TensorDesc, i64), Status> {
    let dims: Vec<i32> = i64s(t.attrs.get(&abi::A_TENSOR_DIMS).ok_or(abi::BAD_PARAM)?).iter().map(|v| *v as i32).collect();
    let strides: Vec<i32> = i64s(t.attrs.get(&abi::A_TENSOR_STRIDES).ok_or(abi::BAD_PARAM)?).iter().map(|v| *v as i32).collect();
    let dtype = i32_of(t.attrs.get(&abi::A_TENSOR_DTYPE).ok_or(abi::BAD_PARAM)?).ok_or(abi::BAD_PARAM)?;
    let uid = t.attrs.get(&abi::A_TENSOR_UID).map(|a| i64s(a)[0]).unwrap_or(-1);
    let mut td = TensorDesc::default();
    let r = td.set_nd(dtype, &dims, &strides);
    if r != abi::SUCCESS {
        return Err(r);
    }
    Ok((td, uid))
}

fn conv_of(c: &BDesc) -> Result<ConvDesc, Status> {
    let get = |n| c.attrs.get(&n).map(i64s).ok_or(abi::BAD_PARAM);
    let to32 = |v: Vec<i64>| v.into_iter().map(|x| x as i32).collect::<Vec<_>>();
    let pre = to32(get(abi::A_CONV_PRE_PADDINGS)?);
    let post = to32(get(abi::A_CONV_POST_PADDINGS)?);
    if pre != post {
        return Err(abi::NOT_SUPPORTED_SHAPE); // asymmetric padding has no kernel
    }
    let mode = i32_of(c.attrs.get(&abi::A_CONV_MODE).ok_or(abi::BAD_PARAM)?).ok_or(abi::BAD_PARAM)?;
    let comp = i32_of(c.attrs.get(&abi::A_CONV_COMP_TYPE).ok_or(abi::BAD_PARAM)?).ok_or(abi::BAD_PARAM)?;
    let mut cd = ConvDesc::default();
    let r = cd.set_nd(&pre, &to32(get(abi::A_CONV_FILTER_STRIDES)?), &to32(get(abi::A_CONV_DILATIONS)?), mode, comp);
    if r != abi::SUCCESS {
        return Err(r);
    }
    Ok(cd)
}

/// Build the executable plan of a one-operation convolution graph.
unsafe fn plan_of_op(op: &BDesc) -> Result<PlanExec, Status> {
    use abi::*;
    let (conv_n, a_n, b_n, o_n, al_n, be_n, dir) = match op.ty {
        BD_OP_CONV_FWD => (A_FWD_CONV, A_FWD_X, A_FWD_W, A_FWD_Y, A_FWD_ALPHA, A_FWD_BETA, Dir::Fwd),
        BD_OP_CONV_BWD_DATA => (A_BWDD_CONV, A_BWDD_DY, A_BWDD_W, A_BWDD_DX, A_BWDD_ALPHA, A_BWDD_BETA, Dir::BwdData),
        BD_OP_CONV_BWD_FILTER => (A_BWDF_CONV, A_BWDF_X, A_BWDF_DY, A_BWDF_DW, A_BWDF_ALPHA, A_BWDF_BETA, Dir::BwdFilter),
        _ => return Err(NOT_SUPPORTED_GRAPH_PATTERN),
    };
    let conv = conv_of(unsafe { target(op, conv_n, BD_CONVOLUTION)? })?;
    let (ta, ua) = tensor_of(unsafe { target(op, a_n, BD_TENSOR)? })?;
    let (tb, ub) = tensor_of(unsafe { target(op, b_n, BD_TENSOR)? })?;
    let (to, uo) = tensor_of(unsafe { target(op, o_n, BD_TENSOR)? })?;
    let alpha = op.attrs.get(&al_n).and_then(scalar_f32).unwrap_or(1.0);
    let beta = op.attrs.get(&be_n).and_then(scalar_f32).unwrap_or(0.0);
    let filt = |t: &TensorDesc| {
        let mut f = FilterDesc::default();
        let r = f.set_nd(t.dtype, abi::NCHW, &t.dims);
        f.strides_override = Some(t.strides.clone());
        (r, f)
    };
    // Backend tensors carry logical NCHW dims; groups follow from the filter's channel count.
    let (x, w, y) = match dir {
        Dir::Fwd => (ta, filt(&tb), to),
        Dir::BwdData => (to, filt(&tb), ta),
        Dir::BwdFilter => (ta, filt(&to), tb),
    };
    if w.0 != abi::SUCCESS {
        return Err(w.0);
    }
    let mut conv = conv;
    if w.1.dims.len() == x.dims.len() && w.1.dims[1] > 0 && x.dims[1] % w.1.dims[1] == 0 {
        conv.groups = x.dims[1] / w.1.dims[1];
    }
    let plan = desc::plan_conv(dir, &x, &w.1, &conv, &y).map_err(|s| match s {
        NOT_SUPPORTED => NOT_SUPPORTED_GRAPH_PATTERN,
        s => s,
    })?;
    Ok(PlanExec { plan, uids: [ua, ub, uo], alpha, beta })
}

/// # Safety
/// Attribute pointers stored in `d` must reference live backend descriptors.
pub unsafe fn finalize(d: &mut BDesc) -> Status {
    use abi::*;
    if d.finalized {
        return SUCCESS;
    }
    let st = (|| -> Result<(), Status> {
        match d.ty {
            BD_TENSOR => {
                let virt = d.attrs.get(&A_TENSOR_VIRTUAL).map(|a| a.bytes.first() == Some(&1)).unwrap_or(false);
                for n in [A_TENSOR_DIMS, A_TENSOR_STRIDES, A_TENSOR_DTYPE] {
                    d.attrs.get(&n).ok_or(BAD_PARAM)?;
                }
                if !virt && !d.attrs.contains_key(&A_TENSOR_UID) {
                    return Err(BAD_PARAM);
                }
                if d.attrs[&A_TENSOR_DIMS].count != d.attrs[&A_TENSOR_STRIDES].count || d.attrs[&A_TENSOR_DIMS].count == 0 {
                    return Err(BAD_PARAM);
                }
            }
            BD_CONVOLUTION => {
                conv_of(d)?;
                let sd = d.attrs.get(&A_CONV_SPATIAL_DIMS).map(|a| i64s(a)[0]).ok_or(BAD_PARAM)?;
                if d.attrs[&A_CONV_FILTER_STRIDES].count != sd {
                    return Err(BAD_PARAM);
                }
            }
            BD_OP_CONV_FWD | BD_OP_CONV_BWD_DATA | BD_OP_CONV_BWD_FILTER => {
                // Operand and descriptor attributes must all be present and finalized.
                unsafe { plan_of_op(d) }.map(|_| ())?;
            }
            BD_OPERATIONGRAPH => {
                if !d.attrs.contains_key(&A_GRAPH_HANDLE) && !d.attrs.contains_key(&A_DEVICEPROP_ANY) {
                    return Err(BAD_PARAM);
                }
                let ops = d.attrs.get(&A_GRAPH_OPS).ok_or(BAD_PARAM)?;
                if ops.count == 0 {
                    return Err(BAD_PARAM);
                }
                for p in ptrs(ops) {
                    let op = unsafe { (p as BPtr).as_ref() }.ok_or(BAD_PARAM)?;
                    if !op.finalized {
                        return Err(BAD_PARAM_NOT_FINALIZED);
                    }
                }
                // A graph is runnable only when it is exactly one supported convolution.
                // Everything derived is copied by value: cudnn_frontend destroys the op, tensor and
                // convolution descriptors (and the graph itself) while plans built from the
                // configs are still alive, so later finalizes must not dereference them.
                let exec = if ops.count == 1 { unsafe { plan_of_op(&*(ptrs(ops)[0] as BPtr)) }.ok() } else { None };
                d.engine_count = exec.is_some() as i64;
                d.exec = exec;
            }
            BD_ENGINEHEUR => {
                let g = unsafe { target(d, A_HEUR_GRAPH, BD_OPERATIONGRAPH)? };
                d.attrs.get(&A_HEUR_MODE).ok_or(BAD_PARAM)?;
                if g.engine_count == 0 {
                    return Err(NOT_SUPPORTED_GRAPH_PATTERN);
                }
                d.engine_count = g.engine_count;
                d.exec = g.exec;
            }
            BD_ENGINE => {
                let g = unsafe { target(d, A_ENGINE_GRAPH, BD_OPERATIONGRAPH)? };
                let idx = d.attrs.get(&A_ENGINE_INDEX).map(|a| i64s(a)[0]).ok_or(BAD_PARAM)?;
                if idx < 0 || idx >= g.engine_count {
                    return Err(if g.engine_count == 0 { NOT_SUPPORTED_GRAPH_PATTERN } else { BAD_PARAM_OUT_OF_BOUND });
                }
                d.exec = g.exec;
                d.engine_count = g.engine_count;
            }
            BD_ENGINECFG => {
                let eng = unsafe { target(d, A_CFG_ENGINE, BD_ENGINE)? };
                d.exec = eng.exec;
                // keep a private copy of the engine: the frontend may free its own
                let own = Box::into_raw(Box::new(eng.clone()));
                if let Some(a) = d.attrs.get_mut(&A_CFG_ENGINE) {
                    a.bytes = (own as usize).to_le_bytes().to_vec();
                }
            }
            BD_EXECUTION_PLAN => {
                // Newer frontends may set neither HANDLE nor DEVICEPROP before finalize.
                let cfg = unsafe { target(d, A_PLAN_CFG, BD_ENGINECFG)? };
                d.exec = Some(cfg.exec.ok_or(NOT_SUPPORTED_GRAPH_PATTERN)?);
            }
            BD_VARIANT_PACK => {
                let u = d.attrs.get(&A_VP_UIDS).ok_or(BAD_PARAM)?;
                let p = d.attrs.get(&A_VP_PTRS).ok_or(BAD_PARAM)?;
                if u.count != p.count {
                    return Err(BAD_PARAM);
                }
            }
            _ => return Err(NOT_SUPPORTED_GRAPH_PATTERN),
        }
        Ok(())
    })();
    match st {
        Ok(()) => {
            d.finalized = true;
            SUCCESS
        }
        Err(s) => s,
    }
}

/// # Safety
/// `out` must have room for `requested * elem_size` bytes (or be null with `requested == 0`);
/// descriptor handles in `out` (ENGINEHEUR_RESULTS) must be live ENGINECFG descriptors.
pub unsafe fn get_attribute(d: &BDesc, name: i32, ty: i32, requested: i64, count: *mut i64, out: *mut u8) -> Status {
    use abi::*;
    // cudnn_frontend queries counts with a null `out` (and requested 0 or the max count).
    if requested < 0 {
        return BAD_PARAM;
    }
    let out_ok = !out.is_null();
    let put_count = |n: i64| {
        if !count.is_null() {
            unsafe { *count = n };
        }
    };
    // Derived (computed) attributes first.
    match name {
        A_GRAPH_ENGINE_COUNT if d.ty == BD_OPERATIONGRAPH => {
            if !d.finalized {
                return BAD_PARAM_NOT_FINALIZED;
            }
            if ty != AT_INT64 {
                return BAD_PARAM_ATTRIBUTE_TYPE;
            }
            put_count(1);
            if requested > 0 && out_ok {
                unsafe { (out as *mut i64).write_unaligned(d.engine_count) };
            }
            return SUCCESS;
        }
        A_HEUR_RESULTS if d.ty == BD_ENGINEHEUR => {
            if !d.finalized {
                return BAD_PARAM_NOT_FINALIZED;
            }
            // requested == 0 (or no output array) is the count query.
            if requested == 0 || !out_ok {
                put_count(d.engine_count);
                return SUCCESS;
            }
            let n = d.engine_count.min(requested);
            for i in 0..n as usize {
                let cfg = unsafe { *(out as *const BPtr).add(i) };
                let Some(cfg) = (unsafe { cfg.as_mut() }) else { return BAD_PARAM };
                if cfg.ty != BD_ENGINECFG {
                    return BAD_PARAM_DESCRIPTOR_TYPE;
                }
                // Materialize engine `i` (by value; no graph pointer is kept).
                let mut eng = BDesc { ty: BD_ENGINE, finalized: true, attrs: HashMap::new(), engine_count: d.engine_count, exec: d.exec };
                eng.attrs.insert(A_ENGINE_INDEX, Attr { ty: AT_INT64, count: 1, bytes: (i as i64).to_le_bytes().to_vec() });
                let ep = (Box::into_raw(Box::new(eng)) as usize).to_le_bytes();
                *cfg = BDesc { ty: BD_ENGINECFG, finalized: true, attrs: HashMap::new(), engine_count: d.engine_count, exec: d.exec };
                cfg.attrs.insert(A_CFG_ENGINE, Attr { ty: AT_BACKEND_DESCRIPTOR, count: 1, bytes: ep.to_vec() });
            }
            put_count(n);
            return SUCCESS;
        }
        A_CFG_WORKSPACE | A_PLAN_WORKSPACE => {
            if !d.finalized {
                return BAD_PARAM_NOT_FINALIZED;
            }
            put_count(1);
            if requested > 0 && out_ok {
                unsafe { (out as *mut i64).write_unaligned(0) };
            }
            return SUCCESS;
        }
        _ => {}
    }
    // Stored attributes.
    let Some(a) = d.attrs.get(&name) else {
        // engine/plan note lists the frontend probes: report "none"
        put_count(0);
        return if matches!(d.ty, BD_ENGINE | BD_ENGINECFG | BD_EXECUTION_PLAN | BD_ENGINEHEUR) { SUCCESS } else { BAD_PARAM };
    };
    if a.ty != ty {
        return BAD_PARAM_ATTRIBUTE_TYPE;
    }
    put_count(a.count);
    let take = requested.min(a.count) as usize;
    if take == 0 || !out_ok {
        return SUCCESS;
    }
    if ty == AT_BACKEND_DESCRIPTOR {
        // The caller pre-creates the destination descriptors; real cuDNN fills them in place
        // (cudnn_frontend relies on this for ENGINECFG_ENGINE and knob lists).
        for (i, src) in ptrs(a).into_iter().take(take).enumerate() {
            let dst = unsafe { *(out as *const BPtr).add(i) };
            match (unsafe { dst.as_mut() }, unsafe { (src as BPtr).as_ref() }) {
                (Some(dst), Some(src)) => {
                    if dst.ty != src.ty {
                        return BAD_PARAM_DESCRIPTOR_TYPE;
                    }
                    *dst = src.clone();
                }
                (None, Some(_)) => unsafe { (out as *mut usize).add(i).write_unaligned(src) },
                _ => return BAD_PARAM,
            }
        }
        return SUCCESS;
    }
    unsafe { std::ptr::copy_nonoverlapping(a.bytes.as_ptr(), out, take * elem_size(ty)) };
    SUCCESS
}

/// Look up the device pointer for each operand UID of a plan in a variant pack.
pub fn resolve_pointers(vp: &BDesc, uids: &[i64; 3]) -> Result<[u64; 3], Status> {
    let u = i64s(vp.attrs.get(&abi::A_VP_UIDS).ok_or(abi::BAD_PARAM)?);
    let p = ptrs(vp.attrs.get(&abi::A_VP_PTRS).ok_or(abi::BAD_PARAM)?);
    let mut out = [0u64; 3];
    for (i, uid) in uids.iter().enumerate() {
        let pos = u.iter().position(|x| x == uid).ok_or(abi::BAD_PARAM)?;
        out[i] = p[pos] as u64;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use abi::*;

    unsafe fn set(d: BPtr, name: i32, ty: i32, vals: &[u8], count: i64) -> Status {
        unsafe { set_attribute(&mut *d, name, ty, count, vals.as_ptr()) }
    }
    fn b64(v: &[i64]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }
    fn desc_ptr(p: BPtr) -> Vec<u8> {
        (p as usize).to_le_bytes().to_vec()
    }

    unsafe fn tensor(uid: i64, dims: &[i64], strides: &[i64], dt: i32) -> BPtr {
        let t = create(BD_TENSOR).unwrap();
        unsafe {
            assert_eq!(set(t, A_TENSOR_DIMS, AT_INT64, &b64(dims), dims.len() as i64), SUCCESS);
            assert_eq!(set(t, A_TENSOR_STRIDES, AT_INT64, &b64(strides), strides.len() as i64), SUCCESS);
            assert_eq!(set(t, A_TENSOR_DTYPE, AT_DATA_TYPE, &dt.to_le_bytes(), 1), SUCCESS);
            assert_eq!(set(t, A_TENSOR_UID, AT_INT64, &b64(&[uid]), 1), SUCCESS);
            assert_eq!(set(t, A_TENSOR_ALIGN, AT_INT64, &b64(&[16]), 1), SUCCESS);
            assert_eq!(finalize(&mut *t), SUCCESS);
        }
        t
    }

    unsafe fn conv_desc() -> BPtr {
        let c = create(BD_CONVOLUTION).unwrap();
        unsafe {
            set(c, A_CONV_SPATIAL_DIMS, AT_INT64, &b64(&[2]), 1);
            set(c, A_CONV_DILATIONS, AT_INT64, &b64(&[1, 1]), 2);
            set(c, A_CONV_FILTER_STRIDES, AT_INT64, &b64(&[1, 1]), 2);
            set(c, A_CONV_PRE_PADDINGS, AT_INT64, &b64(&[1, 1]), 2);
            set(c, A_CONV_POST_PADDINGS, AT_INT64, &b64(&[1, 1]), 2);
            set(c, A_CONV_COMP_TYPE, AT_DATA_TYPE, &DT_FLOAT.to_le_bytes(), 1);
            set(c, A_CONV_MODE, AT_CONVOLUTION_MODE, &CROSS_CORRELATION.to_le_bytes(), 1);
            assert_eq!(finalize(&mut *c), SUCCESS);
        }
        c
    }

    /// Build graph -> heuristics -> config -> plan -> variant pack like cudnn_frontend does.
    #[test]
    fn frontend_style_forward_conv_flow() {
        unsafe {
            let x = tensor(1, &[2, 4, 8, 8], &[256, 64, 8, 1], DT_FLOAT);
            let w = tensor(2, &[6, 4, 3, 3], &[36, 9, 3, 1], DT_FLOAT);
            let y = tensor(3, &[2, 6, 8, 8], &[384, 64, 8, 1], DT_FLOAT);
            let c = conv_desc();
            let op = create(BD_OP_CONV_FWD).unwrap();
            assert_eq!(set(op, A_FWD_X, AT_BACKEND_DESCRIPTOR, &desc_ptr(x), 1), SUCCESS);
            assert_eq!(set(op, A_FWD_W, AT_BACKEND_DESCRIPTOR, &desc_ptr(w), 1), SUCCESS);
            assert_eq!(set(op, A_FWD_Y, AT_BACKEND_DESCRIPTOR, &desc_ptr(y), 1), SUCCESS);
            assert_eq!(set(op, A_FWD_CONV, AT_BACKEND_DESCRIPTOR, &desc_ptr(c), 1), SUCCESS);
            assert_eq!(set(op, A_FWD_ALPHA, AT_FLOAT, &2f32.to_le_bytes(), 1), SUCCESS);
            assert_eq!(set(op, A_FWD_BETA, AT_FLOAT, &0f32.to_le_bytes(), 1), SUCCESS);
            assert_eq!(finalize(&mut *op), SUCCESS);
            let g = create(BD_OPERATIONGRAPH).unwrap();
            assert_eq!(set(g, A_GRAPH_OPS, AT_BACKEND_DESCRIPTOR, &desc_ptr(op), 1), SUCCESS);
            assert_eq!(set(g, A_GRAPH_HANDLE, AT_HANDLE, &1usize.to_le_bytes(), 1), SUCCESS);
            assert_eq!(finalize(&mut *g), SUCCESS);
            let mut cnt = [0u8; 8];
            let mut n = 0i64;
            assert_eq!(get_attribute(&*g, A_GRAPH_ENGINE_COUNT, AT_INT64, 1, &mut n, cnt.as_mut_ptr()), SUCCESS);
            assert_eq!(i64::from_le_bytes(cnt), 1);
            // heuristics
            let h = create(BD_ENGINEHEUR).unwrap();
            assert_eq!(set(h, A_HEUR_GRAPH, AT_BACKEND_DESCRIPTOR, &desc_ptr(g), 1), SUCCESS);
            assert_eq!(set(h, A_HEUR_MODE, 8, &0i32.to_le_bytes(), 1), SUCCESS);
            assert_eq!(finalize(&mut *h), SUCCESS);
            let cfgs: Vec<BPtr> = (0..3).map(|_| create(BD_ENGINECFG).unwrap()).collect();
            assert_eq!(get_attribute(&*h, A_HEUR_RESULTS, AT_BACKEND_DESCRIPTOR, 3, &mut n, cfgs.as_ptr() as *mut u8), SUCCESS);
            assert_eq!(n, 1);
            assert!((*cfgs[0]).finalized);
            // plan
            let p = create(BD_EXECUTION_PLAN).unwrap();
            assert_eq!(set(p, A_PLAN_HANDLE, AT_HANDLE, &1usize.to_le_bytes(), 1), SUCCESS);
            assert_eq!(set(p, A_PLAN_CFG, AT_BACKEND_DESCRIPTOR, &desc_ptr(cfgs[0]), 1), SUCCESS);
            assert_eq!(finalize(&mut *p), SUCCESS);
            let e = (*p).exec.unwrap();
            assert_eq!((e.plan.geom.k, e.plan.geom.groups, e.alpha, e.uids), (6, 1, 2.0, [1, 2, 3]));
            assert_eq!(get_attribute(&*p, A_PLAN_WORKSPACE, AT_INT64, 1, &mut n, cnt.as_mut_ptr()), SUCCESS);
            assert_eq!(i64::from_le_bytes(cnt), 0);
            // variant pack
            let vp = create(BD_VARIANT_PACK).unwrap();
            let ptrs_b: Vec<u8> = [0x1000usize, 0x2000, 0x3000].iter().flat_map(|v| v.to_le_bytes()).collect();
            assert_eq!(set(vp, A_VP_UIDS, AT_INT64, &b64(&[3, 1, 2]), 3), SUCCESS);
            assert_eq!(set(vp, A_VP_PTRS, AT_VOID_PTR, &ptrs_b, 3), SUCCESS);
            assert_eq!(finalize(&mut *vp), SUCCESS);
            assert_eq!(resolve_pointers(&*vp, &e.uids).unwrap(), [0x2000, 0x3000, 0x1000]);
            assert_eq!(resolve_pointers(&*vp, &[9, 1, 2]).unwrap_err(), BAD_PARAM);
        }
    }

    #[test]
    fn rejects_bad_attribute_types_and_unfinalized_use() {
        unsafe {
            let t = create(BD_TENSOR).unwrap();
            assert_eq!(set(t, A_TENSOR_DIMS, AT_FLOAT, &[0; 16], 4), BAD_PARAM_ATTRIBUTE_TYPE);
            assert_eq!(finalize(&mut *t), BAD_PARAM); // missing attributes
            assert_eq!(set(t, A_TENSOR_UID, AT_INT64, &b64(&[1, 2]), 2), BAD_PARAM); // single-valued
            let op = create(BD_OP_CONV_FWD).unwrap();
            assert_eq!(set(op, A_FWD_X, AT_BACKEND_DESCRIPTOR, &desc_ptr(t), 1), SUCCESS);
            assert_eq!(finalize(&mut *op), BAD_PARAM); // conv descriptor missing
            let cd = conv_desc();
            assert_eq!(set(op, A_FWD_CONV, AT_BACKEND_DESCRIPTOR, &desc_ptr(cd), 1), SUCCESS);
            assert_eq!(finalize(&mut *op), BAD_PARAM_NOT_FINALIZED); // x is an unfinalized tensor

            assert_eq!(create(99).unwrap_err(), BAD_PARAM);
            // set after finalize is rejected
            let x = tensor(1, &[1, 1, 1, 1], &[1, 1, 1, 1], DT_FLOAT);
            assert_eq!(set(x, A_TENSOR_UID, AT_INT64, &b64(&[2]), 1), BAD_PARAM);
        }
    }

    #[test]
    fn pointwise_graph_has_no_engines() {
        unsafe {
            // a pointwise op descriptor (type 13) finalizes as unsupported
            let pw = create(13).unwrap();
            assert_eq!(finalize(&mut *pw), NOT_SUPPORTED_GRAPH_PATTERN);
            // a graph over an unsupported conv (double data) finalizes with zero engines;
            // heuristics then report NOT_SUPPORTED_GRAPH_PATTERN
            let x = tensor(1, &[1, 2, 4, 4], &[32, 16, 4, 1], DT_DOUBLE);
            let w = tensor(2, &[2, 2, 1, 1], &[2, 1, 1, 1], DT_DOUBLE);
            let c = conv_desc();
            let op = create(BD_OP_CONV_FWD).unwrap();
            set(op, A_FWD_X, AT_BACKEND_DESCRIPTOR, &desc_ptr(x), 1);
            set(op, A_FWD_W, AT_BACKEND_DESCRIPTOR, &desc_ptr(w), 1);
            let y = tensor(3, &[1, 2, 6, 6], &[72, 36, 6, 1], DT_DOUBLE);
            set(op, A_FWD_Y, AT_BACKEND_DESCRIPTOR, &desc_ptr(y), 1);
            set(op, A_FWD_CONV, AT_BACKEND_DESCRIPTOR, &desc_ptr(c), 1);
            assert_eq!(finalize(&mut *op), NOT_SUPPORTED_GRAPH_PATTERN);
        }
    }

    /// Replays the call sequence of cudnn_frontend 1.x as used by PyTorch's Conv_v8.cpp
    /// (read from EngineHeuristics, Engine, EngineConfig, ExecutionPlan, Filters headers):
    /// heuristics -> count query (requested 0, null out) -> ENGINECFG results -> hasNumericalNote
    /// (ENGINECFG_ENGINE into a pre-created ENGINE, notes with null out) -> plan build
    /// (ENGINECFG_ENGINE again, GLOBAL_INDEX, KNOB_CHOICES, notes, workspace) -> variant pack.
    #[test]
    fn mock_frontend_pytorch_sequence() {
        unsafe {
            for (opty, dt) in [(BD_OP_CONV_FWD, DT_HALF), (BD_OP_CONV_BWD_DATA, DT_FLOAT), (BD_OP_CONV_BWD_FILTER, DT_BFLOAT16)] {
                let x = tensor(b'x' as i64, &[2, 4, 8, 8], &[256, 64, 8, 1], dt);
                let w = tensor(b'w' as i64, &[6, 4, 3, 3], &[36, 9, 3, 1], dt);
                let y = tensor(b'y' as i64, &[2, 6, 8, 8], &[384, 64, 8, 1], dt);
                let c = conv_desc();
                let op = create(opty).unwrap();
                let (xn, yn, wn, cn) = match opty {
                    BD_OP_CONV_FWD => (A_FWD_X, A_FWD_Y, A_FWD_W, A_FWD_CONV),
                    BD_OP_CONV_BWD_DATA => (A_BWDD_DX, A_BWDD_DY, A_BWDD_W, A_BWDD_CONV),
                    _ => (A_BWDF_X, A_BWDF_DY, A_BWDF_DW, A_BWDF_CONV),
                };
                for (n, t) in [(xn, x), (yn, y), (wn, w), (cn, c)] {
                    assert_eq!(set(op, n, AT_BACKEND_DESCRIPTOR, &desc_ptr(t), 1), SUCCESS);
                }
                assert_eq!(finalize(&mut *op), SUCCESS, "op {opty}");
                let g = create(BD_OPERATIONGRAPH).unwrap();
                assert_eq!(set(g, A_GRAPH_HANDLE, AT_HANDLE, &1usize.to_le_bytes(), 1), SUCCESS);
                assert_eq!(set(g, A_GRAPH_OPS, AT_BACKEND_DESCRIPTOR, &desc_ptr(op), 1), SUCCESS);
                assert_eq!(set(g, 803, AT_BOOLEAN, &[0], 1), SUCCESS); // IS_DYNAMIC_SHAPE_ENABLED
                assert_eq!(finalize(&mut *g), SUCCESS);
                for mode in [0i32, 1, 2, 3] {
                    let h = create(BD_ENGINEHEUR).unwrap();
                    assert_eq!(set(h, A_HEUR_GRAPH, AT_BACKEND_DESCRIPTOR, &desc_ptr(g), 1), SUCCESS);
                    assert_eq!(set(h, A_HEUR_MODE, 8, &mode.to_le_bytes(), 1), SUCCESS);
                    assert_eq!(finalize(&mut *h), SUCCESS, "heur mode {mode}");
                    // getEngineConfigCount: requested 0, out null
                    let mut n = -1i64;
                    assert_eq!(get_attribute(&*h, A_HEUR_RESULTS, AT_BACKEND_DESCRIPTOR, 0, &mut n, std::ptr::null_mut()), SUCCESS);
                    assert_eq!(n, 1);
                    let cfgs = [create(BD_ENGINECFG).unwrap()];
                    n = -1;
                    assert_eq!(get_attribute(&*h, A_HEUR_RESULTS, AT_BACKEND_DESCRIPTOR, 1, &mut n, cfgs.as_ptr() as *mut u8), SUCCESS);
                    assert_eq!(n, 1);
                    // hasNumericalNote: fetch ENGINE into a fresh descriptor, query notes (null out, then array)
                    let eng = create(BD_ENGINE).unwrap();
                    n = -1;
                    assert_eq!(get_attribute(&*cfgs[0], A_CFG_ENGINE, AT_BACKEND_DESCRIPTOR, 1, &mut n, [eng].as_ptr() as *mut u8), SUCCESS);
                    assert_eq!(n, 1);
                    assert!((*eng).finalized, "engine descriptor filled in place");
                    let mut notes = [99i32; 8];
                    for out in [std::ptr::null_mut(), notes.as_mut_ptr() as *mut u8] {
                        n = -1;
                        assert_eq!(get_attribute(&*eng, 1303, 11, 8, &mut n, out), SUCCESS); // NUMERICAL_NOTE
                        assert_eq!(n, 0);
                        n = -1;
                        assert_eq!(get_attribute(&*eng, 1305, 19, 8, &mut n, out), SUCCESS); // BEHAVIOR_NOTE
                        assert_eq!(n, 0);
                    }
                    // Engine::build knob query
                    let knobs = [create(8).unwrap(); 1];
                    n = -1;
                    assert_eq!(get_attribute(&*eng, 1302, AT_BACKEND_DESCRIPTOR, 1, &mut n, knobs.as_ptr() as *mut u8), SUCCESS);
                    assert_eq!(n, 0);
                    if mode == 3 {
                        // cudnn_frontend frees the graph, ops, tensors and conv desc before the
                        // plan is built from the (kept) engine config.
                        for d in [h, g, op, x, w, y, c] {
                            destroy(d);
                        }
                    }
                    // ExecutionPlanBuilder
                    let p = create(BD_EXECUTION_PLAN).unwrap();
                    assert_eq!(set(p, A_PLAN_HANDLE, AT_HANDLE, &1usize.to_le_bytes(), 1), SUCCESS);
                    assert_eq!(set(p, A_PLAN_CFG, AT_BACKEND_DESCRIPTOR, &desc_ptr(cfgs[0]), 1), SUCCESS);
                    assert_eq!(finalize(&mut *p), SUCCESS);
                    let eng2 = create(BD_ENGINE).unwrap();
                    assert_eq!(get_attribute(&*cfgs[0], A_CFG_ENGINE, AT_BACKEND_DESCRIPTOR, 1, &mut n, [eng2].as_ptr() as *mut u8), SUCCESS);
                    let mut idx = [0u8; 8];
                    assert_eq!(get_attribute(&*eng2, A_ENGINE_INDEX, AT_INT64, 1, &mut n, idx.as_mut_ptr()), SUCCESS);
                    assert_eq!((n, i64::from_le_bytes(idx)), (1, 0));
                    n = -1;
                    assert_eq!(get_attribute(&*cfgs[0], A_CFG_KNOBS, AT_BACKEND_DESCRIPTOR, 16, &mut n, [std::ptr::null_mut::<BDesc>(); 16].as_ptr() as *mut u8), SUCCESS);
                    assert_eq!(n, 0);
                    let mut ws = -1i64;
                    assert_eq!(get_attribute(&*p, A_PLAN_WORKSPACE, AT_INT64, 1, std::ptr::null_mut(), (&mut ws as *mut i64).cast()), SUCCESS);
                    assert_eq!(ws, 0);
                    // VariantPack uids 'x','y','w' with null workspace
                    let vp = create(BD_VARIANT_PACK).unwrap();
                    set(vp, A_VP_UIDS, AT_INT64, &b64(&[b'x' as i64, b'y' as i64, b'w' as i64]), 3);
                    let pv: Vec<u8> = [0x10usize, 0x20, 0x30].iter().flat_map(|v| v.to_le_bytes()).collect();
                    set(vp, A_VP_PTRS, AT_VOID_PTR, &pv, 3);
                    set(vp, A_VP_WORKSPACE, AT_VOID_PTR, &0usize.to_le_bytes(), 1);
                    assert_eq!(finalize(&mut *vp), SUCCESS);
                    let e = (*p).exec.unwrap();
                    let ptrs = resolve_pointers(&*vp, &e.uids).unwrap();
                    // kernel roles (a, b, out): fwd x,w,y; bwd data dy,w,dx; bwd filter x,dy,dw
                    let want = match opty {
                        BD_OP_CONV_FWD => [0x10, 0x30, 0x20],
                        BD_OP_CONV_BWD_DATA => [0x20, 0x30, 0x10],
                        _ => [0x10, 0x20, 0x30],
                    };
                    assert_eq!(ptrs, want, "op {opty}");
                }
            }
        }
    }
}
