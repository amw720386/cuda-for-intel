"""Generate handwritten single-feature PTX probes into handwritten/<family>/.
Each probe is a tiny sm_80 (or noted target) kernel with one feature under test.
Probes only need to parse/translate; operands are uninitialised registers.
Original work for this repo (MIT); instruction syntax follows the public PTX ISA."""
import shutil
from pathlib import Path

OUT = Path(__file__).parent / "handwritten"
if OUT.exists(): shutil.rmtree(OUT)

REGS = """\t.reg .pred %p<8>;
\t.reg .b16 %h<8>;
\t.reg .b32 %r<32>;
\t.reg .b64 %rd<16>;
\t.reg .f32 %f<16>;
"""
def probe(fam, name, lines, target="sm_80", version="8.0", pre="", entry=None):
    d = OUT / fam; d.mkdir(parents=True, exist_ok=True)
    body = "\n".join("\t" + l for l in lines)
    import re
    init = []
    used = set(re.findall(r"%(?:rd|fd|r|f|h|p)\d+", body)) | {"%r1", "%r31"}
    used -= {"%rd1", "%rd2"}
    for r in sorted(used, key=lambda x: (re.sub(r"\d", "", x), int(re.sub(r"\D", "", x)))):
        if r.startswith("%rd"): init.append(f"mov.u64 {r}, 1;")
        elif r.startswith("%fd"): init.append(f"mov.f64 {r}, 0d3FF0000000000000;")
        elif r.startswith("%r"): init.append(f"mov.u32 {r}, 1;")
        elif r.startswith("%f"): init.append(f"mov.f32 {r}, 0f3F800000;")
        elif r.startswith("%h"): init.append(f"ld.global.u16 {r}, [%rd2];")
    for r in sorted(x for x in used if x.startswith("%p")): init.append(f"setp.eq.u32 {r}, %r31, 1;")
    body = "\n".join("\t" + l for l in init) + "\n" + body
    fdreg = "\t.reg .f64 %fd<8>;\n" if "%fd" in body else ""
    txt = f".version {version}\n.target {target}\n.address_size 64\n{pre}\n.visible .entry {name}(\n\t.param .u64 out\n)\n{{\n{REGS}{fdreg}\tld.param.u64 %rd1, [out];\n\tcvta.to.global.u64 %rd2, %rd1;\n{body}\n\tst.global.u32 [%rd2], %r1;\n\tret;\n}}\n"
    (d / f"{name}.ptx").write_text(txt)

# ---- mma / wmma / ldmatrix
A = "{%r1,%r2,%r3,%r4}"; B = "{%r5,%r6}"; C = "{%f1,%f2,%f3,%f4}"; D = "{%f5,%f6,%f7,%f8}"
for n, t, ab, cd, k in [
 ("m16n8k8_f16_f32","m16n8k8","f16","f32",None),
 ("m16n8k16_f16_f32","m16n8k16","f16","f32",None),
 ("m16n8k16_bf16_f32","m16n8k16","bf16","f32",None),
 ("m16n8k8_bf16_f32","m16n8k8","bf16","f32",None),
 ("m8n8k4_f16_f32","m8n8k4","f16","f32","sm_70"),
 ("m16n8k8_tf32_f32","m16n8k8","tf32","f32",None),
 ("m16n8k4_tf32_f32","m16n8k4","tf32","f32",None),
]:
    shape = t
    # fragment arity per shape/type (approximate; the translator should reject on opcode first)
    probe("mma", f"mma_{n}", [f"mma.sync.aligned.{shape}.row.col.{cd}.{ab}.{ab}.{cd} {D}, {A}, {B}, {C};"], target=k or "sm_80")
probe("mma","mma_m16n8k16_f16_f16acc",["mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%r10,%r11}, {%r1,%r2,%r3,%r4}, {%r5,%r6}, {%r7,%r8};"])
probe("mma","mma_m16n8k32_s8_s32",["mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%r10,%r11,%r12,%r13}, {%r1,%r2,%r3,%r4}, {%r5,%r6}, {%r7,%r8,%r9,%r14};"])
probe("mma","mma_m16n8k32_u8_satfinite",["mma.sync.aligned.m16n8k32.row.col.satfinite.s32.u8.u8.s32 {%r10,%r11,%r12,%r13}, {%r1,%r2,%r3,%r4}, {%r5,%r6}, {%r7,%r8,%r9,%r14};"])
probe("mma","mma_m8n8k16_s8",["mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%r10,%r11}, {%r1}, {%r5}, {%r7,%r8};"])
probe("mma","mma_m16n8k64_s4",["mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {%r10,%r11,%r12,%r13}, {%r1,%r2,%r3,%r4}, {%r5,%r6}, {%r7,%r8,%r9,%r14};"])
probe("mma","mma_m16n8k256_b1_xor_popc",["mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc {%r10,%r11,%r12,%r13}, {%r1,%r2,%r3,%r4}, {%r5,%r6}, {%r7,%r8,%r9,%r14};"])
probe("mma","mma_m16n8k16_f64_m8n8k4",["mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%fd1,%fd2}, {%fd3}, {%fd4}, {%fd5,%fd6};"])
probe("mma","mma_sp_m16n8k32_f16",["mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 {%f5,%f6,%f7,%f8}, {%r1,%r2,%r3,%r4}, {%r5,%r6,%r7,%r8}, {%f1,%f2,%f3,%f4}, %r9, 0x0;"])
probe("mma","mma_m16n8k16_f16_f32_scalar_regs",["mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%f5,%f6,%f7,%f8}, {%r1,%r2,%r3,%r4}, {%r5,%r6}, {%f1,%f2,%f3,%f4};"])
probe("mma","wgmma_m64n8k16_f16",["wgmma.fence.sync.aligned;","wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%f1,%f2,%f3,%f4}, %rd3, %rd4, p, 1, 1, 0, 0;","wgmma.commit_group.sync.aligned;","wgmma.wait_group.sync.aligned 0;"], target="sm_90a", version="8.0")
probe("mma","tcgen05_mma_sm100",["tcgen05.fence::before_thread_sync;"], target="sm_100a", version="8.6")
probe("wmma","wmma_load_a_m16n16k16_f16",["wmma.load.a.sync.aligned.row.m16n16k16.global.f16 {%r1,%r2,%r3,%r4,%r5,%r6,%r7,%r8}, [%rd2], 16;"])
probe("wmma","wmma_mma_m16n16k16_f16_f32",["wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32 {%f1,%f2,%f3,%f4,%f5,%f6,%f7,%f8}, {%r1,%r2,%r3,%r4,%r5,%r6,%r7,%r8}, {%r9,%r10,%r11,%r12,%r13,%r14,%r15,%r16}, {%f1,%f2,%f3,%f4,%f5,%f6,%f7,%f8};"])
probe("wmma","wmma_store_d_m16n16k16_f32",["wmma.store.d.sync.aligned.row.m16n16k16.global.f32 [%rd2], {%f1,%f2,%f3,%f4,%f5,%f6,%f7,%f8}, 16;"])
for x in (1,2,4):
    regs = "{" + ",".join(f"%r{i+1}" for i in range(x)) + "}"
    probe("ldmatrix", f"ldmatrix_x{x}", [f"ldmatrix.sync.aligned.m8n8.x{x}.shared.b16 {regs}, [%rd2];"])
    probe("ldmatrix", f"ldmatrix_x{x}_trans", [f"ldmatrix.sync.aligned.m8n8.x{x}.trans.shared.b16 {regs}, [%rd2];"])
probe("ldmatrix","stmatrix_x4",["stmatrix.sync.aligned.m8n8.x4.shared.b16 [%rd2], {%r1,%r2,%r3,%r4};"], target="sm_90", version="7.8")
probe("ldmatrix","ldmatrix_m16n16_b8",["ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {%r1,%r2}, [%rd2];"], target="sm_100a", version="8.6")

# ---- cp.async / bulk / mbarrier
for sz,cop in [(4,"ca"),(8,"ca"),(16,"ca"),(16,"cg")]:
    probe("cp_async", f"cp_async_{cop}_{sz}", [f"cp.async.{cop}.shared.global [%rd2], [%rd2], {sz};"])
probe("cp_async","cp_async_ca_zfill",["cp.async.ca.shared.global [%rd2], [%rd2], 16, %r2;"])
probe("cp_async","cp_async_cg_l2_128",["cp.async.cg.shared.global.L2::128B [%rd2], [%rd2], 16;"])
probe("cp_async","cp_async_commit_group",["cp.async.commit_group;"])
probe("cp_async","cp_async_wait_group_0",["cp.async.wait_group 0;"])
probe("cp_async","cp_async_wait_group_2",["cp.async.wait_group 2;"])
probe("cp_async","cp_async_wait_all",["cp.async.wait_all;"])
probe("cp_async","cp_async_mbarrier_arrive",["cp.async.mbarrier.arrive.shared.b64 [%rd2];"])
probe("cp_async","cp_async_bulk_g2s",["cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%rd2], [%rd2], 128, [%rd3];"], target="sm_90", version="8.0")
probe("cp_async","cp_async_bulk_tensor_2d",["cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%rd2], [%rd3, {%r1,%r2}], [%rd4];"], target="sm_90", version="8.0")
probe("cp_async","cp_async_bulk_commit_group",["cp.async.bulk.commit_group;","cp.async.bulk.wait_group.read 0;"], target="sm_90", version="8.0")
probe("cp_async","st_async_b32",["st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.b32 [%rd2], %r1, [%rd3];"], target="sm_90", version="8.1")
probe("cp_async","red_async_add_u32",["red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u32 [%rd2], %r1, [%rd3];"], target="sm_90", version="8.1")
probe("cp_async","mbarrier_init",["mbarrier.init.shared.b64 [%rd2], 32;"])
probe("cp_async","mbarrier_arrive",["mbarrier.arrive.shared.b64 %rd3, [%rd2];"])
probe("cp_async","mbarrier_try_wait_parity",["mbarrier.try_wait.parity.shared.b64 %p1, [%rd2], %r1;"], target="sm_90", version="7.8")
probe("cp_async","mbarrier_expect_tx",["mbarrier.expect_tx.relaxed.cta.shared.b64 [%rd2], 128;"], target="sm_90", version="8.0")
probe("cp_async","fence_proxy_async",["fence.proxy.async.shared::cta;"], target="sm_90", version="8.0")
probe("cp_async","prefetch_tensormap",["prefetch.tensormap [%rd2];"], target="sm_90", version="8.0")

# ---- barriers / fences / cluster / misc sync
probe("barrier","bar_sync_0",["bar.sync 0;"])
probe("barrier","bar_sync_count",["bar.sync 1, 128;"])
probe("barrier","bar_sync_reg",["bar.sync %r1, %r2;"])
probe("barrier","barrier_sync",["barrier.sync 0;"])
probe("barrier","barrier_sync_aligned",["barrier.sync.aligned 0;"])
probe("barrier","bar_arrive",["bar.arrive 1, 128;"])
probe("barrier","bar_red_popc",["bar.red.popc.u32 %r1, 0, %p1;"])
probe("barrier","bar_red_and_pred",["bar.red.and.pred %p2, 0, %p1;"])
probe("barrier","bar_warp_sync",["bar.warp.sync 0xffffffff;"])
probe("barrier","barrier_cluster_arrive_wait",["barrier.cluster.arrive;","barrier.cluster.wait;"], target="sm_90", version="7.8")
probe("barrier","membar_gl",["membar.gl;"])
probe("barrier","membar_cta",["membar.cta;"])
probe("barrier","fence_sc_gpu",["fence.sc.gpu;"])
probe("barrier","fence_acq_rel_cta",["fence.acq_rel.cta;"])
probe("barrier","ld_acquire_gpu",["ld.acquire.gpu.global.u32 %r1, [%rd2];"])
probe("barrier","st_release_gpu",["st.release.gpu.global.u32 [%rd2], %r1;"])
probe("barrier","ld_relaxed_sys",["ld.relaxed.sys.global.u32 %r1, [%rd2];"])
probe("barrier","griddepcontrol_wait",["griddepcontrol.wait;","griddepcontrol.launch_dependents;"], target="sm_90", version="7.8")
probe("barrier","elect_sync",["elect.sync %r1|%p1, 0xffffffff;"], target="sm_90", version="8.0")
probe("barrier","setmaxnreg_inc",["setmaxnreg.inc.sync.aligned.u32 240;"], target="sm_90a", version="8.0")
probe("barrier","getctarank",["getctarank.shared::cluster.u32 %r1, %rd2;"], target="sm_90", version="7.8")
probe("barrier","mapa_shared_cluster",["mapa.shared::cluster.u64 %rd3, %rd2, %r1;"], target="sm_90", version="7.8")
probe("barrier","nanosleep",["nanosleep.u32 100;"])
probe("barrier","trap",["trap;"])
probe("barrier","brkpt",["brkpt;"])
probe("barrier","exit",["exit;"])

# ---- warp collectives
for mode in ("idx","up","down","bfly"):
    probe("shfl", f"shfl_sync_{mode}_b32", [f"shfl.sync.{mode}.b32 %r1|%p1, %r2, %r3, %r4, 0xffffffff;"])
    probe("shfl", f"shfl_sync_{mode}_b32_imm", [f"shfl.sync.{mode}.b32 %r1, %r2, 16, 31, 0xffffffff;"])
    probe("shfl", f"shfl_legacy_{mode}_b32", [f"shfl.{mode}.b32 %r1, %r2, 1, 31;"], target="sm_35", version="6.0")
probe("shfl","shfl_sync_bfly_f32",["shfl.sync.bfly.b32 %r1, %r2, 16, 31, 0xffffffff;","mov.b32 %f1, %r1;"])
probe("shfl","shfl_sync_idx_b64_via_mov",["mov.b64 {%r2,%r3}, %rd3;","shfl.sync.idx.b32 %r4, %r2, 0, 31, 0xffffffff;","shfl.sync.idx.b32 %r5, %r3, 0, 31, 0xffffffff;","mov.b64 %rd4, {%r4,%r5};"])
for m in ("ballot","all","any","uni"):
    d = "%r1" if m=="ballot" else "%p2"
    probe("vote", f"vote_sync_{m}", [f"vote.sync.{m}{'.b32' if m=='ballot' else '.pred'} {d}, %p1, 0xffffffff;"])
probe("vote","vote_sync_ballot_member",["vote.sync.ballot.b32 %r1, %p1, %r2;"])
probe("vote","vote_legacy_ballot",["vote.ballot.b32 %r1, %p1;"], target="sm_35", version="6.0")
probe("vote","activemask",["activemask.b32 %r1;"])
probe("vote","match_any_sync_b32",["match.any.sync.b32 %r1, %r2, 0xffffffff;"], target="sm_70", version="6.0")
probe("vote","match_all_sync_b32",["match.all.sync.b32 %r1|%p1, %r2, 0xffffffff;"], target="sm_70", version="6.0")
probe("vote","match_any_sync_b64",["match.any.sync.b64 %r1, %rd3, 0xffffffff;"], target="sm_70", version="6.0")
for op in ("add","min","max","and","or","xor"):
    probe("redux", f"redux_sync_{op}_u32", [f"redux.sync.{op}.u32 %r1, %r2, 0xffffffff;"], target="sm_80", version="7.0")
probe("redux","redux_sync_add_s32",["redux.sync.add.s32 %r1, %r2, 0xffffffff;"])
probe("redux","redux_sync_min_s32",["redux.sync.min.s32 %r1, %r2, 0xffffffff;"])
probe("redux","redux_sync_max_f32_nan",["redux.sync.max.NaN.f32 %f1, %f2, 0xffffffff;"], target="sm_100a", version="8.6")
for n in ("eq","lt","le","ge","gt"):
    probe("sreg", f"lanemask_{n}", [f"mov.u32 %r1, %lanemask_{n};"])
for s in ("laneid","warpid","nwarpid","smid","nsmid","gridid","clock","clock64","globaltimer","total_smem_size","dynamic_smem_size","cluster_ctaid.x","cluster_nctaid.x","cluster_ctarank","is_explicit_cluster","pm0","envreg0"):
    t = "u64" if s in ("clock64","globaltimer") else "u32"
    probe("sreg", "sreg_"+s.replace(".","_"), [f"mov.{t} {'%rd3' if t=='u64' else '%r1'}, %{s};"], target="sm_90" if "cluster" in s or "smem" in s else "sm_80", version="7.8")
for s in ("tid.x","ntid.x","ctaid.x","nctaid.x","tid.y","ctaid.y","ctaid.z","WARP_SZ"):
    probe("sreg", "sreg_"+s.replace(".","_"), [f"mov.u32 %r1, %{s};"])

# ---- bit manipulation / integer extensions
probe("bitmanip","prmt_default",["prmt.b32 %r1, %r2, %r3, %r4;"])
probe("bitmanip","prmt_imm_selector",["prmt.b32 %r1, %r2, %r3, 0x3210;"])
for m in ("f4e","b4e","rc8","ecl","ecr","rc16"):
    probe("bitmanip", f"prmt_{m}", [f"prmt.b32.{m} %r1, %r2, %r3, %r4;"])
probe("bitmanip","dp4a_s32_s32",["dp4a.s32.s32 %r1, %r2, %r3, %r4;"], target="sm_61", version="5.0")
probe("bitmanip","dp4a_u32_u32",["dp4a.u32.u32 %r1, %r2, %r3, %r4;"], target="sm_61", version="5.0")
probe("bitmanip","dp4a_s32_u32",["dp4a.s32.u32 %r1, %r2, %r3, %r4;"], target="sm_61", version="5.0")
probe("bitmanip","dp2a_lo_s32_s32",["dp2a.lo.s32.s32 %r1, %r2, %r3, %r4;"], target="sm_61", version="5.0")
probe("bitmanip","dp2a_hi_u32_u32",["dp2a.hi.u32.u32 %r1, %r2, %r3, %r4;"], target="sm_61", version="5.0")
probe("bitmanip","lop3_b32",["lop3.b32 %r1, %r2, %r3, %r4, 0xe8;"], target="sm_50", version="7.0")
probe("bitmanip","lop3_b32_imm_c",["lop3.b32 %r1, %r2, %r3, 0xf0f0f0f0, 0x96;"], target="sm_50", version="7.0")
probe("bitmanip","lop3_pred",["lop3.b32 %r1, %r2, %r3, %r4, 0xe8;"], target="sm_50", version="7.0")
probe("bitmanip","bfe_u32",["bfe.u32 %r1, %r2, %r3, %r4;"])
probe("bitmanip","bfe_s32",["bfe.s32 %r1, %r2, 8, 8;"])
probe("bitmanip","bfe_u64",["bfe.u64 %rd3, %rd4, %r3, %r4;"])
probe("bitmanip","bfi_b32",["bfi.b32 %r1, %r2, %r3, %r4, %r5;"])
probe("bitmanip","bfi_b64",["bfi.b64 %rd3, %rd4, %rd5, %r4, %r5;"])
probe("bitmanip","fns_b32",["fns.b32 %r1, %r2, %r3, %r4;"], target="sm_30", version="6.0")
probe("bitmanip","clz_b32",["clz.b32 %r1, %r2;"])
probe("bitmanip","clz_b64",["clz.b64 %r1, %rd3;"])
probe("bitmanip","popc_b32",["popc.b32 %r1, %r2;"])
probe("bitmanip","popc_b64",["popc.b64 %r1, %rd3;"])
probe("bitmanip","brev_b32",["brev.b32 %r1, %r2;"])
probe("bitmanip","brev_b64",["brev.b64 %rd3, %rd4;"])
probe("bitmanip","bfind_u32",["bfind.u32 %r1, %r2;"])
probe("bitmanip","bfind_shiftamt_s32",["bfind.shiftamt.s32 %r1, %r2;"])
probe("bitmanip","bmsk_wrap",["bmsk.wrap.b32 %r1, %r2, %r3;"], target="sm_70", version="7.6")
probe("bitmanip","szext_wrap_s32",["szext.wrap.s32 %r1, %r2, %r3;"], target="sm_70", version="7.6")
probe("bitmanip","shf_l_wrap",["shf.l.wrap.b32 %r1, %r2, %r3, %r4;"], target="sm_32", version="3.1")
probe("bitmanip","shf_r_clamp",["shf.r.clamp.b32 %r1, %r2, %r3, %r4;"], target="sm_32", version="3.1")
probe("bitmanip","sad_u32",["sad.u32 %r1, %r2, %r3, %r4;"])
probe("bitmanip","mul24_lo_u32",["mul24.lo.u32 %r1, %r2, %r3;"])
probe("bitmanip","mad24_lo_s32",["mad24.lo.s32 %r1, %r2, %r3, %r4;"])
probe("bitmanip","mul_hi_u64",["mul.hi.u64 %rd3, %rd4, %rd5;"])
probe("bitmanip","mad_hi_s32",["mad.hi.s32 %r1, %r2, %r3, %r4;"])
probe("bitmanip","mad_lo_cc",["mad.lo.cc.u32 %r1, %r2, %r3, %r4;","madc.hi.u32 %r5, %r2, %r3, %r4;"], target="sm_20", version="6.0")
probe("bitmanip","add_cc_addc",["add.cc.u32 %r1, %r2, %r3;","addc.u32 %r4, %r5, %r6;"], target="sm_20", version="6.0")
probe("bitmanip","cnot_b32",["cnot.b32 %r1, %r2;"])
probe("bitmanip","not_b64",["not.b64 %rd3, %rd4;"])
probe("bitmanip","rem_s32",["rem.s32 %r1, %r2, %r3;"])
probe("bitmanip","div_u64",["div.u64 %rd3, %rd4, %rd5;"])
probe("bitmanip","abs_s32",["abs.s32 %r1, %r2;"])
probe("bitmanip","min_max_s64",["min.s64 %rd3, %rd4, %rd5;","max.u64 %rd6, %rd4, %rd5;"])
probe("bitmanip","vabsdiff4_u32",["vabsdiff4.u32.u32.u32 %r1, %r2, %r3, %r4;"], target="sm_30", version="6.0")
probe("bitmanip","vadd2_u32",["vadd2.u32.u32.u32 %r1, %r2, %r3, %r4;"], target="sm_30", version="6.0")
probe("bitmanip","vset4_eq",["vset4.u32.u32.eq %r1, %r2, %r3, %r4;"], target="sm_30", version="6.0")
probe("bitmanip","vmax_s32_s32_s32",["vmax.s32.s32.s32 %r1, %r2, %r3, %r4;"], target="sm_30", version="6.0")
probe("bitmanip","isspacep_shared",["isspacep.shared %p1, %rd2;"])
probe("bitmanip","cvta_to_shared",["cvta.to.shared.u64 %rd3, %rd2;"])
probe("bitmanip","cvta_shared",["cvta.shared.u64 %rd3, %rd2;"])
probe("bitmanip","cvta_const",["cvta.const.u64 %rd3, %rd2;"])
probe("bitmanip","selp_b64",["selp.b64 %rd3, %rd4, %rd5, %p1;"])
probe("bitmanip","setp_and_combine",["setp.lt.and.s32 %p1, %r1, %r2, %p2;"])
probe("bitmanip","slct_f32_s32",["slct.f32.s32 %f1, %f2, %f3, %r1;"])

# ---- tex / surf
probe("tex","tex_1d_v4_f32_s32",["tex.1d.v4.f32.s32 {%f1,%f2,%f3,%f4}, [%rd3, {%r1}];"], pre=".global .texref t;")
probe("tex","tex_2d_v4_f32_f32_texobj",["tex.2d.v4.f32.f32 {%f1,%f2,%f3,%f4}, [%rd3, {%f5,%f6}];"])
probe("tex","tex_3d_v4_s32_s32",["tex.3d.v4.s32.s32 {%r1,%r2,%r3,%r4}, [%rd3, {%r5,%r6,%r7,%r8}];"])
probe("tex","tex_level_2d",["tex.level.2d.v4.f32.f32 {%f1,%f2,%f3,%f4}, [%rd3, {%f5,%f6}], %f7;"])
probe("tex","tld4_r_2d",["tld4.r.2d.v4.f32.f32 {%f1,%f2,%f3,%f4}, [%rd3, {%f5,%f6}];"])
probe("tex","txq_width",["txq.width.b32 %r1, [%rd3];"])
probe("tex","suld_b_1d_b32",["suld.b.1d.b32.trap %r1, [%rd3, {%r2}];"])
probe("tex","sust_b_2d_v4_b32",["sust.b.2d.v4.b32.trap [%rd3, {%r1,%r2}], {%r3,%r4,%r5,%r6};"])
probe("tex","sured_b_1d_add_u32",["sured.b.add.1d.u32.trap [%rd3, {%r1}], %r2;"])

# ---- memory/cache/hints
probe("memory","ld_global_nc_u32",["ld.global.nc.u32 %r1, [%rd2];"])
probe("memory","ld_global_nc_v4_u32",["ld.global.nc.v4.u32 {%r1,%r2,%r3,%r4}, [%rd2];"])
probe("memory","ld_global_v2_u64",["ld.global.v2.u64 {%rd3,%rd4}, [%rd2];"])
probe("memory","ld_global_v4_f32",["ld.global.v4.f32 {%f1,%f2,%f3,%f4}, [%rd2];"])
probe("memory","ld_global_cs",["ld.global.cs.u32 %r1, [%rd2];"])
probe("memory","ld_global_lu",["ld.global.lu.u32 %r1, [%rd2];"])
probe("memory","ld_global_L1_no_allocate",["ld.global.L1::no_allocate.u32 %r1, [%rd2];"])
probe("memory","ld_global_L2_128B",["ld.global.L2::128B.u32 %r1, [%rd2];"])
probe("memory","ld_volatile_global",["ld.volatile.global.u32 %r1, [%rd2];"])
probe("memory","st_global_cs",["st.global.cs.u32 [%rd2], %r1;"])
probe("memory","st_global_wt",["st.global.wt.u32 [%rd2], %r1;"])
probe("memory","st_global_v4_u32",["st.global.v4.u32 [%rd2], {%r1,%r2,%r3,%r4};"])
probe("memory","st_global_u16",["st.global.u16 [%rd2], %h1;"])
probe("memory","ld_shared_v4",["ld.shared.v4.u32 {%r1,%r2,%r3,%r4}, [%rd2];"])
probe("memory","ld_shared_cluster",["ld.shared::cluster.u32 %r1, [%rd2];"], target="sm_90", version="7.8")
probe("memory","ld_const_u32",["ld.const.u32 %r1, [%rd2];"])
probe("memory","ld_param_v2",["ld.param.v2.u32 {%r1,%r2}, [out];"])
probe("memory","prefetch_global_L2",["prefetch.global.L2 [%rd2];"])
probe("memory","prefetchu_L1",["prefetchu.L1 [%rd2];"])
probe("memory","discard_global_L2",["discard.global.L2 [%rd2], 128;"], target="sm_80", version="7.4")
probe("memory","createpolicy_fractional",["createpolicy.fractional.L2::evict_last.b64 %rd3, 1.0;","ld.global.L2::cache_hint.u32 %r1, [%rd2], %rd3;"], target="sm_80", version="7.4")
probe("memory","applypriority_L2",["applypriority.global.L2::evict_normal [%rd2], 128;"], target="sm_80", version="7.4")
probe("memory","multimem_ld_reduce",["multimem.ld_reduce.relaxed.sys.global.add.u32 %r1, [%rd2];"], target="sm_90", version="8.1")
probe("memory","local_depot",["st.local.u32 [%rd2], %r1;"], pre="")
probe("memory","mov_vector_pack_b32",["mov.b32 %r1, {%h1,%h2};"])
probe("memory","mov_b64_pack",["mov.b64 %rd3, {%r1,%r2};"])
probe("memory","call_noreturn_stub",["call.uni __assertfail, (%rd3, %rd4, %r1, %rd5, %rd6);"], pre=".extern .func __assertfail(.param .b64 a,.param .b64 b,.param .b32 c,.param .b64 d,.param .b64 e);")
probe("memory","vprintf_call",["call.uni (%r1), vprintf, (%rd3, %rd4);"], pre=".extern .func (.param .b32 r) vprintf(.param .b64 a,.param .b64 b);")
probe("memory","func_call_internal",["call.uni (%r1), helper, (%r2);"], pre=".func (.reg .b32 r) helper(.reg .b32 a) { add.u32 r, a, 1; ret; }")
probe("memory","global_array_init",["ld.global.u32 %r1, [tbl+8];"], pre=".global .align 4 .b8 tbl[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};")
probe("memory","const_array_init",["mov.u64 %rd3, ctbl;","ld.const.u32 %r1, [%rd3];"], pre=".const .align 4 .u32 ctbl[4] = {1,2,3,4};")
probe("memory","shared_extern_dynamic",["ld.shared.u32 %r1, [dyn];"], pre=".extern .shared .align 16 .b8 dyn[];")

# ---- atomics
for ins in ["atom.global.add.u32 %r1, [%rd2], %r2;","atom.global.add.f32 %f1, [%rd2], %f2;","atom.global.add.f64 %fd1, [%rd2], %fd2;",
  "atom.global.add.u64 %rd3, [%rd2], %rd4;","atom.global.add.noftz.f16x2 %r1, [%rd2], %r2;","atom.global.add.noftz.bf16x2 %r1, [%rd2], %r2;",
  "atom.global.cas.b32 %r1, [%rd2], %r2, %r3;","atom.global.cas.b64 %rd3, [%rd2], %rd4, %rd5;","atom.global.exch.b32 %r1, [%rd2], %r2;",
  "atom.global.max.s32 %r1, [%rd2], %r2;","atom.global.min.u32 %r1, [%rd2], %r2;","atom.global.max.u64 %rd3, [%rd2], %rd4;","atom.global.or.b32 %r1, [%rd2], %r2;",
  "atom.global.and.b32 %r1, [%rd2], %r2;","atom.global.xor.b32 %r1, [%rd2], %r2;","atom.global.inc.u32 %r1, [%rd2], %r2;","atom.global.dec.u32 %r1, [%rd2], %r2;",
  "atom.shared.add.u32 %r1, [%rd2], %r2;","atom.shared.add.f32 %f1, [%rd2], %f2;","atom.shared.cas.b32 %r1, [%rd2], %r2, %r3;",
  "atom.relaxed.gpu.global.add.u32 %r1, [%rd2], %r2;","atom.acq_rel.sys.global.add.u32 %r1, [%rd2], %r2;","atom.global.cas.b128 {%rd3,%rd4}, [%rd2], {%rd5,%rd6}, {%rd7,%rd8};",
  "red.global.add.u32 [%rd2], %r2;","red.global.add.f32 [%rd2], %f2;","red.global.add.noftz.f16x2 [%rd2], %r2;","red.global.max.u32 [%rd2], %r2;","red.global.add.u64 [%rd2], %rd4;",
  "red.shared.add.u32 [%rd2], %r2;","red.global.or.b32 [%rd2], %r2;","red.release.gpu.global.add.u32 [%rd2], %r2;","atom.global.min.f32 %f1, [%rd2], %f2;","atom.global.exch.b64 %rd3, [%rd2], %rd4;"]:
    n = ins.split(" ")[0].replace(".","_") + "_" + ins.split(" ")[1].strip("%,").replace("{","")[:0] + str(len(list((OUT/"atomics").glob("*"))) if (OUT/"atomics").exists() else 0)
    probe("atomics", n, [ins])

# ---- float / half / bf16 / cvt
fl = {
 "fma_rn_f16x2":"fma.rn.f16x2 %r1, %r2, %r3, %r4;","fma_rn_bf16x2":"fma.rn.bf16x2 %r1, %r2, %r3, %r4;","fma_rn_bf16":"fma.rn.bf16 %h1, %h2, %h3, %h4;",
 "add_f16x2":"add.rn.f16x2 %r1, %r2, %r3;","add_bf16":"add.rn.bf16 %h1, %h2, %h3;","mul_bf16x2":"mul.rn.bf16x2 %r1, %r2, %r3;",
 "min_nan_f16x2":"min.NaN.f16x2 %r1, %r2, %r3;","max_bf16":"max.bf16 %h1, %h2, %h3;","abs_bf16":"abs.bf16 %h1, %h2;","neg_f16":"neg.f16 %h1, %h2;",
 "ex2_approx_f16x2":"ex2.approx.f16x2 %r1, %r2;","ex2_approx_ftz_f32":"ex2.approx.ftz.f32 %f1, %f2;","lg2_approx_f32":"lg2.approx.f32 %f1, %f2;",
 "tanh_approx_f32":"tanh.approx.f32 %f1, %f2;","tanh_approx_f16x2":"tanh.approx.f16x2 %r1, %r2;","sin_approx_f32":"sin.approx.f32 %f1, %f2;","cos_approx_f32":"cos.approx.f32 %f1, %f2;",
 "rcp_approx_ftz_f32":"rcp.approx.ftz.f32 %f1, %f2;","rcp_rn_f32":"rcp.rn.f32 %f1, %f2;","rcp_rn_f64":"rcp.rn.f64 %fd1, %fd2;","rsqrt_approx_f32":"rsqrt.approx.ftz.f32 %f1, %f2;",
 "sqrt_rn_f32":"sqrt.rn.f32 %f1, %f2;","sqrt_approx_ftz_f32":"sqrt.approx.ftz.f32 %f1, %f2;","sqrt_rn_f64":"sqrt.rn.f64 %fd1, %fd2;",
 "div_full_f32":"div.full.f32 %f1, %f2, %f3;","div_rn_f32":"div.rn.f32 %f1, %f2, %f3;","div_approx_ftz_f32":"div.approx.ftz.f32 %f1, %f2, %f3;","div_rn_f64":"div.rn.f64 %fd1, %fd2, %fd3;",
 "fma_rn_f32":"fma.rn.f32 %f1, %f2, %f3, %f4;","fma_rn_ftz_f32":"fma.rn.ftz.f32 %f1, %f2, %f3, %f4;","fma_rn_f64":"fma.rn.f64 %fd1, %fd2, %fd3, %fd4;","mad_f32":"mad.f32 %f1, %f2, %f3, %f4;",
 "add_rz_f32":"add.rz.f32 %f1, %f2, %f3;","mul_rm_f32":"mul.rm.f32 %f1, %f2, %f3;","add_sat_f32":"add.sat.f32 %f1, %f2, %f3;","add_f64":"add.f64 %fd1, %fd2, %fd3;",
 "min_f32":"min.f32 %f1, %f2, %f3;","max_ftz_nan_f32":"max.ftz.NaN.f32 %f1, %f2, %f3;","abs_f32":"abs.f32 %f1, %f2;","neg_f64":"neg.f64 %fd1, %fd2;",
 "copysign_f32":"copysign.f32 %f1, %f2, %f3;","testp_nan_f32":"testp.notanumber.f32 %p1, %f1;","testp_inf_f64":"testp.infinite.f64 %p1, %fd1;",
 "setp_lt_f32":"setp.lt.f32 %p1, %f1, %f2;","setp_nan_f32":"setp.nan.f32 %p1, %f1, %f2;","setp_f16":"setp.lt.f16 %p1, %h1, %h2;","set_gt_f16x2":"set.gt.f16x2.f16x2 %r1, %r2, %r3;",
 "selp_f32":"selp.f32 %f1, %f2, %f3, %p1;","fma_relu_f16":"fma.rn.relu.f16 %h1, %h2, %h3, %h4;","ex2_approx_bf16":"ex2.approx.ftz.bf16 %h1, %h2;",
 "cvt_f32_f16":"cvt.f32.f16 %f1, %h1;","cvt_rn_f16_f32":"cvt.rn.f16.f32 %h1, %f1;","cvt_rn_f16x2_f32":"cvt.rn.f16x2.f32 %r1, %f1, %f2;","cvt_rn_bf16_f32":"cvt.rn.bf16.f32 %h1, %f1;",
 "cvt_rn_bf16x2_f32":"cvt.rn.bf16x2.f32 %r1, %f1, %f2;","cvt_f32_bf16":"cvt.f32.bf16 %f1, %h1;","cvt_rna_tf32_f32":"cvt.rna.tf32.f32 %r1, %f1;","cvt_rn_satfinite_e4m3x2_f32":"cvt.rn.satfinite.e4m3x2.f32 %h1, %f1, %f2;",
 "cvt_rn_f16x2_e4m3x2":"cvt.rn.f16x2.e4m3x2 %r1, %h1;","cvt_rni_s32_f32":"cvt.rni.s32.f32 %r1, %f1;","cvt_rzi_s32_f32":"cvt.rzi.s32.f32 %r1, %f1;","cvt_rmi_f32_f32":"cvt.rmi.f32.f32 %f1, %f2;",
 "cvt_rn_f32_s32":"cvt.rn.f32.s32 %f1, %r1;","cvt_f64_f32":"cvt.f64.f32 %fd1, %f1;","cvt_rn_f32_f64":"cvt.rn.f32.f64 %f1, %fd1;","cvt_u8_s32_sat":"cvt.sat.u8.s32 %r1, %r2;",
 "cvt_s64_s32":"cvt.s64.s32 %rd3, %r1;","cvt_u32_u64":"cvt.u32.u64 %r1, %rd3;","cvt_s16_s8":"cvt.s16.s8 %h1, %h2;","cvt_pack_sat_u8_s32":"cvt.pack.sat.u8.s32.b32 %r1, %r2, %r3, %r4;",
 "cvt_rn_satfinite_f16x2_f32":"cvt.rn.satfinite.f16x2.f32 %r1, %f1, %f2;","cvt_rn_relu_f16_f32":"cvt.rn.relu.f16.f32 %h1, %f1;","cvt_rs_f16x2_f32":"cvt.rs.f16x2.f32 %r1, %f1, %f2, %r3;",
 "cvt_rn_e2m1x2_f32":"cvt.rn.satfinite.e2m1x2.f32 %h1, %f1, %f2;",
}
for k, v in fl.items():
    fam = "cvt" if k.startswith("cvt") else ("half_bf16" if any(t in v for t in ("f16","bf16")) else "float_math")
    tgt = "sm_100a" if "e2m1" in k or "cvt_rs" in k else ("sm_89" if "e4m3" in k else "sm_80")
    ver = "8.6" if tgt=="sm_100a" else ("7.8" if tgt=="sm_89" else "7.0")
    probe(fam, k, [v], target=tgt, version=ver)

# ---- control flow
probe("control","bra_uni_loop",["L0:","add.u32 %r1, %r1, 1;","setp.lt.u32 %p1, %r1, 10;","@%p1 bra L0;"])
probe("control","pred_guard_neg",["@!%p1 add.u32 %r1, %r1, 1;"])
probe("control","ret_in_branch",["@%p1 ret;"])
