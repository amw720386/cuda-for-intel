//! Barrier-legality tests for `flow.rs` / `uniform.rs`.
//!
//! Every accepted pattern is a minimal PTX repro of a PyTorch top-k / kthvalue loop. Accepted
//! kernels also run on the multi-threaded SPIR-V simulator (`sim.rs`), which fails on a barrier
//! reached at different instructions by different invocations, so "accepted" is backed by
//! "does not deadlock and computes the right values"; rejected kernels are shown to misbehave.
use super::flow::set_strict_barriers;
use super::sim::{Gpu, Launch, Sim};
use super::translate;

fn kernel(body: &str, extra_regs: &str) -> String {
    format!(
        r#".version 8.8
.target sm_80
.address_size 64
.visible .entry k(
	.param .u64 k_param_0,
	.param .u64 k_param_1
)
{{
	.reg .pred %p<8>;
	.reg .b32 %r<24>;
	.reg .b64 %rd<24>;
{extra_regs}
	ld.param.u64 %rd1, [k_param_0];
	cvta.to.global.u64 %rd2, %rd1;
	ld.param.u64 %rd3, [k_param_1];
	mov.u32 %r2, %tid.x;
	mov.u32 %r3, %ntid.x;
{body}
	ret;
}}
"#
    )
}

/// Stores `%r7` at `out[tid]`.
const STORE: &str = "	mul.wide.u32 %rd10, %r2, 4;
	add.s64 %rd11, %rd2, %rd10;
	st.global.u32 [%rd11], %r7;";

fn run(src: &str, n: u64, block: u32) -> Result<Vec<u32>, String> {
    let t = translate(src).map_err(|e| e.to_string())?;
    let module = rspirv::dr::load_words(&t.words).unwrap();
    let mut gpu = Gpu::default();
    let out = gpu.alloc(4 * block as usize);
    let mut sim = Sim::new(&module);
    sim.run(&mut gpu, &Launch::new([1, 1, 1], [block, 1, 1], vec![out, n]))?;
    Ok(gpu.download_f32(out, block as usize).iter().map(|f| f.to_bits()).collect())
}

fn rejected(src: &str) -> bool {
    translate(src).is_err_and(|e| e.to_string().contains("barrier may execute divergently"))
}

/// `for (i = tid; i < n - n % ntid; i += ntid) { __syncthreads(); acc += i; }`, 32-bit.
fn stride32(bound: &str) -> String {
    kernel(
        &format!(
            "	cvt.u32.u64 %r1, %rd3;
	rem.u32 %r4, %r1, %r3;
	sub.s32 %r5, %r1, %r4;
	mov.u32 %r6, %r2;
	mov.u32 %r7, 0;
$L_loop:
	setp.ge.u32 %p1, %r6, {bound};
	@%p1 bra $L_done;
	bar.sync 0;
	add.u32 %r7, %r7, %r6;
	add.u32 %r6, %r6, %r3;
	bra.uni $L_loop;
$L_done:
	bar.sync 0;
{STORE}"
        ),
        "",
    )
}

fn expected_sum(tid: u32, k: u32, ntid: u32) -> u32 {
    (0..).map(|j| tid + j * ntid).take_while(|&i| i < k).sum()
}

#[test]
fn block_stride_loop_with_rounded_bound_is_uniform_and_runs() {
    set_strict_barriers(true);
    let src = stride32("%r5");
    for (n, block) in [(70u64, 32u32), (5, 32), (96, 32), (0, 64), (200, 64)] {
        let k = (n - n % u64::from(block)) as u32;
        let got = run(&src, n, block).unwrap_or_else(|e| panic!("n={n} block={block}: {e}"));
        let want: Vec<u32> = (0..block).map(|t| expected_sum(t, k, block)).collect();
        assert_eq!(got, want, "n={n} block={block}");
    }
}

#[test]
fn block_stride_loop_with_unrounded_bound_is_rejected_when_strict() {
    set_strict_barriers(true);
    // Trip counts differ between threads when n % ntid != 0: the barrier is reached by
    // different iterations, which the simulator reports as a divergent barrier.
    let src = stride32("%r1");
    assert!(rejected(&src));
    set_strict_barriers(false);
    let accepted = translate(&src).expect("trusted as in CUDA when not strict");
    let module = rspirv::dr::load_words(&accepted.words).unwrap();
    let mut gpu = Gpu::default();
    let out = gpu.alloc(4 * 32);
    let err = Sim::new(&module)
        .run(&mut gpu, &Launch::new([1, 1, 1], [32, 1, 1], vec![out, 40]))
        .expect_err("n=40 over 32 threads: threads 0..8 loop twice");
    assert!(err.contains("divergent barrier"), "{err}");
}

#[test]
fn thread_guard_on_barrier_is_rejected() {
    for strict in [false, true] {
        set_strict_barriers(strict);
        let src = kernel(
            "	mov.u32 %r7, 0;
	setp.ge.u32 %p1, %r2, 5;
	@%p1 bra $L_skip;
	bar.sync 0;
	mov.u32 %r7, 1;
$L_skip:
	bar.sync 0;"
                .to_owned()
                .as_str(),
            "",
        );
        assert!(rejected(&src), "strict={strict}");
    }
}

#[test]
fn tid_compared_with_multiple_of_ntid_is_uniform() {
    // `tid >= n - n % ntid` guards a barrier; 64-bit arithmetic, 32-bit fast path behind the
    // `n >> 32 == 0` test, exactly as nvcc emits `n % ntid` for a 64-bit `n`.
    set_strict_barriers(true);
    let src = kernel(
        "	cvt.u64.u32 %rd4, %r3;
	and.b64 %rd5, %rd3, -4294967296;
	setp.ne.s64 %p2, %rd5, 0;
	@%p2 bra $L_wide;
	cvt.u32.u64 %r10, %rd4;
	cvt.u32.u64 %r11, %rd3;
	rem.u32 %r12, %r11, %r10;
	cvt.u64.u32 %rd6, %r12;
	bra.uni $L_join;
$L_wide:
	rem.u64 %rd6, %rd3, %rd4;
$L_join:
	sub.s64 %rd7, %rd3, %rd6;
	cvt.u64.u32 %rd8, %r2;
	mov.u32 %r7, 0;
	setp.le.u64 %p1, %rd7, %rd8;
	@%p1 bra $L_skip;
	bar.sync 0;
	mov.u32 %r7, 1;
$L_skip:"
            .to_owned()
            .as_str()
            .replace("$L_skip:", &format!("$L_skip:\n{STORE}"))
            .as_str(),
        "",
    );
    for (n, block) in [(70u64, 32u32), (5, 32), (96, 32), (200, 64)] {
        let k = n - n % u64::from(block);
        let got = run(&src, n, block).unwrap_or_else(|e| panic!("n={n} block={block}: {e}"));
        // Threads run the guarded block only while tid < K; K is 0 or >= ntid.
        let want: Vec<u32> = (0..block).map(|t| u32::from(u64::from(t) < k)).collect();
        assert_eq!(got, want, "n={n} block={block}");
    }
}

#[test]
fn bound_not_a_multiple_of_ntid_is_not_uniform() {
    // `tid >= n % ntid + 1`-style guards split the block: must stay rejected.
    set_strict_barriers(true);
    let src = kernel(
        "	cvt.u32.u64 %r1, %rd3;
	rem.u32 %r4, %r1, %r3;
	add.u32 %r5, %r4, 1;
	mov.u32 %r7, 0;
	setp.ge.u32 %p1, %r2, %r5;
	@%p1 bra $L_skip;
	bar.sync 0;
$L_skip:
	bar.sync 0;",
        "",
    );
    assert!(rejected(&src));
}

#[test]
fn reused_counter_in_unrelated_divergent_loop_does_not_poison_uniform_loop() {
    // The same register is the block-stride counter of a barrier loop and, later, the counter
    // of a thread-divergent scan without barriers (gatherKthValue). Only the reaching
    // definitions matter for the barrier loop.
    set_strict_barriers(true);
    let src = kernel(
        "	cvt.u32.u64 %r1, %rd3;
	rem.u32 %r4, %r1, %r3;
	sub.s32 %r5, %r1, %r4;
	mov.u32 %r6, %r2;
	mov.u32 %r7, 0;
$L_a:
	setp.ge.u32 %p1, %r6, %r5;
	@%p1 bra $L_b;
	bar.sync 0;
	add.u32 %r7, %r7, 1;
	add.u32 %r6, %r6, %r3;
	bra.uni $L_a;
$L_b:
	mov.u32 %r6, %r2;
$L_c:
	setp.ge.u32 %p2, %r6, %r1;
	@%p2 bra $L_d;
	add.u32 %r6, %r6, %r3;
	bra.uni $L_c;
$L_d:
	bar.sync 0;"
            .to_owned()
            .replace("$L_d:\n", &format!("$L_d:\n{STORE}\n"))
            .as_str(),
        "",
    );
    for (n, block) in [(70u64, 32u32), (5, 32), (96, 32)] {
        let k = (n - n % u64::from(block)) as u32;
        let got = run(&src, n, block).unwrap_or_else(|e| panic!("n={n} block={block}: {e}"));
        let want: Vec<u32> = (0..block).map(|t| (0..).map(|j| t + j * block).take_while(|&i| i < k).count() as u32).collect();
        assert_eq!(got, want);
    }
}
