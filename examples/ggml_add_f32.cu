// Adapted from ggml/src/ggml-cuda/binbcast.cu at llama.cpp b12111.
// Copyright (c) 2023-2026 The ggml authors. See ggml-LICENSE.
// FP32, non-fused, non-null src0, one channel/sample, no broadcast.
// Keep GGML's row-stride addressing, grid-stride loop, and overflow guard.

static __attribute__((device, always_inline)) float op_add(float a, float b) {
    return a + b;
}

extern "C" __attribute__((global))
void ggml_add_f32(const float *src0, const float *src1, float *dst,
                  unsigned ne0, unsigned ne1, unsigned s01,
                  unsigned s11, unsigned s1) {
    const unsigned i0s = __nvvm_read_ptx_sreg_ntid_x() *
                         __nvvm_read_ptx_sreg_ctaid_x() + __nvvm_read_ptx_sreg_tid_x();
    const unsigned i1 = __nvvm_read_ptx_sreg_ctaid_y();

    if (i0s >= ne0 || i1 >= ne1) {
        return;
    }

    const auto i_src0 = static_cast<unsigned long long>(i1) * s01;
    const auto i_src1 = static_cast<unsigned long long>(i1) * s11;
    const auto i_dst = static_cast<unsigned long long>(i1) * s1;
    const float *src0_row = src0 + i_src0;
    float *dst_row = dst + i_dst;
    const unsigned s0 = __nvvm_read_ptx_sreg_ntid_x() * __nvvm_read_ptx_sreg_nctaid_x();

    for (unsigned i0 = i0s; i0 < ne0; i0 += s0) {
        float result = src0_row[i0];
        result = op_add(result, src1[i_src1 + i0]);
        dst_row[i0] = result;
        if (ne0 - i0 <= s0) {
            break;
        }
    }
}
