extern "C" __attribute__((global))
void vector_add(const float *a, const float *b, float *out, unsigned n) {
    unsigned i = __nvvm_read_ptx_sreg_ctaid_x() * __nvvm_read_ptx_sreg_ntid_x()
               + __nvvm_read_ptx_sreg_tid_x();
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}
