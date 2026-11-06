extern "C" __attribute__((global)) __attribute__((launch_bounds(128)))
void forge_residual(const float *normalized, const float *residual,
                    float *output, unsigned elements) {
    unsigned i = __nvvm_read_ptx_sreg_ctaid_x() * __nvvm_read_ptx_sreg_ntid_x()
                 + __nvvm_read_ptx_sreg_tid_x();
    if (i < elements) {
        output[i] = normalized[i] + residual[i];
    }
}
