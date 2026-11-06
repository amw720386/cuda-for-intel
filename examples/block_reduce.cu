extern "C" __attribute__((global)) __attribute__((launch_bounds(128)))
void block_reduce(const float *input, float *output, unsigned count) {
    __attribute__((shared)) float scratch[128];
    unsigned lane = __nvvm_read_ptx_sreg_tid_x();
    unsigned block = __nvvm_read_ptx_sreg_ctaid_x();
    unsigned width = __nvvm_read_ptx_sreg_ntid_x();
    unsigned i = block * width + lane;
    scratch[lane] = i < count ? input[i] : 0.0f;
    __syncthreads();
    for (unsigned active = width; active > 1;) {
        unsigned half = (active + 1) >> 1;
        if (lane < active - half) {
            scratch[lane] += scratch[lane + half];
        }
        __syncthreads();
        active = half;
    }
    if (lane == 0) {
        output[block] = scratch[0];
    }
}
