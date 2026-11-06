extern "C" __attribute__((global)) __attribute__((launch_bounds(128)))
void forge_reduce(const float *input, float *output, unsigned columns) {
    __attribute__((shared)) float scratch[128];
    unsigned lane = __nvvm_read_ptx_sreg_tid_x();
    unsigned width = __nvvm_read_ptx_sreg_ntid_x();
    unsigned row = __nvvm_read_ptx_sreg_ctaid_x();
    unsigned base = row * columns;

    float sum = 0.0f;
    for (unsigned col = lane; col < columns; col += width) {
        sum += input[base + col];
    }
    scratch[lane] = sum;
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
        output[row] = scratch[0];
    }
}
