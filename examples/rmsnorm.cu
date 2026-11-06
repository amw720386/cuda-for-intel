extern "C" __attribute__((global)) __attribute__((launch_bounds(128)))
void rmsnorm(const float *input, const float *weight, float *output,
             unsigned columns, float epsilon) {
    __attribute__((shared)) float scratch[128];
    unsigned lane = __nvvm_read_ptx_sreg_tid_x();
    unsigned width = __nvvm_read_ptx_sreg_ntid_x();
    unsigned row = __nvvm_read_ptx_sreg_ctaid_x();
    unsigned base = row * columns;

    float sum = 0.0f;
    for (unsigned col = lane; col < columns; col += width) {
        float x = input[base + col];
        sum = __builtin_fmaf(x, x, sum);
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
    float mean = __nvvm_div_rn_f(scratch[0], (float)columns);
    float scale = __nvvm_div_rn_f(1.0f, __nvvm_sqrt_rn_f(mean + epsilon));
    for (unsigned col = lane; col < columns; col += width) {
        output[base + col] = (input[base + col] * scale) * weight[col];
    }
}
