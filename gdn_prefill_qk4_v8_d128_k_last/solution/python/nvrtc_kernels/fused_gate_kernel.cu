/*
 * Fused gate computation — replaces 10 separate PyTorch kernel launches.
 * gate_log[t,h] = -exp(A_log[h]) * log1p(exp(float(a[t,h]) + dt_bias[h]))
 * beta[t,h]     = 1 / (1 + exp(-float(b[t,h])))
 */

typedef unsigned short bf16_t;

__device__ __forceinline__ float bf16_to_float(bf16_t x) {
    unsigned int val = ((unsigned int)x) << 16;
    return __int_as_float(val);
}

extern "C"
__global__ void fused_gate_kernel(
    const float* __restrict__ A_log,      // [8]
    const bf16_t* __restrict__ a_in,      // [T, 8]
    const float* __restrict__ dt_bias,    // [8]
    const bf16_t* __restrict__ b_in,      // [T, 8]
    float* __restrict__ gate_log_out,     // [T, 8]
    float* __restrict__ beta_out,         // [T, 8]
    int T,
    int H                                 // = 8
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * H;
    if (idx >= total) return;

    int h = idx % H;
    float a_val = bf16_to_float(a_in[idx]);
    float b_val = bf16_to_float(b_in[idx]);

    // gate_log = -exp(A_log[h]) * softplus(a + dt_bias[h])
    float x = a_val + dt_bias[h];
    float sp = (x > 20.0f) ? x : log1pf(expf(x));  // softplus with overflow guard
    gate_log_out[idx] = -expf(A_log[h]) * sp;

    // beta = sigmoid(b)
    beta_out[idx] = 1.0f / (1.0f + expf(-b_val));
}
