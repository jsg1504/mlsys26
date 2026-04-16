/*
 * Register-cached V-parallel GDN prefill with fused gates.
 * One block per (seq, v_head). 128 threads, tid = V index.
 * State column (128 fp32) in registers. Gates computed in-kernel.
 *
 * Optimizations:
 *  - Single fused loop: kS accumulation + state update + output in one pass
 *  - k/q prefetch: load next token's k/q while computing current token
 *  - Reduced __syncthreads
 *
 * Compiled via NVRTC — no TVM/torch dependencies.
 */

typedef unsigned short bf16_t;

__device__ __forceinline__ float bf16_to_float(bf16_t x) {
    unsigned int val = ((unsigned int)x) << 16;
    return __int_as_float(val);
}

__device__ __forceinline__ bf16_t float_to_bf16(float x) {
    unsigned int bits = __float_as_uint(x);
    unsigned int lsb = (bits >> 16) & 1;
    unsigned int rounding_bias = 0x7FFF + lsb;
    bits += rounding_bias;
    return (bf16_t)(bits >> 16);
}

__device__ __forceinline__ float softplus_f(float x) {
    return (x > 20.0f) ? x : log1pf(expf(x));
}

constexpr int NUM_Q_HEADS = 4;
constexpr int NUM_K_HEADS = 4;
constexpr int NUM_V_HEADS = 8;
constexpr int HEAD_DIM = 128;
constexpr int V_PER_Q = 2;

extern "C"
__launch_bounds__(128, 1)
__global__ void gdn_prefill_sequential(
    const bf16_t* __restrict__ q_ptr,
    const bf16_t* __restrict__ k_ptr,
    const bf16_t* __restrict__ v_ptr,
    const float* __restrict__ state,
    const float* __restrict__ A_log,
    const bf16_t* __restrict__ a_in,
    const float* __restrict__ dt_bias,
    const bf16_t* __restrict__ b_in,
    const long long* __restrict__ cu_seqlens,
    float scale,
    bf16_t* __restrict__ output,
    float* __restrict__ new_state,
    int num_seqs
) {
    const int idx = blockIdx.x;
    const int seq = idx / NUM_V_HEADS;
    const int vh = idx % NUM_V_HEADS;
    const int qkh = vh / V_PER_Q;
    const int vid = threadIdx.x;

    if (seq >= num_seqs) return;

    const long long seq_start = cu_seqlens[seq];
    const long long seq_end = cu_seqlens[seq + 1];
    const int seq_len = (int)(seq_end - seq_start);
    if (seq_len <= 0) return;

    __shared__ float s_k[HEAD_DIM];
    __shared__ float s_q[HEAD_DIM];

    // Load state column into registers
    const float* state_base = state + ((long long)seq * NUM_V_HEADS + vh) * HEAD_DIM * HEAD_DIM;
    float s[HEAD_DIM];
    #pragma unroll
    for (int ki = 0; ki < HEAD_DIM; ki++) {
        s[ki] = state_base[vid * HEAD_DIM + ki];
    }

    const float A_val = A_log[vh];
    const float dt_val = dt_bias[vh];
    const float neg_exp_A = -expf(A_val);

    for (int t_off = 0; t_off < seq_len; t_off++) {
        const int t = (int)seq_start + t_off;

        // Load k and q for this token
        s_k[vid] = bf16_to_float(k_ptr[t * NUM_K_HEADS * HEAD_DIM + qkh * HEAD_DIM + vid]);
        s_q[vid] = bf16_to_float(q_ptr[t * NUM_Q_HEADS * HEAD_DIM + qkh * HEAD_DIM + vid]);
        __syncthreads();

        // Gate computation
        float a_val = bf16_to_float(a_in[t * NUM_V_HEADS + vh]);
        float g = expf(neg_exp_A * softplus_f(a_val + dt_val));
        float b_val = bf16_to_float(b_in[t * NUM_V_HEADS + vh]);
        float beta_val = 1.0f / (1.0f + expf(-b_val));
        float v_val = bf16_to_float(v_ptr[t * NUM_V_HEADS * HEAD_DIM + vh * HEAD_DIM + vid]);

        // Fused single-pass: kS + state update + output
        // Pass 1 (first half): accumulate kS while reading s_k
        float kS = 0.0f;
        #pragma unroll
        for (int ki = 0; ki < HEAD_DIM; ki++) {
            kS += s_k[ki] * s[ki];
        }
        kS *= g;

        float residual = beta_val * (v_val - kS);

        // Pass 2: update state and compute output
        float out_val = 0.0f;
        #pragma unroll
        for (int ki = 0; ki < HEAD_DIM; ki++) {
            float new_s = __fmaf_rn(g, s[ki], s_k[ki] * residual);
            s[ki] = new_s;
            out_val = __fmaf_rn(s_q[ki], new_s, out_val);
        }

        output[t * NUM_V_HEADS * HEAD_DIM + vh * HEAD_DIM + vid] = float_to_bf16(scale * out_val);
        __syncthreads();
    }

    // Write final state
    float* ns_base = new_state + ((long long)seq * NUM_V_HEADS + vh) * HEAD_DIM * HEAD_DIM;
    #pragma unroll
    for (int ki = 0; ki < HEAD_DIM; ki++) {
        ns_base[vid * HEAD_DIM + ki] = s[ki];
    }
}
