/*
 * Warp-cooperative GDN prefill with 8-thread groups.
 * Grid: num_seqs * 8 * 8 blocks (seq, v_head, v_eighth).
 * 128 threads per block, each group of 8 threads cooperates on 1 V element.
 * Each thread owns 16 fp32 state values (1/8 of the K dimension).
 *
 * Per-token: 16 FMAs for kS partial + shuffle reduce + 16 FMAs for state+output
 *            + shuffle reduce for output = ~32 FMAs + 2 butterfly reductions.
 *
 * kS and output reductions use __shfl_xor_sync with deltas 1,2,4 within
 * groups of 8 consecutive threads. XOR by 1/2/4 flips bits 0/1/2
 * — all stay within groups of 8, so full-warp mask 0xffffffff is safe.
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
constexpr int GROUP_SIZE = 16;
constexpr int K_PER_THREAD = HEAD_DIM / GROUP_SIZE;  // 8
constexpr int V_PER_BLOCK = 128 / GROUP_SIZE;        // 8
constexpr int V_QUARTERS = HEAD_DIM / V_PER_BLOCK;   // 16

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
    int num_seqs,
    const int* __restrict__ seq_idx_map  // nullable: if non-null, remaps block seq index
) {
    // Block indexing: num_seqs * NUM_V_HEADS * V_QUARTERS blocks
    const int idx = blockIdx.x;
    const int v_quarter = idx % V_QUARTERS;
    const int vh = (idx / V_QUARTERS) % NUM_V_HEADS;
    const int local_seq = idx / (NUM_V_HEADS * V_QUARTERS);
    const int qkh = vh / V_PER_Q;

    // Thread indexing within block
    const int lane = threadIdx.x % GROUP_SIZE;     // 0-7 within group
    const int vid = threadIdx.x / GROUP_SIZE;      // 0-15, V index within this eighth
    const int actual_vid = v_quarter * V_PER_BLOCK + vid;  // 0-127, global V index

    if (local_seq >= num_seqs) return;

    // Hybrid dispatch: when seq_idx_map is non-null, the grid covers a subset
    // of sequences (e.g. the "short" group), and each block's logical seq id
    // must be remapped to the actual sequence index in cu_seqlens/state/output.
    const int seq = (seq_idx_map != nullptr) ? seq_idx_map[local_seq] : local_seq;

    const long long seq_start = cu_seqlens[seq];
    const long long seq_end = cu_seqlens[seq + 1];
    const int seq_len = (int)(seq_end - seq_start);
    if (seq_len <= 0) return;

    // Load state: each thread loads 16 values from its K-dimension partition
    const float* state_base = state + ((long long)seq * NUM_V_HEADS + vh) * HEAD_DIM * HEAD_DIM;
    float s[K_PER_THREAD];
    #pragma unroll
    for (int ki = 0; ki < K_PER_THREAD; ki++) {
        s[ki] = state_base[actual_vid * HEAD_DIM + lane * K_PER_THREAD + ki];
    }

    const float A_val = A_log[vh];
    const float dt_val = dt_bias[vh];
    const float neg_exp_A = -expf(A_val);

    for (int t_off = 0; t_off < seq_len; t_off++) {
        const int t = (int)seq_start + t_off;

        // Base pointers for this token's k and q
        const bf16_t* k_token = k_ptr + t * NUM_K_HEADS * HEAD_DIM + qkh * HEAD_DIM;
        const bf16_t* q_token = q_ptr + t * NUM_Q_HEADS * HEAD_DIM + qkh * HEAD_DIM;

        // Gate computation (same for all threads in the block)
        float a_val = bf16_to_float(a_in[t * NUM_V_HEADS + vh]);
        float g = expf(neg_exp_A * softplus_f(a_val + dt_val));
        float b_val = bf16_to_float(b_in[t * NUM_V_HEADS + vh]);
        float beta_val = 1.0f / (1.0f + expf(-b_val));

        // 1. kS partial reduction: each thread does 32 FMAs over its K partition
        float partial_kS = 0.0f;
        #pragma unroll
        for (int ki = 0; ki < K_PER_THREAD; ki++) {
            partial_kS = __fmaf_rn(
                bf16_to_float(__ldg(&k_token[lane * K_PER_THREAD + ki])),
                s[ki],
                partial_kS
            );
        }

        // Butterfly reduce across 16 lanes (XOR 1,2,4,8 stays within groups of 16)
        partial_kS += __shfl_xor_sync(0xffffffff, partial_kS, 1);
        partial_kS += __shfl_xor_sync(0xffffffff, partial_kS, 2);
        partial_kS += __shfl_xor_sync(0xffffffff, partial_kS, 4);
        partial_kS += __shfl_xor_sync(0xffffffff, partial_kS, 8);
        float kS = partial_kS * g;

        // 2. Read v and broadcast from lane 0 to all lanes in the group
        float v_val;
        if (lane == 0) {
            v_val = bf16_to_float(v_ptr[t * NUM_V_HEADS * HEAD_DIM + vh * HEAD_DIM + actual_vid]);
        }
        v_val = __shfl_sync(0xffffffff, v_val, (threadIdx.x & ~15));  // broadcast from lane 0 of this group

        float residual = beta_val * (v_val - kS);

        // 3. State update (32 FMAs) + partial output accumulation (32 FMAs)
        float partial_out = 0.0f;
        #pragma unroll
        for (int ki = 0; ki < K_PER_THREAD; ki++) {
            float k_val = bf16_to_float(__ldg(&k_token[lane * K_PER_THREAD + ki]));
            float new_s = __fmaf_rn(g, s[ki], k_val * residual);
            s[ki] = new_s;
            partial_out = __fmaf_rn(
                bf16_to_float(__ldg(&q_token[lane * K_PER_THREAD + ki])),
                new_s,
                partial_out
            );
        }

        // 4. Output reduction (butterfly across 16 lanes)
        partial_out += __shfl_xor_sync(0xffffffff, partial_out, 1);
        partial_out += __shfl_xor_sync(0xffffffff, partial_out, 2);
        partial_out += __shfl_xor_sync(0xffffffff, partial_out, 4);
        partial_out += __shfl_xor_sync(0xffffffff, partial_out, 8);

        // Only lane 0 writes the output for this V element
        if (lane == 0) {
            output[t * NUM_V_HEADS * HEAD_DIM + vh * HEAD_DIM + actual_vid] =
                float_to_bf16(scale * partial_out);
        }
    }

    // Write final state: each thread stores its 32 values
    float* ns_base = new_state + ((long long)seq * NUM_V_HEADS + vh) * HEAD_DIM * HEAD_DIM;
    #pragma unroll
    for (int ki = 0; ki < K_PER_THREAD; ki++) {
        ns_base[actual_vid * HEAD_DIM + lane * K_PER_THREAD + ki] = s[ki];
    }
}
