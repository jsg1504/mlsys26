/*
 * GDN Prefill Kernel: gdn_prefill_qk4_v8_d128_k_last
 *
 * Variable-length prefill with recurrent state update.
 * State layout: [N, H=8, V=128, K=128] float32 (k-last)
 * GVA: q_heads=4, k_heads=4, v_heads=8 (2 v_heads per q/k head)
 *
 * Simplified formula (per timestep, per head):
 *   g = exp(-exp(A_log) * softplus(a + dt_bias))
 *   beta = sigmoid(b)
 *   temp = g * state
 *   residual = beta * (v - k @ temp)
 *   state_new = temp + k^T @ residual
 *   output = scale * (q @ state_new)
 *
 * Uses sequential per-token update (baseline, not yet chunked).
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/extra/c_env_api.h>

using bf16 = __nv_bfloat16;

constexpr int NUM_Q_HEADS = 4;
constexpr int NUM_K_HEADS = 4;
constexpr int NUM_V_HEADS = 8;
constexpr int HEAD_DIM = 128;
constexpr int V_PER_Q = NUM_V_HEADS / NUM_Q_HEADS;  // 2
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = HEAD_DIM / WARP_SIZE;
constexpr int DEFAULT_STATE_TILE_ROWS = 16;
constexpr int SMALL_STATE_TILE_ROWS = 8;
constexpr int TINY_STATE_TILE_ROWS = 4;
constexpr int MICRO_STATE_TILE_ROWS = 2;
constexpr int MICRO_BLOCK_THREADS = 64;
constexpr int MICRO_K_COLS_PER_LANE = HEAD_DIM / WARP_SIZE;
constexpr int LONG_ROWS_PER_WARP = DEFAULT_STATE_TILE_ROWS / NUM_WARPS;

static_assert(HEAD_DIM % WARP_SIZE == 0, "HEAD_DIM must be warp aligned");
static_assert(DEFAULT_STATE_TILE_ROWS % NUM_WARPS == 0,
              "Default state tile must divide evenly across warps");

__device__ __forceinline__ float softplus(float x) {
    return log1pf(expf(x));
}

/*
 * One block per (seq_idx, v_head, state_row_tile) triple.
 * Grid: (num_seqs * NUM_V_HEADS, HEAD_DIM / STATE_TILE_ROWS)
 * Block: (128,) — one thread per K dimension
 */
template <int STATE_TILE_ROWS>
__global__ void gdn_prefill_kernel(
    const bf16* __restrict__ q,         // [T, 4, 128]
    const bf16* __restrict__ k,         // [T, 4, 128]
    const bf16* __restrict__ v,         // [T, 8, 128]
    const float* __restrict__ state,    // [N, 8, 128, 128] k-last
    const float* __restrict__ A_log,    // [8]
    const bf16* __restrict__ a,         // [T, 8]
    const float* __restrict__ dt_bias,  // [8]
    const bf16* __restrict__ b_gate,    // [T, 8]
    const int64_t* __restrict__ cu_seqlens, // [N+1]
    const float scale,
    bf16* __restrict__ output,          // [T, 8, 128]
    float* __restrict__ new_state,      // [N, 8, 128, 128]
    int num_seqs
) {
    static_assert(HEAD_DIM % STATE_TILE_ROWS == 0,
                  "HEAD_DIM must be divisible by STATE_TILE_ROWS");
    const int idx = blockIdx.x;
    const int row_tile = blockIdx.y;
    const int seq = idx / NUM_V_HEADS;
    const int vh = idx % NUM_V_HEADS;
    const int qkh = vh / V_PER_Q;
    const int tid = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp_idx = tid / WARP_SIZE;
    const int vi_base = row_tile * STATE_TILE_ROWS;

    if (seq >= num_seqs) return;

    const int64_t seq_start = cu_seqlens[seq];
    const int64_t seq_end = cu_seqlens[seq + 1];
    const int seq_len = (int)(seq_end - seq_start);
    if (seq_len <= 0) return;

    const float* state_base = state + (seq * NUM_V_HEADS + vh) * HEAD_DIM * HEAD_DIM;
    float* new_state_base = new_state + (seq * NUM_V_HEADS + vh) * HEAD_DIM * HEAD_DIM;

    __shared__ float s_state[STATE_TILE_ROWS][HEAD_DIM];
    __shared__ float s_reduce[NUM_WARPS][STATE_TILE_ROWS];
    __shared__ float s_kdot_tile[STATE_TILE_ROWS];
    __shared__ float s_v_tile[STATE_TILE_ROWS];
    __shared__ float s_g;
    __shared__ float s_beta;

    #pragma unroll
    for (int r = 0; r < STATE_TILE_ROWS; r++) {
        const int vi = vi_base + r;
        s_state[r][tid] = state_base[vi * HEAD_DIM + tid];
    }
    __syncthreads();

    float A_exp = 0.0f;
    float dt_val = 0.0f;
    if (tid == 0) {
        A_exp = expf(A_log[vh]);
        dt_val = dt_bias[vh];
    }

    for (int t_offset = 0; t_offset < seq_len; t_offset++) {
        int t = (int)seq_start + t_offset;

        float k_val = __bfloat162float(k[t * NUM_K_HEADS * HEAD_DIM + qkh * HEAD_DIM + tid]);
        float q_val = __bfloat162float(q[t * NUM_Q_HEADS * HEAD_DIM + qkh * HEAD_DIM + tid]);
        if (tid == 0) {
            float a_val = __bfloat162float(a[t * NUM_V_HEADS + vh]);
            float b_val = __bfloat162float(b_gate[t * NUM_V_HEADS + vh]);
            s_g = expf(-A_exp * softplus(a_val + dt_val));
            s_beta = 1.0f / (1.0f + expf(-b_val));
        }
        if (tid < STATE_TILE_ROWS) {
            s_v_tile[tid] = __bfloat162float(
                v[t * NUM_V_HEADS * HEAD_DIM + vh * HEAD_DIM + vi_base + tid]);
        }
        __syncthreads();

        #pragma unroll
        for (int r = 0; r < STATE_TILE_ROWS; r++) {
            float temp_val = s_g * s_state[r][tid];
            s_state[r][tid] = temp_val;
            float partial = k_val * temp_val;

            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                partial += __shfl_down_sync(0xffffffff, partial, offset);

            if (lane == 0)
                s_reduce[warp_idx][r] = partial;
        }
        __syncthreads();

        if (tid == 0) {
            #pragma unroll
            for (int r = 0; r < STATE_TILE_ROWS; r++) {
                float sum = 0.0f;
                #pragma unroll
                for (int warp = 0; warp < NUM_WARPS; warp++) {
                    sum += s_reduce[warp][r];
                }
                s_kdot_tile[r] = sum;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int r = 0; r < STATE_TILE_ROWS; r++) {
            const int vi = vi_base + r;
            float residual = s_beta * (s_v_tile[r] - s_kdot_tile[r]);
            float new_s = s_state[r][tid] + k_val * residual;
            s_state[r][tid] = new_s;

            float partial = q_val * new_s;
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                partial += __shfl_down_sync(0xffffffff, partial, offset);

            if (lane == 0)
                s_reduce[warp_idx][r] = partial;
        }
        __syncthreads();

        if (tid == 0) {
            #pragma unroll
            for (int r = 0; r < STATE_TILE_ROWS; r++) {
                const int vi = vi_base + r;
                float sum = 0.0f;
                #pragma unroll
                for (int warp = 0; warp < NUM_WARPS; warp++) {
                    sum += s_reduce[warp][r];
                }
                output[t * NUM_V_HEADS * HEAD_DIM + vh * HEAD_DIM + vi] =
                    __float2bfloat16(scale * sum);
            }
        }
        // The next timestep synchronizes before reusing shared scalars and row tiles.
    }

    // Write final state back
    #pragma unroll
    for (int r = 0; r < STATE_TILE_ROWS; r++) {
        const int vi = vi_base + r;
        new_state_base[vi * HEAD_DIM + tid] = s_state[r][tid];
    }
}

/*
 * Specialized 16-row long-path kernel.
 * Each warp owns every NUM_WARPS-th row in the tile so the per-row dot products
 * stay warp-local instead of serializing cross-warp reductions through tid == 0.
 */
__global__ void gdn_prefill_kernel_long(
    const bf16* __restrict__ q,         // [T, 4, 128]
    const bf16* __restrict__ k,         // [T, 4, 128]
    const bf16* __restrict__ v,         // [T, 8, 128]
    const float* __restrict__ state,    // [N, 8, 128, 128] k-last
    const float* __restrict__ A_log,    // [8]
    const bf16* __restrict__ a,         // [T, 8]
    const float* __restrict__ dt_bias,  // [8]
    const bf16* __restrict__ b_gate,    // [T, 8]
    const int64_t* __restrict__ cu_seqlens, // [N+1]
    const float scale,
    bf16* __restrict__ output,          // [T, 8, 128]
    float* __restrict__ new_state,      // [N, 8, 128, 128]
    int num_seqs
) {
    static_assert(DEFAULT_STATE_TILE_ROWS == 16, "Long kernel assumes 16-row tiles");
    static_assert(LONG_ROWS_PER_WARP == 4, "Long kernel assumes four rows per warp");

    const int idx = blockIdx.x;
    const int row_tile = blockIdx.y;
    const int seq = idx / NUM_V_HEADS;
    const int vh = idx % NUM_V_HEADS;
    const int qkh = vh / V_PER_Q;
    const int tid = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp_idx = tid / WARP_SIZE;
    const int vi_base = row_tile * DEFAULT_STATE_TILE_ROWS;
    const int k_col_base = lane * MICRO_K_COLS_PER_LANE;
    constexpr unsigned int kWarpMask = 0xffffffffu;

    if (seq >= num_seqs) return;

    const int64_t seq_start = cu_seqlens[seq];
    const int64_t seq_end = cu_seqlens[seq + 1];
    const int seq_len = (int)(seq_end - seq_start);
    if (seq_len <= 0) return;

    const float* state_base = state + (seq * NUM_V_HEADS + vh) * HEAD_DIM * HEAD_DIM;
    float* new_state_base = new_state + (seq * NUM_V_HEADS + vh) * HEAD_DIM * HEAD_DIM;

    __shared__ float s_state[DEFAULT_STATE_TILE_ROWS][HEAD_DIM];
    __shared__ float s_q[HEAD_DIM];
    __shared__ float s_k[HEAD_DIM];
    __shared__ float s_g;
    __shared__ float s_beta;

    #pragma unroll
    for (int r = 0; r < DEFAULT_STATE_TILE_ROWS; r++) {
        const int vi = vi_base + r;
        s_state[r][tid] = state_base[vi * HEAD_DIM + tid];
    }
    __syncthreads();

    float A_exp = 0.0f;
    float dt_val = 0.0f;
    if (tid == 0) {
        A_exp = expf(A_log[vh]);
        dt_val = dt_bias[vh];
    }

    for (int t_offset = 0; t_offset < seq_len; t_offset++) {
        if (t_offset > 0) {
            __syncthreads();
        }

        const int t = (int)seq_start + t_offset;
        const int qk_base = t * NUM_K_HEADS * HEAD_DIM + qkh * HEAD_DIM;
        const int q_base = t * NUM_Q_HEADS * HEAD_DIM + qkh * HEAD_DIM;
        s_k[tid] = __bfloat162float(k[qk_base + tid]);
        s_q[tid] = __bfloat162float(q[q_base + tid]);
        if (tid == 0) {
            const float a_val = __bfloat162float(a[t * NUM_V_HEADS + vh]);
            const float b_val = __bfloat162float(b_gate[t * NUM_V_HEADS + vh]);
            s_g = expf(-A_exp * softplus(a_val + dt_val));
            s_beta = 1.0f / (1.0f + expf(-b_val));
        }
        __syncthreads();

        #pragma unroll
        for (int group = 0; group < LONG_ROWS_PER_WARP; group++) {
            const int r = warp_idx + group * NUM_WARPS;
            const int vi = vi_base + r;
            float state_vals[MICRO_K_COLS_PER_LANE];
            float kdot_partial = 0.0f;

            #pragma unroll
            for (int c = 0; c < MICRO_K_COLS_PER_LANE; c++) {
                const int col = k_col_base + c;
                const float temp_val = s_g * s_state[r][col];
                state_vals[c] = temp_val;
                kdot_partial += s_k[col] * temp_val;
            }

            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                kdot_partial += __shfl_down_sync(kWarpMask, kdot_partial, offset);
            }

            float residual = 0.0f;
            if (lane == 0) {
                const float v_val =
                    __bfloat162float(v[t * NUM_V_HEADS * HEAD_DIM + vh * HEAD_DIM + vi]);
                residual = s_beta * (v_val - kdot_partial);
            }
            residual = __shfl_sync(kWarpMask, residual, 0);

            float output_partial = 0.0f;
            #pragma unroll
            for (int c = 0; c < MICRO_K_COLS_PER_LANE; c++) {
                const int col = k_col_base + c;
                const float new_s = state_vals[c] + s_k[col] * residual;
                s_state[r][col] = new_s;
                output_partial += s_q[col] * new_s;
            }

            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                output_partial += __shfl_down_sync(kWarpMask, output_partial, offset);
            }

            if (lane == 0) {
                output[t * NUM_V_HEADS * HEAD_DIM + vh * HEAD_DIM + vi] =
                    __float2bfloat16(scale * output_partial);
            }
        }
    }

    __syncthreads();

    #pragma unroll
    for (int r = 0; r < DEFAULT_STATE_TILE_ROWS; r++) {
        const int vi = vi_base + r;
        new_state_base[vi * HEAD_DIM + tid] = s_state[r][tid];
    }
}

/*
 * Specialized micro-tile kernel for the live 2-row path.
 * Each warp owns one row and reduces across its 128 K columns with four values per lane,
 * eliminating the cross-warp row reductions and CTA barriers inside the timestep loop.
 */
__global__ void gdn_prefill_kernel_micro(
    const bf16* __restrict__ q,         // [T, 4, 128]
    const bf16* __restrict__ k,         // [T, 4, 128]
    const bf16* __restrict__ v,         // [T, 8, 128]
    const float* __restrict__ state,    // [N, 8, 128, 128] k-last
    const float* __restrict__ A_log,    // [8]
    const bf16* __restrict__ a,         // [T, 8]
    const float* __restrict__ dt_bias,  // [8]
    const bf16* __restrict__ b_gate,    // [T, 8]
    const int64_t* __restrict__ cu_seqlens, // [N+1]
    const float scale,
    bf16* __restrict__ output,          // [T, 8, 128]
    float* __restrict__ new_state,      // [N, 8, 128, 128]
    int num_seqs
) {
    static_assert(MICRO_STATE_TILE_ROWS == 2, "Micro kernel assumes 2-row tiles");
    static_assert(MICRO_K_COLS_PER_LANE == 4, "Micro kernel assumes four K columns per lane");

    const int idx = blockIdx.x;
    const int row_tile = blockIdx.y;
    const int seq = idx / NUM_V_HEADS;
    const int vh = idx % NUM_V_HEADS;
    const int qkh = vh / V_PER_Q;
    const int tid = threadIdx.x;
    const int warp_idx = tid / WARP_SIZE;
    const int lane = tid & (WARP_SIZE - 1);
    const int vi = row_tile * MICRO_STATE_TILE_ROWS + warp_idx;
    const int k_col_base = lane * MICRO_K_COLS_PER_LANE;
    constexpr unsigned int kWarpMask = 0xffffffffu;

    if (seq >= num_seqs || warp_idx >= MICRO_STATE_TILE_ROWS) return;

    const int64_t seq_start = cu_seqlens[seq];
    const int64_t seq_end = cu_seqlens[seq + 1];
    const int seq_len = (int)(seq_end - seq_start);
    if (seq_len <= 0) return;

    const float* state_row =
        state + ((seq * NUM_V_HEADS + vh) * HEAD_DIM + vi) * HEAD_DIM;
    float* new_state_row =
        new_state + ((seq * NUM_V_HEADS + vh) * HEAD_DIM + vi) * HEAD_DIM;

    float state_vals[MICRO_K_COLS_PER_LANE];
    #pragma unroll
    for (int c = 0; c < MICRO_K_COLS_PER_LANE; c++) {
        state_vals[c] = state_row[k_col_base + c];
    }

    float A_exp = 0.0f;
    float dt_val = 0.0f;
    if (lane == 0) {
        A_exp = expf(A_log[vh]);
        dt_val = dt_bias[vh];
    }
    A_exp = __shfl_sync(kWarpMask, A_exp, 0);
    dt_val = __shfl_sync(kWarpMask, dt_val, 0);

    for (int t_offset = 0; t_offset < seq_len; t_offset++) {
        const int t = (int)seq_start + t_offset;
        const int qk_base = t * NUM_K_HEADS * HEAD_DIM + qkh * HEAD_DIM + k_col_base;
        const int q_base = t * NUM_Q_HEADS * HEAD_DIM + qkh * HEAD_DIM + k_col_base;

        float g = 0.0f;
        float beta = 0.0f;
        if (lane == 0) {
            const float a_val = __bfloat162float(a[t * NUM_V_HEADS + vh]);
            const float b_val = __bfloat162float(b_gate[t * NUM_V_HEADS + vh]);
            g = expf(-A_exp * softplus(a_val + dt_val));
            beta = 1.0f / (1.0f + expf(-b_val));
        }
        g = __shfl_sync(kWarpMask, g, 0);
        beta = __shfl_sync(kWarpMask, beta, 0);

        float k_vals[MICRO_K_COLS_PER_LANE];
        float q_vals[MICRO_K_COLS_PER_LANE];
        float kdot_partial = 0.0f;
        #pragma unroll
        for (int c = 0; c < MICRO_K_COLS_PER_LANE; c++) {
            k_vals[c] = __bfloat162float(k[qk_base + c]);
            q_vals[c] = __bfloat162float(q[q_base + c]);
            state_vals[c] *= g;
            kdot_partial += k_vals[c] * state_vals[c];
        }

        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            kdot_partial += __shfl_down_sync(kWarpMask, kdot_partial, offset);
        }

        float residual = 0.0f;
        if (lane == 0) {
            const float v_val =
                __bfloat162float(v[t * NUM_V_HEADS * HEAD_DIM + vh * HEAD_DIM + vi]);
            residual = beta * (v_val - kdot_partial);
        }
        residual = __shfl_sync(kWarpMask, residual, 0);

        float output_partial = 0.0f;
        #pragma unroll
        for (int c = 0; c < MICRO_K_COLS_PER_LANE; c++) {
            state_vals[c] += k_vals[c] * residual;
            output_partial += q_vals[c] * state_vals[c];
        }

        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            output_partial += __shfl_down_sync(kWarpMask, output_partial, offset);
        }

        if (lane == 0) {
            output[t * NUM_V_HEADS * HEAD_DIM + vh * HEAD_DIM + vi] =
                __float2bfloat16(scale * output_partial);
        }
    }

    #pragma unroll
    for (int c = 0; c < MICRO_K_COLS_PER_LANE; c++) {
        new_state_row[k_col_base + c] = state_vals[c];
    }
}

template <int STATE_TILE_ROWS>
void launch_gdn_prefill_kernel(
    cudaStream_t stream,
    const bf16* q,
    const bf16* k,
    const bf16* v,
    const float* state,
    const float* A_log,
    const bf16* a,
    const float* dt_bias,
    const bf16* b_gate,
    const int64_t* cu_seqlens,
    float scale,
    bf16* output,
    float* new_state,
    int num_seqs
) {
    dim3 grid(num_seqs * NUM_V_HEADS, HEAD_DIM / STATE_TILE_ROWS);
    dim3 block(HEAD_DIM);
    gdn_prefill_kernel<STATE_TILE_ROWS><<<grid, block, 0, stream>>>(
        q, k, v, state, A_log, a, dt_bias, b_gate, cu_seqlens, scale, output, new_state, num_seqs);
}

void launch_gdn_prefill_kernel_micro(
    cudaStream_t stream,
    const bf16* q,
    const bf16* k,
    const bf16* v,
    const float* state,
    const float* A_log,
    const bf16* a,
    const float* dt_bias,
    const bf16* b_gate,
    const int64_t* cu_seqlens,
    float scale,
    bf16* output,
    float* new_state,
    int num_seqs
) {
    dim3 grid(num_seqs * NUM_V_HEADS, HEAD_DIM / MICRO_STATE_TILE_ROWS);
    dim3 block(MICRO_BLOCK_THREADS);
    gdn_prefill_kernel_micro<<<grid, block, 0, stream>>>(
        q, k, v, state, A_log, a, dt_bias, b_gate, cu_seqlens, scale, output, new_state, num_seqs);
}

void launch_gdn_prefill_kernel_long(
    cudaStream_t stream,
    const bf16* q,
    const bf16* k,
    const bf16* v,
    const float* state,
    const float* A_log,
    const bf16* a,
    const float* dt_bias,
    const bf16* b_gate,
    const int64_t* cu_seqlens,
    float scale,
    bf16* output,
    float* new_state,
    int num_seqs
) {
    dim3 grid(num_seqs * NUM_V_HEADS, HEAD_DIM / DEFAULT_STATE_TILE_ROWS);
    dim3 block(HEAD_DIM);
    gdn_prefill_kernel_long<<<grid, block, 0, stream>>>(
        q, k, v, state, A_log, a, dt_bias, b_gate, cu_seqlens, scale, output, new_state, num_seqs);
}

// TVM FFI entry point (DPS style)
void gdn_prefill(
    tvm::ffi::TensorView q,         // [T, 4, 128] bf16
    tvm::ffi::TensorView k,         // [T, 4, 128] bf16
    tvm::ffi::TensorView v,         // [T, 8, 128] bf16
    tvm::ffi::TensorView state,     // [N, 8, 128, 128] f32
    tvm::ffi::TensorView A_log,     // [8] f32
    tvm::ffi::TensorView a,         // [T, 8] bf16
    tvm::ffi::TensorView dt_bias,   // [8] f32
    tvm::ffi::TensorView b_gate,    // [T, 8] bf16
    tvm::ffi::TensorView cu_seqlens,// [N+1] i64
    double scale,
    tvm::ffi::TensorView output,    // [T, 8, 128] bf16
    tvm::ffi::TensorView new_state  // [N, 8, 128, 128] f32
) {
    int num_seqs = cu_seqlens.size(0) - 1;

    DLDevice dev = q.device();
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(dev.device_type, dev.device_id));

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev.device_id);

    const bf16* q_ptr = static_cast<const bf16*>(q.data_ptr());
    const bf16* k_ptr = static_cast<const bf16*>(k.data_ptr());
    const bf16* v_ptr = static_cast<const bf16*>(v.data_ptr());
    const float* state_ptr = static_cast<const float*>(state.data_ptr());
    const float* A_log_ptr = static_cast<const float*>(A_log.data_ptr());
    const bf16* a_ptr = static_cast<const bf16*>(a.data_ptr());
    const float* dt_bias_ptr = static_cast<const float*>(dt_bias.data_ptr());
    const bf16* b_gate_ptr = static_cast<const bf16*>(b_gate.data_ptr());
    const int64_t* cu_seqlens_ptr = static_cast<const int64_t*>(cu_seqlens.data_ptr());
    bf16* output_ptr = static_cast<bf16*>(output.data_ptr());
    float* new_state_ptr = static_cast<float*>(new_state.data_ptr());

    // Keep the low-N path dense enough to hide CTA barriers, then fall back to the
    // larger row tiles once the grid is no longer severely underfilled.
    const int default_grid_blocks =
        num_seqs * NUM_V_HEADS * (HEAD_DIM / DEFAULT_STATE_TILE_ROWS);
    const int small_grid_blocks =
        num_seqs * NUM_V_HEADS * (HEAD_DIM / SMALL_STATE_TILE_ROWS);
    const int tiny_grid_blocks =
        num_seqs * NUM_V_HEADS * (HEAD_DIM / TINY_STATE_TILE_ROWS);
    if (tiny_grid_blocks < (4 * sm_count)) {
        launch_gdn_prefill_kernel_micro(
            stream,
            q_ptr,
            k_ptr,
            v_ptr,
            state_ptr,
            A_log_ptr,
            a_ptr,
            dt_bias_ptr,
            b_gate_ptr,
            cu_seqlens_ptr,
            static_cast<float>(scale),
            output_ptr,
            new_state_ptr,
            num_seqs);
    } else if (small_grid_blocks < (2 * sm_count)) {
        launch_gdn_prefill_kernel<TINY_STATE_TILE_ROWS>(
            stream,
            q_ptr,
            k_ptr,
            v_ptr,
            state_ptr,
            A_log_ptr,
            a_ptr,
            dt_bias_ptr,
            b_gate_ptr,
            cu_seqlens_ptr,
            static_cast<float>(scale),
            output_ptr,
            new_state_ptr,
            num_seqs);
    } else if (default_grid_blocks < sm_count) {
        launch_gdn_prefill_kernel<SMALL_STATE_TILE_ROWS>(
            stream,
            q_ptr,
            k_ptr,
            v_ptr,
            state_ptr,
            A_log_ptr,
            a_ptr,
            dt_bias_ptr,
            b_gate_ptr,
            cu_seqlens_ptr,
            static_cast<float>(scale),
            output_ptr,
            new_state_ptr,
            num_seqs);
    } else {
        launch_gdn_prefill_kernel_long(
            stream,
            q_ptr,
            k_ptr,
            v_ptr,
            state_ptr,
            A_log_ptr,
            a_ptr,
            dt_bias_ptr,
            b_gate_ptr,
            cu_seqlens_ptr,
            static_cast<float>(scale),
            output_ptr,
            new_state_ptr,
            num_seqs);
    }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel, gdn_prefill);
