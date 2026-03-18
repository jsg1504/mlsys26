"""
Gated Delta Net Decode Triton Kernel for NVIDIA B200 (Blackwell)

Definition: gdn_decode_qk4_v8_d128_k_last

GVA (Grouped Value Attention) configuration from Qwen3-Next (TP=4):
  num_q_heads = num_k_heads = 4, num_v_heads = 8, head_dim = 128
  State layout: k-last [B, H_v, V, K]

Mathematical operation (per batch, per v_head):
  g      = exp(-exp(A_log[h]) * softplus(a[h] + dt_bias[h]))
  beta   = sigmoid(b[h])
  S_dec  = g * state                     // decay
  old_v  = k_exp . S_dec                 // retrieve from decayed state
  delta  = beta * (v - old_v)            // gated prediction error
  S_new  = S_dec + outer(k_exp, delta)   // rank-1 update
  output = scale * q_exp . S_new         // query

Parallelisation strategy:
  Grid:  (B * NUM_V_HEADS * (HEAD_DIM // BLOCK_V),)
  Each program handles BLOCK_V rows of the V dimension and all K columns.
  BLOCK_V is autotuned for optimal B200 occupancy (148 SMs).
"""

import math

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Autotuning configs targeting B200 (148 SMs, 8 TB/s HBM3e)
# For batch_size=1 with 8 v_heads:
#   BLOCK_V=4  -> 256 programs  (good SM fill)
#   BLOCK_V=8  -> 128 programs  (close to SM count)
#   BLOCK_V=16 ->  64 programs  (under-filled, but less overhead)
#   BLOCK_V=32 ->  32 programs  (large blocks)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_V": 2}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_V": 2}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_V": 2}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_V": 2}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_V": 2}, num_warps=1, num_stages=3),
        triton.Config({"BLOCK_V": 2}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_V": 4}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_V": 4}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_V": 4}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_V": 4}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_V": 4}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_V": 8}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_V": 8}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_V": 16}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_V": 32}, num_warps=4, num_stages=1),
    ],
    key=["B"],
)
@triton.jit
def _gdn_decode_kernel(
    # Input pointers
    q_ptr,
    k_ptr,
    v_ptr,
    state_ptr,
    A_log_ptr,
    a_ptr,
    dt_bias_ptr,
    b_ptr,
    scale,
    # Output pointers
    out_ptr,
    new_state_ptr,
    # Runtime size
    B,
    # Compile-time constants
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GVA_FACTOR: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Triton kernel for one block of the GDN decode step."""
    # ---- Program ID decomposition ----
    pid = tl.program_id(0)
    num_v_blocks = HEAD_DIM // BLOCK_V
    total_per_batch = NUM_V_HEADS * num_v_blocks

    batch = pid // total_per_batch
    remainder = pid % total_per_batch
    v_head = remainder // num_v_blocks
    v_block_idx = remainder % num_v_blocks

    # GVA head mapping: each q/k head covers GVA_FACTOR v heads
    qk_head = v_head // GVA_FACTOR
    v_start = v_block_idx * BLOCK_V

    # ---- Gate computation (scalar per v_head) ----
    # a: [B, NUM_V_HEADS] bf16 (reshaped from [B, 1, NUM_V_HEADS])
    a_val = tl.load(a_ptr + batch * NUM_V_HEADS + v_head).to(tl.float32)
    b_val = tl.load(b_ptr + batch * NUM_V_HEADS + v_head).to(tl.float32)
    A_log_val = tl.load(A_log_ptr + v_head)
    dt_bias_val = tl.load(dt_bias_ptr + v_head)

    # g = exp(-exp(A_log) * softplus(a + dt_bias))
    x = a_val + dt_bias_val
    # Numerically stable softplus: for x > 20, softplus(x) approx x
    sp_x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    g = tl.exp(-tl.exp(A_log_val) * sp_x)

    # beta = sigmoid(b)
    beta = tl.sigmoid(b_val)

    # ---- Load q, k vectors [HEAD_DIM] ----
    k_offsets = tl.arange(0, HEAD_DIM)
    q_base = batch * NUM_Q_HEADS * HEAD_DIM + qk_head * HEAD_DIM
    k_base = batch * NUM_K_HEADS * HEAD_DIM + qk_head * HEAD_DIM
    q_vals = tl.load(q_ptr + q_base + k_offsets).to(tl.float32)
    k_vals = tl.load(k_ptr + k_base + k_offsets).to(tl.float32)

    # ---- Load v values for this V-block [BLOCK_V] ----
    v_offsets = v_start + tl.arange(0, BLOCK_V)
    v_base = batch * NUM_V_HEADS * HEAD_DIM + v_head * HEAD_DIM
    v_vals = tl.load(v_ptr + v_base + v_offsets).to(tl.float32)

    # ---- Load state block [BLOCK_V, HEAD_DIM] ----
    # state layout: [B, NUM_V_HEADS, HEAD_DIM(V), HEAD_DIM(K)] - K contiguous
    state_base = (batch * NUM_V_HEADS + v_head) * HEAD_DIM * HEAD_DIM
    state_offsets = v_offsets[:, None] * HEAD_DIM + k_offsets[None, :]
    state_block = tl.load(state_ptr + state_base + state_offsets,
                          eviction_policy="evict_first")

    # ---- Step 1: Decay ----
    state_block = g * state_block

    # ---- Step 2: Retrieve from decayed state ----
    # old_v[v] = sum_k( k[k] * S_decayed[v, k] )
    old_v = tl.sum(state_block * k_vals[None, :], axis=1)

    # ---- Step 3: Gated prediction error ----
    delta = beta * (v_vals - old_v)

    # ---- Step 4: Rank-1 update ----
    # S_new[v, k] = S_decayed[v, k] + k[k] * delta[v]
    state_block = state_block + delta[:, None] * k_vals[None, :]

    # ---- Step 5: Query output ----
    # out[v] = scale * sum_k( q[k] * S_new[v, k] )
    out_vals = scale * tl.sum(state_block * q_vals[None, :], axis=1)

    # ---- Store output [BLOCK_V] as bf16 ----
    out_base = batch * NUM_V_HEADS * HEAD_DIM + v_head * HEAD_DIM
    tl.store(out_ptr + out_base + v_offsets, out_vals.to(tl.bfloat16))

    # ---- Store new state [BLOCK_V, HEAD_DIM] as f32 ----
    ns_base = (batch * NUM_V_HEADS + v_head) * HEAD_DIM * HEAD_DIM
    tl.store(new_state_ptr + ns_base + state_offsets, state_block,
             eviction_policy="evict_first")


def kernel(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state):
    """
    Gated Delta Net decode entry point (Destination Passing Style).

    Inputs:
        q:       [B, 1, 4, 128]   bf16  - Query tensor
        k:       [B, 1, 4, 128]   bf16  - Key tensor
        v:       [B, 1, 8, 128]   bf16  - Value tensor
        state:   [B, 8, 128, 128] f32   - Recurrent state (optional, can be None)
        A_log:   [8]              f32   - Log decay parameter (learnable)
        a:       [B, 1, 8]        bf16  - Input-dependent decay
        dt_bias: [8]              f32   - Decay bias (learnable)
        b:       [B, 1, 8]        bf16  - Update gate input
        scale:   float                  - Scale factor (1/sqrt(head_size))

    Outputs (pre-allocated):
        output:    [B, 1, 8, 128]   bf16  - Attention output
        new_state: [B, 8, 128, 128] f32   - Updated recurrent state
    """
    B_val = q.shape[0]
    if B_val == 0:
        return

    NUM_Q_HEADS = 4
    NUM_K_HEADS = 4
    NUM_V_HEADS = 8
    HEAD_DIM = 128
    GVA_FACTOR = NUM_V_HEADS // NUM_Q_HEADS  # 2

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(HEAD_DIM)

    # Handle optional state (None -> zeros)
    if state is None:
        state = torch.zeros(
            B_val, NUM_V_HEADS, HEAD_DIM, HEAD_DIM,
            dtype=torch.float32, device=q.device,
        )

    # Reshape tensors to remove the seq_len=1 dimension for flat indexing.
    # [B, 1, H, D] -> [B, H, D]  (contiguous view, no copy)
    q_flat = q.reshape(B_val, NUM_Q_HEADS, HEAD_DIM)
    k_flat = k.reshape(B_val, NUM_K_HEADS, HEAD_DIM)
    v_flat = v.reshape(B_val, NUM_V_HEADS, HEAD_DIM)
    a_flat = a.reshape(B_val, NUM_V_HEADS)
    b_flat = b.reshape(B_val, NUM_V_HEADS)
    out_flat = output.reshape(B_val, NUM_V_HEADS, HEAD_DIM)

    # Grid: one program per (batch, v_head, v_block)
    grid = lambda meta: (B_val * NUM_V_HEADS * (HEAD_DIM // meta["BLOCK_V"]),)

    _gdn_decode_kernel[grid](
        q_flat, k_flat, v_flat, state,
        A_log, a_flat, dt_bias, b_flat,
        scale,
        out_flat, new_state,
        B_val,
        NUM_Q_HEADS=NUM_Q_HEADS,
        NUM_K_HEADS=NUM_K_HEADS,
        NUM_V_HEADS=NUM_V_HEADS,
        HEAD_DIM=HEAD_DIM,
        GVA_FACTOR=GVA_FACTOR,
    )
