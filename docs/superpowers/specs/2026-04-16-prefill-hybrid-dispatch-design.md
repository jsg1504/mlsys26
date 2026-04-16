# Prefill Hybrid Dispatch: NVRTC CUDA + CuTe-DSL

Date: 2026-04-16
Target: `gdn_prefill_qk4_v8_d128_k_last` — mean latency < 0.2 ms on B200

## Problem

Current best: **0.334 ms** mean latency (CuTe-DSL with optimized wrapper).
Target: **< 0.2 ms**.

NCU profiling reveals two bottlenecks:

1. **Short sequences (T ≤ ~200, ~60% of workloads)**: CuTe-DSL kernel has a 64 μs floor cost — 227 KB shared memory allocation, TMA descriptor setup, 8-warp pipeline initialization. GPU utilization < 1%.
2. **Host-side gate overhead (~50–80 μs per call)**: 10 separate PyTorch kernel launches for gate computation before the GDN kernel.

## Solution

Dispatch to two kernel paths based on total sequence length:

| Path | Condition | Kernel | Gate Handling |
|------|-----------|--------|---------------|
| Short | `total_seq_len ≤ THRESHOLD` | NVRTC-compiled register-cached sequential CUDA | Fused inside kernel |
| Long | `total_seq_len > THRESHOLD` | CuTe-DSL chunked (existing) | NVRTC-compiled fused gate kernel |

`THRESHOLD` starts at 512, tuned via benchmark.

## Architecture

```
main.py::run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)
│
├── total_seq_len ≤ THRESHOLD
│     └── nvrtc_sequential_kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, output, new_state)
│         Gates computed inside kernel. No PyTorch ops.
│
└── total_seq_len > THRESHOLD
      ├── nvrtc_fused_gate_kernel(A_log, a, dt_bias, b) → gate_log, beta
      └── chunk_gated_delta_rule(q, k, v, gate_log, beta, state, cu_seqlens, scale)
```

### File Layout

```
solution/python/
  main.py                     # Dispatch logic + NVRTC loader
  nvrtc_kernels/
    sequential_kernel.cu      # Register-cached V-parallel GDN (fused gates)
    fused_gate_kernel.cu      # Fused gate computation (replaces 10 PyTorch kernels)
  nvrtc_loader.py             # NVRTC compile/cache/launch utilities
  gdn_blackwell/              # Existing CuTe-DSL kernel (unchanged)
  prefill_contract.py         # Existing (unused in hot path)
```

## Component Details

### 1. nvrtc_loader.py

Responsibilities:
- Compile `.cu` source to PTX via `cuda.bindings.nvrtc`
- Load PTX into `CUmodule` via `cuda.bindings.driver`
- Cache compiled modules (keyed by source hash)
- Provide `launch(kernel_name, grid, block, args, shared_mem, stream)` helper

No torch dependency in the compilation/launch path. Only uses `cuda-python`.

### 2. sequential_kernel.cu

The register-cached V-parallel kernel from the previous iteration. Key properties:
- 128 threads per block (1 per V dimension)
- State column (128 fp32) cached in registers via `__launch_bounds__(128, 1)`
- Gates fused: takes raw `A_log, a, dt_bias, b` and computes `g`, `beta` in-kernel
- Grid: `num_seqs × NUM_V_HEADS`
- No dynamic shared memory (only `__shared__ float s_k[128], s_q[128]`)

Signature:
```c
__global__ void gdn_prefill_sequential(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const float* state, const float* A_log, const __nv_bfloat16* a,
    const float* dt_bias, const __nv_bfloat16* b, const int64_t* cu_seqlens,
    float scale, __nv_bfloat16* output, float* new_state, int num_seqs);
```

### 3. fused_gate_kernel.cu

Single CUDA kernel replacing 10 PyTorch kernels:
```
gate_log[t, h] = -exp(A_log[h]) * softplus(float(a[t,h]) + dt_bias[h])
beta[t, h]     = sigmoid(float(b[t, h]))
```

Grid: `ceil(T * 8 / 256)`, Block: 256.
Simple elementwise kernel, ~5 μs for all workload sizes.

### 4. main.py Dispatch

```python
def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    T = q.shape[0]
    num_seqs = cu_seqlens.shape[0] - 1
    stream = torch.cuda.current_stream().cuda_stream

    if T <= THRESHOLD:
        output = torch.empty(T, 8, 128, dtype=torch.bfloat16, device=q.device)
        new_state = torch.empty_like(state)
        nvrtc_launch("gdn_prefill_sequential", ...)
        return output, new_state
    else:
        gate_log = torch.empty(T, 8, dtype=torch.float32, device=q.device)
        beta = torch.empty(T, 8, dtype=torch.float32, device=q.device)
        nvrtc_launch("fused_gate", ...)
        output, new_state = chunk_gated_delta_rule(...)
        return output.squeeze(0), new_state
```

Torch is used only for tensor allocation and stream management, never for computation.

## Expected Performance

| Workload bucket | Count | Current (ms) | Target (ms) | Savings |
|-----------------|-------|-------------|-------------|---------|
| Short (T ≤ 512) | ~60 | 0.17–0.22 | 0.035–0.10 | ~60% |
| Long (T > 512) | ~40 | 0.35–1.05 | 0.30–1.00 | ~5–10% |
| **Mean** | 100 | **0.334** | **~0.20** | **~40%** |

## Correctness

- Sequential CUDA kernel already verified on all 100 workloads (max atol=1.22e-4)
- CuTe-DSL path unchanged, correctness preserved
- Full 100-workload correctness sweep required after integration

## Phase 2: Chunked CUDA + wmma for Medium Sequences

If Phase 1 yields ~0.22 ms (above 0.2 ms target), add a third dispatch path:

| Path | Condition | Kernel |
|------|-----------|--------|
| Short | `T ≤ 200` | NVRTC sequential (fused gates) |
| **Medium** | `200 < T ≤ 2000` | **NVRTC chunked + wmma tensor core** |
| Long | `T > 2000` | Fused gate + CuTe-DSL |

The medium-path kernel uses:
- chunk_size = 64
- `nvcuda::wmma` (16×16×16 bf16→fp32) for K@S, Q@S, state update matmuls
- Forward substitution for intra-chunk recurrence (CUDA cores, 4×16×16 blocks)
- Register-cached state between chunks

Expected improvement: medium workloads (~22) from ~300 μs to ~150 μs → mean drops ~0.03 ms.

## Risks

1. **NVRTC compilation latency**: First call compiles PTX (~2–5 s). Mitigated by benchmark warmup.
2. **THRESHOLD tuning**: Wrong threshold hurts mean. Mitigated by benchmark-driven tuning.
3. **NVRTC kernel pointer passing**: Must correctly map torch tensor data pointers to CUDA driver API args. Off-by-one or alignment errors cause silent corruption.
4. **wmma chunked correctness**: Forward substitution and gated mask must match reference exactly. Verify against full 100-workload suite.
