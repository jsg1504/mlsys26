# Prefill Hybrid Dispatch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce prefill mean latency from 0.334 ms to < 0.2 ms by dispatching short sequences to a lightweight NVRTC CUDA kernel and replacing PyTorch gate ops with a fused CUDA kernel.

**Architecture:** Two-path dispatch from Python entry point. Short sequences (T <= threshold) use an NVRTC-compiled register-cached sequential CUDA kernel with fused gates. Long sequences use an NVRTC fused gate kernel followed by the existing CuTe-DSL chunked kernel. All CUDA kernels compiled via `cuda-python` NVRTC at first call, cached thereafter.

**Tech Stack:** cuda-python (NVRTC + driver API), CuTe-DSL (existing), CUDA C (kernels), Python (dispatch)

---

### Task 1: Create nvrtc_loader.py — NVRTC Compile/Cache/Launch Utility

**Files:**
- Create: `gdn_prefill_qk4_v8_d128_k_last/solution/python/nvrtc_loader.py`

- [ ] **Step 1: Write nvrtc_loader.py**

```python
"""NVRTC compile, cache, and launch utilities for custom CUDA kernels."""
import ctypes
import hashlib
from pathlib import Path
from typing import Dict, Tuple

import cuda.bindings.driver as drv
import cuda.bindings.nvrtc as nvrtc

_module_cache: Dict[str, Tuple] = {}  # hash -> (CUmodule, {name: CUfunction})

def _check(err):
    """Raise on CUDA/NVRTC error."""
    if isinstance(err, tuple):
        err = err[0]
    if err != 0:
        raise RuntimeError(f"CUDA/NVRTC error: {err}")

def compile_and_load(source: str, kernel_names: list[str], arch: str = "sm_100a") -> dict:
    """Compile CUDA source via NVRTC and return {name: CUfunction} dict.
    
    Results are cached by source hash. Thread-safe for single-threaded Python.
    """
    src_hash = hashlib.md5(source.encode()).hexdigest()
    if src_hash in _module_cache:
        return _module_cache[src_hash][1]

    # Compile to PTX
    err, prog = nvrtc.nvrtcCreateProgram(source.encode(), b"kernel.cu", 0, [], [])
    _check(err)
    
    opts = [f"--gpu-architecture={arch}".encode(), b"--use_fast_math"]
    compile_err = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    if compile_err[0] != 0:
        _, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * log_size
        nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(f"NVRTC compile failed:\n{log.decode()}")

    _, ptx_size = nvrtc.nvrtcGetPTXSize(prog)
    ptx = b" " * ptx_size
    _check(nvrtc.nvrtcGetPTX(prog, ptx))
    nvrtc.nvrtcDestroyProgram(prog)

    # Load module
    err, module = drv.cuModuleLoadData(ptx)
    _check(err)

    # Get kernel functions
    functions = {}
    for name in kernel_names:
        err, func = drv.cuModuleGetFunction(module, name.encode())
        _check(err)
        functions[name] = func

    _module_cache[src_hash] = (module, functions)
    return functions


def launch(func, grid: tuple, block: tuple, args: list,
           shared_mem: int = 0, stream=None):
    """Launch a CUDA kernel via the driver API.
    
    Args:
        func: CUfunction from compile_and_load
        grid: (gx, gy, gz) grid dimensions
        block: (bx, by, bz) block dimensions
        args: list of kernel arguments (int, float, or ctypes pointer values)
        shared_mem: dynamic shared memory bytes
        stream: CUstream (0 for default)
    """
    # Pack arguments as void** array
    arg_ptrs = []
    arg_values = []
    for arg in args:
        if isinstance(arg, int):
            val = ctypes.c_int64(arg)
        elif isinstance(arg, float):
            val = ctypes.c_float(arg)
        else:
            # Assume it's already a ctypes value or pointer
            val = arg
        arg_values.append(val)
        arg_ptrs.append(ctypes.cast(ctypes.pointer(val), ctypes.c_void_p))

    arr = (ctypes.c_void_p * len(arg_ptrs))(*arg_ptrs)

    cu_stream = stream if stream is not None else drv.CUstream(0)

    _check(drv.cuLaunchKernel(
        func,
        grid[0], grid[1], grid[2],
        block[0], block[1], block[2],
        shared_mem,
        cu_stream,
        arr, 0
    ))
```

- [ ] **Step 2: Verify import works**

```bash
cd gdn_prefill_qk4_v8_d128_k_last/solution/python && python -c "import nvrtc_loader; print('OK')"
```
Expected: `OK` (no import errors)

- [ ] **Step 3: Commit**

```bash
git add gdn_prefill_qk4_v8_d128_k_last/solution/python/nvrtc_loader.py
git commit -m "feat(prefill): add NVRTC compile/cache/launch utility"
```

---

### Task 2: Create fused_gate_kernel.cu

**Files:**
- Create: `gdn_prefill_qk4_v8_d128_k_last/solution/python/nvrtc_kernels/fused_gate_kernel.cu`

- [ ] **Step 1: Write fused_gate_kernel.cu**

```cuda
/*
 * Fused gate computation — replaces 10 separate PyTorch kernel launches.
 * gate_log[t,h] = -exp(A_log[h]) * log1p(exp(float(a[t,h]) + dt_bias[h]))
 * beta[t,h]     = 1 / (1 + exp(-float(b[t,h])))
 */

// Note: NVRTC doesn't have cuda_bf16.h in include path by default.
// We define bf16 conversion manually.
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
```

- [ ] **Step 2: Create `nvrtc_kernels/` directory and `__init__.py`**

```bash
mkdir -p gdn_prefill_qk4_v8_d128_k_last/solution/python/nvrtc_kernels
touch gdn_prefill_qk4_v8_d128_k_last/solution/python/nvrtc_kernels/__init__.py
```

- [ ] **Step 3: Commit**

```bash
git add gdn_prefill_qk4_v8_d128_k_last/solution/python/nvrtc_kernels/
git commit -m "feat(prefill): add fused gate CUDA kernel for NVRTC"
```

---

### Task 3: Create sequential_kernel.cu

**Files:**
- Create: `gdn_prefill_qk4_v8_d128_k_last/solution/python/nvrtc_kernels/sequential_kernel.cu`

- [ ] **Step 1: Write sequential_kernel.cu**

The register-cached V-parallel kernel with fused gates. This is the proven kernel from the earlier iteration, adapted for NVRTC (no TVM FFI headers, uses `extern "C"`).

```cuda
/*
 * Register-cached V-parallel GDN prefill with fused gates.
 * One block per (seq, v_head). 128 threads, tid = V index.
 * State column (128 fp32) in registers. Gates computed in-kernel.
 *
 * Compiled via NVRTC — no TVM/torch dependencies.
 */

typedef unsigned short bf16_t;

__device__ __forceinline__ float bf16_to_float(bf16_t x) {
    unsigned int val = ((unsigned int)x) << 16;
    return __int_as_float(val);
}

__device__ __forceinline__ bf16_t float_to_bf16(float x) {
    // Round to nearest even
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

    // Load state column into registers: s[ki] = state[seq][vh][vid][ki]
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

        s_k[vid] = bf16_to_float(k_ptr[t * NUM_K_HEADS * HEAD_DIM + qkh * HEAD_DIM + vid]);
        s_q[vid] = bf16_to_float(q_ptr[t * NUM_Q_HEADS * HEAD_DIM + qkh * HEAD_DIM + vid]);
        __syncthreads();

        float a_val = bf16_to_float(a_in[t * NUM_V_HEADS + vh]);
        float g = expf(neg_exp_A * softplus_f(a_val + dt_val));
        float b_val = bf16_to_float(b_in[t * NUM_V_HEADS + vh]);
        float beta_val = 1.0f / (1.0f + expf(-b_val));
        float v_val = bf16_to_float(v_ptr[t * NUM_V_HEADS * HEAD_DIM + vh * HEAD_DIM + vid]);

        float kS = 0.0f;
        #pragma unroll
        for (int ki = 0; ki < HEAD_DIM; ki++) {
            kS += s_k[ki] * s[ki];
        }
        kS *= g;

        float residual = beta_val * (v_val - kS);

        float out_val = 0.0f;
        #pragma unroll
        for (int ki = 0; ki < HEAD_DIM; ki++) {
            float new_s = g * s[ki] + s_k[ki] * residual;
            s[ki] = new_s;
            out_val += s_q[ki] * new_s;
        }

        output[t * NUM_V_HEADS * HEAD_DIM + vh * HEAD_DIM + vid] = float_to_bf16(scale * out_val);
        __syncthreads();
    }

    // Write final state: new_state[seq][vh][vid][ki] = s[ki]
    float* ns_base = new_state + ((long long)seq * NUM_V_HEADS + vh) * HEAD_DIM * HEAD_DIM;
    #pragma unroll
    for (int ki = 0; ki < HEAD_DIM; ki++) {
        ns_base[vid * HEAD_DIM + ki] = s[ki];
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add gdn_prefill_qk4_v8_d128_k_last/solution/python/nvrtc_kernels/sequential_kernel.cu
git commit -m "feat(prefill): add register-cached sequential CUDA kernel for NVRTC"
```

---

### Task 4: Rewrite main.py with Hybrid Dispatch

**Files:**
- Modify: `gdn_prefill_qk4_v8_d128_k_last/solution/python/main.py`

- [ ] **Step 1: Write the new main.py**

```python
"""GDN prefill entry point — hybrid dispatch.

Short sequences: NVRTC register-cached sequential CUDA kernel (fused gates).
Long sequences:  NVRTC fused gate kernel + CuTe-DSL chunked kernel.
"""
import ctypes
from pathlib import Path

import torch
import cuda.bindings.driver as drv

try:
    from .nvrtc_loader import compile_and_load, launch
    from .gdn_blackwell import chunk_gated_delta_rule
    from .gdn_blackwell.dispatch import choose_path
except ImportError:
    from nvrtc_loader import compile_and_load, launch
    from gdn_blackwell import chunk_gated_delta_rule
    from gdn_blackwell.dispatch import choose_path

THRESHOLD = 512  # total_seq_len threshold for dispatch
_kernels = {}    # lazy-compiled kernel cache

def _get_kernel_dir() -> Path:
    return Path(__file__).parent / "nvrtc_kernels"

def _ensure_kernels():
    """Compile NVRTC kernels on first call."""
    if _kernels:
        return
    kdir = _get_kernel_dir()

    seq_src = (kdir / "sequential_kernel.cu").read_text()
    seq_fns = compile_and_load(seq_src, ["gdn_prefill_sequential"])
    _kernels["sequential"] = seq_fns["gdn_prefill_sequential"]

    gate_src = (kdir / "fused_gate_kernel.cu").read_text()
    gate_fns = compile_and_load(gate_src, ["fused_gate_kernel"])
    _kernels["fused_gate"] = gate_fns["fused_gate_kernel"]


def _ptr(t: torch.Tensor):
    """Get raw device pointer as ctypes for CUDA driver API."""
    return ctypes.c_uint64(t.data_ptr())


def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    _ensure_kernels()

    T = q.shape[0]
    num_seqs = cu_seqlens.shape[0] - 1
    stream = drv.CUstream(torch.cuda.current_stream().cuda_stream)

    if scale is None or scale == 0.0:
        scale = (128) ** -0.5

    if T <= THRESHOLD:
        # --- Short path: NVRTC sequential kernel (fused gates) ---
        output = torch.empty(T, 8, 128, dtype=torch.bfloat16, device=q.device)
        new_state = torch.empty_like(state)

        grid = (num_seqs * 8, 1, 1)
        block = (128, 1, 1)

        launch(
            _kernels["sequential"], grid, block,
            [_ptr(q), _ptr(k), _ptr(v), _ptr(state),
             _ptr(A_log), _ptr(a), _ptr(dt_bias), _ptr(b),
             _ptr(cu_seqlens), ctypes.c_float(scale),
             _ptr(output), _ptr(new_state), ctypes.c_int32(num_seqs)],
            shared_mem=0, stream=stream,
        )
        return output, new_state

    else:
        # --- Long path: fused gate kernel + CuTe-DSL ---
        gate_log = torch.empty(T, 8, dtype=torch.float32, device=q.device)
        beta = torch.empty(T, 8, dtype=torch.float32, device=q.device)

        total_elems = T * 8
        threads = 256
        blocks_gate = (total_elems + threads - 1) // threads

        launch(
            _kernels["fused_gate"], (blocks_gate, 1, 1), (threads, 1, 1),
            [_ptr(A_log), _ptr(a), _ptr(dt_bias), _ptr(b),
             _ptr(gate_log), _ptr(beta),
             ctypes.c_int32(T), ctypes.c_int32(8)],
            shared_mem=0, stream=stream,
        )

        path_name = choose_path(total_seq_len=T, num_seqs=num_seqs)

        output, new_state = chunk_gated_delta_rule(
            q=q.unsqueeze(0),
            k=k.unsqueeze(0),
            v=v.unsqueeze(0),
            g=gate_log.unsqueeze(0),
            beta=beta.unsqueeze(0),
            initial_state=state,
            cu_seqlens=cu_seqlens,
            scale=scale,
            path_name=path_name,
        )

        return output.squeeze(0), new_state
```

- [ ] **Step 2: Commit**

```bash
git add gdn_prefill_qk4_v8_d128_k_last/solution/python/main.py
git commit -m "feat(prefill): hybrid dispatch — NVRTC sequential for short, fused gate + CuTe-DSL for long"
```

---

### Task 5: Benchmark and Verify Correctness

**Files:** None (benchmark only)

- [ ] **Step 1: Run quick smoke benchmark (3 workloads)**

```bash
conda run -n fi-bench modal run scripts/run_modal_subfolder.py --subfolder gdn_prefill_qk4_v8_d128_k_last --limit 3
```

Expected: All 3 PASSED, latency visible.
If FAILED: check error output, fix NVRTC compilation or pointer issues.

- [ ] **Step 2: Run full 100-workload benchmark**

```bash
conda run -n fi-bench modal run scripts/run_modal_subfolder.py --subfolder gdn_prefill_qk4_v8_d128_k_last --no-quick
```

Expected: 100/100 PASSED, mean latency < 0.25 ms.

- [ ] **Step 3: Log results to bench_history.jsonl**

Append result to `logs/prefill/bench_history.jsonl` in standard format.

- [ ] **Step 4: Commit log**

```bash
git add logs/prefill/bench_history.jsonl
git commit -m "bench(prefill): hybrid dispatch benchmark results"
```

---

### Task 6: Tune THRESHOLD via Benchmark

**Files:**
- Modify: `gdn_prefill_qk4_v8_d128_k_last/solution/python/main.py` (THRESHOLD constant)

- [ ] **Step 1: Run benchmark with THRESHOLD=256**

Change `THRESHOLD = 256` in main.py, run full benchmark, record mean latency.

- [ ] **Step 2: Run benchmark with THRESHOLD=1024**

Change `THRESHOLD = 1024` in main.py, run full benchmark, record mean latency.

- [ ] **Step 3: Pick best THRESHOLD**

Compare results from THRESHOLD=256, 512, 1024. Set THRESHOLD to the value with lowest mean latency.

- [ ] **Step 4: Commit final THRESHOLD**

```bash
git add gdn_prefill_qk4_v8_d128_k_last/solution/python/main.py
git commit -m "tune(prefill): set THRESHOLD to optimal value based on benchmark"
```

---

### Task 7 (Phase 2, if needed): Add Chunked wmma Path for Medium Sequences

Only proceed if Phase 1 mean latency > 0.20 ms.

**Files:**
- Create: `gdn_prefill_qk4_v8_d128_k_last/solution/python/nvrtc_kernels/chunked_kernel.cu`
- Modify: `gdn_prefill_qk4_v8_d128_k_last/solution/python/main.py` (add third dispatch path)

- [ ] **Step 1: Implement chunked kernel with wmma**

Chunked delta rule with:
- chunk_size = 64
- `nvcuda::wmma` 16x16x16 bf16→fp32 for K@S, Q@S, K@K^T, state update
- Forward substitution for (I - beta*L)^{-1} using 4x16x16 block decomposition
- Register-cached state between chunks

- [ ] **Step 2: Add 3-path dispatch in main.py**

```python
if T <= SHORT_THRESHOLD:
    # sequential CUDA
elif T <= MEDIUM_THRESHOLD:
    # chunked wmma CUDA
else:
    # fused gate + CuTe-DSL
```

- [ ] **Step 3: Benchmark and verify correctness**

Run full 100-workload benchmark. Verify all pass. Target: mean latency < 0.20 ms.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat(prefill): add chunked wmma kernel for medium sequences"
```
