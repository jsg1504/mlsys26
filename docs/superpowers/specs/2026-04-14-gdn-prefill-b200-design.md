# GDN Prefill B200 Design

Date: 2026-04-14
Target: `gdn_prefill_qk4_v8_d128_k_last`
Repository: `round4`

## Context

The current implementation in `gdn_prefill_qk4_v8_d128_k_last/solution/cuda/kernel.cu` is a single CUDA kernel that assigns one block to each `(seq, v_head)` pair and performs sequential per-token recurrent updates with large dynamic shared memory. It passes correctness, but it does not exploit Blackwell-specific data movement and tensor-core pipelines.

The released FlashInfer baseline for the same definition uses a CuTe/CUTLASS DSL kernel specialized for Blackwell prefill with chunked execution. The workload distribution is mixed: most workloads are short (`total_seq_len` median `139`, `num_seqs` median `2`), but the tail includes `total_seq_len=8192` with `num_seqs` up to `57`. The design therefore needs to preserve short-shape efficiency while also scaling on large variable-length batches.

## Goals

- Maximize average benchmark speedup on Modal B200 for `gdn_prefill_qk4_v8_d128_k_last`.
- Target Blackwell-oriented kernel structure rather than incremental optimization of the current sequential CUDA kernel.
- Keep the implementation focused on the exact contest definition:
  `qk4`, `v8`, `head_dim=128`, `k-last` state layout, variable-length prefill, `bf16` inputs and outputs, `fp32` state.
- Support at most two runtime execution paths to balance average speed and operational complexity.

## Non-Goals

- Preserving the current `language = "cuda"` submission format.
- Building a general-purpose GDN library that supports arbitrary head counts, layouts, or head dimensions.
- Supporting more than two runtime kernel paths.
- Refactoring unrelated decode kernels or repository-wide infrastructure.

## Chosen Approach

Use the released Blackwell CuTe/CUTLASS baseline as the starting point, then aggressively specialize it for this exact definition.

This is preferred over optimizing the existing `kernel.cu` because the baseline already contains the Blackwell-specific mechanisms needed for a higher performance ceiling: chunked execution, TMA-driven global-to-shared movement, TCGen05-oriented shared-memory layouts, and warp-specialized producer/consumer pipelines. Rebuilding those features inside the current sequential CUDA kernel would effectively reimplement the baseline from scratch with lower confidence and slower iteration.

## High-Level Architecture

The optimized submission will be reorganized into three layers.

### 1. Host/API Layer

The host entry point will:

- validate input shapes and dtypes for the single supported definition;
- materialize gate tensors used by the kernel:
  `g = exp(-exp(A_log) * softplus(a + dt_bias))`
  `beta = sigmoid(b)`;
- build or reuse compiled CuTe kernels through a compile cache;
- dispatch to one of two runtime kernel paths.

### 2. Kernel Family Layer

There will be two runtime execution paths only:

- `small-shape` path for short prompts and small numbers of sequences, where launch/compile overhead and tile underfill dominate;
- `large-shape` path for long prompts and/or many sequences, where deep pipelining and higher occupancy matter more.

Both paths will remain definition-specific and share the same external API and tensor layouts.

### 3. Shared Kernel Primitives

The following pieces will be reused from the Blackwell baseline and then trimmed to this definition:

- tile scheduler for variable-length chunk traversal;
- TMA descriptor setup and prefetch logic;
- TCGen05-compatible shared-memory layout helpers;
- state/output store path for `k-last` state layout.

General-purpose options not needed for this definition will be removed to reduce compile surface and planning complexity.

## Kernel And Data Flow Design

### Fixed Problem Shape

The implementation is specialized for:

- `q`: `[T, 4, 128]`, `bf16`
- `k`: `[T, 4, 128]`, `bf16`
- `v`: `[T, 8, 128]`, `bf16`
- `state`: `[N, 8, 128, 128]`, `fp32`
- `A_log`: `[8]`, `fp32`
- `a`: `[T, 8]`, `bf16`
- `dt_bias`: `[8]`, `fp32`
- `b`: `[T, 8]`, `bf16`
- `cu_seqlens`: `[N + 1]`, `int64`

No dynamic support for other head configurations or layouts will be planned.

### Chunking Strategy

Use `chunk_size=128` as the primary tile granularity.

Rationale:

- the released Blackwell baseline already assumes `chunk_size=128` for this kernel family;
- `head_dim=128` aligns naturally with `128 x 128 x 128` matrix-style work decomposition;
- both recurrent state update and output generation map cleanly to the same tile size.

### Host-Side Precompute

The kernel will not recompute `softplus`, `exp`, and `sigmoid` per timestep inside the main recurrence loop. The host path will precompute and materialize:

- `g` in `fp32`
- `beta` in `fp32`

This removes scalar transcendental work from the hottest kernel region and matches the expected interface of the released chunked baseline.

### Variable-Length Scheduling

Inputs remain in flat varlen layout driven by `cu_seqlens`.

The scheduler will assign work in chunk coordinates derived from sequence spans. Tail chunks that extend beyond a sequence boundary are masked or skipped using per-sequence length checks. The design preserves the existing flat input/output ABI while changing only the internal traversal strategy.

### Main Compute Path

Per chunk, the kernel performs:

1. load `Q`, `K`, `V`, gate data, and prior state tiles;
2. compute chunk-local intermediates for the gated delta recurrence;
3. update recurrent state in `fp32`;
4. generate output tiles and store them in `bf16`.

The large-shape path keeps the Blackwell-style structure:

- TMA-based global-to-shared movement;
- TCGen05-compatible shared layouts;
- warp-specialized producer/consumer roles;
- staged pipelines for `Q/K`, `V`, and epilogue/store work.

### State And Output Layout

- `new_state` remains `fp32` in `k-last` layout;
- `output` remains `bf16`;
- the external ABI does not change.

Changing state layout is intentionally excluded because it would expand correctness risk and reduce the reuse value of the released baseline.

## Runtime Dispatch Design

The optimized implementation supports exactly two runtime paths.

### Small-Shape Path

Use when workloads are dominated by short total sequence length and small sequence counts. This path prioritizes:

- low launch overhead;
- reduced wasted work from underfilled tiles;
- lower scheduling complexity.

### Large-Shape Path

Use for longer prompts and/or higher `num_seqs`, especially the `8192`-length tail. This path prioritizes:

- Blackwell pipeline depth;
- sustained throughput;
- better amortization of TMA and staging overhead.

### Dispatch Policy

Dispatch thresholds are empirical and will be tuned on Modal B200. The design intentionally fixes the policy shape without hardcoding the final threshold yet:

- only two paths are allowed;
- thresholds will be expressed using `total_seq_len` and `num_seqs`;
- compile/cache behavior must not explode with shape-dependent variants.

The first implementation milestone will bring up a single specialized baseline path for correctness. The second milestone adds dual-path dispatch and tunes the threshold using benchmark evidence.

## Compile And Cache Strategy

The compile cache is intentionally narrow. Cache keys should include only the attributes that materially change generated code, such as:

- device;
- path type (`small-shape` or `large-shape`);
- dtype and output-state mode;
- varlen mode if the interface still exposes that distinction.

The cache should not key on every concrete sequence length shape. This avoids unnecessary recompilation and reduces first-run overhead noise on Modal.

## Error Handling

Error handling lives primarily in the host/API layer.

- Reject unsupported shapes immediately.
- Reject unsupported dtypes immediately.
- Reject inconsistent `cu_seqlens` metadata immediately.
- Keep kernel fast-path free of optionality that is not required by this definition.

The intent is to fail early before compilation or launch rather than carry defensive branches into the hot kernel path.

## Testing And Verification

Verification is split into three stages.

### 1. Smoke Correctness

Run at least one representative short-shape workload and one representative long-shape workload to verify:

- outputs;
- final state;
- basic dispatch behavior.

### 2. Full Correctness Sweep

Run all available `100` workloads for correctness before claiming the optimization is ready. Short-only validation is explicitly insufficient because the workload set contains long-tail `8192` cases with many sequences.

### 3. Performance Benchmarking

Run Modal B200 benchmarks after correctness is stable. Dual-path thresholds are tuned using benchmark evidence rather than assumed heuristics.

Metrics to record:

- mean speedup;
- mean latency;
- path coverage for representative short and long cases;
- correctness summary (`max_abs_error`, `max_rel_error`, `matched_ratio`).

## Rollout Plan

Implementation should proceed in three phases:

1. Replace the current sequential CUDA path with a single specialized CuTe/CUTLASS baseline-derived path and make it correct.
2. Introduce the second runtime path and add dispatch thresholds for short versus large workloads.
3. Remove leftover generality and dead branches that do not help this definition.

This ordering minimizes ambiguity during planning: first establish a correct specialized baseline, then add the second path only after the shared architecture is working.

## Risks And Mitigations

### Risk: Short-shape regressions due to heavyweight Blackwell pipeline

Mitigation: keep a dedicated `small-shape` path and tune dispatch thresholds using measured B200 results.

### Risk: Compile/cache churn from overspecialization

Mitigation: constrain generated variants to two runtime paths and a narrow cache key.

### Risk: Correctness drift in long-tail varlen workloads

Mitigation: require a full correctness sweep across all workloads before treating performance numbers as meaningful.

### Risk: Excess complexity from baseline generality

Mitigation: prune unused options and retain only the exact configuration needed for this contest definition.

## Planning Readiness

This spec is ready to drive implementation planning for a single focused project:

- migrate the prefill submission from a sequential CUDA kernel to a specialized CuTe/CUTLASS Blackwell path;
- keep exactly two runtime kernel paths;
- optimize for average benchmark speed on Modal B200 while preserving correctness across the full workload set.
