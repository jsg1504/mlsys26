# GDN Prefill Optimization Review

Date: 2026-04-07

## Scope

This document analyzes the current `gdn_prefill_qk4_v8_d128_k_last` project state, Blackwell B200 optimization opportunities, Gated Delta Net workload characteristics, and which methods are realistically applicable to the current implementation.

## Agent Team Split

- Hardware agent: B200 / Blackwell architecture, memory hierarchy, cluster/DSM, occupancy, and async data movement.
- Model/workload agent: Gated Delta Net algorithm, Qwen3-Next linear-attention shape assumptions, and workload distribution.
- Kernel agent: current CUDA kernel structure, bottlenecks, and implementation feasibility.
- Integrator: merge findings into an optimization roadmap for this repo.

## Current Project State

- Active target is CUDA prefill under [gdn_prefill_qk4_v8_d128_k_last/config.toml](/home/jsg1504/workdir/KimJangMaengShin/codex-mlsys26/gdn_prefill_qk4_v8_d128_k_last/config.toml#L1).
- Definition is fixed at `q_heads=4`, `k_heads=4`, `v_heads=8`, `head_dim=128`, `state dtype=float32`, `input dtype=bfloat16`, and varlen batching via `cu_seqlens`: [gdn_prefill_qk4_v8_d128_k_last.json](/home/jsg1504/workdir/KimJangMaengShin/codex-mlsys26/mlsys26-contest/definitions/gdn/gdn_prefill_qk4_v8_d128_k_last.json#L11).
- The current optimize log is effectively empty: [optimization_log.md](/home/jsg1504/workdir/KimJangMaengShin/codex-mlsys26/logs/prefill/optimization_log.md#L1).
- The current kernel is a sequential baseline, not a chunked prefill kernel: [kernel.cu](/home/jsg1504/workdir/KimJangMaengShin/codex-mlsys26/gdn_prefill_qk4_v8_d128_k_last/solution/cuda/kernel.cu#L43).

## Current Kernel Findings

### Structural observations

- One block handles exactly one `(seq, v_head)` pair: [kernel.cu](/home/jsg1504/workdir/KimJangMaengShin/codex-mlsys26/gdn_prefill_qk4_v8_d128_k_last/solution/cuda/kernel.cu#L38).
- A full `128 x 128` fp32 state tile is copied into dynamic shared memory, which is about 64 KB before adding scratch buffers: [kernel.cu](/home/jsg1504/workdir/KimJangMaengShin/codex-mlsys26/gdn_prefill_qk4_v8_d128_k_last/solution/cuda/kernel.cu#L74).
- Tokens are processed strictly sequentially inside the block: [kernel.cu](/home/jsg1504/workdir/KimJangMaengShin/codex-mlsys26/gdn_prefill_qk4_v8_d128_k_last/solution/cuda/kernel.cu#L93).
- For every token, the kernel loops over `vi=0..127` twice. Both loops perform warp reductions and CTA-wide barriers for each `vi`: [kernel.cu](/home/jsg1504/workdir/KimJangMaengShin/codex-mlsys26/gdn_prefill_qk4_v8_d128_k_last/solution/cuda/kernel.cu#L106), [kernel.cu](/home/jsg1504/workdir/KimJangMaengShin/codex-mlsys26/gdn_prefill_qk4_v8_d128_k_last/solution/cuda/kernel.cu#L124).

### Immediate bottlenecks

- The same scalar gate values `a[t,vh]`, `b[t,vh]`, `A_log[vh]`, and `dt_bias[vh]` are recomputed or reread redundantly by all 128 threads.
- The kernel executes 513 `__syncthreads()` per token:
  - 1 after loading `v`
  - 2 * 128 in the `k @ temp` loop
  - 2 * 128 in the update/output loop
- Each `vi` row reduction is serialized through `tid == 0` for both `s_kdot[vi]` and final output writeback.
- There is no tensor-core use, no async copy, no double-buffering, no warp specialization, and no reuse of shared `q/k` across the 2 value heads that map to the same q/k head.
- The kernel pays the full sequential dependency cost of prefill, even though chunked GDN algorithms exist.

### Correctness / interface observations

- The definition marks `state` as optional, but the current kernel unconditionally dereferences it: [gdn_prefill_qk4_v8_d128_k_last.json](/home/jsg1504/workdir/KimJangMaengShin/codex-mlsys26/mlsys26-contest/definitions/gdn/gdn_prefill_qk4_v8_d128_k_last.json#L74), [kernel.cu](/home/jsg1504/workdir/KimJangMaengShin/codex-mlsys26/gdn_prefill_qk4_v8_d128_k_last/solution/cuda/kernel.cu#L71).
- The current implementation is acceptable as a baseline, but it is not aligned with the likely B200 winning strategy.

## GDN / Workload Findings

### Algorithm implications

- GDN prefill is not ordinary attention. It is a recurrent state update with per-token decay and beta gating on a `128 x 128` state matrix.
- Naive token-by-token execution is correct but under-parallelized.
- Chunk-wise Gated Delta Rule implementations parallelize prefill by turning long sequential recurrence into chunk-level matrix operations plus intra-chunk solves/updates.

### Shape implications from this definition

- `head_dim=128` is fixed, which is a favorable size for tensor-core tiling.
- `v_heads=8` and `q/k_heads=4` imply grouped-value attention; two value heads share one q/k head.
- State layout is `k-last [N, H, V, K]`, which already gives contiguous access along `K` for one row update.

### Dataset observations

Workload scan over all 100 prefill cases in `mlsys26-contest/workloads/gdn/gdn_prefill_qk4_v8_d128_k_last.jsonl`:

- `num_seqs`: min 1, p50 2, p90 35, max 57.
- `total_seq_len`: min 6, p50 139, p90 8192, max 8192.
- Per-sequence lengths: p50 41, p90 662, p95 1131, p99 2822, max 5637.
- Sequence-length bucket counts:
  - `1..32`: 338
  - `33..128`: 254
  - `129..512`: 112
  - `513..2048`: 74
  - `2049+`: 19
- Case `num_seqs` bucket counts:
  - `1`: 33 cases
  - `2..4`: 48 cases
  - `17..32`: 4 cases
  - `33..64`: 12 cases

Implication:

- Many sequences are short enough that launch overhead, barriers, and redundant scalar work matter.
- A nontrivial tail of very long sequences exists, so a true chunked/persistent path is still necessary.
- A single strategy is unlikely to be optimal across the full set.

## B200 / Blackwell Findings

### Hardware features that matter here

- Blackwell compute capability 10.0 supports up to 64 resident warps per SM and 64K 32-bit registers per SM.
- B200 supports thread block clusters and Distributed Shared Memory (DSM). Portable cluster size is 8, and B200 can opt into nonportable cluster size 16.
- B200 keeps up to 256 KB unified L1/shared/texture capacity per SM, with up to 227 KB dynamic shared memory per block after opt-in.
- B200 retains L2 persistence controls.
- Blackwell improves tensor-core throughput and exposes newer data movement / kernel-structuring paths that Triton and CUTLASS/CuTe can exploit.

### B200 constraints that matter here

- Shared-memory-heavy kernels can still become occupancy-limited.
- DSM only helps if work is split across multiple CTAs that need to share state; it does not help the current one-block-per-seq-head design.
- Tensor-core wins require reformulating work into tile-level matrix products. The current scalar loop structure does not expose that.

## Strong Evidence From Contest Baseline

The provided contest baseline already includes a Blackwell-specific GDN prefill solution:

- Description says it uses `FlashInfer SM100 CuTile chunk_gated_delta_rule` on B200: [flashinfer_wrapper_123ca6.json](/home/jsg1504/workdir/KimJangMaengShin/codex-mlsys26/mlsys26-contest/solutions/baseline/gdn/gdn_prefill_qk4_v8_d128_k_last/flashinfer_wrapper_123ca6.json#L39).
- The embedded source clearly states:
  - chunked linear attention for Blackwell
  - persistent and non-persistent modes
  - CuTe-DSL / tcgen05 / TMA-oriented structure
  - fixed chunk size 128

This is the most important directional signal in the repo: the likely high-end solution class is not "make the current sequential kernel slightly better", but "move to a chunked Blackwell-native kernel architecture."

## Optimization Methods

### Tier 0: Immediate, low-risk changes in the current CUDA kernel

1. Hoist scalar gate work out of per-thread execution.
   - Load `a[t,vh]` and `b[t,vh]` once per block, compute `g` and `beta` once, then broadcast.
   - Applicability: direct.
   - Expected benefit: small-to-moderate, especially for short sequences.

2. Vectorize state load/store and possibly `q/k/v` accesses.
   - Use `float4` for state rows and `__nv_bfloat162` or wider packed loads where alignment permits.
   - Applicability: direct.
   - Expected benefit: moderate bandwidth and instruction-count reduction.

3. Rework reductions to reduce CTA-wide barriers.
   - Current pattern writes warp partials to shared memory and synchronizes for every row.
   - Replace with warp-group or staged row-tiles so one barrier serves multiple rows.
   - Applicability: moderate refactor.
   - Expected benefit: moderate on all shapes, large on short sequences where barrier cost dominates.

4. Process the two sibling `v_heads` sharing the same `q/k` head together.
   - Reuse `q` and `k` loads across the pair.
   - Applicability: moderate refactor.
   - Expected benefit: modest but real.

5. Add separate fast paths for very short sequences.
   - For `seq_len <= 32` or `<= 64`, favor simpler low-latency kernels over a large shared-memory kernel.
   - Applicability: direct.
   - Expected benefit: good weighted benefit because short sequences are common.

### Tier 1: Medium-complexity optimizations while staying mostly in raw CUDA

1. Tile the state over `V` instead of staging the entire `128 x 128` state.
   - Example: work on `32 x 128` or `64 x 128` row tiles.
   - Reduces shared-memory footprint and may increase active blocks per SM.
   - Applicability: direct but nontrivial.
   - Expected benefit: moderate.

2. Introduce warp specialization inside the CTA.
   - Dedicated warps for load, reduction/update, and epilogue.
   - Applicability: possible in raw CUDA, easier in CuTe/CUTLASS.
   - Expected benefit: moderate, especially with async copy.

3. Use `cp.async` style staging / double-buffering for row tiles.
   - Applicability: possible for Ampere+ style loads, but much less ergonomic than TMA/CuTe.
   - Expected benefit: moderate if state tiling is introduced first.

4. Bucket by sequence length and launch different kernels.
   - Short: low-latency sequential kernel.
   - Long: heavier tiled kernel.
   - Applicability: direct.
   - Expected benefit: high weighted ROI.

### Tier 2: High-upside methods most aligned with B200

1. Replace token-by-token prefill with chunked GDN prefill.
   - Use chunk size 128 first, matching the contest baseline and Blackwell-friendly tiling.
   - Applicability: not a local tweak; this is a rewrite.
   - Expected benefit: highest upside by far.

2. Reformulate chunk math into tensor-core MMA pipelines.
   - `head_dim=128` strongly favors this.
   - Keep accumulators/state in fp32 where needed.
   - Applicability: rewrite required.
   - Expected benefit: very high.

3. Use CuTe-DSL / CuTile / CUTLASS-style TMA + TMEM + warp-specialized pipelines.
   - This is the implementation style shown by the contest baseline.
   - Applicability: practical if the project is allowed to move beyond the current plain `kernel.cu` path.
   - Expected benefit: very high.

4. Persistent kernel scheduling for long-sequence chunks.
   - Good fit for the long-tail cases and for keeping SMs busy when tile count is large.
   - Applicability: rewrite-level.
   - Expected benefit: high on long cases.

5. Cluster / DSM use only after chunk-level multi-CTA partitioning exists.
   - Applicability: not useful yet.
   - Expected benefit: secondary, only after the main chunked design works.

## Applicability To The Current Implementation

### Directly applicable now

- Scalar gate hoisting and broadcast.
- Vectorized memory access.
- Fewer barriers through row blocking.
- Pairing sibling `v_heads`.
- Sequence-length bucketization with separate kernels.
- Short-sequence fast path.

### Applicable, but only with moderate refactor

- `V`-tile state staging rather than full-state staging.
- Warp-specialized CTA structure.
- Double-buffered staged loads.

### Not realistic as an incremental patch

- Full chunked GDN prefill.
- Tensor-core-first reformulation.
- TMA/TMEM/Blackwell-specialized CuTe kernel architecture.
- Cluster/DSM multi-CTA cooperation for one logical head-state.

## Recommended Roadmap

### Option A: Pragmatic short-term path in this CUDA codebase

1. Add a short-sequence kernel for `seq_len <= 64`.
2. Hoist scalar gate computation and broadcast once per token.
3. Vectorize state load/store.
4. Reduce per-row barrier count by row blocking.
5. Launch a separate long-sequence kernel with state tiling over `V`.

Reason:

- This is the fastest way to turn the current baseline into a materially better baseline without a full rewrite.

### Option B: Highest-upside path for leaderboard performance

1. Stop treating current `kernel.cu` as the final architecture.
2. Build a chunked GDN prefill kernel around chunk size 128.
3. Implement it in CuTe-DSL / CuTile / CUTLASS-style infrastructure, or port the FlashInfer baseline ideas into your own solution.
4. Add persistent scheduling for long chunks and a short-sequence fallback for tail latency.

Reason:

- This matches both the hardware and the contest baseline direction.
- B200 advantages are much easier to realize in this form than in the current scalar-loop kernel.

## Priority Ranking

1. Chunked prefill rewrite using Blackwell-native tiling and tensor-core math.
2. Sequence-length bucketization with separate short and long kernels.
3. Barrier reduction and row blocking in the current kernel.
4. Gate hoisting and scalar broadcast.
5. Vectorized loads/stores.
6. Pairing sibling `v_heads`.
7. Cluster / DSM exploration after chunked rewrite only.

## Bottom Line

- The current kernel is useful as a correctness baseline, but it is structurally far from a B200-optimized solution.
- The dominant opportunity is algorithmic: move from sequential token update to chunked GDN prefill.
- The dominant implementation opportunity is architectural: use Blackwell-friendly tensor-core/TMA/warp-specialized tiling, likely via CuTe/CuTile rather than plain hand-written scalar CUDA.
- If you want an immediate next step without rewriting everything, do sequence-length specialization plus barrier/scalar-load cleanup first.

## External References

- NVIDIA Blackwell Tuning Guide:
  - https://docs.nvidia.com/cuda/archive/13.0.2/pdf/Blackwell_Tuning_Guide.pdf
- NVIDIA technical blog on Triton for Blackwell:
  - https://developer.nvidia.com/blog/openai-triton-on-nvidia-blackwell-boosts-ai-performance-and-programmability/
- Hugging Face Qwen3-Next model documentation:
  - https://huggingface.co/docs/transformers/model_doc/qwen3_next
