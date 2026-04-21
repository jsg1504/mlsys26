# GDN Prefill Kernel Optimization Log

Tracking all optimization iterations for the prefill kernel.

---

<!-- Append new entries below this line -->

## 2026-04-21 - Gate Kernel First (CPU/GPU overlap attempt — NEUTRAL, reverted)

- **Idea**: Reorder main.py so the fused_gate_kernel launches BEFORE the multi-seq `cu_seqlens.tolist()` D2H sync. Hypothesis: the ~2us gate kernel GPU time would overlap with the ~3-5us D2H CPU-blocking time.
- **Result**: 0.2054ms → **0.2054ms (0.0%, neutral)**
- **Status**: reverted (no benefit, kept original order for simplicity)
- **Why no overlap happened**: `tensor.cpu()` / `tensor.tolist()` issues `cudaMemcpyAsync` on the CURRENT PyTorch CUDA stream — the SAME stream as the gate kernel launch. CUDA streams are strictly ordered — the D2H memcpy queues AFTER the gate kernel and waits for it to complete. No parallel execution between same-stream operations.
- **Learning**: To actually overlap CPU-blocking D2H with GPU compute, the D2H and the GPU work must be on DIFFERENT streams. But creating a secondary stream context in Python adds ~2-3us overhead per call, likely exceeding the 2us gate kernel time we'd overlap. Net: not worth pursuing. The stream-serialization property means Python-level reordering around `.cpu()` has no effect on GPU timeline — it only affects CPU timeline.

## 2026-04-21 - Detailed NCU + Partial Unroll Experiment (REVERTED)

### Detailed NCU Findings (Workload 80, T=48)

Ran ncu with `--set detailed` for full Memory Workload Analysis, Occupancy, Source Counters, Tile Statistics.

**Memory Workload Analysis**:
```
Local Memory Spilling Requests:     76800.00
Local Memory Spilling Req Overhead: 100.00%   (of mem requests)
L1/TEX Hit Rate:                    96.43%
L2 Hit Rate:                        76.00%
Memory Throughput:                  4.65 GB/s  (VERY LOW)
Mem Busy:                           1.61%      (memory hw 98% idle)
Mem Pipes Busy:                     1.04%      (memory pipes 99% idle)
```

**Key insight**: Although 76,800 register spill requests exist, **96.4% of them hit L1 cache** (not DRAM). Memory subsystem is 99% idle. The "spilling" NCU highlights is an ALLOCATION decision by the compiler, not a bandwidth problem. L1-cached spills cost ~20-30 cycles but that's much less than DRAM-bound spills would.

**Why the kernel is still slow**: Not memory bandwidth (1.04% busy) but **latency hiding failure** — with 4 warps/SM and FMA chains, the scheduler has no eligible warp 89% of the time. The scalar FMA dependency chain (each token's kS accumulator) is the true critical path.

### Experiment: #pragma unroll → #pragma unroll 16

- **Hypothesis**: NCU showed 76,800 spill requests. Reducing unroll factor would limit live-register count and eliminate spilling.
- **Result**: 0.2054 → **0.2471ms (+20% REGRESSION)**. Min latency 0.039 → 0.062ms (short workloads got MUCH slower). Fast workload count (<0.1ms) dropped from ~48 to 22.
- **Status**: reverted
- **Why it failed**:
  1. L1 hit rate was already 96.4% on spills — the spills weren't costly (stayed in L1).
  2. Partial unroll introduced **loop control overhead** (counter increment, branch) that wasn't there with full unroll.
  3. Full unroll enables the compiler to **schedule FMA instructions across the entire loop** with deep pipelining; partial unroll fragments this.
  4. The compiler's full-unroll + spill-to-L1 strategy is actually optimal for this workload.
- **Learning**: NCU's "Spilling Requests" metric is misleading without context — at 96% L1 hit rate, spills ARE the compiler's chosen optimization, and they're cheap. The true bottleneck is scheduler starvation (4 warps can't hide FMA-chain latency), which cannot be fixed without algorithmic changes (more parallelism via chunked scan).

## 2026-04-21 - NCU Hardware Profiling + launch_bounds Experiment (REVERTED)

### NCU Profiling Findings

Ran actual `ncu` (NVIDIA Nsight Compute) on Modal B200 via `scripts/ncu_profile_modal.py` (sections: SpeedOfLight, LaunchStats, Occupancy, SchedulerStats, WarpStateStats, MemoryWorkloadAnalysis, CSV output).

For `gdn_prefill_sequential` (our kernel):

| Metric | Workload 0 (T=6) | Workload 80 (T=48) |
|---|---|---|
| Block Limit Registers | 2.00 | 2.00 |
| Theoretical Active Warps/SM | 8 | 8 |
| Theoretical Occupancy | 12.50% | 12.50% |
| **Achieved Occupancy** | **6.21%** | **6.28%** |
| One or More Eligible | 10.37% | 18.17% |
| **No Eligible** | **89.63%** | **81.83%** |
| Active Warps/Scheduler | 1.00 | 1.00 |
| Eligible Warps/Scheduler | 0.10 | 0.18 |
| Warp Cycles/Issued Instruction | 9.65 | 5.50 |

**Diagnosis**: Kernel is **latency-bound** with **severely low achieved occupancy (6.2%)**. 89% of scheduler cycles have no eligible warp to issue. Only 4 warps/SM active.

### Experiment: __launch_bounds__(128, 1) → (128, 2)

- **Hypothesis**: Block Limit Registers = 2 means current register usage already permits 2 blocks/SM. Changing launch_bounds from (128,1) to (128,2) lets the grid scheduler pack 2 blocks/SM → 8 warps/SM → better latency hiding.
- **Result**: 0.2054 → **0.2192ms (+6.7%)** (though this may be partially Modal GPU noise — a repeat revert run measured 0.2177)
- **Status**: reverted
- **Root cause of no-help**: The critical constraint is the TOTAL number of blocks, not per-SM packing. For single-seq workloads we launch only `1 * 8 = 8 blocks`. Packing 2 blocks/SM would use just 4 SMs instead of 8 — fewer SMs doing more work serially. The "low occupancy" NCU flagged is a symptom of insufficient grid parallelism for the workload (single sequence, sequential recurrence in time), not something fixable by register/occupancy tuning.
- **Learning**: NCU's "Block Limit Registers = 2" implied we had 2x room on each SM, but with only 8 blocks in the grid we CANNOT benefit from allowing multiple per SM — doing so just concentrates work on fewer SMs. To actually improve parallelism would require algorithmic changes (chunked parallel scan splitting the time dimension), which is a much larger rewrite.

## 2026-04-21 - Modal B200 Profiling Analysis (no code change)

Profiled 8 representative workloads on Modal B200 using torch.profiler + CUDA events. Script: `scripts/profile_modal.py`.

### Per-workload overhead breakdown

| Workload | T | num_seqs | Path | Kernel time | Total | **Overhead** |
|---|---|---|---|---|---|---|
| 0  | 6    | 1  | seq  | 30.6us  | 44.8us   | **14.2us** |
| 5  | 8192 | 34 | CuTe | 638us   | 695.6us  | **57us**   |
| 10 | 8192 | 38 | CuTe | 601us   | 664.8us  | **64us**   |
| 20 | 3999 | 13 | CuTe | 284us   | 344.4us  | **60us**   |
| 40 | 139  | 3  | CuTe | 23.6us  | 76.6us   | **53us (69% of total!)** |
| 60 | 35   | 2  | seq  | 55.7us  | 70.5us   | **14.8us** |
| 80 | 48   | 1  | seq  | 116.3us | 131.2us  | **14.9us** |
| 95 | 959  | 4  | CuTe | 92.9us  | 146.5us  | **53.6us** |

### Key findings

1. **Sequential path overhead is a tight ~15us constant** — composed of Python dispatch (~5us), ctypes arg setup (~3us), cuLaunchKernel (~5us), CUDA event recording (~2us). Very hard to reduce further.

2. **CuTe-DSL path overhead is ~50us**, breakdown:
   - `fused_gate_kernel` launch + GPU time: **~5us** (2us GPU + 3us Python launch)
   - Multi-seq `cu_seqlens.tolist()` D2H sync: **~3-5us** (single-seq skips this, per earlier fix)
   - `compiled_gdn` TVM FFI call: **~15-20us** (biggest chunk, marshaling + launch)
   - Other Python dispatch (dict lookups, data_ptr calls, bundle cache): **~15-20us**

3. **For medium CuTe-DSL workloads (T=100-1000), overhead is 50-70% of total time**. These are the leverage points if overhead can be reduced.

4. **For long CuTe-DSL workloads (>0.5ms), overhead is <10%** — they are dominated by kernel compute. Can't be accelerated without modifying the vendored kernel.

### Remaining optimization space (not pursued this iteration)

- **Fuse gate kernel into CuTe-DSL `gb_warp`**: Save ~5us/call on 55 CuTe-DSL workloads = 275us mean = **~1.3% improvement**. **Risk**: modifying the 4600-line vendored kernel; prior attempts at CuTe-DSL surgery (stage tuning) were abandoned due to complexity.
- **Bypass TVM FFI**: Save ~10-15us/call = **~3% improvement**. **Risk**: very high — requires extracting raw CUDA function handle from CuTe-DSL's compiled object, which is not a supported API.
- **Gate kernel on secondary stream (overlap with D2H)**: Save ~2us on multi-seq calls = **~0.3% improvement**. Low-medium risk but also low value.
- **Stable ctypes pointer caching for gate kernel args**: Save ~1-2us/call = **~0.5% improvement**. Low risk but micro-optimization.

### Decision for this iteration

Commit profiling infrastructure (`scripts/profile_modal.py`). No kernel code change — the analysis shows the remaining 3% gap to sub-0.2ms is in overhead chunks that are either (a) hard to reduce without high-risk vendored-kernel surgery, or (b) spread across many small sources that would require many micro-optimizations for compound gains. Current state (0.2054ms, -27.2% from original baseline) is a strong result.

## 2026-04-21 - Minimal inline max_s_q (ACCEPTED, marginal)

- **Idea**: Replace `get_cu_seqlens_metadata(cu_seqlens)` (which caches by id — always misses due to clone-per-iteration — and does dict/weakref bookkeeping and builds a full metadata dict including unused fields) with minimal inline: `cu_seqlens.tolist(); max(h[i+1] - h[i] for i in range(num_seqs))`. Skips weakref creation, dict storage, nondecreasing check, and multi-field dict construction.
- **Result**: 0.2064ms → **0.2054ms (-0.5%)** (mean of 2 runs: 0.2057, 0.2051)
- **Status**: accepted
- **Correctness**: 100/100 pass
- **Cumulative**: 0.282ms → 0.2054ms (-27.2%)
- **Learning**: For a function called in the hot path, the Python-side overhead of dict lookup + weakref creation + closure + multi-field dict construction is meaningful (~3-6us). When the cache is known to always miss (clone-per-iteration harness pattern), the caching infrastructure is pure overhead. This is a micro-optimization but clean and safe.

## 2026-04-21 - max_s_q=T upper-bound shortcut (REVERTED — major regression)

- **Idea**: Use `max_s_q = T` for ALL workloads (including multi-seq). T is always a safe upper bound on per-sequence length. Eliminates ALL GPU→CPU syncs.
- **Result**: 0.2064ms → **0.4076ms (+97% MAJOR REGRESSION)**. Max latency 0.86ms → 2.80ms.
- **Status**: reverted
- **Root cause**: CuTe-DSL kernel is very sensitive to `max_s_q`. The tile scheduler allocates `ceil(max_s_q / chunk_size)` tiles per sequence. For multi-seq workloads with T >> max_s_q (e.g., T=500, num_seqs=10, actual max_s_q=50), using `max_s_q=T=500` inflates tiles per seq from 1 to 4, producing 3 extra empty tiles per sequence. Each empty tile still goes through the tile-scheduler pipeline (bounds check, mbarrier sync, early exit), costing ~5-10us per empty tile. Aggregated across all tiles: ~200ms of extra scheduling work on the most affected workloads.
- **Learning**: The CuTe-DSL persistent tile scheduler is NOT free per tile — empty tiles still have significant setup overhead. Any overestimate of `max_s_q` directly multiplies the tile count and measurably hurts performance. The "safe overestimate" approach cannot be used here.

## 2026-04-21 - Inline (cu_seqlens[1:] - cu_seqlens[:-1]).max() (REVERTED)

- **Idea**: Replace `get_cu_seqlens_metadata` with `int((cu_seqlens[1:] - cu_seqlens[:-1]).max())`. Compute diff+max on GPU, sync only 1 element.
- **Result**: 0.2064ms → **0.2313ms (+12% REGRESSION)**
- **Status**: reverted
- **Root cause**: The subtract and max operations each launch a GPU kernel. Kernel launch overhead on Blackwell is ~5-8us per kernel. Two extra kernel launches = ~10-16us overhead per call, much more than the D2H sync cost (~5us) of `.cpu()` on a small tensor. The original `.cpu().tolist()` does a SINGLE fast D2H transfer followed by pure-CPU computation.
- **Learning**: For small tensors, `.cpu().tolist()` is faster than multi-op GPU reductions because the D2H latency dominates small-tensor transfers anyway, and CPU-side Python arithmetic on a small list is faster than launching two GPU kernels. Kernel launch overhead matters for these micro-operations.

## 2026-04-21 - Single-seq max_s_q shortcut (ACCEPTED)

- **Idea**: When `num_seqs == 1`, `max_s_q == T` exactly. Skip the `get_cu_seqlens_metadata` call entirely (which does `.detach().cpu().tolist()` — a GPU→CPU sync) and use `max_s_q = T` directly.
- **Context discovered**: The flashinfer_bench harness uses `_clone_args` (timing.py:214-220) to clone all tensor arguments before each timed iteration. `get_cu_seqlens_metadata` caches by `id(cu_seqlens)`, so every clone causes a cache miss → forced GPU→CPU sync inside the timed region, costing ~5-10us per miss × 55 CuTe-DSL workloads × 3 timed iters = ~0.8-1.6ms total overhead.
- **Implementation**: Simple branch in `main.py` — if `num_seqs == 1`, use `max_s_q = T`; else fall through to the original `get_cu_seqlens_metadata` call.
- **Result**: 0.2098ms → **0.2064ms (-1.6%)**
- **Status**: accepted
- **Correctness**: 100/100 pass, max_atol=0.00909 (unchanged)
- **Distribution**: 48 workloads <0.1ms (was 45), 17 workloads >0.5ms (unchanged). Min 0.041→0.039ms.
- **Cumulative**: 0.282ms → 0.2064ms (-26.8%)
- **Learning**: The flashinfer_bench clone-per-iteration pattern renders `id`-keyed caches useless. A value-stable cache key that preserves correctness across workloads is required. For `max_s_q`, the only fully-safe zero-sync shortcut is `num_seqs == 1` (where `max_s_q = T` algebraically). For multi-seq workloads, content-based caching is not viable without a sync to read the content.

## 2026-04-21 - (T, num_seqs) Cache for max_s_q (REVERTED — correctness failure)

- **Idea**: Cache `max_s_q` by `(T, num_seqs)` across all workloads to bypass the `get_cu_seqlens_metadata` sync for BOTH single-seq and multi-seq cases.
- **Implementation**: Added `_max_s_q_cache = {}` dict in main.py; on cache miss compute via `get_cu_seqlens_metadata`, else return cached value.
- **Result**: Mean 0.1917ms on 98 workloads (would beat sub-0.2ms target) BUT 2 workloads (`5835a2bc`, `cd979341`) got **RUNTIME_ERROR** with GPU Xid 13 "Illegal Instruction Parameter" and context corruption.
- **Status**: reverted
- **Root cause**: `(T, num_seqs)` is NOT unique across the 100-workload benchmark. Two different workloads with same `(T, num_seqs)` but different per-sequence length distributions have different `max_s_q`. Cache returned wrong value for the second workload → CuTe-DSL kernel received mismatched problem_size → compiled kernel assumed wrong tile layout → OOB access or illegal instruction → GPU context corruption.
- **Learning**: Any Python-side cache that spans workload boundaries must use a collision-free key. Tensor metadata (shape, dtype, device) is insufficient for collision-free identification — the underlying data matters. Content-based keying requires a GPU→CPU sync, defeating the purpose. This is a fundamental limitation: we can cache within a workload (e.g., by tensor id/data_ptr) or across workloads by a metadata key, but not both safely without content comparison.

## 2026-04-21 - 4-way Dot Product Accumulation in Sequential Kernel (REVERTED)

- **Idea**: Break the 128-long serial FMA dependency chains (on `kS` and `out_val` accumulators in the sequential kernel) into 4 independent 32-long chains. Researcher hypothesized this would 4x the effective FMA pipeline utilization on Blackwell's 4-cycle FMA latency.
- **Implementation**: Modified `sequential_kernel.cu` loops 1 and 2 to use 4 accumulators each with `ki += 4` stride. Preserved `#pragma unroll` directives.
- **Result**: 0.2098ms → **0.2256ms (+7.5% REGRESSION)**
- **Status**: reverted
- **Correctness**: 100/100 pass, max_atol=0.00909 (unchanged)
- **Distribution**: 44 workloads <0.1ms (was 45), 17 workloads >0.5ms (unchanged). Min 0.041→0.043ms.
- **Why it failed**:
  1. NVCC likely already breaks serial FMA chains via instruction scheduling when it has enough ILP headroom — the explicit 4-way source form produced no additional benefit.
  2. Blackwell FMA latency may be shorter than the researcher's 4-cycle assumption (possibly 2-3 cycles), so the 128-long chain was NOT the bottleneck. With 4 warps × `__launch_bounds__(128,1)`, the scheduler already hid FMA latency via warp interleaving.
  3. The 4-way version holds 4 live k values + 4 accumulators simultaneously (vs 1 of each in the serial version), increasing register pressure. With 128 state registers per thread + these extras, the compiler may have spilled, slowing the loop.
  4. Min latency regressed 0.041→0.043ms, confirming the sequential path itself got slower (not just variance).
- **Learning**: Microarchitectural assumptions about FMA pipeline depth must be verified against actual SASS before making source-level changes. A source transformation that SHOULD help based on a textbook dependency-chain analysis can hurt when the compiler was already achieving the optimal schedule. For future kernel tuning, inspect SASS with `cuobjdump` rather than relying on hypothetical pipeline models.

## 2026-04-21 - Threshold Sweep + Vectorized Loads (ALL REVERTED)

Attempted two optimizations after L1 broadcast; both regressed.

| Attempt | Mean (ms) | Δ vs 0.2098 baseline |
|---|---|---|
| T=64 (baseline, re-run) | 0.2090 | -0.4% (noise) |
| T=72 | 0.2130 | +1.5% |
| T=96 | 0.2369 | +12.9% |
| T=128 | 0.2280 | +8.7% |
| Vectorized uint4 loads | 0.2181 | +4.0% |

- **Threshold sweep**: Even with the faster L1-broadcast kernel, CuTe-DSL remains faster than sequential for T>64. CuTe-DSL processes 1 chunk (128 tokens) in constant time regardless of T=65-128, while sequential scales linearly with T. Crossover is still at T≈64.
- **Vectorized uint4 loads**: Loading 8 bf16 per uint4 reduced instruction count but increased register pressure, likely causing spills. The compiler was already vectorizing implicitly.
- **Status**: all reverted, T=64 kept
- **Stable mean**: 0.209ms ±0.001ms (confirmed with 2 runs)
- **Remaining gap**: 5% to sub-0.2ms target
- **Learning**: At 0.209ms, the remaining gap is in the 17 CuTe-DSL workloads (>0.5ms each, contributing ~60% of total time). Closing the gap requires either (a) modifying the vendored CuTe-DSL kernel (infeasible), or (b) writing a custom chunked CUDA kernel with parallel-scan inter-chunk processing for T=65-512 (significant engineering, multi-session effort).

## 2026-04-20 - L1 Broadcast in Sequential Kernel (ACCEPTED)

- **Idea**: Replace shared memory k/q broadcast with direct L1 cache reads. The sequential kernel used `__shared__ float s_k[128], s_q[128]` + two `__syncthreads()` per token to broadcast k/q to all 128 threads. Since all threads need to read the same 128 k values and 128 q values, we instead have all threads read directly from global memory using `__ldg()`. With `__launch_bounds__(128,1)` only 1 block/SM, the full 228KB L1 is available. k/q vectors (256B each = 2 cache lines) hit L1 after the first warp's access.
- **Changes**: (1) Removed `__shared__ float s_k[HEAD_DIM]`, `s_q[HEAD_DIM]`, (2) Removed both `__syncthreads()` calls, (3) All k/q accesses go through `__ldg()` on global pointers, (4) k is read twice per token (kS accumulation + state update) — second read is L1 hit.
- **Result**: 0.2228ms → **0.2098ms (-5.8%)**
- **Status**: accepted
- **Correctness**: 100/100 pass, max_atol=0.00909, matched_ratio=1.0
- **Distribution**: 45 workloads <0.1ms, 17 workloads >0.5ms (CuTe-DSL, unchanged). Min 0.041ms.
- **Cumulative**: 0.282ms → 0.2098ms (-25.6%)
- **Learning**: On Blackwell with single-occupancy blocks, L1 broadcast reads are as efficient as shared memory broadcast but eliminate costly barriers. The 2 `__syncthreads()` per token were a significant fraction of sequential kernel time, especially for short sequences (T=6-64 where barrier overhead dominates over compute). The fast workloads (<0.1ms) improved from ~40 to 45 count, showing the barrier elimination primarily helps the shortest sequences.

## 2026-04-15 00:58 +09:00

- HEAD `4660730`
- Final structure in the worktree keeps the submission under `gdn_prefill_qk4_v8_d128_k_last/solution/python`, with `prefill_contract.py` handling validation and gate preparation, `main.py` handling varlen adaptation and dispatch, and `gdn_blackwell/` carrying the specialized CuTe/CUTLASS Blackwell runtime.
- Dispatch remains a strict two-path policy from `gdn_blackwell/dispatch.py`: `small` when `total_seq_len <= 1024` and `num_seqs <= 8`, otherwise `large`.
- Modal B200 quick smoke with `BenchmarkConfig(warmup_runs=1, iterations=3, num_trials=1)` and a 3-workload quick cap passed all 3 sampled workloads.
- Observed quick-smoke metrics: mean latency `0.253 ms`, mean speedup `63.74x`, min/max speedup `7.15x` / `101.52x`, max abs/rel error `1.51e-03` / `1.57e+02`.
- Trade-off note: short-workload validation is now fast and repeatable through the quick helper path, while a full-suite long-workload benchmark was intentionally skipped after the user requested quick-only benchmarking.

## 2026-04-15 01:27 +09:00

- HEAD `4660730`
- Re-ran the full 100-workload set on Modal B200 after switching the benchmark helper to the lighter `BenchmarkConfig(warmup_runs=1, iterations=3, num_trials=1)`.
- All 100 workloads passed.
- Full-workload metrics: mean latency `0.3426 ms`, mean speedup `488.44x`, min/max latency `0.1924 / 1.0576 ms`, min/max speedup `6.50x / 2149.36x`.
- Correctness envelope on this run: max abs error `9.09e-03`, max rel error `3.29e+03`.
- This full run still uses the same dispatch split: `small` when `total_seq_len <= 1024` and `num_seqs <= 8`, otherwise `large`.

## 2026-04-19 — Model C Exploration (CuTe DSL pingpong / stage tuning) — BLOCKED by scope

- **Hypothesis**: "Increasing pipeline stage depth (e.g., epi_stage 1→2, kv_stage 1→2) or enabling cluster_shape_mnk=(2,1,1) in the vendored CuTe DSL kernel will hide more non-MMA latency behind MMA, reducing per-chunk time by ~20%."
- **Phase**: Exploration (attempt 2)
- **Status**: NOT ATTEMPTED — scope blocker identified
- **Why not attempted**:
  1. Stage parameters (q_stage, kv_stage, qk_stage, epi_stage, mma_cudacore_stage, mma_qk_stage) are interdependent with mbarrier allocation sizes (line 4259-4261), TMEM column layout (line 134-149), and pipeline producer-consumer classes (line 263-333). A single integer bump risks silent deadlocks or races.
  2. `cluster_shape_mnk` change propagates through `tcgen05.make_trivial_tiled_mma` cta_group argument (line 3932), cluster_layout_vmnk (3936), TMA descriptor setup, and kernel launch (line 4391). Multi-site coordinated edit required.
  3. Pingpong warpgroup refactor (FA3-style) requires splitting the 128-thread cudacore warp group into two 64-thread sub-groups with alternating MMA/non-MMA work. Rebalance of barrier IDs, pipeline stages, and TMEM allocation. Multi-session effort.
  4. Without local B200 access, each blind modification requires a ~10min Modal benchmark to validate correctness. The exploration budget (3 iterations per §2.6) cannot absorb the search space of inter-dependent stage parameters.
- **Hypothesis 판정**: not applicable (not attempted)
- **Status decision**: Session exploration budget exhausted after Model B falsification. Further work on Model C/A requires a dedicated multi-session effort with careful incremental CuTe DSL modifications and local B200 access for rapid iteration.
- **Learning**: When a vendored kernel uses complex DSL pipelines with tightly-coupled stage/cluster/TMEM/mbarrier parameters, optimizing it remotely (no local GPU) via blind edit-and-benchmark loops is impractical within a single session budget. Future work needs either (a) local B200 for fast iteration, (b) a from-scratch rewrite in a simpler framework (not Triton — falsified), or (c) CUTLASS example code that demonstrates the exact cluster + pingpong pattern for us to port.

## 2026-04-19 — Model B Exploration (FLA Triton adaptation) (REVERTED)

- **Hypothesis**: "FLA's Triton chunk_gated_delta_rule (chunk_size=64, forward-substitution solve, SSD-adjacent structure) will reduce long-tail 17 workloads' mean latency by ≥15% because it (a) halves per-chunk MMA-inversion cost, (b) enables better SM utilization via smaller tiles."
- **Phase**: Exploration (iteration 1)
- **바꾼 structural idea**: Replace vendored CuTe-DSL chunked kernel (chunk_size=128, 2-level blocked inversion) with FLA's Triton kernel (chunk_size=64, forward-substitution). Expand q/k from 4 heads to 8 to fit FLA's non-GVA head contract.
- **성능 변화**:
  - 100-workload mean: 0.2228 → **0.6593 ms (+196%)** — MAJOR REGRESSION
  - Long-tail >0.5ms count: 17 → 72 (55 extra workloads became long-tail)
  - Max latency: 0.9 → 1.06 ms
- **Correctness**: 100/100 pass (atol=1, rtol=0.3)
- **Hypothesis 판정**: **FALSIFIED**
- **근거**:
  1. Triton's Blackwell (sm_100a) backend does NOT use tcgen05 WGMMA as efficiently as CuTe DSL. The vendored kernel's 10+ tcgen05 MMAs per chunk are faster than FLA's Triton `tl.dot`.
  2. FLA's chunk_size=64 produces 2x more chunks for long sequences (128 vs 64 chunks at T=8192), doubling inter-chunk sequential overhead.
  3. FLA is not GVA-aware; q/k expansion 4→8 heads doubles the q·k^T MMA work.
  4. Regression is consistent across ALL T>64 workloads, not isolated — indicates fundamental backend mismatch, not tuning issue.
- **Status**: REVERTED. USE_FLA default=0; FLA adapter code kept in main.py:227-249 as dead code for future reference.
- **다음 iteration**: 
  - **버림**: FLA-Triton path (falsified, cannot retry per §2.7).
  - **계속**: Pivot to Model C (FA3 pingpong warpgroup inside CuTe DSL) or Model A (TFLA intra-chunk sub-tiling). Both require deep modification of the vendored kernel (gdn.py). Next iteration: analyze feasibility of a minimal pingpong refactor.
- **Learning**: When the baseline is a highly-tuned CuTe DSL kernel on Blackwell, replacing it with a Triton port rarely wins — Triton's codegen for tcgen05 is still maturing. The remaining 10% gap must be found INSIDE the vendored kernel (warp-group layout, barrier relaxation, cluster usage) rather than via replacement.

## 2026-04-19 - max_s_q-based dispatch (REVERTED, all three variants)

Explored per-sequence-aware dispatch to route multi-seq batches with short individual sequences away from CuTe-DSL. Three variants tested:

| Variant | Mean (ms) | Δ vs 0.2228 baseline | Status |
|---|---|---|---|
| SEQ_THRESHOLD=128 (max_s_q<=128 → sequential) | 0.2314 | +3.9% | REGRESSION |
| SEQ_THRESHOLD=64 (max_s_q<=64 → sequential) | 0.2211 | -0.8% | noise |
| MULTI_SEQ_THRESHOLD=128 (multi-seq only) | 0.2234 | +0.3% | noise |

- **Idea**: Current dispatch `T <= 64` sends multi-seq batches (e.g., 10 sequences each of length 20, T=200, max_s_q=20) to CuTe-DSL unnecessarily. Routing on max_s_q captures these.
- **Why SEQ_THRESHOLD=128 regressed**: Single-seq workloads with T=65-128 moved to sequential where per-token cost × max_s_q (~5.7us/token × 128 = 730us) exceeds CuTe-DSL's single-chunk cost (~100-200us).
- **Why SEQ_THRESHOLD=64 was neutral**: The workload distribution (median T=139, median num_seqs=2) contains few multi-seq batches with max_s_q ≤ 64 AND T > 64. Capture set is essentially empty.
- **Why MULTI_SEQ_THRESHOLD=128 was neutral**: Same — very few multi-seq batches with max_s_q in (64, 128] exist in the test set.
- **Status**: all three reverted
- **Learning**: The current 100-workload benchmark's shape distribution doesn't expose the dispatch-heuristic lever. To reach sub-0.2ms, the remaining 10% reduction must come from either (a) speeding up the CuTe-DSL kernel itself (4500-line vendored code; previously enabled persistent scheduling but further tuning requires deep modification), or (b) a custom chunked CUDA kernel replacing CuTe-DSL for medium-length workloads (significant engineering, uncertain benefit).

## 2026-04-19 - Persistent scheduling in CuTe-DSL kernel (ACCEPTED)

- **Idea**: The vendored GDN kernel raised `ValueError("Task 5 runtime only supports non-persistent mode")` but the tile scheduler, barrier synchronization (epi_load_sync_bar_id), and work-tile loops all implement persistent mode. Removing the artificial restriction enables grid reduction from `(1, 8, num_seqs)` to `(min(SM_count, num_tiles), 1, 1)`.
- **Changes**: (1) Removed ValueError in GDN.__init__, (2) default `is_persistent=True`, (3) changed `_compute_grid` to pass `True` to `create_gdn_static_tile_scheduler_params`.
- **Result**: First run 0.2175ms (-2.4%, likely warm-cache outlier); stable runs ~0.222-0.224ms (tied with previous). Max latency reduced from 0.94ms → 0.90ms across runs, suggesting real improvement for the slowest workloads.
- **Status**: accepted (correctness preserved; no regression; potential upside for multi-sequence workloads)
- **Cumulative**: 0.282ms → 0.2228ms (-21.0%)
- **Learning**: The vendored kernel had all infrastructure for persistent mode but the author disabled it (possibly for safety). Persistent mode helps multi-sequence workloads by reducing grid size and improving scheduling locality, but the gain is small for this benchmark because single-sequence workloads only have 8 tiles (same as non-persistent).

## 2026-04-19 - CUDA graph capture for long path (REVERTED)

- **Idea**: Capture both fused_gate_kernel and TVM-FFI compiled_gdn launches into a CUDA graph via `cuStreamBeginCapture`/`cuStreamEndCapture`, then replay on subsequent iterations. Should eliminate ~10-15us TVM FFI + launch overhead per long-path call.
- **Implementation**: Three-tier dispatch (direct launch → capture on 2nd call → replay on 3rd+). Pre-allocated graph input buffers with `.copy_(non_blocking=True)` to update inputs before replay.
- **Result**: Capture failed for 59/60 shapes. Mean unchanged (0.2235ms ≈ baseline 0.2229ms).
- **Status**: reverted
- **Why it failed**: TVM FFI's `compiled_gdn(...)` call does host-side work (Python→C++ argument marshaling, dict lookups inside the compiled PackedFunc) that breaks CUDA stream capture. `cuStreamBeginCapture` returned SUCCESS, but the subsequent compiled_gdn call caused the stream to exit capture mode with an invalid state, so `cuStreamEndCapture` failed.
- **Learning**: CUDA graphs are incompatible with CuTe-DSL + TVM FFI compiled kernels. To use graphs, would need to extract the raw CUDA module/function handle from the compiled object (not exposed by CuTe-DSL's public API) and call `cuLaunchKernel` directly. Infeasible without deep TVM FFI internals modification.

## 2026-04-18 (session 3) - Threshold re-tune after dispatch opts (ACCEPTED)

- **Idea**: With the fast-launch and bundle-cache reducing CuTe-DSL per-call overhead, the optimal sequential/CuTe-DSL crossover may have shifted.
- **Sweep results**: T=16→0.2265, T=32→~0.225, T=64→**~0.223**, T=128→0.2265
- **Result**: 0.2265ms → 0.2229ms (-1.6%)
- **Status**: accepted, THRESHOLD=64 is new optimum
- **Cumulative**: 0.282 → 0.2229ms (-21.0%)
- **Remaining gap to 0.2ms**: 11%

## 2026-04-18 (session 2) - Deep Python dispatch + fast-launch (ACCEPTED, cumulative)

Second optimization pass targeting all remaining Python-side overhead.

| Step | Change | Mean (ms) | Δ vs prior | Δ vs baseline |
|---|---|---|---|---|
| starting | After session 1 (4D views cached) | 0.2629 | — | -6.8% |
| 1 | Fast-launch: _SeqArgs/_GateArgs pre-allocated ctypes + direct drv.cuLaunchKernel | 0.2365 | -10.0% | -16.1% |
| 2 | get_gdn_bundle fast-path: cached (compiled_gdn, output, output_state) tuple | 0.2347 | -0.8% | -16.8% |
| 3 | Skip unsqueeze at call site (data_ptr identical for 3D and 4D view) | 0.2277 | -3.0% | -19.3% |
| 4 | Cache output.squeeze(0) in bundle | 0.2287 | noise | -18.9% |
| 5 | Cache data_ptr() for stable gate_log/beta/output/output_state tensors | **0.2265** | -1.0% | **-19.7%** |

- **Final result**: 0.282ms → **0.2265ms (-19.7%)**
- **Status**: accepted (100% correctness preserved)
- **Target**: 0.2ms (still 13% away)
- **Key techniques**:
  1. **Pre-allocated ctypes + direct cuLaunchKernel**: Biggest win. The generic `launch()` helper was creating ctypes objects per call; a pre-allocated struct of ctypes slots with a pre-built `c_void_p` array eliminates this. Saves ~27us per NVRTC kernel launch.
  2. **Bundle cache**: Cache `(compiled_gdn, output, output_state, problem_size, scale, output_3d, output_ptr, output_state_ptr)` keyed by (problem_size, scale, path_name). Skips the chunk_gated_delta_rule wrapper on cache hits.
  3. **Skip redundant view operations**: `.unsqueeze(0)` and `.squeeze(0)` are cheap but non-free. The kernel was compiled against 4D iterators but only uses data_ptr at launch — so 3D tensors work identically.
  4. **data_ptr caching**: For tensors that live in shape-keyed caches (never freed), cache their device pointer once.

- **Why sub-0.2ms wasn't reached**: The 17 CuTe-DSL long-path workloads contribute ~13ms out of 22.6ms total. Mean of fast workloads is already 0.115ms. Reaching 0.2ms requires cutting CuTe-DSL time by ~40%, which needs GPU-kernel modifications (chunk-level parallelism via WY decomposition + cooperative scheduling) in the 4500-line vendored kernel.

## 2026-04-18 (session 1) - Initial Python-dispatch optimization pass (ACCEPTED, cumulative; superseded by session 2)

Iterative optimizations to eliminate Python-side overhead, guided by the insight that CUDA event timing wraps the entire `run()` call (every microsecond of dispatch appears as GPU idle time).

| Step | Change | Mean (ms) | Δ |
|---|---|---|---|
| baseline | THRESHOLD=128 tuned kernel | 0.282 | — |
| 1 | main.py: cached CUstream, pre-built ctypes, get_cu_seqlens_metadata for max_s_q, pre-allocated gate_log/beta, pre-computed _DEFAULT_SCALE | 0.2732 | -3.1% |
| 2 | gdn.py: cached _cached_stream in chunk_gated_delta_rule | 0.2723 | -0.3% |
| 3 | gdn.py: _output_cache, _state_cache keyed by (shape, dtype, device) | 0.2662 | -2.2% |
| 4 | main.py: _seq_output_cache, _seq_state_cache for T<=128 path | 0.2639 | -0.9% |
| 5 | main.py: pre-computed 4D unsqueeze views for gate_log/beta | 0.2629 | -0.4% |

- **Final result**: 0.282ms → 0.2629ms (**-6.8%**)
- **Status**: accepted (all safe, correctness preserved across 100 workloads)
- **Target**: 0.2ms (still 24% away)
- **Learnings**:
  - Python dispatch is measured. Even `.item()`, `torch.empty()`, `drv.CUstream()` per call add up.
  - `get_cu_seqlens_metadata` cache hits across warmup+iterations of the same workload — eliminates GPU sync.
  - Tensor caching (output, state, gate, beta) by shape saves torch.empty overhead on repeat calls.
  - 4D unsqueeze views can be pre-computed and cached to save per-call view creation.
  - Min latency dropped from 0.083ms → 0.064ms (~23% improvement on short workloads), showing per-call overhead was the dominant cost there.
- **Why 0.2ms wasn't reached**: The 17 workloads >0.5ms each (CuTe-DSL long path) contribute ~13ms out of ~26ms total. Even eliminating ALL Python overhead (~2-3ms savings max) leaves us at ~0.24ms. Reaching 0.2ms requires speeding up the CuTe-DSL GPU kernel itself — which is infeasible without major surgery on the vendored 4500-line kernel. The CuTe-DSL kernel uses only 8 SMs for single-sequence workloads (grid is (1,8,1)) because chunks are processed sequentially within each block rather than in parallel; fundamentally addressing this would require implementing a custom chunked kernel with WY representation for inter-chunk parallelism.
- **Future directions (not attempted here)**:
  1. Hand-written CUDA chunked kernel replacing CuTe-DSL for medium seq (128<T<1024)
  2. cuLA's KDA kernel as a potential drop-in replacement
  3. CUDA graphs via torch.cuda.CUDAGraph (unclear if compatible with TVM FFI)
  4. Enable persistent scheduling in CuTe-DSL (currently explicitly disabled in vendored code)

## 2026-04-18 - Double-buffered sequential kernel (REVERTED)

- **Idea**: Reduce per-token sync in NVRTC sequential kernel via double-buffered k/q shared memory + scalar (v/a/b) prefetch. Cut from 2 syncs/token to 1, overlap next-token load with current compute.
- **Result**: 0.282ms → 0.296ms (+4.9% regression)
- **Status**: reverted
- **Why it didn't work**: The sequential kernel only handles T≤128 workloads (already fast, 0.05-0.15ms each). The added complexity (preload phase, buffer indexing, conditional sync) added overhead that dominates savings for very short sequences (T=6-32). Meanwhile, the 18 workloads >0.5ms (all in CuTe-DSL path) remain the bottleneck — they contribute ~60% of total time and were unaffected by this change.
- **Learning**: To hit sub-0.2ms, optimization must target the long-sequence CuTe-DSL path (>0.5ms workloads), not the already-fast sequential path. Possible directions:
  1. Hand-written CUDA chunked kernel replacing CuTe-DSL for medium sequences (T=128-2048)
  2. Eliminate Python/GPU-sync overhead in the long-path dispatch
  3. Investigate whether the CuTe-DSL kernel has tunable parameters (chunk_size, tile sizes) that benefit our specific problem shape
