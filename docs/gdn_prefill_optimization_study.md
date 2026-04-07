# GDN Prefill Kernel Optimization Study

## Scope

This document analyzes the current `gdn_prefill_qk4_v8_d128_k_last` implementation and lists optimization directions that are worth trying on NVIDIA B200.

Investigation tracks:

1. B200 / Blackwell hardware characteristics relevant to this kernel
2. GDN prefill workload characteristics in this repo
3. Current CUDA kernel structure and bottlenecks
4. Optimization methods synthesized from 1-3
5. Applicability assessment against the current implementation

## Agent Team

The analysis was split into three roles:

1. `Hardware agent`: B200 / Blackwell tuning characteristics and CUDA feature survey
2. `Workload agent`: GDN prefill math, tensor layout, workload distribution, and algorithmic constraints
3. `Kernel agent`: current CUDA implementation, bottlenecks, and implementation feasibility

## Current Repo State

The real work-in-progress prefill kernel is:

- `gdn_prefill_qk4_v8_d128_k_last/solution/cuda/kernel.cu`

The top-level `solution/cuda/kernel.cu` is still only a template and is not relevant for the current prefill implementation.

Current optimization log status:

- `logs/prefill/optimization_log.md`: header only, no recorded iterations
- `logs/decode/optimization_log.md`: exists, but this document focuses on prefill

This means the present optimization plan must be inferred from code and workload definitions rather than prior measured experiments.

## GDN Prefill Workload Characterization

### Math and state structure

From the definition:

- Q heads: 4
- K heads: 4
- V heads: 8
- Head size: 128
- State layout: `[num_seqs, num_v_heads, 128, 128]` in `float32`
- Output layout: `[total_seq_len, num_v_heads, 128]` in `bfloat16`

Reference implementation structure:

1. Compute `g = exp(-exp(A_log) * softplus(a + dt_bias))`
2. Compute `beta = sigmoid(b)`
3. For each sequence and timestep, decay old state
4. Compute `old_v = k @ old_state`
5. Update with rank-1 form using `k^T @ residual`
6. Emit `output = scale * q @ state_new`

Important consequence:

- Prefill is sequential along time within each sequence. There is no legal parallelization across timesteps of the same sequence without reformulating the algorithm.
- The kernel is dominated by repeated operations on a `128 x 128` fp32 state per `(sequence, v_head)`.

### GVA mapping

The workload uses grouped value attention style mapping:

- `num_q_heads = num_k_heads = 4`
- `num_v_heads = 8`
- Therefore each q/k head serves two v-heads

This creates a reuse opportunity:

- For two `v_head`s that map to the same `qkh`, the same `q` and `k` vectors are read and converted twice in the current implementation.

### Workload distribution

The workload file contains 100 cases. Local aggregation of `mlsys26-contest/workloads/gdn/gdn_prefill_qk4_v8_d128_k_last.jsonl` shows:

- `total_seq_len`: min 6, max 8192, median 139, mean 1830.65
- `num_seqs`: min 1, max 57, median 2, mean 7.97
- 16 workloads have `total_seq_len = 8192`
- 18 workloads have `total_seq_len >= 4000`
- 16 workloads have `num_seqs >= 16`
- 56 workloads have `total_seq_len < 256`

Interpretation:

- There is a long tail of short cases, so launch overhead and over-specialized kernels can matter.
- But a meaningful portion of the evaluation set is large enough that steady-state throughput on the recurrent state update likely dominates total runtime.
- A good solution probably needs at least two regimes:
  - short / low-parallelism path
  - long / high-parallelism path

## Current Kernel Strategy

The current CUDA prefill kernel launches:

- one block per `(seq, v_head)`
- 128 threads per block
- one thread per `K` dimension

The full `128 x 128` state matrix for a `(seq, v_head)` tile is loaded into shared memory once, updated token-by-token, and written back at the end.

### What the kernel does well

1. It avoids reloading the recurrent state from global memory at every timestep.
2. It keeps the time-recursive update within a single block, which is semantically simple and correct.
3. It matches the natural `K=128` thread mapping for dot products over the key dimension.

### What the kernel does poorly

1. It uses a large dynamic shared memory allocation:
   - state: `128 * 128 * 4 = 65536` bytes
   - scratch: about `1040` bytes
   - total: about `66.6 KB` per block
2. It computes `k @ state` using a loop over all 128 output rows, with a block-wide reduction and full-block synchronization for every row.
3. It computes `q @ state_new` with another full loop over all 128 rows, again with repeated full-block synchronization.
4. It serializes the whole sequence inside one block, so each `(seq, v_head)` contributes only one block of parallelism regardless of sequence length.
5. It reloads `q`, `k`, and gates separately for the two `v_head`s that share the same `qkh`.

## Bottleneck Analysis Against Current Code

### 1. Synchronization-heavy inner loops

The worst bottleneck is repeated block-wide synchronization in the innermost loops.

In `gdn_prefill_qk4_v8_d128_k_last/solution/cuda/kernel.cu`:

- lines 107-122 compute `s_kdot[vi]` for each `vi`
- lines 125-143 compute output for each `vi`

For each token:

- 128 iterations for `k @ state`
- 128 iterations for `q @ state_new`
- multiple `__syncthreads()` inside both loops

That is structurally expensive even before considering math throughput.

### 2. Under-utilization due to one block per `(seq, v_head)`

The current mapping gives grid size:

- `num_seqs * 8`

For small `num_seqs`, the GPU is underfilled. This is acute because many workloads have `num_seqs <= 3`.

For long sequences, the work remains trapped in a single block per head because the sequential time loop is inside the block:

- lines 93-145

So long sequences do not create more parallelism, only longer block residency.

### 3. Shared-memory pressure limits occupancy

The kernel opts into about 66.6 KB dynamic shared memory per block:

- lines 177-180

On Blackwell compute capability 10.0, the shared-memory-per-SM ceiling is 228 KB and per-block ceiling is 227 KB. This kernel fits easily, but the per-block footprint still restricts active blocks per SM.

This is not catastrophic on B200, but it means occupancy is constrained before register pressure is even considered.

### 4. Scalar transcendental gate computation in the token loop

Per token, per block, the kernel computes:

- `softplus`
- `expf`
- `sigmoid`

This is not the dominant cost relative to the `128 x 128` state math, but for short sequences it is not free. It also makes the short-case path more latency-sensitive.

### 5. Duplicate q/k work across paired v-heads

The mapping `qkh = vh / 2` means `vh=0,1` share the same q/k head, `vh=2,3` share another, etc.

In the current kernel:

- `q` and `k` are loaded separately by each `v_head` block
- the same bf16-to-fp32 conversions are repeated

That is a clean reuse opportunity if the execution mapping changes.

### 6. No measurement infrastructure for prefill yet

There is no recorded prefill optimization history and no checked-in prefill profiling results:

- no NCU snapshots
- no occupancy reports
- no correctness/perf deltas by attempt

This is a process bottleneck rather than a kernel bottleneck, but it matters because several candidate optimizations require quick reject/accept cycles.

## B200 / Blackwell Characteristics That Matter Here

From the current Blackwell tuning guidance:

1. Compute capability 10.0 supports up to 64 resident warps per SM and 32 thread blocks per SM.
2. Shared memory per SM is 228 KB, with up to 227 KB addressable by a single block after opt-in.
3. Blackwell supports thread block clusters and distributed shared memory.
4. B200 supports runtime shared-memory carveout tuning.
5. Blackwell keeps the Hopper-style model where advanced architecture-specific or family-specific features may require dedicated compiler targets.

Why this matters here:

- The current kernel already relies on large dynamic shared memory, so Blackwell is a better fit than older architectures.
- However, this kernel is not currently exploiting the features that would differentiate B200 from a conventional shared-memory implementation.
- In particular, the current kernel does not exploit tensor-core MMA style decomposition, clustered execution, or family-specific code generation.

## Candidate Optimization Methods

The list below combines hardware opportunities and workload-specific structure.

### A. Split the kernel into at least two execution regimes

Idea:

- Use one kernel strategy for short sequences / small `num_seqs`
- Use another for long sequences / large `num_seqs`

Why it fits:

- The workload distribution is bimodal enough to justify specialization.
- The current single strategy is poor for both extremes.

Applicability now:

- High

Expected implementation effort:

- Moderate

Comments:

- This is likely the safest first structural improvement.
- Even a simple threshold on `seq_len` or `total_seq_len / num_seqs` is worth trying.

### B. Warp-specialize the two inner loops to reduce full-block barriers

Idea:

- Replace block-wide reductions plus `__syncthreads()` per row with warp-scoped decomposition and fewer full-block sync points.
- Use multiple warps to cooperatively process different `vi` rows.

Why it fits:

- The current kernel spends a large fraction of its control flow in repeated reduction + barrier patterns.

Applicability now:

- High

Expected implementation effort:

- Moderate to high

Comments:

- This is the most directly applicable optimization without changing algorithm semantics.
- The current `(128 threads, K=128)` mapping is already warp-aligned; the main issue is decomposition.

### C. Tile the state matrix instead of iterating one `vi` row at a time

Idea:

- Process the `128 x 128` state in tiles, for example `64 x 64`, `64 x 128`, or `32 x 128`.
- Accumulate partial `k @ state` and `q @ state_new` over tiles.

Why it fits:

- The kernel currently serializes the `V=128` dimension in software loops.
- Tiling can expose more instruction-level parallelism and reduce synchronization frequency.

Applicability now:

- High

Expected implementation effort:

- High

Comments:

- This is the natural path toward a more tensor-core-friendly decomposition as well.

### D. Fuse the paired `v_head`s that share the same q/k head

Idea:

- Compute two `v_head`s together inside one block or cooperative group because they share the same `q` and `k`.

Why it fits:

- `NUM_V_HEADS = 8`, `NUM_Q_HEADS = NUM_K_HEADS = 4`
- Each `qkh` serves exactly two value heads

Benefits:

- Reuse of `q` / `k`
- Less redundant bf16 conversion
- Better arithmetic intensity

Applicability now:

- Medium to high

Expected implementation effort:

- Moderate

Comments:

- This is one of the cleanest workload-specific optimizations.
- It may increase register or shared-memory pressure, so it should be paired with occupancy checks.

### E. Precompute or hoist gate terms where possible

Idea:

- Reduce recomputation of `expf`, `softplus`, and `sigmoid`
- Possibly precompute `g` and `beta` in a lightweight preprocessing pass or fuse more cheaply into the main loop

Applicability now:

- Medium

Expected implementation effort:

- Low to moderate

Comments:

- Useful mostly for short-sequence latency.
- Not likely the top throughput win for long-sequence heavy cases.

### F. Use vectorized loads and conversion-friendly packing for q/k/v

Idea:

- Load `bf16` values in packed form where alignment permits
- Reduce instruction count around scalar conversion

Applicability now:

- Medium

Expected implementation effort:

- Moderate

Comments:

- This is a secondary optimization and should follow structural fixes.

### G. Rebuild around tensor-core MMA / tile programming

Idea:

- Reformulate the `k @ state` and `q @ state_new` operations as matrix fragments suitable for tensor-core execution.
- Potentially use CUTLASS/CUTE-style tile decomposition or lower-level family-specific Blackwell paths.

Why it is attractive:

- The recurrent core repeatedly performs dense dot-product style math on fixed `128 x 128` state blocks.

Why it is hard:

- The update is not just GEMM; it interleaves decay, rank-1 update, and output projection.
- A full rewrite is required.

Applicability now:

- Medium

Expected implementation effort:

- Very high

Comments:

- This is the highest-upside path if the goal is leaderboard-level performance.
- It is not the best first move if the current implementation has not yet been profiled and stabilized.

### H. Use thread block clusters / distributed shared memory

Idea:

- Spread one logical `(seq, qkh)` or `(seq, qkh pair)` computation across multiple blocks in a cluster and share intermediate state through distributed shared memory.

Why it is interesting on B200:

- Blackwell supports clusters and distributed shared memory.
- B200 allows nonportable cluster size 16 with opt-in.

Why it is difficult:

- The current recurrence is naturally local to a single state block.
- Clustered execution only pays off if we repartition work across blocks carefully.

Applicability now:

- Low to medium

Expected implementation effort:

- Very high

Comments:

- This is a second-wave experiment, not a first-wave optimization.

### I. Use L2 persistence / cache control for streamed q/k/v or state-adjacent metadata

Idea:

- Mark frequently reused data for L2 persistence where CUDA APIs and access patterns make sense.

Applicability now:

- Low to medium

Expected implementation effort:

- Moderate

Comments:

- The recurrent state is already staged in shared memory for the whole sequence, so the best L2 candidates are not obvious.
- This is more likely a minor gain than a major one.

### J. Async copy / TMA-style staging for state tiles

Idea:

- Use asynchronous global-to-shared transfers for tiled state movement and overlap data movement with compute.

Applicability now:

- Medium in principle
- Low against the current codebase

Expected implementation effort:

- High

Comments:

- This becomes more compelling only after the kernel is tiled.
- In the current implementation, state is copied once, then reused for the whole sequence; there is not enough structured tile streaming yet to make this the first optimization.

## Applicability Matrix

| Method | Expected upside | Effort | Fit to current code | Notes |
|---|---:|---:|---|---|
| Two-regime kernel split | High | Medium | Strong | Best first structural move |
| Warp-specialized inner loops | High | Medium-High | Strong | Directly attacks current barrier-heavy loops |
| State tiling | High | High | Strong | Needed before more advanced pipelines |
| Pair two v-heads per q/k head | Medium-High | Medium | Strong | Exploits workload-specific reuse |
| Gate precompute / hoisting | Medium | Low-Medium | Strong | Helps short cases more than long ones |
| Vectorized q/k/v loads | Medium | Medium | Medium | Secondary optimization |
| Tensor-core MMA rewrite | Very High | Very High | Medium | Likely required for top-end performance |
| Thread block clusters / DSM | Medium | Very High | Weak-Medium | B200-specific but structurally invasive |
| L2 persistence hints | Low-Medium | Medium | Weak | Likely incremental only |
| Async copy / TMA pipeline | Medium-High | High | Weak-Medium | Depends on first introducing tiles |

## What Is Immediately Applicable to the Current Kernel

These are the optimizations that can be attempted without first redesigning the whole kernel around a new programming model:

1. Split into short-path and long-path kernels
2. Reduce full-block synchronizations in the two `vi` loops
3. Fuse paired `v_head`s sharing the same `qkh`
4. Add vectorized input loads and cheaper conversion patterns
5. Hoist gate-related scalar work where profitable

These are likely to require a structural rewrite first:

1. Tensor-core-centered implementation
2. Async tiled pipeline
3. Thread block cluster / distributed shared memory decomposition

## Recommended Optimization Roadmap

### Phase 0: Measurement baseline

Before changing math structure, add actual prefill measurements:

1. correctness on representative short and long workloads
2. latency breakdown by workload bucket
3. Nsight Compute capture for one short case and one long case
4. occupancy, smem usage, and barrier stall inspection

Without this, the team is optimizing blind.

### Phase 1: Safe structural improvements

Recommended first experiments:

1. kernel split by workload regime
2. paired-`v_head` execution
3. warp-specialized reduction/output loops
4. vectorized q/k/v access cleanup

This phase has the best effort-to-upside ratio.

### Phase 2: Stateful tile redesign

If Phase 1 is insufficient:

1. redesign around state tiling
2. reduce barrier count sharply
3. create a form that can later use async staged copies

### Phase 3: B200-specific aggressive path

Only after a tiled implementation exists:

1. evaluate async copy / TMA-like staging
2. evaluate tensor-core fragment mapping
3. evaluate clustered multi-block decomposition if single-block saturation remains insufficient

## Gaps in Current Evidence

Current repo gaps:

1. no prefill optimization history in `logs/prefill/optimization_log.md`
2. no prefill Nsight Compute outputs
3. no workload bucketing or summary scripts checked in
4. no prefill correctness/performance regression tests
5. no A/B experiments against the FlashInfer baseline documented in repo

These should be fixed before major kernel rewrites, otherwise it will be difficult to know which ideas actually help.

## Bottom Line

The current prefill kernel is a correct shared-memory baseline, not a B200-tuned implementation.

The best methods to try next are:

1. split the kernel into short and long execution regimes
2. fuse the two `v_head`s that share one q/k head
3. redesign the inner loops to reduce full-block barriers
4. move toward tiled state processing

The highest-upside but highest-risk path is a tensor-core-oriented tiled rewrite. B200 hardware features make that attractive, but the current kernel is not yet structurally ready to benefit from the more advanced Blackwell-specific mechanisms.

## Sources

Repo-local sources:

- `gdn_prefill_qk4_v8_d128_k_last/solution/cuda/kernel.cu`
- `gdn_decode_qk4_v8_d128_k_last/solution/cuda/kernel.cu`
- `mlsys26-contest/definitions/gdn/gdn_prefill_qk4_v8_d128_k_last.json`
- `mlsys26-contest/workloads/gdn/gdn_prefill_qk4_v8_d128_k_last.jsonl`
- `logs/prefill/optimization_log.md`
- `EVALUATION.md`

External sources:

- NVIDIA Blackwell Tuning Guide: https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html
- CUDA 12.8 Blackwell support blog: https://developer.nvidia.com/blog/cuda-toolkit-12-8-delivers-nvidia-blackwell-support/
- Blackwell family-specific feature guidance: https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/
- Blackwell Ultra architecture overview: https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/
