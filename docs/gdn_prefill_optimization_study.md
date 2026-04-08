# GDN Prefill Optimization Study

Updated: 2026-04-08

## Scope

This document replaces the older stale study and re-analyzes the current
`gdn_prefill_qk4_v8_d128_k_last` implementation using:

- the current kernel source
- the current prefill optimization log and benchmark history
- the current workload set
- the current Blackwell/B200 baseline signal in the contest repo
- official Blackwell / CUDA documentation and the Gated Delta Networks paper

The goal is not to repeat every historical idea. The goal is to identify what is
still worth trying now.

## Agent Team

The analysis was split into four roles:

1. `Hardware agent`
   - B200 / Blackwell features, compiler constraints, and what is realistic in
     the current codebase
2. `Model agent`
   - GDN recurrence, workload structure, grouped-value-attention reuse, and
     chunking constraints
3. `Kernel agent`
   - current CUDA implementation, live dispatch paths, and ideas already ruled
     out by the optimization log
4. `Integrator`
   - merge the above into a ranked methodology list and applicability assessment

## Sources Reviewed

### Local repo

- `gdn_prefill_qk4_v8_d128_k_last/solution/cuda/kernel.cu`
- `logs/prefill/optimization_log.md`
- `logs/prefill/bench_history.jsonl`
- `mlsys26-contest/definitions/gdn/gdn_prefill_qk4_v8_d128_k_last.json`
- `mlsys26-contest/workloads/gdn/gdn_prefill_qk4_v8_d128_k_last.jsonl`
- `mlsys26-contest/solutions/baseline/gdn/gdn_prefill_qk4_v8_d128_k_last/flashinfer_wrapper_123ca6.json`
- `docs/gemini_gdn_prefill_b200.md`
- `scripts/profile_kernel.py`

### External references

- NVIDIA Blackwell Tuning Guide:
  https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html
- CUDA C++ Programming Guide:
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- CUDA cluster launch control:
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cluster-launch-control
- CUTLASS Blackwell functionality:
  https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html
- NVIDIA Blackwell family-specific feature blog:
  https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/
- Gated Delta Networks paper:
  https://arxiv.org/abs/2412.06464
- Official GatedDeltaNet repo:
  https://github.com/NVlabs/GatedDeltaNet

## Current Repo State

### Active implementation

The real prefill kernel is:

- `gdn_prefill_qk4_v8_d128_k_last/solution/cuda/kernel.cu`

The current best committed quick result in the log is:

- Iteration 29
- `mean_speedup = 170.18544`
- `mean_latency_ms = 3.05327`

The latest attempt after that was:

- Iteration 30
- `q/k` pair-load vectorization on the micro path
- correctness passed
- weighted speedup regressed
- decision: revert

### Optimization history is now rich, not empty

The prefill log now contains 30 measured iterations. That matters because many
candidate ideas that looked attractive in older studies have already been tested
and rejected.

### Live dispatch paths under the current wrapper

The current host wrapper dispatches by `num_seqs` only:

- `N <= 2` effectively lands on the `2-row micro` path
- `N >= 3` effectively lands on the `16-row default` path

Applying the current wrapper logic to the 100 contest workloads gives:

- `61/100` workloads on `micro2`
- `39/100` workloads on `default16`

For the actual workload set, the `4-row` and `8-row` kernels are effectively dead
paths today.

### Profiling caveat

The checked-in `scripts/profile_kernel.py` does not mirror the current host
wrapper dispatch. For templated prefill kernels it still hardcodes an older
`gdn_prefill_kernel<8|4|2|1>` launch scheme inside the profiling harness instead
of calling the current wrapper logic.

Implication:

- fresh NCU runs from that script are still useful for broad launch/occupancy
  intuition
- but they are not an exact profile of the current end-to-end dispatch behavior

For current-path decisions, the latest notes embedded in
`logs/prefill/optimization_log.md` are more reliable than the profiling harness as
it stands.

## Current Implementation Summary

### Common math

Per token and per value head, the kernel computes:

1. `g = exp(-exp(A_log) * softplus(a + dt_bias))`
2. `beta = sigmoid(b)`
3. `temp = g * state`
4. `kdot = k @ temp`
5. `residual = beta * (v - kdot)`
6. `state_new = temp + k^T @ residual`
7. `output = scale * (q @ state_new)`

This recurrence is sequential within each sequence. There is no legal naive
token-parallel prefill path without reformulating the algorithm.

### Short path: `2-row micro warp-per-row`

The current short path is the Iteration 29 kernel:

- one warp owns one state row
- each lane owns four `K` columns
- the kernel uses warp shuffles instead of cross-warp shared-memory reductions
- this removed the biggest low-`N` barrier bottleneck and produced the current
  largest measured improvement

This is the strongest part of the current implementation.

### Long path: `16-row shared-memory scalar kernel`

The current long path still:

- stages a `16 x 128` state tile in shared memory
- computes `k @ temp` via cross-warp reduction to shared memory
- computes `q @ state_new` via another reduction path
- serializes the final accumulation through `tid == 0`

It is much better than the old full-state kernel, but it is still a scalar
recurrent kernel, not a Blackwell-native tiled kernel.

## What The History Already Established

### Proven wins

These ideas already worked:

- row tiling instead of one CTA owning the full `128 x 128` state
- deeper low-`N` fallback so short workloads get more CTAs
- removing a redundant end-of-timestep CTA barrier
- rewriting the `2-row` path into a warp-per-row micro kernel

### Repeated regressions or dead ends

These are currently low-ROI to revisit without a materially different structure:

- more `4/8/1/32-row` launch-heuristic tuning
- vectorization-only patches for state I/O or micro `q/k` loads
- register tiling in the current micro or `4-row` scalar paths
- `launch_bounds` tuning on the current templated long path
- reviving the currently dead `4-row` or `8-row` kernels

### Correctness failures

These should be treated as unsafe until the math is reformulated more carefully:

- algebraic output fusion of the form
  `q @ state_new = q @ temp + (q · k) * residual`

That family already failed correctness multiple times in the quick benchmark log.

## B200 + GDN Synthesis

### What GDN says

GDN prefill is fundamentally a recurrent state update. That means:

- direct timestep parallelization is not available
- the only path to substantially more GPU parallelism is a chunked reformulation
- grouped-value attention (`q_heads = k_heads = 4`, `v_heads = 8`) creates a real
  reuse opportunity because sibling value heads share the same `q/k` head

### What B200 says

Blackwell/B200 offers strong upside for:

- chunked tile-level matrix pipelines
- TMA-style bulk async movement
- TMEM / `tcgen05` / CUTLASS-CuTe Blackwell paths
- persistent scheduling and cluster-level execution once work is tiled

But those features are not free wins on a scalar recurrent raw-CUDA loop. They
become valuable only after the kernel is rewritten to expose tile-level work.

### Strong directional signal from the contest baseline

The contest baseline for this exact definition is not a token-serial scalar kernel.
It is a Blackwell-specific `chunk_gated_delta_rule` implementation with:

- chunk size 128
- persistent and non-persistent modes
- CuTe / CUTLASS / SM100 Blackwell machinery

That is the strongest available repo-local signal for the long-term architecture.

## Best Current-Code Experiments

These are ideas that are still worth trying without throwing away the current CUDA
kernel family.

### 1. Precompute `g` and `beta` once per `(t, v_head)`

Why it is interesting:

- the current row-tiled kernels recompute the same gate values independently in
  every row-tile CTA
- on the live `2-row micro` path, that means the same `(t, vh)` gate is computed
  across 64 row tiles
- on the `16-row` path, it is still recomputed across 8 row tiles
- B200 is especially asymmetric between massive math throughput and much weaker
  transcendental/SFU throughput, so moving gate generation out of the hot update
  kernel is structurally attractive

Why it fits GDN:

- `g` and `beta` depend only on input tensors and per-head parameters, not on the
  recurrent state
- therefore they are legal to precompute

Applicability now:

- high
- moderate implementation effort
- good candidate for the next single experiment

Correctness notes:

- keep `g` and `beta` in fp32
- preserve the exact gate formula first before trying any approximation

### 2. Rewrite only the `16-row` long path around warp-owned row groups

Why it is interesting:

- the biggest remaining structural weakness is no longer the short path
- the current long path still spends substantial control flow on cross-warp
  shared-memory reductions and `tid == 0` serialization
- the micro path already demonstrated that warp-local ownership is the right style
  for this problem family

Applicability now:

- medium
- more invasive than a scalar cleanup, but still within the current kernel family

What to preserve:

- do not reintroduce algebraic output fusion
- change mapping and reduction structure, not the math

### 3. Reuse `q/k` across sibling value heads

Why it is interesting:

- grouped-value attention means two `v_head`s share one `q/k` head
- the current implementation duplicates `q/k` loads and bf16-to-f32 conversions
  across those sibling heads

Applicability now:

- medium
- best targeted at the long path, not the underfilled short micro path

Important caveat:

- mapping one CTA to two value heads can cut grid density in half, which is bad on
  the already underfilled short path
- this is therefore better as a long-path reuse optimization than as a micro-path
  change

## Best Rewrite-Level Directions

These are the methods most aligned with B200, but they are not incremental
patches.

### 1. Keep the current micro short path, replace the long path with chunked GDN prefill

Recommended architecture:

- keep the Iteration 29 micro kernel for the short/low-`N` regime
- add a new chunked long path for larger or longer workloads
- start with `chunk_size = 128`, because that is the strongest signal from the
  baseline and from the fixed `head_dim = 128`

Why this is the best overall direction:

- it respects GDN recurrence semantics
- it is the first structure that can actually unlock B200-specific tensor/memory
  features
- it avoids spending more time on launch-heuristic microtuning that the log has
  already mostly exhausted

Applicability now:

- high at the repo strategy level
- low as a small patch

### 2. Add persistent scheduling only after chunking or true tile decomposition

Why it is interesting:

- workload lengths are highly uneven
- static assignment wastes hardware once short tiles finish

Why it is not first:

- in the current scalar recurrence structure, there is not enough movable work to
  justify the complexity
- it becomes natural only after chunk or tile decomposition exists

### 3. Use CuTe/CUTLASS Blackwell paths only after the algorithm is in chunk form

This includes:

- TMA-style movement
- TMEM / `tcgen05`
- Blackwell family-specific code generation
- warp-specialized producer/consumer pipelines

These are attractive, but only after the long path becomes a tile pipeline instead
of a token-serial scalar loop.

## Recommended Ranking

### Highest-value direct experiments

1. precompute `g` and `beta` once per `(t, vh)`
2. warp-specialized rewrite of the `16-row` long path
3. sibling-`v_head` `q/k` reuse on the long path

### Highest-value architectural directions

1. keep current micro short path, add chunked long path
2. persistent scheduler after chunking
3. CuTe/CUTLASS Blackwell path after chunking

## Not Recommended Right Now

Avoid spending more cycles on the following unless the surrounding architecture
changes first:

- more launch-only tuning of `4/8/1/32-row` variants
- vectorization-only patches on the current paths
- register-only tiling of the current scalar micro path
- `launch_bounds` microtuning
- early DSM / cluster / TMA experiments on the current scalar kernel
- output algebraic fusion without a different numerical formulation

## Bottom Line

The current prefill kernel is no longer a raw baseline. It already contains one
strong specialized path: the `2-row micro warp-per-row` kernel. That means the
best remaining opportunities have shifted.

The next useful work is not "more random row-tile heuristics." It is:

1. one more round of structurally informed current-code experiments, led by gate
   precompute and a real long-path mapping rewrite
2. then a deliberate move toward a two-regime architecture:
   `current micro short path + new chunked Blackwell long path`

That is the cleanest synthesis of:

- the current kernel
- the optimization log
- the workload distribution
- GDN algorithmic constraints
- and what B200 is actually good at
