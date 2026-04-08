# Prefill Optimization Candidates for the Current Kernel

Updated: 2026-04-08

## Goal

This note analyzes what is still worth trying for
`gdn_prefill_qk4_v8_d128_k_last` in its current repo state.

It combines:

- the live CUDA kernel
- the latest committed quick benchmark baseline
- the optimization log including the most recent reverted ideas
- a fresh required Modal profile
- the contest workload mix
- the contest Blackwell baseline
- official NVIDIA and Gated Delta Networks primary sources

## Agent Team

The work was split into five roles:

1. `Hardware agent`
   - B200 / Blackwell features, compiler constraints, and realistic leverage
2. `Model agent`
   - GDN recurrence legality, chunking, grouped-value-attention reuse
3. `Kernel agent`
   - current CUDA structure, live dispatch, and already-tested ideas
4. `Baseline agent`
   - workload distribution and the contest FlashInfer Blackwell baseline
5. `Integrator`
   - merge all signals into a ranked applicability list

## What Was Reviewed

### Local repo

- `gdn_prefill_qk4_v8_d128_k_last/solution/cuda/kernel.cu`
- `logs/prefill/bench_history.jsonl`
- `logs/prefill/optimization_log.md`
- `scripts/profile_kernel.py`
- `mlsys26-contest/definitions/gdn/gdn_prefill_qk4_v8_d128_k_last.json`
- `mlsys26-contest/workloads/gdn/gdn_prefill_qk4_v8_d128_k_last.jsonl`
- `mlsys26-contest/solutions/baseline/gdn/gdn_prefill_qk4_v8_d128_k_last/flashinfer_wrapper_123ca6.json`
- `docs/gdn_prefill_optimization_study.md`
- `docs/gemini_gdn_prefill_b200.md`

### External primary sources

- Gated Delta Networks paper:
  https://proceedings.iclr.cc/paper_files/paper/2025/file/4904fad153f6434a7bcf04465d4be2cc-Paper-Conference.pdf
- Official GatedDeltaNet repo:
  https://github.com/NVlabs/GatedDeltaNet
- NVIDIA CUDA Programming Guide:
  https://docs.nvidia.com/cuda/cuda-programming-guide/
- NVIDIA Blackwell family-specific feature blog:
  https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/
- NVIDIA CUTLASS Blackwell functionality:
  https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html
- NVIDIA CUTLASS Blackwell cluster launch control:
  https://docs.nvidia.com/cutlass/media/docs/cpp/blackwell_cluster_launch_control.html

## Current Repo State

### Live kernel shape

The checked-in kernel is still a sequential per-token recurrent update:

- `g = exp(-exp(A_log) * softplus(a + dt_bias))`
- `beta = sigmoid(b)`
- `temp = g * state`
- `residual = beta * (v - k @ temp)`
- `state_new = temp + k^T @ residual`
- `output = scale * (q @ state_new)`

This is explicit in the kernel header comment.

### Best committed quick benchmark baseline

The latest comparable committed quick benchmark is Iteration 41:

- `mean_speedup = 269.0450130872929`
- `mean_latency_ms = 0.9570018161771198`
- correctness matched all workloads

This matters because later ideas must beat Iteration 41, not an older baseline.

### What the latest log already ruled out

Iterations 42 through 46 all regressed and were reverted:

- persistent-style fixed-CTA scheduling
- software pipelining of warp-private `q/k` and gate reloads
- vectorized long-path `q/k` pair loads
- vectorized long-path state `float4` I/O
- Blackwell cluster launch control work stealing on the current scalar family

So those ideas are no longer good default recommendations for the current kernel.

## Live Dispatch and Workload Share

Applying the current wrapper logic to the 100 contest workloads gives:

- `micro2`: `61/100` workloads, `21,605 / 183,065` tokens, `11.8%` of token volume
- `long16`: `39/100` workloads, `161,460 / 183,065` tokens, `88.2%` of token volume
- `4-row` and `8-row` templated paths are effectively dead on the contest set

Implication:

- the short path matters for weighted score stability
- the long path dominates total work
- any serious next optimization has to be justified against `long16`

## Fresh Modal Profile

Required run:

- `conda run -n fi-bench modal run scripts/profile_kernel.py --kernel prefill`

Important caveat:

- the profiling harness still does not mirror the real wrapper dispatch
- it still hardcodes templated launches `gdn_prefill_kernel<8|4|2|1>`
- so the profile is directional, not authoritative, for the live `micro2/long16` code

Even with that caveat, the signal is stable and useful:

- `N=1, T=16`: grid `64`, compute throughput `0.54%`, memory throughput `0.55%`, achieved occupancy `6.22%`
- `N=1, T=2107`: grid `64`, compute throughput `0.60%`, memory throughput `0.61%`, achieved occupancy `6.25%`
- `N=43, T=8192`: grid `344`, compute throughput `7.33%`, memory throughput `8.24%`, achieved occupancy `14.33%`
- `N=57, T=8192`: grid `456`, compute throughput `9.66%`, memory throughput `10.86%`, achieved occupancy `18.84%`

Bottom line:

- the family is still small-grid
- it is still weakly latency-hidden
- it is not bandwidth-saturated
- fixed per-tile overhead still matters more than raw memory bandwidth

## B200 / Blackwell Findings

The hardware signal is clear:

1. Blackwell-specific upside is real, but it is mostly tied to tile-level matrix work.
   - `tcgen05.mma`, TMA, TMEM, and CLC are strongest when the kernel exposes large structured tiles.
2. Family-specific features are not a magic compiler switch.
   - Using them directly requires either PTX or a library path that already does so.
3. The current raw-CUDA recurrent loop does not expose enough tensor-core-friendly work.
   - The live kernel mostly executes scalar bf16-to-f32 loads, SFU gate math, warp reductions, and row-wise updates.
4. CUTLASS Blackwell machinery is most valuable after the algorithm is chunked.
   - That is where warp-specialized producer/consumer pipelines, TMA, TMEM, and dynamic persistence actually fit.

## GDN Findings

The model-side constraints are equally clear:

1. Naive token-parallel prefill is illegal.
   - `state_t` depends directly on `state_{t-1}`.
2. GDN already has a hardware-efficient chunkwise formulation in the paper.
   - That is the first mathematically valid route to larger parallel tiles.
3. Grouped-value attention is real reuse.
   - `q_heads = 4`
   - `k_heads = 4`
   - `v_heads = 8`
   - two sibling `v_head`s share the same `q/k` head
4. Exact gate precompute is legal but not automatically profitable.
   - `g` and `beta` depend only on `a`, `b`, `A_log`, and `dt_bias`
   - the current repo already showed that a separate gate-cache pass regresses

## Contest Baseline Signal

The contest baseline for this exact definition is highly informative.

It does not use a token-serial scalar kernel. It uses:

- `chunk_gated_delta_rule`
- `chunk_size = 128`
- persistent and non-persistent scheduling support
- CuTe / CUTLASS / SM100 Blackwell machinery
- gate values prepared outside the recurrent kernel

That is the strongest repo-local clue about the right long-term architecture.

## Ranked Optimization Methodologies

| Rank | Methodology | Why it is still worth attention | Applicability to current code | Verdict |
| --- | --- | --- | --- | --- |
| 1 | Keep `micro2`, replace only `long16` with a chunked GDN long path starting at `chunk_size = 128` | `long16` owns `88.2%` of token volume and chunking is the first legal structure that exposes real Blackwell-friendly work | rewrite-level, but preserves the public interface and can keep the short path unchanged | `best-next-direction` |
| 2 | Implement the chunked long path with CuTe/CUTLASS SM100 using `tcgen05`, TMA, and TMEM | This is where B200-specific upside actually lives | rewrite-level only | `recommended-after-1` |
| 3 | Add persistent scheduling or dynamic tile scheduling only after chunk tiling exists | Variable-length imbalance matters, but current scalar persistent and CLC attempts already regressed | rewrite-level only | `recommended-after-1` |
| 4 | Revisit sibling-`v_head` `q/k` reuse only inside chunk tiles | GVA reuse is real, but the current paired-head scalar attempt already regressed | rewrite-level only | `conditional-after-1` |
| 5 | Revisit gate generation only as an overlapped pipeline stage inside the chunked backend | Exact and legal, but the separate gate-cache pass already regressed | rewrite-level only | `conditional-after-1` |
| 6 | Try a narrow `long16` L2 persisting-cache window for repeatedly reused `q/k/a/b` inputs | This is the main Blackwell-flavored current-family idea that has not yet been tested and does not perturb the recurrence math | current code, long-path only, low expected ROI | `optional-try-now` |
| 7 | Fix `scripts/profile_kernel.py` to profile the real `micro2/long16` dispatch | Not a speedup by itself, but required before spending more loops on current-family ranking | immediate engineering task | `do-now` |
| 8 | More current-family scheduling, pipelining, vectorization, or CLC on the scalar long path | Already tested in Iterations 42-46 and all regressed | technically possible, but weak evidence | `not-now` |
| 9 | More row-heuristic tuning or dead-path revival (`4-row`, `8-row`, `1-row`, broader fallbacks) | Extensively explored already | current kernel family | `not-now` |
| 10 | Standalone exact gate-cache precompute pass | Legal but empirically regressed badly | current kernel family | `not-now` |
| 11 | Algebraic output fusion | Already failed correctness multiple times | any near-term path | `unsafe` |

## Applicability Against the Current Implementation

### Immediately applicable

- try a host-side L2 persisting-cache window for the `long16` path only

This is the only still-plausible Blackwell-specific experiment that fits the
current scalar kernel family. It is low-risk to correctness because it does not
change the recurrence math or launch mapping, but it is also low-confidence
because the remaining bottleneck is still mostly latency hiding rather than raw
HBM pressure.

- fix the prefill profiling harness so it follows the real wrapper dispatch

This is not a kernel speedup, but it is the only high-confidence immediate task
that improves decision quality instead of repeating recently failed ideas.

### Compatible with the current public interface, but rewrite-level

- keep the current `micro2` kernel for `N <= 2`
- replace only the `N >= 3` long path with a chunked backend
- preserve:
  - current tensor shapes
  - current state layout `[N, 8, 128, 128]`
  - current GVA mapping
  - current TVM FFI entry point

This is the cleanest way to move forward without throwing away the part of the
kernel family that already works well.

### Not recommended in the current scalar family

- another persistent-scheduling variant
- another software-pipelining pass
- more vectorization-only follow-ups
- more launch-heuristic churn
- another separate gate-cache pass
- direct CLC/cluster scheduling on the current scalar row-tile loop

The repo has already spent several measured iterations here, and the evidence is
consistently negative.

### Unsafe right now

- algebraic output fusion without a materially different numerical formulation

## Recommendation

The synthesis is straightforward:

1. stop treating the current scalar long path as an open-ended tuning surface
2. keep the current `micro2` short path
3. fix the profile harness so future rankings are based on the real dispatch
4. move the next real optimization effort to:
   - `chunked GDN long path`
   - `chunk_size = 128`
   - `CuTe/CUTLASS SM100`
   - Blackwell-native scheduling only after chunk tiling exists

That is the best match to:

- the current benchmark history
- the actual workload mix
- GDN recurrence legality
- the repo-local baseline architecture
- and what B200 is actually good at

## Checklist

### Recommended

- [ ] Optionally try a `long16` L2 persisting-cache window for reused `q/k/a/b`
- [ ] Fix `scripts/profile_kernel.py` to mirror the real `micro2/long16` dispatch
- [ ] Keep the current `micro2` short path
- [ ] Rewrite only the `long16` regime as chunked GDN prefill
- [ ] Start the rewrite with `chunk_size = 128`
- [ ] Build the new long path with CuTe/CUTLASS SM100 `tcgen05`
- [ ] Use TMA and TMEM in the new chunked backend, not in the current scalar kernel
- [ ] Add persistent or dynamic scheduling only after chunk tiling exists
- [ ] Revisit sibling-`v_head` `q/k` reuse only inside the chunked backend
- [ ] Revisit gate precompute only as an overlapped pipeline stage

### Do Not Prioritize

- [ ] Do not spend another loop on row-heuristic tuning
- [ ] Do not spend another loop on vectorization-only patches in the scalar family
- [ ] Do not retry standalone gate-cache precompute in the current family
- [ ] Do not retry CLC directly on the current scalar long path

### Unsafe

- [ ] Do not retry algebraic output fusion without a new and carefully validated numerical formulation

## Sources

### Local

- `gdn_prefill_qk4_v8_d128_k_last/solution/cuda/kernel.cu`
- `logs/prefill/bench_history.jsonl`
- `logs/prefill/optimization_log.md`
- `scripts/profile_kernel.py`
- `mlsys26-contest/definitions/gdn/gdn_prefill_qk4_v8_d128_k_last.json`
- `mlsys26-contest/workloads/gdn/gdn_prefill_qk4_v8_d128_k_last.jsonl`
- `mlsys26-contest/solutions/baseline/gdn/gdn_prefill_qk4_v8_d128_k_last/flashinfer_wrapper_123ca6.json`
- `docs/gdn_prefill_optimization_study.md`
- `docs/gemini_gdn_prefill_b200.md`

### External

- Gated Delta Networks paper:
  https://proceedings.iclr.cc/paper_files/paper/2025/file/4904fad153f6434a7bcf04465d4be2cc-Paper-Conference.pdf
- Official GatedDeltaNet repo:
  https://github.com/NVlabs/GatedDeltaNet
- NVIDIA CUDA Programming Guide:
  https://docs.nvidia.com/cuda/cuda-programming-guide/
- NVIDIA Blackwell Tuning Guide:
  https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html
- CUDA L2 cache control:
  https://docs.nvidia.com/cuda/archive/13.1.1/cuda-programming-guide/04-special-topics/l2-cache-control.html
- NVIDIA Blackwell and CUDA 12.9 family-specific features:
  https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/
- NVIDIA CUTLASS Blackwell functionality:
  https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html
- NVIDIA CUTLASS Blackwell cluster launch control:
  https://docs.nvidia.com/cutlass/media/docs/cpp/blackwell_cluster_launch_control.html
