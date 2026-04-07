# GDN Prefill Kernel Optimization Log

Tracking all optimization iterations for the prefill kernel.

---

<!-- Append new entries below this line -->

## 2026-04-07 Iteration 1

- Idea: cache the per-timestep decayed shared state `temp = g * state` in `s_state` during the k-dot pass, then reuse it in the update pass.
- Why: the baseline recomputed `g * s_state[...]` for every `(vi, tid)` during the second loop even though the decayed value had already been produced in the first loop.
- Result: quick Modal benchmark passed all 100 workloads with `mean_speedup 6.42` versus the comparable logged baseline `6.37`.
- Decision: commit.

## 2026-04-08 Iteration 2

- Idea: row-block the `vi` loop in tiles of 8 rows so one set of CTA-wide barriers serves multiple k-dot and output reductions.
- Why: the current kernel is barrier-heavy, and local review notes identified the repeated per-row `__syncthreads()` pattern as the most direct near-term bottleneck to attack without changing recurrence semantics.
- Result: quick Modal benchmark passed all 100 workloads with `mean_speedup 10.02` and `mean_latency_ms 24.21` versus the latest comparable logged quick baseline `6.42` and `34.55`.
- Decision: commit.

## 2026-04-08 Iteration 3

- Idea: split the recurrent state over `16`-row tiles so each CTA owns one `(seq, v_head, row_tile)` instead of the full `128x128` state.
- Why: the required Modal NCU profile showed theoretical occupancy capped at `18.75%` by `66.21 KB` of shared memory per block, plus low achieved occupancy (`6.25%` to `17.81%`) and underfilled launches; moving to row-tiled CTAs cuts per-block shared state sharply and exposes `8x` more blocks from the same workload.
- Result: quick Modal benchmark passed all 100 workloads with `mean_speedup 61.32` and `mean_latency_ms 3.598` versus the latest comparable logged quick baseline `10.02` and `24.21`.
- Decision: commit.

## 2026-04-08 Iteration 4

- Idea: hoist the per-token gate scalar work so each CTA computes `g` and `beta` once per timestep and broadcasts them from shared memory instead of recomputing the same values in all `128` threads.
- Why: after the row-tiled rewrite, the inner loop still duplicated the `softplus`/`exp`/sigmoid chain and the `a`/`b` loads across every thread in the CTA even though those scalars are identical for a given `(t, vh)` tile.
- Result: quick Modal benchmark passed all 100 workloads with `mean_speedup 63.84` and `mean_latency_ms 3.854` versus the latest comparable logged quick baseline `61.32` and `3.598`.
- Decision: commit.

## 2026-04-08 Iteration 5

- Idea: specialize underfilled launches to `8`-row tiles when the default `16`-row prefill grid would launch fewer than one block per SM.
- Why: the corrected Modal NCU profile showed that low-parallelism shapes still launched only `64` blocks at `N=1` with the `16`-row kernel and reached about `6.2%` achieved occupancy, while large shapes were already near-balanced, so the best next single change was to increase short-case block count without disturbing the main `16`-row path.
- Result: quick Modal benchmark passed all 100 workloads with `mean_speedup 76.45` and `mean_latency_ms 3.506` versus the latest comparable logged quick baseline `63.84` and `3.854`.
- Decision: commit.

## 2026-04-08 Iteration 6

- Idea: add a `4`-row tile fallback when the existing `8`-row prefill launch would still expose fewer than roughly two blocks per SM.
- Why: the latest Modal NCU profile still flagged the kernel as small-grid and synchronization-heavy, and NVIDIA's CUDA guidance says multiple resident blocks per SM help hide `__syncthreads()` stalls; moving the most underfilled shapes from `8` rows to `4` rows is the smallest change that increases block count again without touching the established `8`- and `16`-row paths.
- Result: quick Modal benchmark passed all 100 workloads with `mean_speedup 102.1511` and `mean_latency_ms 3.33782` versus the latest comparable logged quick baseline `76.45` and `3.506`.
- Decision: commit.

## 2026-04-08 Iteration 7

- Idea: broaden the `8`-row launch heuristic so the `16`-row prefill path only runs once it can expose roughly two blocks per SM.
- Why: the required Modal NCU profile still showed low achieved occupancy on the short, synchronization-heavy prefill shapes, and Nsight explicitly recommended keeping at least two resident blocks per SM for the `__syncthreads()` cases; extending the existing `8`-row path to `N=3..4` was the narrowest launch-only experiment that matched that guidance.
- Result: quick Modal benchmark passed all 100 workloads with `mean_speedup 91.0193` and `mean_latency_ms 3.07649` versus the latest comparable logged quick baseline `102.1511` and `3.33782`.
- Decision: revert because the weighted `mean_speedup` regressed despite the lower `mean_latency_ms`.

## 2026-04-08 Iteration 8

- Idea: hoist the per-row residual scalar `beta * (v - kdot)` into the single-thread reduction phase and reuse it from shared memory in the update pass.
- Why: after the recent launch-tuning passes, the hot inner loop still recomputed the same residual scalar in all `128` threads for every row and timestep, so this was the narrowest safe way to remove redundant work without changing the launch heuristic or synchronization structure.
- Result: quick Modal benchmark passed all 100 workloads with `mean_speedup 90.58838` and `mean_latency_ms 3.34223` versus the latest comparable logged quick baseline `102.1511` and `3.33782`.
- Decision: revert because correctness held but the weighted `mean_speedup` regressed versus the latest comparable quick baseline.

## 2026-04-08 Iteration 9

- Idea: add a `2`-row tile fallback when the existing `4`-row prefill launch would still expose fewer than roughly two blocks per SM.
- Why: the required Modal profile still showed the shortest synchronization-heavy shapes running with achieved occupancy around `6.2%`, while Nsight explicitly recommends keeping at least two resident blocks per SM for `__syncthreads()`-heavy kernels; the workload set is also dominated by low-`N` cases, so specializing the single-sequence tail was the narrowest launch-only experiment left to test.
- Result: quick Modal benchmark passed all 100 workloads with `mean_speedup 99.86472` and `mean_latency_ms 3.30985` versus the latest comparable logged quick baseline `102.1511` and `3.33782`.
- Decision: revert because correctness held but the weighted `mean_speedup` regressed versus the latest comparable quick baseline.

## 2026-04-08 Iteration 10

- Idea: keep each thread's `4`-row prefill state tile in registers instead of shared memory during the recurrent loop.
- Why: the required Modal profile again showed the kernel was underfilled and latency-bound rather than bandwidth-bound, and the local prefill survey marks register tiling as a viable next scalar-kernel experiment; specializing only the existing `4`-row path was the narrowest way to cut repeated shared-memory traffic in the low-`N` runtime path without changing the established launch heuristic or the `16`-row steady-state kernel.
- Result: quick Modal benchmark passed all 100 workloads with `mean_speedup 90.66050` and `mean_latency_ms 3.30741` versus the latest comparable logged quick baseline `102.1511` and `3.33782`.
- Decision: revert because correctness held but the weighted `mean_speedup` regressed versus the latest comparable quick baseline.

## 2026-04-08 Iteration 11

- Idea: remove the redundant end-of-timestep CTA barrier after the output write in the recurrent loop.
- Why: the required Modal NCU profile still showed low achieved occupancy across the workload mix (`6.18%` to `18.86%`) and Nsight continued to flag the kernel as underfilled and `__syncthreads()`-sensitive; after the output-reduction barrier, only `tid == 0` performs a global output store, while the next timestep already has a barrier before any thread reuses shared scalars or row tiles, so the trailing barrier was extra synchronization.
- Result: quick Modal benchmark passed all 100 workloads with `mean_speedup 105.33746` and `mean_latency_ms 3.25857` versus the latest comparable logged quick baseline `102.1511` and `3.33782`.
- Decision: commit.

## 2026-04-08 Iteration 12

- Idea: parallelize each tile's row-finalization epilogues across warp `0` instead of serializing both cross-warp reductions through `tid == 0`.
- Why: the required Modal NCU profile still showed the kernel underfilled and latency-bound, with achieved occupancy ranging from about `6.19%` to `18.85%` and both SM and memory throughput staying low, so the next single-change opportunity was to shorten the per-timestep serialized critical path rather than chase bandwidth; the current kernel still had one thread summing every row tile twice per token.
- Result: quick Modal benchmark passed all 100 workloads with `mean_speedup 94.71138` and `mean_latency_ms 3.10382` versus the latest comparable logged quick baseline `105.33746` and `3.25857`.
- Decision: revert because correctness held but the weighted `mean_speedup` regressed versus the latest comparable quick baseline.

## 2026-04-08 Iteration 13

- Idea: split the low-`N` launch heuristic by sequence length so the currently dead `8`-row kernel becomes reachable again for longer `num_seqs == 2` workloads while the short underfilled cases keep the `4`-row fallback.
- Why: the required Modal profile still showed the kernel was underfilled and `__syncthreads()`-sensitive on low-`N` shapes, while the repo's prefill study explicitly recommends separate execution regimes for short and long workloads; after Iteration 6, the dispatch never selected the `8`-row path, so the narrowest new experiment was to revive it only where the extra per-token row phases of the `4`-row kernel were most likely to dominate.
- Result: quick Modal benchmark passed all 100 workloads with `mean_speedup 86.5064` and `mean_latency_ms 3.36934` versus the latest comparable logged quick baseline `105.33746` and `3.25857`.
- Decision: revert because correctness held but the weighted `mean_speedup` regressed versus the latest comparable quick baseline.

## 2026-04-08 Iteration 14

- Idea: fuse the output path algebraically by accumulating `q @ temp` beside `k @ temp`, then use `q @ state_new = q @ temp + (q · k) * residual` to avoid the second per-row output reduction.
- Why: the required Modal profile again showed the kernel was underfilled and synchronization-sensitive across the workload mix, and the current tile kernel still spends one full cross-warp reduction per row on the output path; this was the narrowest single change that directly removed that reduction without touching the launch heuristic.
- Result: the quick Modal benchmark failed correctness with `INCORRECT_NUMERICAL` on `17/100` workloads, with observed worst-case `max_atol 1.86` and `max_rtol 1.60e5`, so it was not comparable on performance to the latest logged quick baseline `mean_speedup 105.33746`.
- Decision: revert because correctness regressed.

## 2026-04-08 Iteration 15

- Idea: fuse the output path algebraically by accumulating both `q @ temp` and `q · k` during the existing `k @ temp` pass, then emit `q @ state_new` as `q @ temp + (q · k) * residual` so the kernel no longer needs the second per-row cross-warp reduction.
- Why: the required Modal profile still showed the kernel underfilled and synchronization-sensitive, with Nsight continuing to call out poor latency hiding on the `__syncthreads()`-heavy launch shapes, so the most direct remaining single-change target was still the serialized output reduction path inside each timestep.
- Result: the quick Modal benchmark failed correctness with `INCORRECT_NUMERICAL` on `16/100` workloads, with observed worst-case `max_atol 1.56` and `max_rtol 1.48e5`, so it was not comparable on performance to the latest logged quick baseline `mean_speedup 105.33746`.
- Decision: revert because correctness regressed.

## 2026-04-08 Iteration 16

- Idea: broaden the `2`-row micro-tile fallback so the current `4`-row low-`N` prefill path switches to `2`-row tiles whenever the `4`-row launch would still expose fewer than roughly four blocks per SM.
- Why: the accurate current-dispatch Modal NCU profile showed the existing `4`-row path was still badly underfilled on the representative low-`N` cases that dominate the short-workload side of the benchmark mix: `N=1` launched `256` total blocks, reached only about `10.8%` achieved occupancy, and Nsight explicitly recommended at least two blocks per SM for the `__syncthreads()`-heavy kernel, while the corrected `N=2` profile was still grid-small at `512` total blocks and only `0.22` full waves; in contrast, the established `16`-row path had already climbed to `53%` to `64%` achieved occupancy on the large-`N` shapes, so the narrowest next change was to deepen only the low-parallelism fallback.
- Result: quick Modal benchmark passed all `100/100` workloads with `mean_speedup 126.76536`, `mean_latency_ms 3.18747`, `min_speedup 47.73991`, and `max_speedup 266.28173` versus the latest comparable logged quick baseline `mean_speedup 105.33746` and `mean_latency_ms 3.25857`.
- Decision: commit.
