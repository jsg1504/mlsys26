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
