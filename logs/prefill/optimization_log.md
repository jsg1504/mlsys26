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
