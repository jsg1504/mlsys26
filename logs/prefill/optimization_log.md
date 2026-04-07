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

## 2026-04-08 Iteration 17

- Idea: revive the `4`-row fallback for underfilled `N=3..4` launches whenever the default `16`-row grid would still expose fewer than roughly two blocks per SM, while keeping the current `2`-row micro path for `N=1..2`.
- Why: the required Modal NCU profile still showed the kernel was grid-small and `__syncthreads()`-sensitive on the representative low-`N` shapes, with achieved occupancy around `6.2%` on the smallest launches and Nsight explicitly recommending at least two blocks per SM for this synchronization-heavy kernel; local workload inspection also showed `81/100` benchmark cases have `num_seqs <= 4`, and the current wrapper had effectively left the `4`-row path unused, so the narrowest next launch-only experiment was to revive it only where the `16`-row grid stayed below that two-blocks-per-SM target.
- Result: the quick Modal benchmark passed all `100/100` workloads with `mean_speedup 117.99770`, `mean_latency_ms 2.80145`, `min_speedup 41.02`, and `max_speedup 192.85` versus the latest comparable logged quick baseline `mean_speedup 126.76536` and `mean_latency_ms 3.18747`.
- Decision: revert because correctness held but the weighted `mean_speedup` regressed versus the latest comparable quick baseline.

## 2026-04-08 Iteration 18

- Idea: route long `num_seqs <= 2` workloads away from the current `2`-row micro path and onto the existing `4`-row tile path once `total_seq_len >= 512`.
- Why: the required Modal profile still showed the kernel underfilled and `__syncthreads()`-sensitive, but it also exposed that the current dispatch sends both `N=1, T=16` and `N=1, T=2107` through the same micro-tiled regime; local workload inspection showed a clear low-`N` length gap (`N=1` jumps from `134` to `525`, `N=2` from `461` to `574`), so the narrowest single change left was a length-aware escape hatch that preserves the short low-`N` specialization while reducing row-phase count on the long low-`N` tail.
- Result: the quick Modal benchmark passed all `100/100` workloads with `mean_speedup 109.6773`, `mean_latency_ms 3.24218`, `min_speedup 42.29`, and `max_speedup 199.99` versus the latest comparable logged quick baseline `mean_speedup 126.76536` and `mean_latency_ms 3.18747`.
- Decision: revert because correctness held but the weighted `mean_speedup` regressed versus the latest comparable quick baseline.

## 2026-04-08 Iteration 19

- Idea: add a `1`-row ultra-micro fallback for very short low-`N` launches when `num_seqs <= 2` and `total_seq_len <= 128`, while leaving the current Iteration 16 dispatch unchanged for longer low-`N` and higher-`N` cases.
- Why: the required Modal profile still showed the kernel was heavily underfilled and `__syncthreads()`-sensitive on the shortest representative shapes, with Nsight repeatedly flagging small grids and low achieved occupancy; local workload inspection also showed `28/33` of the `N=1` cases stay at or below `128` total tokens, so the narrowest remaining launch-only experiment was to deepen the micro-tiling only for that very short tail instead of re-routing the longer low-`N` workloads that had already regressed.
- Result: the quick Modal benchmark passed all `100/100` workloads with `mean_speedup 120.6829`, `mean_latency_ms 3.17876`, `min_speedup 43.36`, and `max_speedup 217.32` versus the latest comparable logged quick baseline `mean_speedup 126.76536` and `mean_latency_ms 3.18747`.
- Decision: revert because correctness held but the weighted `mean_speedup` regressed versus the latest comparable quick baseline.

## 2026-04-08 Iteration 20

- Idea: fuse the output path algebraically by computing both `q @ temp` and `q·k` during the existing `k @ temp` pass, then emit `q @ state_new = q @ temp + (q·k) * residual` so the timestep no longer performs the per-row output reduction.
- Why: the required Modal profile still showed the kernel underfilled and synchronization-sensitive, with achieved occupancy around `6.1%` on the shortest representative launch and only about `18.8%` even on the largest one, while both compute and memory throughput stayed low; the remaining per-row output reduction was therefore still the narrowest non-dispatch critical-path target in the timestep loop.
- Result: the quick Modal benchmark failed correctness on `16/100` workloads with `max_atol 1.44`, `max_rtol 2.67e5`, and `matched_ratio 0.84`, so it was not comparable on performance to the latest comparable quick baseline `mean_speedup 126.76536`.
- Decision: revert because correctness regressed.

## 2026-04-08 Iteration 21

- Idea: add `__launch_bounds__(128, 13)` to the templated prefill kernel so the compiler has an explicit incentive to trim register pressure on the active `16`-row long-path kernel.
- Why: the corrected current-dispatch Modal NCU profile showed that this B200 run is effectively using only two regimes, `2`-row tiles for `N <= 2` and `16`-row tiles otherwise; the long-path `16`-row kernel already reached about `53.10%` to `64.14%` achieved occupancy, but Nsight still flagged its `75%` theoretical occupancy as register-limited at `40` registers per thread, so a launch-bounds hint was the narrowest single change left that could plausibly unlock another resident block without touching math, indexing, or dispatch.
- Result: the quick Modal benchmark passed all `100/100` workloads with `mean_speedup 116.35188`, `mean_latency_ms 3.19538`, `min_speedup 41.23719`, and `max_speedup 215.63628` versus the latest comparable logged quick baseline `mean_speedup 126.76536` and `mean_latency_ms 3.18747`.
- Decision: revert because correctness held but the weighted `mean_speedup` regressed versus the latest comparable quick baseline.

## 2026-04-08 Iteration 22

- Idea: shrink only the existing `2`-row micro prefill CTA to `64` threads so each thread owns two K columns instead of keeping the low-`N` path at `128` threads.
- Why: the required Modal profile again showed the representative low-`N` prefill workloads were heavily underfilled, with achieved occupancy around `6.25%`, very low compute and memory throughput, and an explicit Nsight recommendation to keep at least two blocks per SM for the `__syncthreads()`-heavy launch; combined with the current wrapper's `N <= 2` dispatch to the `2`-row micro path, the narrowest new experiment was to halve only that micro CTA width without changing the proven row-tiling heuristic.
- Result: the quick Modal benchmark passed all `100/100` workloads with `mean_speedup 107.83794`, `mean_latency_ms 3.19421`, `min_speedup 42.17675`, and `max_speedup 202.30264` versus the latest comparable logged quick baseline `mean_speedup 126.76536` and `mean_latency_ms 3.18747`.
- Decision: revert because correctness held but the weighted `mean_speedup` regressed versus the latest comparable quick baseline.

## 2026-04-08 Iteration 23

- Idea: keep each thread's existing `2`-row micro prefill state tile in registers instead of shared memory during the recurrent loop.
- Why: the current-dispatch profiling and workload scan still point to the `N <= 2` micro path as the most leveraged regime, with `61/100` benchmark cases in that bucket and the representative low-`N` launches remaining underfilled and synchronization-heavy; NVIDIA's CUDA programming guide also notes that registers are thread-local SM storage, so the narrowest remaining micro-kernel change was to eliminate the repeated shared-memory round-trips for the two per-thread state scalars without touching the proven dispatch thresholds or the `16`-row steady-state path.
- Result: the quick Modal benchmark passed all `100/100` workloads with `mean_speedup 70.54863`, `mean_latency_ms 3.16302`, `min_speedup 24.91403`, and `max_speedup 133.39675` versus the latest comparable logged quick baseline `mean_speedup 126.76536` and `mean_latency_ms 3.18747`.
- Decision: revert because correctness held but the weighted `mean_speedup` regressed sharply versus the latest comparable quick baseline.

## 2026-04-08 Iteration 24

- Idea: route only the underfilled `N=3..4` prefill launches to the existing `8`-row kernel whenever the default `16`-row grid would still expose fewer than roughly two blocks per SM.
- Why: the required Modal profile still showed the current kernel was underfilled and latency-hiding limited across representative prefill shapes, with achieved occupancy staying around `6.24%` to `18.85%`, compute and memory throughput both low, and Nsight continuing to flag small-grid launches; combined with the current dispatch math on this `148`-SM B200 (`16`-row grids of `192` and `256` blocks for `N=3` and `N=4`, versus `8`-row grids of `384` and `512` blocks), the narrowest untried launch-only experiment left was to fill the `N=3..4` gap with the existing `8`-row path while avoiding the previously regressing `4`-row fallback.
- Result: the quick Modal benchmark passed all `100/100` workloads with `mean_speedup 106.7757`, `mean_latency_ms 2.9264`, `min_speedup 40.05`, and `max_speedup 182.61` versus the latest comparable logged quick baseline `mean_speedup 126.76536` and `mean_latency_ms 3.18747`.
- Decision: revert because correctness held but the weighted `mean_speedup` regressed versus the latest comparable quick baseline.

## 2026-04-08 Iteration 25

- Idea: vectorize the prefill state-tile prologue and epilogue with aligned `float4` global loads/stores so both the active `2`-row and `16`-row paths issue fewer fixed-overhead state memory instructions per CTA.
- Why: the required Modal profile again showed the kernel was grid-small and latency-hiding limited, while the current wrapper still routes the real workload set through only two regimes (`2`-row on `61/100` cases and `16`-row on `39/100` cases); after the recent dispatch experiments regressed, the narrowest untried idea left that still helps both live paths was to trim the fixed state load/store instruction count without changing math, synchronization, or launch heuristics. NVIDIA's CUDA guidance also recommends coalesced, aligned vector accesses where possible to reduce memory-instruction overhead.
- Result: the quick Modal benchmark passed all `100/100` workloads with `mean_speedup 112.79125`, `mean_latency_ms 3.17595`, `min_speedup 43.15171`, and `max_speedup 201.44738` versus the latest comparable logged quick baseline `mean_speedup 126.76536` and `mean_latency_ms 3.18747`.
- Decision: revert because correctness held but the weighted `mean_speedup` regressed versus the latest comparable quick baseline.

## 2026-04-08 Iteration 26

- Idea: keep the existing `2`-row micro prefill state tile in registers only for very short `seq_len <= 64` blocks, while leaving longer low-`N` launches and all larger row-tile kernels on the baseline shared-memory path.
- Why: the required Modal profile still showed representative current prefill launches were underfilled and latency-hiding limited, with achieved occupancy around `6.22%` to `18.84%`, grid sizes of only `64` to `456` blocks on the profiled shapes, and both compute and memory throughput staying under roughly `11%`; the workload study also shows many per-sequence lengths fall at or below `64`, while the earlier broad `2`-row register-tiling experiment regressed because it also hit the long low-`N` tail. The narrowest remaining short-path experiment was therefore to confine that register-tiling idea to very short micro-kernel blocks only.
- Result: the quick Modal benchmark passed all `100/100` workloads with `mean_speedup 107.03050`, `mean_latency_ms 3.19030`, `min_speedup 39.54`, and `max_speedup 190.02` versus the latest comparable logged quick baseline `mean_speedup 126.76536` and `mean_latency_ms 3.18747`.
- Decision: revert because correctness held but the weighted `mean_speedup` regressed versus the latest comparable quick baseline.

## 2026-04-08 Iteration 27

- Idea: add a `32`-row long-path prefill tile when the coarsened grid still leaves at least two blocks per SM, while leaving the existing `2`-row micro path and the current `16`-row low-to-mid path unchanged.
- Why: the required Modal profile still showed the kernel was underfilled and synchronization-limited overall, while the recent current-dispatch notes in this log showed the live kernel had effectively collapsed to a `2`-row micro path for `N <= 2` and a `16`-row path otherwise, with the long-path `16`-row launches already reaching roughly `53%` to `64%` achieved occupancy. That made the narrowest remaining long-path experiment a coarser row tile for only the well-filled large-`N` cases, trading some excess grid density for fewer fixed state prologue/epilogue phases per updated `128x128` state matrix.
- Result: the quick Modal benchmark passed all `100/100` workloads with `mean_speedup 110.21769`, `mean_latency_ms 4.65542`, `min_speedup 33.42358`, and `max_speedup 228.61807` versus the latest comparable logged quick baseline `mean_speedup 126.76536` and `mean_latency_ms 3.18747`.
- Decision: revert because correctness held but the weighted `mean_speedup` regressed versus the latest comparable quick baseline.

## 2026-04-08 Iteration 28

- Idea: vectorize only the live `2`-row micro prefill state-tile prologue and epilogue with aligned `float4` global loads/stores, while leaving the proven `16`-row path and the recurrent math unchanged.
- Why: the required fresh Modal profile still showed representative low-`N` launches were small-grid and latency-hiding limited, with achieved occupancy around `6.2%` and Nsight again recommending at least two runnable blocks per SM for the `__syncthreads()`-heavy kernel. The workload scan also still shows `61/100` benchmark cases in the live `N <= 2` micro path and `45/100` of those at `total_seq_len <= 128`, so the narrowest remaining fixed-overhead experiment was to keep the earlier state-I/O vectorization idea confined to that dominant short path instead of perturbing the `16`-row steady-state kernel that had already regressed when vectorized together.
- Result: the quick Modal benchmark passed all `100/100` workloads with `mean_speedup 114.04340`, `mean_latency_ms 3.19428`, `min_speedup 43.55`, and `max_speedup 207.27` versus the latest comparable logged quick baseline `mean_speedup 126.76536` and `mean_latency_ms 3.18747`.
- Decision: revert because correctness held but the weighted `mean_speedup` regressed versus the latest comparable quick baseline.

## 2026-04-08 Iteration 29

- Idea: specialize only the live `2`-row micro prefill path into a `64`-thread warp-per-row kernel, so each warp owns one row, processes four K columns per lane, and uses warp shuffles instead of the existing cross-warp row reductions and CTA barriers inside the timestep loop.
- Why: the required fresh Modal profile and the recent current-dispatch notes still showed the dominant `N <= 2` micro path was small-grid and synchronization-heavy, with representative low-`N` launches stuck around `6.2%` achieved occupancy while the long-path `16`-row kernel was already much healthier. The Nsight Compute profiling guide also explicitly calls out barrier stalls and shared-memory short-scoreboard stalls as optimization targets, while the CUDA programming guide documents `__shfl_sync` as the warp-local data-exchange primitive that removes the need for shared-memory reductions. With `61/100` workloads still landing on that live micro path, the narrowest remaining high-impact experiment was therefore to rewrite only the `2`-row kernel mapping around warp-local row ownership instead of perturbing the proven `4`-, `8`-, and `16`-row paths.
- Result: the quick Modal benchmark passed all `100/100` workloads with `mean_speedup 170.18544`, `mean_latency_ms 3.05327`, `min_speedup 43.89814`, and `max_speedup 368.71060` versus the latest comparable logged quick baseline `mean_speedup 126.76536` and `mean_latency_ms 3.18747`.
- Decision: commit because correctness held and the weighted `mean_speedup` improved strongly versus the latest comparable quick baseline.
