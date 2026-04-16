# GDN Prefill Kernel Optimization Log

Tracking all optimization iterations for the prefill kernel.

---

<!-- Append new entries below this line -->

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
