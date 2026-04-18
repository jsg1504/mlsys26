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

## 2026-04-18 - Python dispatch optimization (ACCEPTED)

- **Idea**: Minimize all per-call overhead in `main.py`. Cache CUstream, pre-create ctypes constants (`_INT32_8`, `_BLOCK_128`, `_BLOCK_256`), use `get_cu_seqlens_metadata` for cached `max_s_q` (avoids GPU sync on repeat calls of same cu_seqlens), pre-allocate `gate_log`/`beta` tensors per `(T, device)`, pre-compute `_DEFAULT_SCALE` constant.
- **Result**: 0.282ms → 0.2732ms (-3.1%)
- **Status**: accepted
- **Learnings**: Python dispatch overhead IS measured by the CUDA event timer (appears as GPU idle time between `record()` and kernel start). Every `.item()` GPU sync, `torch.empty()` alloc, `drv.CUstream()` construction adds measurable cost. 17 workloads >0.5ms (down from 18). Min latency dropped meaningfully (0.083→0.070ms) showing the per-call overhead was nontrivial for fast workloads.

## 2026-04-18 - Double-buffered sequential kernel (REVERTED)

- **Idea**: Reduce per-token sync in NVRTC sequential kernel via double-buffered k/q shared memory + scalar (v/a/b) prefetch. Cut from 2 syncs/token to 1, overlap next-token load with current compute.
- **Result**: 0.282ms → 0.296ms (+4.9% regression)
- **Status**: reverted
- **Why it didn't work**: The sequential kernel only handles T≤128 workloads (already fast, 0.05-0.15ms each). The added complexity (preload phase, buffer indexing, conditional sync) added overhead that dominates savings for very short sequences (T=6-32). Meanwhile, the 18 workloads >0.5ms (all in CuTe-DSL path) remain the bottleneck — they contribute ~60% of total time and were unaffected by this change.
- **Learning**: To hit sub-0.2ms, optimization must target the long-sequence CuTe-DSL path (>0.5ms workloads), not the already-fast sequential path. Possible directions:
  1. Hand-written CUDA chunked kernel replacing CuTe-DSL for medium sequences (T=128-2048)
  2. Eliminate Python/GPU-sync overhead in the long-path dispatch
  3. Investigate whether the CuTe-DSL kernel has tunable parameters (chunk_size, tile sizes) that benefit our specific problem shape
