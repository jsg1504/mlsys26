# Prefill Workflow Template

## Target

- Kernel: `prefill`
- Source: `gdn_prefill_qk4_v8_d128_k_last/solution/cuda/kernel.cu`
- Config: `gdn_prefill_qk4_v8_d128_k_last/config.toml`
- Bench log: `logs/prefill/bench_history.jsonl`
- Optimization log: `logs/prefill/optimization_log.md`
- Profile command:
  - `conda run -n fi-bench modal run scripts/profile_kernel.py --kernel prefill`
- Benchmark command:
  - `conda run -n fi-bench modal run scripts/run_modal_subfolder.py --subfolder gdn_prefill_qk4_v8_d128_k_last`

## Explorer Message: Research

```text
Read `gdn_prefill_qk4_v8_d128_k_last/solution/cuda/kernel.cu`, `logs/prefill/bench_history.jsonl`, and `logs/prefill/optimization_log.md`.
External research mode: `auto` by default. Override with `local_only` to forbid external web research or `required` to mandate it.
Before ranking ideas, run:
`conda run -n fi-bench modal run scripts/profile_kernel.py --kernel prefill`
Also read any relevant materials under `docs/` before ranking ideas.
Only consult external official documentation and papers if the active external research mode allows it.
Treat the latest comparable committed quick benchmark entry as baseline when one exists. If none exists yet, use the latest quick benchmark entry and call out that bootstrap assumption explicitly.
Use the current profile results to identify the dominant bottlenecks and cite the relevant profile findings, but treat the checked-in prefill profiling harness as directional if it diverges from the current wrapper dispatch or the latest logged quick-benchmark evidence.
Identify the most likely current bottleneck in the prefill kernel and rank 2-3 next optimization ideas for a single iteration.
Return the top recommendation, expected impact, implementation sketch, any correctness constraints, which benchmark entry is the baseline, which external research mode was used, which `docs/` files informed the recommendation, whether external sources were consulted and why, and which official documents or papers materially influenced it if any were used.
```

## Worker Message: Implement

```text
Implement exactly one optimization in `gdn_prefill_qk4_v8_d128_k_last/solution/cuda/kernel.cu`.
Use the selected prefill optimization idea only. Preserve correctness, interfaces, tensor layout, and destination-passing behavior.
Do not mix unrelated refactors or cleanup into this edit.
At the end, report files changed, what changed, and any assumptions that must remain true.
```

## Explorer Message: Review

```text
Review the edited `gdn_prefill_qk4_v8_d128_k_last/solution/cuda/kernel.cu` before benchmarking.
Check indexing, varlen sequence handling, shapes, GDN formula preservation, q/k to v-head mapping, synchronization safety, and obvious performance regressions.
Return pass or fail, findings by severity, and whether benchmarking is safe.
```

## Worker Message: Evaluate

```text
Run the quick Modal benchmark for prefill:
`conda run -n fi-bench modal run scripts/run_modal_subfolder.py --subfolder gdn_prefill_qk4_v8_d128_k_last`
Use the active external research mode from the research step and carry forward a short note describing whether external sources were consulted.
Parse the benchmark output, compare it against the latest comparable committed quick benchmark baseline already in `logs/prefill/bench_history.jsonl` when one exists, append a new entry to `logs/prefill/bench_history.jsonl`, append a short note to `logs/prefill/optimization_log.md`, and recommend `revert` or `commit`.
The appended JSON line must include `decision` with that final recommendation and `research_mode` with the active external research mode.
The optimization-log note must mention the active research mode and whether external sources were consulted.
Use correctness regression or no performance improvement as `revert`.
```
