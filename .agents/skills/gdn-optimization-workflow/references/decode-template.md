# Decode Workflow Template

## Target

- Kernel: `decode`
- Source: `gdn_decode_qk4_v8_d128_k_last/solution/cuda/kernel.cu`
- Config: `gdn_decode_qk4_v8_d128_k_last/config.toml`
- Bench log: `logs/decode/bench_history.jsonl`
- Optimization log: `logs/decode/optimization_log.md`
- Benchmark command:
  - `conda run -n fi-bench modal run scripts/run_modal_subfolder.py --subfolder gdn_decode_qk4_v8_d128_k_last`

## Explorer Message: Research

```text
Read `gdn_decode_qk4_v8_d128_k_last/solution/cuda/kernel.cu`, `logs/decode/bench_history.jsonl`, and `logs/decode/optimization_log.md`.
Also read any relevant materials under `docs/` before ranking ideas.
Actively consult relevant official documentation and papers if they help validate bottlenecks or improve the proposed optimization.
Treat the latest comparable quick benchmark entry as baseline.
Identify the most likely current bottleneck in the decode kernel and rank 2-3 next optimization ideas for a single iteration.
Return the top recommendation, expected impact, implementation sketch, any correctness constraints, which `docs/` files informed the recommendation, and which official documents or papers materially influenced it.
```

## Worker Message: Implement

```text
Implement exactly one optimization in `gdn_decode_qk4_v8_d128_k_last/solution/cuda/kernel.cu`.
Use the selected decode optimization idea only. Preserve correctness, interfaces, tensor layout, and destination-passing behavior.
Do not mix unrelated refactors or cleanup into this edit.
At the end, report files changed, what changed, and any assumptions that must remain true.
```

## Explorer Message: Review

```text
Review the edited `gdn_decode_qk4_v8_d128_k_last/solution/cuda/kernel.cu` before benchmarking.
Check indexing, shapes, GDN formula preservation, q/k to v-head mapping, synchronization safety, and obvious performance regressions.
Return pass or fail, findings by severity, and whether benchmarking is safe.
```

## Worker Message: Evaluate

```text
Run the quick Modal benchmark for decode:
`conda run -n fi-bench modal run scripts/run_modal_subfolder.py --subfolder gdn_decode_qk4_v8_d128_k_last`
Parse the benchmark output, append a new entry to `logs/decode/bench_history.jsonl`, compare it against the latest comparable quick benchmark baseline already in that log, and recommend `revert` or `commit`.
The appended JSON line must include `decision` with that final recommendation.
Use correctness regression or no performance improvement as `revert`.
```
