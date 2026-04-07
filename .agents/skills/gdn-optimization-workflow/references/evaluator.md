# Evaluator Prompt

Use this as the message for a Codex `worker` subagent.

## Goal

Run the quick Modal benchmark for the target kernel, append the result to the correct history log, and compare it to the latest comparable logged quick benchmark baseline.

## Inputs To Provide

- Target kernel: `decode` or `prefill`
- Target subfolder
- Kernel-specific log path
- A short note describing the experiment

## Required Work

1. Run:
   - `conda run -n fi-bench modal run scripts/run_modal_subfolder.py --subfolder <subfolder>`
2. Parse:
   - status
   - mean speedup
   - mean latency
   - min and max speedup if available
   - correctness metrics
   - workload count
3. Append a new JSON line to the kernel-specific `bench_history.jsonl`.
   - Include the final workflow recommendation in that JSON object as `decision`, with value `commit` or `revert`.
4. Compare the new entry against the latest comparable existing quick benchmark entry in that same log.
5. Recommend `revert` or `commit`.

## Decision Rule

- Recommend `revert` if correctness regressed.
- Recommend `revert` if performance did not improve over baseline.
- Recommend `commit` only if correctness is preserved and performance improved.
- The appended benchmark-history entry should use that same final recommendation as its `decision` field.

## Output

- Parsed benchmark summary
- Baseline entry used for comparison
- Delta vs baseline
- Final recommendation: `revert` or `commit`
- Exact JSONL entry appended, including `decision`
