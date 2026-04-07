---
name: gdn-optimization-workflow
description: Use when optimizing the GDN decode or prefill CUDA kernels in this repository and the work should follow the project's research, single-change, review, quick-benchmark, revert-or-commit loop.
---

# GDN Optimization Workflow

Use this skill for the two project kernels:
- `gdn_decode_qk4_v8_d128_k_last/solution/cuda/kernel.cu`
- `gdn_prefill_qk4_v8_d128_k_last/solution/cuda/kernel.cu`

## Workflow

1. Read the target kernel and the latest comparable quick benchmark entry from:
   - `logs/decode/bench_history.jsonl`
   - `logs/prefill/bench_history.jsonl`
2. Read the matching optimization log:
   - `logs/decode/optimization_log.md`
   - `logs/prefill/optimization_log.md`
3. Research the next optimization candidate.
   - During research, read relevant documents under `docs/` first when they apply to the target kernel or optimization topic.
   - Then actively consult relevant official documentation and papers when they can sharpen the bottleneck analysis or proposed optimization.
4. Choose exactly one idea for the iteration.
5. Implement only that one idea.
6. Review the edited kernel before benchmarking.
7. Run the quick Modal benchmark:
   - `conda run -n fi-bench modal run scripts/run_modal_subfolder.py --subfolder <subfolder>`
8. Append the benchmark result to the correct kernel log.
   - Include the final workflow decision in the JSONL entry as `decision: "commit"` or `decision: "revert"` for optimization iterations.
9. Compare against the latest comparable quick benchmark baseline.
10. Revert if correctness regressed or performance did not improve.
11. Commit only if correctness is preserved and performance improved.

## Hard Rules

- One optimization idea per iteration.
- No full benchmark in this workflow.
- No unrelated cleanup mixed into the experiment.
- Baseline comes from the latest comparable logged quick benchmark result, not memory.
- Correctness regression always loses to speed.
- Research subagents should consult relevant `docs/` materials before proposing new optimization ideas.
- Research subagents should actively use relevant official documentation and papers, not just local notes, when evaluating optimization ideas.
- Optimization-iteration entries appended to `bench_history.jsonl` should include a `decision` field with `commit` or `revert`.

## Delegation

Use Codex built-in subagents, not custom agent types:
- `explorer` for research and static review
- `worker` for code changes or benchmark execution

Read the matching prompt template before spawning:
- Research: `references/researcher.md`
- Implementation: `references/kernel-writer.md`
- Review: `references/kernel-reviewer.md`
- Benchmark and compare: `references/evaluator.md`

## Ready-To-Use Templates

Use these project-local templates when you want to run the workflow immediately:
- Decode runbook: `references/decode-template.md`
- Prefill runbook: `references/prefill-template.md`

Each runbook includes:
- target paths
- baseline log path
- benchmark command
- suggested `spawn_agent` messages for research, implementation, review, and evaluation

## Expected Outputs Per Iteration

- Selected optimization idea and rationale
- Kernel diff implementing exactly one change
- Review findings or explicit review pass
- New quick benchmark entry in the kernel-specific log
- Benchmark-history entry should include the final `decision`
- Decision: `revert` or `commit`
- Short optimization-log note summarizing the experiment
