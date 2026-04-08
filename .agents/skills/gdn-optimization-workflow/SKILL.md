---
name: gdn-optimization-workflow
description: Use when iterating on the GDN decode or prefill CUDA kernels in this repository.
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
3. Profile the current kernel on Modal before optimization research:
   - `conda run -n fi-bench modal run scripts/profile_kernel.py --kernel <decode|prefill>`
   - Use the profile to identify the current launch, occupancy, memory, and synchronization bottlenecks.
   - For prefill, treat the checked-in profiling harness as directional rather than authoritative when it diverges from the current wrapper dispatch or the latest logged quick-benchmark evidence.
4. Research the next optimization candidate.
   - During research, read relevant documents under `docs/` first when they apply to the target kernel or optimization topic.
   - Then actively consult relevant official documentation and papers when they can sharpen the bottleneck analysis or proposed optimization.
   - Ground the recommendation in the latest profile findings, but resolve prefill conflicts using the current kernel plus the latest benchmark and optimization logs.
5. Choose exactly one idea for the iteration.
6. Implement only that one idea.
7. Review the edited kernel before benchmarking.
8. Run the quick Modal benchmark:
   - `conda run -n fi-bench modal run scripts/run_modal_subfolder.py --subfolder <subfolder>`
9. Append the benchmark result to the correct kernel log.
   - Include the final workflow decision in the JSONL entry as `decision: "commit"` or `decision: "revert"` for optimization iterations.
10. Append a short note to the matching optimization log summarizing the experiment, result, and decision.
11. Compare against the latest comparable quick benchmark baseline.
12. Revert only the current iteration's optimization edits if correctness regressed or performance did not improve.
13. Commit only if correctness is preserved and performance improved.

## Hard Rules

- One optimization idea per iteration.
- No full benchmark in this workflow.
- No unrelated cleanup mixed into the experiment.
- Baseline comes from the latest comparable logged quick benchmark result, not memory.
- A comparable baseline is the latest logged quick benchmark for the same kernel and workload mode whose `decision` is `commit`.
- If no optimization-iteration `commit` entry exists yet for that kernel, fall back to the latest available quick benchmark entry and call out that bootstrap assumption explicitly.
- Profile the current kernel before researching the next optimization idea.
- For prefill, `scripts/profile_kernel.py` is still required, but its results are only directional if they disagree with the current dispatch behavior or the latest benchmark history.
- Correctness regression always loses to speed.
- Research subagents should consult relevant `docs/` materials before proposing new optimization ideas.
- Research subagents should actively use relevant official documentation and papers, not just local notes, when evaluating optimization ideas.
- Research recommendations should cite the current profile findings that motivated the idea.
- Research recommendations should state which benchmark entry is being used as the comparable baseline.
- Optimization-iteration entries appended to `bench_history.jsonl` should include a `decision` field with `commit` or `revert`.
- Reverts in this workflow should back out only the current iteration's experiment and must not disturb unrelated local changes.

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
- Short optimization-log note in the kernel-specific markdown log
- Benchmark-history entry should include the final `decision`
- Decision: `revert` or `commit`
