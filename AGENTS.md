# Project Agents

This repository uses project-local Codex guidance only. Do not rely on `.claude/` workflows for Codex work.

## Primary Workflow

For GDN kernel optimization work, use the project-local skill:
- `.agents/skills/gdn-optimization-workflow/SKILL.md`

Apply this loop:
1. Read the current decode or prefill kernel.
2. Read the latest comparable quick benchmark entry from the kernel-specific history log.
3. Profile the current kernel on Modal before researching optimizations.
4. Research the next high-impact optimization.
5. Implement one optimization only.
6. Review the edited kernel before benchmarking.
7. Run the quick Modal benchmark and log the result.
8. Revert if correctness regressed or speedup did not improve.
9. Commit only if correctness is preserved and performance improved.

## Project Facts

- Target kernels:
  - `gdn_decode_qk4_v8_d128_k_last/solution/cuda/kernel.cu`
  - `gdn_prefill_qk4_v8_d128_k_last/solution/cuda/kernel.cu`
- Benchmark logs:
  - `logs/decode/bench_history.jsonl`
  - `logs/prefill/bench_history.jsonl`
- Optimization notes:
  - `logs/decode/optimization_log.md`
  - `logs/prefill/optimization_log.md`
- Use quick Modal benchmark only in this workflow.
- Baseline must come from the most recent comparable logged quick benchmark result.

## Subagent Use

When delegation helps, use built-in Codex subagents only:
- `explorer` for research and code reading
- `worker` for implementation or benchmark execution

Prompt templates for those roles live under:
- `.agents/skills/gdn-optimization-workflow/references/`

Ready-to-use kernel runbooks:
- `.agents/skills/gdn-optimization-workflow/references/decode-template.md`
- `.agents/skills/gdn-optimization-workflow/references/prefill-template.md`
