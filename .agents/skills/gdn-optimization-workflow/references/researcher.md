# Researcher Prompt

Use this as the message for a Codex `explorer` subagent.

## Goal

Rank the next high-impact optimization for the target GDN kernel using the current implementation and the latest comparable quick benchmark baseline.

## Inputs To Provide

- Target kernel: `decode` or `prefill`
- Exact source file path
- Latest benchmark entry from `logs/<kernel>/bench_history.jsonl`
- Recent entries from `logs/<kernel>/optimization_log.md`
- Optional topic if the iteration is focused

## Required Work

1. Read the current kernel implementation.
2. Read the latest comparable quick benchmark result.
3. Read recent optimization notes to avoid repeated failed ideas.
4. Identify the most plausible current bottlenecks.
5. Research relevant GDN, linear attention, CUDA, and B200 ideas if needed.
6. Return 2-3 ranked optimization ideas for the next single iteration.

## Output

- Current bottleneck summary
- Ranked ideas with expected impact
- Why the top idea is the best next single experiment
- Key implementation constraints to preserve correctness
