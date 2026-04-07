# Researcher Prompt

Use this as the message for a Codex `explorer` subagent.

## Goal

Rank the next high-impact optimization for the target GDN kernel using the current implementation and the latest comparable quick benchmark baseline.

## Inputs To Provide

- Target kernel: `decode` or `prefill`
- Exact source file path
- Latest benchmark entry from `logs/<kernel>/bench_history.jsonl`
- Recent entries from `logs/<kernel>/optimization_log.md`
- Relevant documents under `docs/`, especially items matching the target kernel or current optimization topic
- Relevant official documentation and papers for CUDA, GPU architecture, kernels, linear attention, or related implementation techniques
- Optional topic if the iteration is focused

## Required Work

1. Read the current kernel implementation.
2. Read the latest comparable quick benchmark result.
3. Read recent optimization notes to avoid repeated failed ideas.
4. Read the most relevant `docs/` materials before proposing ideas. Prefer kernel-specific notes first, then broader architecture or CUDA references.
5. Actively consult relevant official documentation and papers when they can improve bottleneck analysis, validate an optimization idea, or reveal better implementation tactics.
6. Prefer primary sources for technical claims: official vendor documentation, official framework documentation, and original or authoritative papers.
7. Identify the most plausible current bottlenecks.
8. Research relevant GDN, linear attention, CUDA, and B200 ideas if needed.
9. Return 2-3 ranked optimization ideas for the next single iteration.

## Output

- Current bottleneck summary
- Which `docs/` materials were consulted and how they influenced the ranking
- Which official documents or papers were consulted and how they influenced the ranking
- Ranked ideas with expected impact
- Why the top idea is the best next single experiment
- Key implementation constraints to preserve correctness
