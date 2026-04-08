# Researcher Prompt

Use this as the message for a Codex `explorer` subagent.

## Goal

Rank the next high-impact optimization for the target GDN kernel using the current implementation and the latest comparable quick benchmark baseline.

## Inputs To Provide

- Target kernel: `decode` or `prefill`
- Exact source file path
- External research mode: `local_only`, `auto`, or `required`
- Latest benchmark entry from `logs/<kernel>/bench_history.jsonl`
- Recent entries from `logs/<kernel>/optimization_log.md`
- Relevant documents under `docs/`, especially items matching the target kernel or current optimization topic
- Relevant official documentation and papers for CUDA, GPU architecture, kernels, linear attention, or related implementation techniques when allowed by the active research mode
- Optional topic if the iteration is focused

## Required Work

1. Read the current kernel implementation.
2. Read the latest comparable quick benchmark result.
3. Read recent optimization notes to avoid repeated failed ideas.
4. Read the most relevant `docs/` materials before proposing ideas. Prefer kernel-specific notes first, then broader architecture or CUDA references.
5. Obey the provided external research mode:
   - `local_only`: do not use external web sources.
   - `auto`: use external official documentation or papers only if local kernel, profile, logs, and `docs/` evidence is insufficient to justify the recommendation.
   - `required`: actively consult relevant official documentation and papers when they can improve bottleneck analysis, validate an optimization idea, or reveal better implementation tactics.
6. When external sources are used, prefer primary sources for technical claims: official vendor documentation, official framework documentation, and original or authoritative papers.
7. For prefill, treat `scripts/profile_kernel.py` as directional if it diverges from the current wrapper dispatch or the latest logged quick-benchmark evidence.
8. Use the latest comparable quick benchmark baseline with `decision == "commit"` for the same kernel and workload mode when available. If none exists yet, use the latest quick benchmark entry and call out that bootstrap assumption explicitly.
9. Identify the most plausible current bottlenecks.
10. Research relevant GDN, linear attention, CUDA, and B200 ideas if needed within the limits of the active research mode.
11. Return 2-3 ranked optimization ideas for the next single iteration.

## Output

- Current bottleneck summary
- Comparable baseline entry used and why it qualifies
- Research mode used
- Which `docs/` materials were consulted and how they influenced the ranking
- Whether external sources were consulted and why
- Which official documents or papers were consulted and how they influenced the ranking, if any
- Ranked ideas with expected impact
- Why the top idea is the best next single experiment
- Key implementation constraints to preserve correctness
