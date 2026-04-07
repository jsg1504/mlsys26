# Kernel Writer Prompt

Use this as the message for a Codex `worker` subagent.

## Goal

Implement exactly one selected optimization in the target GDN kernel.

## Inputs To Provide

- Target file path
- Chosen optimization idea
- Short implementation sketch
- Any constraints discovered during research or review

## Rules

- You are not alone in the codebase. Do not revert unrelated edits.
- Modify only the files needed for this optimization.
- Preserve interfaces, tensor layouts, and destination-passing behavior.
- Keep state math in float32.
- Do not mix unrelated refactors or formatting-only cleanup into this change.
- Stop at one optimization idea.

## Handoff

Return:
- Files changed
- What changed
- What assumptions must remain true for correctness
- Any areas that need careful review before benchmarking
