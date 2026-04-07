# Kernel Reviewer Prompt

Use this as the message for a Codex `explorer` subagent.

## Goal

Review the edited GDN kernel before benchmarking.

## Inputs To Provide

- Target kernel name
- Edited file path
- Summary of the intended optimization

## Review Checklist

1. Tensor shapes and indexing still match the definition.
2. Decode or prefill signature still matches the task.
3. GDN formula is preserved for:
   - `g`
   - `beta`
   - state update
   - output projection
4. Query/key to value-head mapping is still correct.
5. Shared memory, synchronization, and race safety still hold.
6. The edit does not introduce an obvious performance regression such as extra global passes or inflated smem use.

## Output

- `pass` or `fail`
- Findings ordered by severity
- Code regions to revisit if any
- Whether benchmarking is safe
