# GDN Prefill B200 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current sequential CUDA prefill kernel with a `qk4 / v8 / d128 / k-last`-specific Python + CuTe/CUTLASS Blackwell submission that packs cleanly, runs on Modal B200, and supports exactly two runtime execution paths.

**Architecture:** Keep the submission self-contained under `gdn_prefill_qk4_v8_d128_k_last/solution/python`. `prefill_contract.py` owns pure-Python validation and gate preparation for the exact `qk4 / v8 / d128 / k-last` contract, `main.py` owns the benchmark entrypoint and varlen adaptation, and `gdn_blackwell/` owns compile/cache, dispatch, and the vendored Blackwell kernel code. Repository scripts are extended only enough to pack Python solutions and benchmark them on Modal with the required dependencies.

**Tech Stack:** Python 3.12, PyTorch, CuTe/CUTLASS DSL, cuda-python, FlashInfer-Bench, Modal B200

---

## File Structure

- Modify: `scripts/pack_solution.py`
  Purpose: Add `language = "python"` support, map Python solutions to `solution/python`, and forward `dependencies` plus `destination_passing_style`.
- Modify: `scripts/run_modal_subfolder.py`
  Purpose: Ensure the Modal development image has the Python runtime dependencies needed for the CuTe/CUTLASS solution.
- Modify: `gdn_prefill_qk4_v8_d128_k_last/config.toml`
  Purpose: Switch the prefill submission from CUDA to Python entrypoint metadata.
- Create: `gdn_prefill_qk4_v8_d128_k_last/solution/python/prefill_contract.py`
  Purpose: Own exact-definition validation, gate/beta preparation, and varlen shape normalization without importing CuTe.
- Create: `gdn_prefill_qk4_v8_d128_k_last/solution/python/main.py`
  Purpose: Benchmark entrypoint `run`, host-side preprocessing, and integration with the kernel package.
- Create: `gdn_prefill_qk4_v8_d128_k_last/solution/python/gdn_blackwell/__init__.py`
  Purpose: Package export surface for the kernel runtime.
- Create: `gdn_prefill_qk4_v8_d128_k_last/solution/python/gdn_blackwell/dispatch.py`
  Purpose: Hold pure-Python path selection and cache-key helpers for the two runtime paths.
- Create: `gdn_prefill_qk4_v8_d128_k_last/solution/python/gdn_blackwell/gdn.py`
  Purpose: Vendored baseline-derived kernel runtime, compile/cache integration, and path-specific launch logic.
- Create: `gdn_prefill_qk4_v8_d128_k_last/solution/python/gdn_blackwell/gdn_helpers.py`
  Purpose: Shared layout and helper utilities copied from the released baseline, then trimmed.
- Create: `gdn_prefill_qk4_v8_d128_k_last/solution/python/gdn_blackwell/gdn_tile_scheduler.py`
  Purpose: Varlen chunk scheduler copied from the baseline, then trimmed to this definition.
- Create: `gdn_prefill_qk4_v8_d128_k_last/tests/test_pack_python.py`
  Purpose: Verify repository tooling can locate and describe Python solutions.
- Create: `gdn_prefill_qk4_v8_d128_k_last/tests/test_prefill_contract.py`
  Purpose: Verify pure-Python validation and gate preparation logic.
- Create: `gdn_prefill_qk4_v8_d128_k_last/tests/test_main_contract.py`
  Purpose: Verify `main.run` preserves the benchmark ABI and varlen shape semantics.
- Create: `gdn_prefill_qk4_v8_d128_k_last/tests/test_dispatch.py`
  Purpose: Verify the two-path selection and cache-key policy without requiring CuTe imports.
- Create: `gdn_prefill_qk4_v8_d128_k_last/tests/test_pack_end_to_end.py`
  Purpose: Verify the subfolder packs into a Python `solution.json` with the expected metadata.

## Task 1: Add Python Solution Tooling Support

**Files:**
- Modify: `scripts/pack_solution.py`
- Modify: `scripts/run_modal_subfolder.py`
- Test: `gdn_prefill_qk4_v8_d128_k_last/tests/test_pack_python.py`

- [ ] **Step 1: Write the failing packer support test**

```python
from pathlib import Path
import scripts.pack_solution as ps

project_root = Path(__file__).resolve().parents[1]
assert ps._get_source_dir("python", project_root) == project_root / "solution" / "python"
spec = ps._build_spec_from_config(
    {
        "language": "python",
        "entry_point": "main.py::run",
        "dependencies": ["cutlass", "cuda-python"],
        "destination_passing_style": False,
    }
)
assert spec.language == "python"
assert spec.entry_point == "main.py::run"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python gdn_prefill_qk4_v8_d128_k_last/tests/test_pack_python.py`
Expected: FAIL with `AttributeError` because `_get_source_dir` and `_build_spec_from_config` do not exist yet.

- [ ] **Step 3: Write the minimal tooling implementation**

```python
def _get_source_dir(language: str, project_root: Path) -> Path:
    mapping = {
        "triton": project_root / "solution" / "triton",
        "cuda": project_root / "solution" / "cuda",
        "python": project_root / "solution" / "python",
    }
    return mapping[language]

def _build_spec_from_config(build_config: dict) -> BuildSpec:
    return BuildSpec(
        language=build_config["language"],
        target_hardware=["cuda"],
        entry_point=build_config["entry_point"],
        dependencies=build_config.get("dependencies", []),
        destination_passing_style=build_config.get("destination_passing_style", True),
    )
```

Also update the Modal image in `scripts/run_modal_subfolder.py` to install `cutlass` and `cuda-python` so quick B200 benchmarks can import the Python solution during development.

- [ ] **Step 4: Run the test and a syntax check**

Run: `python gdn_prefill_qk4_v8_d128_k_last/tests/test_pack_python.py`
Expected: PASS

Run: `python -m compileall scripts/pack_solution.py scripts/run_modal_subfolder.py`
Expected: PASS with both files compiled.

- [ ] **Step 5: Commit**

```bash
git add scripts/pack_solution.py scripts/run_modal_subfolder.py gdn_prefill_qk4_v8_d128_k_last/tests/test_pack_python.py
git commit -m "feat: add python solution tooling support"
```

## Task 2: Lock The Pure-Python Prefill Contract

**Files:**
- Modify: `gdn_prefill_qk4_v8_d128_k_last/config.toml`
- Create: `gdn_prefill_qk4_v8_d128_k_last/solution/python/prefill_contract.py`
- Test: `gdn_prefill_qk4_v8_d128_k_last/tests/test_prefill_contract.py`

- [ ] **Step 1: Write the failing contract test**

```python
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "solution" / "python"))
from prefill_contract import prepare_g_beta, validate_inputs

T = 16
q = torch.empty((T, 4, 128), dtype=torch.bfloat16)
k = torch.empty((T, 4, 128), dtype=torch.bfloat16)
v = torch.empty((T, 8, 128), dtype=torch.bfloat16)
state = torch.empty((1, 8, 128, 128), dtype=torch.float32)
A_log = torch.zeros((8,), dtype=torch.float32)
a = torch.zeros((T, 8), dtype=torch.bfloat16)
dt_bias = torch.zeros((8,), dtype=torch.float32)
b = torch.zeros((T, 8), dtype=torch.bfloat16)
cu_seqlens = torch.tensor([0, T], dtype=torch.int64)

gate_log, beta = prepare_g_beta(A_log, a, dt_bias, b)
validate_inputs(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens)
assert gate_log.dtype == torch.float32
assert beta.dtype == torch.float32
assert gate_log.shape == (T, 8)
assert beta.shape == (T, 8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python gdn_prefill_qk4_v8_d128_k_last/tests/test_prefill_contract.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'prefill_contract'`.

- [ ] **Step 3: Implement the contract module and switch config**

```python
def prepare_g_beta(A_log, a, dt_bias, b):
    x = a.float() + dt_bias.float()
    gate_log = -torch.exp(A_log.float()) * F.softplus(x)
    beta = torch.sigmoid(b.float())
    return gate_log, beta

def validate_inputs(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens):
    assert tuple(q.shape[1:]) == (4, 128)
    assert tuple(k.shape[1:]) == (4, 128)
    assert tuple(v.shape[1:]) == (8, 128)
    assert state.dtype == torch.float32
```

Update `config.toml` to:

```toml
[build]
language = "python"
entry_point = "main.py::run"
dependencies = ["cutlass", "cuda-python"]
destination_passing_style = false
```

- [ ] **Step 4: Run the contract test**

Run: `python gdn_prefill_qk4_v8_d128_k_last/tests/test_prefill_contract.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add gdn_prefill_qk4_v8_d128_k_last/config.toml \
  gdn_prefill_qk4_v8_d128_k_last/solution/python/prefill_contract.py \
  gdn_prefill_qk4_v8_d128_k_last/tests/test_prefill_contract.py
git commit -m "feat: add prefill python contract module"
```

## Task 3: Wire The Benchmark Entrypoint Before The Kernel

**Files:**
- Create: `gdn_prefill_qk4_v8_d128_k_last/solution/python/main.py`
- Create: `gdn_prefill_qk4_v8_d128_k_last/solution/python/gdn_blackwell/__init__.py`
- Create: `gdn_prefill_qk4_v8_d128_k_last/solution/python/gdn_blackwell/gdn.py`
- Test: `gdn_prefill_qk4_v8_d128_k_last/tests/test_main_contract.py`

- [ ] **Step 1: Write the failing entrypoint contract test**

```python
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "solution" / "python"))
import main

calls = {}

def fake_chunk_gated_delta_rule(**kwargs):
    calls.update(kwargs)
    out = torch.empty_like(kwargs["v"])
    state = torch.empty_like(kwargs["initial_state"])
    return out, state

main.chunk_gated_delta_rule = fake_chunk_gated_delta_rule
T = 12
q = torch.empty((T, 4, 128), dtype=torch.bfloat16)
k = torch.empty((T, 4, 128), dtype=torch.bfloat16)
v = torch.empty((T, 8, 128), dtype=torch.bfloat16)
state = torch.empty((1, 8, 128, 128), dtype=torch.float32)
A_log = torch.zeros((8,), dtype=torch.float32)
a = torch.zeros((T, 8), dtype=torch.bfloat16)
dt_bias = torch.zeros((8,), dtype=torch.float32)
b = torch.zeros((T, 8), dtype=torch.bfloat16)
cu_seqlens = torch.tensor([0, T], dtype=torch.int64)

output, new_state = main.run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, None)
assert output.shape == (T, 8, 128)
assert new_state.shape == state.shape
assert calls["q"].shape == (1, T, 4, 128)
assert calls["k"].shape == (1, T, 4, 128)
assert calls["v"].shape == (1, T, 8, 128)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python gdn_prefill_qk4_v8_d128_k_last/tests/test_main_contract.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'main'`.

- [ ] **Step 3: Implement the entrypoint and a stub kernel module**

```python
from prefill_contract import prepare_g_beta, validate_inputs
from gdn_blackwell.gdn import chunk_gated_delta_rule

def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    validate_inputs(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens)
    gate_log, beta = prepare_g_beta(A_log, a, dt_bias, b)
    q_b, k_b, v_b = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
    output, new_state = chunk_gated_delta_rule(
        q=q_b,
        k=k_b,
        v=v_b,
        g=gate_log,
        beta=beta,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        scale=scale,
    )
    return output.squeeze(0), new_state
```

Create a minimal `gdn_blackwell/gdn.py` stub with a `chunk_gated_delta_rule` symbol so imports work before the real kernel runtime is vendored in the next task.

- [ ] **Step 4: Run the entrypoint contract test**

Run: `python gdn_prefill_qk4_v8_d128_k_last/tests/test_main_contract.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add gdn_prefill_qk4_v8_d128_k_last/solution/python/main.py \
  gdn_prefill_qk4_v8_d128_k_last/solution/python/gdn_blackwell/__init__.py \
  gdn_prefill_qk4_v8_d128_k_last/solution/python/gdn_blackwell/gdn.py \
  gdn_prefill_qk4_v8_d128_k_last/tests/test_main_contract.py
git commit -m "feat: wire python prefill entrypoint"
```

## Task 4: Vendor The Blackwell Baseline Runtime

**Files:**
- Modify: `gdn_prefill_qk4_v8_d128_k_last/solution/python/gdn_blackwell/gdn.py`
- Create: `gdn_prefill_qk4_v8_d128_k_last/solution/python/gdn_blackwell/gdn_helpers.py`
- Create: `gdn_prefill_qk4_v8_d128_k_last/solution/python/gdn_blackwell/gdn_tile_scheduler.py`
- Test: `gdn_prefill_qk4_v8_d128_k_last/tests/test_pack_end_to_end.py`

- [ ] **Step 1: Write the failing pack end-to-end test**

```python
from pathlib import Path
import json
import scripts.pack_solution as ps

project_root = Path(__file__).resolve().parents[1]
ps.PROJECT_ROOT = project_root
output_path = project_root / "solution.json"
ps.pack_solution(output_path)
packed = json.loads(output_path.read_text())
assert packed["spec"]["language"] == "python"
assert packed["spec"]["entry_point"] == "main.py::run"
assert any(src["path"] == "gdn_blackwell/gdn.py" for src in packed["sources"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python gdn_prefill_qk4_v8_d128_k_last/tests/test_pack_end_to_end.py`
Expected: FAIL because the vendored `gdn_blackwell` source tree is still incomplete.

- [ ] **Step 3: Replace the stub runtime with the released baseline-derived files**

Copy in the released files as the starting point:

```text
main.py                        -> already adapted in Task 3
gdn_blackwell/gdn.py          -> vendor and then trim later
gdn_blackwell/gdn_helpers.py  -> vendor
gdn_blackwell/gdn_tile_scheduler.py -> vendor
```

Immediately after vendoring:

- change imports to work from `solution/python`;
- keep the current runtime single-path until Task 6;
- preserve the released kernel’s gate convention (`g`/`gate_log`) consistently with `prefill_contract.py`;
- do not add extra feature support beyond this definition.

- [ ] **Step 4: Run pack and syntax checks**

Run: `python gdn_prefill_qk4_v8_d128_k_last/tests/test_pack_end_to_end.py`
Expected: PASS

Run: `python -m compileall gdn_prefill_qk4_v8_d128_k_last/solution/python`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add gdn_prefill_qk4_v8_d128_k_last/solution/python/gdn_blackwell \
  gdn_prefill_qk4_v8_d128_k_last/tests/test_pack_end_to_end.py
git commit -m "feat: vendor blackwell prefill runtime"
```

## Task 5: Specialize The Kernel To The Exact Contest Definition

**Files:**
- Modify: `gdn_prefill_qk4_v8_d128_k_last/solution/python/prefill_contract.py`
- Modify: `gdn_prefill_qk4_v8_d128_k_last/solution/python/main.py`
- Modify: `gdn_prefill_qk4_v8_d128_k_last/solution/python/gdn_blackwell/gdn.py`
- Test: `gdn_prefill_qk4_v8_d128_k_last/tests/test_prefill_contract.py`

- [ ] **Step 1: Extend the failing contract test with rejection cases**

```python
bad_v = torch.empty((T, 7, 128), dtype=torch.bfloat16)
try:
    validate_inputs(q, k, bad_v, state, A_log, a, dt_bias, b, cu_seqlens)
except ValueError as exc:
    assert "v" in str(exc)
else:
    raise AssertionError("expected validate_inputs to reject unsupported v-head count")
```

- [ ] **Step 2: Run the contract test to verify it fails**

Run: `python gdn_prefill_qk4_v8_d128_k_last/tests/test_prefill_contract.py`
Expected: FAIL because the current validation is still too loose.

- [ ] **Step 3: Tighten host and kernel specialization**

Implement the exact-definition checks and remove unused generality:

```python
if tuple(q.shape[1:]) != (4, 128):
    raise ValueError("expected q shape [T, 4, 128]")
if tuple(v.shape[1:]) != (8, 128):
    raise ValueError("expected v shape [T, 8, 128]")
if cu_seqlens.dtype != torch.int64:
    raise ValueError("cu_seqlens must be int64")
```

In `gdn_blackwell/gdn.py`, prune unused code paths so the runtime only needs to support:

- varlen prefill;
- `head_dim = 128`;
- `q_heads = k_heads = 4`;
- `v_heads = 8`;
- `fp32` state output.

- [ ] **Step 4: Run the strengthened contract test**

Run: `python gdn_prefill_qk4_v8_d128_k_last/tests/test_prefill_contract.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add gdn_prefill_qk4_v8_d128_k_last/solution/python/prefill_contract.py \
  gdn_prefill_qk4_v8_d128_k_last/solution/python/main.py \
  gdn_prefill_qk4_v8_d128_k_last/solution/python/gdn_blackwell/gdn.py \
  gdn_prefill_qk4_v8_d128_k_last/tests/test_prefill_contract.py
git commit -m "feat: specialize prefill kernel to qk4 v8 d128"
```

## Task 6: Add Two-Path Dispatch And Narrow Cache Keys

**Files:**
- Create: `gdn_prefill_qk4_v8_d128_k_last/solution/python/gdn_blackwell/dispatch.py`
- Modify: `gdn_prefill_qk4_v8_d128_k_last/solution/python/main.py`
- Modify: `gdn_prefill_qk4_v8_d128_k_last/solution/python/gdn_blackwell/gdn.py`
- Test: `gdn_prefill_qk4_v8_d128_k_last/tests/test_dispatch.py`

- [ ] **Step 1: Write the failing dispatch policy test**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "solution" / "python"))
from gdn_blackwell.dispatch import choose_path, make_cache_key

assert choose_path(total_seq_len=64, num_seqs=1) == "small"
assert choose_path(total_seq_len=8192, num_seqs=32) == "large"
assert make_cache_key("small", "bf16", True, True)[0] == "small"
assert make_cache_key("large", "bf16", True, True)[0] == "large"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python gdn_prefill_qk4_v8_d128_k_last/tests/test_dispatch.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'gdn_blackwell.dispatch'`.

- [ ] **Step 3: Implement the dispatch helper and integrate it**

```python
def choose_path(total_seq_len: int, num_seqs: int) -> str:
    if total_seq_len <= SMALL_TOTAL_SEQ_LEN and num_seqs <= SMALL_NUM_SEQS:
        return "small"
    return "large"

def make_cache_key(path_name: str, dtype_name: str, is_varlen: bool, has_state: bool):
    return (path_name, dtype_name, is_varlen, has_state)
```

Then thread `path_name` through `main.py` and `gdn.py` so compile caching produces exactly two kernel families, not one kernel per concrete shape.

- [ ] **Step 4: Run the dispatch test**

Run: `python gdn_prefill_qk4_v8_d128_k_last/tests/test_dispatch.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add gdn_prefill_qk4_v8_d128_k_last/solution/python/gdn_blackwell/dispatch.py \
  gdn_prefill_qk4_v8_d128_k_last/solution/python/main.py \
  gdn_prefill_qk4_v8_d128_k_last/solution/python/gdn_blackwell/gdn.py \
  gdn_prefill_qk4_v8_d128_k_last/tests/test_dispatch.py
git commit -m "feat: add dual-path dispatch for prefill"
```

## Task 7: Run Verification, Benchmarks, And Logging

**Files:**
- Modify: `logs/prefill/bench_history.jsonl`
- Modify: `logs/prefill/optimization_log.md`
- Test: `gdn_prefill_qk4_v8_d128_k_last/tests/test_pack_end_to_end.py`

- [ ] **Step 1: Re-run the local pack check before hitting Modal**

Run: `python gdn_prefill_qk4_v8_d128_k_last/tests/test_pack_end_to_end.py`
Expected: PASS and `solution.json` exists under the prefill subfolder.

- [ ] **Step 2: Run the quick Modal smoke benchmark**

Run: `modal run scripts/run_modal_subfolder.py --subfolder gdn_prefill_qk4_v8_d128_k_last`
Expected: PASS on representative workloads with no import/build errors.

- [ ] **Step 3: Run the full Modal benchmark**

Run: `modal run scripts/run_modal_subfolder.py --subfolder gdn_prefill_qk4_v8_d128_k_last --no-quick`
Expected: PASS across the full workload set with correctness metrics present.

- [ ] **Step 4: Record the benchmark results and perform final verification**

Use `@superpowers:verification-before-completion` before claiming success.

Append:

- one JSON line to `logs/prefill/bench_history.jsonl` with timestamp, commit, mean speedup, mean latency, correctness summary, and notes about the active dispatch thresholds;
- one entry to `logs/prefill/optimization_log.md` summarizing the final kernel structure and the observed trade-off between short and long workloads.

- [ ] **Step 5: Commit**

```bash
git add gdn_prefill_qk4_v8_d128_k_last \
  scripts/pack_solution.py \
  scripts/run_modal_subfolder.py \
  logs/prefill/bench_history.jsonl \
  logs/prefill/optimization_log.md
git commit -m "feat: optimize GDN prefill for B200"
```

## Notes For Execution

- Do not reintroduce the old CUDA `kernel.cu` path into the runtime flow; the Python solution is the intended submission format for this optimization pass.
- Keep the gate representation consistent end-to-end. The released baseline uses a log-space gate convention; `prefill_contract.py`, `main.py`, and `gdn_blackwell/gdn.py` must all agree.
- If the Modal quick run fails because `cutlass` or `cuda-python` is still unavailable inside the benchmark image, fix the image first rather than debugging the kernel blindly.
- Keep the runtime limited to exactly two path names: `"small"` and `"large"`.
- Do not tune thresholds until the single-path specialized runtime is correct and benchmarkable.
