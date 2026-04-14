import sys
from pathlib import Path

import torch

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

expected_gate_log = torch.full((T, 8), -torch.log(torch.tensor(2.0)), dtype=torch.float32)
expected_beta = torch.full((T, 8), 0.5, dtype=torch.float32)
torch.testing.assert_close(gate_log, expected_gate_log)
torch.testing.assert_close(beta, expected_beta)

try:
    validate_inputs(
        q,
        k,
        torch.empty((T + 1, 8, 128), dtype=torch.bfloat16),
        state,
        A_log,
        a,
        dt_bias,
        b,
        cu_seqlens,
    )
except ValueError as exc:
    assert "v.shape" in str(exc)
else:
    raise AssertionError("Expected ValueError for invalid v batch shape")

try:
    validate_inputs(
        q,
        k,
        v,
        torch.empty((2, 8, 128, 128), dtype=torch.float32),
        A_log,
        a,
        dt_bias,
        b,
        cu_seqlens,
    )
except ValueError as exc:
    assert "state.shape" in str(exc)
else:
    raise AssertionError("Expected ValueError for invalid state shape")

try:
    validate_inputs(
        q.float(),
        k,
        v,
        state,
        A_log,
        a,
        dt_bias,
        b,
        cu_seqlens,
    )
except TypeError as exc:
    assert "q.dtype" in str(exc)
else:
    raise AssertionError("Expected TypeError for invalid q dtype")

multi_lengths = torch.tensor([0, 5, T], dtype=torch.int64)
multi_state = torch.empty((2, 8, 128, 128), dtype=torch.float32)
validate_inputs(q, k, v, multi_state, A_log, a, dt_bias, b, multi_lengths)

try:
    prepare_g_beta(A_log, a.float(), dt_bias, b)
except TypeError as exc:
    assert "a.dtype" in str(exc)
else:
    raise AssertionError("Expected TypeError for invalid a dtype")

try:
    prepare_g_beta(A_log, a, dt_bias, b.float())
except TypeError as exc:
    assert "b.dtype" in str(exc)
else:
    raise AssertionError("Expected TypeError for invalid b dtype")

try:
    prepare_g_beta(
        A_log,
        torch.zeros((T, 8), dtype=torch.bfloat16),
        dt_bias,
        torch.zeros((T + 1, 8), dtype=torch.bfloat16),
    )
except ValueError as exc:
    assert "shape" in str(exc)
else:
    raise AssertionError("Expected ValueError for mismatched a and b shapes")
