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
assert calls["g"].shape == (1, T, 8)
assert calls["beta"].shape == (1, T, 8)
expected_g = torch.full((1, T, 8), -torch.log(torch.tensor(2.0)), dtype=torch.float32)
expected_beta = torch.full((1, T, 8), 0.5, dtype=torch.float32)
torch.testing.assert_close(calls["g"], expected_g)
torch.testing.assert_close(calls["beta"], expected_beta)
assert calls["initial_state"] is state
assert calls["cu_seqlens"] is cu_seqlens
assert calls["scale"] == 1.0

try:
    main.run(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        state,
        A_log,
        a,
        dt_bias,
        b,
        cu_seqlens,
        None,
    )
except ValueError as exc:
    assert "flat varlen" in str(exc)
else:
    raise AssertionError("Expected ValueError for pseudo-batched main.run inputs")


def fake_bad_chunk_gated_delta_rule(**kwargs):
    bad_output = torch.empty((T, 8, 128), dtype=torch.bfloat16)
    bad_state = torch.empty((1, 8, 128), dtype=torch.float32)
    return bad_output, bad_state


main.chunk_gated_delta_rule = fake_bad_chunk_gated_delta_rule

try:
    main.run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, None)
except ValueError as exc:
    assert "output" in str(exc) or "new_state" in str(exc)
else:
    raise AssertionError("Expected ValueError for malformed chunk_gated_delta_rule return")
