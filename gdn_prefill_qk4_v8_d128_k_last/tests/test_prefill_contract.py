import gc
import sys
import weakref
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "solution" / "python"))
import prefill_contract as pc
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
        torch.empty((T, 7, 128), dtype=torch.bfloat16),
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
    raise AssertionError("Expected ValueError for invalid v head count")

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

pc._CU_SEQLENS_METADATA_CACHE.clear()
tolist_calls = 0
original_tolist = torch.Tensor.tolist


def counting_tolist(self, *args, **kwargs):
    global tolist_calls
    tolist_calls += 1
    return original_tolist(self, *args, **kwargs)


with mock.patch.object(torch.Tensor, "item", side_effect=AssertionError("validate_inputs should not call Tensor.item()")), \
     mock.patch("torch.any", side_effect=AssertionError("validate_inputs should not call torch.any()")), \
     mock.patch.object(torch.Tensor, "tolist", counting_tolist):
    validate_inputs(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens)
    validate_inputs(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens)

assert tolist_calls == 1

pc._CU_SEQLENS_METADATA_CACHE.clear()
large_metadata = None
large_cu = torch.tensor([0, 128, 256, 384, 512, 640, 768, 896, 1025], dtype=torch.int64)
large_cu_key = id(large_cu)
large_cu_ref = weakref.ref(large_cu)
with mock.patch.object(torch.Tensor, "data_ptr", return_value=7):
    large_metadata = pc.get_cu_seqlens_metadata(large_cu)
assert large_metadata["sum_s_q"] == 1025
assert large_cu_key in pc._CU_SEQLENS_METADATA_CACHE
del large_cu
gc.collect()
assert large_cu_ref() is None
assert large_cu_key not in pc._CU_SEQLENS_METADATA_CACHE

small_cu = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14, 16], dtype=torch.int64)
small_state = torch.empty((8, 8, 128, 128), dtype=torch.float32)
small_q = torch.empty((16, 4, 128), dtype=torch.bfloat16)
small_k = torch.empty((16, 4, 128), dtype=torch.bfloat16)
small_v = torch.empty((16, 8, 128), dtype=torch.bfloat16)
small_a = torch.zeros((16, 8), dtype=torch.bfloat16)
small_b = torch.zeros((16, 8), dtype=torch.bfloat16)
with mock.patch.object(torch.Tensor, "data_ptr", return_value=7):
    small_metadata = pc.get_cu_seqlens_metadata(small_cu)

assert small_metadata["sum_s_q"] == 16
assert small_metadata["max_s_q"] == 2
validate_inputs(small_q, small_k, small_v, small_state, A_log, small_a, dt_bias, small_b, small_cu)
