import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "solution" / "python"))
import gdn_blackwell.gdn as gdn


def make_inputs(T=12, N=2):
    q = torch.empty((1, T, 4, 128), dtype=torch.bfloat16)
    k = torch.empty((1, T, 4, 128), dtype=torch.bfloat16)
    v = torch.empty((1, T, 8, 128), dtype=torch.bfloat16)
    g = torch.empty((1, T, 8), dtype=torch.float32)
    beta = torch.empty((1, T, 8), dtype=torch.float32)
    state = torch.empty((N, 8, 128, 128), dtype=torch.float32)
    cu = torch.tensor([0, 5, T], dtype=torch.int64)
    return q, k, v, g, beta, state, cu


runtime_calls = {"launches": 0, "cache_keys": []}
compiled_cache = {}


def fake_get_compiled(problem_size, dtype, is_varlen, is_initial_state, is_output_state, scale):
    key = (problem_size, dtype, is_varlen, is_initial_state, is_output_state, scale)
    runtime_calls["cache_keys"].append(key)
    if key not in compiled_cache:
        def fake_compiled(*args, **kwargs):
            runtime_calls["launches"] += 1

        compiled_cache[key] = {"compiled_gdn": fake_compiled}
    return compiled_cache[key]


gdn.GDN.can_implement = staticmethod(lambda *args, **kwargs: True)
gdn._get_compiled_gdn_prefill_kernel = fake_get_compiled
gdn.cuda.CUstream = lambda stream: stream
gdn.torch.cuda.current_stream = lambda: SimpleNamespace(cuda_stream=0)

if hasattr(gdn, "_CU_SEQLENS_METADATA_CACHE"):
    gdn._CU_SEQLENS_METADATA_CACHE.clear()

q, k, v, gate, beta, state, cu_seqlens = make_inputs()
original_tolist = torch.Tensor.tolist
tolist_calls = 0


def counting_tolist(self, *args, **kwargs):
    global tolist_calls
    tolist_calls += 1
    return original_tolist(self, *args, **kwargs)


with mock.patch.object(torch.Tensor, "tolist", counting_tolist):
    output, output_state = gdn.chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=gate,
        beta=beta,
        initial_state=state,
        cu_seqlens=cu_seqlens,
        scale=1.0,
    )
    second_output, second_output_state = gdn.chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=gate,
        beta=beta,
        initial_state=state,
        cu_seqlens=cu_seqlens,
        scale=1.0,
    )

assert output.shape == v.shape
assert output_state.shape == state.shape
assert second_output.shape == v.shape
assert second_output_state.shape == state.shape
assert runtime_calls["launches"] == 2
assert len(set(runtime_calls["cache_keys"])) == 1
assert tolist_calls == 1

bad_q = torch.empty((1, q.shape[1], 5, 128), dtype=torch.bfloat16)
try:
    gdn.chunk_gated_delta_rule(
        q=bad_q,
        k=bad_q,
        v=v,
        g=gate,
        beta=beta,
        initial_state=state,
        cu_seqlens=cu_seqlens,
        scale=1.0,
    )
except ValueError as exc:
    assert "q.shape" in str(exc)
else:
    raise AssertionError("Expected wrapper to reject out-of-scope q/k shape before launch")

bad_g = gate.to(dtype=torch.bfloat16)
try:
    gdn.chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=bad_g,
        beta=beta,
        initial_state=state,
        cu_seqlens=cu_seqlens,
        scale=1.0,
    )
except TypeError as exc:
    assert "g.dtype" in str(exc)
else:
    raise AssertionError("Expected wrapper to reject g dtype before launch")

bad_cu = torch.tensor([0, q.shape[1] - 1, q.shape[1] - 1], dtype=torch.int64)
try:
    gdn.chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=gate,
        beta=beta,
        initial_state=state,
        cu_seqlens=bad_cu,
        scale=1.0,
    )
except ValueError as exc:
    assert "end at T" in str(exc)
else:
    raise AssertionError("Expected wrapper to reject mismatched cu_seqlens end before launch")

assert runtime_calls["launches"] == 2
