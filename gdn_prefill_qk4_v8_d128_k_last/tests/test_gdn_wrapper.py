import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "solution" / "python"))
import gdn_blackwell.gdn as gdn
import prefill_contract as pc


def make_inputs(T=12, cu=None):
    q = torch.empty((1, T, 4, 128), dtype=torch.bfloat16)
    k = torch.empty((1, T, 4, 128), dtype=torch.bfloat16)
    v = torch.empty((1, T, 8, 128), dtype=torch.bfloat16)
    g = torch.empty((1, T, 8), dtype=torch.float32)
    beta = torch.empty((1, T, 8), dtype=torch.float32)
    if cu is None:
        cu = torch.tensor([0, 5, T], dtype=torch.int64)
    state = torch.empty((cu.numel() - 1, 8, 128, 128), dtype=torch.float32)
    return q, k, v, g, beta, state, cu


class FakeCompile:
    def __init__(self, runtime_calls):
        self.runtime_calls = runtime_calls

    def __getitem__(self, options):
        def compile_impl(
            gdn_kernel,
            q_iterator,
            k_iterator,
            v_iterator,
            o_iterator,
            gate_iterator,
            beta_iterator,
            problem_size,
            state_iterator,
            state_output_iterator,
            normalized_scale,
            cu_seqlens_tensor,
            stream,
        ):
            artifact_id = len(self.runtime_calls["compile_calls"]) + 1
            self.runtime_calls["compile_calls"].append(
                {
                    "problem_size": problem_size,
                    "scale": normalized_scale,
                }
            )

            def fake_compiled(*args, **kwargs):
                self.runtime_calls["launches"].append(
                    {
                        "problem_size": args[6],
                        "scale": args[9],
                    }
                )

            fake_compiled.artifact_id = artifact_id
            return fake_compiled

        return compile_impl


runtime_calls = {"compile_calls": [], "launches": []}


def fake_from_dlpack(tensor, assumed_align=16, enable_tvm_ffi=True):
    return SimpleNamespace(iterator=object())


gdn._get_compiled_gdn_prefill_kernel.cache_clear()
pc._CU_SEQLENS_METADATA_CACHE.clear()
original_tolist = torch.Tensor.tolist
tolist_calls = 0


def counting_tolist(self, *args, **kwargs):
    global tolist_calls
    tolist_calls += 1
    return original_tolist(self, *args, **kwargs)


with (
    mock.patch.object(gdn, "from_dlpack", side_effect=fake_from_dlpack),
    mock.patch.object(gdn, "cute", wraps=gdn.cute) as cute_module,
    mock.patch.object(gdn.cuda, "CUstream", side_effect=lambda stream: stream),
    mock.patch.object(gdn.torch.cuda, "current_stream", return_value=SimpleNamespace(cuda_stream=0)),
    mock.patch.object(torch.Tensor, "tolist", counting_tolist),
):
    cute_module.compile = FakeCompile(runtime_calls)

    q_small, k_small, v_small, gate_small, beta_small, state_small, cu_small = make_inputs()
    output, output_state = gdn.chunk_gated_delta_rule(
        q=q_small,
        k=k_small,
        v=v_small,
        g=gate_small,
        beta=beta_small,
        initial_state=state_small,
        cu_seqlens=cu_small,
        scale=1.0,
    )
    same_shape_output, same_shape_state = gdn.chunk_gated_delta_rule(
        q=q_small,
        k=k_small,
        v=v_small,
        g=gate_small,
        beta=beta_small,
        initial_state=state_small,
        cu_seqlens=cu_small,
        scale=1.0,
    )

    q_small_2, k_small_2, v_small_2, gate_small_2, beta_small_2, state_small_2, cu_small_2 = make_inputs(
        T=24,
        cu=torch.tensor([0, 12, 24], dtype=torch.int64),
    )
    cross_shape_output, cross_shape_state = gdn.chunk_gated_delta_rule(
        q=q_small_2,
        k=k_small_2,
        v=v_small_2,
        g=gate_small_2,
        beta=beta_small_2,
        initial_state=state_small_2,
        cu_seqlens=cu_small_2,
        scale=1.0,
    )
    default_scale_output, default_scale_state = gdn.chunk_gated_delta_rule(
        q=q_small,
        k=k_small,
        v=v_small,
        g=gate_small,
        beta=beta_small,
        initial_state=state_small,
        cu_seqlens=cu_small,
        scale=None,
    )

    q_large, k_large, v_large, gate_large, beta_large, state_large, cu_large = make_inputs(
        T=1025,
        cu=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 1025], dtype=torch.int64),
    )
    large_output, large_output_state = gdn.chunk_gated_delta_rule(
        q=q_large,
        k=k_large,
        v=v_large,
        g=gate_large,
        beta=beta_large,
        initial_state=state_large,
        cu_seqlens=cu_large,
        scale=1.0,
    )

    compile_calls_before_invalid = len(runtime_calls["compile_calls"])
    try:
        gdn.chunk_gated_delta_rule(
            q=q_small,
            k=k_small,
            v=v_small,
            g=gate_small,
            beta=beta_small,
            initial_state=state_small,
            cu_seqlens=cu_small,
            scale=1.0,
            path_name="large",
        )
    except ValueError as exc:
        assert "path_name" in str(exc)
    else:
        raise AssertionError("Expected mismatched path_name override to be rejected")
    assert len(runtime_calls["compile_calls"]) == compile_calls_before_invalid

    bad_q = torch.empty((1, q_small.shape[1], 5, 128), dtype=torch.bfloat16)
    try:
        gdn.chunk_gated_delta_rule(
            q=bad_q,
            k=bad_q,
            v=v_small,
            g=gate_small,
            beta=beta_small,
            initial_state=state_small,
            cu_seqlens=cu_small,
            scale=1.0,
        )
    except ValueError as exc:
        assert "q.shape" in str(exc)
    else:
        raise AssertionError("Expected wrapper to reject out-of-scope q/k shape before compile lookup")
    assert len(runtime_calls["compile_calls"]) == compile_calls_before_invalid

    bad_g = gate_small.to(dtype=torch.bfloat16)
    try:
        gdn.chunk_gated_delta_rule(
            q=q_small,
            k=k_small,
            v=v_small,
            g=bad_g,
            beta=beta_small,
            initial_state=state_small,
            cu_seqlens=cu_small,
            scale=1.0,
        )
    except TypeError as exc:
        assert "g.dtype" in str(exc)
    else:
        raise AssertionError("Expected wrapper to reject g dtype before compile lookup")
    assert len(runtime_calls["compile_calls"]) == compile_calls_before_invalid

    bad_cu = torch.tensor([0, q_small.shape[1] - 1, q_small.shape[1] - 1], dtype=torch.int64)
    try:
        gdn.chunk_gated_delta_rule(
            q=q_small,
            k=k_small,
            v=v_small,
            g=gate_small,
            beta=beta_small,
            initial_state=state_small,
            cu_seqlens=bad_cu,
            scale=1.0,
        )
    except ValueError as exc:
        assert "end at T" in str(exc)
    else:
        raise AssertionError("Expected wrapper to reject mismatched cu_seqlens end before compile lookup")
    assert len(runtime_calls["compile_calls"]) == compile_calls_before_invalid


assert output.shape == v_small.shape
assert output.dtype == torch.bfloat16
assert output_state.shape == state_small.shape
assert output_state.dtype == torch.float32
assert same_shape_output.shape == v_small.shape
assert same_shape_state.shape == state_small.shape
assert cross_shape_output.shape == v_small_2.shape
assert cross_shape_state.shape == state_small_2.shape
assert default_scale_output.shape == v_small.shape
assert default_scale_state.shape == state_small.shape
assert large_output.shape == v_large.shape
assert large_output_state.shape == state_large.shape
assert len(runtime_calls["launches"]) == 5
assert len(runtime_calls["compile_calls"]) == 4
assert runtime_calls["compile_calls"][0]["problem_size"] == (2, 7, 12, 4, 8, 128)
assert runtime_calls["compile_calls"][1]["problem_size"] == (2, 12, 24, 4, 8, 128)
assert runtime_calls["compile_calls"][2]["problem_size"] == (2, 7, 12, 4, 8, 128)
assert runtime_calls["compile_calls"][2]["scale"] == q_small.shape[-1] ** -0.5
assert runtime_calls["compile_calls"][3]["problem_size"] == (9, 1017, 1025, 4, 8, 128)
assert runtime_calls["launches"][2]["problem_size"] == (2, 12, 24, 4, 8, 128)
assert runtime_calls["launches"][3]["scale"] == q_small.shape[-1] ** -0.5
assert gdn._get_compiled_gdn_prefill_kernel.cache_info().currsize == 2
assert tolist_calls <= 4
