import torch

from gdn_blackwell import chunk_gated_delta_rule
from prefill_contract import prepare_g_beta, validate_inputs


def _prepare_runtime_inputs(q, k, v, g, beta):
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("main.run currently accepts only flat varlen inputs for q, k, and v.")
    return {
        "q": q.unsqueeze(0),
        "k": k.unsqueeze(0),
        "v": v.unsqueeze(0),
        "g": g.unsqueeze(0),
        "beta": beta.unsqueeze(0),
    }


def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    if isinstance(q, torch.Tensor) and isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor):
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError("main.run currently accepts only flat varlen inputs for q, k, and v.")
    validate_inputs(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens)
    gate_log, beta = prepare_g_beta(A_log, a, dt_bias, b)
    runtime_inputs = _prepare_runtime_inputs(q, k, v, gate_log, beta)

    output, new_state = chunk_gated_delta_rule(
        **runtime_inputs,
        initial_state=state,
        cu_seqlens=cu_seqlens,
        scale=1.0 if scale is None else scale,
    )

    expected_output_shape = runtime_inputs["v"].shape
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"chunk_gated_delta_rule output must be a torch.Tensor, got {type(output).__name__}")
    if output.shape != expected_output_shape:
        raise ValueError(
            f"chunk_gated_delta_rule output.shape must be {tuple(expected_output_shape)}, got {tuple(output.shape)}"
        )
    if output.dtype != runtime_inputs["v"].dtype:
        raise ValueError(
            f"chunk_gated_delta_rule output.dtype must be {runtime_inputs['v'].dtype}, got {output.dtype}"
        )
    if output.device != runtime_inputs["v"].device:
        raise ValueError(
            f"chunk_gated_delta_rule output.device must be {runtime_inputs['v'].device}, got {output.device}"
        )
    if not isinstance(new_state, torch.Tensor):
        raise TypeError(
            f"chunk_gated_delta_rule new_state must be a torch.Tensor, got {type(new_state).__name__}"
        )
    if new_state.shape != state.shape:
        raise ValueError(
            f"chunk_gated_delta_rule new_state.shape must be {tuple(state.shape)}, got {tuple(new_state.shape)}"
        )
    if new_state.dtype != state.dtype:
        raise ValueError(
            f"chunk_gated_delta_rule new_state.dtype must be {state.dtype}, got {new_state.dtype}"
        )
    if new_state.device != state.device:
        raise ValueError(
            f"chunk_gated_delta_rule new_state.device must be {state.device}, got {new_state.device}"
        )

    return output.squeeze(0), new_state
