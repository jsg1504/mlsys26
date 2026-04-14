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

    return output.squeeze(0), new_state
