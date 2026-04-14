from gdn_blackwell import chunk_gated_delta_rule
from prefill_contract import prepare_g_beta, validate_inputs


def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    validate_inputs(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens)
    gate_log, beta = prepare_g_beta(A_log, a, dt_bias, b)
    is_varlen_flat = q.ndim == 3

    if is_varlen_flat:
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)

    output, new_state = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=gate_log,
        beta=beta,
        initial_state=state,
        cu_seqlens=cu_seqlens,
        scale=1.0 if scale is None else scale,
    )

    if is_varlen_flat:
        output = output.squeeze(0)

    return output, new_state
