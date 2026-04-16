import torch
import torch.nn.functional as F

try:
    from .gdn_blackwell import chunk_gated_delta_rule
    from .gdn_blackwell.dispatch import choose_path
except ImportError:
    from gdn_blackwell import chunk_gated_delta_rule
    from gdn_blackwell.dispatch import choose_path


def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    # Gate computation — minimal kernel launches
    # a: [T, 8] bf16, dt_bias: [8] f32, A_log: [8] f32, b: [T, 8] bf16
    gate_log = -(A_log.exp() * F.softplus(a.float() + dt_bias))
    beta = b.float().sigmoid_()

    # cu_seqlens metadata — avoid full CPU sync
    num_seqs = cu_seqlens.shape[0] - 1
    sum_s_q = q.shape[0]

    path_name = choose_path(total_seq_len=sum_s_q, num_seqs=num_seqs)

    output, new_state = chunk_gated_delta_rule(
        q=q.unsqueeze(0),
        k=k.unsqueeze(0),
        v=v.unsqueeze(0),
        g=gate_log.unsqueeze(0),
        beta=beta.unsqueeze(0),
        initial_state=state,
        cu_seqlens=cu_seqlens,
        scale=scale,
        path_name=path_name,
    )

    return output.squeeze(0), new_state
