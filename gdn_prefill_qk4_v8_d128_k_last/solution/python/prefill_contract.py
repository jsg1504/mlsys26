import torch
import torch.nn.functional as F


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
