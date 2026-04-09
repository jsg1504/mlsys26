import sys
from pathlib import Path
import nvidia_cutlass_dsl

for _base in nvidia_cutlass_dsl.__path__:
    _candidate = Path(_base) / "python_packages"
    if _candidate.exists():
        sys.path.insert(0, str(_candidate))

import torch
import triton
import triton.language as tl
from .gdn_blackwell.gdn import chunk_gated_delta_rule


@triton.jit
def _prep_kernel(a_ptr, b_ptr, a_log_ptr, dt_bias_ptr, g_ptr, beta_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    head = offs % 8

    a = tl.load(a_ptr + offs, mask=mask, other=0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0).to(tl.float32)
    a_log = tl.load(a_log_ptr + head, mask=mask, other=0).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + head, mask=mask, other=0).to(tl.float32)

    x = a + dt_bias
    softplus = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    g = -tl.exp(a_log) * softplus
    beta = 1.0 / (1.0 + tl.exp(-b))

    tl.store(g_ptr + offs, g, mask=mask)
    tl.store(beta_ptr + offs, beta, mask=mask)


def _prepare_gate_tensors(A_log, a, dt_bias, b):
    g = torch.empty_like(a, dtype=torch.float32)
    beta = torch.empty_like(b, dtype=torch.float32)
    n = a.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    _prep_kernel[grid](a, b, A_log.float(), dt_bias.float(), g, beta, n, BLOCK=1024, num_warps=4)
    return g, beta


def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    g, beta = _prepare_gate_tensors(A_log, a, dt_bias, b)

    varlen = cu_seqlens is not None and q.dim() == 3
    if varlen:
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)

    output, new_state = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=None,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
    )

    if varlen:
        output = output.squeeze(0)

    return output, new_state
