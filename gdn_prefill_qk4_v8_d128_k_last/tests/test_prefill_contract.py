import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "solution" / "python"))
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
