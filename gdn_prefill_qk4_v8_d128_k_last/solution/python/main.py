"""GDN prefill entry point — hybrid dispatch.

Short sequences: NVRTC register-cached sequential CUDA kernel (fused gates).
Long sequences:  NVRTC fused gate kernel + CuTe-DSL chunked kernel.
"""
import ctypes
from pathlib import Path

import torch
import cuda.bindings.driver as drv

try:
    from .nvrtc_loader import compile_and_load, launch
    from .gdn_blackwell import chunk_gated_delta_rule
    from .gdn_blackwell.dispatch import choose_path
except ImportError:
    from nvrtc_loader import compile_and_load, launch
    from gdn_blackwell import chunk_gated_delta_rule
    from gdn_blackwell.dispatch import choose_path

THRESHOLD = 128
_kernels = {}

def _get_kernel_dir() -> Path:
    return Path(__file__).parent / "nvrtc_kernels"

def _ensure_kernels():
    if _kernels:
        return
    kdir = _get_kernel_dir()

    seq_src = (kdir / "sequential_kernel.cu").read_text()
    seq_fns = compile_and_load(seq_src, ["gdn_prefill_sequential"])
    _kernels["sequential"] = seq_fns["gdn_prefill_sequential"]

    gate_src = (kdir / "fused_gate_kernel.cu").read_text()
    gate_fns = compile_and_load(gate_src, ["fused_gate_kernel"])
    _kernels["fused_gate"] = gate_fns["fused_gate_kernel"]


def _ptr(t: torch.Tensor):
    return ctypes.c_uint64(t.data_ptr())


def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    _ensure_kernels()

    T = q.shape[0]
    num_seqs = cu_seqlens.shape[0] - 1
    stream = drv.CUstream(torch.cuda.current_stream().cuda_stream)

    if scale is None or scale == 0.0:
        scale = 128 ** -0.5

    if T <= THRESHOLD:
        output = torch.empty(T, 8, 128, dtype=torch.bfloat16, device=q.device)
        new_state = torch.empty_like(state)

        grid = (num_seqs * 8, 1, 1)
        block = (128, 1, 1)

        launch(
            _kernels["sequential"], grid, block,
            [_ptr(q), _ptr(k), _ptr(v), _ptr(state),
             _ptr(A_log), _ptr(a), _ptr(dt_bias), _ptr(b),
             _ptr(cu_seqlens), ctypes.c_float(scale),
             _ptr(output), _ptr(new_state), ctypes.c_int32(num_seqs)],
            shared_mem=0, stream=stream,
        )
        return output, new_state

    else:
        gate_log = torch.empty(T, 8, dtype=torch.float32, device=q.device)
        beta = torch.empty(T, 8, dtype=torch.float32, device=q.device)

        total_elems = T * 8
        threads = 256
        blocks_gate = (total_elems + threads - 1) // threads

        launch(
            _kernels["fused_gate"], (blocks_gate, 1, 1), (threads, 1, 1),
            [_ptr(A_log), _ptr(a), _ptr(dt_bias), _ptr(b),
             _ptr(gate_log), _ptr(beta),
             ctypes.c_int32(T), ctypes.c_int32(8)],
            shared_mem=0, stream=stream,
        )

        path_name = choose_path(total_seq_len=T, num_seqs=num_seqs)

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
