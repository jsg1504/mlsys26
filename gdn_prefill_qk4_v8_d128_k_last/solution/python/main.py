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
    from .prefill_contract import get_cu_seqlens_metadata
except ImportError:
    from nvrtc_loader import compile_and_load, launch
    from gdn_blackwell import chunk_gated_delta_rule
    from prefill_contract import get_cu_seqlens_metadata

THRESHOLD = 128
_DEFAULT_SCALE = 128 ** -0.5
_kernels = {}
_cached_stream = None
_gate_cache = {}
_seq_output_cache = {}
_seq_state_cache = {}

# Pre-created ctypes constants
_INT32_8 = ctypes.c_int32(8)
_BLOCK_128 = (128, 1, 1)
_BLOCK_256 = (256, 1, 1)

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


def _get_stream():
    global _cached_stream
    if _cached_stream is None:
        _cached_stream = drv.CUstream(torch.cuda.current_stream().cuda_stream)
    return _cached_stream


def _get_gate_tensors(T, device):
    key = (T, device)
    cached = _gate_cache.get(key)
    if cached is not None:
        return cached
    tensors = (
        torch.empty(T, 8, dtype=torch.float32, device=device),
        torch.empty(T, 8, dtype=torch.float32, device=device),
    )
    _gate_cache[key] = tensors
    return tensors


def _get_seq_output(T, device):
    key = (T, device)
    cached = _seq_output_cache.get(key)
    if cached is not None:
        return cached
    t = torch.empty(T, 8, 128, dtype=torch.bfloat16, device=device)
    _seq_output_cache[key] = t
    return t


def _get_seq_state(state):
    key = (tuple(state.shape), state.dtype, state.device)
    cached = _seq_state_cache.get(key)
    if cached is not None:
        return cached
    t = torch.empty_like(state)
    _seq_state_cache[key] = t
    return t


def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    _ensure_kernels()

    T = q.shape[0]
    num_seqs = cu_seqlens.shape[0] - 1
    stream = _get_stream()

    if scale is None or scale == 0.0:
        scale = _DEFAULT_SCALE

    if T <= THRESHOLD:
        output = _get_seq_output(T, q.device)
        new_state = _get_seq_state(state)

        grid = (num_seqs * 8, 1, 1)

        launch(
            _kernels["sequential"], grid, _BLOCK_128,
            [_ptr(q), _ptr(k), _ptr(v), _ptr(state),
             _ptr(A_log), _ptr(a), _ptr(dt_bias), _ptr(b),
             _ptr(cu_seqlens), ctypes.c_float(scale),
             _ptr(output), _ptr(new_state), ctypes.c_int32(num_seqs)],
            shared_mem=0, stream=stream,
        )
        return output, new_state

    else:
        # Use cached metadata (no GPU sync on cache hit)
        meta = get_cu_seqlens_metadata(cu_seqlens)
        max_s_q = meta["max_s_q"]

        # Reuse pre-allocated gate tensors
        gate_log, beta = _get_gate_tensors(T, q.device)

        total_elems = T * 8
        blocks_gate = (total_elems + 255) // 256

        launch(
            _kernels["fused_gate"], (blocks_gate, 1, 1), _BLOCK_256,
            [_ptr(A_log), _ptr(a), _ptr(dt_bias), _ptr(b),
             _ptr(gate_log), _ptr(beta),
             ctypes.c_int32(T), _INT32_8],
            shared_mem=0, stream=stream,
        )

        path_name = "small" if T <= 1024 and num_seqs <= 8 else "large"

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
            precomputed_max_s_q=max_s_q,
        )

        return output.squeeze(0), new_state
