"""GDN prefill entry point — hybrid dispatch.

Short sequences: NVRTC register-cached sequential CUDA kernel (fused gates).
Long sequences:  NVRTC fused gate kernel + CuTe-DSL chunked kernel.
"""
import ctypes
from pathlib import Path

import torch
import cuda.bindings.driver as drv

try:
    from .nvrtc_loader import compile_and_load
    from .gdn_blackwell import chunk_gated_delta_rule, get_gdn_bundle
    from .prefill_contract import get_cu_seqlens_metadata
except ImportError:
    from nvrtc_loader import compile_and_load
    from gdn_blackwell import chunk_gated_delta_rule, get_gdn_bundle
    from prefill_contract import get_cu_seqlens_metadata

THRESHOLD = 128
_DEFAULT_SCALE = 128 ** -0.5
_kernels = {}
_cached_stream = None
_gate_cache = {}
_seq_output_cache = {}
_seq_state_cache = {}

_c_uint64 = ctypes.c_uint64
_c_int32 = ctypes.c_int32
_c_float = ctypes.c_float
_c_void_p = ctypes.c_void_p
_addressof = ctypes.addressof
_cuLaunchKernel = drv.cuLaunchKernel


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


def _get_stream():
    global _cached_stream
    if _cached_stream is None:
        _cached_stream = drv.CUstream(torch.cuda.current_stream().cuda_stream)
    return _cached_stream


# Pre-allocated ctypes args for sequential kernel (13 params)
class _SeqArgs:
    __slots__ = ('p0','p1','p2','p3','p4','p5','p6','p7','p8','p9','p10','p11','p12','arr')
    def __init__(self):
        self.p0 = _c_uint64(0)
        self.p1 = _c_uint64(0)
        self.p2 = _c_uint64(0)
        self.p3 = _c_uint64(0)
        self.p4 = _c_uint64(0)
        self.p5 = _c_uint64(0)
        self.p6 = _c_uint64(0)
        self.p7 = _c_uint64(0)
        self.p8 = _c_uint64(0)
        self.p9 = _c_float(0)
        self.p10 = _c_uint64(0)
        self.p11 = _c_uint64(0)
        self.p12 = _c_int32(0)
        self.arr = (_c_void_p * 13)(
            _addressof(self.p0), _addressof(self.p1), _addressof(self.p2),
            _addressof(self.p3), _addressof(self.p4), _addressof(self.p5),
            _addressof(self.p6), _addressof(self.p7), _addressof(self.p8),
            _addressof(self.p9), _addressof(self.p10), _addressof(self.p11),
            _addressof(self.p12),
        )

_seq_args = _SeqArgs()

# Pre-allocated ctypes args for fused gate kernel (8 params)
class _GateArgs:
    __slots__ = ('p0','p1','p2','p3','p4','p5','p6','p7','arr')
    def __init__(self):
        self.p0 = _c_uint64(0)
        self.p1 = _c_uint64(0)
        self.p2 = _c_uint64(0)
        self.p3 = _c_uint64(0)
        self.p4 = _c_uint64(0)
        self.p5 = _c_uint64(0)
        self.p6 = _c_int32(0)
        self.p7 = _c_int32(8)
        self.arr = (_c_void_p * 8)(
            _addressof(self.p0), _addressof(self.p1), _addressof(self.p2),
            _addressof(self.p3), _addressof(self.p4), _addressof(self.p5),
            _addressof(self.p6), _addressof(self.p7),
        )

_gate_args = _GateArgs()


def _get_gate_tensors(T, device):
    key = (T, device)
    cached = _gate_cache.get(key)
    if cached is not None:
        return cached
    gate_log = torch.empty(T, 8, dtype=torch.float32, device=device)
    beta = torch.empty(T, 8, dtype=torch.float32, device=device)
    # Cache data_ptr values alongside tensors (stable for cached tensors)
    tensors = (gate_log, beta, gate_log.unsqueeze(0), beta.unsqueeze(0),
               gate_log.data_ptr(), beta.data_ptr())
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

        sa = _seq_args
        sa.p0.value = q.data_ptr()
        sa.p1.value = k.data_ptr()
        sa.p2.value = v.data_ptr()
        sa.p3.value = state.data_ptr()
        sa.p4.value = A_log.data_ptr()
        sa.p5.value = a.data_ptr()
        sa.p6.value = dt_bias.data_ptr()
        sa.p7.value = b.data_ptr()
        sa.p8.value = cu_seqlens.data_ptr()
        sa.p9.value = scale
        sa.p10.value = output.data_ptr()
        sa.p11.value = new_state.data_ptr()
        sa.p12.value = num_seqs

        _cuLaunchKernel(
            _kernels["sequential"],
            num_seqs * 8, 1, 1,
            128, 1, 1,
            0, stream, sa.arr, 0,
        )
        return output, new_state

    else:
        meta = get_cu_seqlens_metadata(cu_seqlens)
        max_s_q = meta["max_s_q"]

        gate_log, beta, gate_log_4d, beta_4d, gate_log_ptr, beta_ptr = _get_gate_tensors(T, q.device)

        ga = _gate_args
        ga.p0.value = A_log.data_ptr()
        ga.p1.value = a.data_ptr()
        ga.p2.value = dt_bias.data_ptr()
        ga.p3.value = b.data_ptr()
        ga.p4.value = gate_log_ptr
        ga.p5.value = beta_ptr
        ga.p6.value = T

        _cuLaunchKernel(
            _kernels["fused_gate"],
            (T * 8 + 255) // 256, 1, 1,
            256, 1, 1,
            0, stream, ga.arr, 0,
        )

        path_name = "small" if T <= 1024 and num_seqs <= 8 else "large"

        # Fast path: get cached compiled kernel + output/state tensors directly.
        (compiled_gdn, output, output_state, problem_size, normalized_scale,
         output_3d, output_ptr, output_state_ptr) = get_gdn_bundle(
            q, k, v, gate_log_4d, beta_4d, state, cu_seqlens,
            num_seqs, T, max_s_q, path_name, scale,
        )

        compiled_gdn(
            q.data_ptr(),
            k.data_ptr(),
            v.data_ptr(),
            output_ptr,
            gate_log_ptr,
            beta_ptr,
            problem_size,
            state.data_ptr(),
            output_state_ptr,
            normalized_scale,
            cu_seqlens,
            stream=stream,
        )

        return output_3d, output_state
