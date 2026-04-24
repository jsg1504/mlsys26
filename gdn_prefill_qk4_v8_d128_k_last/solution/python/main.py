"""GDN prefill entry point — hybrid dispatch.

Short sequences: NVRTC register-cached sequential CUDA kernel (fused gates).
Long sequences:  CuTe-DSL chunked kernel (gates fused into gb_warp).
"""
import ctypes
from pathlib import Path

import torch
import cuda.bindings.driver as drv

try:
    from .nvrtc_loader import compile_and_load
    from .gdn_blackwell import chunk_gated_delta_rule, get_gdn_bundle
except ImportError:
    from nvrtc_loader import compile_and_load
    from gdn_blackwell import chunk_gated_delta_rule, get_gdn_bundle

THRESHOLD = 192
_DEFAULT_SCALE = 128 ** -0.5
_kernels = {}
_cached_stream = None
_seq_output_cache = {}
_seq_state_cache = {}

_c_uint64 = ctypes.c_uint64
_c_int32 = ctypes.c_int32
_c_float = ctypes.c_float
_c_void_p = ctypes.c_void_p
_addressof = ctypes.addressof
_cuLaunchKernel = drv.cuLaunchKernel
_cuMemcpyDtoH = drv.cuMemcpyDtoH

# Unbound data_ptr — avoids per-call attribute lookup + bound method creation
_data_ptr = torch.Tensor.data_ptr

# Pre-allocated host buffer for cu_seqlens D2H (max 101 int64 entries)
_cu_seqlens_host = (ctypes.c_longlong * 101)()
_cu_seqlens_host_ptr = _addressof(_cu_seqlens_host)


def _get_kernel_dir() -> Path:
    return Path(__file__).parent / "nvrtc_kernels"


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


def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    global _cached_stream

    if not _kernels:
        kdir = _get_kernel_dir()
        seq_src = (kdir / "sequential_kernel.cu").read_text()
        seq_fns = compile_and_load(seq_src, ["gdn_prefill_sequential"])
        _kernels["sequential"] = seq_fns["gdn_prefill_sequential"]

    T = q.shape[0]
    num_seqs = cu_seqlens.shape[0] - 1

    stream = _cached_stream
    if stream is None:
        _cached_stream = drv.CUstream(torch.cuda.current_stream().cuda_stream)
        stream = _cached_stream

    if scale is None or scale == 0.0:
        scale = _DEFAULT_SCALE

    if T <= THRESHOLD:
        cached_out = _seq_output_cache.get(T)
        if cached_out is not None:
            output, output_ptr = cached_out
        else:
            output = torch.empty(T, 8, 128, dtype=torch.bfloat16, device=q.device)
            output_ptr = _data_ptr(output)
            _seq_output_cache[T] = (output, output_ptr)

        cached_st = _seq_state_cache.get(num_seqs)
        if cached_st is not None:
            new_state, new_state_ptr = cached_st
        else:
            new_state = torch.empty_like(state)
            new_state_ptr = _data_ptr(new_state)
            _seq_state_cache[num_seqs] = (new_state, new_state_ptr)

        sa = _seq_args
        sa.p0.value = _data_ptr(q)
        sa.p1.value = _data_ptr(k)
        sa.p2.value = _data_ptr(v)
        sa.p3.value = _data_ptr(state)
        sa.p4.value = _data_ptr(A_log)
        sa.p5.value = _data_ptr(a)
        sa.p6.value = _data_ptr(dt_bias)
        sa.p7.value = _data_ptr(b)
        sa.p8.value = _data_ptr(cu_seqlens)
        sa.p9.value = scale
        sa.p10.value = output_ptr
        sa.p11.value = new_state_ptr
        sa.p12.value = num_seqs

        _cuLaunchKernel(
            _kernels["sequential"],
            num_seqs * 128, 1, 1,
            128, 1, 1,
            0, stream, sa.arr, 0,
        )
        return output, new_state

    else:
        if num_seqs == 1:
            max_s_q = T
        else:
            n = num_seqs + 1
            _cuMemcpyDtoH(_cu_seqlens_host_ptr, _data_ptr(cu_seqlens), n * 8)
            h = _cu_seqlens_host
            max_s_q = max(h[i+1] - h[i] for i in range(num_seqs))

        # Fusion: skip fused_gate launch; CuTe-DSL kernel now computes gates inline.

        path_name = "small" if T <= 1024 and num_seqs <= 8 else "large"

        (compiled_gdn, output, output_state, problem_size, normalized_scale,
         output_3d, output_ptr, output_state_ptr) = get_gdn_bundle(
            q, k, v, a, b, A_log, dt_bias, state, cu_seqlens,
            num_seqs, T, max_s_q, path_name, scale,
        )

        compiled_gdn(
            _data_ptr(q),
            _data_ptr(k),
            _data_ptr(v),
            output_ptr,
            _data_ptr(a),
            _data_ptr(b),
            _data_ptr(A_log),
            _data_ptr(dt_bias),
            problem_size,
            _data_ptr(state),
            output_state_ptr,
            normalized_scale,
            cu_seqlens,
            stream=stream,
        )

        return output_3d, output_state
