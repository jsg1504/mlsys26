"""GDN prefill entry point — hybrid dispatch.

Short sequences: NVRTC register-cached sequential CUDA kernel (fused gates).
Long sequences:  CuTe-DSL chunked kernel (gates fused into gb_warp).
"""
import ctypes
import os
from pathlib import Path

import numpy as np
import torch
import cuda.bindings.driver as drv

try:
    from .nvrtc_loader import compile_and_load
    from .gdn_blackwell import chunk_gated_delta_rule, get_gdn_bundle
except ImportError:
    from nvrtc_loader import compile_and_load
    from gdn_blackwell import chunk_gated_delta_rule, get_gdn_bundle

THRESHOLD = 192
# Hybrid split: per-sequence dispatch when batch has bimodal length distribution.
# SHORT_LEN matches CuTe-DSL chunk_size=128 — seqs at/below this occupy a single
# (under-utilized) tile. MIN_SHORT guards against splitting when the short group
# is too small to justify the split overhead. MIN_T rejects medium-T batches where
# the sequential kernel's wall-clock (~max_short_len × 5.7µs/token) exceeds the
# CuTe-DSL tile reclaim — empirically confirmed on WL 6 (T=4124) and WL 20 (T=3999).
HYBRID_SHORT_LEN = 128
HYBRID_MIN_SHORT = 4
HYBRID_MIN_T = 4200
HYBRID_ENABLED = os.environ.get("GDN_HYBRID_DISABLE", "0") != "1"
_HYB_NOT_CACHED = object()  # sentinel for cache-miss (distinct from cached-None)

_DEFAULT_SCALE = 128 ** -0.5
_kernels = {}
_cached_stream = None
_seq_output_cache = {}
_seq_state_cache = {}
# Hybrid-split caches
_hyb_meta_cache = {}    # (T, num_seqs, lens_tuple) → (short_idx_t, long_idx_t, long_row_idx, cu_seqlens_long, T_long, num_long, max_s_q_long, path_name_long)
_hyb_output_cache = {}  # T → output buffer [T, 8, 128] bf16
_hyb_state_cache = {}   # num_seqs → new_state buffer (same shape as state)

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

# Step 1a: env-gated per-workload seq-length histogram logging (measurement only).
# Enable with GDN_LOG_SEQ_HIST=1 and optionally GDN_LOG_SEQ_HIST_SHORT_LEN=N (default 128).
_SEQ_HIST_ENABLED = os.environ.get("GDN_LOG_SEQ_HIST", "0") == "1"
_SEQ_HIST_SHORT_LEN = int(os.environ.get("GDN_LOG_SEQ_HIST_SHORT_LEN", "128"))
_seq_hist_seen = set()
_seq_hist_host = (ctypes.c_longlong * 101)()
_seq_hist_host_ptr = _addressof(_seq_hist_host)


def _log_seq_hist(T, num_seqs, cu_seqlens, path_name):
    if num_seqs == 1:
        key = (T, 1, T)
        if key in _seq_hist_seen:
            return
        _seq_hist_seen.add(key)
        L = T
        num_short = 1 if L <= _SEQ_HIST_SHORT_LEN else 0
        num_long = 1 - num_short
        short_tok = L if num_short else 0
        long_tok = 0 if num_short else L
        print(
            f"[seq-hist] T={T} num_seqs=1 max_s_q={T} "
            f"num_short={num_short} short_tok={short_tok} "
            f"num_long={num_long} long_tok={long_tok} "
            f"path={path_name}",
            flush=True,
        )
        return

    n = num_seqs + 1
    _cuMemcpyDtoH(_seq_hist_host_ptr, _data_ptr(cu_seqlens), n * 8)
    h = _seq_hist_host
    lens = tuple(h[i + 1] - h[i] for i in range(num_seqs))
    key = (T, num_seqs, lens)
    if key in _seq_hist_seen:
        return
    _seq_hist_seen.add(key)
    max_s_q = max(lens)
    short = [L for L in lens if L <= _SEQ_HIST_SHORT_LEN]
    long_ = [L for L in lens if L > _SEQ_HIST_SHORT_LEN]
    print(
        f"[seq-hist] T={T} num_seqs={num_seqs} max_s_q={max_s_q} "
        f"num_short={len(short)} short_tok={sum(short)} "
        f"num_long={len(long_)} long_tok={sum(long_)} "
        f"path={path_name} lens={lens}",
        flush=True,
    )


def _get_kernel_dir() -> Path:
    return Path(__file__).parent / "nvrtc_kernels"


# Pre-allocated ctypes args for sequential kernel (14 params; p13 = seq_idx_map ptr, 0=nullptr)
class _SeqArgs:
    __slots__ = ('p0','p1','p2','p3','p4','p5','p6','p7','p8','p9','p10','p11','p12','p13','arr')
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
        self.p13 = _c_uint64(0)  # seq_idx_map device pointer (0 = nullptr = identity map)
        self.arr = (_c_void_p * 14)(
            _addressof(self.p0), _addressof(self.p1), _addressof(self.p2),
            _addressof(self.p3), _addressof(self.p4), _addressof(self.p5),
            _addressof(self.p6), _addressof(self.p7), _addressof(self.p8),
            _addressof(self.p9), _addressof(self.p10), _addressof(self.p11),
            _addressof(self.p12), _addressof(self.p13),
        )

_seq_args = _SeqArgs()


def _build_hybrid_meta(T, num_seqs, lens_tuple, cu_seqlens_host, device):
    """Classify seqs into short/long and build device-side metadata tensors.

    Returns None if the split gate fails (num_short < MIN_SHORT or num_long == 0).
    """
    short_idx = [i for i, L in enumerate(lens_tuple) if L <= HYBRID_SHORT_LEN]
    long_idx = [i for i, L in enumerate(lens_tuple) if L > HYBRID_SHORT_LEN]
    if len(short_idx) < HYBRID_MIN_SHORT or len(long_idx) < 1:
        return None

    # Build compact cu_seqlens for the long subset (host-side cumsum).
    cu_long_host = [0]
    for i in long_idx:
        cu_long_host.append(cu_long_host[-1] + lens_tuple[i])
    T_long = cu_long_host[-1]
    num_long = len(long_idx)
    max_s_q_long = max(lens_tuple[i] for i in long_idx)

    # Row indices in the original [T, ...] tensors that belong to long seqs.
    long_rows_np = np.concatenate([
        np.arange(cu_seqlens_host[i], cu_seqlens_host[i + 1], dtype=np.int64)
        for i in long_idx
    ])
    long_row_idx = torch.from_numpy(long_rows_np).to(device, non_blocking=True)

    short_idx_t = torch.tensor(short_idx, dtype=torch.int32, device=device)
    long_idx_t = torch.tensor(long_idx, dtype=torch.int64, device=device)
    cu_seqlens_long = torch.tensor(cu_long_host, dtype=torch.int64, device=device)
    path_name_long = "small" if T_long <= 1024 and num_long <= 8 else "large"
    return (short_idx_t, long_idx_t, long_row_idx, cu_seqlens_long,
            T_long, num_long, max_s_q_long, path_name_long)


def _run_hybrid(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale,
                T, num_seqs, meta, stream):
    """Hybrid dispatch: sequential kernel on short seqs, CuTe-DSL on long seqs."""
    (short_idx_t, long_idx_t, long_row_idx, cu_seqlens_long,
     T_long, num_long, max_s_q_long, path_name_long) = meta
    device = q.device
    num_short = short_idx_t.shape[0]

    # Pre-allocate/cache output and new_state (shared between both kernels).
    output = _hyb_output_cache.get(T)
    if output is None:
        output = torch.empty(T, 8, 128, dtype=torch.bfloat16, device=device)
        _hyb_output_cache[T] = output

    new_state = _hyb_state_cache.get(num_seqs)
    if new_state is None:
        new_state = torch.empty_like(state)
        _hyb_state_cache[num_seqs] = new_state

    # Gather long-subset inputs (q/k/v/a/b are laid out by the original cu_seqlens).
    q_long = q.index_select(0, long_row_idx)
    k_long = k.index_select(0, long_row_idx)
    v_long = v.index_select(0, long_row_idx)
    a_long = a.index_select(0, long_row_idx)
    b_long = b.index_select(0, long_row_idx)
    state_long = state.index_select(0, long_idx_t)

    # Launch sequential kernel on SHORT subset.
    # Grid = num_short * 128 blocks; each block's logical seq is remapped to the
    # real seq index via seq_idx_map = short_idx_t (int32 device ptr).
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
    sa.p10.value = _data_ptr(output)
    sa.p11.value = _data_ptr(new_state)
    sa.p12.value = num_short
    sa.p13.value = _data_ptr(short_idx_t)
    _cuLaunchKernel(
        _kernels["sequential"],
        num_short * 128, 1, 1,
        128, 1, 1,
        0, stream, sa.arr, 0,
    )

    # Launch CuTe-DSL on LONG subset (compact inputs → compact temp output).
    (compiled_gdn, _out_long_4d, output_state_long, problem_size, normalized_scale,
     output_3d_long, output_ptr_long, output_state_ptr_long) = get_gdn_bundle(
        q_long, k_long, v_long, a_long, b_long, A_log, dt_bias, state_long,
        cu_seqlens_long, num_long, T_long, max_s_q_long, path_name_long, scale,
    )
    compiled_gdn(
        _data_ptr(q_long),
        _data_ptr(k_long),
        _data_ptr(v_long),
        output_ptr_long,
        _data_ptr(a_long),
        _data_ptr(b_long),
        _data_ptr(A_log),
        _data_ptr(dt_bias),
        problem_size,
        _data_ptr(state_long),
        output_state_ptr_long,
        normalized_scale,
        cu_seqlens_long,
        stream=stream,
    )

    # Scatter long-subset outputs back into their original row/slot positions.
    output.index_copy_(0, long_row_idx, output_3d_long)
    new_state.index_copy_(0, long_idx_t, output_state_long)
    return output, new_state


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
        if _SEQ_HIST_ENABLED:
            _log_seq_hist(T, num_seqs, cu_seqlens, "sequential")

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
        sa.p13.value = 0  # seq_idx_map = nullptr (identity: local_seq == seq)

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
            lens_tuple = None
        else:
            n = num_seqs + 1
            _cuMemcpyDtoH(_cu_seqlens_host_ptr, _data_ptr(cu_seqlens), n * 8)
            h = _cu_seqlens_host
            lens_tuple = tuple(h[i + 1] - h[i] for i in range(num_seqs))
            max_s_q = max(lens_tuple)

        # Hybrid dispatch: when the batch is multi-seq with bimodal length
        # distribution, split short seqs to sequential kernel and long seqs to
        # CuTe-DSL (on a compacted sub-batch). Falls through to the single-kernel
        # path if the gate fails (num_short < MIN_SHORT or num_long == 0).
        if HYBRID_ENABLED and lens_tuple is not None and T >= HYBRID_MIN_T:
            key = (T, num_seqs, lens_tuple)
            meta = _hyb_meta_cache.get(key, _HYB_NOT_CACHED)
            if meta is _HYB_NOT_CACHED:
                meta = _build_hybrid_meta(T, num_seqs, lens_tuple, h, q.device)
                _hyb_meta_cache[key] = meta  # cache None too (fall-through decisions)
            if meta is not None:
                if _SEQ_HIST_ENABLED:
                    _log_seq_hist(T, num_seqs, cu_seqlens, "hybrid_split")
                return _run_hybrid(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens,
                                   scale, T, num_seqs, meta, stream)

        # Fusion: skip fused_gate launch; CuTe-DSL kernel now computes gates inline.

        path_name = "small" if T <= 1024 and num_seqs <= 8 else "large"

        if _SEQ_HIST_ENABLED:
            _log_seq_hist(T, num_seqs, cu_seqlens, f"cute_dsl_{path_name}")

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
