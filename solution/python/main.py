"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import functools
import math
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack

TILE_V = 8
TILE_K = 128
NUM_STAGES = 2
NUM_THREADS = 128
NUM_BLOCKS_PER_STATE = 8


@cute.kernel
def gdn_decode_kernel_small_batch_pretranspose(
    tiled_copy_load: cute.TiledCopy,
    h0_source: cute.Tensor,
    smem_layout_staged: cute.Layout,
    vec_size: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b: cute.Tensor,
    o: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
):
    tidx, _, _ = cute.arch.thread_idx()
    lane_id = tidx % 32
    block_idx, _, _ = cute.arch.block_idx()
    batch_idx = block_idx // NUM_BLOCKS_PER_STATE
    batch_inner = block_idx % NUM_BLOCKS_PER_STATE
    num_v_tiles_per_block = num_v_tiles // NUM_BLOCKS_PER_STATE
    i_n = batch_idx // HV
    i_hv = batch_idx % HV
    i_h = i_hv // (HV // H)
    i_t = 0

    smem = cutlass.utils.SmemAllocator()
    s_data = smem.allocate_tensor(cutlass.Float32, smem_layout_staged, 128)
    s_output = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((V,)), 16)
    s_v = smem.allocate_tensor(cutlass.Float32, cute.make_layout((V,)), 16)

    r_k = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_q = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_h = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_q_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_k_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_v_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )

    k_start = lane_id * vec_size
    r_A_log = cutlass.Float32(A_log[i_hv])
    r_a = cutlass.Float32(a[i_n, i_t, i_hv])
    r_dt_bias = cutlass.Float32(dt_bias[i_hv])
    r_b = cutlass.Float32(b[i_n, i_t, i_hv])

    cute.arch.barrier()

    g_src_batch = h0_source[(batch_idx, None, None)]
    g_dst = cute.local_tile(h0_source, (1, TILE_V, TILE_K), (batch_idx, None, 0))
    g_src = cute.local_tile(g_src_batch, (TILE_V, TILE_K), (None, 0))
    thr_copy_load = tiled_copy_load.get_slice(tidx)

    start_v_tiles = batch_inner * num_v_tiles_per_block
    prefetch_count = cutlass.min(NUM_STAGES - 1, num_v_tiles_per_block)
    for v_tiles in range(start_v_tiles, start_v_tiles + prefetch_count):
        stage = (v_tiles - start_v_tiles) % NUM_STAGES
        g_src_tile = g_src[(None, None, v_tiles)]
        s_data_stage = s_data[(None, None, stage)]
        thr_g_src = thr_copy_load.partition_S(g_src_tile)
        thr_s_data = thr_copy_load.partition_D(s_data_stage)
        cute.copy(tiled_copy_load, thr_g_src, thr_s_data)
        cute.arch.cp_async_commit_group()

    q_tile = cute.local_tile(q, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_id))
    k_tile = cute.local_tile(k, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_id))
    cute.autovec_copy(q_tile, r_q_bf16)
    cute.autovec_copy(k_tile, r_k_bf16)
    for i in cutlass.range_constexpr(vec_size):
        r_q[i] = cutlass.Float32(r_q_bf16[i])
        r_k[i] = cutlass.Float32(r_k_bf16[i])

    v_tile = cute.local_tile(v, (1, 1, 1, vec_size), (i_n, i_t, i_hv, lane_id))
    cute.autovec_copy(v_tile, r_v_bf16)
    for i in cutlass.range_constexpr(vec_size):
        s_v[k_start + i] = cutlass.Float32(r_v_bf16[i])

    cute.arch.barrier()

    r_g = 0.0
    r_beta = 0.0
    if lane_id == 0:
        x = r_a + r_dt_bias
        beta_x = softplus_beta * x
        softplus_x = 0.0
        if beta_x <= softplus_threshold:
            exp_beta_x = cute.exp(beta_x, fastmath=True)
            log_input = cutlass.Float32(1.0 + exp_beta_x)
            log_result = cutlass.Float32(cute.log(log_input, fastmath=True))
            softplus_x = cutlass.Float32(
                (cutlass.Float32(1.0) / softplus_beta) * log_result
            )
        else:
            softplus_x = x
        r_g_value = -cute.exp(r_A_log, fastmath=True) * softplus_x
        r_beta = 1.0 / (1.0 + cute.exp(-r_b, fastmath=True))
        r_g = cute.exp(r_g_value, fastmath=True)

    r_g = cute.arch.shuffle_sync(r_g, 0)
    r_beta = cute.arch.shuffle_sync(r_beta, 0)

    if use_qk_l2norm:
        sum_q = 0.0
        sum_k = 0.0
        for i in cutlass.range_constexpr(vec_size):
            sum_q += r_q[i] * r_q[i]
            sum_k += r_k[i] * r_k[i]
        for offset in [16, 8, 4, 2, 1]:
            sum_q += cute.arch.shuffle_sync_bfly(
                sum_q, offset=offset, mask=-1, mask_and_clamp=31
            )
            sum_k += cute.arch.shuffle_sync_bfly(
                sum_k, offset=offset, mask=-1, mask_and_clamp=31
            )
        inv_norm_q = cute.rsqrt(sum_q + 1e-6, fastmath=True)
        inv_norm_k = cute.rsqrt(sum_k + 1e-6, fastmath=True)
        for i in cutlass.range_constexpr(vec_size):
            r_q[i] = r_q[i] * inv_norm_q
            r_k[i] = r_k[i] * inv_norm_k

    for i in cutlass.range_constexpr(vec_size):
        r_q[i] = r_q[i] * scale

    end_v_tiles = start_v_tiles + num_v_tiles_per_block
    for v_tiles in range(start_v_tiles, end_v_tiles):
        stage = (v_tiles - start_v_tiles) % NUM_STAGES
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        next_v_tiles = v_tiles + prefetch_count
        if next_v_tiles < end_v_tiles:
            next_stage = (next_v_tiles - start_v_tiles) % NUM_STAGES
            g_src_next = g_src[(None, None, next_v_tiles)]
            s_data_next = s_data[(None, None, next_stage)]
            thr_g_src = thr_copy_load.partition_S(g_src_next)
            thr_s_data = thr_copy_load.partition_D(s_data_next)
            cute.copy(tiled_copy_load, thr_g_src, thr_s_data)
            cute.arch.cp_async_commit_group()

        for row in cutlass.range_constexpr(0, TILE_V, 4):
            row_offset = tidx // 32
            sum_hk = 0.0
            s_data_tile = cute.local_tile(
                s_data, (1, vec_size, 1), (row + row_offset, lane_id, stage)
            )
            cute.autovec_copy(s_data_tile, r_h)
            for i in cutlass.range_constexpr(vec_size):
                r_h[i] = r_h[i] * r_g
                sum_hk += r_h[i] * r_k[i]
            for offset in [16, 8, 4, 2, 1]:
                sum_hk += cute.arch.shuffle_sync_bfly(
                    sum_hk, offset=offset, mask=-1, mask_and_clamp=31
                )

            v_new = s_v[v_tiles * TILE_V + row + row_offset] - sum_hk
            v_new = v_new * r_beta

            sum_hq = 0.0
            for i in cutlass.range_constexpr(vec_size):
                r_h[i] += r_k[i] * v_new
                sum_hq += r_h[i] * r_q[i]

            g_dst_tile = cute.local_tile(
                g_dst, (1, 1, vec_size, 1), (0, row + row_offset, lane_id, v_tiles)
            )
            cute.autovec_copy(r_h, g_dst_tile)

            for offset in [16, 8, 4, 2, 1]:
                sum_hq += cute.arch.shuffle_sync_bfly(
                    sum_hq, offset=offset, mask=-1, mask_and_clamp=31
                )

            o_idx = v_tiles * TILE_V + row + row_offset
            if lane_id == 0 and o_idx < V:
                s_output[o_idx] = cutlass.BFloat16(sum_hq)

    cute.arch.barrier()
    if tidx >= start_v_tiles * TILE_V and tidx < end_v_tiles * TILE_V:
        o[(i_n, i_t, i_hv, tidx)] = s_output[tidx]


@cute.jit
def run_gdn_decode_kernel_small_batch_pretranspose(
    h0_source: cute.Tensor,
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b: cute.Tensor,
    o: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    batch_size, v_dim, k_dim = (
        h0_source.layout.shape[0],
        h0_source.layout.shape[1],
        h0_source.layout.shape[2],
    )

    copy_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.Float32,
        num_bits_per_copy=128,
    )
    thread_layout = cute.make_layout((4, 32), stride=(32, 1))
    val_layout = cute.make_layout((1, 4))
    tiled_copy_load = cute.make_tiled_copy_tv(copy_atom, thread_layout, val_layout)
    num_v_tiles = cute.ceil_div(v_dim, TILE_V)
    vec_size = TILE_K // 32
    smem_layout_staged = cute.make_layout(
        (TILE_V, TILE_K, NUM_STAGES), stride=(TILE_K, 1, TILE_V * TILE_K)
    )
    smem_bytes = 4 * TILE_V * TILE_K * NUM_STAGES + 4 * k_dim + 2 * v_dim + 32

    gdn_decode_kernel_small_batch_pretranspose(
        tiled_copy_load,
        h0_source,
        smem_layout_staged,
        vec_size,
        num_v_tiles,
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        o,
        softplus_beta,
        softplus_threshold,
        scale,
        HV,
        B,
        T,
        H,
        K,
        V,
        use_qk_l2norm,
    ).launch(
        grid=(batch_size * NUM_BLOCKS_PER_STATE, 1, 1),
        block=[NUM_THREADS, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


@functools.cache
def _get_compiled_decode_kernel(
    B: int,
    T: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    dtype: torch.dtype,
    scale: float,
    use_qk_l2norm: bool,
):
    return {}


def gated_delta_rule_decode_pretranspose(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
    use_qk_l2norm: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K = q.shape
    _, _, HV, V = v.shape

    assert state.shape == (B, HV, V, K), (
        f"Expected state shape [B={B}, HV={HV}, V={V}, K={K}], got {state.shape}"
    )
    assert T == 1, f"Decode only supports T=1, got T={T}"
    assert state.dtype == torch.float32, f"state must be float32, got {state.dtype}"
    assert K >= 128, f"K must be at least 128, got K={K}"
    assert V >= 128, f"V must be at least 128, got V={V}"
    assert V % TILE_V == 0, f"V must be divisible by {TILE_V}, got V={V}"
    assert q.dtype in (torch.float16, torch.bfloat16), (
        f"q must be float16/bfloat16, got {q.dtype}"
    )
    assert A_log.dtype == torch.float32, f"A_log must be float32, got {A_log.dtype}"

    if scale is None:
        scale = K**-0.5

    output_provided = output is not None
    target_dtype = output.dtype if output_provided else q.dtype
    if output is None:
        output = torch.zeros((B, T, HV, V), dtype=torch.bfloat16, device=q.device)

    h0_source = state.reshape(B * HV, V, K)
    cache_key = (B, T, H, HV, K, V, q.dtype, scale, use_qk_l2norm)
    cache = _get_compiled_decode_kernel(*cache_key)

    if "compiled" not in cache:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        h0_source_tensor = from_dlpack(h0_source, assumed_align=16)
        A_log_tensor = from_dlpack(A_log, assumed_align=16)
        a_tensor = from_dlpack(a, assumed_align=16)
        dt_bias_tensor = from_dlpack(dt_bias, assumed_align=16)
        q_tensor = from_dlpack(q, assumed_align=16)
        k_tensor = from_dlpack(k, assumed_align=16)
        v_tensor = from_dlpack(v, assumed_align=16)
        b_tensor = from_dlpack(b, assumed_align=16)
        o_tensor = from_dlpack(output, assumed_align=16)

        compiled = cute.compile(
            run_gdn_decode_kernel_small_batch_pretranspose,
            h0_source_tensor,
            A_log_tensor,
            a_tensor,
            dt_bias_tensor,
            q_tensor,
            k_tensor,
            v_tensor,
            b_tensor,
            o_tensor,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            scale=scale,
            HV=HV,
            B=B,
            T=T,
            H=H,
            K=K,
            V=V,
            use_qk_l2norm=use_qk_l2norm,
            stream=stream,
            options="--enable-tvm-ffi",
        )
        cache["compiled"] = compiled

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    cache["compiled"](h0_source, A_log, a, dt_bias, q, k, v, b, output, stream)

    if not state.is_contiguous():
        state.copy_(h0_source.reshape(B, HV, V, K))
    if output.dtype != target_dtype:
        output = output.to(target_dtype)
    return output, state


def run(q, k, v, state, A_log, a, dt_bias, b, scale):
    if isinstance(scale, torch.Tensor):
        scale = float(scale.item())
    else:
        scale = float(scale)
    if scale == 0.0:
        scale = 1.0 / math.sqrt(q.shape[-1])

    batch_size, seq_len, num_v_heads, head_size = v.shape
    output = torch.empty(
        batch_size,
        seq_len,
        num_v_heads,
        head_size,
        dtype=q.dtype,
        device=q.device,
    )
    return gated_delta_rule_decode_pretranspose(
        q=q,
        k=k,
        v=v,
        state=state,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        scale=scale,
        output=output,
        use_qk_l2norm=False,
    )
