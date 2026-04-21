"""
Probe the compiled_gdn TVM FFI object on Modal B200:
1. Compare call overhead with same vs cloned tensors
2. Inspect _tvm_ffi_function and jit_module for CUfunction extraction
3. Check if __cubin__ becomes non-None after first call
4. Full kernel_info dump

Usage:
    modal run scripts/probe_compiled_gdn.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("flashinfer-probe-compiled-gdn")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.2.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .pip_install(
        "flashinfer-bench",
        "torch",
        "triton",
        "numpy",
        "nvidia-cutlass-dsl",
        "cuda-python",
    )
)


@app.function(image=image, gpu="B200:1", timeout=600, volumes={TRACE_SET_PATH: trace_volume})
def probe_compiled(solution_json: str, definition_name: str) -> str:
    import json
    import torch
    import time
    import traceback

    from flashinfer_bench import Solution, TraceSet
    from flashinfer_bench.compile.registry import BuilderRegistry
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
    from flashinfer_bench.logging import configure_logging
    import logging
    configure_logging(level=logging.WARNING)

    solution = Solution.model_validate_json(solution_json)
    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    definition = trace_set.definitions[definition_name]
    workloads = trace_set.workloads.get(definition_name, [])

    registry = BuilderRegistry.get_instance()
    runnable = registry.build(definition, solution)

    out = []

    # Pick workload 40 (CuTe-DSL, medium)
    idx = 40
    trace = workloads[idx]
    workload = trace.workload
    safe_tensors = load_safetensors(definition, workload, trace_set.root) if any(
        s.type == "safetensors" for s in workload.inputs.values()
    ) else None
    inputs = gen_inputs(definition, workload, device="cuda", safe_tensors=safe_tensors)
    q = inputs[0]
    T_val = q.shape[0]
    cu_seqlens = inputs[8]
    num_seqs = cu_seqlens.shape[0] - 1
    out.append(f"Workload {idx}: T={T_val}, num_seqs={num_seqs}")

    # Warmup
    for _ in range(5):
        cloned = [x.clone() if isinstance(x, torch.Tensor) else x for x in inputs]
        runnable(*cloned)
    torch.cuda.synchronize()

    # Find modules
    main_mod = gdn_mod = None
    for name, mod in sys.modules.items():
        if hasattr(mod, '_kernels') and hasattr(mod, 'run'):
            main_mod = mod
        if hasattr(mod, '_bundle_cache') and hasattr(mod, 'get_gdn_bundle'):
            gdn_mod = mod

    if not main_mod or not gdn_mod:
        return f"ERROR: main_mod={main_mod is not None}, gdn_mod={gdn_mod is not None}"

    bundle_cache = gdn_mod._bundle_cache
    key = next(iter(bundle_cache))
    bundle = bundle_cache[key]
    compiled_gdn = bundle[0]

    # ====================================================
    # PROBE 1: kernel_info full dump
    # ====================================================
    out.append("\n=== kernel_info ===")
    ki = compiled_gdn.kernel_info
    if ki:
        for k2, v in ki.items():
            out.append(f"  kernel name: {k2[:120]}")
            if isinstance(v, dict):
                for k3, v3 in v.items():
                    out.append(f"    {k3}: {repr(v3)[:200]}")
            else:
                out.append(f"    value: {repr(v)[:200]}")

    # ====================================================
    # PROBE 2: __cubin__ and __ptx__ after compilation
    # ====================================================
    out.append("\n=== CUBIN/PTX after warmup ===")
    cubin = compiled_gdn.__cubin__
    ptx = compiled_gdn.__ptx__
    out.append(f"  __cubin__: {type(cubin)} len={len(cubin) if cubin else 'None'}")
    out.append(f"  __ptx__: {type(ptx)} len={len(ptx) if ptx else 'None'}")
    if ptx:
        ptx_str = ptx.decode() if isinstance(ptx, bytes) else str(ptx)
        # Find .visible .entry and .param declarations
        lines = ptx_str.split('\n')
        for i, line in enumerate(lines):
            if '.visible .entry' in line or '.param' in line and '.entry' in '\n'.join(lines[max(0,i-5):i+1]):
                out.append(f"  PTX[{i}]: {line.strip()}")
                if i > 0 and '.visible .entry' in lines[i-1]:
                    out.append(f"  PTX[{i-1}]: {lines[i-1].strip()}")

    # ====================================================
    # PROBE 3: _tvm_ffi_function and jit_module
    # ====================================================
    out.append("\n=== _tvm_ffi_function ===")
    try:
        tvm_fn = compiled_gdn._tvm_ffi_function
        out.append(f"  type: {type(tvm_fn)}")
        out.append(f"  dir: {[a for a in dir(tvm_fn) if not a.startswith('__')][:20]}")
        if hasattr(tvm_fn, 'handle'):
            out.append(f"  handle: {tvm_fn.handle}")
        for attr in ['release_gil', 'is_global', 'num_args']:
            if hasattr(tvm_fn, attr):
                out.append(f"  {attr}: {getattr(tvm_fn, attr)}")
    except Exception as e:
        out.append(f"  error: {e}")

    out.append("\n=== jit_module ===")
    try:
        jm = compiled_gdn.jit_module
        out.append(f"  type: {type(jm)}")
        out.append(f"  dir: {[a for a in dir(jm) if not a.startswith('__')][:30]}")
        for attr in ['module', 'device_module', 'cu_module', 'lib', 'functions', 'get_function']:
            if hasattr(jm, attr):
                val = getattr(jm, attr)
                out.append(f"  {attr}: type={type(val)} repr={repr(val)[:200]}")
    except Exception as e:
        out.append(f"  error: {e}")

    out.append("\n=== engine ===")
    try:
        eng = compiled_gdn.engine
        out.append(f"  type: {type(eng)}")
        out.append(f"  dir: {[a for a in dir(eng) if not a.startswith('__')][:30]}")
    except Exception as e:
        out.append(f"  error: {e}")

    out.append("\n=== artifacts ===")
    try:
        arts = compiled_gdn.artifacts
        out.append(f"  type: {type(arts)}")
        if isinstance(arts, dict):
            for ak, av in arts.items():
                out.append(f"  key={ak}: type={type(av)} repr={repr(av)[:150]}")
        elif isinstance(arts, (list, tuple)):
            for i, a in enumerate(arts[:5]):
                out.append(f"  [{i}]: type={type(a)} repr={repr(a)[:150]}")
        else:
            out.append(f"  repr: {repr(arts)[:300]}")
    except Exception as e:
        out.append(f"  error: {e}")

    out.append("\n=== capi_func ===")
    try:
        cf = compiled_gdn.capi_func
        out.append(f"  type: {type(cf)}")
        out.append(f"  repr: {repr(cf)[:200]}")
        out.append(f"  dir: {[a for a in dir(cf) if not a.startswith('__')][:20]}")
    except Exception as e:
        out.append(f"  error: {e}")

    # ====================================================
    # PROBE 4: Call overhead - SAME tensors vs CLONED tensors
    # ====================================================
    out.append("\n=== Call overhead comparison ===")

    stream = main_mod._get_stream()
    gate_log, beta, gate_log_4d, beta_4d, gate_log_ptr, beta_ptr = main_mod._get_gate_tensors(T_val, q.device)
    path_name = "small" if T_val <= 1024 and num_seqs <= 8 else "large"

    get_bundle = gdn_mod.get_gdn_bundle
    state = inputs[3]

    # Get bundle for this workload
    if num_seqs == 1:
        max_s_q = T_val
    else:
        h = cu_seqlens.tolist()
        max_s_q = max(h[i+1] - h[i] for i in range(num_seqs))

    result_b = get_bundle(
        q, inputs[1], inputs[2], gate_log_4d, beta_4d, state, cu_seqlens,
        num_seqs, T_val, max_s_q, path_name, main_mod._DEFAULT_SCALE,
    )
    cgdn = result_b[0]
    output_ptr = result_b[6]
    output_state_ptr = result_b[7]
    problem_size = result_b[3]
    normalized_scale = result_b[4]

    # Test A: Same tensors (data_ptr constant)
    torch.cuda.synchronize()
    times_same = []
    for _ in range(200):
        t0 = time.perf_counter_ns()
        cgdn(
            q.data_ptr(), inputs[1].data_ptr(), inputs[2].data_ptr(),
            output_ptr, gate_log_ptr, beta_ptr,
            problem_size, state.data_ptr(), output_state_ptr,
            normalized_scale, cu_seqlens, stream=stream,
        )
        t1 = time.perf_counter_ns()
        times_same.append((t1 - t0) / 1000)
    torch.cuda.synchronize()

    # Test B: Cloned tensors (data_ptr changes each call, like benchmark)
    times_cloned = []
    for _ in range(200):
        q_c = q.clone()
        k_c = inputs[1].clone()
        v_c = inputs[2].clone()
        s_c = state.clone()
        cs_c = cu_seqlens.clone()
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        cgdn(
            q_c.data_ptr(), k_c.data_ptr(), v_c.data_ptr(),
            output_ptr, gate_log_ptr, beta_ptr,
            problem_size, s_c.data_ptr(), output_state_ptr,
            normalized_scale, cs_c, stream=stream,
        )
        t1 = time.perf_counter_ns()
        times_cloned.append((t1 - t0) / 1000)
    torch.cuda.synchronize()

    # Test C: Same tensors, positional stream (no kwargs)
    times_pos = []
    # Determine if compiled_gdn has _kwargs_wrapper attribute
    # and if there's a way to call without kwargs
    has_kw_wrapper = hasattr(cgdn, '_kwargs_wrapper')
    out.append(f"  has _kwargs_wrapper: {has_kw_wrapper}")

    # Try calling the underlying _tvm_ffi_function directly
    times_direct = []
    try:
        tvm_fn = cgdn._tvm_ffi_function
        if tvm_fn is not None:
            torch.cuda.synchronize()
            for _ in range(200):
                t0 = time.perf_counter_ns()
                tvm_fn(
                    q.data_ptr(), inputs[1].data_ptr(), inputs[2].data_ptr(),
                    output_ptr, gate_log_ptr, beta_ptr,
                    problem_size, state.data_ptr(), output_state_ptr,
                    normalized_scale, cu_seqlens, stream,
                )
                t1 = time.perf_counter_ns()
                times_direct.append((t1 - t0) / 1000)
            torch.cuda.synchronize()
    except Exception as e:
        out.append(f"  _tvm_ffi_function direct call error: {e}")
        traceback.print_exc()

    # Report results
    def stats(times):
        s = sorted(times)
        return f"avg={sum(s)/len(s):.1f}us  p50={s[len(s)//2]:.1f}us  p95={s[int(len(s)*0.95)]:.1f}us  min={s[0]:.1f}us  max={s[-1]:.1f}us"

    out.append(f"\n  Same tensors (200 calls): {stats(times_same)}")
    out.append(f"    first 5: {[f'{t:.1f}' for t in times_same[:5]]}")
    out.append(f"  Cloned tensors (200 calls): {stats(times_cloned)}")
    out.append(f"    first 5: {[f'{t:.1f}' for t in times_cloned[:5]]}")
    if times_direct:
        out.append(f"  _tvm_ffi_function direct (200 calls): {stats(times_direct)}")
        out.append(f"    first 5: {[f'{t:.1f}' for t in times_direct[:5]]}")

    # ====================================================
    # PROBE 5: Full run() timing breakdown with cloned tensors
    # ====================================================
    out.append("\n=== Full run() breakdown with cloned tensors ===")
    from cuda.bindings import driver as drv

    # Measure individual steps
    _cuLaunchKernel = drv.cuLaunchKernel
    gate_kernel = main_mod._kernels.get("fused_gate")
    ga = main_mod._gate_args

    torch.cuda.synchronize()
    step_times = {
        'max_s_q': [], 'gate_setup': [], 'gate_launch': [],
        'bundle_get': [], 'compiled_call': [], 'total': []
    }

    for _ in range(100):
        q_c = q.clone()
        k_c = inputs[1].clone()
        v_c = inputs[2].clone()
        s_c = state.clone()
        cs_c = cu_seqlens.clone()
        A_log = inputs[4]
        a_in = inputs[5]
        dt_bias = inputs[6]
        b_in = inputs[7]
        torch.cuda.synchronize()

        t_total_start = time.perf_counter_ns()

        # Step 1: max_s_q
        t0 = time.perf_counter_ns()
        if num_seqs == 1:
            ms_q = T_val
        else:
            h = cs_c.tolist()
            ms_q = max(h[i+1] - h[i] for i in range(num_seqs))
        t1 = time.perf_counter_ns()
        step_times['max_s_q'].append((t1 - t0) / 1000)

        # Step 2: gate setup
        t0 = time.perf_counter_ns()
        gl, bt, gl4, bt4, glp, btp = main_mod._get_gate_tensors(T_val, q_c.device)
        ga.p0.value = A_log.data_ptr()
        ga.p1.value = a_in.data_ptr()
        ga.p2.value = dt_bias.data_ptr()
        ga.p3.value = b_in.data_ptr()
        ga.p4.value = glp
        ga.p5.value = btp
        ga.p6.value = T_val
        t1 = time.perf_counter_ns()
        step_times['gate_setup'].append((t1 - t0) / 1000)

        # Step 3: gate launch
        t0 = time.perf_counter_ns()
        _cuLaunchKernel(
            gate_kernel,
            (T_val * 8 + 255) // 256, 1, 1,
            256, 1, 1,
            0, stream, ga.arr, 0,
        )
        t1 = time.perf_counter_ns()
        step_times['gate_launch'].append((t1 - t0) / 1000)

        # Step 4: bundle get
        t0 = time.perf_counter_ns()
        pn = "small" if T_val <= 1024 and num_seqs <= 8 else "large"
        rb = get_bundle(
            q_c, k_c, v_c, gl4, bt4, s_c, cs_c,
            num_seqs, T_val, ms_q, pn, main_mod._DEFAULT_SCALE,
        )
        cg = rb[0]
        op = rb[6]
        osp = rb[7]
        ps2 = rb[3]
        ns = rb[4]
        t1 = time.perf_counter_ns()
        step_times['bundle_get'].append((t1 - t0) / 1000)

        # Step 5: compiled_gdn call
        t0 = time.perf_counter_ns()
        cg(
            q_c.data_ptr(), k_c.data_ptr(), v_c.data_ptr(),
            op, glp, btp,
            ps2, s_c.data_ptr(), osp,
            ns, cs_c, stream=stream,
        )
        t1 = time.perf_counter_ns()
        step_times['compiled_call'].append((t1 - t0) / 1000)

        t_total_end = time.perf_counter_ns()
        step_times['total'].append((t_total_end - t_total_start) / 1000)

    torch.cuda.synchronize()

    for step, times in step_times.items():
        s = sorted(times)
        out.append(f"  {step:20s}: {stats(s)}")

    return "\n".join(out)


@app.local_entrypoint()
def main():
    import scripts.pack_solution as ps

    subfolder = "gdn_prefill_qk4_v8_d128_k_last"
    ps.PROJECT_ROOT = PROJECT_ROOT / subfolder

    print("[probe] Packing solution...")
    solution_path = ps.pack_solution()
    solution_json = solution_path.read_text()

    from flashinfer_bench import Solution
    solution = Solution.model_validate_json(solution_json)

    print(f"[probe] Running probe on Modal B200...")
    result = probe_compiled.remote(solution_json, solution.definition)
    print(result)
