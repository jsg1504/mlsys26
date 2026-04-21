"""
NCU profiling on Modal B200 — profiles representative workloads.

Usage:
    modal run scripts/profile_modal.py --subfolder gdn_prefill_qk4_v8_d128_k_last
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("flashinfer-ncu-profile")

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


@app.function(image=image, gpu="B200:1", timeout=1800, volumes={TRACE_SET_PATH: trace_volume})
def run_profiling(solution_json: str, definition_name: str, workload_indices: list[int]) -> str:
    """Profile specific workloads with torch profiler and CUDA events."""
    import json
    import torch
    import time
    import logging

    from flashinfer_bench import Solution, TraceSet
    from flashinfer_bench.compile.registry import BuilderRegistry
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
    from flashinfer_bench.logging import configure_logging

    configure_logging(level=logging.WARNING)

    solution = Solution.model_validate_json(solution_json)
    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    definition = trace_set.definitions[definition_name]
    workloads = trace_set.workloads.get(definition_name, [])

    registry = BuilderRegistry.get_instance()
    runnable = registry.build(definition, solution)

    output_lines = []
    output_lines.append(f"=== NCU-style Profiling on B200 ===")
    output_lines.append(f"Total workloads available: {len(workloads)}")
    output_lines.append(f"Profiling indices: {workload_indices}")
    output_lines.append("")

    for idx in workload_indices:
        if idx >= len(workloads):
            output_lines.append(f"[{idx}] SKIP — index out of range")
            continue

        trace = workloads[idx]  # actually Trace objects
        workload = trace.workload
        safe_tensors = load_safetensors(definition, workload, trace_set.root) if any(
            s.type == "safetensors" for s in workload.inputs.values()
        ) else None
        inputs = gen_inputs(definition, workload, device="cuda", safe_tensors=safe_tensors)

        # Extract workload shape info
        q = inputs[0] if len(inputs) > 0 else None
        cu_seqlens = inputs[8] if len(inputs) > 8 else None
        T = q.shape[0] if q is not None else "?"
        num_seqs = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else "?"

        output_lines.append(f"--- Workload {idx} (uuid={workload.uuid[:8]}) ---")
        output_lines.append(f"  T={T}, num_seqs={num_seqs}")

        if cu_seqlens is not None and isinstance(num_seqs, int):
            h = cu_seqlens.tolist()
            seq_lens = [h[i+1] - h[i] for i in range(num_seqs)]
            output_lines.append(f"  seq_lens={seq_lens[:10]}{'...' if len(seq_lens) > 10 else ''}")
            output_lines.append(f"  max_s_q={max(seq_lens)}")

        # Warmup
        for _ in range(3):
            runnable(*[x.clone() if isinstance(x, torch.Tensor) else x for x in inputs])
        torch.cuda.synchronize()

        # Profile with torch profiler
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=False,
        ) as prof:
            for _ in range(5):
                cloned = [x.clone() if isinstance(x, torch.Tensor) else x for x in inputs]
                runnable(*cloned)
            torch.cuda.synchronize()

        # Extract kernel-level info
        output_lines.append(f"  --- Kernel Trace (5 iterations) ---")

        events = prof.key_averages()
        # Sort by self_device_time_total (GPU time attributable to the op itself)
        all_events = sorted(events, key=lambda e: e.self_device_time_total, reverse=True)

        for ev in all_events[:20]:
            if ev.self_device_time_total <= 0:
                continue
            name = ev.key[:85]
            count = ev.count
            total_us = ev.self_device_time_total
            avg_us = total_us / count if count > 0 else 0
            output_lines.append(
                f"    {name:<85s} count={count:3d}  total={total_us:8.0f}us  avg={avg_us:7.1f}us"
            )

        # Also do CUDA event timing per iteration
        output_lines.append(f"  --- Per-iteration CUDA event timing ---")
        iter_times = []
        for _ in range(10):
            cloned = [x.clone() if isinstance(x, torch.Tensor) else x for x in inputs]
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            runnable(*cloned)
            end.record()
            torch.cuda.synchronize()
            iter_times.append(start.elapsed_time(end))
        avg_ms = sum(iter_times) / len(iter_times)
        min_ms = min(iter_times)
        max_ms = max(iter_times)
        output_lines.append(f"    10 iters: avg={avg_ms:.4f}ms, min={min_ms:.4f}ms, max={max_ms:.4f}ms")
        output_lines.append(f"    raw: {[round(t, 4) for t in iter_times]}")
        output_lines.append("")

    return "\n".join(output_lines)


@app.local_entrypoint()
def main(subfolder: str, indices: str = "0,10,30,50,70,90"):
    """Profile specific workload indices.

    Args:
        subfolder: e.g. gdn_prefill_qk4_v8_d128_k_last
        indices: comma-separated workload indices to profile (default: 0,10,30,50,70,90)
    """
    import scripts.pack_solution as ps

    ps.PROJECT_ROOT = PROJECT_ROOT / subfolder

    print("[profile] Packing solution...")
    solution_path = ps.pack_solution()
    solution_json = solution_path.read_text()

    from flashinfer_bench import Solution
    solution = Solution.model_validate_json(solution_json)
    definition_name = solution.definition

    idx_list = [int(x.strip()) for x in indices.split(",")]
    print(f"[profile] Profiling workloads {idx_list} on Modal B200...")

    result = run_profiling.remote(solution_json, definition_name, idx_list)
    print(result)
