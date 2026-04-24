"""
Run actual NVIDIA Nsight Compute (ncu) profiling on Modal B200.
Collects hardware-level kernel metrics: SM occupancy, memory throughput,
compute throughput, stall reasons, register usage.

Usage:
    modal run scripts/ncu_profile_modal.py --subfolder gdn_prefill_qk4_v8_d128_k_last
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
def run_ncu_profiling(solution_json: str, definition_name: str, workload_indices: list[int]) -> str:
    import json
    import subprocess
    import tempfile
    import os
    import torch

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

    output_lines = []

    # Check ncu availability
    ncu_paths = [
        "/usr/local/cuda/bin/ncu",
        "/opt/nvidia/nsight-compute/ncu",
        "ncu",
    ]
    ncu_bin = None
    for p in ncu_paths:
        try:
            r = subprocess.run([p, "--version"], capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                ncu_bin = p
                output_lines.append(f"ncu found: {p}")
                output_lines.append(r.stdout.strip()[:200])
                break
        except Exception:
            continue

    if ncu_bin is None:
        output_lines.append("WARNING: ncu not found. Using CUPTI-based profiling via torch.profiler instead.")
        output_lines.append("Checking nsight-compute packages...")
        r = subprocess.run(["find", "/usr", "/opt", "-name", "ncu", "-type", "f"], capture_output=True, text=True, timeout=10)
        output_lines.append(f"find result: {r.stdout.strip()[:300]}")

    output_lines.append("")

    # For each workload, profile with detailed CUPTI metrics
    for idx in workload_indices:
        if idx >= len(workloads):
            continue

        trace = workloads[idx]
        workload = trace.workload
        safe_tensors = load_safetensors(definition, workload, trace_set.root) if any(
            s.type == "safetensors" for s in workload.inputs.values()
        ) else None
        inputs = gen_inputs(definition, workload, device="cuda", safe_tensors=safe_tensors)

        q = inputs[0] if len(inputs) > 0 else None
        cu_seqlens = inputs[8] if len(inputs) > 8 else None
        T = q.shape[0] if q is not None else "?"
        num_seqs = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else "?"

        output_lines.append(f"=== Workload {idx} (uuid={workload.uuid[:8]}, T={T}, num_seqs={num_seqs}) ===")

        if cu_seqlens is not None and isinstance(num_seqs, int):
            h = cu_seqlens.tolist()
            seq_lens = [h[i+1] - h[i] for i in range(num_seqs)]
            max_s_q = max(seq_lens) if seq_lens else 0
            output_lines.append(f"  max_s_q={max_s_q}, seq_lens={seq_lens[:5]}{'...' if len(seq_lens) > 5 else ''}")

        # Warmup
        for _ in range(3):
            runnable(*[x.clone() if isinstance(x, torch.Tensor) else x for x in inputs])
        torch.cuda.synchronize()

        # Profile with torch.profiler using CUPTI for detailed kernel info
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=False,
            with_flops=True,
        ) as prof:
            for _ in range(3):
                cloned = [x.clone() if isinstance(x, torch.Tensor) else x for x in inputs]
                runnable(*cloned)
            torch.cuda.synchronize()

        # Detailed kernel events
        output_lines.append("  --- CUDA Kernel Events ---")
        events = prof.key_averages()
        all_events = sorted(events, key=lambda e: e.self_device_time_total, reverse=True)

        for ev in all_events[:12]:
            if ev.self_device_time_total <= 0:
                continue
            name = ev.key[:90]
            count = ev.count
            total_us = ev.self_device_time_total
            avg_us = total_us / count if count > 0 else 0
            cpu_total = ev.cpu_time_total
            flops = ev.flops if hasattr(ev, 'flops') and ev.flops else 0
            output_lines.append(
                f"    {name}"
            )
            output_lines.append(
                f"      calls={count}  gpu_total={total_us:.0f}us  gpu_avg={avg_us:.1f}us  cpu_total={cpu_total:.0f}us  flops={flops}"
            )

        # If ncu is available, run it on a small standalone script
        if ncu_bin:
            output_lines.append("  --- NCU Hardware Metrics ---")
            # Save solution to file (avoid JSON escaping)
            sol_path = "/tmp/_ncu_solution.json"
            with open(sol_path, "w") as sf:
                sf.write(solution_json)
            # Create profiling script
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir='/tmp') as f:
                f.write(
                    "import torch\n"
                    "import sys\n"
                    "sys.path.insert(0, '/root')\n"
                    "from flashinfer_bench import Solution, TraceSet\n"
                    "from flashinfer_bench.compile.registry import BuilderRegistry\n"
                    "from flashinfer_bench.bench.utils import gen_inputs, load_safetensors\n"
                    "import logging\n"
                    "logging.disable(logging.CRITICAL)\n"
                    f"solution = Solution.model_validate_json(open('{sol_path}').read())\n"
                    f"trace_set = TraceSet.from_path('{TRACE_SET_PATH}')\n"
                    f"definition = trace_set.definitions['{definition_name}']\n"
                    f"workloads = trace_set.workloads.get('{definition_name}', [])\n"
                    f"trace = workloads[{idx}]\n"
                    "workload = trace.workload\n"
                    "safe_tensors = load_safetensors(definition, workload, trace_set.root) if any(\n"
                    "    s.type == 'safetensors' for s in workload.inputs.values()\n"
                    ") else None\n"
                    "inputs = gen_inputs(definition, workload, device='cuda', safe_tensors=safe_tensors)\n"
                    "registry = BuilderRegistry.get_instance()\n"
                    "runnable = registry.build(definition, solution)\n"
                    "for _ in range(3):\n"
                    "    runnable(*[x.clone() if isinstance(x, torch.Tensor) else x for x in inputs])\n"
                    "torch.cuda.synchronize()\n"
                    "torch.cuda.cudart().cudaProfilerStart()\n"
                    "cloned = [x.clone() if isinstance(x, torch.Tensor) else x for x in inputs]\n"
                    "torch.cuda.synchronize()\n"
                    "runnable(*cloned)\n"
                    "torch.cuda.synchronize()\n"
                    "torch.cuda.cudart().cudaProfilerStop()\n"
                )
                script_path = f.name

            try:
                # Focus on SpeedOfLight only (user-requested section)
                # Filter to our kernels only (not Activity Buffer etc)
                ncu_cmd = [
                    ncu_bin,
                    "--section", "SpeedOfLight",
                    "--kernel-name", "regex:gdn_prefill_sequential|fused_gate_kernel|kernel_cutlass",
                    "--profile-from-start", "off",
                    "--target-processes", "all",
                    "--print-summary", "per-kernel",
                    "python3", script_path,
                ]
                result = subprocess.run(
                    ncu_cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
                )
                if result.returncode == 0:
                    output_lines.append(result.stdout)
                else:
                    output_lines.append(f"  ncu failed (rc={result.returncode})")
                    output_lines.append(f"  stderr: {result.stderr[:1000]}")
                    output_lines.append(f"  stdout: {result.stdout[:1000]}")
            except subprocess.TimeoutExpired:
                output_lines.append("  ncu timed out (300s)")
            except Exception as e:
                output_lines.append(f"  ncu error: {e}")
            finally:
                os.unlink(script_path)
        else:
            output_lines.append("  (ncu not available — hardware metrics skipped)")

        # CUDA event timing
        output_lines.append("  --- CUDA Event Timing ---")
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
        output_lines.append(f"    10 iters: avg={avg_ms:.4f}ms  min={min_ms:.4f}ms")
        output_lines.append("")

    return "\n".join(output_lines)


@app.local_entrypoint()
def main(subfolder: str, indices: str = "0,40,80"):
    """Run ncu profiling on representative workloads.

    Default indices: 0 (short seq), 40 (medium CuTe-DSL), 80 (medium seq)
    """
    import scripts.pack_solution as ps

    ps.PROJECT_ROOT = PROJECT_ROOT / subfolder

    print("[ncu] Packing solution...")
    solution_path = ps.pack_solution()
    solution_json = solution_path.read_text()

    from flashinfer_bench import Solution
    solution = Solution.model_validate_json(solution_json)

    idx_list = [int(x.strip()) for x in indices.split(",")]
    print(f"[ncu] Profiling workloads {idx_list} on Modal B200...")

    result = run_ncu_profiling.remote(solution_json, solution.definition, idx_list)
    print(result)
