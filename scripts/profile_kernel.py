"""
Profile GDN decode or prefill kernel with NCU on Modal B200.

Usage:
  modal run scripts/profile_kernel.py --kernel decode
  modal run scripts/profile_kernel.py --kernel prefill
  modal run scripts/profile_kernel.py --kernel both
  modal run scripts/profile_kernel.py --kernel decode --max-workloads 3

Uses flashinfer_bench's solution runner for build/execution,
but invokes NCU directly (without NVTX filtering) for profiling.
Workloads come from the actual contest trace set.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Solution, TraceSet

app = modal.App("ncu-profile")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .pip_install(
        "flashinfer-bench", "torch", "triton", "numpy",
        "nvidia-cutlass-dsl", "cuda-python",
    )
)

SECTIONS = ["SpeedOfLight", "MemoryWorkloadAnalysis", "LaunchStats", "Occupancy"]

# Kernel name regex patterns for NCU --kernel-name
# CuTe-DSL JIT kernels are named "kernel_cutlass_kernel_fib_pyt..."
KERNEL_NAMES = {
    "gdn_decode_qk4_v8_d128_k_last": "gdn_decode_kernel",
    "gdn_prefill_qk4_v8_d128_k_last": "regex:kernel_cutlass",
}


@app.function(
    image=image, gpu="B200:1", timeout=1800,
    volumes={TRACE_SET_PATH: trace_volume},
)
def profile(solution: Solution, max_workloads: int = 0):
    """Profile a solution against its workloads using NCU."""
    import subprocess, tempfile

    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    definition_name = solution.definition

    if definition_name not in trace_set.definitions:
        print(f"ERROR: Definition '{definition_name}' not found in trace set")
        return {}

    definition = trace_set.definitions[definition_name]
    kernel_name = KERNEL_NAMES.get(definition_name, "kernel")

    workload_traces = trace_set.workloads.get(definition_name, [])
    if not workload_traces:
        print(f"ERROR: No workloads found for '{definition_name}'")
        return {}

    if max_workloads > 0:
        workload_traces = workload_traces[:max_workloads]

    print(f"Profiling {definition_name}: {len(workload_traces)} workloads")
    print(f"Kernel filter: {kernel_name}")
    print(f"Sections: {SECTIONS}")

    results = {}
    for i, trace in enumerate(workload_traces):
        workload = trace.workload
        axes_str = ", ".join(f"{k}={v}" for k, v in workload.axes.items())
        print(f"\n{'='*60}")
        print(f"  [{i+1}/{len(workload_traces)}] {axes_str}")
        print(f"{'='*60}")

        with tempfile.TemporaryDirectory(prefix="ncu_") as data_dir:
            data_path = Path(data_dir)
            (data_path / "definition.json").write_text(definition.model_dump_json())
            (data_path / "solution.json").write_text(solution.model_dump_json())
            (data_path / "workload.json").write_text(workload.model_dump_json())

            # Build NCU command: profile the actual GDN kernel
            # For prefill: skip the first GDN launch (warmup/compile),
            # profile the second one (warm cache)
            skip = 1 if "prefill" in definition_name else 1
            cmd = ["ncu", "--page", "details", "--kernel-name", kernel_name,
                   "--launch-skip", str(skip), "--launch-count", "1", "-f"]
            for s in SECTIONS:
                cmd.extend(["--section", s])
            cmd.extend([
                sys.executable, "-u", "-m",
                "flashinfer_bench.agents._solution_runner",
                "--data-dir", data_dir,
                "--device", "cuda:0",
                "--trace-set-path", TRACE_SET_PATH,
            ])

            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=300
                )
                output = result.stdout + result.stderr
                print(output)
                results[workload.uuid[:8]] = output
            except subprocess.TimeoutExpired:
                print(f"TIMEOUT after 300s")
                results[workload.uuid[:8]] = "TIMEOUT"

    return results


def pack_subfolder(subfolder: str) -> Solution:
    """Pack a subfolder's solution."""
    import scripts.pack_solution as ps
    ps.PROJECT_ROOT = PROJECT_ROOT / subfolder
    solution_path = ps.pack_solution()
    return Solution.model_validate_json(solution_path.read_text())


@app.local_entrypoint()
def main(kernel: str = "both", max_workloads: int = 0):
    """Profile GDN kernels with NCU.

    Args:
        kernel: Which kernel to profile (decode, prefill, both).
        max_workloads: Max workloads to profile per kernel (0 = all).
    """
    targets = {
        "decode": "gdn_decode_qk4_v8_d128_k_last",
        "prefill": "gdn_prefill_qk4_v8_d128_k_last",
    }

    if kernel == "both":
        to_profile = list(targets.items())
    elif kernel in targets:
        to_profile = [(kernel, targets[kernel])]
    else:
        print(f"Unknown kernel: {kernel}. Use decode, prefill, or both.")
        return

    for name, subfolder in to_profile:
        print(f"\n=== Packing {name} solution ===")
        solution = pack_subfolder(subfolder)
        print(f"  {solution.name} ({solution.definition})")

        print(f"\n=== Profiling {name} on Modal B200 ===")
        results = profile.remote(solution, max_workloads)
        if not results:
            print(f"No results for {name}!")
