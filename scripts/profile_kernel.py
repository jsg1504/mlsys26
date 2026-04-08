"""
Profile GDN decode or prefill kernel with NCU on Modal B200.

Usage:
  modal run scripts/profile_kernel.py --kernel decode
  modal run scripts/profile_kernel.py --kernel prefill
  modal run scripts/profile_kernel.py --kernel both
  modal run scripts/profile_kernel.py --kernel decode --max-workloads 3
  modal run scripts/profile_kernel.py --kernel prefill --prefill-path micro2 --max-workloads 3
  modal run scripts/profile_kernel.py --kernel prefill --prefill-path long16 --max-workloads 3

Uses flashinfer_bench's solution runner for build/execution,
but invokes NCU directly (without NVTX filtering) for profiling.
Workloads come from the actual contest trace set.
"""

import sys
from collections import Counter
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
        "nvidia/cuda:13.2.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
)

SECTIONS = ["SpeedOfLight", "MemoryWorkloadAnalysis", "LaunchStats", "Occupancy"]

NUM_V_HEADS = 8
HEAD_DIM = 128
DEFAULT_STATE_TILE_ROWS = 16
SMALL_STATE_TILE_ROWS = 8
TINY_STATE_TILE_ROWS = 4
PREFILL_DEFINITION = "gdn_prefill_qk4_v8_d128_k_last"
VALID_PREFILL_PATHS = ("all", "micro2", "long16")

# Kernel name regex patterns for NCU --kernel-name
KERNEL_NAMES = {
    "gdn_decode_qk4_v8_d128_k_last": "gdn_decode_kernel",
}
PREFILL_KERNEL_REGEX = {
    "all": r"gdn_prefill_kernel.*",
    "micro2": r"gdn_prefill_kernel_micro",
    "long16": r"gdn_prefill_kernel_long",
}


@app.function(
    image=image, gpu="B200:1", timeout=1800,
    volumes={TRACE_SET_PATH: trace_volume},
)
def profile(solution: Solution, max_workloads: int = 0, prefill_path: str = "all"):
    """Profile a solution against its workloads using NCU."""
    import subprocess
    import tempfile

    import torch

    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    definition_name = solution.definition

    if definition_name not in trace_set.definitions:
        print(f"ERROR: Definition '{definition_name}' not found in trace set")
        return {}

    definition = trace_set.definitions[definition_name]
    workload_traces = trace_set.workloads.get(definition_name, [])
    if not workload_traces:
        print(f"ERROR: No workloads found for '{definition_name}'")
        return {}

    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    dispatch_counts = Counter()
    if definition_name == PREFILL_DEFINITION:
        filtered_workloads = []
        for trace in workload_traces:
            dispatch = classify_prefill_dispatch(trace.workload.axes["num_seqs"], sm_count)
            dispatch_counts[dispatch] += 1
            if prefill_path == "all" or dispatch == prefill_path:
                filtered_workloads.append(trace)
        workload_traces = filtered_workloads

    if max_workloads > 0:
        workload_traces = workload_traces[:max_workloads]

    kernel_name = resolve_kernel_name(definition_name, prefill_path)

    print(f"Profiling {definition_name}: {len(workload_traces)} workloads")
    print(f"Kernel filter: {kernel_name}")
    print(f"Sections: {SECTIONS}")
    if definition_name == PREFILL_DEFINITION:
        print(f"Device SM count: {sm_count}")
        print(
            "Prefill dispatch counts: "
            f"micro2={dispatch_counts.get('micro2', 0)}, "
            f"4-row={dispatch_counts.get('4-row', 0)}, "
            f"8-row={dispatch_counts.get('8-row', 0)}, "
            f"long16={dispatch_counts.get('long16', 0)}"
        )
        print(f"Selected prefill path: {prefill_path}")

    results = {}
    for i, trace in enumerate(workload_traces):
        workload = trace.workload
        axes_parts = [f"{k}={v}" for k, v in workload.axes.items()]
        if definition_name == PREFILL_DEFINITION:
            axes_parts.insert(
                0, f"dispatch={classify_prefill_dispatch(workload.axes['num_seqs'], sm_count)}"
            )
        axes_str = ", ".join(axes_parts)
        print(f"\n{'='*60}")
        print(f"  [{i+1}/{len(workload_traces)}] {axes_str}")
        print(f"{'='*60}")

        with tempfile.TemporaryDirectory(prefix="ncu_") as data_dir:
            data_path = Path(data_dir)
            (data_path / "definition.json").write_text(definition.model_dump_json())
            (data_path / "solution.json").write_text(solution.model_dump_json())
            (data_path / "workload.json").write_text(workload.model_dump_json())

            # Build NCU command: use --kernel-name + --launch-skip instead of NVTX
            cmd = ["ncu", "--page", "details", "--kernel-name", kernel_name,
                   "--launch-skip", "1", "--launch-count", "1", "-f"]
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


def classify_prefill_dispatch(num_seqs: int, sm_count: int) -> str:
    """Mirror the live prefill wrapper's dispatch thresholds."""
    default_grid_blocks = num_seqs * NUM_V_HEADS * (HEAD_DIM // DEFAULT_STATE_TILE_ROWS)
    small_grid_blocks = num_seqs * NUM_V_HEADS * (HEAD_DIM // SMALL_STATE_TILE_ROWS)
    tiny_grid_blocks = num_seqs * NUM_V_HEADS * (HEAD_DIM // TINY_STATE_TILE_ROWS)
    if tiny_grid_blocks < (4 * sm_count):
        return "micro2"
    if small_grid_blocks < (2 * sm_count):
        return "4-row"
    if default_grid_blocks < sm_count:
        return "8-row"
    return "long16"


def resolve_kernel_name(definition_name: str, prefill_path: str) -> str:
    """Pick the NCU kernel-name regex for the current target."""
    if definition_name == PREFILL_DEFINITION:
        return PREFILL_KERNEL_REGEX[prefill_path]
    return KERNEL_NAMES.get(definition_name, "kernel")


def pack_subfolder(subfolder: str) -> Solution:
    """Pack a subfolder's solution."""
    import scripts.pack_solution as ps
    ps.PROJECT_ROOT = PROJECT_ROOT / subfolder
    solution_path = ps.pack_solution()
    return Solution.model_validate_json(solution_path.read_text())


@app.local_entrypoint()
def main(kernel: str = "both", max_workloads: int = 0, prefill_path: str = "all"):
    """Profile GDN kernels with NCU.

    Args:
        kernel: Which kernel to profile (decode, prefill, both).
        max_workloads: Max workloads to profile per kernel (0 = all).
        prefill_path: Prefill dispatch path to profile (all, micro2, long16).
    """
    if prefill_path not in VALID_PREFILL_PATHS:
        print(
            f"Unknown prefill path: {prefill_path}. "
            f"Use one of: {', '.join(VALID_PREFILL_PATHS)}."
        )
        return

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
        results = profile.remote(solution, max_workloads, prefill_path)
        if not results:
            print(f"No results for {name}!")
