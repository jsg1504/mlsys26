"""Profile specific workload indices on Modal B200 with NCU."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Solution, TraceSet

app = modal.App("ncu-selective")
trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.from_registry("nvidia/cuda:13.2.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
    .env({"TVM_FFI_CUDA_ARCH_LIST": "10.0a"})
)

SECTIONS = ["SpeedOfLight", "MemoryWorkloadAnalysis", "LaunchStats", "Occupancy"]
KERNEL_NAMES = {
    "gdn_decode_qk4_v8_d128_k_last": "gdn_decode_kernel",
    "gdn_prefill_qk4_v8_d128_k_last": "gdn_prefill_kernel",
}

@app.function(image=image, gpu="B200:1", timeout=1800, volumes={TRACE_SET_PATH: trace_volume})
def profile_selected(solution: Solution, indices: list):
    import subprocess, tempfile
    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    definition_name = solution.definition
    definition = trace_set.definitions[definition_name]
    kernel_name = KERNEL_NAMES.get(definition_name, "kernel")
    workload_traces = trace_set.workloads.get(definition_name, [])

    print(f"Profiling {definition_name}: indices {indices} out of {len(workload_traces)}")
    results = {}

    for idx in indices:
        if idx >= len(workload_traces):
            print(f"Index {idx} out of range, skipping")
            continue
        trace = workload_traces[idx]
        workload = trace.workload
        axes_str = ", ".join(f"{k}={v}" for k, v in workload.axes.items())
        print(f"\n{'='*60}")
        print(f"  [{idx}] {axes_str}")
        print(f"{'='*60}")

        with tempfile.TemporaryDirectory(prefix="ncu_") as data_dir:
            data_path = Path(data_dir)
            (data_path / "definition.json").write_text(definition.model_dump_json())
            (data_path / "solution.json").write_text(solution.model_dump_json())
            (data_path / "workload.json").write_text(workload.model_dump_json())

            cmd = ["ncu", "--page", "details", "--kernel-name", kernel_name,
                   "--launch-skip", "1", "--launch-count", "1", "-f"]
            for s in SECTIONS:
                cmd.extend(["--section", s])
            cmd.extend([
                sys.executable, "-u", "-m",
                "flashinfer_bench.agents._solution_runner",
                "--data-dir", data_dir, "--device", "cuda:0",
                "--trace-set-path", TRACE_SET_PATH,
            ])
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                output = result.stdout + result.stderr
                print(output)
                results[f"idx{idx}_{workload.uuid[:8]}"] = output
            except subprocess.TimeoutExpired:
                print(f"TIMEOUT after 300s")
                results[f"idx{idx}_{workload.uuid[:8]}"] = "TIMEOUT"
    return results

def pack_subfolder(subfolder: str) -> Solution:
    import scripts.pack_solution as ps
    ps.PROJECT_ROOT = PROJECT_ROOT / subfolder
    solution_path = ps.pack_solution()
    return Solution.model_validate_json(solution_path.read_text())

@app.local_entrypoint()
def main(kernel: str = "decode", indices: str = "0,10,18,25,32,39,46"):
    targets = {
        "decode": "gdn_decode_qk4_v8_d128_k_last",
        "prefill": "gdn_prefill_qk4_v8_d128_k_last",
    }
    subfolder = targets.get(kernel)
    if not subfolder:
        print(f"Unknown kernel: {kernel}")
        return

    idx_list = [int(x.strip()) for x in indices.split(",")]
    print(f"=== Profiling {kernel} at indices {idx_list} ===")
    solution = pack_subfolder(subfolder)
    results = profile_selected.remote(solution, idx_list)
    if not results:
        print("No results!")
