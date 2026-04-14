"""
Run Modal benchmark for a subfolder-based solution.

Usage:
    modal run scripts/run_modal_subfolder.py --subfolder gdn_decode_qk4_v8_d128_k_last
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet
from flashinfer_bench.logging import configure_logging

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"
DEFAULT_BENCHMARK_TIMEOUT_SECONDS = 300
QUICK_BENCHMARK_TIMEOUT_SECONDS = 120
QUICK_WORKLOAD_LIMIT = 3

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
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


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark(solution: Solution, config: BenchmarkConfig = None, workload_limit: int | None = None) -> dict:
    """Run benchmark on Modal B200 and return results."""
    if config is None:
        config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)

    if workload_limit is None and (
        config.warmup_runs == 1
        and config.iterations == 3
        and config.num_trials == 1
        and config.timeout_seconds == QUICK_BENCHMARK_TIMEOUT_SECONDS
    ):
        workload_limit = QUICK_WORKLOAD_LIMIT

    configure_logging(level=logging.INFO)

    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    if workload_limit is not None:
        workloads = workloads[:workload_limit]

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if trace.evaluation:
            entry = {
                "status": trace.evaluation.status.value,
                "solution": trace.solution,
            }
            if trace.evaluation.log:
                entry["log"] = trace.evaluation.log
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            results[definition.name][trace.workload.uuid] = entry

    return results


def build_benchmark_config(quick: bool) -> BenchmarkConfig:
    """Build the benchmark config for quick or full Modal runs."""
    if quick:
        return BenchmarkConfig(
            warmup_runs=1,
            iterations=3,
            num_trials=1,
            timeout_seconds=QUICK_BENCHMARK_TIMEOUT_SECONDS,
        )

    return BenchmarkConfig(
        warmup_runs=3,
        iterations=100,
        num_trials=5,
        timeout_seconds=DEFAULT_BENCHMARK_TIMEOUT_SECONDS,
    )


def print_phase(message: str):
    """Print a short phase marker for Modal quick runs."""
    print(f"[modal] {message}")


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        for workload_uuid, result in traces.items():
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")

            if result.get("latency_ms") is not None:
                print(f" | {result['latency_ms']:.3f} ms", end="")

            if result.get("speedup_factor") is not None:
                print(f" | {result['speedup_factor']:.2f}x speedup", end="")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")

            print()

            log = result.get("log")
            if log:
                print("    evaluation.log:")
                for line in log.rstrip().splitlines():
                    print(f"      {line}")


@app.local_entrypoint()
def main(subfolder: str, quick: bool = True):
    """Load pre-packed solution.json from subfolder and run benchmark on Modal.

    Args:
        subfolder: Which subfolder to benchmark (e.g. gdn_decode_qk4_v8_d128_k_last)
        quick: Use fast config for development (default). Pass --no-quick for final submission.
    """
    import scripts.pack_solution as ps

    ps.PROJECT_ROOT = PROJECT_ROOT / subfolder

    print_phase("Packing solution from source files...")
    solution_path = ps.pack_solution()

    print_phase("Loading packed solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    config = build_benchmark_config(quick)
    mode = "quick" if quick else "full"
    workload_limit = QUICK_WORKLOAD_LIMIT if quick else None
    workload_note = f", {workload_limit} workloads max" if workload_limit is not None else ""
    print_phase(
        f"Submitting benchmark to Modal B200 ({mode} mode, timeout {config.timeout_seconds}s{workload_note})..."
    )

    results = run_benchmark.remote(solution, config, workload_limit)
    print_phase("Remote benchmark complete.")

    if not results:
        print("No results returned!")
        return

    print_results(results)
