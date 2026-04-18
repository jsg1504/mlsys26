"""List all workloads with their axes for a given definition."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import TraceSet

app = modal.App("list-workloads")
trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.from_registry("nvidia/cuda:13.2.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
)

@app.function(image=image, gpu="B200:1", timeout=300, volumes={TRACE_SET_PATH: trace_volume})
def list_workloads(definition_name: str):
    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    workloads = trace_set.workloads.get(definition_name, [])
    results = []
    for i, w in enumerate(workloads):
        results.append({"index": i, "uuid": w.workload.uuid[:8], "axes": dict(w.workload.axes)})
    return results

@app.local_entrypoint()
def main(definition: str = "gdn_decode_qk4_v8_d128_k_last"):
    import json, collections
    workloads = list_workloads.remote(definition)
    batch_counts = collections.Counter()
    for w in workloads:
        b = w["axes"].get("batch_size", "?")
        batch_counts[b] += 1
        axes_str = ", ".join(f"{k}={v}" for k, v in w["axes"].items())
        print(f"  [{w['index']:2d}] {w['uuid']} | {axes_str}")
    print(f"\nTotal: {len(workloads)} workloads")
    print("Batch size distribution:")
    for b in sorted(batch_counts.keys()):
        print(f"  B={b}: {batch_counts[b]} workloads")
