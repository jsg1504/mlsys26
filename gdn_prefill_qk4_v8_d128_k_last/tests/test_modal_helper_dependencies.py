import importlib.util
import json
import sys
import types
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_modal_subfolder.py"
script_text = script_path.read_text(encoding="utf-8")

assert '"nvidia-cutlass-dsl"' in script_text
assert '"cutlass"' not in script_text
assert "timeout_seconds=QUICK_BENCHMARK_TIMEOUT_SECONDS" in script_text
assert "print_phase(\"Packing solution from source files...\")" in script_text


def load_runner_module(monkeypatch):
    modal = types.ModuleType("modal")

    class DummyApp:
        def __init__(self, *args, **kwargs):
            pass

        def function(self, *args, **kwargs):
            def decorator(fn):
                return fn

            return decorator

        def local_entrypoint(self):
            def decorator(fn):
                return fn

            return decorator

    class DummyVolume:
        @classmethod
        def from_name(cls, *args, **kwargs):
            return object()

    class DummyImage:
        @classmethod
        def from_registry(cls, *args, **kwargs):
            return cls()

        def pip_install(self, *args, **kwargs):
            return self

    modal.App = DummyApp
    modal.Volume = DummyVolume
    modal.Image = DummyImage

    flashinfer_bench = types.ModuleType("flashinfer_bench")

    class BenchmarkConfig:
        def __init__(
            self,
            *,
            warmup_runs=10,
            iterations=50,
            num_trials=3,
            rtol=0.01,
            atol=0.01,
            log_dir="",
            use_isolated_runner=False,
            required_matched_ratio=None,
            sampling_validation_trials=100,
            sampling_tvd_threshold=0.2,
            definitions=None,
            solutions=None,
            timeout_seconds=300,
            profile_baseline=True,
        ):
            self.warmup_runs = warmup_runs
            self.iterations = iterations
            self.num_trials = num_trials
            self.rtol = rtol
            self.atol = atol
            self.log_dir = log_dir
            self.use_isolated_runner = use_isolated_runner
            self.required_matched_ratio = required_matched_ratio
            self.sampling_validation_trials = sampling_validation_trials
            self.sampling_tvd_threshold = sampling_tvd_threshold
            self.definitions = definitions
            self.solutions = solutions
            self.timeout_seconds = timeout_seconds
            self.profile_baseline = profile_baseline

    class Solution:
        def __init__(self, name, definition):
            self.name = name
            self.definition = definition

        @classmethod
        def model_validate_json(cls, text):
            payload = json.loads(text)
            return cls(payload["name"], payload["definition"])

    class TraceSet:
        @classmethod
        def from_path(cls, *args, **kwargs):
            return cls()

    class Benchmark:
        def __init__(self, *args, **kwargs):
            pass

        def run_all(self, *args, **kwargs):
            return types.SimpleNamespace(traces={})

    flashinfer_bench.Benchmark = Benchmark
    flashinfer_bench.BenchmarkConfig = BenchmarkConfig
    flashinfer_bench.Solution = Solution
    flashinfer_bench.TraceSet = TraceSet

    monkeypatch.setitem(sys.modules, "modal", modal)
    monkeypatch.setitem(sys.modules, "flashinfer_bench", flashinfer_bench)

    spec = importlib.util.spec_from_file_location("run_modal_subfolder_test", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_benchmark_config_helper_uses_shorter_quick_timeout(monkeypatch):
    module = load_runner_module(monkeypatch)

    quick_config = module.build_benchmark_config(True)
    full_config = module.build_benchmark_config(False)

    assert quick_config.warmup_runs == 1
    assert quick_config.iterations == 3
    assert quick_config.num_trials == 1
    assert quick_config.timeout_seconds == 120

    assert full_config.warmup_runs == 3
    assert full_config.iterations == 100
    assert full_config.num_trials == 5
    assert full_config.timeout_seconds == 300


def test_main_emits_phase_progress_and_uses_quick_timeout(monkeypatch, tmp_path):
    module = load_runner_module(monkeypatch)

    solution_path = tmp_path / "solution.json"
    solution_path.write_text('{"name": "demo", "definition": "demo_def"}', encoding="utf-8")

    fake_pack_solution = types.ModuleType("scripts.pack_solution")
    fake_pack_solution.PROJECT_ROOT = Path("/initial")
    fake_pack_solution.pack_solution = lambda: solution_path

    fake_scripts = types.ModuleType("scripts")
    fake_scripts.__path__ = []
    monkeypatch.setitem(sys.modules, "scripts", fake_scripts)
    monkeypatch.setitem(sys.modules, "scripts.pack_solution", fake_pack_solution)

    remote_calls = []

    def fake_remote(solution, config):
        remote_calls.append((solution, config))
        return {"demo_def": {}}

    module.run_benchmark.remote = fake_remote

    output = StringIO()
    with redirect_stdout(output):
        module.main(subfolder="gdn_prefill_qk4_v8_d128_k_last", quick=True)

    text = output.getvalue()
    assert "[modal] Packing solution from source files..." in text
    assert "[modal] Loading packed solution..." in text
    assert "[modal] Submitting benchmark to Modal B200 (quick mode, timeout 120s)..." in text
    assert "[modal] Remote benchmark complete." in text
    assert remote_calls[0][0].name == "demo"
    assert remote_calls[0][0].definition == "demo_def"
    assert remote_calls[0][1].timeout_seconds == 120
