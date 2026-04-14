import logging
import sys
import types
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_modal_subfolder.py"

def load_runner_module(monkeypatch):
    modal = types.ModuleType("modal")

    class DummyApp:
        def __init__(self, *args, **kwargs):
            pass

        def function(self, *args, **kwargs):
            def decorator(fn):
                fn.remote = lambda *remote_args, **remote_kwargs: None
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

    monkeypatch.setitem(sys.modules, "modal", modal)
    monkeypatch.delitem(sys.modules, "scripts", raising=False)
    monkeypatch.delitem(sys.modules, "scripts.pack_solution", raising=False)

    import importlib.util

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

    captured_payload = {}

    def fake_model_validate_json(text):
        captured_payload["text"] = text

        class FakeSolution:
            name = "demo"
            definition = "demo_def"

        return FakeSolution()

    monkeypatch.setattr(module.Solution, "model_validate_json", staticmethod(fake_model_validate_json))

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

    monkeypatch.setattr(module.run_benchmark, "remote", fake_remote)

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
    assert captured_payload["text"] == solution_path.read_text(encoding="utf-8")


def test_print_results_includes_evaluation_log(monkeypatch):
    module = load_runner_module(monkeypatch)

    output = StringIO()
    results = {
        "demo_def": {
            "workload-uuid": {
                "status": "RUNTIME_ERROR",
                "log": "first line\nsecond line",
            }
        }
    }

    with redirect_stdout(output):
        module.print_results(results)

    text = output.getvalue()
    assert "demo_def:" in text
    assert "Workload workload..." in text
    assert "evaluation.log:" in text
    assert "first line" in text
    assert "second line" in text


def test_run_benchmark_configures_package_logger_and_preserves_evaluation_log(monkeypatch):
    module = load_runner_module(monkeypatch)

    package_logger = logging.getLogger("flashinfer-bench")
    underscored_logger = logging.getLogger("flashinfer_bench")

    original_package_handlers = list(package_logger.handlers)
    original_package_level = package_logger.level
    original_package_propagate = package_logger.propagate
    original_underscore_handlers = list(underscored_logger.handlers)
    original_underscore_level = underscored_logger.level

    class FakeLog:
        status = types.SimpleNamespace(value="RUNTIME_ERROR")
        log = "traceback here\nmore detail"
        performance = None
        correctness = None

    class FakeTrace:
        workload = types.SimpleNamespace(uuid="12345678-abcdef")
        solution = "demo_solution"
        evaluation = FakeLog()

    class FakeBenchmark:
        def __init__(self, trace_set, config):
            self.trace_set = trace_set
            self.config = config

        def run_all(self, dump_traces=True):
            return types.SimpleNamespace(
                traces={"demo_def": [FakeTrace()]},
            )

    class FakeTraceSet:
        def __init__(self, root=None, definitions=None, solutions=None, workloads=None, traces=None):
            self.root = root
            self.definitions = definitions or {}
            self.solutions = solutions or {}
            self.workloads = workloads or {}
            self.traces = traces or {}

        @classmethod
        def from_path(cls, path):
            return cls(
                root=Path("/tmp"),
                definitions={"demo_def": types.SimpleNamespace(name="demo_def")},
                workloads={"demo_def": [types.SimpleNamespace(uuid="12345678-abcdef")]},
            )

    monkeypatch.setattr(module, "Benchmark", FakeBenchmark)
    monkeypatch.setattr(module, "TraceSet", FakeTraceSet)

    class FakeSolution:
        definition = "demo_def"

    observed_package_level = None
    observed_package_handlers = None
    observed_underscore_handlers = None

    try:
        result = module.run_benchmark(FakeSolution(), None)
        observed_package_level = package_logger.level
        observed_package_handlers = list(package_logger.handlers)
        observed_underscore_handlers = list(underscored_logger.handlers)
    finally:
        package_logger.handlers = original_package_handlers
        package_logger.setLevel(original_package_level)
        package_logger.propagate = original_package_propagate
        underscored_logger.handlers = original_underscore_handlers
        underscored_logger.setLevel(original_underscore_level)

    assert observed_package_level == logging.INFO
    assert any(isinstance(handler, logging.StreamHandler) for handler in observed_package_handlers)
    assert all(not isinstance(handler, logging.StreamHandler) for handler in observed_underscore_handlers)
    assert result["demo_def"]["12345678-abcdef"]["log"] == "traceback here\nmore detail"
