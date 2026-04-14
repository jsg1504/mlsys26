from pathlib import Path
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import scripts.pack_solution as ps


project_root = Path(__file__).resolve().parents[1]
assert ps._get_source_dir("python", project_root) == project_root / "solution" / "python"
try:
    ps._get_source_dir("fortran", project_root)
except ValueError as exc:
    assert str(exc) == "Unsupported language: fortran"
else:
    raise AssertionError("Expected ValueError for unsupported language")

spec = ps._build_spec_from_config(
    {
        "language": "python",
        "entry_point": "main.py::run",
        "dependencies": ["nvidia-cutlass-dsl", "cuda-python"],
        "destination_passing_style": False,
    }
)
assert spec.language == "python"
assert spec.entry_point == "main.py::run"
assert spec.dependencies == ["nvidia-cutlass-dsl", "cuda-python"]
assert spec.destination_passing_style is False

with tempfile.TemporaryDirectory() as tmpdir:
    tmp_path = Path(tmpdir)
    (tmp_path / "main.py").write_text("def run():\n    pass\n")
    nested = tmp_path / "gdn_blackwell"
    nested.mkdir()
    deeper = nested / "submodule"
    deeper.mkdir()
    (nested / "__init__.py").write_text("from .gdn import chunk_gated_delta_rule\n")
    (nested / "gdn.py").write_text("value = 1\n")
    (nested / "kernel.cc").write_text("int kernel() { return 0; }\n")
    (deeper / "helper.cxx").write_text("int helper() { return 1; }\n")

    packed = ps._pack_python_solution_from_files(
        path=str(tmp_path),
        spec=spec,
        name="demo",
        definition="demo-def",
        author="demo-author",
    )

    assert [source.path for source in packed.sources] == [
        "gdn_blackwell/__init__.py",
        "gdn_blackwell/gdn.py",
        "gdn_blackwell/kernel.cc",
        "gdn_blackwell/submodule/helper.cxx",
        "main.py",
    ]
