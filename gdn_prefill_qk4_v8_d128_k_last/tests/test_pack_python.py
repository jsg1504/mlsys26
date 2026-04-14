from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import scripts.pack_solution as ps


project_root = Path(__file__).resolve().parents[1]
assert ps._get_source_dir("python", project_root) == project_root / "solution" / "python"
spec = ps._build_spec_from_config(
    {
        "language": "python",
        "entry_point": "main.py::run",
        "dependencies": ["cutlass", "cuda-python"],
        "destination_passing_style": False,
    }
)
assert spec.language == "python"
assert spec.entry_point == "main.py::run"
