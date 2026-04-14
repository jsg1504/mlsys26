from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import scripts.pack_solution as ps


project_root = Path(__file__).resolve().parents[1]
output_path = project_root / "tests" / "solution.json"
old_project_root = ps.PROJECT_ROOT
try:
    ps.PROJECT_ROOT = project_root
    if output_path.exists():
        output_path.unlink()

    result_path = ps.pack_solution(output_path=output_path)
    data = json.loads(result_path.read_text())

    assert result_path == output_path
    assert data["spec"]["entry_point"] == "main.py::run"
    assert {source["path"] for source in data["sources"]} >= {
        "main.py",
        "prefill_contract.py",
        "gdn_blackwell/__init__.py",
        "gdn_blackwell/gdn.py",
    }
finally:
    ps.PROJECT_ROOT = old_project_root
    if output_path.exists():
        output_path.unlink()
