import importlib
import shutil
import sys
import tempfile
from pathlib import Path


solution_python = Path(__file__).resolve().parents[1] / "solution" / "python"


with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir_path = Path(tmpdir)
    package_name = "packaged_runtime_case"
    package_root = tmpdir_path / package_name
    shutil.copytree(solution_python, package_root)
    (package_root / "__init__.py").write_text("", encoding="utf-8")

    sys.path.insert(0, str(tmpdir_path))
    try:
        imported_main = importlib.import_module(f"{package_name}.main")
    except Exception as exc:
        raise AssertionError(f"Expected packaged import of {package_name}.main to succeed, got: {exc!r}") from exc
    finally:
        sys.path.pop(0)
        for module_name in list(sys.modules):
            if module_name == package_name or module_name.startswith(f"{package_name}."):
                sys.modules.pop(module_name, None)

assert hasattr(imported_main, "run")
