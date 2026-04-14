from pathlib import Path


script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_modal_subfolder.py"
script_text = script_path.read_text(encoding="utf-8")

assert '"nvidia-cutlass-dsl"' in script_text
assert '"cutlass"' not in script_text
