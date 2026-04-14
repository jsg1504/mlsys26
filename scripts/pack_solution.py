"""
Pack solution source files into solution.json.

Reads configuration from config.toml and packs the appropriate source files
for the configured solution language into a Solution JSON file for submission.
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from flashinfer_bench import BuildSpec
from flashinfer_bench import Solution, SourceFile
from flashinfer_bench.agents import pack_solution_from_files


def _get_source_dir(language: str, project_root: Path) -> Path:
    mapping = {
        "triton": project_root / "solution" / "triton",
        "cuda": project_root / "solution" / "cuda",
        "python": project_root / "solution" / "python",
    }
    try:
        return mapping[language]
    except KeyError as exc:
        raise ValueError(f"Unsupported language: {language}") from exc


def _build_spec_from_config(build_config: dict) -> BuildSpec:
    return BuildSpec(
        language=build_config["language"],
        target_hardware=["cuda"],
        entry_point=build_config["entry_point"],
        dependencies=build_config.get("dependencies", []),
        destination_passing_style=build_config.get("destination_passing_style", True),
    )


def _pack_python_solution_from_files(
    path: str, spec: BuildSpec, name: str, definition: str, author: str, description: str = ""
) -> Solution:
    """Pack a Python solution directory recursively, preserving relative paths."""
    path_obj = Path(path)

    if not path_obj.exists():
        raise ValueError(f"Path does not exist: {path}")

    if not path_obj.is_dir():
        raise ValueError(f"Path is not a directory: {path}")

    sources = []

    for file_path in sorted(path_obj.rglob("*")):
        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()
        if ext not in {".py", ".cu", ".cuh", ".cpp", ".c", ".h", ".hpp"}:
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        sources.append(
            SourceFile(path=file_path.relative_to(path_obj).as_posix(), content=content)
        )

    if not sources:
        raise ValueError(f"No source files found in directory: {path}")

    return Solution(
        name=name,
        definition=definition,
        author=author,
        description=description,
        spec=spec,
        sources=sources,
    )


def load_config() -> dict:
    """Load configuration from config.toml."""
    config_path = PROJECT_ROOT / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "rb") as f:
        return tomllib.load(f)


def pack_solution(output_path: Path = None) -> Path:
    """Pack solution files into a Solution JSON."""
    config = load_config()

    solution_config = config["solution"]
    build_config = config["build"]

    language = build_config["language"]
    source_dir = _get_source_dir(language, PROJECT_ROOT)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # Create build spec
    spec = _build_spec_from_config(build_config)

    # Pack the solution
    if language == "python":
        solution = _pack_python_solution_from_files(
            path=str(source_dir),
            spec=spec,
            name=solution_config["name"],
            definition=solution_config["definition"],
            author=solution_config["author"],
        )
    else:
        solution = pack_solution_from_files(
            path=str(source_dir),
            spec=spec,
            name=solution_config["name"],
            definition=solution_config["definition"],
            author=solution_config["author"],
        )

    # Write to output file
    if output_path is None:
        output_path = PROJECT_ROOT / "solution.json"

    output_path.write_text(solution.model_dump_json(indent=2))
    print(f"Solution packed: {output_path}")
    print(f"  Name: {solution.name}")
    print(f"  Definition: {solution.definition}")
    print(f"  Author: {solution.author}")
    print(f"  Language: {language}")

    return output_path


def main():
    """Entry point for pack_solution script."""
    import argparse

    parser = argparse.ArgumentParser(description="Pack solution files into solution.json")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path for solution.json (default: ./solution.json)"
    )
    args = parser.parse_args()

    try:
        pack_solution(args.output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
