from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path

ML_MARKERS = ("ml_ocr", "ml_pdf_model", "ml_vlm", "ml_asr")
SUITE_MARKERS = {
    "ocr": "ml_ocr",
    "pdf-model": "ml_pdf_model",
    "vlm": "ml_vlm",
    "asr": "ml_asr",
}
MARKER_SUITES = {marker: suite for suite, marker in SUITE_MARKERS.items()}


def parse_bool(value: str) -> bool:
    return value.lower() == "true"


def is_pytest_mark_attribute(node: ast.AST, marker: str) -> bool:
    if not isinstance(node, ast.Attribute) or node.attr != marker:
        return False
    if not isinstance(node.value, ast.Attribute) or node.value.attr != "mark":
        return False
    return isinstance(node.value.value, ast.Name) and node.value.value.id == "pytest"


def markers_in_node(node: ast.AST) -> set[str]:
    markers: set[str] = set()
    for child in ast.walk(node):
        for marker in ML_MARKERS:
            if is_pytest_mark_attribute(child, marker):
                markers.add(marker)
    return markers


def module_level_ml_markers(tree: ast.Module) -> set[str]:
    markers: set[str] = set()
    for statement in tree.body:
        value: ast.AST | None = None
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "pytestmark"
            for target in statement.targets
        ):
            value = statement.value
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == "pytestmark"
        ):
            value = statement.value

        if value is not None:
            markers.update(markers_in_node(value))

    return markers


def detect_ml_markers(path: Path) -> set[str]:
    if not path.exists() or path.suffix != ".py":
        return set()

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    all_markers = markers_in_node(tree)
    module_markers = module_level_ml_markers(tree)
    if all_markers != module_markers:
        raise ValueError(
            f"{path}: ML pytest markers must be declared with module-level "
            "`pytestmark` so CI can select whole test modules."
        )
    return module_markers


def discover_test_markers(repo_root: Path) -> dict[str, list[Path]]:
    discovered: dict[str, list[Path]] = {marker: [] for marker in ML_MARKERS}
    tests_dir = repo_root / "tests"
    if not tests_dir.exists():
        return discovered

    for path in sorted(tests_dir.rglob("*.py")):
        markers = detect_ml_markers(path)
        for marker in markers:
            discovered[marker].append(path.relative_to(repo_root))

    return discovered


def build_ml_suites(*, run_all_ml: bool) -> list[str]:
    if not run_all_ml:
        return []

    return [MARKER_SUITES[marker] for marker in ML_MARKERS]


def write_github_output(name: str, value: str) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path is None:
        print(f"{name}={value}")
        return

    with Path(output_path).open("a", encoding="utf-8") as output_file:
        output_file.write(f"{name}={value}\n")


def print_paths(paths: list[Path]) -> None:
    for path in paths:
        print(path.as_posix())


def run_matrix(args: argparse.Namespace) -> None:
    suites = build_ml_suites(run_all_ml=parse_bool(args.run_all_ml))
    write_github_output("ml_suites", json.dumps(suites, separators=(",", ":")))


def run_core_ignore_args(args: argparse.Namespace) -> None:
    discovered = discover_test_markers(args.repo_root)
    marked_paths = sorted({path for paths in discovered.values() for path in paths})
    for path in marked_paths:
        print(f"--ignore={path.as_posix()}")


def run_suite_args(args: argparse.Namespace) -> None:
    if args.suite not in SUITE_MARKERS:
        raise ValueError(f"Unknown ML suite: {args.suite}")

    discovered = discover_test_markers(args.repo_root)
    print_paths(discovered[SUITE_MARKERS[args.suite]])


def run_suite_marker(args: argparse.Namespace) -> None:
    if args.suite not in SUITE_MARKERS:
        raise ValueError(f"Unknown ML suite: {args.suite}")

    print(SUITE_MARKERS[args.suite])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select pytest modules for Docling's marker-based CI lanes."
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    subparsers = parser.add_subparsers(dest="command", required=True)

    matrix_parser = subparsers.add_parser("matrix")
    matrix_parser.add_argument("--run-all-ml", default="false")
    matrix_parser.set_defaults(func=run_matrix)

    core_parser = subparsers.add_parser("core-ignore-args")
    core_parser.set_defaults(func=run_core_ignore_args)

    suite_parser = subparsers.add_parser("suite-args")
    suite_parser.add_argument("suite")
    suite_parser.set_defaults(func=run_suite_args)

    marker_parser = subparsers.add_parser("suite-marker")
    marker_parser.add_argument("suite")
    marker_parser.set_defaults(func=run_suite_marker)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
