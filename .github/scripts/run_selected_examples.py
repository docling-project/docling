#!/usr/bin/env python3
"""Run a filtered set of example scripts from docs/examples."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run selected Docling example scripts from docs/examples.",
    )
    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=Path("docs/examples"),
        help="Directory containing the example scripts.",
    )
    parser.add_argument(
        "--changed-files-json",
        default="[]",
        help="JSON array of changed file paths relative to the repository root.",
    )
    parser.add_argument(
        "--include-pattern",
        default="",
        help="Regex matched against the example filename. Empty means include all.",
    )
    parser.add_argument(
        "--exclude-pattern",
        default="",
        help="Regex matched against the example filename. Empty means exclude none.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("runtime_summary.log"),
        help="Path to the runtime summary output file.",
    )
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=Path("scratch"),
        help="Scratch directory created before running examples.",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run all eligible examples instead of only changed files.",
    )
    return parser.parse_args()


def compile_pattern(pattern: str) -> re.Pattern[str] | None:
    if not pattern:
        return None
    return re.compile(pattern)


def load_changed_files(raw_json: str) -> list[Path]:
    loaded = json.loads(raw_json)
    if not isinstance(loaded, list):
        msg = "--changed-files-json must decode to a JSON list."
        raise ValueError(msg)
    return [Path(str(item)) for item in loaded]


def select_candidates(
    examples_dir: Path,
    run_all: bool,
    changed_files: list[Path],
) -> list[Path]:
    if run_all:
        return sorted(path for path in examples_dir.glob("*.py") if path.is_file())

    return sorted(
        path
        for path in changed_files
        if path.suffix == ".py" and path.parent.as_posix() == examples_dir.as_posix()
    )


def filter_examples(
    candidates: list[Path],
    include_pattern: re.Pattern[str] | None,
    exclude_pattern: re.Pattern[str] | None,
) -> list[Path]:
    selected: list[Path] = []
    for path in candidates:
        name = path.name
        if include_pattern is not None and not include_pattern.match(name):
            continue
        if exclude_pattern is not None and exclude_pattern.match(name):
            continue
        selected.append(path)
    return selected


def write_summary(summary_path: Path, lines: list[str]) -> None:
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_examples(selected: list[Path], scratch_dir: Path, summary_path: Path) -> int:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    summary_lines = ["--- Example Runtimes ---"]

    if not selected:
        message = "No runnable example scripts were selected."
        print(message)
        summary_lines.append(message)
        write_summary(summary_path, summary_lines)
        return 0

    for path in selected:
        print(f"--- Running example {path.name} ---")
        start_time = time.perf_counter()
        completed = subprocess.run(
            ["uv", "run", "--no-sync", "python", str(path)],
            check=False,
        )
        duration = int(time.perf_counter() - start_time)
        print(f"Finished in {duration}s.")
        summary_lines.append(f"{path.name}: {duration}s")
        write_summary(summary_path, summary_lines)
        if completed.returncode != 0:
            return completed.returncode

    print()
    print("===================================")
    print("       Final Runtime Summary       ")
    print("===================================")
    print(summary_path.read_text(encoding="utf-8"), end="")
    print("===================================")
    return 0


def main() -> int:
    args = parse_args()
    changed_files = load_changed_files(args.changed_files_json)
    include_pattern = compile_pattern(args.include_pattern)
    exclude_pattern = compile_pattern(args.exclude_pattern)
    candidates = select_candidates(args.examples_dir, args.run_all, changed_files)
    selected = filter_examples(candidates, include_pattern, exclude_pattern)
    return run_examples(selected, args.scratch_dir, args.summary_path)


if __name__ == "__main__":
    raise SystemExit(main())
