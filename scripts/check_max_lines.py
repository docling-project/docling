#!/usr/bin/env python3
"""Pre-commit hook to enforce a maximum number of lines per file."""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
from pathlib import Path

CHECKED_EXTENSIONS = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".rs",
    ".json",
    ".sql",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
}

SKIP_DIRS = {
    ".cache",
    ".git",
    ".hypothesis",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "build",
    "dist",
    "node_modules",
    "__pycache__",
}

DEFAULT_MAX_LINES = 1000
DEFAULT_IGNORE_FILE = ".max-lines-ignore"


def load_ignore_patterns(ignore_file: Path) -> tuple[list[str], list[str]]:
    """Load silent and warning-only glob patterns from the ignore file."""
    silent_patterns: list[str] = []
    warn_patterns: list[str] = []

    if not ignore_file.exists():
        return silent_patterns, warn_patterns

    current_section = "silent"
    for line in ignore_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        section = stripped.lower()
        if section == "[silent]":
            current_section = "silent"
            continue
        if section == "[warn]":
            current_section = "warn"
            continue

        if current_section == "warn":
            warn_patterns.append(stripped)
        else:
            silent_patterns.append(stripped)

    return silent_patterns, warn_patterns


def is_ignored(file_path: str, patterns: list[str]) -> bool:
    """Return whether a relative file path matches any ignore pattern."""
    return any(
        fnmatch.fnmatch(file_path, pattern)
        or fnmatch.fnmatch(Path(file_path).name, pattern)
        for pattern in patterns
    )


def count_lines(file_path: Path) -> int:
    """Count text lines while tolerating encoding errors in fixtures."""
    try:
        return len(file_path.read_text(encoding="utf-8", errors="replace").splitlines())
    except OSError:
        return 0


def find_files(repo_root: Path) -> list[Path]:
    """Walk the repository and collect checked file types."""
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix in CHECKED_EXTENSIONS:
                files.append(path)
    return files


def filter_candidate_files(repo_root: Path, file_args: list[str]) -> list[Path]:
    """Resolve pre-commit file arguments to checked files under the repo root."""
    files: list[Path] = []
    for raw_path in file_args:
        path = Path(raw_path)
        if not path.is_absolute():
            path = (repo_root / path).resolve()

        if not path.is_file():
            continue

        try:
            rel_parts = path.relative_to(repo_root).parts
        except ValueError:
            continue

        if any(part in SKIP_DIRS for part in rel_parts):
            continue
        if path.suffix not in CHECKED_EXTENSIONS:
            continue

        files.append(path)

    return list(dict.fromkeys(files))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-lines",
        type=int,
        default=DEFAULT_MAX_LINES,
        help=f"Maximum allowed lines per file (default: {DEFAULT_MAX_LINES}).",
    )
    parser.add_argument(
        "--ignore-file",
        default=DEFAULT_IGNORE_FILE,
        help=f"Path to ignore file (default: {DEFAULT_IGNORE_FILE}).",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Optional file paths to check. Defaults to scanning the repo.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    ignore_file = repo_root / args.ignore_file
    silent_patterns, warn_patterns = load_ignore_patterns(ignore_file)
    files = (
        filter_candidate_files(repo_root, args.files)
        if args.files
        else find_files(repo_root)
    )

    violations: list[tuple[str, int]] = []
    warnings: list[tuple[str, int]] = []

    for file_path in sorted(files):
        rel_path = file_path.relative_to(repo_root).as_posix()
        line_count = count_lines(file_path)
        if line_count <= args.max_lines:
            continue

        if is_ignored(rel_path, warn_patterns):
            warnings.append((rel_path, line_count))
        elif not is_ignored(rel_path, silent_patterns):
            violations.append((rel_path, line_count))

    if warnings:
        print(f"Ignored files exceeding {args.max_lines} line limit (TODO refactor):")
        for path, line_count in warnings:
            print(f"  WARN: {path} has {line_count} lines")
        print()

    if violations:
        print(f"Files exceeding {args.max_lines} line limit:")
        for path, line_count in violations:
            print(f"  FAIL: {path} has {line_count} lines")
        print(f"\n{len(violations)} file(s) over the limit.")
        print(f"To exempt existing debt, add patterns to {args.ignore_file}.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
