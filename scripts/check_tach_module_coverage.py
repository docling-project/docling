#!/usr/bin/env python3
"""Fail when Python modules are not covered by a Tach module."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PACKAGE_ROOT = "docling"
TACH_CONFIG = Path("tach.toml")
SKIP_DIRS = {"__pycache__"}


def load_toml(path: Path) -> dict[str, Any]:
    """Load TOML with a small fallback for older local Python installs."""
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib
        except ModuleNotFoundError:
            return {"modules": [{"path": p} for p in parse_module_paths(path)]}

    with path.open("rb") as stream:
        return tomllib.load(stream)


def parse_module_paths(path: Path) -> list[str]:
    """Extract [[modules]].path values when no TOML parser is available."""
    module_paths: list[str] = []
    in_module = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "[[modules]]":
            in_module = True
            continue
        if line.startswith("[["):
            in_module = False
            continue
        if in_module and line.startswith("path"):
            _, value = line.split("=", maxsplit=1)
            module_paths.append(value.strip().strip('"'))
    return module_paths


def module_name(path: Path) -> str:
    parts = list(path.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def iter_python_modules(package_root: Path) -> list[str]:
    modules: set[str] = set()
    for path in package_root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        name = module_name(path)
        if name != PACKAGE_ROOT:
            modules.add(name)
    return sorted(modules)


def is_covered(module: str, tach_modules: list[str]) -> bool:
    return any(
        module == tach_module or module.startswith(f"{tach_module}.")
        for tach_module in tach_modules
    )


def module_exists(module: str) -> bool:
    path = Path(*module.split("."))
    return path.with_suffix(".py").is_file() or (path / "__init__.py").is_file()


def main() -> int:
    config = load_toml(TACH_CONFIG)
    tach_modules = sorted(module["path"] for module in config.get("modules", []))

    duplicate_modules = sorted(
        module for module in set(tach_modules) if tach_modules.count(module) > 1
    )
    stale_modules = [module for module in tach_modules if not module_exists(module)]
    uncovered_modules = [
        module
        for module in iter_python_modules(Path(PACKAGE_ROOT))
        if not is_covered(module, tach_modules)
    ]

    if not duplicate_modules and not stale_modules and not uncovered_modules:
        return 0

    print("Tach module coverage check failed.")
    if duplicate_modules:
        print("\nDuplicate [[modules]] entries:")
        for module in duplicate_modules:
            print(f"  - {module}")
    if stale_modules:
        print("\nConfigured Tach modules without a matching Python module/package:")
        for module in stale_modules:
            print(f"  - {module}")
    if uncovered_modules:
        print("\nPython modules not covered by any Tach [[modules]].path prefix:")
        for module in uncovered_modules:
            print(f"  - {module}")
        print(
            "\nAdd a Tach module entry in tach.toml, or move the code under an "
            "existing Tach module deliberately."
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
