from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_fast_checks_module() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[1] / ".github/scripts/run_pr_fast_checks.py"
    )
    spec = importlib.util.spec_from_file_location("run_pr_fast_checks", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fast_checks = load_fast_checks_module()


def write_file(repo_root: Path, relative_path: str, content: str = "pass\n") -> None:
    file_path = repo_root / relative_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")


def test_collect_targets_limits_scope_to_supported_paths(tmp_path: Path) -> None:
    write_file(tmp_path, "docling/existing_module.py")
    write_file(tmp_path, "tests/test_fast_checks.py")
    write_file(tmp_path, "docs/examples/demo.ipynb", "{}\n")
    write_file(tmp_path, ".github/scripts/helper.py")

    changed_paths = [
        "README.md",
        "docling/existing_module.py",
        "docling/new_module.py",
        "tests/test_fast_checks.py",
        "docs/examples/demo.ipynb",
        ".github/scripts/helper.py",
        "other/outside_scope.py",
    ]

    (
        ruff_targets,
        mypy_targets,
        existing_ruff_targets,
        existing_mypy_targets,
        used_tooling_smoke_targets,
    ) = fast_checks.collect_targets(tmp_path, changed_paths)

    assert ruff_targets == [
        "docling/existing_module.py",
        "docling/new_module.py",
        "tests/test_fast_checks.py",
        "docs/examples/demo.ipynb",
        ".github/scripts/helper.py",
    ]
    assert mypy_targets == [
        "docling/existing_module.py",
        "docling/new_module.py",
        ".github/scripts/helper.py",
    ]
    assert existing_ruff_targets == [
        "docling/existing_module.py",
        "tests/test_fast_checks.py",
        "docs/examples/demo.ipynb",
        ".github/scripts/helper.py",
    ]
    assert existing_mypy_targets == [
        "docling/existing_module.py",
        ".github/scripts/helper.py",
    ]
    assert used_tooling_smoke_targets is False


def test_collect_targets_uses_smoke_target_for_tooling_only_changes(
    tmp_path: Path,
) -> None:
    write_file(tmp_path, fast_checks.SMOKE_CHECK_TARGET)

    (
        ruff_targets,
        mypy_targets,
        existing_ruff_targets,
        existing_mypy_targets,
        used_tooling_smoke_targets,
    ) = fast_checks.collect_targets(tmp_path, ["pyproject.toml"])

    assert ruff_targets == [fast_checks.SMOKE_CHECK_TARGET]
    assert mypy_targets == [fast_checks.SMOKE_CHECK_TARGET]
    assert existing_ruff_targets == [fast_checks.SMOKE_CHECK_TARGET]
    assert existing_mypy_targets == [fast_checks.SMOKE_CHECK_TARGET]
    assert used_tooling_smoke_targets is True


def test_collect_targets_skips_unrelated_changes(tmp_path: Path) -> None:
    (
        ruff_targets,
        mypy_targets,
        existing_ruff_targets,
        existing_mypy_targets,
        used_tooling_smoke_targets,
    ) = fast_checks.collect_targets(tmp_path, ["README.md"])

    assert ruff_targets == []
    assert mypy_targets == []
    assert existing_ruff_targets == []
    assert existing_mypy_targets == []
    assert used_tooling_smoke_targets is False


def test_build_check_units_uses_fast_mypy_flags(monkeypatch) -> None:
    monkeypatch.setattr(
        fast_checks,
        "resolve_executable",
        lambda repo_root, executable_name: Path(f"/tmp/{executable_name}"),
    )

    units = fast_checks.build_check_units(Path("/tmp/repo"))
    mypy_unit = next(unit for unit in units if unit.name == "mypy")

    assert mypy_unit.command == [
        "/tmp/mypy",
        "--config-file",
        "/tmp/repo/pyproject.toml",
        "--follow-imports",
        "skip",
        "--ignore-missing-imports",
    ]


def test_significant_regression_requires_same_successful_target_set() -> None:
    base = fast_checks.CommandResult(
        unit_name="mypy",
        label="base",
        targets=["docling/module.py"],
        duration_seconds=3.0,
        returncode=0,
        stdout="",
        stderr="",
    )
    head = fast_checks.CommandResult(
        unit_name="mypy",
        label="head",
        targets=["docling/module.py"],
        duration_seconds=5.2,
        returncode=0,
        stdout="",
        stderr="",
    )
    different_targets = fast_checks.CommandResult(
        unit_name="mypy",
        label="head",
        targets=["docling/other.py"],
        duration_seconds=5.2,
        returncode=0,
        stdout="",
        stderr="",
    )

    assert fast_checks.is_significant_regression(base, head) is True
    assert fast_checks.is_significant_regression(base, different_targets) is False
