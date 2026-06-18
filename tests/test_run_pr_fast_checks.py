from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


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


def run_git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


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
        ty_targets,
        existing_ruff_targets,
        existing_ty_targets,
        used_tooling_smoke_targets,
    ) = fast_checks.collect_targets(tmp_path, changed_paths)

    assert ruff_targets == [
        "docling/existing_module.py",
        "docling/new_module.py",
        "tests/test_fast_checks.py",
        "docs/examples/demo.ipynb",
        ".github/scripts/helper.py",
    ]
    assert ty_targets == [
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
    assert existing_ty_targets == [
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
        ty_targets,
        existing_ruff_targets,
        existing_ty_targets,
        used_tooling_smoke_targets,
    ) = fast_checks.collect_targets(tmp_path, ["pyproject.toml"])

    assert ruff_targets == [fast_checks.SMOKE_CHECK_TARGET]
    assert ty_targets == [fast_checks.SMOKE_CHECK_TARGET]
    assert existing_ruff_targets == [fast_checks.SMOKE_CHECK_TARGET]
    assert existing_ty_targets == [fast_checks.SMOKE_CHECK_TARGET]
    assert used_tooling_smoke_targets is True


def test_collect_targets_skips_unrelated_changes(tmp_path: Path) -> None:
    (
        ruff_targets,
        ty_targets,
        existing_ruff_targets,
        existing_ty_targets,
        used_tooling_smoke_targets,
    ) = fast_checks.collect_targets(tmp_path, ["README.md"])

    assert ruff_targets == []
    assert ty_targets == []
    assert existing_ruff_targets == []
    assert existing_ty_targets == []
    assert used_tooling_smoke_targets is False


def test_build_check_units_uses_ty_check(monkeypatch) -> None:
    monkeypatch.setattr(
        fast_checks,
        "resolve_executable",
        lambda repo_root, executable_name: Path(f"/tmp/{executable_name}"),
    )

    units = fast_checks.build_check_units(Path("/tmp/repo"))
    ty_unit = next(unit for unit in units if unit.name == "ty")

    assert ty_unit.command == [
        "/tmp/ty",
        "check",
        "--project",
        "/tmp/repo",
    ]


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="PR fast checks run on Ubuntu CI.",
)
def test_git_helpers_accept_synthetic_merge_tree(tmp_path: Path) -> None:
    run_git(tmp_path, "init")
    run_git(tmp_path, "config", "user.name", "Test User")
    run_git(tmp_path, "config", "user.email", "test@example.com")

    write_file(tmp_path, "docling/module.py", "VALUE = 1\n")
    run_git(tmp_path, "add", ".")
    run_git(tmp_path, "commit", "-m", "base")
    base_ref = run_git(tmp_path, "rev-parse", "HEAD")

    run_git(tmp_path, "switch", "-c", "pr")
    write_file(tmp_path, "docling/module.py", "VALUE = 2\n")
    write_file(tmp_path, "tests/test_module.py", "def test_value():\n    pass\n")
    run_git(tmp_path, "add", ".")
    run_git(tmp_path, "commit", "-m", "pr")
    head_ref = run_git(tmp_path, "rev-parse", "HEAD")

    merge_tree = run_git(
        tmp_path,
        "merge-tree",
        "--write-tree",
        "--merge-base",
        base_ref,
        base_ref,
        head_ref,
    )

    assert fast_checks.get_changed_paths(tmp_path, base_ref, merge_tree) == [
        "docling/module.py",
        "tests/test_module.py",
    ]

    run_git(tmp_path, "switch", "--detach", base_ref)
    fast_checks.overlay_head_files(
        tmp_path,
        merge_tree,
        ["docling/module.py", "tests/test_module.py"],
    )

    assert (tmp_path / "docling/module.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert (tmp_path / "tests/test_module.py").exists()


def test_log_result_suppresses_success_output(capsys) -> None:
    result = fast_checks.CommandResult(
        unit_name="ty",
        label="base",
        targets=["docling/module.py"],
        duration_seconds=1.0,
        returncode=0,
        stdout="allowed warning\n",
        stderr="",
    )

    fast_checks.log_result(result)

    captured = capsys.readouterr()
    assert "exit code 0" in captured.out
    assert "allowed warning" not in captured.out


def test_log_result_prints_failure_output(capsys) -> None:
    result = fast_checks.CommandResult(
        unit_name="ty",
        label="head",
        targets=["docling/module.py"],
        duration_seconds=1.0,
        returncode=1,
        stdout="type error\n",
        stderr="",
    )

    fast_checks.log_result(result)

    captured = capsys.readouterr()
    assert "exit code 1" in captured.out
    assert "type error" in captured.out


def test_significant_regression_requires_same_successful_target_set() -> None:
    base = fast_checks.CommandResult(
        unit_name="ty",
        label="base",
        targets=["docling/module.py"],
        duration_seconds=3.0,
        returncode=0,
        stdout="",
        stderr="",
    )
    head = fast_checks.CommandResult(
        unit_name="ty",
        label="head",
        targets=["docling/module.py"],
        duration_seconds=5.2,
        returncode=0,
        stdout="",
        stderr="",
    )
    different_targets = fast_checks.CommandResult(
        unit_name="ty",
        label="head",
        targets=["docling/other.py"],
        duration_seconds=5.2,
        returncode=0,
        stdout="",
        stderr="",
    )

    assert fast_checks.is_significant_regression(base, head) is True
    assert fast_checks.is_significant_regression(base, different_targets) is False
