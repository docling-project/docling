from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

RUFF_DIRECTORIES = ("docling", "tests", "docs/examples", ".github/scripts")
MYPY_DIRECTORIES = ("docling", ".github/scripts")
TOOLING_SMOKE_TRIGGER_PATHS = (
    ".github/scripts/run_pr_fast_checks.py",
    ".github/workflows/pr-fast-checks.yml",
    ".pre-commit-config.yaml",
    "pyproject.toml",
    "uv.lock",
)
SMOKE_CHECK_TARGET = ".github/scripts/run_pr_fast_checks.py"
RELATIVE_INCREASE_THRESHOLD = 0.5
ABSOLUTE_INCREASE_THRESHOLD_SECONDS = 2.0


@dataclass(slots=True)
class CheckUnit:
    name: str
    command: list[str]
    base_targets: list[str]
    head_targets: list[str]


@dataclass(slots=True)
class CommandResult:
    unit_name: str
    label: str
    targets: list[str]
    duration_seconds: float
    returncode: int
    stdout: str
    stderr: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run cheap PR-target lint and typing checks on changed files only, "
            "then compare timings against the base snapshot."
        )
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--base-ref", required=True)
    parser.add_argument("--head-ref", required=True)
    return parser.parse_args()


def run_command(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    unit_name: str,
    label: str,
    targets: list[str],
) -> CommandResult:
    start = time.perf_counter()
    completed = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    duration_seconds = time.perf_counter() - start

    return CommandResult(
        unit_name=unit_name,
        label=label,
        targets=targets,
        duration_seconds=duration_seconds,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def run_git_text(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout


def run_git_bytes(repo_root: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        check=True,
    )
    return completed.stdout


def get_changed_paths(repo_root: Path, base_ref: str, head_ref: str) -> list[str]:
    output = run_git_text(
        repo_root,
        "diff",
        "--name-only",
        "--diff-filter=ACMR",
        base_ref,
        head_ref,
    )
    return [line for line in output.splitlines() if line]


def is_python_or_notebook_file(path: str, directories: tuple[str, ...]) -> bool:
    suffix = Path(path).suffix
    if suffix not in {".py", ".ipynb"}:
        return False

    return any(
        path == directory or path.startswith(f"{directory}/")
        for directory in directories
    )


def is_mypy_target(path: str) -> bool:
    if Path(path).suffix != ".py":
        return False

    return any(
        path == directory or path.startswith(f"{directory}/")
        for directory in MYPY_DIRECTORIES
    )


def filter_existing(repo_root: Path, paths: list[str]) -> list[str]:
    return [path for path in paths if (repo_root / path).exists()]


def overlay_head_files(repo_root: Path, head_ref: str, paths: list[str]) -> None:
    for relative_path in paths:
        output_path = repo_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(
            run_git_bytes(repo_root, "show", f"{head_ref}:{relative_path}")
        )


def resolve_executable(repo_root: Path, executable_name: str) -> Path:
    venv_executable = repo_root / f".venv/bin/{executable_name}"
    if venv_executable.exists():
        return venv_executable

    system_executable = shutil.which(executable_name)
    if system_executable is None:
        raise FileNotFoundError(
            f"Could not find `{executable_name}` in .venv or on PATH."
        )

    return Path(system_executable)


def build_check_units(repo_root: Path) -> list[CheckUnit]:
    ruff_executable = resolve_executable(repo_root, "ruff")
    mypy_executable = resolve_executable(repo_root, "mypy")
    config_path = repo_root / "pyproject.toml"

    return [
        CheckUnit(
            name="ruff-format",
            command=[
                str(ruff_executable),
                "format",
                "--check",
                "--config",
                str(config_path),
            ],
            base_targets=[],
            head_targets=[],
        ),
        CheckUnit(
            name="ruff-lint",
            command=[
                str(ruff_executable),
                "check",
                "--config",
                str(config_path),
            ],
            base_targets=[],
            head_targets=[],
        ),
        CheckUnit(
            name="mypy",
            command=[
                str(mypy_executable),
                "--config-file",
                str(config_path),
                "--follow-imports",
                "skip",
                "--ignore-missing-imports",
            ],
            base_targets=[],
            head_targets=[],
        ),
    ]


def collect_targets(
    repo_root: Path, changed_paths: list[str]
) -> tuple[list[str], list[str], list[str], list[str], bool]:
    ruff_targets = [
        path
        for path in changed_paths
        if is_python_or_notebook_file(path, RUFF_DIRECTORIES)
    ]
    mypy_targets = [path for path in changed_paths if is_mypy_target(path)]

    used_tooling_smoke_targets = (
        not ruff_targets
        and not mypy_targets
        and any(path in TOOLING_SMOKE_TRIGGER_PATHS for path in changed_paths)
    )
    if used_tooling_smoke_targets:
        ruff_targets = [SMOKE_CHECK_TARGET]
        mypy_targets = [SMOKE_CHECK_TARGET]

    existing_ruff_targets = filter_existing(repo_root, ruff_targets)
    existing_mypy_targets = filter_existing(repo_root, mypy_targets)

    return (
        ruff_targets,
        mypy_targets,
        existing_ruff_targets,
        existing_mypy_targets,
        used_tooling_smoke_targets,
    )


def populate_targets(
    units: list[CheckUnit],
    *,
    ruff_targets: list[str],
    mypy_targets: list[str],
    existing_ruff_targets: list[str],
    existing_mypy_targets: list[str],
) -> None:
    for unit in units:
        if unit.name.startswith("ruff"):
            unit.base_targets = existing_ruff_targets
            unit.head_targets = ruff_targets
        else:
            unit.base_targets = existing_mypy_targets
            unit.head_targets = mypy_targets


def render_target_list(paths: list[str]) -> str:
    if not paths:
        return "_none_"

    return "<br>".join(f"`{path}`" for path in paths)


def is_significant_regression(
    base: CommandResult | None, head: CommandResult | None
) -> bool:
    if base is None or head is None:
        return False
    if base.returncode != 0 or head.returncode != 0:
        return False
    if base.targets != head.targets:
        return False

    delta_seconds = head.duration_seconds - base.duration_seconds
    if delta_seconds < ABSOLUTE_INCREASE_THRESHOLD_SECONDS:
        return False

    if base.duration_seconds <= 0:
        return True

    relative_increase = delta_seconds / base.duration_seconds
    return relative_increase >= RELATIVE_INCREASE_THRESHOLD


def format_result_cell(result: CommandResult | None) -> str:
    if result is None:
        return "_skipped_"

    status = "passed" if result.returncode == 0 else f"failed ({result.returncode})"
    return f"{result.duration_seconds:.2f}s<br>{status}"


def append_summary(summary_path: Path | None, text: str) -> None:
    if summary_path is None:
        return

    with summary_path.open("a", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def log_result(result: CommandResult) -> None:
    print(
        f"[{result.label}] {result.unit_name} on {len(result.targets)} target(s) "
        f"finished in {result.duration_seconds:.2f}s with exit code {result.returncode}."
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    summary_file = Path(summary_path) if summary_path else None

    changed_paths = get_changed_paths(repo_root, args.base_ref, args.head_ref)
    (
        ruff_targets,
        mypy_targets,
        existing_ruff_targets,
        existing_mypy_targets,
        used_tooling_smoke_targets,
    ) = collect_targets(repo_root, changed_paths)

    all_head_targets = sorted(set(ruff_targets).union(mypy_targets))

    append_summary(summary_file, "## PR Fast Checks\n")
    append_summary(summary_file, "### Changed paths\n")
    append_summary(
        summary_file,
        "\n".join(f"- `{path}`" for path in changed_paths) + "\n",
    )
    append_summary(summary_file, "\n### Check targets\n")
    if used_tooling_smoke_targets:
        append_summary(
            summary_file,
            "- Tooling-only change detected; using the smoke target set.\n",
        )
    if all_head_targets:
        append_summary(
            summary_file,
            "\n".join(f"- `{target}`" for target in all_head_targets) + "\n",
        )
    else:
        append_summary(
            summary_file,
            "- No Python or notebook targets matched the fast-check rules.\n",
        )
        print("No Python or notebook targets matched the fast-check rules.")
        return 0

    units = build_check_units(repo_root)
    populate_targets(
        units,
        ruff_targets=ruff_targets,
        mypy_targets=mypy_targets,
        existing_ruff_targets=existing_ruff_targets,
        existing_mypy_targets=existing_mypy_targets,
    )

    env = os.environ.copy()
    env.setdefault("MYPY_FORCE_COLOR", "0")
    env.setdefault("PYTHONUTF8", "1")

    base_results: dict[str, CommandResult | None] = {}
    head_results: dict[str, CommandResult | None] = {}

    for unit in units:
        if unit.base_targets:
            base_result = run_command(
                [*unit.command, *unit.base_targets],
                cwd=repo_root,
                env=env,
                unit_name=unit.name,
                label="base",
                targets=unit.base_targets,
            )
            log_result(base_result)
            base_results[unit.name] = base_result
        else:
            base_results[unit.name] = None

    overlay_head_files(repo_root, args.head_ref, all_head_targets)

    head_failures: list[CommandResult] = []
    for unit in units:
        if unit.head_targets:
            head_result = run_command(
                [*unit.command, *unit.head_targets],
                cwd=repo_root,
                env=env,
                unit_name=unit.name,
                label="head",
                targets=unit.head_targets,
            )
            log_result(head_result)
            head_results[unit.name] = head_result
            if head_result.returncode != 0:
                head_failures.append(head_result)
        else:
            head_results[unit.name] = None

    append_summary(summary_file, "\n### Timing comparison\n")
    append_summary(
        summary_file,
        "| Unit | Base targets | Head targets | Base | Head | Delta | Timing note |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n",
    )

    regressions: list[str] = []
    for unit in units:
        summary_base_result = base_results[unit.name]
        summary_head_result = head_results[unit.name]

        delta_cell = "_n/a_"
        note = ""
        if summary_base_result is not None and summary_head_result is not None:
            delta_seconds = (
                summary_head_result.duration_seconds
                - summary_base_result.duration_seconds
            )
            delta_cell = f"{delta_seconds:+.2f}s"
            if summary_base_result.targets != summary_head_result.targets:
                note = "target set changed"
            elif is_significant_regression(summary_base_result, summary_head_result):
                note = "significant increase"
                regressions.append(
                    f"`{unit.name}` increased from "
                    f"{summary_base_result.duration_seconds:.2f}s to "
                    f"{summary_head_result.duration_seconds:.2f}s on the same "
                    f"target set."
                )
            else:
                note = "within threshold"

        append_summary(
            summary_file,
            "| "
            + " | ".join(
                [
                    unit.name,
                    render_target_list(unit.base_targets),
                    render_target_list(unit.head_targets),
                    format_result_cell(summary_base_result),
                    format_result_cell(summary_head_result),
                    delta_cell,
                    note or "_n/a_",
                ]
            )
            + " |\n",
        )

    if regressions:
        append_summary(summary_file, "\n### Timing regressions\n")
        append_summary(
            summary_file,
            "\n".join(f"- {regression}" for regression in regressions) + "\n",
        )
        for regression in regressions:
            print(f"::warning::{regression}")
    else:
        append_summary(
            summary_file,
            "\n### Timing regressions\n- None detected on unchanged target sets.\n",
        )

    if head_failures:
        failing_units = ", ".join(failure.unit_name for failure in head_failures)
        print(f"Head checks failed: {failing_units}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
