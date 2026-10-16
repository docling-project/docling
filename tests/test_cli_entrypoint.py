# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for the ``docling`` console entry point and its ``--version`` fast path."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from click.utils import strip_ansi
from typer.testing import CliRunner

import docling
from docling.cli import entrypoint
from docling.cli.main import app

REPO_ROOT = Path(__file__).resolve().parent.parent

runner = CliRunner()

_FULL_LABELS = (
    "Docling version:",
    "Docling Core version:",
    "Docling IBM Models version:",
    "Docling Parse version:",
    "Python:",
    "Platform:",
)


def _run_entrypoint(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], *args: str
) -> str:
    monkeypatch.setattr(sys, "argv", ["docling", *args])
    entrypoint.main()
    return capsys.readouterr().out


@pytest.mark.parametrize(
    "args",
    [
        ["--version"],
        ["--version", "--full"],
        ["--full", "--version"],
        ["convert", "--version", "--full"],
    ],
)
def test_version_fast_path_matches_typer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], args: list[str]
) -> None:
    """The fast path must print exactly what the Typer app prints."""
    fast_output = _run_entrypoint(monkeypatch, capsys, *args)

    result = runner.invoke(app, args)

    assert result.exit_code == 0
    assert fast_output == result.output


def test_version_is_one_line_and_full_adds_components(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    short_output = _run_entrypoint(monkeypatch, capsys, "--version")
    full_output = _run_entrypoint(monkeypatch, capsys, "--version", "--full")

    assert short_output == f"Docling version: {docling.__version__}\n"
    full_lines = full_output.splitlines()
    assert len(full_lines) == len(_FULL_LABELS)
    for line, label in zip(full_lines, _FULL_LABELS):
        assert line.startswith(label)


def test_version_fast_path_skips_conversion_stack() -> None:
    """``docling --version`` must not import the CLI app or the model stack.

    A fresh subprocess is needed: other tests already imported them here.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "import sys\n"
        "sys.argv = ['docling', '--version', '--full']\n"
        "from docling.cli.entrypoint import main\n"
        "main()\n"
        "heavy = ('docling.cli.main', 'torch', 'transformers')\n"
        "print('LOADED:', [m for m in heavy if m in sys.modules])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "LOADED: []" in result.stdout


def test_entrypoint_passes_other_arguments_to_typer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        _run_entrypoint(monkeypatch, capsys, "--help")

    assert exc_info.value.code == 0
    assert "convert" in strip_ansi(capsys.readouterr().out)


def test_full_without_version_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "doc.md"
    source.write_text("# Title\n", encoding="utf-8")

    result = runner.invoke(app, [str(source), "--full"])

    assert result.exit_code == 2
    assert "--full requires --version." in strip_ansi(result.output)
