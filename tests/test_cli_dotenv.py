# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for the `.env` loading of the `docling` CLI.

The service client is faked, so these tests need no docling-serve instance.
"""

import os

import pytest
from typer.testing import CliRunner

from docling.cli.main import app

runner = CliRunner(env={"TERM": "dumb"})


class _HealthStopped(Exception):
    pass


class _RecordingClient:
    """Records the constructor arguments and stops the command at the health check."""

    kwargs: dict | None = None

    def __init__(self, **kwargs):
        _RecordingClient.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def health(self):
        raise _HealthStopped("stop after the client is built")


@pytest.fixture
def _record_client(monkeypatch):
    _RecordingClient.kwargs = None
    monkeypatch.setattr("docling.cli.remote.DoclingServiceClient", _RecordingClient)
    return _RecordingClient


def _clear_env(monkeypatch, *names: str) -> None:
    # setenv first so monkeypatch restores the absent state after the test,
    # also for keys that the CLI sets.
    for name in names:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


def test_remote_reads_only_service_settings_from_dotenv(
    tmp_path, monkeypatch, _record_client
):
    _clear_env(
        monkeypatch,
        "DOCLING_SERVICE_URL",
        "DOCLING_SERVICE_API_KEY",
        "DOCLING_TEST_UNRELATED",
    )
    project = tmp_path / "project"
    workdir = project / "sub"
    workdir.mkdir(parents=True)
    (project / ".env").write_text(
        "DOCLING_SERVICE_URL=https://dotenv.example.com\n"
        "DOCLING_SERVICE_API_KEY=dotenv-key\n"
        "DOCLING_TEST_UNRELATED=1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DOCLING_SERVICE_API_KEY", "env-key")
    monkeypatch.chdir(workdir)
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-1.4")

    result = runner.invoke(
        app, ["convert-remote", str(source), "--output", str(tmp_path / "out")]
    )

    # The fake stops the command at the health check, after the client is built.
    assert result.exit_code == 1, result.output
    assert _record_client.kwargs is not None
    assert _record_client.kwargs["url"] == "https://dotenv.example.com"
    # A real environment variable beats the .env value.
    assert _record_client.kwargs["api_key"] == "env-key"
    assert "DOCLING_TEST_UNRELATED" not in os.environ


def test_local_convert_does_not_read_dotenv(tmp_path, monkeypatch):
    _clear_env(monkeypatch, "DOCLING_SERVICE_URL", "DOCLING_TEST_UNRELATED")
    (tmp_path / ".env").write_text(
        "DOCLING_SERVICE_URL=https://dotenv.example.com\nDOCLING_TEST_UNRELATED=1\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["convert", "missing.pdf", "--output", "out"])

    assert result.exit_code == 1, result.output
    assert "DOCLING_SERVICE_URL" not in os.environ
    assert "DOCLING_TEST_UNRELATED" not in os.environ
