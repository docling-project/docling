# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for the env-file options of `docling convert-remote`.

The service client is faked, so these tests need no docling-serve instance.
"""

import os

import pytest
from typer.testing import CliRunner

from docling.cli.main import app

runner = CliRunner(env={"TERM": "dumb", "COLUMNS": "200"})


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
def client(monkeypatch):
    _RecordingClient.kwargs = None
    monkeypatch.setattr("docling.cli.remote.DoclingServiceClient", _RecordingClient)
    for name in ("DOCLING_SERVICE_URL", "DOCLING_SERVICE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return _RecordingClient


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "report.pdf"
    path.write_bytes(b"%PDF-1.4")
    return path


def _remote(source, *args: str):
    return runner.invoke(app, ["convert-remote", str(source), *args])


def _text(result) -> str:
    # Rich wraps messages at the console width; compare without line breaks.
    return " ".join(result.output.split())


def test_dotenv_is_not_read_without_an_option(tmp_path, monkeypatch, client, source):
    (tmp_path / ".env").write_text(
        "DOCLING_SERVICE_URL=https://dotenv.example.com\n", encoding="utf-8"
    )
    workdir = tmp_path / "sub"
    workdir.mkdir()
    for cwd in (tmp_path, workdir):
        monkeypatch.chdir(cwd)
        result = _remote(source)
        assert result.exit_code == 2, result.output
        assert "No service URL" in _text(result)
    assert client.kwargs is None


def test_dotenv_reads_only_the_working_directory(tmp_path, monkeypatch, client, source):
    (tmp_path / ".env").write_text(
        "DOCLING_SERVICE_URL=https://parent.example.com\n", encoding="utf-8"
    )
    workdir = tmp_path / "sub"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    result = _remote(source, "--dotenv")

    assert result.exit_code == 2, result.output
    assert "no .env file in this directory" in _text(result)
    assert client.kwargs is None


def test_env_file_fills_unset_settings_and_names_its_source(
    tmp_path, monkeypatch, client, source
):
    env_file = tmp_path / "service.env"
    env_file.write_text(
        "DOCLING_SERVICE_URL=https://file.example.com\n"
        "DOCLING_SERVICE_API_KEY=file-key\n"
        "DOCLING_LIBREOFFICE_CMD=/bin/false\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("DOCLING_LIBREOFFICE_CMD", raising=False)
    monkeypatch.setenv("DOCLING_SERVICE_API_KEY", "env-key")

    result = _remote(source, "--env-file", str(env_file))

    assert result.exit_code == 1, result.output
    assert client.kwargs["url"] == "https://file.example.com"
    # A real environment variable beats the file.
    assert client.kwargs["api_key"] == "env-key"
    assert f"https://file.example.com (from {env_file.resolve()})" in result.output
    # Other keys in the file do not reach the process environment.
    assert "DOCLING_LIBREOFFICE_CMD" not in os.environ


def test_env_file_values_are_not_interpolated(tmp_path, monkeypatch, client, source):
    monkeypatch.setenv("DOCLING_TEST_OTHER_SECRET", "other-secret")
    (tmp_path / ".env").write_text(
        "DOCLING_SERVICE_URL=https://file.example.com/${DOCLING_TEST_OTHER_SECRET}\n"
        "DOCLING_SERVICE_API_KEY=${DOCLING_TEST_OTHER_SECRET}\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    result = _remote(source, "--dotenv")

    assert result.exit_code == 1, result.output
    assert (
        client.kwargs["url"] == "https://file.example.com/${DOCLING_TEST_OTHER_SECRET}"
    )
    assert client.kwargs["api_key"] == "${DOCLING_TEST_OTHER_SECRET}"


def test_local_convert_does_not_read_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("DOCLING_TEST_UNRELATED", raising=False)
    (tmp_path / ".env").write_text("DOCLING_TEST_UNRELATED=1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["convert", "missing.pdf", "--output", "out"])

    assert result.exit_code == 1, result.output
    assert "DOCLING_TEST_UNRELATED" not in os.environ
