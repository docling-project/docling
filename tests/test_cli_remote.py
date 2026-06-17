"""Tests for the `docling convert-remote` command.

The service client is faked so these run without a live docling-serve instance.
"""

import re
from pathlib import Path, PurePath

import pytest
from typer.testing import CliRunner

from docling.cli.main import app
from docling.cli.remote import _parse_page_range
from docling.datamodel.base_models import ConversionStatus, InputFormat, OutputFormat

runner = CliRunner()
pytestmark = pytest.mark.external_service
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


class _FakeDoc:
    def save_as_markdown(self, *, filename, image_mode):
        Path(filename).write_text("# converted\n", encoding="utf-8")

    def save_as_json(self, *, filename, image_mode):
        Path(filename).write_text("{}", encoding="utf-8")


class _FakeResult:
    def __init__(self, name: str):
        self.status = ConversionStatus.SUCCESS
        self.input = type("Inp", (), {"file": PurePath(name)})()
        self.document = _FakeDoc()
        self.errors: list = []
        self.timings: dict = {}


class _FakeClient:
    """Captures constructor and convert_all arguments for assertions."""

    instances: list["_FakeClient"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.health_called = False
        self.captured_sources = None
        self.captured_options = None
        self.health_error: Exception | None = None
        _FakeClient.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def health(self):
        self.health_called = True
        if self.health_error is not None:
            raise self.health_error
        return {"status": "ok"}

    def convert_all(self, sources, options=None, **kwargs):
        self.captured_sources = list(sources)
        self.captured_options = options
        return iter(
            [
                _FakeResult(s if isinstance(s, str) else s.name)
                for s in self.captured_sources
            ]
            or [_FakeResult("report.pdf")]
        )


@pytest.fixture(autouse=True)
def _reset_fake():
    _FakeClient.instances.clear()
    yield
    _FakeClient.instances.clear()


@pytest.fixture
def _patch_client(monkeypatch):
    monkeypatch.setattr("docling.cli.remote.DoclingServiceClient", _FakeClient)
    return _FakeClient


def test_remote_help_is_self_sufficient():
    result = runner.invoke(app, ["convert-remote", "--help"])
    assert result.exit_code == 0
    help_text = _strip_ansi(result.output)
    for marker in (
        "Authentication",
        "Exit codes",
        "Examples",
        "DOCLING_SERVICE_URL",
        "DOCLING_SERVICE_API_KEY",
        "--service-url",
    ):
        assert marker in help_text


def test_remote_missing_url_exits_2(monkeypatch):
    monkeypatch.delenv("DOCLING_SERVICE_URL", raising=False)
    result = runner.invoke(app, ["convert-remote", "report.pdf"])
    assert result.exit_code == 2
    assert "No service URL" in result.output


def test_remote_converts_local_file(tmp_path, _patch_client):
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-1.4")
    output = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "convert-remote",
            str(source),
            "--service-url",
            "https://docling.example.com",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output / "report.md").exists()
    client = _FakeClient.instances[-1]
    assert client.health_called
    # Local files are sent as Path objects.
    assert client.captured_sources == [source]


def test_remote_url_source_stays_string(tmp_path, _patch_client):
    output = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "convert-remote",
            "https://example.com/doc.pdf",
            "--service-url",
            "https://docling.example.com",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    client = _FakeClient.instances[-1]
    assert client.captured_sources == ["https://example.com/doc.pdf"]


def test_remote_maps_conversion_options(tmp_path, _patch_client):
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-1.4")
    output = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "convert-remote",
            str(source),
            "--service-url",
            "https://docling.example.com",
            "--from",
            "pdf",
            "--from",
            "docx",
            "--to",
            "md",
            "--to",
            "json",
            "--no-ocr",
            "--no-tables",
            "--ocr-lang",
            "en,de",
            "--page-range",
            "2-5",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    opts = _FakeClient.instances[-1].captured_options
    assert opts.from_formats == [InputFormat.PDF, InputFormat.DOCX]
    assert opts.to_formats == [OutputFormat.MARKDOWN, OutputFormat.JSON]
    assert opts.do_ocr is False
    assert opts.do_table_structure is False
    assert opts.ocr_lang == ["en", "de"]
    assert opts.page_range == (2, 5)
    # Both requested formats are written locally.
    assert (output / "report.md").exists()
    assert (output / "report.json").exists()


def test_remote_credentials_from_env(tmp_path, monkeypatch, _patch_client):
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-1.4")
    output = tmp_path / "out"
    monkeypatch.setenv("DOCLING_SERVICE_URL", "https://env.example.com")
    monkeypatch.setenv("DOCLING_SERVICE_API_KEY", "secret-key")

    result = runner.invoke(
        app, ["convert-remote", str(source), "--output", str(output)]
    )
    assert result.exit_code == 0, result.output
    client = _FakeClient.instances[-1]
    assert client.kwargs["url"] == "https://env.example.com"
    assert client.kwargs["api_key"] == "secret-key"


def test_remote_health_failure_exits_1(tmp_path, monkeypatch):
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-1.4")

    class _UnreachableClient(_FakeClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.health_error = ConnectionError("connection refused")

    monkeypatch.setattr("docling.cli.remote.DoclingServiceClient", _UnreachableClient)

    result = runner.invoke(
        app,
        [
            "convert-remote",
            str(source),
            "--service-url",
            "https://docling.example.com",
        ],
    )
    assert result.exit_code == 1
    assert "Cannot reach service" in result.output


def test_remote_missing_input_exits_1(tmp_path, _patch_client):
    result = runner.invoke(
        app,
        [
            "convert-remote",
            str(tmp_path / "does-not-exist.pdf"),
            "--service-url",
            "https://docling.example.com",
        ],
    )
    assert result.exit_code == 1
    assert " ".join(result.output.split()).endswith("does not exist.")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, None), ("1-4", (1, 4)), ("3", (3, 3)), (" 2-7 ", (2, 7))],
)
def test_parse_page_range_valid(raw, expected):
    assert _parse_page_range(raw) == expected


@pytest.mark.parametrize("raw", ["abc", "0-3", "4-2", "1-", "-3", "1-x"])
def test_parse_page_range_invalid(raw):
    import typer

    with pytest.raises(typer.BadParameter):
        _parse_page_range(raw)
