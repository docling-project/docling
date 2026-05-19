"""RhwpConverter HTTP client 단위 테스트 (이슈 #199)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from genon.preprocessor.converters.hwp_to_pdf import availability, rhwp as rhwp_mod
from genon.preprocessor.converters.hwp_to_pdf.rhwp import (
    CONVERT_PATH,
    RhwpConverter,
)

ENDPOINT = "http://rhwp-pdf-api:7878"
FAKE_PDF = b"%PDF-1.4\n%fake\n"


@pytest.fixture
def hwp_input(tmp_path: Path) -> Path:
    p = tmp_path / "doc.hwp"
    p.write_bytes(b"HWP-BYTES")
    return p


@pytest.fixture
def url_set(monkeypatch):
    monkeypatch.setenv("RHWP_PDF_API_URL", ENDPOINT)


@pytest.mark.unit
def test_is_available_reflects_env(monkeypatch):
    monkeypatch.delenv("RHWP_PDF_API_URL", raising=False)
    assert RhwpConverter().is_available() is False
    monkeypatch.setenv("RHWP_PDF_API_URL", ENDPOINT)
    assert RhwpConverter().is_available() is True


@pytest.mark.unit
def test_is_available_false_for_empty_or_whitespace(monkeypatch):
    monkeypatch.setenv("RHWP_PDF_API_URL", "   ")
    assert RhwpConverter().is_available() is False


@pytest.mark.unit
def test_convert_posts_hwp_bytes_and_writes_pdf(url_set, hwp_input):
    captured = {}

    def fake_post(url, data, headers, timeout):
        captured["url"] = url
        captured["data"] = data
        captured["headers"] = headers
        captured["timeout"] = timeout
        return MagicMock(status_code=200, content=FAKE_PDF, text="")

    with patch.object(requests, "post", side_effect=fake_post):
        result = RhwpConverter().convert(str(hwp_input))

    assert result == str(hwp_input.with_suffix(".pdf"))
    assert Path(result).read_bytes() == FAKE_PDF
    assert captured["url"] == ENDPOINT + CONVERT_PATH
    assert captured["data"] == b"HWP-BYTES"
    assert captured["headers"] == {"Content-Type": "application/octet-stream"}


@pytest.mark.unit
def test_convert_strips_trailing_slash_in_url(monkeypatch, hwp_input):
    monkeypatch.setenv("RHWP_PDF_API_URL", ENDPOINT + "/")
    seen = {}

    def fake_post(url, **_):
        seen["url"] = url
        return MagicMock(status_code=200, content=FAKE_PDF, text="")

    with patch.object(requests, "post", side_effect=fake_post):
        RhwpConverter().convert(str(hwp_input))
    assert seen["url"] == ENDPOINT + CONVERT_PATH


@pytest.mark.unit
def test_convert_returns_none_without_url(monkeypatch, hwp_input):
    monkeypatch.delenv("RHWP_PDF_API_URL", raising=False)
    assert RhwpConverter().convert(str(hwp_input)) is None


@pytest.mark.unit
def test_convert_returns_none_on_non_200(url_set, hwp_input):
    resp = MagicMock(status_code=500, content=b"server error", text="boom")
    with patch.object(requests, "post", return_value=resp):
        assert RhwpConverter().convert(str(hwp_input)) is None


@pytest.mark.unit
def test_convert_returns_none_on_non_pdf_body(url_set, hwp_input):
    resp = MagicMock(status_code=200, content=b"not a pdf", text="")
    with patch.object(requests, "post", return_value=resp):
        assert RhwpConverter().convert(str(hwp_input)) is None


@pytest.mark.unit
def test_convert_returns_none_on_empty_body(url_set, hwp_input):
    resp = MagicMock(status_code=200, content=b"", text="")
    with patch.object(requests, "post", return_value=resp):
        assert RhwpConverter().convert(str(hwp_input)) is None


@pytest.mark.unit
def test_convert_handles_timeout(url_set, hwp_input):
    with patch.object(requests, "post", side_effect=requests.Timeout()):
        assert RhwpConverter().convert(str(hwp_input)) is None


@pytest.mark.unit
def test_convert_handles_connection_error(url_set, hwp_input):
    with patch.object(requests, "post", side_effect=requests.ConnectionError("refused")):
        assert RhwpConverter().convert(str(hwp_input)) is None


@pytest.mark.unit
def test_convert_returns_none_when_input_missing(url_set, tmp_path):
    nonexistent = tmp_path / "missing.hwp"
    assert RhwpConverter().convert(str(nonexistent)) is None


@pytest.mark.unit
def test_convert_honors_timeout_env(url_set, hwp_input, monkeypatch):
    monkeypatch.setenv("HWP_TO_PDF_TIMEOUT_SEC", "12")
    seen = {}

    def fake_post(url, data, headers, timeout):
        seen["timeout"] = timeout
        return MagicMock(status_code=200, content=FAKE_PDF, text="")

    with patch.object(requests, "post", side_effect=fake_post):
        RhwpConverter().convert(str(hwp_input))
    assert seen["timeout"] == 12.0
