"""hwp_to_pdf.availability 단위 테스트."""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from genon.preprocessor.converters.hwp_to_pdf import availability


@pytest.fixture
def make_executable(tmp_path: Path):
    def _factory(name: str) -> Path:
        p = tmp_path / name
        p.write_text("#!/bin/sh\nexit 0\n")
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return p
    return _factory


@pytest.mark.unit
def test_pdf_sdk_available_true_when_binary_executable(monkeypatch, make_executable, tmp_path):
    sdk_home = tmp_path
    (sdk_home / "pdfConverter")  # placeholder
    binary = make_executable("pdfConverter")
    monkeypatch.setenv("PDF_SDK_HOME", str(sdk_home))
    assert availability.pdf_sdk_binary() == binary
    assert availability.pdf_sdk_available() is True


@pytest.mark.unit
def test_pdf_sdk_available_false_when_binary_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("PDF_SDK_HOME", str(tmp_path))
    assert availability.pdf_sdk_available() is False


@pytest.mark.unit
def test_pdf_sdk_available_false_when_not_executable(monkeypatch, tmp_path):
    monkeypatch.setenv("PDF_SDK_HOME", str(tmp_path))
    (tmp_path / "pdfConverter").write_text("not exec")
    # chmod 없이 — 실행 권한 없음
    assert availability.pdf_sdk_available() is False


@pytest.mark.unit
def test_rhwp_available_uses_env_override(monkeypatch, make_executable):
    binary = make_executable("rhwp")
    monkeypatch.setenv("RHWP_BIN", str(binary))
    assert availability.rhwp_binary() == binary
    assert availability.rhwp_available() is True


@pytest.mark.unit
def test_rhwp_available_false_when_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("RHWP_BIN", str(tmp_path / "no_such_rhwp"))
    assert availability.rhwp_available() is False


@pytest.mark.unit
def test_libreoffice_available_via_which(monkeypatch):
    monkeypatch.setattr(availability.shutil, "which", lambda name: "/usr/bin/soffice" if name == "soffice" else None)
    assert availability.libreoffice_available() is True


@pytest.mark.unit
def test_libreoffice_unavailable_when_which_returns_none(monkeypatch):
    monkeypatch.setattr(availability.shutil, "which", lambda name: None)
    assert availability.libreoffice_available() is False
