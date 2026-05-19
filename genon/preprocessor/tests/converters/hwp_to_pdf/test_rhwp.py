"""RhwpConverter 단위 테스트."""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from genon.preprocessor.converters.hwp_to_pdf import rhwp as rhwp_mod
from genon.preprocessor.converters.hwp_to_pdf.rhwp import RhwpConverter


@pytest.fixture
def rhwp_bin(monkeypatch, tmp_path):
    """rhwp_binary 와 rhwp_available 를 동시에 모킹해 가용으로 둠."""
    fake = tmp_path / "rhwp"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    monkeypatch.setattr(rhwp_mod, "rhwp_binary", lambda: fake)
    monkeypatch.setattr(rhwp_mod, "rhwp_available", lambda: True)
    return fake


@pytest.mark.unit
def test_is_available_uses_availability_module(monkeypatch):
    monkeypatch.setattr(rhwp_mod, "rhwp_available", lambda: True)
    assert RhwpConverter().is_available() is True

    monkeypatch.setattr(rhwp_mod, "rhwp_available", lambda: False)
    assert RhwpConverter().is_available() is False


@pytest.mark.unit
def test_convert_invokes_export_pdf_subcommand(rhwp_bin, tmp_path):
    in_file = tmp_path / "doc.hwp"
    in_file.write_bytes(b"\x00")  # dummy
    out_file = tmp_path / "doc.pdf"

    def fake_run(cmd, **kwargs):
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"%PDF-1.4 stub")
        proc = MagicMock()
        proc.returncode = 0
        proc.stdout = ""
        proc.stderr = ""
        return proc

    with patch.object(subprocess, "run", side_effect=fake_run) as run_mock:
        result = RhwpConverter().convert(str(in_file))

    assert result == str(out_file)
    call_args = run_mock.call_args.args[0]
    assert call_args[0] == str(rhwp_bin)
    assert call_args[1] == "export-pdf"
    assert call_args[2] == str(in_file)
    assert "-o" in call_args


@pytest.mark.unit
def test_convert_returns_none_on_nonzero_returncode(rhwp_bin, tmp_path):
    in_file = tmp_path / "doc.hwp"
    in_file.write_bytes(b"\x00")

    proc = MagicMock(returncode=1, stdout="", stderr="parse error")
    with patch.object(subprocess, "run", return_value=proc):
        assert RhwpConverter().convert(str(in_file)) is None


@pytest.mark.unit
def test_convert_returns_none_when_output_missing(rhwp_bin, tmp_path):
    in_file = tmp_path / "doc.hwp"
    in_file.write_bytes(b"\x00")

    # rc=0 이지만 output 파일 미생성
    proc = MagicMock(returncode=0, stdout="", stderr="")
    with patch.object(subprocess, "run", return_value=proc):
        assert RhwpConverter().convert(str(in_file)) is None


@pytest.mark.unit
def test_convert_returns_none_when_output_empty(rhwp_bin, tmp_path):
    in_file = tmp_path / "doc.hwp"
    in_file.write_bytes(b"\x00")
    out_file = tmp_path / "doc.pdf"
    out_file.write_bytes(b"")  # 0 byte

    proc = MagicMock(returncode=0, stdout="", stderr="")
    with patch.object(subprocess, "run", return_value=proc):
        assert RhwpConverter().convert(str(in_file)) is None


@pytest.mark.unit
def test_convert_handles_timeout(rhwp_bin, tmp_path):
    in_file = tmp_path / "doc.hwp"
    in_file.write_bytes(b"\x00")

    with patch.object(subprocess, "run", side_effect=subprocess.TimeoutExpired(cmd="rhwp", timeout=1)):
        assert RhwpConverter().convert(str(in_file)) is None


@pytest.mark.unit
def test_convert_honors_timeout_env(rhwp_bin, tmp_path, monkeypatch):
    in_file = tmp_path / "doc.hwp"
    in_file.write_bytes(b"\x00")
    monkeypatch.setenv("HWP_TO_PDF_TIMEOUT_SEC", "12")

    seen = {}

    def fake_run(cmd, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        proc = MagicMock(returncode=1, stdout="", stderr="")
        return proc

    with patch.object(subprocess, "run", side_effect=fake_run):
        RhwpConverter().convert(str(in_file))

    assert seen["timeout"] == 12
