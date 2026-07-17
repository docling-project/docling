import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from docling.backend.docx.drawingml import utils as drawingml_utils


def _track_mkdtemp(monkeypatch) -> list[Path]:
    created_dirs: list[Path] = []
    real_mkdtemp = drawingml_utils.mkdtemp

    def tracking_mkdtemp(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        created_dirs.append(Path(path))
        return path

    monkeypatch.setattr(drawingml_utils, "mkdtemp", tracking_mkdtemp)
    return created_dirs


def test_convert_with_libreoffice_uses_timeout_and_isolated_profile(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        drawingml_utils, "get_libreoffice_cmd", lambda: "/usr/bin/soffice"
    )
    created_profile_dirs = _track_mkdtemp(monkeypatch)

    captured_args: list[str] = []
    captured_kwargs: dict = {}

    def fake_run(args, **kwargs):
        captured_args.extend(args)
        captured_kwargs.update(kwargs)
        # The isolated profile dir must exist while the "conversion" runs.
        assert created_profile_dirs[-1].exists()
        # Mirror LibreOffice's real behavior of writing the PDF next to the
        # input, named after the input's stem.
        (tmp_path / "drawing_only.pdf").write_bytes(b"%PDF-1.4")
        return MagicMock(returncode=0)

    monkeypatch.setattr(drawingml_utils.subprocess, "run", fake_run)

    converter = drawingml_utils.get_docx_to_pdf_converter()
    assert converter is not None

    input_path = tmp_path / "drawing_only.docx"
    output_path = tmp_path / "drawing_only.pdf"
    input_path.write_bytes(b"")

    converter(input_path, output_path)

    assert captured_kwargs["timeout"] == drawingml_utils.LIBREOFFICE_TIMEOUT_S

    profile_flag = next(
        a for a in captured_args if str(a).startswith("-env:UserInstallation=")
    )
    assert created_profile_dirs[-1].as_uri() in profile_flag

    # The isolated profile directory is cleaned up after the call.
    assert not created_profile_dirs[-1].exists()


def test_convert_with_libreoffice_cleans_up_profile_on_timeout(monkeypatch, tmp_path):
    monkeypatch.setattr(
        drawingml_utils, "get_libreoffice_cmd", lambda: "/usr/bin/soffice"
    )
    created_profile_dirs = _track_mkdtemp(monkeypatch)

    def fake_run(args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args, timeout=kwargs.get("timeout"))

    monkeypatch.setattr(drawingml_utils.subprocess, "run", fake_run)

    converter = drawingml_utils.get_docx_to_pdf_converter()
    assert converter is not None

    input_path = tmp_path / "drawing_only.docx"
    output_path = tmp_path / "drawing_only.pdf"
    input_path.write_bytes(b"")

    with pytest.raises(subprocess.TimeoutExpired):
        converter(input_path, output_path)

    # A hung/killed conversion must not leak its profile directory.
    assert not created_profile_dirs[-1].exists()
