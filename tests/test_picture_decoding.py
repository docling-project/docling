# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Pictures in documents are decoded only with the raster image decoders.

A picture selects its Pillow decoder from its first bytes. A PostScript picture
would select the EPS decoder, which runs the ``gs`` program to draw it. These
tests put a fake ``gs`` on ``PATH`` that leaves a marker file, and replace the
PNG picture of a DOCX and a PPTX package with a PostScript one.
"""

import os
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from docx import Document
from PIL import EpsImagePlugin, Image
from pptx import Presentation
from pptx.util import Inches

from docling.backend.image_backend import ImageDocumentBackend
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import InputDocument
from docling.document_converter import DocumentConverter

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the fake gs program is a shell script"
)

_EPS = b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 16 16\nshowpage\n"


def _png() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (16, 16), "red").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def gs_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Put a fake ``gs`` first on PATH and return the file it creates when run."""
    marker = tmp_path / "gs_was_run"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gs = bin_dir / "gs"
    gs.write_text(f'#!/bin/sh\ntouch "{marker}"\nexit 1\n')
    gs.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir), prepend=os.pathsep)
    # Pillow remembers the result of its first search for gs.
    monkeypatch.setattr(EpsImagePlugin, "gs_binary", None)
    return marker


def _docx(path: Path) -> Path:
    doc = Document()
    doc.add_paragraph("Before the picture")
    doc.add_picture(BytesIO(_png()))
    doc.add_paragraph("After the picture")
    doc.save(str(path))
    return path


def _pptx(path: Path) -> Path:
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = "Quarterly results"
    slide.shapes.add_picture(BytesIO(_png()), Inches(1), Inches(2))
    prs.save(str(path))
    return path


def _replace_media(src: Path, dst: Path, payload: bytes) -> Path:
    """Copy a package and replace each PNG media part with ``payload``."""
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if "/media/" in info.filename and info.filename.endswith(".png"):
                data = payload
            zout.writestr(info, data)
    return dst


@pytest.mark.parametrize(
    ("build", "fmt", "text"),
    [
        (_docx, InputFormat.DOCX, "After the picture"),
        (_pptx, InputFormat.PPTX, "Quarterly results"),
    ],
    ids=["docx", "pptx"],
)
def test_postscript_picture_is_not_decoded(
    tmp_path: Path, gs_marker: Path, build, fmt: InputFormat, text: str
):
    source = build(tmp_path / f"png.{fmt.value}")
    converter = DocumentConverter(allowed_formats=[fmt])

    # A raster picture still converts with its image.
    png_doc = converter.convert(source).document
    assert [p.image is not None for p in png_doc.pictures] == [True]

    eps_source = _replace_media(source, tmp_path / f"eps.{fmt.value}", _EPS)
    result = converter.convert(eps_source)

    assert result.status == ConversionStatus.SUCCESS
    assert text in [t.text for t in result.document.texts]
    assert all(p.image is None for p in result.document.pictures)
    assert not gs_marker.exists()


def test_postscript_image_input_is_rejected(tmp_path: Path, gs_marker: Path):
    path = tmp_path / "picture.png"
    path.write_bytes(_EPS)
    in_doc = InputDocument(
        path_or_stream=path, format=InputFormat.IMAGE, backend=ImageDocumentBackend
    )

    assert not in_doc.valid
    assert not gs_marker.exists()
