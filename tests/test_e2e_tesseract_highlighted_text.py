# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import importlib.util
import re
import shutil
from pathlib import Path

import pytest

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    OcrMode,
    PdfPipelineOptions,
    TesseractCliOcrOptions,
    TesseractOcrOptions,
)
from docling.document_converter import DocumentConverter, ImageFormatOption

pytestmark = pytest.mark.ml_ocr

_FIXTURE = Path("./tests/data/ocr/sources/highlighted_text.png")

_LINES = [
    "The parties agree that the landlord will repair the roof before winter.",
    "Rent is due on the first day of each month at the office of the manager.",
    "Late payments accrue a fee of five percent after a grace period of ten days.",
    "The tenant must give sixty days written notice before moving out of the unit.",
    "Pets are allowed only with the written consent of the landlord in advance.",
    "The security deposit will be returned within thirty days after the lease ends.",
    "Either party may terminate this agreement for a material breach of its terms.",
    "This agreement is governed by the laws of the state where the unit is located.",
]
_HIGHLIGHTED_ROWS = (2, 3, 4)

_SKIP_NO_TESSERACT = pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="tesseract not installed"
)
_SKIP_NO_TESSEROCR = pytest.mark.skipif(
    importlib.util.find_spec("tesserocr") is None, reason="tesserocr not installed"
)


def _generate_fixture():
    from PIL import Image, ImageDraw, ImageFont

    dpi = 200
    image = Image.new("RGB", (round(8.5 * dpi), round(11 * dpi)), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=28)
    y = dpi
    for row, line in enumerate(_LINES):
        if row in _HIGHLIGHTED_ROWS:
            left, top, right, bottom = draw.textbbox((dpi, y), line, font=font)
            draw.rectangle((left - 4, top - 6, right + 4, bottom + 6), fill="yellow")
        draw.text((dpi, y), line, fill="black", font=font)
        y += 50
    image.save(_FIXTURE, dpi=(dpi, dpi))


def _words(text):
    return re.findall(r"[a-z]+", text.lower())


@pytest.mark.parametrize(
    "ocr_options_cls",
    [
        pytest.param(TesseractCliOcrOptions, marks=_SKIP_NO_TESSERACT, id="tesseract"),
        pytest.param(TesseractOcrOptions, marks=_SKIP_NO_TESSEROCR, id="tesserocr"),
    ],
)
def test_full_page_ocr_reads_highlighted_lines(ocr_options_cls):
    options = PdfPipelineOptions()
    options.do_ocr = True
    options.ocr_options = ocr_options_cls(lang=["eng"], mode=OcrMode.FULL_PAGE)
    converter = DocumentConverter(
        format_options={InputFormat.IMAGE: ImageFormatOption(pipeline_options=options)}
    )

    found = set(_words(converter.convert(_FIXTURE).document.export_to_markdown()))

    highlighted = [word for row in _HIGHLIGHTED_ROWS for word in _words(_LINES[row])]
    missing = [word for word in highlighted if word not in found]
    assert len(missing) <= len(highlighted) // 10, missing


if __name__ == "__main__":
    _generate_fixture()
