# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Replacing a broken PDF text layer with OCR (OcrOptions.replace_broken_text_layer).

The test page shows "Proxy Statement" as an image, over a text layer that either
matches it or is the output of a font decoded as accented Latin letters.
"""

import ctypes
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
import pytest
from PIL import Image, ImageDraw, ImageFont

from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractCliOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

pytestmark = pytest.mark.ml_ocr

_SHOWN = "Proxy Statement"
_BROKEN_LAYER = " ".join(
    [
        "\N{LATIN SMALL LETTER A WITH ACUTE}\N{LATIN SMALL LETTER A WITH DIAERESIS}"
        "\N{LATIN SMALL LETTER C WITH CEDILLA}\N{LATIN SMALL LETTER E WITH GRAVE}"
        "\N{LATIN SMALL LETTER I WITH CIRCUMFLEX}"
    ]
    * 12
)


def _page_pdf(path: Path, text_layer: str) -> Path:
    """One page: `_SHOWN` drawn as an image, under a text layer of `text_layer`."""
    width, height = 400, 200
    image = Image.new("RGB", (width * 3, height * 3), "white")
    font = ImageFont.load_default(size=110)
    ImageDraw.Draw(image).text((60, 230), _SHOWN, fill="black", font=font)

    pdf = pdfium.PdfDocument.new()
    page = pdf.new_page(width, height)
    picture = pdfium.PdfImage.new(pdf)
    picture.set_bitmap(pdfium.PdfBitmap.from_pil(image))
    picture.set_matrix(pdfium.PdfMatrix().scale(width, height))
    page.insert_obj(picture)

    helvetica = pdfium_c.FPDFText_LoadStandardFont(pdf.raw, b"Helvetica")
    text = pdfium_c.FPDFPageObj_CreateTextObj(pdf.raw, helvetica, 9.0)
    utf16 = (text_layer + "\0").encode("utf-16-le")
    pdfium_c.FPDFText_SetText(
        text, ctypes.cast(ctypes.c_char_p(utf16), ctypes.POINTER(pdfium_c.FPDF_WCHAR))
    )
    # A broken layer still draws the right glyphs; white ink keeps this one from
    # adding marks of its own to the image the OCR reads.
    pdfium_c.FPDFPageObj_SetFillColor(text, 255, 255, 255, 255)
    pdfium_c.FPDFPageObj_Transform(text, 1, 0, 0, 1, 20, 112)
    pdfium_c.FPDFPage_InsertObject(page.raw, text)
    page.gen_content()
    pdf.save(path)
    return path


def _convert(path: Path, *, replace: bool) -> ConversionResult:
    ocr = TesseractCliOcrOptions(lang=["eng"], replace_broken_text_layer=replace)
    options = PdfPipelineOptions(do_ocr=True, do_table_structure=False, ocr_options=ocr)
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )
    return converter.convert(path)


def test_broken_text_layer_is_replaced_by_ocr(tmp_path: Path) -> None:
    result = _convert(_page_pdf(tmp_path / "broken.pdf", _BROKEN_LAYER), replace=True)

    assert result.confidence.pages[1].parse_score == 0.0
    markdown = result.document.export_to_markdown()
    assert _SHOWN in markdown
    assert _BROKEN_LAYER[:5] not in markdown


def test_option_off_keeps_the_text_layer_and_its_score(tmp_path: Path) -> None:
    result = _convert(_page_pdf(tmp_path / "broken.pdf", _BROKEN_LAYER), replace=False)

    assert result.confidence.pages[1].parse_score == 1.0
    assert _BROKEN_LAYER[:5] in result.document.export_to_markdown()


def test_good_text_layer_is_kept(tmp_path: Path) -> None:
    layer = "Annual Proxy Statement of the Company"
    result = _convert(_page_pdf(tmp_path / "good.pdf", layer), replace=True)

    assert result.confidence.pages[1].parse_score == 1.0
    assert layer in result.document.export_to_markdown()
