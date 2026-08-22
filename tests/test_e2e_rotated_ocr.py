"""Regression test for docling#3839 — OCR text on a rotated page is dropped.

A rotated page places its body text in a margin, where the layout model
classifies it as ``page_footer``/``page_header`` -> ``ContentLayer.FURNITURE``.
``export_to_markdown()`` omits the FURNITURE layer from its default (BODY)
output, so the correctly-recognized OCR text is silently dropped and the
document exports empty. It is recoverable only via ``included_content_layers``.

The failure is in the *label*, not OCR or coordinate handling: the text is
recognized verbatim, and it reproduces on two independent OCR backends
(Tesseract and RapidOCR — the latter applies no OSD / rotation / coordinate
transform at all, ruling out a coordinate mismatch). For accessible-PDF output
the impact is worse than an empty string: FURNITURE becomes ``/Artifact`` in the
tag tree, so a "successful" conversion yields a document that assistive
technology reads as completely empty.

The image is generated synthetically (deterministic, no fixture file): a few
lines of small text at the top margin land at the bottom margin after a 180°
rotation, where the layout model classifies them ``page_footer``. The text is
deliberately dense enough (multiple long lines) to clear Tesseract OSD's
minimum-character floor — so OSD detects the rotation and the text is OCR'd
correctly, isolating the *label* bug. (A single sparse line can fall below OSD's
floor and fail for the wrong reason — mirrored OCR text — instead of exercising
the furniture drop; dense text at 90°/270° is instead re-classified as body, so
this test pins the robust 180° case.)

Marked ``xfail`` so it merges independently of the fix and gives that PR a
red/green target: the furniture-drop assertion fails on current ``main`` and
xpasses once pre-layout orientation detection lands — remove the marker then.
"""

import importlib.util
import shutil

import pytest
from PIL import Image, ImageDraw, ImageFont

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions,
    TesseractCliOcrOptions,
)
from docling.document_converter import DocumentConverter, ImageFormatOption

_LINE = "Certified 2026 reference ZQXPHOENIX 7742 north garage roof warranty batch A"
_MARKER = "ZQXPHOENIX"

_BACKENDS = [
    pytest.param(
        TesseractCliOcrOptions,
        marks=pytest.mark.skipif(
            shutil.which("tesseract") is None, reason="tesseract not installed"
        ),
        id="tesseract",
    ),
    pytest.param(
        RapidOcrOptions,
        marks=pytest.mark.skipif(
            importlib.util.find_spec("rapidocr_onnxruntime") is None
            and importlib.util.find_spec("rapidocr") is None,
            reason="rapidocr not installed",
        ),
        id="rapidocr",
    ),
]


def _make_rotated_image(tmp_path, angle):
    """An upright page with a small dense top-margin block, saved rotated ``angle``°."""
    img = Image.new("RGB", (1700, 2200), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    y = 15
    for _ in range(3):  # enough characters to clear OSD's floor; thin edge strip
        draw.text((60, y), _LINE, fill="black", font=font)
        y += 33
    path = tmp_path / f"rotated_{angle}.png"
    img.rotate(angle, expand=True).save(path)
    return path


def _convert(path, ocr_options_cls):
    options = PdfPipelineOptions()
    options.do_ocr = True
    options.ocr_options = ocr_options_cls(force_full_page_ocr=True)
    converter = DocumentConverter(
        format_options={InputFormat.IMAGE: ImageFormatOption(pipeline_options=options)}
    )
    return converter.convert(path).document


@pytest.mark.xfail(
    reason="docling#3839: rotated-page OCR text is classified as header/footer "
    "FURNITURE and dropped from the default export; remove when pre-layout "
    "orientation detection lands",
    strict=False,
)
@pytest.mark.parametrize("ocr_options_cls", _BACKENDS)
@pytest.mark.parametrize("angle", [180])
def test_rotated_page_ocr_text_reaches_default_export(tmp_path, angle, ocr_options_cls):
    from docling_core.types.doc import ContentLayer

    doc = _convert(_make_rotated_image(tmp_path, angle), ocr_options_cls)

    # Sanity: the text IS recognized — recoverable via the furniture layer.
    recovered = doc.export_to_markdown(
        included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}
    )
    assert _MARKER in recovered, f"OCR should recognize the text at {angle}°"

    # The bug: it is classified page_header/page_footer -> FURNITURE and dropped
    # from the default (BODY) export, so the conversion silently yields an empty
    # document.
    assert _MARKER in doc.export_to_markdown(), (
        f"rotated-page ({angle}°) OCR text is silently dropped from the default "
        f"export (docling#3839) — it lands in a header/footer FURNITURE item"
    )
