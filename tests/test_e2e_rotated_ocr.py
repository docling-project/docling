# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

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

The fixtures (``tests/data/ocr/sources/rotated_{90,180,270}_dense_text.png``,
viewable in the repo; regenerate by running this file as a script) are one
upright page with a few lines of small top-margin text, saved at each rotation.
The text is deliberately dense enough (multiple long lines) to clear Tesseract
OSD's minimum-character floor — so OSD detects the rotation and the text is
OCR'd correctly, isolating the *label* bug. (A single sparse line can fall below
OSD's floor and fail for the wrong reason — mirrored OCR text — instead of
exercising the furniture drop.)

All angles run the same assertions; only the marks differ, pinning the
problem-range boundary as measured on current ``main``:

    angle  tesseract              rapidocr
    90     passes (body)          excluded: garbled at the OCR stage
    180    xfail: furniture drop  xfail: furniture drop
    270    passes (body)          passes (body)

At 90°/270° the layout model classifies the (sideways) dense block as body
text, so the marker reaches the default export — those cases are plain tests
guarding the boundary. 90°-RapidOCR is excluded because its line-angle
classifier only handles 0°/180°: recognition itself garbles, a failure mode
out of scope for a labeling test.

The 180° cases are marked ``xfail`` so the test merges independently of the
fix and gives that PR a red/green target: the furniture-drop assertion fails
on current ``main`` and xpasses once pre-layout orientation detection lands —
remove the marker then.
"""

import importlib.util
import shutil
from pathlib import Path

import pytest

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions,
    TesseractCliOcrOptions,
)
from docling.document_converter import DocumentConverter, ImageFormatOption

# Executes OCR model code (Tesseract / RapidOCR): route to the ml_ocr CI lane,
# not the core lane (which ignores ML-marked modules).
pytestmark = pytest.mark.ml_ocr

_LINE = "Certified 2026 reference ZQXPHOENIX 7742 north garage roof warranty batch A"
_MARKER = "ZQXPHOENIX"

_FIXTURES = {
    angle: Path(f"./tests/data/ocr/sources/rotated_{angle}_dense_text.png")
    for angle in (90, 180, 270)
}

_SKIP_NO_TESSERACT = pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="tesseract not installed"
)
_SKIP_NO_RAPIDOCR = pytest.mark.skipif(
    importlib.util.find_spec("rapidocr_onnxruntime") is None
    and importlib.util.find_spec("rapidocr") is None,
    reason="rapidocr not installed",
)
_XFAIL_3839 = pytest.mark.xfail(
    reason="docling#3839: rotated-page OCR text is classified as header/footer "
    "FURNITURE and dropped from the default export; remove when pre-layout "
    "orientation detection lands",
    strict=False,
)

# The full angle x backend boundary, minus 90°-RapidOCR (garbles at the OCR
# stage — a recognition failure out of scope for this labeling test).
_CASES = [
    pytest.param(
        90, TesseractCliOcrOptions, marks=[_SKIP_NO_TESSERACT], id="90-tesseract"
    ),
    pytest.param(
        180,
        TesseractCliOcrOptions,
        marks=[_SKIP_NO_TESSERACT, _XFAIL_3839],
        id="180-tesseract",
    ),
    pytest.param(
        270, TesseractCliOcrOptions, marks=[_SKIP_NO_TESSERACT], id="270-tesseract"
    ),
    pytest.param(
        180, RapidOcrOptions, marks=[_SKIP_NO_RAPIDOCR, _XFAIL_3839], id="180-rapidocr"
    ),
    pytest.param(270, RapidOcrOptions, marks=[_SKIP_NO_RAPIDOCR], id="270-rapidocr"),
]


def _generate_fixture(angle):
    """Regenerate the checked-in fixture: an upright page with a small dense
    top-margin block, saved rotated ``angle``°. Run this file as a script."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (1700, 2200), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("DejaVuSans.ttf", 22)
    y = 15
    for _ in range(3):  # enough characters to clear OSD's floor; thin edge strip
        draw.text((60, y), _LINE, fill="black", font=font)
        y += 33
    img.rotate(angle, expand=True).save(_FIXTURES[angle])


def _convert(path, ocr_options_cls):
    options = PdfPipelineOptions()
    options.do_ocr = True
    options.ocr_options = ocr_options_cls(force_full_page_ocr=True)
    converter = DocumentConverter(
        format_options={InputFormat.IMAGE: ImageFormatOption(pipeline_options=options)}
    )
    return converter.convert(path).document


@pytest.mark.parametrize(("angle", "ocr_options_cls"), _CASES)
def test_rotated_page_ocr_text_reaches_default_export(angle, ocr_options_cls):
    from docling_core.types.doc import ContentLayer

    doc = _convert(_FIXTURES[angle], ocr_options_cls)

    # Sanity: the text IS recognized — recoverable via the furniture layer.
    recovered = doc.export_to_markdown(
        included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}
    )
    assert _MARKER in recovered, f"OCR should recognize the text at {angle}°"

    # At 180° (the xfail cases) the text is classified page_header/page_footer
    # -> FURNITURE and dropped from the default (BODY) export, so the conversion
    # silently yields an empty document. At 90°/270° this passes today and pins
    # the boundary.
    assert _MARKER in doc.export_to_markdown(), (
        f"rotated-page ({angle}°) OCR text is silently dropped from the default "
        f"export (docling#3839) — it lands in a header/footer FURNITURE item"
    )


if __name__ == "__main__":
    for _angle in _FIXTURES:
        _generate_fixture(_angle)
