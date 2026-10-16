# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Read the baseline and drawn size of each character of a PDF page with PDFium.

Neither PDF backend exposes these on its text cells: docling-parse character boxes follow
each glyph's ink, and the pypdfium2 backend builds no character cells.
"""

from __future__ import annotations

import ctypes
import importlib.util
import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from docling_core.types.doc import Size

from docling.utils.locks import pypdfium2_lock

if TYPE_CHECKING:
    import pypdfium2 as pdfium

# Characters rotated by more than this (radians) are not horizontal text.
_MAX_HORIZONTAL_ANGLE = 0.01


@dataclass(frozen=True)
class PdfChar:
    """One character, in PDFium's unrotated user space (bottom-left origin)."""

    text: str
    # Box from the font metrics, so its height does not depend on the glyph's shape.
    left: float
    bottom: float
    right: float
    top: float
    baseline: float
    # Font size scaled by the text matrix: many PDFs set size 1 and scale the matrix.
    size: float
    horizontal: bool


@dataclass(frozen=True)
class PdfPageChars:
    """The characters of one page, with what is needed to map them onto Docling's page."""

    chars: list[PdfChar]
    rotation: int
    display_size: Size
    origin: tuple[float, float]

    def to_display_frame(
        self, box: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        """Map a box to the page's display frame (bottom-left origin)."""
        # lazy import: pypdfium2 is an optional dependency
        from docling.backend.pypdfium2_backend import _rect_to_display_frame

        return _rect_to_display_frame(
            box, self.rotation, self.display_size, self.origin
        )


def pypdfium2_available() -> bool:
    """Tell whether pypdfium2, an optional dependency, is installed."""
    return importlib.util.find_spec("pypdfium2") is not None


@contextmanager
def open_pdf(
    path_or_stream: Union[Path, BytesIO], password: Optional[str] = None
) -> Iterator[pdfium.PdfDocument]:
    """Open a PDF with PDFium for the duration of the block."""
    import pypdfium2 as pdfium

    source = (
        path_or_stream.getvalue()
        if isinstance(path_or_stream, BytesIO)
        else path_or_stream
    )
    with pypdfium2_lock:
        pdf = pdfium.PdfDocument(source, password=password)
    try:
        yield pdf
    finally:
        with pypdfium2_lock:
            pdf.close()


def read_page_chars(pdf: pdfium.PdfDocument, page_index: int) -> PdfPageChars:
    """Return the characters of a page (0-based), in content order."""
    import pypdfium2.raw as pdfium_c

    with pypdfium2_lock:
        page = pdf[page_index]
        try:
            width, height = page.get_size()
            rotation = page.get_rotation()
            left_x, bottom_y = page.get_bbox()[:2]
            text_page = page.get_textpage()
            try:
                chars = _read_chars(text_page.raw, text_page.count_chars(), pdfium_c)
            finally:
                text_page.close()
        finally:
            page.close()
    return PdfPageChars(
        chars=chars,
        rotation=rotation,
        display_size=Size(width=width, height=height),
        origin=(left_x, bottom_y),
    )


def _read_chars(raw, count: int, pdfium_c) -> list[PdfChar]:
    box = pdfium_c.FS_RECTF()
    matrix = pdfium_c.FS_MATRIX()
    origin_x, origin_y = ctypes.c_double(), ctypes.c_double()
    chars: list[PdfChar] = []
    for index in range(count):
        code = pdfium_c.FPDFText_GetUnicode(raw, index)
        if code <= 0 or not pdfium_c.FPDFText_GetLooseCharBox(raw, index, box):
            continue
        if not pdfium_c.FPDFText_GetCharOrigin(raw, index, origin_x, origin_y):
            continue
        if 0xDC00 <= code <= 0xDFFF and chars and _is_high_surrogate(chars[-1]):
            # Where wchar_t is 16 bits, PDFium splits a character beyond the BMP (a math
            # italic x) into a surrogate pair; join the halves.
            high = chars.pop()
            chars.append(
                replace(
                    high,
                    text=chr(
                        0x10000 + ((ord(high.text) - 0xD800) << 10) + (code - 0xDC00)
                    ),
                    right=max(high.right, box.right),
                    top=max(high.top, box.top),
                )
            )
            continue
        size = float(pdfium_c.FPDFText_GetFontSize(raw, index))
        if pdfium_c.FPDFText_GetMatrix(raw, index, matrix):
            size *= math.sqrt(abs(matrix.a * matrix.d - matrix.b * matrix.c))
        angle = float(pdfium_c.FPDFText_GetCharAngle(raw, index))
        chars.append(
            PdfChar(
                text=chr(code),
                left=box.left,
                bottom=box.bottom,
                right=box.right,
                top=box.top,
                baseline=origin_y.value,
                size=size,
                horizontal=0 <= angle <= _MAX_HORIZONTAL_ANGLE
                or abs(angle - 2 * math.pi) <= _MAX_HORIZONTAL_ANGLE,
            )
        )
    # A half of a surrogate pair left on its own comes from a broken ToUnicode map.
    return [
        replace(c, text="\ufffd") if 0xD800 <= ord(c.text) <= 0xDFFF else c
        for c in chars
    ]


def _is_high_surrogate(char: PdfChar) -> bool:
    return 0xD800 <= ord(char.text) <= 0xDBFF
