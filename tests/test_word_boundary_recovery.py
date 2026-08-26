# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Word-boundary recovery for PDFs that omit space glyphs.

The fixtures are written as raw PDF bytes rather than shipped as binary files so
that the content stream under test is visible in the diff: the whole point is which
operators the producer emitted, and a checked-in PDF would hide that.
"""

from pathlib import Path

import pytest

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

CAPTION_WORDS = [
    "Cause",
    "of",
    "death",
    "per",
    "legal",
    "outcome",
    "for",
    "non-human",
    "cases",
    "in",
    "the",
    "Netherlands",
]
CAPTION = " ".join(CAPTION_WORDS)


def _write_pdf(path: Path, text_ops: str) -> Path:
    """Write a one-page Helvetica PDF whose content stream is ``text_ops``."""
    content = f"BT\n/F1 12 Tf\n72 700 Td\n{text_ops}\nET\n"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        "/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        f"<< /Length {len(content)} >>\nstream\n{content}endstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")

    startxref = len(out)
    size = len(objects) + 1
    out += f"xref\n0 {size}\n0000000000 65535 f \n".encode("latin-1")
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode("latin-1")
    out += (
        f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{startxref}\n%%EOF\n"
    ).encode("latin-1")

    path.write_bytes(bytes(out))
    return path


def _kerned_run(words: list[str], kern: int) -> str:
    """A single TJ run whose words are separated only by a kerning advance."""
    parts: list[str] = []
    for index, word in enumerate(words):
        if index:
            parts.append(str(kern))
        parts.append(f"({word})")
    return "[" + " ".join(parts) + "] TJ"


def _line_texts(pdf: Path) -> list[str]:
    in_doc = InputDocument(
        path_or_stream=pdf,
        format=InputFormat.PDF,
        backend=DoclingParseDocumentBackend,
        filename=pdf.name,
    )
    backend = DoclingParseDocumentBackend(in_doc, pdf)
    page = backend.load_page(0)
    try:
        return [cell.text for cell in page.get_text_cells()]
    finally:
        page.unload()
        backend.unload()


@pytest.mark.parametrize("kern", [-170, -150, -120])
def test_kerned_run_without_space_glyphs_keeps_word_boundaries(
    tmp_path: Path, kern: int
) -> None:
    """A caption written as one kerned TJ run still reads as separate words.

    Word segmentation would otherwise return the whole run as a single token,
    because the producer never painted a space glyph anywhere in the stream.
    """
    pdf = _write_pdf(tmp_path / "kerned.pdf", _kerned_run(CAPTION_WORDS, kern))

    assert _line_texts(pdf) == [CAPTION]


def test_real_space_glyphs_are_left_alone(tmp_path: Path) -> None:
    """Text the producer spaced normally is not re-spaced."""
    pdf = _write_pdf(tmp_path / "spaced.pdf", f"({CAPTION}) Tj")

    assert _line_texts(pdf) == [CAPTION]


def test_single_word_line_is_not_split(tmp_path: Path) -> None:
    """A line that legitimately holds one word keeps its normal letter spacing."""
    pdf = _write_pdf(tmp_path / "one_word.pdf", "(Netherlands) Tj")

    assert _line_texts(pdf) == ["Netherlands"]
