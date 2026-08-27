# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Rejoining a drop cap with the word it begins.

The fixtures are written as raw PDF bytes rather than shipped as binary files so
that the content stream under test is visible in the diff: what matters is that
the oversized initial letter is a separate text run set at a separate font size,
and a checked-in PDF would hide that.
"""

from pathlib import Path

import pytest

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

BODY_SIZE = 12
DROP_SIZE = 34
LEFT = 72.0
BODY_BASELINE = 690.0
#: A drop cap sits on a lower baseline than the text beside it, which is what
#: makes it span two lines. Runs that share a baseline are merged by the parser
#: itself, so the offset is what puts the case under test in reach.
DROP_BASELINE = 676.0
#: Where text set beside the drop cap starts. Helvetica "R" at 34pt is 24.55pt
#: wide, so this abuts the letter with no gap, exactly as a real drop cap does.
BESIDE = 96.0

TAIL = "ealized through a combination of careful engineering"


def _show(x: float, y: float, size: int, text: str) -> str:
    """A single positioned text run."""
    return f"BT\n/F1 {size} Tf\n{x:.2f} {y:.2f} Td\n({text}) Tj\nET\n"


def _write_pdf(path: Path, content: str) -> Path:
    """Write a one-page Helvetica PDF whose content stream is ``content``."""
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


def _line_texts(path: Path) -> list[str]:
    """The text of every line cell the PDF backend produces for page one."""
    in_doc = InputDocument(
        path_or_stream=path,
        format=InputFormat.PDF,
        backend=DoclingParseDocumentBackend,
        filename=path.name,
    )
    backend = DoclingParseDocumentBackend(in_doc, path)
    try:
        return [cell.text for cell in backend.load_page(0).get_text_cells()]
    finally:
        backend.unload()


def test_drop_cap_rejoins_with_its_word(tmp_path: Path) -> None:
    """An oversized initial letter is not a word of its own."""
    content = _show(LEFT, DROP_BASELINE, DROP_SIZE, "R") + _show(
        BESIDE, BODY_BASELINE, BODY_SIZE, TAIL
    )
    lines = _line_texts(_write_pdf(tmp_path / "drop_cap.pdf", content))

    assert any(line.startswith("Realized through") for line in lines), lines
    assert not any("R ealized" in line for line in lines), lines


def test_body_size_paragraph_is_untouched(tmp_path: Path) -> None:
    """The same sentence set as one run must survive unchanged."""
    content = _show(LEFT, BODY_BASELINE, BODY_SIZE, "R" + TAIL)
    lines = _line_texts(_write_pdf(tmp_path / "plain.pdf", content))

    assert any(line.startswith("Realized through") for line in lines), lines


def test_modest_size_jump_keeps_its_space(tmp_path: Path) -> None:
    """A letter half again as tall as the body text is not a drop cap.

    Runs whose boxes touch turn up throughout ordinary layout -- the test corpus
    has them in rotated diagram labels, where two genuinely separate words abut.
    The jump in height is what tells a drop cap apart from those, so an 18pt
    letter against 12pt body text keeps its space.
    """
    content = _show(LEFT, 681, 18, "R") + _show(
        LEFT + 13, BODY_BASELINE, BODY_SIZE, TAIL
    )
    lines = _line_texts(_write_pdf(tmp_path / "modest.pdf", content))

    assert any(line.startswith("R ealized") for line in lines), lines


def test_large_initial_separated_by_a_real_gap_keeps_its_space(tmp_path: Path) -> None:
    """A big letter with white space after it is a heading, not a drop cap."""
    content = _show(LEFT, DROP_BASELINE, DROP_SIZE, "R") + _show(
        BESIDE + 12, BODY_BASELINE, BODY_SIZE, TAIL
    )
    lines = _line_texts(_write_pdf(tmp_path / "gapped.pdf", content))

    assert any(line.startswith("R ealized") for line in lines), lines


@pytest.mark.parametrize("initial", ["1", "-"])
def test_only_letters_are_treated_as_drop_caps(tmp_path: Path, initial: str) -> None:
    """A digit or a dash at the head of a line is a list marker, not a drop cap."""
    content = _show(LEFT, DROP_BASELINE, DROP_SIZE, initial) + _show(
        BESIDE, BODY_BASELINE, BODY_SIZE, TAIL
    )
    lines = _line_texts(_write_pdf(tmp_path / "marker.pdf", content))

    assert not any(line.startswith(f"{initial}ealized") for line in lines), lines
