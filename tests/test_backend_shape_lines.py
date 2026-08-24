"""Tests for DoclingParsePageBackend.get_shape_lines (issue #4028).

The base-class contract declares get_shape_lines but the docling-parse page
backend returned None ("cannot answer"). Ruled tables commonly draw their row
separators either as stroked lines or as thin filled rectangles; both must be
reported so that table row reconciliation can see them.
"""

from pathlib import Path

import pytest

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

PAGE_W, PAGE_H = 595.0, 842.0


def _write_pdf(path: Path, content: str) -> None:
    """Write a single-page PDF with the given content stream (stdlib only)."""
    stream = content.encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] "
            "/Resources << >> /Contents 4 0 R >>"
        ).encode("latin-1"),
        b"<< /Length "
        + str(len(stream)).encode()
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n"
    ).encode()
    path.write_bytes(bytes(out))


@pytest.fixture
def ruled_page(tmp_path: Path) -> Path:
    """One filled-rectangle rule, one stroked rule, and one vertical stroke.

    PDF y-coordinates are bottom-up; a rule at pdf y=742 is at y=100 in
    docling's top-left origin.
    """
    pdf_path = tmp_path / "rules.pdf"
    _write_pdf(
        pdf_path,
        "\n".join(
            [
                "70.00 741.75 460.00 0.5 re f",  # filled rect rule, top-left y≈100
                "70.00 642.00 m 530.00 642.00 l 0.5 w S",  # stroked rule, y=200
                "70.00 100.00 m 70.00 700.00 l 0.5 w S",  # vertical stroke, x=70
            ]
        ),
    )
    return pdf_path


def _load_page(pdf_path: Path):
    in_doc = InputDocument(
        path_or_stream=pdf_path,
        format=InputFormat.PDF,
        backend=DoclingParseDocumentBackend,
    )
    return in_doc._backend.load_page(0)


def test_reports_filled_and_stroked_horizontal_rules(ruled_page: Path):
    page = _load_page(ruled_page)
    lines = page.get_shape_lines(horizontal=True, vertical=False)
    assert lines is not None
    ys = sorted(line.t for line in lines)
    assert len(ys) == 2
    assert ys[0] == pytest.approx(100.0, abs=1.0)
    assert ys[1] == pytest.approx(200.0, abs=1.0)
    for line in lines:
        # Degenerate (zero-height) boxes spanning the drawn width.
        assert line.b == pytest.approx(line.t, abs=1.0)
        assert line.r - line.l == pytest.approx(460.0, abs=2.0)


def test_reports_vertical_rules_separately(ruled_page: Path):
    page = _load_page(ruled_page)
    lines = page.get_shape_lines(horizontal=False, vertical=True)
    assert lines is not None
    assert len(lines) == 1
    assert lines[0].l == pytest.approx(70.0, abs=1.0)
    assert lines[0].r == pytest.approx(lines[0].l, abs=1.0)


def test_text_is_not_reported_as_shape_lines(tmp_path: Path):
    pdf_path = tmp_path / "no-rules.pdf"
    _write_pdf(pdf_path, "BT 100 700 Td ET")
    page = _load_page(pdf_path)
    assert page.get_shape_lines(horizontal=True, vertical=True) == []
