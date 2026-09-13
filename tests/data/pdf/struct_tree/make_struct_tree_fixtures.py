# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Regenerate the tagged-PDF fixtures used by ``tests/test_pdf_struct_tree.py``.

Run with ``python tests/data/pdf/struct_tree/make_struct_tree_fixtures.py``. The PDFs are
written by hand rather than by a library so the structure tree is explicit and reviewable:
what matters is the ``/StructTreeRoot``, the ``Formula`` structure elements, and where the
MathML sits, and no PDF writer available here emits those.

``formula_mathml_tagged.pdf`` mimics Microsoft's PDF/UA output: the MathML rides in an
``MSFT_MathML`` attribute. Its first formula also carries a ``/BBox`` layout attribute while
the second does not, so both ways of locating an element on the page are covered.
"""

from pathlib import Path

MATHML_FRAC = (
    '<math xmlns="http://www.w3.org/1998/Math/MathML">'
    "<mfrac><mi>x</mi><mi>y</mi></mfrac></math>"
)
MATHML_SUM = (
    '<math xmlns="http://www.w3.org/1998/Math/MathML">'
    "<mrow><mi>a</mi><mo>+</mo><mi>b</mi></mrow></math>"
)


def _pdf_string(text: str) -> str:
    """Escape a value for a PDF literal string."""
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _build(objs: dict[int, str | None], stream_obj: int, stream: bytes) -> bytes:
    """Assemble numbered objects into a PDF with a correct xref table."""
    out = bytearray(b"%PDF-1.7\n")
    offsets: dict[int, int] = {}
    for num in sorted(objs):
        offsets[num] = len(out)
        if num == stream_obj:
            out += f"{num} 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode()
            out += stream + b"\nendstream\nendobj\n"
        else:
            out += f"{num} 0 obj\n{objs[num]}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for num in sorted(objs):
        out += f"{offsets[num]:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode()
    return bytes(out)


def tagged_with_formulas() -> bytes:
    """Two ``Formula`` elements: the first carries /BBox, the second only marked content."""
    stream = (
        b"/Formula <</MCID 0>> BDC BT /F1 12 Tf 20 150 Td (x/y) Tj ET EMC\n"
        b"/Formula <</MCID 1>> BDC BT /F1 12 Tf 20 100 Td (a+b) Tj ET EMC"
    )
    objs: dict[int, str | None] = {
        1: (
            "<< /Type /Catalog /Pages 2 0 R /StructTreeRoot 6 0 R "
            "/MarkInfo << /Marked true >> /Lang (en-US) >>"
        ),
        2: "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        3: (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R "
            "/Resources << /Font << /F1 5 0 R >> >> /StructParents 0 >>"
        ),
        4: None,
        5: "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        6: "<< /Type /StructTreeRoot /K [7 0 R] /ParentTree 9 0 R >>",
        7: "<< /Type /StructElem /S /Document /P 6 0 R /K [8 0 R 10 0 R] >>",
        8: (
            "<< /Type /StructElem /S /Formula /P 7 0 R /Pg 3 0 R /K [0] "
            "/Alt (x over y) /A [ "
            f"<< /O /MSFT_MathML /MSFT_MathML ({_pdf_string(MATHML_FRAC)}) >> "
            "<< /O /Layout /BBox [15 140 85 170] /Placement /Block >> ] >>"
        ),
        9: "<< /Nums [0 [8 0 R 10 0 R]] >>",
        10: (
            "<< /Type /StructElem /S /Formula /P 7 0 R /Pg 3 0 R /K [1] "
            "/ActualText (a plus b) /A << /O /MSFT_MathML "
            f"/MSFT_MathML ({_pdf_string(MATHML_SUM)}) >> >>"
        ),
    }
    return _build(objs, 4, stream)


def tagged_without_formulas() -> bytes:
    """Tagged, but the only structure element is a paragraph."""
    stream = b"/P <</MCID 0>> BDC BT /F1 12 Tf 20 150 Td (hello) Tj ET EMC"
    objs: dict[int, str | None] = {
        1: (
            "<< /Type /Catalog /Pages 2 0 R /StructTreeRoot 6 0 R "
            "/MarkInfo << /Marked true >> >>"
        ),
        2: "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        3: (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R "
            "/Resources << /Font << /F1 5 0 R >> >> /StructParents 0 >>"
        ),
        4: None,
        5: "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        6: "<< /Type /StructTreeRoot /K [7 0 R] /ParentTree 9 0 R >>",
        7: "<< /Type /StructElem /S /Document /P 6 0 R /K [8 0 R] >>",
        8: "<< /Type /StructElem /S /P /P 7 0 R /Pg 3 0 R /K [0] >>",
        9: "<< /Nums [0 [8 0 R]] >>",
    }
    return _build(objs, 4, stream)


def main() -> None:
    here = Path(__file__).parent
    (here / "formula_mathml_tagged.pdf").write_bytes(tagged_with_formulas())
    (here / "formula_mathml_untagged_struct.pdf").write_bytes(tagged_without_formulas())


if __name__ == "__main__":
    main()
