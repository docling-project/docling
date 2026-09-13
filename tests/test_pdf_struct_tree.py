# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Reading ``Formula`` structure elements out of a tagged PDF.

The fixtures under ``tests/data/pdf/struct_tree/`` are minimal hand-built tagged PDFs; see
``make_struct_tree_fixtures.py`` in the same directory for how they are produced. They live
outside ``tests/data/pdf/sources/`` on purpose -- that tree is swept by the end-to-end
conversion test, which would demand groundtruth for them and run the layout model over two
lines of text to no purpose.
"""

from pathlib import Path

import pypdfium2 as pdfium
import pytest
from docling_core.types.doc import CoordOrigin

from docling.backend.docling_parse_backend import (
    DoclingParseDocumentBackend,
    ThreadedDoclingParseDocumentBackend,
)
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.utils.pdf_struct_tree import (
    _sanitize_mathml,
    extract_formula_structs_from_pdfium,
)

TAGGED = Path("tests/data/pdf/struct_tree/formula_mathml_tagged.pdf")
TAGGED_NO_FORMULA = Path(
    "tests/data/pdf/struct_tree/formula_mathml_untagged_struct.pdf"
)
UNTAGGED = Path("tests/data/pdf/sources/2203.01017v2.pdf")


def _extract(path: Path, page_nos=(1,)):
    doc = pdfium.PdfDocument(path)
    try:
        return extract_formula_structs_from_pdfium(doc, page_nos)
    finally:
        doc.close()


def test_extracts_both_formula_elements():
    records = _extract(TAGGED)

    assert len(records) == 2
    assert all(r.page_no == 1 for r in records)
    assert "<mfrac><mi>x</mi><mi>y</mi></mfrac>" in records[0].mathml
    assert "<mi>a</mi><mo>+</mo><mi>b</mi>" in records[1].mathml


def test_reads_alt_and_actual_text():
    first, second = _extract(TAGGED)

    # The fixture gives the first element /Alt and the second /ActualText, so both getters
    # are exercised. PDFium NUL-pads these buffers; the decode must strip that.
    assert first.alt_text == "x over y"
    assert first.actual_text is None
    assert second.actual_text == "a plus b"
    assert second.alt_text is None


def test_bbox_attribute_is_converted_to_top_left_origin():
    """The first element carries ``/BBox [15 140 85 170]`` on a 200pt-high page."""
    first = _extract(TAGGED)[0]

    assert first.bbox is not None
    assert first.bbox.coord_origin == CoordOrigin.TOPLEFT
    assert (first.bbox.l, first.bbox.t, first.bbox.r, first.bbox.b) == (15, 30, 85, 60)


def test_bbox_falls_back_to_marked_content_bounds():
    """The second element has no ``BBox``; its box comes from the text it marks."""
    second = _extract(TAGGED)[1]

    assert second.bbox is not None
    assert second.bbox.coord_origin == CoordOrigin.TOPLEFT
    # The run is drawn at (20, 100) in PDF space on a 200pt page.
    assert 18 < second.bbox.l < 25
    assert 88 < second.bbox.t < 102


def test_tagged_pdf_without_formula_elements_yields_nothing():
    assert _extract(TAGGED_NO_FORMULA) == []


def test_untagged_pdf_yields_nothing():
    assert _extract(UNTAGGED, page_nos=(1, 2)) == []


def test_pages_outside_the_document_are_ignored():
    assert _extract(TAGGED, page_nos=(0, 5, 99)) == []


def test_only_requested_pages_are_read():
    assert _extract(TAGGED, page_nos=()) == []


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "   ",
        "<math><mfrac>",  # not well formed
        "<div><p>not mathml</p></div>",  # wrong root
        "<math><script>alert(1)</script></math>",  # script smuggling
        '<math><annotation-xml encoding="text/html"><b>x</b></annotation-xml></math>',
    ],
)
def test_unusable_mathml_is_rejected(payload: str):
    assert _sanitize_mathml(payload) is None


@pytest.mark.parametrize(
    "payload",
    [
        "<math><mi>x</mi></math>",
        '<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>x</mi></math>',
        "  <math><mi>x</mi></math>  ",
    ],
)
def test_usable_mathml_is_accepted(payload: str):
    assert _sanitize_mathml(payload) == payload.strip()


def _backend(cls, path: Path):
    in_doc = InputDocument(
        path_or_stream=path,
        format=InputFormat.PDF,
        backend=cls,
        filename=path.name,
    )
    return cls(in_doc, path)


@pytest.mark.parametrize(
    "backend_cls",
    [
        DoclingParseDocumentBackend,
        ThreadedDoclingParseDocumentBackend,
        PyPdfiumDocumentBackend,
    ],
)
def test_every_pdf_backend_exposes_the_same_records(backend_cls):
    """Including the threaded backend, which holds no pypdfium2 handle of its own."""
    backend = _backend(backend_cls, TAGGED)
    try:
        records = backend.get_formula_structures([1])
    finally:
        backend.unload()

    assert len(records) == 2
    assert [r.mathml for r in records] == [r.mathml for r in _extract(TAGGED)]


@pytest.mark.parametrize(
    "backend_cls",
    [
        DoclingParseDocumentBackend,
        ThreadedDoclingParseDocumentBackend,
        PyPdfiumDocumentBackend,
    ],
)
def test_backends_return_nothing_for_a_tagged_pdf_without_formulas(backend_cls):
    backend = _backend(backend_cls, TAGGED_NO_FORMULA)
    try:
        assert backend.get_formula_structures([1]) == []
    finally:
        backend.unload()
