"""Tests for PDF-bookmark / ToC heading inference and list-item promotion."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DoclingDocument,
    ProvenanceItem,
    Size,
)
from docling_core.types.doc.document import ListItem, SectionHeaderItem

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat, PdfOutlineItem
from docling.datamodel.document import InputDocument
from docling.datamodel.pipeline_options import HeadingHierarchyOptions
from docling.models.stages.heading_hierarchy.heading_hierarchy_model import (
    HeadingHierarchyModel,
    _match_score,
)
from docling.utils.pdf_outline import (
    extract_outline_from_pdfium,
    outline_from_docling_parse,
)

SAMPLE_PDF = Path("./tests/data/pdf/bookmark_sample.pdf")


def _bbox(top: float) -> BoundingBox:
    return BoundingBox(
        l=100, t=top, r=300, b=top + 15, coord_origin=CoordOrigin.TOPLEFT
    )


def _prov(page_no: int, text: str, top: float) -> ProvenanceItem:
    return ProvenanceItem(page_no=page_no, charspan=(0, len(text)), bbox=_bbox(top))


def _model(**opts) -> HeadingHierarchyModel:
    base = dict(enabled=True, use_style=False)
    base.update(opts)
    return HeadingHierarchyModel(options=HeadingHierarchyOptions(**base))


# --------------------------------------------------------------------------- matching


def test_match_score_exact_and_marker_stripped():
    # Bookmark titles routinely drop the on-page numbering marker.
    assert _match_score("1.1 Definitions", "Definitions") == pytest.approx(1.0)
    assert _match_score("PART I Introduction", "Introduction") >= 0.9


def test_match_score_truncated_containment():
    # Bookmarks are frequently truncated relative to the on-page heading.
    assert (
        _match_score(
            "Representations and Warranties of the Seller",
            "Representations and Warranties",
        )
        >= 0.9
    )


def test_match_score_rejects_unrelated():
    # Unrelated titles must score below the default match threshold (so they are ignored).
    threshold = HeadingHierarchyOptions().bookmark_match_threshold
    assert _match_score("Termination", "Definitions") < threshold
    assert _match_score("Governing Law", "Schedule of Assets") < threshold


# ----------------------------------------------------------------------- inference


def test_bookmark_promotes_listitem_and_sets_levels():
    # A heading mis-classified as a list-item ("Definitions") must be promoted and nested
    # under its parent section, exactly the case the layout model gets wrong.
    doc = DoclingDocument(name="t")
    doc.add_page(page_no=1, size=Size(width=600, height=800))
    doc.add_page(page_no=2, size=Size(width=600, height=800))
    doc.add_heading(
        text="PART I Introduction", prov=_prov(1, "PART I Introduction", 50)
    )
    group = doc.add_list_group(name="list")
    doc.add_list_item(
        text="Definitions", parent=group, prov=_prov(1, "Definitions", 120)
    )
    doc.add_heading(text="Conclusion", prov=_prov(2, "Conclusion", 50))

    outline = [
        PdfOutlineItem(title="Introduction", level=0, page_no=1),
        PdfOutlineItem(title="Definitions", level=1, page_no=1),
        PdfOutlineItem(title="Conclusion", level=0, page_no=2),
    ]
    _model().assign_heading_levels(doc, outline=outline)

    headings = {h.text: h.level for h in doc.texts if isinstance(h, SectionHeaderItem)}
    assert headings == {
        "PART I Introduction": 1,
        "Definitions": 2,  # promoted list-item, nested one level deeper
        "Conclusion": 1,
    }
    # the original list-item is gone, replaced by the heading
    assert not any(
        isinstance(i, ListItem) and i.text == "Definitions" for i in doc.texts
    )


def test_bookmark_overrides_numbering():
    # Numbering alone would set "1. Scope" to level 1 and leave "Preamble" unlevelled;
    # a confident bookmark match is authoritative and wins.
    doc = DoclingDocument(name="t")
    doc.add_page(page_no=1, size=Size(width=600, height=800))
    doc.add_heading(text="Preamble", prov=_prov(1, "Preamble", 40))
    doc.add_heading(text="1. Scope", prov=_prov(1, "1. Scope", 120))

    outline = [
        PdfOutlineItem(title="Preamble", level=0, page_no=1),
        PdfOutlineItem(title="Scope", level=1, page_no=1),
    ]
    _model().assign_heading_levels(doc, outline=outline)

    assert {h.text: h.level for h in doc.texts} == {"Preamble": 1, "1. Scope": 2}


def test_unmatched_bookmark_falls_back_to_numbering():
    # A noisy/partial outline that matches nothing must not degrade the numbering result.
    doc = DoclingDocument(name="t")
    doc.add_page(page_no=1, size=Size(width=600, height=800))
    doc.add_heading(text="1. Scope", prov=_prov(1, "1. Scope", 40))
    doc.add_heading(text="1.1 Definitions", prov=_prov(1, "1.1 Definitions", 120))

    outline = [PdfOutlineItem(title="Totally Unrelated Bookmark", level=0, page_no=1)]
    _model().assign_heading_levels(doc, outline=outline)

    assert [h.level for h in doc.texts] == [1, 2]  # numbering still applies


def test_wrong_page_bookmark_does_not_match():
    # Same title but a different target page must not be matched.
    doc = DoclingDocument(name="t")
    doc.add_page(page_no=1, size=Size(width=600, height=800))
    doc.add_page(page_no=2, size=Size(width=600, height=800))
    doc.add_heading(text="Scope", prov=_prov(1, "Scope", 40))

    outline = [PdfOutlineItem(title="Scope", level=0, page_no=2)]
    _model().assign_heading_levels(doc, outline=outline)

    # No numbering, no style, no matched bookmark -> level unchanged.
    assert [h.level for h in doc.texts] == [1]


def test_use_bookmarks_false_ignores_outline():
    doc = DoclingDocument(name="t")
    doc.add_page(page_no=1, size=Size(width=600, height=800))
    doc.add_heading(text="Alpha", prov=_prov(1, "Alpha", 40))
    doc.add_heading(text="Beta", prov=_prov(1, "Beta", 120))

    outline = [
        PdfOutlineItem(title="Alpha", level=0, page_no=1),
        PdfOutlineItem(title="Beta", level=1, page_no=1),
    ]
    model = _model(use_numbering=False, use_bookmarks=False)
    model.assign_heading_levels(doc, outline=outline)

    assert [h.level for h in doc.texts] == [1, 1]  # outline ignored, nothing applies


# ----------------------------------------------------------------------- extraction


def test_extract_outline_from_generated_pdf(tmp_path):
    reportlab_canvas = pytest.importorskip("reportlab.pdfgen.canvas")
    import pypdfium2 as pdfium
    from reportlab.lib.pagesizes import letter

    path = tmp_path / "outlined.pdf"
    c = reportlab_canvas.Canvas(str(path), pagesize=letter)
    for key, title, level, page_text in [
        ("p1", "Chapter 1", 0, "Chapter 1"),
        ("p2", "Section 1.1", 1, "Section 1.1"),
        ("p3", "Chapter 2", 0, "Chapter 2"),
    ]:
        c.bookmarkPage(key)
        c.addOutlineEntry(title, key, level=level)
        c.drawString(72, 720, page_text)
        c.showPage()
    c.save()

    pdoc = pdfium.PdfDocument(str(path))
    items = [(i.title, i.level, i.page_no) for i in extract_outline_from_pdfium(pdoc)]

    assert items == [
        ("Chapter 1", 0, 1),
        ("Section 1.1", 1, 2),
        ("Chapter 2", 0, 3),
    ]


# ------------------------------------------------------------------------- wiring


def test_call_reads_outline_from_conversion_result():
    # __call__ must pull the outline off ConversionResult.pdf_outline and honor use_bookmarks.
    doc = DoclingDocument(name="t")
    doc.add_page(page_no=1, size=Size(width=600, height=800))
    doc.add_heading(text="Alpha", prov=_prov(1, "Alpha", 40))
    doc.add_heading(text="Beta", prov=_prov(1, "Beta", 120))
    conv_res = SimpleNamespace(
        document=doc,
        pages=[],
        pdf_outline=[
            PdfOutlineItem(title="Alpha", level=0, page_no=1),
            PdfOutlineItem(title="Beta", level=1, page_no=1),
        ],
    )

    model = HeadingHierarchyModel(
        options=HeadingHierarchyOptions(
            enabled=True, use_numbering=False, use_style=False
        )
    )
    out = model(conv_res)
    assert [h.level for h in out.texts] == [
        1,
        2,
    ]  # hierarchy comes solely from bookmarks


# --------------------------------------------------------------------- real PDF


# Expected outline tree shared by both backend extractors (pypdfium2 adds page/position).
EXPECTED_OUTLINE = [
    ("PART I - DEFINITIONS", 0),
    ("1. Interpretation", 1),
    ("2. Construction of Terms", 1),
    ("PART II - OBLIGATIONS", 0),
    ("3. Payment Terms", 1),
    ("3.1 Payment Schedule", 2),
    ("4. Termination", 1),
    ("PART III - MISCELLANEOUS", 0),
]


def test_pypdfium_backend_outline_from_sample_pdf():
    # pypdfium2 backend: rich extraction with title, depth, target page and vertical position.
    in_doc = InputDocument(
        path_or_stream=SAMPLE_PDF,
        format=InputFormat.PDF,
        backend=PyPdfiumDocumentBackend,
    )
    outline = in_doc._backend.get_document_outline()

    assert [(o.title, o.level) for o in outline] == EXPECTED_OUTLINE
    assert [o.page_no for o in outline] == [1, 1, 1, 2, 2, 2, 3, 3]
    # XYZ destinations carry a vertical target, captured as a top-left-origin y_top.
    assert all(o.y_top is not None and o.y_top > 0 for o in outline)


def test_docling_parse_native_outline_from_sample_pdf():
    # docling-parse backends use the native get_table_of_contents() (no pypdfium2). It carries
    # titles + hierarchy only, so page_no/y_top are None. Loaded via the parser directly because
    # the same tree drives DoclingParseDocumentBackend.get_document_outline().
    from docling_parse.pdf_parser import DoclingPdfParser

    dp_doc = DoclingPdfParser(loglevel="fatal").load(str(SAMPLE_PDF))
    try:
        outline = outline_from_docling_parse(dp_doc)
    finally:
        dp_doc.unload()

    assert [(o.title, o.level) for o in outline] == EXPECTED_OUTLINE
    assert all(o.page_no is None and o.y_top is None for o in outline)
