import pytest
from docling_core.types.doc import DoclingDocument, ProvenanceItem
from docling_core.types.doc.base import BoundingBox, Size
from docling_core.types.doc.labels import DocItemLabel

from tests.verify_utils import verify_docitems


def _make_doc_with_bbox(
    *, left: float, page_width: float = 612.0, page_height: float = 792.0
) -> DoclingDocument:
    doc = DoclingDocument(name="test")
    doc.add_page(page_no=1, size=Size(width=page_width, height=page_height))
    doc.add_text(
        label=DocItemLabel.PARAGRAPH,
        text="bbox check",
        orig="bbox check",
        prov=ProvenanceItem(
            page_no=1,
            bbox=BoundingBox(l=left, t=20.0, r=30.0, b=40.0),
            charspan=(0, 10),
        ),
    )
    return doc


def test_verify_docitems_allows_small_bbox_variance_for_non_fuzzy_docs():
    verify_docitems(
        doc_pred=_make_doc_with_bbox(left=11.53),
        doc_true=_make_doc_with_bbox(left=10.0),
        fuzzy=False,
        pdf_filename="fixture.json",
    )


def test_verify_docitems_rejects_large_bbox_variance_for_non_fuzzy_docs():
    with pytest.raises(AssertionError, match="BBox left mismatch"):
        verify_docitems(
            doc_pred=_make_doc_with_bbox(left=12.01),
            doc_true=_make_doc_with_bbox(left=10.0),
            fuzzy=False,
            pdf_filename="fixture.json",
        )


def test_verify_docitems_allows_reasonable_bbox_variance_for_fuzzy_docs():
    verify_docitems(
        doc_pred=_make_doc_with_bbox(left=17.23, page_width=2000.0, page_height=2829.0),
        doc_true=_make_doc_with_bbox(left=10.0, page_width=2000.0, page_height=2829.0),
        fuzzy=True,
        pdf_filename="fixture.json",
    )


def test_verify_docitems_rejects_gross_bbox_variance_for_fuzzy_docs():
    with pytest.raises(AssertionError, match="BBox left mismatch"):
        verify_docitems(
            doc_pred=_make_doc_with_bbox(
                left=25.0, page_width=2000.0, page_height=2829.0
            ),
            doc_true=_make_doc_with_bbox(
                left=10.0, page_width=2000.0, page_height=2829.0
            ),
            fuzzy=True,
            pdf_filename="fixture.json",
        )


def test_verify_docitems_rejects_bbox_presence_mismatch():
    doc_true = _make_doc_with_bbox(left=10.0)
    doc_pred = _make_doc_with_bbox(left=10.0)
    doc_pred.texts[0].prov[0].bbox = None

    with pytest.raises(AssertionError, match="BBox presence mismatch"):
        verify_docitems(
            doc_pred=doc_pred,
            doc_true=doc_true,
            fuzzy=False,
            pdf_filename="fixture.json",
        )
