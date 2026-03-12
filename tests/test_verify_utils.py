import pytest
from docling_core.types.doc import DoclingDocument, ProvenanceItem
from docling_core.types.doc.base import BoundingBox
from docling_core.types.doc.labels import DocItemLabel

from tests.verify_utils import verify_docitems


def _make_doc_with_bbox(*, left: float) -> DoclingDocument:
    doc = DoclingDocument(name="test")
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
        doc_pred=_make_doc_with_bbox(left=10.06),
        doc_true=_make_doc_with_bbox(left=10.0),
        fuzzy=False,
        pdf_filename="fixture.json",
    )


def test_verify_docitems_rejects_large_bbox_variance_for_non_fuzzy_docs():
    with pytest.raises(AssertionError, match="BBox left mismatch"):
        verify_docitems(
            doc_pred=_make_doc_with_bbox(left=10.11),
            doc_true=_make_doc_with_bbox(left=10.0),
            fuzzy=False,
            pdf_filename="fixture.json",
        )


def test_verify_docitems_allows_reasonable_bbox_variance_for_fuzzy_docs():
    verify_docitems(
        doc_pred=_make_doc_with_bbox(left=12.65),
        doc_true=_make_doc_with_bbox(left=10.0),
        fuzzy=True,
        pdf_filename="fixture.json",
    )


def test_verify_docitems_rejects_gross_bbox_variance_for_fuzzy_docs():
    with pytest.raises(AssertionError, match="BBox left mismatch"):
        verify_docitems(
            doc_pred=_make_doc_with_bbox(left=20.01),
            doc_true=_make_doc_with_bbox(left=10.0),
            fuzzy=True,
            pdf_filename="fixture.json",
        )
