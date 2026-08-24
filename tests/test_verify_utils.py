import pytest
from docling_core.types.doc import DoclingDocument, ImageRef, ProvenanceItem
from docling_core.types.doc.base import BoundingBox, Size
from docling_core.types.doc.labels import DocItemLabel
from PIL import Image

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


def _make_doc_with_picture(*, image_size: tuple[int, int]) -> DoclingDocument:
    doc = DoclingDocument(name="test")
    doc.add_picture(
        image=ImageRef.from_pil(Image.new("RGB", image_size, "red"), dpi=72)
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
                left=500.0, page_width=2000.0, page_height=2829.0
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


def test_verify_docitems_rejects_picture_count_mismatch():
    doc_true = _make_doc_with_picture(image_size=(2, 2))
    doc_pred = DoclingDocument(name="test")

    with pytest.raises(AssertionError, match="Picture lengths do not match"):
        verify_docitems(
            doc_pred=doc_pred,
            doc_true=doc_true,
            fuzzy=False,
            pdf_filename="fixture.json",
        )


def test_verify_docitems_uses_predicted_picture_image() -> None:
    doc_true = _make_doc_with_picture(image_size=(2, 2))
    doc_pred = _make_doc_with_picture(image_size=(3, 2))

    with pytest.raises(AssertionError, match="Image width mismatch"):
        verify_docitems(
            doc_pred=doc_pred,
            doc_true=doc_true,
            fuzzy=False,
            pdf_filename="fixture.json",
        )


@pytest.mark.parametrize(
    "true_size,pred_size,should_pass,expected_error",
    [
        # Tolerance is 1.5% of image dimension.
        # For 254x267 image: 3px = 1.18% width, 4px = 1.50% height
        ((254, 267), (251, 267), True, None),  # 3px = 1.18% width: passes
        ((254, 267), (250, 267), False, "Image width mismatch"),  # 4px = 1.57%: fails
        (
            (254, 267),
            (254, 263),
            True,
            None,
        ),  # 4px = 1.50% height: passes (at boundary)
        ((254, 267), (254, 262), False, "Image height mismatch"),  # 5px = 1.87%: fails
        # Small images: percentage-based tolerance is precise
        ((10, 10), (9, 9), False, "Image width mismatch"),  # 1px = 10%: fails (>> 1.5%)
        ((100, 100), (99, 99), True, None),  # 1px = 1%: passes (< 1.5%)
    ],
)
def test_verify_docitems_image_size_strict(
    true_size: tuple[int, int],
    pred_size: tuple[int, int],
    should_pass: bool,
    expected_error: str | None,
) -> None:
    """Test image size verification with percentage-based tolerance in strict (non-fuzzy) mode."""
    doc_true = _make_doc_with_picture(image_size=true_size)
    doc_pred = _make_doc_with_picture(image_size=pred_size)

    if should_pass:
        verify_docitems(
            doc_pred=doc_pred,
            doc_true=doc_true,
            fuzzy=False,
            pdf_filename="fixture.json",
        )
    else:
        with pytest.raises(AssertionError, match=expected_error):
            verify_docitems(
                doc_pred=doc_pred,
                doc_true=doc_true,
                fuzzy=False,
                pdf_filename="fixture.json",
            )


def test_verify_docitems_fuzzy_skips_image_size_check() -> None:
    """In fuzzy mode image sizes are not compared — any size is accepted as long as
    the predicted image exists.  This covers LibreOffice-based backends
    (MsWordDocumentBackend, MsExcelDocumentBackend, MsPowerPointDocumentBackend)
    whose rendered pixel dimensions vary across platforms and LibreOffice versions.
    """
    # Wildly different sizes: would fail in strict mode but must pass in fuzzy mode.
    doc_true = _make_doc_with_picture(image_size=(254, 267))
    doc_pred = _make_doc_with_picture(image_size=(100, 50))

    verify_docitems(
        doc_pred=doc_pred,
        doc_true=doc_true,
        fuzzy=True,
        pdf_filename="fixture.json",
    )
