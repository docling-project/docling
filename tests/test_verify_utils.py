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
    """Test that image size mismatches are detected."""
    doc_true = _make_doc_with_picture(image_size=(10, 10))
    doc_pred = _make_doc_with_picture(image_size=(15, 10))  # 5 pixel difference

    with pytest.raises(AssertionError, match="Image width mismatch"):
        verify_docitems(
            doc_pred=doc_pred,
            doc_true=doc_true,
            fuzzy=False,
            pdf_filename="fixture.json",
        )


@pytest.mark.parametrize(
    "true_size,pred_size,fuzzy,should_pass,expected_error",
    [
        # Strict mode (fuzzy=False): tolerance is 1.5% of image dimension
        # For 254x267 image: width_tol=3.8px, height_tol=4px
        (
            (254, 267),
            (251, 267),
            False,
            True,
            None,
        ),  # 3px width diff: passes (within 1.5%)
        ((254, 267), (250, 267), False, False, "Image width mismatch"),  # 4px: fails
        (
            (254, 267),
            (254, 263),
            False,
            True,
            None,
        ),  # 4px height diff: passes (within 1.5%)
        ((254, 267), (254, 262), False, False, "Image height mismatch"),  # 5px: fails
        # Fuzzy mode (fuzzy=True): tolerance is 5% of image dimension
        # For 254x267 image: width_tol=12px, height_tol=13px
        ((254, 267), (242, 254), True, True, None),  # 12-13px diff: passes (within 5%)
        ((254, 267), (241, 267), True, False, "Image width mismatch"),  # 13px: fails
        ((254, 267), (254, 253), True, False, "Image height mismatch"),  # 14px: fails
        # Small images should have at least 1 pixel tolerance
        ((10, 10), (9, 9), False, True, None),  # 1px diff on small image: passes
    ],
)
def test_verify_docitems_image_size_fuzziness(
    true_size: tuple(int, int),
    pred_size: tuple(int, int),
    fuzzy: bool,
    should_pass: bool,
    expected_error: str | None,
) -> None:
    """Test image size verification with percentage-based tolerance in strict and fuzzy modes."""
    doc_true = _make_doc_with_picture(image_size=true_size)
    doc_pred = _make_doc_with_picture(image_size=pred_size)

    if should_pass:
        verify_docitems(
            doc_pred=doc_pred,
            doc_true=doc_true,
            fuzzy=fuzzy,
            pdf_filename="fixture.json",
        )
    else:
        with pytest.raises(AssertionError, match=expected_error):
            verify_docitems(
                doc_pred=doc_pred,
                doc_true=doc_true,
                fuzzy=fuzzy,
                pdf_filename="fixture.json",
            )
