"""Tests for recovering extractable text the layout model mislabeled as a PICTURE.

These exercise ``LayoutPostprocessor`` end-to-end (real cell assignment, special
cluster handling and the false-picture discriminator), since the bug they guard
against is the silent dropping of programmatic text swallowed by a picture.
"""

from docling_core.types.doc import BoundingBox, CoordOrigin, DocItemLabel, Size
from docling_core.types.doc.page import (
    BitmapResource,
    BoundingRectangle,
    PdfPageBoundaryType,
    PdfPageGeometry,
    SegmentedPdfPage,
    TextCell,
)

from docling.datamodel.base_models import Cluster, Page
from docling.datamodel.pipeline_options import LayoutOptions, TextInPictureHandling

PAGE_SIZE = 100.0


def _rect(left: float, top: float, right: float, bottom: float) -> BoundingRectangle:
    return BoundingRectangle.from_bounding_box(
        BoundingBox(l=left, t=top, r=right, b=bottom, coord_origin=CoordOrigin.TOPLEFT)
    )


def _cell(index: int, box: tuple[float, float, float, float]) -> TextCell:
    return TextCell(
        index=index,
        rect=_rect(*box),
        text=f"text-{index}",
        orig=f"text-{index}",
        from_ocr=False,
    )


def _cluster(
    cid: int, label: DocItemLabel, box: tuple[float, float, float, float]
) -> Cluster:
    return Cluster(
        id=cid,
        label=label,
        bbox=BoundingBox(
            l=box[0], t=box[1], r=box[2], b=box[3], coord_origin=CoordOrigin.TOPLEFT
        ),
        confidence=0.9,
        cells=[],
    )


def _page(*, bitmaps: list[BitmapResource] | None = None) -> Page:
    full = BoundingBox(
        l=0, t=0, r=PAGE_SIZE, b=PAGE_SIZE, coord_origin=CoordOrigin.TOPLEFT
    )
    geometry = PdfPageGeometry(
        angle=0.0,
        rect=_rect(0, 0, PAGE_SIZE, PAGE_SIZE),
        boundary_type=PdfPageBoundaryType.CROP_BOX,
        art_bbox=full,
        bleed_bbox=full,
        crop_bbox=full,
        media_bbox=full,
        trim_bbox=full,
    )
    parsed = SegmentedPdfPage(
        dimension=geometry,
        char_cells=[],
        word_cells=[],
        textline_cells=[_cell(0, (10, 10, 90, 90))],
        bitmap_resources=bitmaps or [],
    )
    page = Page(page_no=0)
    page.size = Size(width=PAGE_SIZE, height=PAGE_SIZE)
    page.parsed_page = parsed
    return page


def _clusters() -> list[Cluster]:
    # A PICTURE box that fully encloses a TEXT box covering most of its area.
    return [
        _cluster(1, DocItemLabel.PICTURE, (8, 8, 92, 92)),
        _cluster(2, DocItemLabel.TEXT, (10, 10, 90, 90)),
    ]


def _run(page: Page, handling: TextInPictureHandling) -> list[Cluster]:
    from docling.utils.layout_postprocessor import LayoutPostprocessor

    options = LayoutOptions(text_in_picture_handling=handling)
    clusters, _ = LayoutPostprocessor(page, _clusters(), options).postprocess()
    return clusters


def test_absorb_keeps_default_drop_behavior() -> None:
    clusters = _run(_page(), TextInPictureHandling.ABSORB)

    pictures = [c for c in clusters if c.label == DocItemLabel.PICTURE]
    texts = [c for c in clusters if c.label == DocItemLabel.TEXT]

    assert len(pictures) == 1
    # Text is swallowed by the picture (current behavior) — no standalone text.
    assert texts == []
    assert pictures[0].cells, "absorbed text cells should live on the picture"


def test_demote_recovers_text_and_drops_false_picture() -> None:
    clusters = _run(_page(), TextInPictureHandling.DEMOTE)

    pictures = [c for c in clusters if c.label == DocItemLabel.PICTURE]
    texts = [c for c in clusters if c.label == DocItemLabel.TEXT]

    assert pictures == []
    assert len(texts) == 1
    assert any(not cell.from_ocr for cell in texts[0].cells)


def test_keep_text_retains_both_picture_and_text() -> None:
    clusters = _run(_page(), TextInPictureHandling.KEEP_TEXT)

    pictures = [c for c in clusters if c.label == DocItemLabel.PICTURE]
    texts = [c for c in clusters if c.label == DocItemLabel.TEXT]

    assert len(pictures) == 1
    assert len(texts) == 1


def test_real_bitmap_region_is_left_untouched() -> None:
    # An embedded bitmap underlies the picture: it is a genuine image, so even
    # in DEMOTE mode the text stays absorbed and the picture is preserved.
    bitmap = BitmapResource(rect=_rect(8, 8, 92, 92))
    clusters = _run(_page(bitmaps=[bitmap]), TextInPictureHandling.DEMOTE)

    pictures = [c for c in clusters if c.label == DocItemLabel.PICTURE]
    texts = [c for c in clusters if c.label == DocItemLabel.TEXT]

    assert len(pictures) == 1
    assert texts == []
