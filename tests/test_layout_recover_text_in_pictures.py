"""Regression tests for issue #3699.

The layout model sometimes detects a banner-style heading inside a filled box as a
PICTURE. Its text cells are then nested under the picture and dropped from every
content layer (BODY and FURNITURE alike). With ``recover_text_in_pictures`` enabled,
a PICTURE whose area is largely covered by text-only clusters is treated as spurious:
the picture is discarded and its text clusters survive so they serialize normally.

These tests drive ``LayoutPostprocessor`` directly with synthetic clusters, so they
are deterministic and do not require the layout model or any PDF fixture.
"""

from types import SimpleNamespace

from docling_core.types.doc import DocItemLabel, Size
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.base_models import BoundingBox, Cluster
from docling.utils.layout_postprocessor import LayoutPostprocessor, SpatialClusterIndex


def _cell(index: int, bbox: tuple[float, float, float, float]) -> TextCell:
    left, top, right, bottom = bbox
    return TextCell(
        index=index,
        rect=BoundingRectangle(
            r_x0=left,
            r_y0=top,
            r_x1=right,
            r_y1=top,
            r_x2=right,
            r_y2=bottom,
            r_x3=left,
            r_y3=bottom,
        ),
        text=str(index),
        orig=str(index),
        from_ocr=False,
    )


def _cluster(
    cid: int,
    label: DocItemLabel,
    bbox: tuple[float, float, float, float],
    cells: list[TextCell] | None = None,
    confidence: float = 0.8,
) -> Cluster:
    left, top, right, bottom = bbox
    return Cluster(
        id=cid,
        label=label,
        bbox=BoundingBox(l=left, t=top, r=right, b=bottom),
        confidence=confidence,
        cells=cells or [],
    )


# --- the decision helper -----------------------------------------------------


def test_is_text_only_picture_detects_text_box() -> None:
    # A heading rendered inside a filled box: the contained text covers most of
    # the picture area -> spurious picture.
    processor = object.__new__(LayoutPostprocessor)
    picture = _cluster(1, DocItemLabel.PICTURE, (300, 30, 560, 90))
    header = _cluster(
        2,
        DocItemLabel.SECTION_HEADER,
        (318, 41, 500, 85),
        cells=[_cell(0, (318, 41, 500, 85))],
    )

    assert processor._is_text_only_picture(picture, [header]) is True


def test_is_text_only_picture_keeps_genuine_figure() -> None:
    # A real figure with only a small caption inside: low text coverage -> kept.
    processor = object.__new__(LayoutPostprocessor)
    picture = _cluster(1, DocItemLabel.PICTURE, (10, 10, 210, 210))
    caption = _cluster(
        2,
        DocItemLabel.CAPTION,
        (20, 195, 120, 207),
        cells=[_cell(0, (20, 195, 120, 207))],
    )

    assert processor._is_text_only_picture(picture, [caption]) is False


def test_is_text_only_picture_keeps_picture_with_non_text() -> None:
    # A picture containing a formula must be left alone even at high coverage.
    processor = object.__new__(LayoutPostprocessor)
    picture = _cluster(1, DocItemLabel.PICTURE, (300, 30, 560, 90))
    formula = _cluster(
        2,
        DocItemLabel.FORMULA,
        (318, 41, 500, 85),
        cells=[_cell(0, (318, 41, 500, 85))],
    )

    assert processor._is_text_only_picture(picture, [formula]) is False


# --- end-to-end through _process_special_clusters ----------------------------


def _make_processor(recover: bool) -> LayoutPostprocessor:
    """A processor with a single PICTURE drawn around a SECTION_HEADER's text."""
    processor = object.__new__(LayoutPostprocessor)
    picture = _cluster(1, DocItemLabel.PICTURE, (300, 30, 560, 90), confidence=0.63)
    header = _cluster(
        2,
        DocItemLabel.SECTION_HEADER,
        (318, 41, 500, 85),
        cells=[_cell(0, (318, 41, 500, 85))],
        confidence=0.5,
    )
    processor.regular_clusters = [header]
    processor.special_clusters = [picture]
    processor.page_size = Size(width=600, height=800)
    processor.options = SimpleNamespace(
        recover_text_in_pictures=recover, skip_cell_assignment=False
    )
    processor.picture_index = SpatialClusterIndex([picture])
    processor.wrapper_index = SpatialClusterIndex([])
    return processor


def test_enabled_drops_spurious_picture_and_keeps_text() -> None:
    processor = _make_processor(recover=True)

    result = processor._process_special_clusters()

    # The spurious picture is gone...
    assert all(c.label != DocItemLabel.PICTURE for c in result)
    # ...and it never claimed the heading as a child, so the heading remains in the
    # regular set and will serialize as a normal section header.
    absorbed = {child.id for c in result for child in c.children}
    assert 2 not in absorbed
    assert processor.regular_clusters[0].id == 2


def test_disabled_absorbs_text_into_picture_unchanged() -> None:
    processor = _make_processor(recover=False)

    result = processor._process_special_clusters()

    # Default behaviour is unchanged: the picture survives and swallows the heading.
    pictures = [c for c in result if c.label == DocItemLabel.PICTURE]
    assert len(pictures) == 1
    assert 2 in {child.id for child in pictures[0].children}
