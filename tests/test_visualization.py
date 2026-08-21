"""Tests for the layout debug visualisation helpers.

These only run when a ``settings.debug.visualize_*`` flag is set, which no
other test does, so the drawing code is otherwise never executed. The
assertions target what the helpers are for: producing a side-by-side image
that splits region clusters from the rest, written where the debug settings
point.
"""

from pathlib import Path

import pytest
from docling_core.types.doc import BoundingBox, CoordOrigin, DocItemLabel, Size
from docling_core.types.doc.page import BoundingRectangle, TextCell
from PIL import Image

from docling.datamodel.base_models import Cluster, Page
from docling.datamodel.settings import settings
from docling.utils.visualization import (
    draw_clusters,
    draw_clusters_and_cells_side_by_side,
)

PAGE_WIDTH = 120
PAGE_HEIGHT = 80


def _cell(x0: float, y0: float, x1: float, y1: float, text: str = "cell") -> TextCell:
    return TextCell(
        index=0,
        text=text,
        orig=text,
        from_ocr=False,
        rect=BoundingRectangle.from_bounding_box(
            BoundingBox(l=x0, t=y0, r=x1, b=y1, coord_origin=CoordOrigin.TOPLEFT)
        ),
    )


def _cluster(
    cluster_id: int,
    label: DocItemLabel,
    bbox: tuple[float, float, float, float] = (10, 10, 60, 40),
    confidence: float = 0.9,
    cells: list[TextCell] | None = None,
    children: list[Cluster] | None = None,
) -> Cluster:
    x0, y0, x1, y1 = bbox
    return Cluster(
        id=cluster_id,
        label=label,
        bbox=BoundingBox(l=x0, t=y0, r=x1, b=y1, coord_origin=CoordOrigin.TOPLEFT),
        confidence=confidence,
        cells=cells or [],
        children=children or [],
    )


def _page_with_image() -> Page:
    page = Page(page_no=0, size=Size(width=PAGE_WIDTH, height=PAGE_HEIGHT))
    image = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), color=(255, 255, 255))
    page._image_cache = {1.0: image}
    return page


def test_draw_clusters_marks_the_cluster_area():
    image = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), color=(255, 255, 255))
    before = image.copy()

    draw_clusters(image, [_cluster(1, DocItemLabel.TEXT)], scale_x=1.0, scale_y=1.0)

    assert image.tobytes() != before.tobytes()
    # A pixel well inside the cluster is tinted; one outside is untouched.
    assert image.getpixel((40, 30)) != (255, 255, 255)
    assert image.getpixel((110, 75)) == (255, 255, 255)


def test_draw_clusters_on_an_empty_list_leaves_the_image_untouched():
    image = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), color=(255, 255, 255))
    before = image.copy()

    draw_clusters(image, [], scale_x=1.0, scale_y=1.0)

    assert image.tobytes() == before.tobytes()


def test_draw_clusters_renders_children_and_cells():
    child = _cluster(2, DocItemLabel.LIST_ITEM, bbox=(70, 50, 110, 70))
    parent = _cluster(
        1,
        DocItemLabel.TEXT,
        cells=[_cell(12, 12, 55, 20)],
        children=[child],
    )
    image = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), color=(255, 255, 255))

    draw_clusters(image, [parent], scale_x=1.0, scale_y=1.0)

    # The child cluster sits outside the parent box and must also be drawn.
    assert image.getpixel((90, 60)) != (255, 255, 255)


def test_draw_clusters_normalises_inverted_boxes():
    """A box given bottom-right first must still be drawn, not skipped."""
    inverted = _cluster(1, DocItemLabel.TEXT, bbox=(60, 40, 10, 10))
    image = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), color=(255, 255, 255))

    draw_clusters(image, [inverted], scale_x=1.0, scale_y=1.0)

    assert image.getpixel((40, 30)) != (255, 255, 255)


def test_draw_clusters_applies_the_scale_factors():
    image = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), color=(255, 255, 255))
    # At half scale the same cluster covers a quarter of the area, so a point
    # inside the unscaled box falls outside the scaled one.
    draw_clusters(image, [_cluster(1, DocItemLabel.TEXT)], scale_x=0.5, scale_y=0.5)

    assert image.getpixel((15, 15)) != (255, 255, 255)
    assert image.getpixel((50, 35)) == (255, 255, 255)


@pytest.fixture
def debug_output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(settings.debug, "debug_output_path", str(tmp_path))
    return tmp_path


def test_side_by_side_writes_a_double_width_image(debug_output_dir):
    page = _page_with_image()

    draw_clusters_and_cells_side_by_side(
        input_file=Path("report.pdf"),
        page=page,
        clusters=[_cluster(1, DocItemLabel.TEXT)],
        mode_prefix="postprocessed",
    )

    out_file = debug_output_dir / "debug_report" / "postprocessed_layout_page_00000.png"
    assert out_file.exists()
    with Image.open(out_file) as written:
        assert written.size == (PAGE_WIDTH * 2, PAGE_HEIGHT)


def test_side_by_side_splits_region_labels_onto_the_right_half(debug_output_dir):
    """FORM, KEY_VALUE_REGION and PICTURE go right; everything else goes left."""
    page = _page_with_image()

    draw_clusters_and_cells_side_by_side(
        input_file=Path("report.pdf"),
        page=page,
        clusters=[
            _cluster(1, DocItemLabel.TEXT, bbox=(10, 10, 50, 30)),
            _cluster(2, DocItemLabel.PICTURE, bbox=(10, 10, 50, 30)),
        ],
        mode_prefix="raw",
    )

    out_file = debug_output_dir / "debug_report" / "raw_layout_page_00000.png"
    with Image.open(out_file) as written:
        left = written.crop((0, 0, PAGE_WIDTH, PAGE_HEIGHT))
        right = written.crop((PAGE_WIDTH, 0, PAGE_WIDTH * 2, PAGE_HEIGHT))
        # Both halves are drawn on, but with different label colours.
        assert left.getpixel((30, 20)) != (255, 255, 255)
        assert right.getpixel((30, 20)) != (255, 255, 255)
        assert left.getpixel((30, 20)) != right.getpixel((30, 20))


def test_side_by_side_names_the_file_after_the_page_number(debug_output_dir):
    page = _page_with_image()
    page.page_no = 12

    draw_clusters_and_cells_side_by_side(
        input_file=Path("nested/dir/report.pdf"),
        page=page,
        clusters=[],
        mode_prefix="raw",
    )

    # The directory is derived from the file stem, dropping any parent path.
    assert (debug_output_dir / "debug_report" / "raw_layout_page_00012.png").exists()


def test_side_by_side_reuses_an_existing_debug_directory(debug_output_dir):
    page = _page_with_image()
    out_dir = debug_output_dir / "debug_report"
    out_dir.mkdir(parents=True)
    (out_dir / "sentinel.txt").write_text("kept")

    draw_clusters_and_cells_side_by_side(
        input_file=Path("report.pdf"),
        page=page,
        clusters=[],
        mode_prefix="raw",
    )

    assert (out_dir / "sentinel.txt").read_text() == "kept"
    assert (out_dir / "raw_layout_page_00000.png").exists()


def test_side_by_side_show_mode_does_not_write_a_file(
    debug_output_dir, monkeypatch: pytest.MonkeyPatch
):
    page = _page_with_image()
    shown: list[Image.Image] = []
    monkeypatch.setattr(
        Image.Image, "show", lambda self, *args, **kwargs: shown.append(self)
    )

    draw_clusters_and_cells_side_by_side(
        input_file=Path("report.pdf"),
        page=page,
        clusters=[],
        mode_prefix="raw",
        show=True,
    )

    assert len(shown) == 1
    assert not (debug_output_dir / "debug_report").exists()
