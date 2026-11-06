# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for keeping PowerPoint shape groups in the document tree."""

from io import BytesIO
from pathlib import Path

from docling_core.types.doc import (
    DoclingDocument,
    GroupItem,
    GroupLabel,
    PictureItem,
    TextItem,
)
from PIL import Image
from pptx import Presentation
from pptx.enum.shapes import MSO_CONNECTOR
from pptx.util import Inches

from .test_backend_pptx import get_converter


def _add_text(shapes, text: str, top: float) -> None:
    box = shapes.add_textbox(Inches(1), Inches(top), Inches(4), Inches(0.5))
    box.text_frame.text = text


def _tree(doc: DoclingDocument) -> list[tuple[int, str]]:
    """Return the slide content as (depth, group name, text or "picture") pairs."""
    tree = []
    for item, depth in doc.iterate_items(with_groups=True):
        if item is doc.body:
            continue
        if isinstance(item, GroupItem) and item.label == GroupLabel.UNSPECIFIED:
            tree.append((depth, item.name))
        elif isinstance(item, TextItem):
            tree.append((depth, item.text))
        elif isinstance(item, PictureItem):
            tree.append((depth, "picture"))
    return tree


def test_pptx_grouped_shapes_stay_in_their_group(tmp_path: Path):
    """A picture grouped with its label stays with it, nested groups too."""
    png = BytesIO()
    Image.new("RGB", (40, 20), (200, 30, 30)).save(png, format="PNG")

    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_text(slide.shapes, "Before", 0.5)
    group = slide.shapes.add_group_shape()
    group.name = "Figure"
    group.shapes.add_picture(png, Inches(1), Inches(1.5), Inches(2), Inches(1))
    _add_text(group.shapes, "Figure label", 2.6)
    legend = group.shapes.add_group_shape()
    legend.name = "Legend"
    _add_text(legend.shapes, "Legend text", 3.5)
    _add_text(slide.shapes, "After", 5)
    pptx_path = tmp_path / "groups.pptx"
    prs.save(pptx_path)

    doc = get_converter().convert(pptx_path).document

    assert _tree(doc) == [
        (2, "Before"),
        (2, "Figure"),
        (3, "picture"),
        (3, "Figure label"),
        (3, "Legend"),
        (4, "Legend text"),
        (2, "After"),
    ]
    assert doc.export_to_markdown() == (
        "Before\n\n<!-- image -->\n\nFigure label\n\nLegend text\n\nAfter"
    )


def test_pptx_group_without_content_is_left_out(tmp_path: Path):
    """A group of connectors only, also inside another group, adds no group."""
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    lines = slide.shapes.add_group_shape()
    lines.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(1), Inches(1), Inches(2), Inches(2)
    )
    outer = slide.shapes.add_group_shape()
    nested_lines = outer.shapes.add_group_shape()
    nested_lines.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(3), Inches(1), Inches(4), Inches(2)
    )
    kept = slide.shapes.add_group_shape()
    kept.name = "Kept"
    _add_text(kept.shapes, "Text", 3)
    pptx_path = tmp_path / "empty_groups.pptx"
    prs.save(pptx_path)

    doc = get_converter().convert(pptx_path).document

    assert _tree(doc) == [(2, "Kept"), (3, "Text")]
    DoclingDocument.model_validate(doc.export_to_dict())
