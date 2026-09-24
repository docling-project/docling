# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from docling_core.types.doc.base import BoundingBox, CoordOrigin
from docling_core.types.doc.document import DoclingDocument, ListItem, ProvenanceItem
from docling_core.types.doc.labels import DocItemLabel

from docling.models.postprocessing.list_marker_processor import (
    ListItemMarkerProcessor,
)


# Example usage and testing
def test_listitem_marker_model():
    """Example of how to use the ListItemMarkerProcessor."""

    # Create a sample document
    doc = DoclingDocument(name="Sample Document")

    doc.add_text(
        label=DocItemLabel.TEXT,
        text="• Second item with bullet and content",  # Marker and content together
        prov=ProvenanceItem(
            page_no=0,
            bbox=BoundingBox(l=0, t=15, r=200, b=25, coord_origin=CoordOrigin.TOPLEFT),
            charspan=(0, 37),
        ),
    )

    doc.add_list_item(
        text="• Third item with bullet and content",  # Marker and content together
        prov=ProvenanceItem(
            page_no=0,
            bbox=BoundingBox(l=0, t=15, r=200, b=25, coord_origin=CoordOrigin.TOPLEFT),
            charspan=(0, 37),
        ),
    )

    # Add some sample text items that should be converted to list items
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="1.",  # Marker only
        prov=ProvenanceItem(
            page_no=0,
            bbox=BoundingBox(l=0, t=0, r=10, b=10, coord_origin=CoordOrigin.TOPLEFT),
            charspan=(0, 2),
        ),
    )

    doc.add_text(
        label=DocItemLabel.TEXT,
        text="First item content",  # Content only
        prov=ProvenanceItem(
            page_no=0,
            bbox=BoundingBox(l=15, t=0, r=100, b=10, coord_origin=CoordOrigin.TOPLEFT),
            charspan=(0, 18),
        ),
    )

    # Process the document
    processor = ListItemMarkerProcessor()
    processed_doc = processor.process_document(doc, merge_items=True)

    # print(" ---------- document: \n", processed_doc.export_to_markdown(), "\n ---------- \n")

    assert len(processed_doc.texts) == 3, "len(processed_doc.texts)==3"

    assert processed_doc.texts[0].text == "• Second item with bullet and content"

    assert isinstance(processed_doc.texts[1], ListItem)
    assert processed_doc.texts[1].text == "Third item with bullet and content"
    assert processed_doc.texts[1].marker == "•"
    assert not processed_doc.texts[1].enumerated

    assert isinstance(processed_doc.texts[2], ListItem)
    assert processed_doc.texts[2].label == DocItemLabel.LIST_ITEM
    assert processed_doc.texts[2].text == "First item content"
    assert processed_doc.texts[2].marker == "1."
    assert processed_doc.texts[2].enumerated


def test_compound_list_item_markers():
    """Compound/hierarchical markers must be split off like the simple ones.

    Without these patterns the marker stays fused into the text and `marker` is left
    empty, which makes downstream Markdown serialization prepend a second, position-based
    number (e.g. "7. 9a. Compute ...").
    """
    doc = DoclingDocument(name="Compound markers")
    group = doc.add_list_group(name="list")
    for text in [
        "1. Get the minimal grid dimensions.",
        "3.a. If all IOU scores are below the threshold, discard.",
        "9a. Compute the top and bottom boundary.",
        "9b) Intersect the orphan's bounding box.",
        "(9c) Compute the left and right boundary.",
        "1.2.3 Deeply nested item.",
        "2.1. Dotted item with trailing dot.",
    ]:
        doc.add_list_item(text=text, parent=group)

    ListItemMarkerProcessor().process_document(doc)

    items = [item for item in doc.texts if isinstance(item, ListItem)]

    assert [item.marker for item in items] == [
        "1.",
        "3.a.",
        "9a.",
        "9b)",
        "(9c)",
        "1.2.3",
        "2.1.",
    ]
    assert [item.text for item in items] == [
        "Get the minimal grid dimensions.",
        "If all IOU scores are below the threshold, discard.",
        "Compute the top and bottom boundary.",
        "Intersect the orphan's bounding box.",
        "Compute the left and right boundary.",
        "Deeply nested item.",
        "Dotted item with trailing dot.",
    ]
    assert all(item.enumerated for item in items)


def test_simple_markers_unchanged_by_compound_patterns():
    """Compound patterns are matched first, so guard the simple markers against shadowing."""
    doc = DoclingDocument(name="Simple markers")
    group = doc.add_list_group(name="list")
    for text in [
        "1. One",
        "12. Twelve",
        "2) Two",
        "(3) Three",
        "[4] Four",
        "i. Five",
        "II. Six",
        "a. Seven",
        "B) Eight",
        "• Nine",
    ]:
        doc.add_list_item(text=text, parent=group)

    ListItemMarkerProcessor().process_document(doc)

    items = [item for item in doc.texts if isinstance(item, ListItem)]

    assert [item.marker for item in items] == [
        "1.",
        "12.",
        "2)",
        "(3)",
        "[4]",
        "i.",
        "II.",
        "a.",
        "B)",
        "•",
    ]
    assert [item.enumerated for item in items] == [True] * 9 + [False]


def test_multiline_item_text_is_not_truncated():
    """A list item that spans several lines keeps all of its text, not just line one."""
    doc = DoclingDocument(name="Multiline items")
    group = doc.add_list_group(name="list")
    raws = [
        "[15] Author, Title, Journal\r\nvol. 436 (2012), 2963-2965.",
        "1. First step of the procedure\r\ncontinued on the next line.",
        "• Layerwise quantization. Finds optimal weights\r\nthat minimize the loss.",
    ]
    for raw in raws:
        doc.add_list_item(text=raw, parent=group)

    ListItemMarkerProcessor().process_document(doc)

    items = [item for item in doc.texts if isinstance(item, ListItem)]

    assert [item.marker for item in items] == ["[15]", "1.", "•"]
    # marker + the separating whitespace + text must account for the whole original.
    for item in items:
        assert item.orig == f"{item.marker} {item.text}"
