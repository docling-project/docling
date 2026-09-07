# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from docling_core.types.doc import CoordOrigin, Size
from docling_core.types.doc.page import (
    Coord2D,
    PdfDestination,
    PdfDestinationKind,
    PdfTableOfContents,
)

from docling.utils.pdf_outline import extract_outline_from_docling_parse

PAGE_SIZE = Size(width=612.0, height=792.0)


def _node(
    text: str,
    *,
    children: list[PdfTableOfContents] | None = None,
    destination: PdfDestination | None = None,
) -> PdfTableOfContents:
    return PdfTableOfContents(
        text=text, destination=destination, children=children or []
    )


def _dest(
    page_no: int,
    *,
    kind: PdfDestinationKind = PdfDestinationKind.XYZ,
    y_bottom_left: float | None = 726.0,
) -> PdfDestination:
    """A destination as docling-parse reports it: the target page's own frame, bottom-left."""
    return PdfDestination(
        page_no=page_no,
        kind=kind,
        point=None if y_bottom_left is None else Coord2D(x=0.0, y=y_bottom_left),
        coord_origin=CoordOrigin.BOTTOMLEFT,
        page_size=PAGE_SIZE,
    )


def _build_chain(depth: int) -> PdfTableOfContents:
    """Build a linear chain of nested nodes depth levels deep:
    root -> child -> child -> ... (depth - 1 named children below root)."""
    node = _node(f"level_{depth - 1}")
    for i in range(depth - 2, -1, -1):
        node = _node(f"level_{i}", children=[node])
    return node


def test_outline_no_toc_returns_empty_list():
    assert extract_outline_from_docling_parse(None) == []


def test_outline_flat_structure():
    root = _node("root", children=[_node("First"), _node("Second"), _node("Third")])
    items = extract_outline_from_docling_parse(root)
    assert [(item.title, item.level) for item in items] == [
        ("First", 0),
        ("Second", 0),
        ("Third", 0),
    ]


def test_outline_nested_structure_preserves_order_and_levels():
    root = _node(
        "root",
        children=[
            _node("Chapter 1", children=[_node("1.1"), _node("1.2")]),
            _node("Chapter 2"),
        ],
    )
    items = extract_outline_from_docling_parse(root)
    assert [(item.title, item.level) for item in items] == [
        ("Chapter 1", 0),
        ("1.1", 1),
        ("1.2", 1),
        ("Chapter 2", 0),
    ]


def test_outline_blank_and_whitespace_titles_are_excluded():
    root = _node("root", children=[_node(""), _node("   "), _node("  Real Title  ")])
    items = extract_outline_from_docling_parse(root)
    assert [(item.title, item.level) for item in items] == [("Real Title", 0)]


def test_outline_untitled_node_still_deepens_its_children():
    """A skipped title must not collapse the level of the subtree below it."""
    root = _node("root", children=[_node("", children=[_node("Buried")])])
    items = extract_outline_from_docling_parse(root)
    assert [(item.title, item.level) for item in items] == [("Buried", 1)]


def test_destination_yields_page_and_top_left_position():
    """docling-parse reports bottom-left coordinates; matching needs a top-left origin."""
    root = _node("root", children=[_node("Chapter 1", destination=_dest(3))])
    (item,) = extract_outline_from_docling_parse(root)
    assert item.page_no == 3
    assert item.y_top == PAGE_SIZE.height - 726.0


def test_destination_without_a_position_still_yields_its_page():
    """FIT and FIT_B encode no coordinate, but the target page is still authoritative."""
    root = _node(
        "root",
        children=[
            _node(
                "Chapter 1",
                destination=_dest(2, kind=PdfDestinationKind.FIT, y_bottom_left=None),
            )
        ],
    )
    (item,) = extract_outline_from_docling_parse(root)
    assert item.page_no == 2
    assert item.y_top is None


def test_entry_without_a_destination_leaves_page_and_position_unset():
    root = _node("root", children=[_node("Unresolvable")])
    (item,) = extract_outline_from_docling_parse(root)
    assert item.page_no is None
    assert item.y_top is None


def test_outline_deep_chain_does_not_raise_recursion_error():
    """Regression test: a naive recursive walk over the outline tree raises
    RecursionError once the tree is deeper than Python's call-stack limit
    (default 1000). Large real-world documents can legitimately have this
    many nested heading levels. Use a depth well past the default limit to
    make sure this is actually exercised regardless of interpreter
    settings."""
    depth = 5000
    root = _build_chain(depth)

    items = extract_outline_from_docling_parse(root)

    assert len(items) == depth - 1
    assert items[0].title == "level_1"
    assert items[0].level == 0
    assert items[-1].title == f"level_{depth - 1}"
    assert items[-1].level == depth - 2
