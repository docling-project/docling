# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Regression tests for issue #4185: DOCX list starting at indent level > 0.

A numbered Word list whose items sit at ``w:ilvl`` 1, 2, 1 (i.e. it never
touches indent level 0) used to drop the item that returns to the *starting*
level: the backend logged
"Parent element of the list item is not a ListGroup. The list item will be ignored."
and discarded the paragraph. Only a list that starts at level 0 was handled.

The fixtures used here live in ``tests/data/docx/sources/`` with the matching
ground truth under ``tests/data/docx/groundtruth/``, following the conventions
of the other Word list fixtures.
"""

from pathlib import Path

from docling_core.types.doc import ListGroup, ListItem

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter

_DOCX_ROOT = Path("./tests/data/docx/sources")


def _convert(name: str):
    docx_path = _DOCX_ROOT / f"{name}.docx"
    assert docx_path.exists()
    return (
        DocumentConverter(allowed_formats=[InputFormat.DOCX])
        .convert(docx_path)
        .document
    )


def _list_items(doc):
    return [item for item, _ in doc.iterate_items() if isinstance(item, ListItem)]


def test_list_returning_to_starting_level_above_zero_keeps_items():
    """A list at levels 1, 2, 1 must not drop the item returning to level 1.

    Regression for #4185: a numbered list that starts at ``w:ilvl`` 1 (never
    touching level 0) used to lose the third item, because the level-1 slot
    between the list base and the level-2 sub-list group was left empty.
    """

    converted = _convert("docx_list_starts_above_level_zero")

    # All three items must survive, in order.
    assert [t.text for t in converted.texts] == ["Item A", "Item B", "Item C"]

    list_items = _list_items(converted)
    assert [item.text for item in list_items] == ["Item A", "Item B", "Item C"]

    # The item that returns to the starting level must rejoin the starting
    # level's ListGroup (the same one Item A lives in) -- not be dropped.
    group_of_a = list_items[0].parent.resolve(converted)
    group_of_c = list_items[2].parent.resolve(converted)
    assert isinstance(group_of_a, ListGroup)
    assert group_of_c.get_ref() == group_of_a.get_ref()

    # The deeper item must nest inside the same list (as a sub-group of the
    # outer ListGroup), not become a sibling top-level list -- which is what
    # the bug produced for Item A/Item B.
    group_of_b = list_items[1].parent.resolve(converted)
    assert isinstance(group_of_b, ListGroup)
    assert group_of_b.parent == group_of_a.get_ref()

    # The markdown must reflect a single nested list, not two separate lists.
    markdown = converted.export_to_markdown()
    assert "Item C" in markdown
    # Item B is rendered indented under Item A; Item C returns to the top level.
    lines = [line for line in markdown.splitlines() if line.strip()]
    assert lines[0].startswith("- ") and "Item A" in lines[0]
    assert lines[1].startswith("    - ") and "Item B" in lines[1]
    assert lines[2].startswith("- ") and "Item C" in lines[2]


def test_list_returning_to_starting_level_zero_still_works():
    """Control case: levels 0, 1, 2, 1 must keep working unchanged."""

    converted = _convert("docx_list_starts_at_level_zero")

    assert [t.text for t in converted.texts] == [
        "Item A",
        "Item B",
        "Item C",
        "Item D",
    ]

    list_items = _list_items(converted)
    # Item D (level 1) rejoins Item B's (level 1) group.
    group_of_b = list_items[1].parent.resolve(converted)
    group_of_d = list_items[3].parent.resolve(converted)
    assert group_of_d.get_ref() == group_of_b.get_ref()
