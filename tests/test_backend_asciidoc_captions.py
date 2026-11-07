# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for the placement of block title captions in the AsciiDoc backend."""

import pytest
from docling_core.types.doc import DocItemLabel

from .test_backend_asciidoc import _convert_adoc


@pytest.mark.parametrize(
    ("block", "label"),
    [
        ("image::x.png[]", DocItemLabel.PICTURE),
        ("|===\n| a | b\n|===", DocItemLabel.TABLE),
        ("----\nx = 1\n----", DocItemLabel.CODE),
    ],
    ids=["picture", "table", "code"],
)
def test_block_title_caption_stays_in_the_section_of_its_block(
    block: str, label: DocItemLabel
) -> None:
    # The caption must be a sibling of its block, not a child of the body
    # collected after the last section.
    src = f"= Title\n\n== One\n\n.Block title\n{block}\n\n== Two\n\nText.\n"
    doc = _convert_adoc(src.encode())

    item = next(item for item, _ in doc.iterate_items() if item.label == label)
    caption = item.captions[0].resolve(doc)
    assert caption.text == "Block title"
    assert caption.parent == item.parent

    section = item.parent.resolve(doc)
    assert section.text == "One"
    assert [ref.cref for ref in section.children] == [caption.self_ref, item.self_ref]
