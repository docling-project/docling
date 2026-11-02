# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import pytest
from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    GroupLabel,
    ListItem,
    ProvenanceItem,
)
from docling_core.types.doc.base import BoundingBox, CoordOrigin

from docling.datamodel.base_models import Cluster, TextElement
from docling.models.stages.reading_order.readingorder_model import (
    ReadingOrderModel,
    ReadingOrderOptions,
)


def _bounding_box() -> BoundingBox:
    return BoundingBox(l=0, t=0, r=10, b=10, coord_origin=CoordOrigin.TOPLEFT)


def _list_item(text: str):
    document = DoclingDocument(
        name="test",
        origin=DocumentOrigin(
            filename="test.pdf", mimetype="application/pdf", binary_hash="0" * 64
        ),
    )
    list_group = document.add_group(label=GroupLabel.LIST, name="list")
    return document.add_list_item(
        text=text,
        prov=ProvenanceItem(page_no=1, charspan=(0, len(text)), bbox=_bounding_box()),
        parent=list_group,
    )


def _element(cluster_id: int, text: str) -> TextElement:
    return TextElement(
        id=cluster_id,
        page_no=1,
        label=DocItemLabel.LIST_ITEM,
        text=text,
        cluster=Cluster(
            id=cluster_id,
            label=DocItemLabel.LIST_ITEM,
            bbox=_bounding_box(),
        ),
    )


@pytest.mark.parametrize(
    ("prefix", "continuation", "expected"),
    [
        ("algo-", "rithms", "algorithms"),
        ("algo\u00ad", "rithms", "algorithms"),
        ("algo-", "Rithms", "algo- Rithms"),
    ],
)
def test_merge_elements_dehyphenates_lowercase_continuations(
    prefix: str, continuation: str, expected: str
) -> None:
    item = _list_item(prefix)
    model = ReadingOrderModel(ReadingOrderOptions())

    model._merge_elements(
        _element(1, prefix),
        _element(2, continuation),
        item,
        page_height=100,
    )

    assert item.text == expected
    assert item.orig == expected


@pytest.mark.parametrize(
    ("prefix", "continuation", "expected_spans"),
    [
        ("algo-", "rithms", [(0, 4), (4, 10)]),
        ("algo\u00ad", "rithms", [(0, 4), (4, 10)]),
        ("algo-", "Rithms", [(0, 5), (6, 12)]),
        ("algo", "rithms", [(0, 4), (5, 11)]),
    ],
)
def test_merged_provenance_covers_the_resulting_text(
    prefix: str, continuation: str, expected_spans: list[tuple[int, int]]
) -> None:
    """Removing a hyphen must update both source and continuation spans."""
    item = _list_item(prefix)
    model = ReadingOrderModel(ReadingOrderOptions())

    model._merge_elements(
        _element(1, prefix), _element(2, continuation), item, page_height=100
    )

    assert [prov.charspan for prov in item.prov] == expected_spans
    assert "".join(item.text[start:end] for start, end in expected_spans).replace(
        " ", ""
    ) == item.text.replace(" ", "")


@pytest.mark.parametrize("marker", ["", "- ", "1. "])
def test_repeated_merges_keep_earlier_spans_valid(marker: str) -> None:
    """A later split-word merge must not overlap an earlier source span."""
    model = ReadingOrderModel(ReadingOrderOptions())
    item, _ = model._handle_text_element(
        _element(1, marker + "algo-"),
        out_doc=DoclingDocument(name="repeated-merge"),
        current_list=None,
        parent=None,
        page_height=100,
    )
    assert isinstance(item, ListItem)
    model._merge_elements(
        _element(1, "algo-"), _element(2, "rithms and pro-"), item, page_height=100
    )
    model._merge_elements(
        _element(1, "algo-"), _element(3, "grams"), item, page_height=100
    )

    assert item.text == "algorithms and programs"
    assert [prov.charspan for prov in item.prov] == [(0, 4), (4, 18), (18, 23)]


@pytest.mark.parametrize("marker", ["", "- ", "1. "])
@pytest.mark.parametrize(
    ("prefix", "continuation", "expected", "expected_spans"),
    [
        ("algo-", "rithms", "algorithms", [(0, 4), (4, 10)]),
        ("algo\u00ad", "rithms", "algorithms", [(0, 4), (4, 10)]),
        ("algo-", "Rithms", "algo- Rithms", [(0, 5), (6, 12)]),
        ("algo", "rithms", "algo rithms", [(0, 4), (5, 11)]),
    ],
)
def test_list_marker_spans_index_text(
    marker: str,
    prefix: str,
    continuation: str,
    expected: str,
    expected_spans: list[tuple[int, int]],
) -> None:
    """List provenance indexes marker-free text, while orig retains the marker."""
    document = DoclingDocument(name="list-marker")
    model = ReadingOrderModel(ReadingOrderOptions())
    element = _element(1, marker + prefix)
    item, list_group = model._handle_text_element(
        element, out_doc=document, current_list=None, parent=None, page_height=100
    )
    assert isinstance(item, ListItem)
    assert list_group is not None
    assert item.parent == list_group.get_ref()
    assert item.text == prefix
    assert item.orig == marker + prefix
    assert item.prov[0].charspan == (0, len(prefix))

    model._merge_elements(element, _element(2, continuation), item, page_height=100)

    assert item.text == expected
    assert item.orig == marker + expected
    assert [prov.charspan for prov in item.prov] == expected_spans
    assert expected in document.export_to_markdown()
    restored = DoclingDocument.model_validate(document.export_to_dict())
    assert restored.texts[0].prov == item.prov
