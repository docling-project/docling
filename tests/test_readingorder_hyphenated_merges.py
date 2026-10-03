# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from pathlib import PurePath

import pytest
from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    GroupLabel,
    ProvenanceItem,
    Size,
)
from docling_core.types.doc.base import BoundingBox, CoordOrigin

from docling.datamodel.base_models import (
    AssembledUnit,
    Cluster,
    InputFormat,
    Page,
    TextElement,
)
from docling.datamodel.document import ConversionResult, InputDocument
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
    ("prefix", "continuation", "expected", "spans"),
    [
        ("algo-", "rithms", "algorithms", [(0, 4), (4, 10)]),
        ("algo\u00ad", "rithms", "algorithms", [(0, 4), (4, 10)]),
        ("algo-", "Rithms", "algo- Rithms", [(0, 5), (6, 12)]),
        ("two", "words", "two words", [(0, 3), (4, 9)]),
        ("algo-", "  rithms", "algo  rithms", [(0, 4), (4, 12)]),
        ("algo-", "", "algo- ", [(0, 5), (6, 6)]),
        ("algo\u00ad", "", "algo", [(0, 4), (4, 4)]),
        ("", "words", " words", [(0, 0), (1, 6)]),
        ("\u00ad", "word", "word", [(0, 0), (0, 4)]),
    ],
)
def test_merged_provenance_tracks_text(
    prefix: str,
    continuation: str,
    expected: str,
    spans: list[tuple[int, int]],
) -> None:
    item = _list_item(prefix)
    left = _element(1, prefix)
    right = _element(2, continuation)
    right.page_no = 2
    model = ReadingOrderModel(ReadingOrderOptions())

    model._merge_elements(left, right, item, page_height=100)

    assert item.text == item.orig == expected
    assert [prov.charspan for prov in item.prov] == spans
    assert [prov.page_no for prov in item.prov] == [1, 2]
    assert item.prov[0].bbox == _bounding_box()
    assert item.prov[1].bbox == _bounding_box().to_bottom_left_origin(100)
    assert item.text[slice(*item.prov[1].charspan)] == continuation


@pytest.mark.parametrize("hyphen", ["-", "\u00ad"])
def test_repeated_dehyphenation_keeps_earlier_provenance(hyphen: str) -> None:
    item = _list_item(f"re{hyphen}")
    first = _element(1, item.text)
    model = ReadingOrderModel(ReadingOrderOptions())

    for page_no, continuation in enumerate([f"con{hyphen}", "struction"], start=2):
        right = _element(page_no, continuation)
        right.page_no = page_no
        model._merge_elements(first, right, item, page_height=100)

    assert item.text == item.orig == "reconstruction"
    assert [prov.charspan for prov in item.prov] == [(0, 2), (2, 5), (5, 14)]
    assert [item.text[slice(*prov.charspan)] for prov in item.prov] == [
        "re",
        "con",
        "struction",
    ]
    assert [prov.page_no for prov in item.prov] == [1, 2, 3]


@pytest.mark.parametrize("prefix", ["algo-", "algo\u00ad", "algo"])
def test_merge_without_initial_provenance(prefix: str) -> None:
    item = _list_item(prefix)
    item.prov.clear()
    model = ReadingOrderModel(ReadingOrderOptions())

    model._merge_elements(
        _element(1, prefix), _element(2, "rithms"), item, page_height=100
    )

    expected = "algo rithms" if prefix == "algo" else "algorithms"
    assert item.text == expected
    assert [prov.charspan for prov in item.prov] == [(len(expected) - 6, len(expected))]


@pytest.mark.parametrize("same_link", [True, False])
def test_dehyphenation_preserves_hyperlink_merge_policy(same_link: bool) -> None:
    item = _list_item("algo-")
    item.hyperlink = "https://example.com/first"
    right = _element(2, "rithms")
    right.hyperlink = item.hyperlink if same_link else "https://example.com/second"

    ReadingOrderModel(ReadingOrderOptions())._merge_elements(
        _element(1, "algo-"), right, item, page_height=100
    )

    assert item.text == "algorithms"
    assert item.hyperlink == ("https://example.com/first" if same_link else None)
    assert [prov.charspan for prov in item.prov] == [(0, 4), (4, 10)]


@pytest.mark.parametrize(
    ("parts", "expected", "spans"),
    [
        (["re-", "con-", "struction"], "reconstruction", [(0, 2), (2, 5), (5, 14)]),
        (
            ["re\u00ad", "con\u00ad", "struction"],
            "reconstruction",
            [(0, 2), (2, 5), (5, 14)],
        ),
        (["one", "two", "three"], "one two three", [(0, 3), (4, 7), (8, 13)]),
    ],
)
def test_reading_order_merges_pages_with_valid_provenance(
    parts: list[str], expected: str, spans: list[tuple[int, int]]
) -> None:
    elements = []
    for page_no, text in enumerate(parts, start=1):
        element = _element(page_no, text)
        element.page_no = page_no
        element.label = element.cluster.label = DocItemLabel.TEXT
        elements.append(element)
    result = ConversionResult(
        input=InputDocument.model_construct(
            file=PurePath("input.pdf"),
            document_hash="0" * 64,
            valid=True,
            format=InputFormat.PDF,
        ),
        pages=[Page(page_no=i, size=Size(width=100, height=100)) for i in range(1, 4)],
        assembled=AssembledUnit(elements=elements, body=elements),
    )

    document = ReadingOrderModel(ReadingOrderOptions())(result)

    assert len(document.texts) == 1
    item = document.texts[0]
    assert item.text == item.orig == expected
    assert [prov.charspan for prov in item.prov] == spans
    assert [prov.page_no for prov in item.prov] == [1, 2, 3]
    assert all(
        prov.bbox == _bounding_box().to_bottom_left_origin(100) for prov in item.prov
    )
    restored = DoclingDocument.model_validate_json(document.model_dump_json())
    assert [prov.charspan for prov in restored.texts[0].prov] == spans
