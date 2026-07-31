import pytest
from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    GroupLabel,
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
