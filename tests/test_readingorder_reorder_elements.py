# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from pathlib import PurePath

from docling_core.types.doc import BoundingBox, DocItemLabel, Size
from docling_core.types.doc.document import TextItem

from docling.datamodel.base_models import (
    AssembledUnit,
    Cluster,
    InputFormat,
    Page,
    PageElement,
    TextElement,
)
from docling.datamodel.document import ConversionResult, InputDocument
from docling.models.stages.reading_order.readingorder_model import (
    ReadingOrderModel,
    ReadingOrderOptions,
)


def _conversion_result(elements: list[PageElement]) -> ConversionResult:
    input_doc = InputDocument.model_construct(
        file=PurePath("input.pdf"),
        document_hash="0" * 64,
        valid=True,
        format=InputFormat.PDF,
    )
    return ConversionResult(
        input=input_doc,
        pages=[Page(page_no=1, size=Size(width=500, height=500))],
        assembled=AssembledUnit(elements=elements, body=elements),
    )


def _text(cid: int, text: str, top: float, bottom: float) -> TextElement:
    return TextElement(
        label=DocItemLabel.TEXT,
        id=cid,
        text=text,
        page_no=1,
        cluster=Cluster(
            id=cid,
            label=DocItemLabel.TEXT,
            bbox=BoundingBox(l=10, t=top, r=150, b=bottom),
        ),
    )


def _assembled_out_of_reading_order() -> list[PageElement]:
    """Two paragraphs assembled bottom-first, so reading order must swap them."""
    lower = _text(1, "lower", top=300, bottom=340)
    upper = _text(2, "upper", top=100, bottom=140)
    return [lower, upper]


def _texts(doc) -> list[str]:
    return [item.text for item in doc.texts if isinstance(item, TextItem)]


def test_reorder_elements_enabled_sorts_into_reading_order() -> None:
    conv_res = _conversion_result(_assembled_out_of_reading_order())

    doc = ReadingOrderModel(ReadingOrderOptions())(conv_res)

    assert _texts(doc) == ["upper", "lower"]


def test_reorder_elements_disabled_keeps_assembled_order() -> None:
    conv_res = _conversion_result(_assembled_out_of_reading_order())

    doc = ReadingOrderModel(ReadingOrderOptions(reorder_elements=False))(conv_res)

    assert _texts(doc) == ["lower", "upper"]
