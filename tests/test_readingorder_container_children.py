from pathlib import PurePath

import pytest
from docling_core.types.doc import (
    BoundingBox,
    CodeItem,
    DocItemLabel,
    GroupLabel,
    Size,
    TableCell,
)
from docling_core.types.doc.document import GroupItem, PictureItem, TableItem

from docling.datamodel.base_models import (
    AssembledUnit,
    Cluster,
    ContainerElement,
    FigureElement,
    InputFormat,
    Page,
    PageElement,
    Table,
    TextElement,
)
from docling.datamodel.document import ConversionResult, InputDocument
from docling.models.stages.reading_order.readingorder_model import (
    ReadingOrderModel,
    ReadingOrderOptions,
)


def _cluster(cid: int, label: DocItemLabel, bbox: tuple[float, ...]) -> Cluster:
    left, top, right, bottom = bbox
    return Cluster(
        id=cid,
        label=label,
        bbox=BoundingBox(l=left, t=top, r=right, b=bottom),
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


def test_container_preserves_typed_table_and_picture_children() -> None:
    table_cluster = _cluster(2, DocItemLabel.TABLE, (10, 10, 150, 100))
    picture_cluster = _cluster(3, DocItemLabel.PICTURE, (200, 200, 300, 300))
    container_cluster = _cluster(1, DocItemLabel.FORM, (0, 0, 400, 400))
    container_cluster.children = [table_cluster, picture_cluster]

    table = Table(
        label=DocItemLabel.TABLE,
        id=2,
        page_no=1,
        cluster=table_cluster,
        otsl_seq=[],
        num_rows=1,
        num_cols=1,
        table_cells=[
            TableCell(
                text="value",
                start_row_offset_idx=0,
                end_row_offset_idx=1,
                start_col_offset_idx=0,
                end_col_offset_idx=1,
            )
        ],
    )
    picture = FigureElement(
        label=DocItemLabel.PICTURE,
        id=3,
        page_no=1,
        cluster=picture_cluster,
    )
    container = ContainerElement(
        label=DocItemLabel.FORM,
        id=1,
        page_no=1,
        cluster=container_cluster,
    )
    conv_res = _conversion_result([container, table, picture])
    model = object.__new__(ReadingOrderModel)
    ro_elements = model._assembled_to_readingorder_elements(conv_res)

    doc = model._readingorder_elements_to_docling_doc(
        conv_res,
        ro_elements,
        el_to_captions_mapping={},
        el_to_footnotes_mapping={},
        el_merges_mapping={},
    )

    form = next(
        item
        for item in doc.groups
        if isinstance(item, GroupItem) and item.label == GroupLabel.FORM_AREA
    )
    children = [child.resolve(doc) for child in form.children]
    assert [type(child) for child in children] == [TableItem, PictureItem]
    assert len(doc.body.children) == 1
    assert len(doc.tables) == 1
    assert doc.tables[0].data.table_cells[0].text == "value"
    assert len(doc.pictures) == 1


@pytest.mark.parametrize("caption_is_child", [False, True])
def test_container_does_not_interrupt_caption_assignment(
    caption_is_child: bool,
) -> None:
    container_bottom = 490 if caption_is_child else 460
    container_cluster = _cluster(0, DocItemLabel.FORM, (0, 330, 160, container_bottom))
    table_cluster = _cluster(1, DocItemLabel.TABLE, (10, 350, 150, 450))
    caption_cluster = _cluster(2, DocItemLabel.CAPTION, (10, 465, 150, 480))
    container_cluster.children = [table_cluster]
    if caption_is_child:
        container_cluster.children.append(caption_cluster)

    table = Table(
        label=DocItemLabel.TABLE,
        id=1,
        page_no=1,
        cluster=table_cluster,
        otsl_seq=[],
        num_rows=1,
        num_cols=1,
        table_cells=[],
    )
    container = ContainerElement(
        label=DocItemLabel.FORM,
        id=0,
        page_no=1,
        cluster=container_cluster,
    )
    caption = TextElement(
        label=DocItemLabel.CAPTION,
        id=2,
        text="Table caption",
        page_no=1,
        cluster=caption_cluster,
    )

    doc = ReadingOrderModel(ReadingOrderOptions())(
        _conversion_result([container, table, caption])
    )

    assert len(doc.tables) == 1
    assert [item.resolve(doc).text for item in doc.tables[0].captions] == [
        "Table caption"
    ]
    assert len([text for text in doc.texts if text.label == DocItemLabel.CAPTION]) == 1


def test_container_children_follow_predicted_reading_order() -> None:
    container_cluster = _cluster(1, DocItemLabel.FORM, (0, 0, 300, 300))
    lower_cluster = _cluster(2, DocItemLabel.PICTURE, (10, 200, 100, 250))
    upper_cluster = _cluster(3, DocItemLabel.PICTURE, (10, 10, 100, 60))
    container_cluster.children = [lower_cluster, upper_cluster]

    container = ContainerElement(
        label=DocItemLabel.FORM,
        id=1,
        page_no=1,
        cluster=container_cluster,
    )
    lower = FigureElement(
        label=DocItemLabel.PICTURE,
        id=2,
        page_no=1,
        cluster=lower_cluster,
    )
    upper = FigureElement(
        label=DocItemLabel.PICTURE,
        id=3,
        page_no=1,
        cluster=upper_cluster,
    )

    doc = ReadingOrderModel(ReadingOrderOptions())(
        _conversion_result([container, lower, upper])
    )

    form = next(group for group in doc.groups if group.label == GroupLabel.FORM_AREA)
    children = [child.resolve(doc) for child in form.children]
    assert [child.prov[0].bbox.t for child in children] == [490, 300]


def test_container_regular_children_keep_cluster_order() -> None:
    container_cluster = _cluster(1, DocItemLabel.FORM, (0, 0, 300, 300))
    lower_cluster = _cluster(2, DocItemLabel.TEXT, (10, 200, 100, 250))
    upper_cluster = _cluster(3, DocItemLabel.TEXT, (10, 10, 100, 60))
    container_cluster.children = [lower_cluster, upper_cluster]

    container = ContainerElement(
        label=DocItemLabel.FORM,
        id=1,
        page_no=1,
        cluster=container_cluster,
    )
    lower = TextElement(
        label=DocItemLabel.TEXT,
        id=2,
        text="lower",
        page_no=1,
        cluster=lower_cluster,
    )
    upper = TextElement(
        label=DocItemLabel.TEXT,
        id=3,
        text="upper",
        page_no=1,
        cluster=upper_cluster,
    )

    doc = ReadingOrderModel(ReadingOrderOptions())(
        _conversion_result([container, lower, upper])
    )

    form = next(group for group in doc.groups if group.label == GroupLabel.FORM_AREA)
    children = [child.resolve(doc) for child in form.children]
    assert [child.text for child in children] == ["lower", "upper"]


def test_container_boundary_prevents_text_merge() -> None:
    external_cluster = _cluster(1, DocItemLabel.TEXT, (10, 10, 100, 50))
    container_cluster = _cluster(2, DocItemLabel.FORM, (140, 0, 260, 60))
    child_cluster = _cluster(3, DocItemLabel.TEXT, (150, 10, 250, 50))
    container_cluster.children = [child_cluster]

    external = TextElement(
        label=DocItemLabel.TEXT,
        id=1,
        text="hello",
        page_no=1,
        cluster=external_cluster,
    )
    container = ContainerElement(
        label=DocItemLabel.FORM,
        id=2,
        page_no=1,
        cluster=container_cluster,
    )
    child = TextElement(
        label=DocItemLabel.TEXT,
        id=3,
        text="world",
        page_no=1,
        cluster=child_cluster,
    )

    doc = ReadingOrderModel(ReadingOrderOptions())(
        _conversion_result([external, container, child])
    )

    assert [text.text for text in doc.texts] == ["hello", "world"]


def test_container_preserves_nested_typed_child_body_order() -> None:
    container_cluster = _cluster(0, DocItemLabel.FORM, (0, 150, 400, 400))
    table_cluster = _cluster(1, DocItemLabel.TABLE, (20, 300, 80, 380))
    outside_cluster = _cluster(2, DocItemLabel.TEXT, (350, 100, 410, 140))
    container_cluster.children = [table_cluster]

    container = ContainerElement(
        label=DocItemLabel.FORM,
        id=0,
        page_no=1,
        cluster=container_cluster,
    )
    table = Table(
        label=DocItemLabel.TABLE,
        id=1,
        page_no=1,
        cluster=table_cluster,
        otsl_seq=[],
        num_rows=1,
        num_cols=1,
        table_cells=[],
    )
    outside = TextElement(
        label=DocItemLabel.TEXT,
        id=2,
        text="outside",
        page_no=1,
        cluster=outside_cluster,
    )

    doc = ReadingOrderModel(ReadingOrderOptions())(
        _conversion_result([container, table, outside])
    )

    body_items = [child.resolve(doc) for child in doc.body.children]
    assert [item.label for item in body_items] == [
        GroupLabel.FORM_AREA,
        DocItemLabel.TEXT,
    ]


def test_container_code_preserves_caption() -> None:
    container_cluster = _cluster(0, DocItemLabel.FORM, (0, 330, 160, 490))
    code_cluster = _cluster(1, DocItemLabel.CODE, (10, 350, 150, 450))
    caption_cluster = _cluster(2, DocItemLabel.CAPTION, (10, 465, 150, 480))
    container_cluster.children = [code_cluster, caption_cluster]

    container = ContainerElement(
        label=DocItemLabel.FORM,
        id=0,
        page_no=1,
        cluster=container_cluster,
    )
    code = TextElement(
        label=DocItemLabel.CODE,
        id=1,
        text="print('x')",
        page_no=1,
        cluster=code_cluster,
    )
    caption = TextElement(
        label=DocItemLabel.CAPTION,
        id=2,
        text="Example",
        page_no=1,
        cluster=caption_cluster,
    )

    doc = ReadingOrderModel(ReadingOrderOptions())(
        _conversion_result([container, code, caption])
    )

    code_item = next(item for item in doc.texts if item.label == DocItemLabel.CODE)
    assert isinstance(code_item, CodeItem)
    assert [item.resolve(doc).text for item in code_item.captions] == ["Example"]


def test_container_between_texts_prevents_merge() -> None:
    left_cluster = _cluster(0, DocItemLabel.TEXT, (0, 350, 100, 400))
    container_cluster = _cluster(1, DocItemLabel.FORM, (120, 340, 200, 410))
    right_cluster = _cluster(2, DocItemLabel.TEXT, (220, 350, 300, 400))

    left = TextElement(
        label=DocItemLabel.TEXT,
        id=0,
        text="foo",
        page_no=1,
        cluster=left_cluster,
    )
    container = ContainerElement(
        label=DocItemLabel.FORM,
        id=1,
        page_no=1,
        cluster=container_cluster,
    )
    right = TextElement(
        label=DocItemLabel.TEXT,
        id=2,
        text="bar",
        page_no=1,
        cluster=right_cluster,
    )

    doc = ReadingOrderModel(ReadingOrderOptions())(
        _conversion_result([left, container, right])
    )

    assert [text.text for text in doc.texts] == ["foo", "bar"]
