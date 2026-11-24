# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import json
from pathlib import Path, PurePath
from unittest.mock import MagicMock

import pytest
from docling_core.transforms.serializer.doclang import DocLangDocSerializer
from docling_core.types.doc import (
    BoundingBox,
    DocItemLabel,
    DoclingDocument,
    ListItem,
    NodeItem,
    RichTableCell,
    Size,
    TableCell,
)
from docling_core.types.doc.items.form import (
    FieldItem,
    FieldRegionItem,
    FieldValueItem,
)
from docling_core.types.doc.page import (
    BoundingRectangle,
    PdfHyperlink,
    PdfWidget,
    TextCell,
)

from docling.datamodel.base_models import (
    AssembledUnit,
    Cluster,
    FieldItemPrediction,
    FieldRegionElement,
    InputFormat,
    LayoutPrediction,
    Page,
    Table,
    TableStructurePrediction,
)
from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.models.postprocessing.reading_order_rb import ReadingOrderPredictor
from docling.models.stages.form_field import form_field_model
from docling.models.stages.form_field.form_field_model import PdfFormFieldModel
from docling.models.stages.form_field.keying import PUSHBUTTON_FLAG
from docling.models.stages.page_assemble.page_assemble_model import (
    PageAssembleModel,
    PageAssembleOptions,
)
from docling.models.stages.reading_order.readingorder_model import (
    ReadingOrderModel,
    ReadingOrderOptions,
)

FORM_PDF = Path("tests/data/pdf/sources/acroform_sample.pdf")
EXPECTED_NAMES = [
    "applicant_name",
    "agree_terms",
    "newsletter",
    "mail_optin",
]


def _convert(*, extract_form_fields: bool) -> ConversionResult:
    options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=False,
        extract_form_fields=extract_form_fields,
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options),
        }
    ).convert(FORM_PDF)


def _regions(page: Page) -> list[tuple[int | None, list[FieldItemPrediction]]]:
    """The page's field items by route: host paragraph or FORM id, then page-wide."""
    fields = page.predictions.form_fields
    if fields is None:
        return []
    return [
        *((host_id, [item]) for host_id, item in fields.inline_items.items()),
        *fields.form_items.items(),
        *([(None, fields.loose_items)] if fields.loose_items else []),
    ]


def _conversion_result(page: Page) -> ConversionResult:
    input_doc = InputDocument.model_construct(
        file=PurePath("input.pdf"),
        document_hash="0" * 64,
        valid=True,
        format=InputFormat.PDF,
    )
    return ConversionResult(input=input_doc, pages=[page])


def _widget(
    index: int,
    bbox: BoundingBox,
    text: str,
    *,
    field_type: str = "/Tx",
    appearance_state: str | None = None,
    field_flags: int = 0,
) -> PdfWidget:
    return PdfWidget(
        index=index,
        rect=BoundingRectangle.from_bounding_box(
            bbox.to_bottom_left_origin(page_height=100)
        ),
        widget_text=text,
        widget_field_type=field_type,
        widget_appearance_state=appearance_state,
        widget_field_flags=field_flags,
    )


def test_field_mapping_assembly_and_materialization_preserve_regions_and_order() -> (
    None
):
    form_1 = Cluster(
        id=1,
        label=DocItemLabel.FORM,
        bbox=BoundingBox(l=0, t=0, r=60, b=60),
    )
    form_2 = Cluster(
        id=2,
        label=DocItemLabel.FORM,
        bbox=BoundingBox(l=40, t=0, r=100, b=60),
    )
    form_3 = Cluster(
        id=3,
        label=DocItemLabel.FORM,
        bbox=BoundingBox(l=45, t=30, r=55, b=45),
    )
    widgets = [
        _widget(0, BoundingBox(l=10, t=10, r=20, b=20), "first"),
        _widget(1, BoundingBox(l=80, t=10, r=90, b=20), "second"),
        _widget(2, BoundingBox(l=20, t=30, r=30, b=40), "third"),
        _widget(3, BoundingBox(l=45, t=10, r=55, b=20), "tie"),
        _widget(
            4,
            BoundingBox(l=47, t=32, r=53, b=40),
            "/Off",
            field_type="/Btn",
            appearance_state="/On",
        ),
        _widget(
            5,
            BoundingBox(l=10, t=80, r=20, b=90),
            "/Yes",
            field_type="/Btn",
            appearance_state="/Off",
        ),
        # Zero-size widgets are artifacts (Well-Tagged PDF 1.0, 8.9.2.4.13).
        _widget(6, BoundingBox(l=30, t=30, r=30, b=30), "ghost"),
        # Push buttons carry no value.
        _widget(
            7,
            BoundingBox(l=60, t=80, r=90, b=90),
            "Submit",
            field_type="/Btn",
            field_flags=PUSHBUTTON_FLAG,
        ),
    ]
    # A FORM keeps its child blocks, as layout postprocessing leaves them.
    visible_child = Cluster(
        id=4,
        label=DocItemLabel.TEXT,
        bbox=BoundingBox(l=5, t=5, r=35, b=9),
    )
    form_1.children = [visible_child]
    page = Page(page_no=1, size=Size(width=100, height=100))
    page.parsed_page = MagicMock(widgets=widgets)
    page.predictions.layout = LayoutPrediction(
        clusters=[form_1, form_2, form_3, visible_child]
    )
    conv_res = _conversion_result(page)

    assert list(PdfFormFieldModel(enabled=True)(conv_res, [page])) == [page]
    assert [source for source, _ in _regions(page)] == [
        1,
        2,
        3,
        None,
    ]

    # Keyless items: one value each, one item per widget -- flat shape preserved.
    def _region_values(items):
        return [value for item in items for value in item.values]

    regions = _regions(page)
    assert [
        [value.text for value in _region_values(items)] for _, items in regions
    ] == [["first", "third", "tie"], ["second"], [""], [""]]
    assert [
        [value.checkbox for value in _region_values(items)] for _, items in regions
    ] == [
        [None, None, None],
        [None],
        [DocItemLabel.CHECKBOX_SELECTED],
        [DocItemLabel.CHECKBOX_UNSELECTED],
    ]
    assert all(item.key_text == "" for _, items in regions for item in items)
    assert regions[2][1][0].values[0].orig == "/On"
    assert regions[3][1][0].values[0].orig == "/Off"
    # The zero-size and push-button widgets are dropped: no extra region and
    # neither their text nor orig reaches any value.
    assert not any(
        value.orig in {"ghost", "Submit"}
        for _, items in regions
        for value in _region_values(items)
    )

    layout_before_assembly = page.predictions.layout.model_copy(deep=True)
    page._backend = MagicMock()
    page._backend.is_valid.return_value = True

    assert list(PageAssembleModel(PageAssembleOptions())(conv_res, [page])) == [page]
    assert page.predictions.layout == layout_before_assembly
    assert page.assembled is not None
    region_elements = [
        element
        for element in page.assembled.elements
        if isinstance(element, FieldRegionElement)
    ]
    # Each FORM block with fields becomes a region element; the page-wide
    # field is placed by the reading order.
    assert [element.id for element in region_elements] == [1, 2, 3]
    assert region_elements[0].cluster.children == [visible_child]
    assert len({element.id for element in page.assembled.elements}) == len(
        page.assembled.elements
    )

    conv_res.assembled = AssembledUnit(
        elements=page.assembled.elements,
        body=page.assembled.body,
        headers=page.assembled.headers,
    )
    doc = ReadingOrderModel(ReadingOrderOptions())(conv_res)
    # One field region per FORM, plus one for the page-wide field, which the
    # reading order places between the blocks by geometry.
    assert len(doc.field_regions) == 4
    assert len(doc.field_items) == 6
    assert not any(item.label == DocItemLabel.FIELD_KEY for item in doc.texts)

    def _value_repr(value: FieldValueItem) -> str:
        # Text fields carry their text; a checkbox value is empty and nests a
        # CHECKBOX_* child that carries the selection state.
        checkbox_children = [
            child.resolve(doc)
            for child in value.children
            if child.resolve(doc).label
            in {DocItemLabel.CHECKBOX_SELECTED, DocItemLabel.CHECKBOX_UNSELECTED}
        ]
        if checkbox_children:
            assert value.text == ""
            return checkbox_children[0].label.value
        return value.text

    values_by_region = []
    for region in doc.field_regions:
        field_items = [
            child.resolve(doc)
            for child in region.children
            if isinstance(child.resolve(doc), FieldItem)
        ]
        values_by_region.append(
            [_value_repr(field.children[0].resolve(doc)) for field in field_items]
        )
    assert values_by_region == [
        [DocItemLabel.CHECKBOX_SELECTED.value],
        ["first", "third", "tie"],
        [DocItemLabel.CHECKBOX_UNSELECTED.value],
        ["second"],
    ]
    doc.validate_document()


def _text_cluster(cluster_id: int, bbox: BoundingBox, text: str) -> Cluster:
    cell = TextCell(
        index=0,
        rect=BoundingRectangle.from_bounding_box(bbox),
        text=text,
        orig=text,
        from_ocr=False,
    )
    return Cluster(id=cluster_id, label=DocItemLabel.TEXT, bbox=bbox, cells=[cell])


def test_suppresses_rendered_duplicate_of_filled_widget() -> None:
    # A filled widget paints its /V into the raster; the layout model re-detects
    # it as a plain text cluster inside the widget rect -> suppress it.
    widget_bbox = BoundingBox(l=10, t=10, r=90, b=20)
    form = Cluster(
        id=1, label=DocItemLabel.FORM, bbox=BoundingBox(l=0, t=0, r=100, b=60)
    )
    inside = _text_cluster(2, BoundingBox(l=12, t=11, r=30, b=19), "111.00")
    # same string but well outside any widget rect -> ordinary printed text, keep
    outside = _text_cluster(3, BoundingBox(l=10, t=80, r=30, b=90), "111.00")
    form.children = [inside]

    widgets = [_widget(0, widget_bbox, "111.00")]
    page = Page(page_no=1, size=Size(width=100, height=100))
    page.parsed_page = MagicMock(widgets=widgets)
    page.predictions.layout = LayoutPrediction(clusters=[form, inside, outside])
    conv_res = _conversion_result(page)

    list(PdfFormFieldModel(enabled=True)(conv_res, [page]))

    remaining = page.predictions.layout.clusters
    assert inside not in remaining  # contained duplicate dropped
    assert outside in remaining  # coincidental match kept
    assert form.children == []  # child ref pruned so assembly stays consistent
    # A printed number is never a key: the value stays keyless.
    assert _regions(page)[0][1][0].key_text == ""


def _checkbox_cluster(cluster_id: int, bbox: BoundingBox) -> Cluster:
    # The layout model's checkbox clusters carry the detected mark, no text.
    return Cluster(id=cluster_id, label=DocItemLabel.CHECKBOX_SELECTED, bbox=bbox)


def test_checkbox_widget_keys_its_option_key_and_lets_as_override_classifier() -> None:
    # A /Btn widget with its printed option key to its right. The layout
    # model also detected the box as CHECKBOX_SELECTED (a mark glyph), but /AS
    # ("/Off") -- not the classifier -- decides the state. The key becomes
    # the key and leaves the body; the value keeps the widget's own rect.
    widget_bbox = BoundingBox(l=10, t=10, r=18, b=18)
    key_bbox = BoundingBox(l=22, t=10, r=80, b=18)
    key = _text_cluster(2, key_bbox, "Married filing jointly")
    glyph = _checkbox_cluster(3, BoundingBox(l=9, t=9, r=19, b=19))
    widget = _widget(0, widget_bbox, "/Off", field_type="/Btn", appearance_state="/Off")
    page = Page(page_no=1, size=Size(width=100, height=100))
    page.parsed_page = MagicMock(widgets=[widget])
    page.predictions.layout = LayoutPrediction(clusters=[key, glyph])
    conv_res = _conversion_result(page)

    list(PdfFormFieldModel(enabled=True)(conv_res, [page]))

    ((_, (item,)),) = _regions(page)
    (value,) = item.values
    assert item.key_text == "Married filing jointly"
    assert item.key_bbox == key_bbox
    assert value.text == ""
    assert (
        value.checkbox == DocItemLabel.CHECKBOX_UNSELECTED
    )  # from /AS, overriding the classifier
    assert value.bbox == widget_bbox
    assert key not in page.predictions.layout.clusters  # promoted to a key
    assert glyph in page.predictions.layout.clusters  # the mark is no key


def test_pipeline_materializes_format_neutral_fields() -> None:
    result = _convert(extract_form_fields=True)

    assert result.pages[0].assembled is not None
    # The layout detector merges the three checkbox lines and the note below
    # them into one text cluster, but each checkbox has its own key printed
    # beside it: they are options,
    # not blanks of one sentence, so each keeps its own key. The text field
    # binds its detached "Full name:" key. All four land in the page-wide
    # region; the keys leave the merged paragraph, and its last line, which
    # keys nothing, stays in the body.
    assert [source for source, _ in _regions(result.pages[0])] == [None]
    assert len(result.document.field_regions) == 1
    items = result.document.field_items
    keys = [
        next(
            (
                child.resolve(result.document).text
                for child in item.children
                if child.resolve(result.document).label == DocItemLabel.FIELD_KEY
            ),
            "",
        )
        for item in items
    ]
    assert keys == [
        "Full name:",
        "I agree to the terms",
        "Subscribe to newsletter",
        "Monthly mail digest",
    ]

    values = [
        item for item in result.document.texts if isinstance(item, FieldValueItem)
    ]
    # Checkbox values carry no text; their state rides on a nested CHECKBOX_* child.
    assert [value.text for value in values] == ["Ada Lovelace", "", "", ""]
    checkbox_labels = [
        [
            child.resolve(result.document).label
            for child in value.children
            if child.resolve(result.document).label
            in {DocItemLabel.CHECKBOX_SELECTED, DocItemLabel.CHECKBOX_UNSELECTED}
        ]
        for value in values
    ]
    assert checkbox_labels == [
        [],
        [DocItemLabel.CHECKBOX_SELECTED],
        [DocItemLabel.CHECKBOX_UNSELECTED],
        [DocItemLabel.CHECKBOX_SELECTED],
    ]
    assert [value.orig for value in values] == [
        "Ada Lovelace",
        "/Yes",
        "/Off",
        "/On",
    ]
    assert all(value.kind == "fillable" for value in values)
    assert not any(
        name in {item.text for item in result.document.texts} for name in EXPECTED_NAMES
    )
    body = [
        item.text
        for item in result.document.texts
        if item.label in {DocItemLabel.TEXT, DocItemLabel.PARAGRAPH}
    ]
    assert body == ["See example.org for details"]
    serialized = json.dumps(result.document.export_to_dict())
    assert "widget_field_name" not in serialized
    assert "widget_appearance_state" not in serialized

    disabled = _convert(extract_form_fields=False)
    assert _regions(disabled.pages[0]) == []
    assert disabled.document.field_regions == []
    assert disabled.document.field_items == []
    # The hosting paragraph is not dropped -- it is reinterpreted as a field_item
    # at materialization. The only cluster extraction removes is the "Full name:"
    # key promoted out of the body to become the text field's bound key.
    enabled_clusters = result.pages[0].predictions.layout.clusters
    disabled_clusters = disabled.pages[0].predictions.layout.clusters
    enabled_ids = {cluster.id for cluster in enabled_clusters}
    dropped = [
        PdfFormFieldModel._cluster_text(cluster)
        for cluster in disabled_clusters
        if cluster.id not in enabled_ids
    ]
    assert dropped == ["Fullname:"]


def test_widgets_inlined_in_paragraph_group_into_one_keyed_item() -> None:
    # IRS Sch.1 line 7 shape: a sentence inlines a checkbox and fillable
    # amounts. The layout detects the whole line as one text cluster enclosing
    # the widgets; a FORM cluster also wraps the page. The paragraph keys all
    # of them as one clause, hosts the item in place (its text is the key) and
    # stays in the layout -- not three keyless FORM fields.
    paragraph = _text_cluster(
        1,
        BoundingBox(l=10, t=10, r=90, b=20),
        "check here and enter amount repaid and interest:",
    )
    page_form = Cluster(
        id=2, label=DocItemLabel.FORM, bbox=BoundingBox(l=0, t=0, r=100, b=100)
    )
    widgets = [
        _widget(
            0,
            BoundingBox(l=30, t=12, r=34, b=18),
            "",
            field_type="/Btn",
            appearance_state="/Off",
        ),
        _widget(1, BoundingBox(l=55, t=12, r=68, b=18), "1221.00"),
        _widget(2, BoundingBox(l=74, t=12, r=87, b=18), "12.50"),
    ]
    page = Page(page_no=1, size=Size(width=100, height=100))
    page.parsed_page = MagicMock(widgets=widgets)
    page.predictions.layout = LayoutPrediction(clusters=[page_form, paragraph])
    conv_res = _conversion_result(page)

    list(PdfFormFieldModel(enabled=True)(conv_res, [page]))

    keyed = [
        (source, item)
        for source, items in _regions(page)
        for item in items
        if item.key_text
    ]
    assert len(keyed) == 1
    source, item = keyed[0]
    assert source == paragraph.id
    assert item.key_text == "check here and enter amount repaid and interest:"
    assert item.key_bbox == paragraph.bbox
    assert [(v.text, v.checkbox) for v in item.values] == [
        ("", DocItemLabel.CHECKBOX_UNSELECTED),
        ("1221.00", None),
        ("12.50", None),
    ]
    # The paragraph stays in the layout -- it is reinterpreted as a field_item in
    # place at materialization, keeping its position in its list/container.
    assert paragraph in page.predictions.layout.clusters


def _table_page(
    *, with_structure: bool, amount_cell: str | None = None
) -> tuple[Page, Cluster]:
    """A two-by-two table whose amount cell holds a widget.

    TableFormer cell boxes cover their text, so the widget sits in the row and
    column bands of the table without being inside any detected cell. Like
    TableFormer, the table lists no cell for the empty amount position unless
    ``amount_cell`` gives that cell a text.
    """

    def cell(text, bbox, row, col, *, header=False):
        return TableCell(
            text=text,
            bbox=bbox,
            start_row_offset_idx=row,
            end_row_offset_idx=row + 1,
            start_col_offset_idx=col,
            end_col_offset_idx=col + 1,
            column_header=header,
        )

    cells = [
        cell("Item", BoundingBox(l=2, t=2, r=20, b=8), 0, 0, header=True),
        cell("Amount", BoundingBox(l=52, t=2, r=70, b=8), 0, 1, header=True),
        cell("Total income", BoundingBox(l=2, t=22, r=40, b=28), 1, 0),
    ]
    if amount_cell is not None:
        cells.append(cell(amount_cell, BoundingBox(l=52, t=22, r=60, b=28), 1, 1))
    table = Cluster(
        id=1, label=DocItemLabel.TABLE, bbox=BoundingBox(l=0, t=0, r=100, b=40)
    )
    page = Page(page_no=1, size=Size(width=100, height=100))
    page.parsed_page = MagicMock(
        widgets=[_widget(0, BoundingBox(l=62, t=20, r=82, b=30), "1,000")]
    )
    page.predictions.layout = LayoutPrediction(clusters=[table])
    if with_structure:
        page.predictions.tablestructure = TableStructurePrediction(
            table_map={
                table.id: Table(
                    label=table.label,
                    id=table.id,
                    page_no=page.page_no,
                    cluster=table,
                    otsl_seq=[],
                    num_rows=2,
                    num_cols=2,
                    table_cells=cells,
                )
            }
        )
    return page, table


def _materialize(page: Page) -> DoclingDocument:
    """Run the form stage, page assembly and reading order on one page."""
    conv_res = _conversion_result(page)
    list(PdfFormFieldModel(enabled=True)(conv_res, [page]))
    page._backend = MagicMock()
    page._backend.is_valid.return_value = True
    list(PageAssembleModel(PageAssembleOptions())(conv_res, [page]))
    assert page.assembled is not None
    conv_res.assembled = AssembledUnit(
        elements=page.assembled.elements,
        body=page.assembled.body,
        headers=page.assembled.headers,
    )
    return ReadingOrderModel(ReadingOrderOptions())(conv_res)


def test_value_grid_materializes_column_headers_as_field_hints() -> None:
    page = Page(page_no=1, size=Size(width=300, height=100))
    boxes = [
        BoundingBox(l=x, t=y, r=x + 60, b=y + 12) for y in (30, 65) for x in (100, 200)
    ]
    page.parsed_page = MagicMock(
        widgets=[_widget(i, bbox, str(i + 1)) for i, bbox in enumerate(boxes)]
    )
    page.predictions.layout = LayoutPrediction(
        clusters=[
            _text_cluster(1, BoundingBox(l=10, t=31, r=80, b=41), "Revenue"),
            _text_cluster(2, BoundingBox(l=10, t=66, r=80, b=76), "Expenses"),
            _text_cluster(3, BoundingBox(l=100, t=5, r=160, b=15), "This year"),
            _text_cluster(4, BoundingBox(l=200, t=5, r=260, b=15), "Last year"),
        ]
    )

    doc = _materialize(page)

    assert [
        [(child.resolve(doc).label, child.resolve(doc).text) for child in item.children]
        for item in doc.field_items
    ] == [
        [
            (DocItemLabel.FIELD_KEY, key),
            (DocItemLabel.FIELD_VALUE, str(i + 1)),
            (DocItemLabel.FIELD_HINT, hint),
        ]
        for i, (key, hint) in enumerate(
            [
                ("Revenue", "This year"),
                ("Revenue", "Last year"),
                ("Expenses", "This year"),
                ("Expenses", "Last year"),
            ]
        )
    ]


def _cell_region(doc: DoclingDocument, row: int, col: int) -> NodeItem:
    """The item a rich cell of the document's only table points to."""
    (table,) = doc.tables
    (cell,) = [
        c
        for c in table.data.table_cells
        if c.start_row_offset_idx == row and c.start_col_offset_idx == col
    ]
    assert isinstance(cell, RichTableCell)
    return cell.ref.resolve(doc)


def test_widget_in_table_goes_into_its_cell_with_value_only(caplog) -> None:
    # The row header names the value in another column and the header names
    # the column: the table already says both, so the empty cell TableFormer
    # did not list becomes a rich cell holding only the value.
    page, table = _table_page(with_structure=True)

    with caplog.at_level("WARNING"):
        doc = _materialize(page)
    assert "Form field" not in caplog.text  # the keying ran, not the fallback

    assert _regions(page) == []
    assert page.predictions.tablestructure is not None
    (cell_fields,) = page.predictions.tablestructure.table_map[table.id].field_cells
    assert [(i.key_text, i.hint_text) for i in cell_fields.items] == [("", "")]
    # The table keeps its cells: nothing is promoted out of the layout.
    assert [cluster.id for cluster in page.predictions.layout.clusters] == [1]

    region = _cell_region(doc, 1, 1)
    assert region.label == DocItemLabel.FIELD_REGION
    (field_item,) = doc.field_items
    assert field_item.parent is not None
    assert field_item.parent.cref == region.self_ref
    children = [child.resolve(doc) for child in field_item.children]
    assert [(child.label, child.text) for child in children] == [
        (DocItemLabel.FIELD_VALUE, "1,000")
    ]
    doc.validate_document()
    doclang = doc.export_to_doclang()
    assert doclang.index("<field_region>") < doclang.index("</table>")


@pytest.mark.parametrize(
    ("amount_cell", "key", "kept_text"),
    [
        # Text in the widget's own cell is its key, not repeated as text.
        ("Paid", "Paid", []),
        # A code without letters never keys a value, but stays in its cell.
        ("250", "", ["250"]),
    ],
)
def test_text_in_the_widgets_own_cell_stays_in_the_rich_cell(
    amount_cell: str, key: str, kept_text: list[str]
) -> None:
    page, _ = _table_page(with_structure=True, amount_cell=amount_cell)

    doc = _materialize(page)

    region = _cell_region(doc, 1, 1)
    children = [child.resolve(doc) for child in region.children]
    assert [c.text for c in children if c.label == DocItemLabel.TEXT] == kept_text
    (field_item,) = [c for c in children if c.label == DocItemLabel.FIELD_ITEM]
    keys = [
        child.resolve(doc).text
        for child in field_item.children
        if child.resolve(doc).label == DocItemLabel.FIELD_KEY
    ]
    assert keys == ([key] if key else [])
    doc.validate_document()


def test_widget_between_columns_of_an_incomplete_grid_stays_out_of_the_table() -> None:
    # A three-column grid whose middle column has no text has no band there:
    # a widget in that column, overlapping neither neighbour's band, has no
    # reliable cell and stays after the table, without a key: the row header
    # sits in another cell and never keys it.
    def cell(text, bbox, row, col):
        return TableCell(
            text=text,
            bbox=bbox,
            start_row_offset_idx=row,
            end_row_offset_idx=row + 1,
            start_col_offset_idx=col,
            end_col_offset_idx=col + 1,
        )

    table = Cluster(
        id=1, label=DocItemLabel.TABLE, bbox=BoundingBox(l=0, t=0, r=100, b=40)
    )
    page = Page(page_no=1, size=Size(width=100, height=100))
    page.parsed_page = MagicMock(
        widgets=[_widget(0, BoundingBox(l=44, t=20, r=56, b=30), "1,000")]
    )
    page.predictions.layout = LayoutPrediction(clusters=[table])
    page.predictions.tablestructure = TableStructurePrediction(
        table_map={
            table.id: Table(
                label=table.label,
                id=table.id,
                page_no=page.page_no,
                cluster=table,
                otsl_seq=[],
                num_rows=2,
                num_cols=3,
                table_cells=[
                    cell("Item", BoundingBox(l=2, t=2, r=20, b=8), 0, 0),
                    cell("Total", BoundingBox(l=62, t=2, r=80, b=8), 0, 2),
                    cell("Total income", BoundingBox(l=2, t=22, r=40, b=28), 1, 0),
                ],
            )
        }
    )

    list(PdfFormFieldModel(enabled=True)(_conversion_result(page), [page]))

    assert page.predictions.tablestructure is not None
    assert page.predictions.tablestructure.table_map[table.id].field_cells == []
    ((_, (item,)),) = _regions(page)
    assert item.key_text == ""
    assert [value.text for value in item.values] == ["1,000"]


def test_widget_in_table_without_structure_stays_keyless() -> None:
    # Table keys come from TableFormer cells: without table structure the
    # widget keeps its value and no free-form key crosses into the table.
    page, _ = _table_page(with_structure=False)

    list(PdfFormFieldModel(enabled=True)(_conversion_result(page), [page]))

    ((_, (item,)),) = _regions(page)
    assert item.key_text == ""
    assert item.hint_text == ""
    assert [value.text for value in item.values] == ["1,000"]


def test_keying_failure_keeps_values_without_keys(caplog, monkeypatch) -> None:
    # The page must not fail when the keying raises: its values come through
    # unkeyed and the key stays in the body.
    def failing(*args, **kwargs):
        raise RuntimeError("keying failed")

    monkeypatch.setattr(form_field_model, "assign", failing)
    key = _text_cluster(2, BoundingBox(l=22, t=10, r=80, b=18), "Full name")
    form = Cluster(
        id=1, label=DocItemLabel.FORM, bbox=BoundingBox(l=0, t=0, r=100, b=60)
    )
    widgets = [
        _widget(0, BoundingBox(l=10, t=10, r=18, b=18), "Ada"),
        _widget(1, BoundingBox(l=10, t=30, r=18, b=38), "Lovelace"),
    ]
    page = Page(page_no=1, size=Size(width=100, height=100))
    page.parsed_page = MagicMock(widgets=widgets)
    page.predictions.layout = LayoutPrediction(clusters=[form, key])

    with caplog.at_level("WARNING"):
        list(PdfFormFieldModel(enabled=True)(_conversion_result(page), [page]))

    assert "Form field keying failed on page 1" in caplog.text
    ((source, items),) = _regions(page)
    assert source == form.id
    assert [(item.key_text, item.values[0].text) for item in items] == [
        ("", "Ada"),
        ("", "Lovelace"),
    ]
    assert key in page.predictions.layout.clusters


def _options_page() -> tuple[Page, Cluster]:
    """Three checkboxes with their option keys, then a note, in one cluster.

    acroform_sample.pdf at a quarter of its size: the layout model merges the
    four text lines into one text cluster.
    """
    lines = [
        ("I agree to the terms", 35, 49.75),
        ("Subscribe to newsletter", 42.5, 55),
        ("Monthly mail digest", 50, 49.5),
    ]
    note = ("See example.org for details", BoundingBox(l=18, t=55, r=54.5, b=57.75))
    boxes = [
        (text, BoundingBox(l=23.75, t=top, r=right, b=top + 2.75))
        for text, top, right in lines
    ]
    cells = [
        TextCell(
            index=i,
            rect=BoundingRectangle.from_bounding_box(box),
            text=text,
            orig=text,
            from_ocr=False,
        )
        for i, (text, box) in enumerate([*boxes, note])
    ]
    paragraph = Cluster(
        id=1,
        label=DocItemLabel.TEXT,
        bbox=BoundingBox(l=18, t=35, r=55, b=57.75),
        cells=cells,
    )
    widgets = [
        _widget(
            i,
            BoundingBox(l=18, t=top, r=21.5, b=top + 3.5),
            "",
            field_type="/Btn",
            appearance_state="/Off",
        )
        for i, (_, top, _) in enumerate(lines)
    ]
    page = Page(page_no=1, size=Size(width=100, height=100))
    page.parsed_page = MagicMock(widgets=widgets)
    page.predictions.layout = LayoutPrediction(clusters=[paragraph])
    return page, paragraph


def test_option_keys_leave_a_partly_used_paragraph() -> None:
    # Each checkbox uses its own option key, but the note is no key, so
    # the paragraph is only partly used: the keys leave it, and the note
    # stays in the body with a box fitted to it, instead of the whole paragraph
    # repeating the keys after the fields.
    page, paragraph = _options_page()

    list(PdfFormFieldModel(enabled=True)(_conversion_result(page), [page]))

    assert [item.key_text for _, items in _regions(page) for item in items] == [
        "I agree to the terms",
        "Subscribe to newsletter",
        "Monthly mail digest",
    ]
    assert page.predictions.layout.clusters == [paragraph]
    assert [cell.text for cell in paragraph.cells] == ["See example.org for details"]
    assert paragraph.bbox == BoundingBox(l=18, t=55, r=54.5, b=57.75)


def test_filled_values_do_not_remain_as_a_paragraph_after_their_keys() -> None:
    # A filled form: the page paints each widget's value, and the layout model
    # merges two key-and-value lines into one text cluster. Both keys
    # become keys and the painted values are the fields' own values, so
    # nothing of the cluster is left for the body.
    def cell(index: int, text: str, bbox: BoundingBox) -> TextCell:
        return TextCell(
            index=index,
            rect=BoundingRectangle.from_bounding_box(bbox),
            text=text,
            orig=text,
            from_ocr=False,
        )

    paragraph = Cluster(
        id=1,
        label=DocItemLabel.TEXT,
        bbox=BoundingBox(l=10, t=10, r=50, b=20),
        cells=[
            cell(0, "Full name:", BoundingBox(l=10, t=10, r=28, b=14)),
            cell(1, "John Smith", BoundingBox(l=31, t=10, r=50, b=14)),
            cell(2, "City:", BoundingBox(l=10, t=16, r=20, b=20)),
            cell(3, "Paris", BoundingBox(l=31, t=16, r=40, b=20)),
        ],
    )
    widgets = [
        _widget(0, BoundingBox(l=30, t=9.5, r=70, b=14.5), "John Smith"),
        _widget(1, BoundingBox(l=30, t=15.5, r=70, b=20.5), "Paris"),
    ]
    page = Page(page_no=1, size=Size(width=100, height=100))
    page.parsed_page = MagicMock(widgets=widgets)
    page.predictions.layout = LayoutPrediction(clusters=[paragraph])

    list(PdfFormFieldModel(enabled=True)(_conversion_result(page), [page]))

    assert [
        (item.key_text, [value.text for value in item.values])
        for _, items in _regions(page)
        for item in items
    ] == [("Full name:", ["John Smith"]), ("City:", ["Paris"])]
    assert page.predictions.layout.clusters == []


def test_failed_commit_restores_a_trimmed_paragraph(caplog, monkeypatch) -> None:
    # Writing the plan fails after the keys left the paragraph: the page
    # is left unchanged, so the paragraph gets its cells and box back.
    page, paragraph = _options_page()
    cells, bbox = list(paragraph.cells), paragraph.bbox

    def failing(*args, **kwargs):
        raise RuntimeError("commit failed")

    monkeypatch.setattr(PdfFormFieldModel, "_suppress_duplicate_text", failing)
    with caplog.at_level("WARNING"):
        list(PdfFormFieldModel(enabled=True)(_conversion_result(page), [page]))

    assert "Form field extraction failed on page 1" in caplog.text
    assert page.predictions.form_fields is None
    assert page.predictions.layout.clusters == [paragraph]
    assert paragraph.cells == cells
    assert paragraph.bbox == bbox


def test_inline_host_is_not_dropped_as_rendered_duplicate() -> None:
    # A filled widget whose painted value the layout absorbed into a paragraph
    # slightly larger than the widget: the paragraph hosts the item in place,
    # so dropping it as a duplicate of the value would lose the value.
    host = _text_cluster(1, BoundingBox(l=10, t=10, r=90, b=20), "Amount 12")
    widgets = [_widget(0, BoundingBox(l=12, t=11, r=88, b=19), "Amount 12")]
    page = Page(page_no=1, size=Size(width=100, height=100))
    page.parsed_page = MagicMock(widgets=widgets)
    page.predictions.layout = LayoutPrediction(clusters=[host])

    list(PdfFormFieldModel(enabled=True)(_conversion_result(page), [page]))

    ((source, items),) = _regions(page)
    assert source == host.id
    assert [value.text for value in items[0].values] == ["Amount 12"]
    assert host in page.predictions.layout.clusters


def _field_region_page(page_wide: bool) -> Page:
    """Name: [box] / a note / Date: [box], in a FORM region or on the page."""
    clusters = [
        _text_cluster(1, BoundingBox(l=10, t=10, r=30, b=20), "Name:"),
        _text_cluster(2, BoundingBox(l=10, t=30, r=90, b=40), "Write in capitals."),
        _text_cluster(3, BoundingBox(l=10, t=50, r=30, b=60), "Date:"),
    ]
    if not page_wide:
        form = Cluster(
            id=4,
            label=DocItemLabel.FORM,
            bbox=BoundingBox(l=5, t=5, r=95, b=65),
            children=clusters,
        )
        clusters = [form, *clusters]
    page = Page(page_no=1, size=Size(width=100, height=100))
    page.parsed_page = MagicMock(
        widgets=[
            _widget(0, BoundingBox(l=35, t=10, r=90, b=20), "Ada"),
            _widget(1, BoundingBox(l=35, t=50, r=90, b=60), "2026-10-12"),
        ],
        hyperlinks=[],
    )
    page.predictions.layout = LayoutPrediction(clusters=clusters)
    return page


@pytest.mark.parametrize("page_wide", [False, True])
def test_fields_go_between_the_text_they_are_printed_between(page_wide: bool) -> None:
    # The keys leave the text, so the reading order has only the note left.
    # The note is printed between the two fields, so it is emitted there,
    # in a FORM region as on the page.
    doc = _materialize(_field_region_page(page_wide))

    order = []
    for item, _ in doc.iterate_items(with_groups=False):
        if isinstance(item, FieldItem):
            order.append(item.children[0].resolve(doc).text)
        elif item.label == DocItemLabel.TEXT:
            order.append(item.text)
    assert order == ["Name:", "Write in capitals.", "Date:"]
    # On the page each run of fields has a field region of its own.
    assert len(doc.field_regions) == (2 if page_wide else 1)


def _host_between_paragraphs(
    link: str | None = None,
) -> tuple[Page, Cluster, Cluster, Cluster]:
    """Three stacked paragraphs; the middle one hosts a checkbox widget.

    ``link`` puts a hyperlink annotation over the hosting paragraph.
    """
    host_bbox = BoundingBox(l=10, t=30, r=90, b=40)
    previous = _text_cluster(
        1, BoundingBox(l=10, t=10, r=90, b=20), "Applicants who live in"
    )
    host = _text_cluster(2, host_bbox, "check here if you rent")
    following = _text_cluster(
        3, BoundingBox(l=10, t=50, r=90, b=60), "and attach the lease"
    )
    hyperlinks = (
        []
        if link is None
        else [
            PdfHyperlink(
                index=0,
                rect=BoundingRectangle.from_bounding_box(
                    host_bbox.to_bottom_left_origin(page_height=100)
                ),
                uri=link,
            )
        ]
    )
    page = Page(page_no=1, size=Size(width=100, height=100))
    page.parsed_page = MagicMock(
        widgets=[
            _widget(
                0,
                BoundingBox(l=80, t=32, r=86, b=38),
                "",
                field_type="/Btn",
                appearance_state="/Off",
            )
        ],
        hyperlinks=hyperlinks,
    )
    page.predictions.layout = LayoutPrediction(clusters=[previous, host, following])
    return page, previous, host, following


@pytest.mark.parametrize("host_is_source", [True, False])
def test_reading_order_never_merges_an_inline_host(host_is_source: bool) -> None:
    # predict_merges joins a paragraph to its continuation at the top of the
    # next column, and the merged paragraph is not emitted on its own. A host
    # becomes a field item and takes no merged text: merged from, its
    # continuation would vanish; merged into, its widgets would.
    left, right = (10, 90), (110, 190)
    host_column, host_top = (left, 30) if host_is_source else (right, 10)
    host_bbox = BoundingBox(
        l=host_column[0], t=host_top, r=host_column[1], b=host_top + 10
    )
    previous = _text_cluster(
        1,
        BoundingBox(
            l=10, t=10 if host_is_source else 30, r=90, b=20 if host_is_source else 40
        ),
        "Applicants who live in",
    )
    host = _text_cluster(2, host_bbox, "check here if you rent")
    following = _text_cluster(
        3,
        BoundingBox(
            l=110, t=10 if host_is_source else 30, r=190, b=20 if host_is_source else 40
        ),
        "and attach the lease",
    )
    page = Page(page_no=1, size=Size(width=200, height=100))
    page.parsed_page = MagicMock(
        widgets=[
            _widget(
                0,
                BoundingBox(
                    l=host_bbox.r - 10,
                    t=host_top + 2,
                    r=host_bbox.r - 4,
                    b=host_top + 8,
                ),
                "",
                field_type="/Btn",
                appearance_state="/Off",
            )
        ],
        hyperlinks=[],
    )
    page.predictions.layout = LayoutPrediction(clusters=[previous, host, following])

    doc = _materialize(page)

    texts = [(text.label, text.text) for text in doc.texts]
    assert (DocItemLabel.TEXT, "Applicants who live in") in texts
    assert (DocItemLabel.FIELD_KEY, "check here if you rent") in texts
    assert (DocItemLabel.TEXT, "and attach the lease") in texts
    (item,) = doc.field_items
    assert [child.resolve(doc).label for child in item.children] == [
        DocItemLabel.FIELD_KEY,
        DocItemLabel.FIELD_VALUE,
    ]


def test_inline_host_keeps_its_hyperlink() -> None:
    page, _, _, _ = _host_between_paragraphs(link="https://example.org/rent")

    doc = _materialize(page)

    (key,) = [text for text in doc.texts if text.label == DocItemLabel.FIELD_KEY]
    assert str(key.hyperlink) == "https://example.org/rent"


def test_list_item_host_keeps_the_list_made_of_list_items() -> None:
    # A list item that hosts a checkbox becomes a field item in place. List
    # serializers expect list items under a list, so the field item sits in an
    # empty list item of its own.
    texts = ["Read the terms first", "I agree to the terms", "Sign below"]
    clusters = [
        _text_cluster(n + 1, BoundingBox(l=10, t=10 + 15 * n, r=90, b=20 + 15 * n), t)
        for n, t in enumerate(texts)
    ]
    for cluster in clusters:
        cluster.label = DocItemLabel.LIST_ITEM
    page = Page(page_no=1, size=Size(width=100, height=100))
    page.parsed_page = MagicMock(
        widgets=[
            _widget(
                0,
                BoundingBox(l=82, t=27, r=88, b=33),
                "",
                field_type="/Btn",
                appearance_state="/Off",
            )
        ],
        hyperlinks=[],
    )
    page.predictions.layout = LayoutPrediction(clusters=clusters)

    doc = _materialize(page)

    (group,) = doc.groups
    children = [child.resolve(doc) for child in group.children]
    assert all(isinstance(child, ListItem) for child in children)
    assert [child.text for child in children] == [texts[0], "", texts[2]]
    (item,) = [grandchild.resolve(doc) for grandchild in children[1].children]
    assert isinstance(item, FieldItem)
    key = item.children[0].resolve(doc)
    assert (key.label, key.text) == (DocItemLabel.FIELD_KEY, texts[1])


@pytest.mark.parametrize(
    ("label", "widget"),
    [
        # A text box beside a heading, and a checkbox inside one: a heading
        # never hosts a field in place, so both go to a field region.
        (DocItemLabel.SECTION_HEADER, "beside"),
        (DocItemLabel.SECTION_HEADER, "inside"),
        (DocItemLabel.FORMULA, "beside"),
    ],
)
def test_heading_or_formula_key_is_a_plain_key_in_a_field_region(
    label: DocItemLabel, widget: str
) -> None:
    key = _text_cluster(1, BoundingBox(l=10, t=10, r=60, b=20), "Applicant name")
    key.label = label
    widgets = (
        [_widget(0, BoundingBox(l=65, t=10, r=95, b=20), "Jane Doe")]
        if widget == "beside"
        else [
            _widget(
                0,
                BoundingBox(l=52, t=12, r=58, b=18),
                "",
                field_type="/Btn",
                appearance_state="/Off",
            )
        ]
    )
    page = Page(page_no=1, size=Size(width=100, height=100))
    page.parsed_page = MagicMock(widgets=widgets, hyperlinks=[])
    page.predictions.layout = LayoutPrediction(clusters=[key])

    doc = _materialize(page)

    (item,) = doc.field_items
    assert isinstance(item.parent.resolve(doc), FieldRegionItem)
    field_key = item.children[0].resolve(doc)
    assert field_key.label == DocItemLabel.FIELD_KEY
    assert field_key.text == "Applicant name"
    assert field_key.prov
    assert not field_key.children
    assert "Applicant name" in doc.export_to_markdown()


@pytest.mark.parametrize("boundary", ["ordinary", "zero_area", "table_scope"])
@pytest.mark.parametrize("abstain", [False, True])
def test_painted_cells_survive_candidate_filters_and_abstention(
    boundary: str, abstain: bool, monkeypatch
) -> None:
    # Recognition must cover cells cleanup can read even when candidate filters
    # reject them. Painted cells must not become atoms shared with keys.
    key = _text_cluster(1, BoundingBox(l=10, t=10, r=28, b=14), "Full name:")
    painted = _text_cluster(
        2,
        BoundingBox(l=31, t=31, r=31 if boundary == "zero_area" else 50, b=34),
        " Paris ",
    )
    painted.cells[0].index = 2
    key.cells += painted.cells
    key.bbox = BoundingBox.enclosing_bbox([key.bbox, painted.bbox])
    clusters = [key]
    if boundary == "table_scope":
        clusters.append(
            Cluster(
                id=3,
                label=DocItemLabel.TABLE,
                bbox=BoundingBox(l=30, t=30, r=70, b=35),
            )
        )
    page = Page(page_no=1, size=Size(width=100, height=100))
    page.parsed_page = MagicMock(
        widgets=[
            _widget(0, BoundingBox(l=30, t=9.5, r=70, b=14.5), "John Smith"),
            _widget(1, BoundingBox(l=30, t=30, r=70, b=35), "Paris"),
        ]
    )
    page.predictions.layout = LayoutPrediction(clusters=clusters)
    if abstain:
        monkeypatch.setattr(
            "docling.models.stages.form_field.keying.inputs.MAX_KEY_TEXTS", 0
        )
        monkeypatch.setattr(
            "docling.models.stages.form_field.keying.solver.MAX_KEY_TEXTS", 0
        )

    assignment = form_field_model._assign(page)

    assert assignment.painted_cells == {(key.id, 2)}
    assert all(key_text.text == "Full name:" for key_text in assignment.key_texts)
    assert set().union(*assignment.sources.values()) == {(key.id, 0)}
    assert [value.native.index for value in assignment.values] == [0, 1]
    assert (assignment.solver_status == "optimal") is not abstain
    list(PdfFormFieldModel(enabled=True)(_conversion_result(page), [page]))
    assert (key in page.predictions.layout.clusters) is abstain
    assert [
        (item.key_text, [value.text for value in item.values])
        for _, items in _regions(page)
        for item in items
    ] == [("" if abstain else "Full name:", ["John Smith"]), ("", ["Paris"])]
