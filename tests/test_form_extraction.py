# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import json
from pathlib import Path, PurePath
from unittest.mock import MagicMock

from docling_core.types.doc import BoundingBox, DocItemLabel, Size
from docling_core.types.doc.items.form import FieldItem, FieldValueItem
from docling_core.types.doc.page import BoundingRectangle, PdfWidget, TextCell

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.datamodel.base_models import (
    AssembledUnit,
    Cluster,
    FieldRegionElement,
    InputFormat,
    LayoutPrediction,
    Page,
)
from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.models.stages.form_field.form_field_model import PdfFormFieldModel
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
) -> PdfWidget:
    return PdfWidget(
        index=index,
        rect=BoundingRectangle.from_bounding_box(
            bbox.to_bottom_left_origin(page_height=100)
        ),
        widget_text=text,
        widget_field_type=field_type,
        widget_appearance_state=appearance_state,
    )


def test_docling_parse_exposes_complete_widget_contract_in_native_order() -> None:
    input_doc = InputDocument(
        path_or_stream=FORM_PDF,
        format=InputFormat.PDF,
        backend=DoclingParseDocumentBackend,
    )
    backend = input_doc._backend
    page = backend.load_page(0)
    try:
        segmented_page = page.get_segmented_page()
        assert segmented_page is not None
        assert [widget.widget_field_name for widget in segmented_page.widgets] == (
            EXPECTED_NAMES
        )
        assert [widget.index for widget in segmented_page.widgets] == [0, 1, 2, 3]
        assert segmented_page.widgets[0].widget_text == "Ada Lovelace"
        assert segmented_page.widgets[0].widget_description == "Full legal name"
        assert segmented_page.widgets[0].widget_field_type == "/Tx"
        assert segmented_page.widgets[0].widget_field_flags == 2
        assert segmented_page.widgets[1].widget_appearance_state == "/Yes"
        assert segmented_page.widgets[2].widget_appearance_state == "/Off"
        assert segmented_page.widgets[3].widget_text in {None, ""}
        assert segmented_page.widgets[3].widget_appearance_state == "/On"
    finally:
        page.unload()
        backend.unload()


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
    ]
    page = Page(page_no=1, size=Size(width=100, height=100))
    page.parsed_page = MagicMock(widgets=widgets)
    page.predictions.layout = LayoutPrediction(clusters=[form_1, form_2, form_3])
    conv_res = _conversion_result(page)

    assert list(PdfFormFieldModel(enabled=True)(conv_res, [page])) == [page]
    assert [
        region.source_container_id for region in page.predictions.field_regions
    ] == [
        1,
        2,
        3,
        None,
    ]

    # Keyless items: one value each, one item per widget -- flat shape preserved.
    def _region_values(region):
        return [value for item in region.items for value in item.values]

    assert [
        [value.text for value in _region_values(region)]
        for region in page.predictions.field_regions
    ] == [["first", "third", "tie"], ["second"], [""], [""]]
    assert [
        [value.checkbox for value in _region_values(region)]
        for region in page.predictions.field_regions
    ] == [[None, None, None], [None], ["selected"], ["unselected"]]
    assert all(
        item.key_text == ""
        for region in page.predictions.field_regions
        for item in region.items
    )
    assert page.predictions.field_regions[2].items[0].values[0].orig == "/On"
    assert page.predictions.field_regions[3].items[0].values[0].orig == "/Off"

    visible_child = Cluster(
        id=4,
        label=DocItemLabel.TEXT,
        bbox=BoundingBox(l=5, t=5, r=35, b=9),
    )
    retained_form_1 = form_1.model_copy(
        update={
            "bbox": BoundingBox(l=5, t=5, r=35, b=45),
            "children": [visible_child],
        }
    )
    retained_form_2 = form_2.model_copy(
        update={"bbox": BoundingBox(l=60, t=5, r=95, b=45)}
    )
    page.predictions.layout = LayoutPrediction(
        clusters=[retained_form_1, retained_form_2, visible_child]
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
    assert len(region_elements) == 4
    assert region_elements[0].id == form_1.id
    assert region_elements[0].cluster.bbox == form_1.bbox
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
    assert ["first", "third", "tie"] in values_by_region
    assert [DocItemLabel.CHECKBOX_SELECTED.value] in values_by_region
    assert [DocItemLabel.CHECKBOX_UNSELECTED.value] in values_by_region
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


def _checkbox_cluster(
    cluster_id: int, bbox: BoundingBox, *, label: str, mark: str
) -> Cluster:
    cells = [
        TextCell(
            index=0,
            rect=BoundingRectangle.from_bounding_box(bbox),
            text=label,
            orig=label,
            from_ocr=False,
        ),
        TextCell(
            index=1,
            rect=BoundingRectangle.from_bounding_box(bbox),
            text=mark,
            orig=mark,
            from_ocr=False,
        ),
    ]
    return Cluster(
        id=cluster_id,
        label=DocItemLabel.CHECKBOX_SELECTED,
        bbox=bbox,
        cells=cells,
    )


def test_checkbox_widget_lifts_cluster_label_and_lets_as_override_classifier() -> None:
    # The widget rect sits inside a CHECKBOX_SELECTED cluster that also absorbed
    # the option label "4797" and the mark glyph. The cluster's text is lifted
    # verbatim onto the value and the cluster is promoted out of the plain-text
    # stream; but /AS ("/Off") -- not the classifier's "selected" -- decides state.
    widget_bbox = BoundingBox(l=10, t=10, r=18, b=18)
    cluster_bbox = BoundingBox(l=8, t=8, r=40, b=20)
    cb_cluster = _checkbox_cluster(2, cluster_bbox, label="4797", mark="✔")
    widget = _widget(0, widget_bbox, "/Off", field_type="/Btn", appearance_state="/Off")
    page = Page(page_no=1, size=Size(width=100, height=100))
    page.parsed_page = MagicMock(widgets=[widget])
    page.predictions.layout = LayoutPrediction(clusters=[cb_cluster])
    conv_res = _conversion_result(page)

    list(PdfFormFieldModel(enabled=True)(conv_res, [page]))

    value = page.predictions.field_regions[0].items[0].values[0]
    assert value.text == ""
    assert value.checkbox == "unselected"  # from /AS, overriding the classifier
    assert value.checkbox_label == "4797 ✔"  # cluster text lifted as-is
    assert value.bbox == cluster_bbox  # prov wraps the whole cluster, not the widget
    assert cb_cluster not in page.predictions.layout.clusters  # promoted, not left


def test_pipeline_materializes_format_neutral_fields() -> None:
    result = _convert(extract_form_fields=True)

    assert result.pages[0].assembled is not None
    # The layout detector merges the three checkbox lines into one text cluster,
    # so their widgets share an enclosing paragraph and group under one keyed
    # field_item materialized in place of that paragraph. The text field has no
    # enclosing paragraph and stays a keyless value in the page-wide region.
    assert [
        region.source_container_id
        for region in result.pages[0].predictions.field_regions
    ] == [0, None]
    # Only the text field's page-wide region survives as a field_region; the
    # paragraph item lives inline in the body, not wrapped in a region of its own.
    assert len(result.document.field_regions) == 1
    # One keyed inline item (paragraph key + three checkbox values) and one
    # keyless single-value item for the text field.
    assert len(result.document.field_items) == 2
    keys = [
        item for item in result.document.texts if item.label == DocItemLabel.FIELD_KEY
    ]
    assert len(keys) == 1
    assert "I agree to the terms" in keys[0].text
    # The keyed field_item materializes in the paragraph's own place, NOT wrapped
    # in a field_region of its own (it would be, if pulled out into a region).
    keyed_item = keys[0].parent.resolve(result.document)
    assert keyed_item.label == DocItemLabel.FIELD_ITEM
    assert keyed_item.parent.resolve(result.document).label != DocItemLabel.FIELD_REGION

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
    serialized = json.dumps(result.document.export_to_dict())
    assert "widget_field_name" not in serialized
    assert "widget_appearance_state" not in serialized

    disabled = _convert(extract_form_fields=False)
    assert disabled.pages[0].predictions.field_regions == []
    assert disabled.document.field_regions == []
    assert disabled.document.field_items == []
    # The hosting paragraph is not dropped -- it is reinterpreted as a field_item
    # at materialization -- so extraction leaves the layout clusters untouched.
    assert result.pages[0].predictions.layout == disabled.pages[0].predictions.layout


def test_widgets_inlined_in_paragraph_group_into_one_keyed_item() -> None:
    # IRS Sch.1 line 7 shape: a sentence inlines a checkbox and a fillable amount.
    # The layout detects the whole line as one text cluster enclosing both
    # widgets; a FORM cluster also wraps the page. The smaller text cluster is the
    # more specific host, so both widgets group under one keyed item -- key = the
    # paragraph, two values (checkbox + amount) -- not two keyless FORM fields.
    paragraph = _text_cluster(
        1,
        BoundingBox(l=10, t=10, r=90, b=20),
        "check here and enter amount repaid:",
    )
    page_form = Cluster(
        id=2, label=DocItemLabel.FORM, bbox=BoundingBox(l=0, t=0, r=100, b=100)
    )
    widgets = [
        _widget(
            0,
            BoundingBox(l=40, t=12, r=44, b=18),
            "",
            field_type="/Btn",
            appearance_state="/Off",
        ),
        _widget(1, BoundingBox(l=70, t=12, r=85, b=18), "1221.00"),
    ]
    page = Page(page_no=1, size=Size(width=100, height=100))
    page.parsed_page = MagicMock(widgets=widgets)
    page.predictions.layout = LayoutPrediction(clusters=[page_form, paragraph])
    conv_res = _conversion_result(page)

    list(PdfFormFieldModel(enabled=True)(conv_res, [page]))

    keyed = [
        (region, item)
        for region in page.predictions.field_regions
        for item in region.items
        if item.key_text
    ]
    assert len(keyed) == 1
    region, item = keyed[0]
    assert region.source_container_id == paragraph.id
    assert item.key_text == "check here and enter amount repaid:"
    assert item.key_bbox == paragraph.bbox
    assert [(v.text, v.checkbox) for v in item.values] == [
        ("", "unselected"),
        ("1221.00", None),
    ]
    # The paragraph stays in the layout -- it is reinterpreted as a field_item in
    # place at materialization, keeping its position in its list/container.
    assert paragraph in page.predictions.layout.clusters
