# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from pathlib import PurePath
from unittest.mock import Mock

from docling_core.types.doc import (
    BoundingBox,
    DocItemLabel,
    GroupItem,
    Size,
    TableCell,
    TableItem,
    TextItem,
)
from docling_core.types.doc.page import BoundingRectangle, TextCell
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import (
    AssembledUnit,
    Cluster,
    InputFormat,
    Page,
    Table,
)
from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options import TableStructureOptions
from docling.models.stages.reading_order.readingorder_model import (
    ReadingOrderModel,
    ReadingOrderOptions,
)
from docling.models.stages.table_structure.table_structure_model import (
    TableStructureModel,
)


def _text_cell(
    index: int,
    text: str,
    bbox: tuple[float, float, float, float],
) -> TextCell:
    left, top, right, bottom = bbox
    return TextCell(
        index=index,
        rect=BoundingRectangle.from_bounding_box(
            BoundingBox(l=left, t=top, r=right, b=bottom)
        ),
        text=text,
        orig=text,
        from_ocr=True,
    )


def _predict_table(cells: list[TextCell]) -> Table:
    model = TableStructureModel(
        enabled=False,
        artifacts_path=None,
        options=TableStructureOptions(),
        accelerator_options=AcceleratorOptions(),
    )
    model.scale = 1.0
    model.tf_predictor = Mock()
    table_cell = TableCell(
        text="matched",
        bbox=None,
        start_row_offset_idx=0,
        end_row_offset_idx=1,
        start_col_offset_idx=0,
        end_col_offset_idx=1,
    )
    model.tf_predictor.multi_table_predict.return_value = [
        {
            "tf_responses": [table_cell.model_dump()],
            "predict_details": {
                "matches": {"10": [{"table_cell_id": 0}]},
                "num_rows": 1,
                "num_cols": 1,
                "prediction": {"rs_seq": ["fcel", "nl"]},
            },
        }
    ]
    cluster = Cluster(
        id=1,
        label=DocItemLabel.TABLE,
        bbox=BoundingBox(l=0, t=0, r=100, b=100),
        cells=cells,
    )

    return model._do_prediction_on_image_to_table(
        table_image=Image.new("RGB", (100, 100)),
        table_cluster=cluster,
        page_no=1,
    )


def _render_table(table: Table, *, recover: bool):
    conv_res = ConversionResult(
        input=InputDocument.model_construct(
            file=PurePath("table.pdf"),
            document_hash="0" * 64,
            format=InputFormat.PDF,
        ),
        pages=[Page(page_no=1, size=Size(width=100, height=100))],
        assembled=AssembledUnit(elements=[table], body=[table]),
    )
    model = ReadingOrderModel(
        options=ReadingOrderOptions(recover_orphaned_table_text=recover)
    )
    elements = model._assembled_to_readingorder_elements(conv_res)
    return model._readingorder_elements_to_docling_doc(
        conv_res,
        ordered_siblings={None: elements},
        el_to_captions_mapping={},
        el_to_footnotes_mapping={},
        el_merges_mapping={},
    )


def test_tableformer_prediction_carries_unmatched_input_cells():
    matched = _text_cell(10, "matched", (0, 0, 10, 10))
    unmatched = _text_cell(11, "Signature John Doe", (20, 0, 40, 10))
    empty = _text_cell(12, "  ", (50, 0, 60, 10))

    table = _predict_table([matched, unmatched, empty])

    assert table.unmatched_table_cells == [unmatched]


def test_recovery_is_disabled_by_default():
    unmatched = _text_cell(11, "Signature John Doe", (20, 0, 40, 10))

    doc = _render_table(_predict_table([unmatched]), recover=False)

    assert doc.groups == []
    assert doc.texts == []
    assert "Signature John Doe" not in doc.export_to_markdown()


def test_recovery_appends_unmatched_input_text_after_table():
    unmatched = [
        _text_cell(11, "Signature", (20, 0, 30, 10)),
        _text_cell(12, "John Doe", (31, 0, 50, 10)),
    ]

    doc = _render_table(_predict_table(unmatched), recover=True)

    body_items = [child.resolve(doc) for child in doc.body.children]
    assert len(body_items) == 2
    assert isinstance(body_items[0], TableItem)
    assert isinstance(body_items[1], GroupItem)

    recovered_items = [child.resolve(doc) for child in body_items[1].children]
    assert len(recovered_items) == 1
    assert isinstance(recovered_items[0], TextItem)
    assert recovered_items[0].text == "Signature John Doe"
    assert "Signature John Doe" in doc.export_to_markdown()
    doc.validate_document()
