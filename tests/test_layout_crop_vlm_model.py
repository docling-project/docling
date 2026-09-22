# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from docling_core.types.doc import BoundingBox, DocItemLabel

from docling.datamodel.base_models import Cluster, Page
from docling.experimental.datamodel.layout_crop_vlm_pipeline_options import (
    DOCLANG_FORMULA_TASK,
    DOCLANG_OCR_TASK,
    DOCLANG_TABLE_TASK,
    LayoutCropVlmOptions,
)
from docling.experimental.models.layout_crop_vlm_model import LayoutCropVlmModel


def _model() -> LayoutCropVlmModel:
    # A disabled stage builds no engine; the reply handling does not need one.
    model = LayoutCropVlmModel(
        enabled=False,
        enable_remote_services=False,
        artifacts_path=None,
        options=LayoutCropVlmOptions(),
        accelerator_options=None,  # type: ignore[arg-type]
    )
    return model


def _cluster(label: DocItemLabel) -> Cluster:
    return Cluster(id=7, label=label, bbox=BoundingBox(l=10, t=20, r=200, b=80))


def test_task_follows_layout_label():
    model = _model()
    assert model._task_for(_cluster(DocItemLabel.TABLE)) == DOCLANG_TABLE_TASK
    assert model._task_for(_cluster(DocItemLabel.FORMULA)) == DOCLANG_FORMULA_TASK
    assert model._task_for(_cluster(DocItemLabel.SECTION_HEADER)) == DOCLANG_OCR_TASK
    assert model._task_for(_cluster(DocItemLabel.PICTURE)) is None


def test_text_reply_without_closing_tag():
    # Servers strip the `</doclang>` stop string from the reply.
    cluster = _cluster(DocItemLabel.TEXT)
    _model()._apply_text(cluster, "<doclang><text>Hello crop world.</text>")
    assert [cell.text for cell in cluster.cells] == ["Hello crop world."]


def test_truncated_reply_keeps_its_text():
    cluster = _cluster(DocItemLabel.TEXT)
    _model()._apply_text(cluster, "<doclang><text>cut off by the token bud")
    assert cluster.cells[0].text == "cut off by the token bud"


def test_table_reply_becomes_table_prediction():
    page = Page(page_no=1)
    cluster = _cluster(DocItemLabel.TABLE)
    reply = (
        "<doclang><table><ched/>Name<ched/>Value<nl/>"
        "<fcel/>alpha<fcel/>1<nl/><fcel/>beta<ecel/><nl/></table>"
    )
    _model()._apply_table(page, cluster, reply)

    assert page.predictions.tablestructure is not None
    table = page.predictions.tablestructure.table_map[cluster.id]
    assert (table.num_rows, table.num_cols) == (3, 2)
    assert table.cluster is cluster
    texts = {
        (cell.start_row_offset_idx, cell.start_col_offset_idx): cell.text
        for cell in table.table_cells
    }
    assert texts[(0, 0)] == "Name"
    assert texts[(1, 1)] == "1"
    assert texts[(2, 0)] == "beta"
