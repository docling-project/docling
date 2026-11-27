# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest
from docling_core.types.doc import (
    BoundingBox,
    CodeLanguageLabel,
    DocItemLabel,
    DoclingDocument,
    Size,
)
from PIL import Image as PILImage
from pydantic import ValidationError

from docling.backend.docling_parse_backend import ThreadedDoclingParseDocumentBackend
from docling.datamodel.base_models import Cluster, InputFormat, Page, TextElement
from docling.datamodel.document import InputDocument
from docling.datamodel.pipeline_options import VlmConvertOptions
from docling.experimental.datamodel.layout_crop_vlm_pipeline_options import (
    DOCLANG_FORMULA_TASK,
    DOCLANG_OCR_TASK,
    DOCLANG_TABLE_TASK,
    CropTask,
    LayoutCropVlmOptions,
)
from docling.experimental.models.layout_crop_vlm_model import (
    LayoutCropVlmModel,
    native_tile_canvas,
)
from docling.models.inference_engines.vlm import VlmEngineType
from docling.models.stages.reading_order.readingorder_model import (
    ReadingOrderModel,
    ReadingOrderOptions,
)


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
    formula = model._task_for(_cluster(DocItemLabel.FORMULA))
    assert formula is not None and formula.prompt == DOCLANG_FORMULA_TASK.prompt
    header = model._task_for(_cluster(DocItemLabel.SECTION_HEADER))
    assert header is not None and header.prompt == DOCLANG_OCR_TASK.prompt
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


def test_500m_caps_text_tiles_and_pads_at_least_4_points():
    options = LayoutCropVlmOptions()
    assert options.crop_min_padding == 4.0
    assert options.default_task.max_tiles == 8
    assert options.tasks[DocItemLabel.FORMULA].max_tiles == 8
    assert options.tasks[DocItemLabel.CODE].max_tiles == 8
    assert options.tasks[DocItemLabel.TABLE].max_tiles is None
    assert options.tasks[DocItemLabel.DOCUMENT_INDEX].max_tiles is None
    # The shared task constants stay uncapped.
    assert DOCLANG_OCR_TASK.max_tiles is None


def test_2b_keeps_the_processor_limits():
    options = LayoutCropVlmOptions(
        vlm_options=VlmConvertOptions.from_preset("granite_for_docling_2b")
    )
    assert options.crop_min_padding == 0.0
    assert options.default_task.max_tiles is None
    assert all(task.max_tiles is None for task in options.tasks.values())


def test_explicit_crop_settings_are_kept():
    options = LayoutCropVlmOptions(crop_min_padding=0.0, default_task=DOCLANG_OCR_TASK)
    assert options.crop_min_padding == 0.0
    assert options.default_task.max_tiles is None
    assert options.tasks[DocItemLabel.FORMULA].max_tiles == 8


_OCR_8_TILES = DOCLANG_OCR_TASK.model_copy(update={"max_tiles": 8})


def test_tile_cap_goes_to_the_processor():
    base = {"extra_processor_kwargs": {"fine_route": False}, "top_k": 1}
    config = LayoutCropVlmModel._generation_config(
        base, _OCR_8_TILES, VlmEngineType.TRANSFORMERS
    )
    assert config["extra_processor_kwargs"] == {
        "fine_route": False,
        "max_patches": 8,
    }
    assert config["top_k"] == 1
    assert base["extra_processor_kwargs"] == {"fine_route": False}
    table = LayoutCropVlmModel._generation_config(
        base, DOCLANG_TABLE_TASK, VlmEngineType.TRANSFORMERS
    )
    assert table is base
    # vLLM rejects a per-request tile limit.
    vllm = LayoutCropVlmModel._generation_config(base, _OCR_8_TILES, VlmEngineType.VLLM)
    assert vllm is base


_CROPBOXES: list[BoundingBox | None] = []


class _RecordingPage(Page):
    """Page whose rendering only records the crop box it was asked for."""

    def get_image(  # type: ignore[override]
        self, scale=1.0, max_size=None, cropbox=None
    ):
        _CROPBOXES.append(cropbox)
        if cropbox is None:
            return None
        return PILImage.new(
            "RGB", (round(cropbox.width * scale), round(cropbox.height * scale))
        )


def test_padding_is_relative_with_a_minimum():
    page = _RecordingPage(page_no=1, size=Size(width=600, height=800))
    model = _model()
    # 190 x 60 points: 2 % is 3.8 x 1.2 points, below the 4-point minimum.
    model._crop(page, _cluster(DocItemLabel.TEXT), DOCLANG_OCR_TASK)
    box = _CROPBOXES[-1]
    assert (box.l, box.t, box.r, box.b) == (6, 16, 204, 84)
    # 500 x 400 points: 2 % (10 x 8 points) is above the minimum.
    big = Cluster(
        id=8, label=DocItemLabel.TABLE, bbox=BoundingBox(l=50, t=50, r=550, b=450)
    )
    model._crop(page, big, DOCLANG_TABLE_TASK)
    box = _CROPBOXES[-1]
    assert (box.l, box.t, box.r, box.b) == (40, 42, 560, 458)


def test_vertical_crop_goes_on_a_native_canvas():
    page = _RecordingPage(page_no=1, size=Size(width=600, height=800))
    model = _model()
    vertical = Cluster(
        id=9, label=DocItemLabel.TEXT, bbox=BoundingBox(l=100, t=100, r=120, b=500)
    )
    image = model._crop(page, vertical, model.options.default_task)
    assert image is not None
    assert image.width % 512 == 0 and image.height % 512 == 0
    assert (image.width // 512) * (image.height // 512) <= 8
    # A wide crop is left to the processor.
    image = model._crop(page, _cluster(DocItemLabel.TEXT), DOCLANG_OCR_TASK)
    assert image is not None and image.size == (396, 136)


def test_native_canvas_keeps_the_aspect_ratio():
    crop = PILImage.new("RGB", (30, 600), (255, 255, 255))
    canvas = native_tile_canvas(crop, max_tiles=8)
    # Short side 30 px -> upscaled to 40 px: 40 x 800, one column of two tiles.
    assert canvas.size == (512, 1024)
    assert canvas.getpixel((0, 0)) == (255, 255, 255)


def test_furniture_reply_keeps_text():
    cluster = _cluster(DocItemLabel.PAGE_HEADER)
    _model()._apply_text(
        cluster,
        '<doclang><page_header><layer value="furniture"/>Quarterly report</page_header>',
    )
    assert [cell.text for cell in cluster.cells] == ["Quarterly report"]


def test_code_reply_keeps_language_through_assembly():
    cluster = _cluster(DocItemLabel.CODE)
    _model()._apply_text(
        cluster, '<doclang><code><label value="Python"/>print(1)</code>'
    )
    element = TextElement(
        id=cluster.id,
        page_no=1,
        label=cluster.label,
        cluster=cluster,
        text=cluster.cells[0].text,
    )
    doc = DoclingDocument(name="code")
    doc.add_page(page_no=1, size=Size(width=200, height=300))
    item = ReadingOrderModel(ReadingOrderOptions())._add_code_element(element, doc)
    assert item.text == "print(1)"
    assert item.code_language == CodeLanguageLabel.PYTHON


@pytest.mark.parametrize("value", [0, -1])
def test_crop_limits_require_positive_values(value):
    with pytest.raises(ValidationError, match="max_tiles"):
        CropTask(prompt="ocr", max_tiles=value)
    with pytest.raises(ValidationError, match="engine_batch_size"):
        LayoutCropVlmOptions(engine_batch_size=value)
    with pytest.raises(ValueError, match="max_tiles"):
        native_tile_canvas(PILImage.new("RGB", (30, 300)), max_tiles=value)


@pytest.mark.parametrize("engine", [VlmEngineType.API, VlmEngineType.API_OPENAI])
def test_api_crop_limit_is_reported(engine, caplog, monkeypatch):
    monkeypatch.setattr(LayoutCropVlmModel, "_warned_api_tiles", False)
    base = {"api_params": {"model": "served-model"}}
    for _ in range(2):
        assert LayoutCropVlmModel._generation_config(base, _OCR_8_TILES, engine) == base
    assert caplog.text.count("server's tile limit") == 1


def test_form_crop_keeps_matching_pdf_line_positions():
    source = InputDocument(
        path_or_stream=Path("tests/data/pdf/sources/acroform_sample.pdf"),
        format=InputFormat.PDF,
        backend=ThreadedDoclingParseDocumentBackend,
    )
    backend = source._backend
    page_backend = next(backend.iter_pages())
    try:
        page = Page(page_no=1, parsed_page=page_backend.get_segmented_page())
        cluster = Cluster(
            id=1, label=DocItemLabel.TEXT, bbox=BoundingBox(l=90, t=140, r=300, b=190)
        )
        model = _model()
        model._apply_text(
            cluster,
            "<doclang><text>I agree to the terms</text><text>Subscribe to newsletter</text>",
            page,
        )
        assert [cell.text.strip() for cell in cluster.cells] == [
            "I agree to the terms",
            "Subscribe to newsletter",
        ]
        assert cluster.cells[0].rect != cluster.cells[1].rect

        model._apply_text(
            cluster, "<doclang><text>Different recognized text</text>", page
        )
        assert [cell.text for cell in cluster.cells] == ["Different recognized text"]
    finally:
        page_backend.unload()
        backend.unload()
