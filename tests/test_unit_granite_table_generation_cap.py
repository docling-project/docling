# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Granite Vision table generation must terminate, and a failed table must
not take the text under it along.

The granite-vision-4.1-4b tokenizer carries the transformers placeholder
``model_max_length`` (1e30), and the stage used to pass it as
``max_new_tokens``. A table crop that never emits end-of-text then generated
until the process was killed (issue #4657). With a budget in place the same
crops still produced truncated or looping OTSL whose few parsed rows replaced
the page text under the table while the conversion reported success. These
tests stand in for the model call and check the budget, the loop detector,
the output cleanup, the fallback to the page text, and the error report.
"""

import logging
from unittest.mock import MagicMock

import torch
from docling_core.types.doc import BoundingBox, DocItemLabel, Size, TableCell
from docling_core.types.doc.page import (
    BoundingRectangle,
    PdfPageBoundaryType,
    PdfPageGeometry,
    SegmentedPdfPage,
    TextCell,
)
from PIL import Image
from transformers import StoppingCriteriaList

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import (
    Cluster,
    DoclingComponentType,
    ErrorItem,
    FailureCategory,
    LayoutPrediction,
    Page,
    Table,
)
from docling.datamodel.pipeline_options import GraniteVisionTableStructureOptions
from docling.models.stages.table_structure.table_structure_model_granite_vision import (
    GraniteVisionTableStructureModel,
    native_text_recall,
)
from docling.models.utils.hf_stopping_criteria import HFStoppingCriteriaWrapper

PROMPT_TOKENS = 5
PLACEHOLDER_MODEL_MAX_LENGTH = int(1e30)
PAGE_WIDTH, PAGE_HEIGHT = 600, 800
TABLE_BBOX = BoundingBox(l=100, t=100, r=400, b=300)

HEADER_ROW = "<ched>Name</ched><ched>Value</ched><nl>"
DATA_ROW = "<fcel>Foo</fcel><fcel>42</fcel><nl>"
LOOP_ROW = "<fcel>x</fcel><fcel>x</fcel><nl>"
MORE_ROWS = "<fcel>Bar</fcel><fcel>43</fcel><nl><fcel>Baz</fcel><fcel>44</fcel><nl>"
# A results row as a laboratory report produces it.
LAB_ROW = "<fcel>Hemoglobin</fcel><fcel>13.5 g/dL</fcel><fcel>12.0 - 16.0</fcel><nl>"
# Measured on the granite-vision-4.1-4b tokenizer: the OTSL tags are three to
# four tokens each, so a window of generated tokens decodes to about this many
# characters per token on a row of numbers.
OTSL_CHARS_PER_TOKEN = 1.7


def _model_with_mocked_inference(
    options: GraniteVisionTableStructureOptions,
    *,
    decoded_text: str,
    generated_tokens: int = 8,
) -> GraniteVisionTableStructureModel:
    model = GraniteVisionTableStructureModel(
        enabled=False,
        artifacts_path=None,
        options=options,
        accelerator_options=AcceleratorOptions(),
    )
    model.enabled = True
    model.device = "cpu"

    processor = MagicMock()
    processor.tokenizer.model_max_length = PLACEHOLDER_MODEL_MAX_LENGTH
    processor.tokenizer.pad_token_id = 0
    processor.apply_chat_template.return_value = "<tables_otsl>"
    processor.return_value.to.return_value = {
        "input_ids": torch.ones((1, PROMPT_TOKENS), dtype=torch.long)
    }
    processor.decode.return_value = decoded_text
    model._processor = processor

    model._model = MagicMock()
    model._model.generate.return_value = torch.ones(
        (1, PROMPT_TOKENS + generated_tokens), dtype=torch.long
    )
    return model


def _mock_page() -> tuple[MagicMock, Cluster]:
    """A page without any text: the stage only sees the table crop."""
    cluster = Cluster(
        id=0, label=DocItemLabel.TABLE, bbox=BoundingBox(l=0, t=0, r=10, b=10)
    )
    page = MagicMock()
    page.page_no = 1
    page._backend.is_valid.return_value = True
    page.predictions.layout.clusters = [cluster]
    page.predictions.tablestructure = None
    page.get_image.return_value = Image.new("RGB", (10, 10))
    page.cells = []
    return page, cluster


def _text_cell(
    index: int, bbox: BoundingBox, text: str, *, from_ocr: bool = False
) -> TextCell:
    return TextCell(
        index=index,
        rect=BoundingRectangle.from_bounding_box(bbox),
        text=text,
        orig=text,
        from_ocr=from_ocr,
    )


def _lab_cells(*, from_ocr: bool = False) -> list[TextCell]:
    """The eight words of a 4x2 table, one text line per cell."""
    words = ["Name", "Value", "Foo", "42", "Bar", "43", "Baz", "44"]
    cells = []
    for index, word in enumerate(words):
        row, col = divmod(index, 2)
        top = 110 + row * 40
        left = 110 + col * 150
        cells.append(
            _text_cell(
                index,
                BoundingBox(l=left, t=top, r=left + 130, b=top + 20),
                word,
                from_ocr=from_ocr,
            )
        )
    return cells


class _PageBackend:
    """The slice of PdfPageBackend the Granite stage uses while predicting."""

    def is_valid(self) -> bool:
        return True

    def get_page_image(self, scale: float = 1.0, cropbox=None) -> Image.Image:
        return Image.new("RGB", (10, 10))


def _page_with_text(cells: list[TextCell]) -> tuple[Page, Cluster]:
    """A real page whose text lines sit inside the table box."""
    full = BoundingBox(l=0, t=0, r=PAGE_WIDTH, b=PAGE_HEIGHT)
    geometry = PdfPageGeometry(
        angle=0,
        boundary_type=PdfPageBoundaryType.CROP_BOX,
        rect=BoundingRectangle.from_bounding_box(full),
        art_bbox=full,
        bleed_bbox=full,
        crop_bbox=full,
        media_bbox=full,
        trim_bbox=full,
    )
    cluster = Cluster(
        id=3, label=DocItemLabel.TABLE, bbox=TABLE_BBOX, confidence=0.8, cells=cells
    )
    page = Page(page_no=1, size=Size(width=PAGE_WIDTH, height=PAGE_HEIGHT))
    page._backend = _PageBackend()  # type: ignore[assignment]
    page.parsed_page = SegmentedPdfPage(
        dimension=geometry,
        textline_cells=cells,
        char_cells=[],
        word_cells=[],
        has_chars=False,
        has_words=False,
        has_lines=True,
    )
    page.predictions.layout = LayoutPrediction(clusters=[cluster])
    return page, cluster


def _predict(
    model: GraniteVisionTableStructureModel, page, cluster: Cluster
) -> tuple[Table, list[ErrorItem]]:
    conv_res = MagicMock()
    conv_res.errors = []
    predictions = model.predict_tables(conv_res, [page])
    return predictions[0].table_map[cluster.id], conv_res.errors


def _predicted_table(model: GraniteVisionTableStructureModel) -> Table:
    page, cluster = _mock_page()
    return _predict(model, page, cluster)[0]


def _cell_texts(table: Table) -> list[str]:
    return [c.text for c in table.table_cells]


def _only_error(errors: list[ErrorItem]) -> ErrorItem:
    assert len(errors) == 1
    (error,) = errors
    assert error.component_type == DoclingComponentType.MODEL
    assert error.category == FailureCategory.INFERENCE_FAILURE
    assert error.module_name == GraniteVisionTableStructureModel.__name__
    return error


def test_generation_uses_the_configured_budget_not_the_tokenizer_placeholder():
    options = GraniteVisionTableStructureOptions(max_new_tokens=512)
    model = _model_with_mocked_inference(options, decoded_text=HEADER_ROW)

    _predicted_table(model)

    gen_kwargs = model._model.generate.call_args.kwargs
    assert gen_kwargs["max_new_tokens"] == 512
    stopping = gen_kwargs["stopping_criteria"]
    assert isinstance(stopping, StoppingCriteriaList)
    assert len(stopping) == 1
    assert isinstance(stopping[0], HFStoppingCriteriaWrapper)


def test_default_budget_is_bounded():
    assert GraniteVisionTableStructureOptions().max_new_tokens == 8192


def test_loop_detector_window_holds_enough_copies_of_a_results_row():
    """The first fix inspected 1024 tokens: too few characters for 32 copies
    of a row of numbers, so such loops ran to the budget (issue #4657)."""
    model = _model_with_mocked_inference(
        GraniteVisionTableStructureOptions(), decoded_text=HEADER_ROW
    )
    stopper = model._repetition_stopper
    assert stopper is not None
    looping = LAB_ROW * 200

    old_window = looping[-int(1024 * OTSL_CHARS_PER_TOKEN) :]
    window = looping[-int(stopper.lookback_tokens() * OTSL_CHARS_PER_TOKEN) :]

    assert not stopper.should_stop(old_window)
    assert stopper.should_stop(window)


def test_stop_on_repetition_can_be_disabled():
    options = GraniteVisionTableStructureOptions(
        stop_on_repetition=False, min_text_recall=0
    )
    looping = HEADER_ROW + DATA_ROW + LOOP_ROW * 60
    model = _model_with_mocked_inference(options, decoded_text=looping)

    table = _predicted_table(model)

    assert "stopping_criteria" not in model._model.generate.call_args.kwargs
    assert table.num_rows == 62


def test_repeated_rows_are_dropped_but_their_first_copy_is_kept():
    """The loop starts on a row the model really read, then repeats it."""
    looping = HEADER_ROW + DATA_ROW + LOOP_ROW * 60
    model = _model_with_mocked_inference(
        GraniteVisionTableStructureOptions(), decoded_text=looping
    )

    table = _predicted_table(model)

    assert table.num_rows == 3
    assert table.num_cols == 2
    assert _cell_texts(table) == ["Name", "Value", "Foo", "42", "x", "x"]


def test_partial_loop_row_from_a_mid_unit_stop_is_cut_at_the_row_break():
    looping = HEADER_ROW + DATA_ROW + LOOP_ROW * 60 + LOOP_ROW[:11]
    model = _model_with_mocked_inference(
        GraniteVisionTableStructureOptions(), decoded_text=looping
    )

    table = _predicted_table(model)

    assert table.num_rows == 3
    assert _cell_texts(table) == ["Name", "Value", "Foo", "42", "x", "x"]


def test_loop_inside_a_row_drops_the_unfinished_row():
    looping = HEADER_ROW + DATA_ROW + "<fcel>x</fcel>" * 80
    model = _model_with_mocked_inference(
        GraniteVisionTableStructureOptions(), decoded_text=looping
    )

    table = _predicted_table(model)

    assert table.num_rows == 2
    assert _cell_texts(table) == ["Name", "Value", "Foo", "42"]


def test_short_runs_of_identical_rows_are_kept():
    """A real table may repeat rows; only a long run counts as a loop."""
    text = HEADER_ROW + DATA_ROW * 12
    model = _model_with_mocked_inference(
        GraniteVisionTableStructureOptions(), decoded_text=text
    )

    table = _predicted_table(model)

    assert table.num_rows == 13


def test_trimmed_loop_without_page_text_is_reported():
    looping = HEADER_ROW + DATA_ROW + LOOP_ROW * 60
    model = _model_with_mocked_inference(
        GraniteVisionTableStructureOptions(), decoded_text=looping
    )
    page, cluster = _mock_page()

    table, errors = _predict(model, page, cluster)

    assert table.num_rows == 3
    error = _only_error(errors)
    assert error.page_no == 1
    assert "repeated the same fragment" in error.error_message
    assert "could not be checked against the PDF text" in error.error_message


def test_trimmed_loop_confirmed_by_the_page_text_is_only_logged(caplog):
    cells = [
        *_lab_cells()[:4],
        _text_cell(8, BoundingBox(l=110, t=190, r=240, b=210), "x"),
        _text_cell(9, BoundingBox(l=260, t=190, r=390, b=210), "x"),
    ]
    page, cluster = _page_with_text(cells)
    looping = HEADER_ROW + DATA_ROW + LOOP_ROW * 60
    model = _model_with_mocked_inference(
        GraniteVisionTableStructureOptions(), decoded_text=looping
    )

    with caplog.at_level(logging.WARNING):
        table, errors = _predict(model, page, cluster)

    assert errors == []
    assert table.num_rows == 3
    assert cluster.children == []
    assert any("match the PDF text" in r.message for r in caplog.records)


def test_hitting_the_budget_keeps_the_page_text_and_reports_the_table():
    page, cluster = _page_with_text(_lab_cells())
    options = GraniteVisionTableStructureOptions(max_new_tokens=8)
    model = _model_with_mocked_inference(
        options, decoded_text=HEADER_ROW, generated_tokens=8
    )

    table, errors = _predict(model, page, cluster)

    assert (table.num_rows, table.num_cols) == (0, 0)
    assert table.table_cells == []
    (child,) = cluster.children
    assert child.label == DocItemLabel.TEXT
    assert [c.text for c in child.cells] == [c.text for c in _lab_cells()]
    error = _only_error(errors)
    assert error.page_no == 1
    assert "max_new_tokens limit (8)" in error.error_message
    assert "kept as plain text" in error.error_message


def test_hitting_the_budget_without_page_text_keeps_the_partial_structure():
    options = GraniteVisionTableStructureOptions(max_new_tokens=8)
    model = _model_with_mocked_inference(
        options, decoded_text=HEADER_ROW, generated_tokens=8
    )
    page, cluster = _mock_page()

    table, errors = _predict(model, page, cluster)

    assert table.num_rows == 1
    assert cluster.children == []
    error = _only_error(errors)
    assert "partial structure was kept" in error.error_message


def test_finishing_under_the_budget_is_not_reported(caplog):
    options = GraniteVisionTableStructureOptions(max_new_tokens=8)
    model = _model_with_mocked_inference(
        options, decoded_text=HEADER_ROW, generated_tokens=3
    )
    page, cluster = _mock_page()

    with caplog.at_level(logging.WARNING):
        table, errors = _predict(model, page, cluster)

    assert errors == []
    assert table.num_rows == 1
    assert not [r for r in caplog.records if "GraniteVision table" in r.message]


def test_transcription_that_lost_the_page_text_is_replaced_by_that_text():
    """A header-only grid parses fine but holds two of the eight words."""
    page, cluster = _page_with_text(_lab_cells())
    model = _model_with_mocked_inference(
        GraniteVisionTableStructureOptions(), decoded_text=HEADER_ROW
    )

    table, errors = _predict(model, page, cluster)

    assert (table.num_rows, table.num_cols) == (0, 0)
    (child,) = cluster.children
    assert [c.text for c in child.cells] == [c.text for c in _lab_cells()]
    error = _only_error(errors)
    assert "2 of the 8 words" in error.error_message
    assert "min_text_recall=0.5" in error.error_message


def test_transcription_that_matches_the_page_text_is_kept():
    page, cluster = _page_with_text(_lab_cells())
    model = _model_with_mocked_inference(
        GraniteVisionTableStructureOptions(),
        decoded_text=HEADER_ROW + DATA_ROW + MORE_ROWS,
    )

    table, errors = _predict(model, page, cluster)

    assert errors == []
    assert (table.num_rows, table.num_cols) == (4, 2)
    assert cluster.children == []


def test_page_text_check_ignores_ocr_cells():
    """OCR is a second reading of the pixels, not ground truth."""
    page, cluster = _page_with_text(_lab_cells(from_ocr=True))
    model = _model_with_mocked_inference(
        GraniteVisionTableStructureOptions(), decoded_text=HEADER_ROW
    )

    table, errors = _predict(model, page, cluster)

    assert errors == []
    assert table.num_rows == 1
    assert cluster.children == []


def test_page_text_check_can_be_disabled():
    page, cluster = _page_with_text(_lab_cells())
    model = _model_with_mocked_inference(
        GraniteVisionTableStructureOptions(min_text_recall=0),
        decoded_text=HEADER_ROW,
    )

    table, errors = _predict(model, page, cluster)

    assert errors == []
    assert table.num_rows == 1


def test_output_without_a_table_falls_back_to_the_page_text():
    page, cluster = _page_with_text(_lab_cells())
    model = _model_with_mocked_inference(
        GraniteVisionTableStructureOptions(),
        decoded_text="The image shows a laboratory report.",
    )

    table, errors = _predict(model, page, cluster)

    assert (table.num_rows, table.num_cols) == (0, 0)
    assert len(cluster.children) == 1
    error = _only_error(errors)
    assert "holds no table structure" in error.error_message


def test_text_recall_compares_words_not_spacing_or_punctuation():
    native = [
        _text_cell(0, TABLE_BBOX, "Hemoglobin 13.5 g/dL"),
        _text_cell(1, TABLE_BBOX, "12.0 - 16.0"),
    ]
    transcribed = [
        TableCell(
            text=t,
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=i,
            end_col_offset_idx=i + 1,
        )
        for i, t in enumerate(["HEMOGLOBIN", "13.5  G / DL", "12.0-16.0"])
    ]

    recall = native_text_recall(native, transcribed)

    assert recall is not None
    assert (recall.found, recall.total) == (9, 9)


def test_text_recall_needs_a_few_words_to_judge():
    assert native_text_recall([_text_cell(0, TABLE_BBOX, "Total 42")], []) is None


def test_cells_outside_a_nested_cluster_are_kept_as_well():
    """The layout stage nests the text clusters that sit inside the table
    box; the fallback must not stop at them."""
    cells = _lab_cells()
    page, cluster = _page_with_text(cells)
    nested = Cluster(
        id=7,
        label=DocItemLabel.TEXT,
        bbox=BoundingBox(l=110, t=110, r=390, b=130),
        cells=cells[:2],
    )
    cluster.children = [nested]
    options = GraniteVisionTableStructureOptions(max_new_tokens=8)
    model = _model_with_mocked_inference(
        options, decoded_text=HEADER_ROW, generated_tokens=8
    )

    table, errors = _predict(model, page, cluster)

    assert (table.num_rows, table.num_cols) == (0, 0)
    kept_nested, added = cluster.children
    assert kept_nested is nested
    assert added.id == 8
    assert [c.text for c in added.cells] == [c.text for c in cells[2:]]
    assert "kept as plain text" in _only_error(errors).error_message


def test_crop_uses_the_configured_scale():
    model = _model_with_mocked_inference(
        GraniteVisionTableStructureOptions(image_scale=2.0), decoded_text=HEADER_ROW
    )
    page, cluster = _mock_page()

    _predict(model, page, cluster)

    page.get_image.assert_called_once_with(scale=2.0, cropbox=cluster.bbox)
