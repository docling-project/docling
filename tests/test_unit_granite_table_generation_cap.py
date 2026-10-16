# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Granite Vision table generation must terminate on a pathological crop.

The granite-vision-4.1-4b tokenizer carries the transformers placeholder
``model_max_length`` (1e30), and the stage used to pass it as
``max_new_tokens``. A table crop that never emits end-of-text then generated
until the process was killed (issue #4657). These tests stand in for the
model call and check the budget, the loop detector, and the output cleanup.
"""

import logging
from unittest.mock import MagicMock

import torch
from docling_core.types.doc import BoundingBox, DocItemLabel
from PIL import Image
from transformers import StoppingCriteriaList

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Cluster
from docling.datamodel.pipeline_options import GraniteVisionTableStructureOptions
from docling.models.stages.table_structure.table_structure_model_granite_vision import (
    GraniteVisionTableStructureModel,
)
from docling.models.utils.hf_stopping_criteria import HFStoppingCriteriaWrapper

PROMPT_TOKENS = 5
PLACEHOLDER_MODEL_MAX_LENGTH = int(1e30)

HEADER_ROW = "<ched>Name</ched><ched>Value</ched><nl>"
DATA_ROW = "<fcel>Foo</fcel><fcel>42</fcel><nl>"
LOOP_ROW = "<fcel>x</fcel><fcel>x</fcel><nl>"


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


def _page_with_one_table() -> tuple[MagicMock, Cluster]:
    cluster = Cluster(
        id=0, label=DocItemLabel.TABLE, bbox=BoundingBox(l=0, t=0, r=10, b=10)
    )
    page = MagicMock()
    page.page_no = 0
    page._backend.is_valid.return_value = True
    page.predictions.layout.clusters = [cluster]
    page.predictions.tablestructure = None
    page.get_image.return_value = Image.new("RGB", (10, 10))
    return page, cluster


def _predicted_table(model: GraniteVisionTableStructureModel):
    page, cluster = _page_with_one_table()
    predictions = model.predict_tables(MagicMock(), [page])
    return predictions[0].table_map[cluster.id]


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


def test_stop_on_repetition_can_be_disabled():
    options = GraniteVisionTableStructureOptions(stop_on_repetition=False)
    looping = HEADER_ROW + DATA_ROW + LOOP_ROW * 60
    model = _model_with_mocked_inference(options, decoded_text=looping)

    table = _predicted_table(model)

    assert "stopping_criteria" not in model._model.generate.call_args.kwargs
    assert table.num_rows == 62


def test_repeated_rows_are_dropped_without_damaging_the_last_real_row():
    looping = HEADER_ROW + DATA_ROW + LOOP_ROW * 60
    model = _model_with_mocked_inference(
        GraniteVisionTableStructureOptions(), decoded_text=looping
    )

    table = _predicted_table(model)

    assert table.num_rows == 2
    assert table.num_cols == 2
    assert [c.text for c in table.table_cells] == ["Name", "Value", "Foo", "42"]


def test_partial_loop_row_from_a_mid_unit_stop_is_cut_at_the_row_break():
    looping = HEADER_ROW + DATA_ROW + LOOP_ROW * 60 + LOOP_ROW[:11]
    model = _model_with_mocked_inference(
        GraniteVisionTableStructureOptions(), decoded_text=looping
    )

    table = _predicted_table(model)

    assert table.num_rows == 2
    assert [c.text for c in table.table_cells] == ["Name", "Value", "Foo", "42"]


def test_short_runs_of_identical_rows_are_kept():
    """A real table may repeat rows; only a long run counts as a loop."""
    text = HEADER_ROW + DATA_ROW * 12
    model = _model_with_mocked_inference(
        GraniteVisionTableStructureOptions(), decoded_text=text
    )

    table = _predicted_table(model)

    assert table.num_rows == 13


def test_hitting_the_budget_is_logged(caplog):
    options = GraniteVisionTableStructureOptions(max_new_tokens=8)
    model = _model_with_mocked_inference(
        options, decoded_text=HEADER_ROW, generated_tokens=8
    )

    with caplog.at_level(logging.WARNING):
        table = _predicted_table(model)

    assert table.num_rows == 1
    assert any("max_new_tokens limit (8)" in r.message for r in caplog.records)


def test_finishing_under_the_budget_is_not_logged(caplog):
    options = GraniteVisionTableStructureOptions(max_new_tokens=8)
    model = _model_with_mocked_inference(
        options, decoded_text=HEADER_ROW, generated_tokens=3
    )

    with caplog.at_level(logging.WARNING):
        _predicted_table(model)

    assert not [r for r in caplog.records if "max_new_tokens" in r.message]
