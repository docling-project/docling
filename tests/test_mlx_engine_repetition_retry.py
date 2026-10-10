# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for the MLX engine retry of looping generations."""

from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image as PILImage

from docling.datamodel.stage_model_specs import VLM_CONVERT_GRANITE_DOCLING
from docling.datamodel.vlm_engine_options import MlxVlmEngineOptions
from docling.models.inference_engines.vlm.base import VlmEngineInput, VlmEngineType
from docling.models.inference_engines.vlm.mlx_engine import MlxVlmEngine
from docling.models.utils.generation_utils import (
    EmptyPictureRunStopper,
    GenerationStopper,
)

pytestmark = pytest.mark.ml_vlm

LOOP_PICTURES = "".join(
    f"<picture><loc_259><loc_{y}><loc_261><loc_{y + 5}></picture>\n"
    for y in range(223, 343, 10)
)
RETRIED_TEXT = "<doctag><text>after the figure</text></doctag>"


def _engine(
    calls: list[dict[str, Any]],
    chunked: bool = False,
    options: MlxVlmEngineOptions | None = None,
    loop_suffix: str = "",
) -> MlxVlmEngine:
    engine = MlxVlmEngine.__new__(MlxVlmEngine)
    engine._initialized = True
    engine.options = options or MlxVlmEngineOptions()
    engine.model_config = None
    engine.vlm_model = object()
    engine.processor = object()
    engine.config = object()
    engine.apply_chat_template = lambda processor, config, prompt, num_images: prompt

    def stream_generate(model, processor, prompt, images, **kwargs):
        calls.append(kwargs)
        text = (
            RETRIED_TEXT
            if kwargs.get("repetition_penalty") == 1.1
            else LOOP_PICTURES + loop_suffix
        )
        # The streaming detokenizer can hold text back until generation ends.
        pieces = ["", text] if chunked else list(text)
        for piece in pieces:
            yield SimpleNamespace(text=piece)

    engine.stream_generate = stream_generate
    return engine


def _input(extra_generation_config: dict[str, Any]) -> VlmEngineInput:
    return VlmEngineInput(
        image=PILImage.new("RGB", (8, 8), "white"),
        prompt="convert",
        temperature=0.0,
        max_new_tokens=512,
        stop_strings=["</doctag>"],
        extra_generation_config=extra_generation_config,
    )


def test_granite_mlx_retries_looping_page_with_repetition_penalty() -> None:
    calls: list[dict[str, Any]] = []
    config = VLM_CONVERT_GRANITE_DOCLING.model_spec.get_runtime_input_extra_config(
        VlmEngineType.MLX
    )

    output = _engine(calls).predict_batch([_input(config)])[0]

    assert [c.get("repetition_penalty") for c in calls] == [None, 1.1]
    assert output.text == RETRIED_TEXT
    assert output.stop_reason == "stop_string"


def test_granite_mlx_retries_loop_seen_only_in_finished_output() -> None:
    calls: list[dict[str, Any]] = []
    config = VLM_CONVERT_GRANITE_DOCLING.model_spec.get_runtime_input_extra_config(
        VlmEngineType.MLX
    )

    output = _engine(calls, chunked=True).predict_batch([_input(config)])[0]

    assert [c.get("repetition_penalty") for c in calls] == [None, 1.1]
    assert output.text == RETRIED_TEXT


def test_retry_penalty_replaces_engine_penalty_and_keeps_context_size() -> None:
    calls: list[dict[str, Any]] = []
    config = VLM_CONVERT_GRANITE_DOCLING.model_spec.get_runtime_input_extra_config(
        VlmEngineType.MLX
    )
    options = MlxVlmEngineOptions(repetition_penalty=1.03, repetition_context_size=64)

    _engine(calls, options=options).predict_batch([_input(config)])

    assert [
        (c.get("repetition_penalty"), c.get("repetition_context_size")) for c in calls
    ] == [(1.03, 64), (1.1, 64)]


def test_loop_ended_by_stop_string_in_one_piece_is_retried() -> None:
    calls: list[dict[str, Any]] = []
    config = VLM_CONVERT_GRANITE_DOCLING.model_spec.get_runtime_input_extra_config(
        VlmEngineType.MLX
    )

    output = _engine(calls, chunked=True, loop_suffix="</doctag>").predict_batch(
        [_input(config)]
    )[0]

    assert [c.get("repetition_penalty") for c in calls] == [None, 1.1]
    assert output.text == RETRIED_TEXT


def test_failing_stopper_is_skipped() -> None:
    calls: list[dict[str, Any]] = []

    class FailingStopper(GenerationStopper):
        def should_stop(self, s: str) -> bool:
            raise ValueError("boom")

    config = {"custom_stopping_criteria": [FailingStopper]}

    output = _engine(calls).predict_batch([_input(config)])[0]

    assert output.text == LOOP_PICTURES
    assert output.stop_reason == "unspecified"


def test_looping_page_without_retry_config_keeps_stopped_output() -> None:
    calls: list[dict[str, Any]] = []
    config = {"custom_stopping_criteria": [EmptyPictureRunStopper]}

    output = _engine(calls).predict_batch([_input(config)])[0]

    assert [c.get("repetition_penalty") for c in calls] == [None]
    assert output.stop_reason == "custom_criteria"
