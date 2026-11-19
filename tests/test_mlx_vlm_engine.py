# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import VlmStopReason
from docling.datamodel.pipeline_options_vlm_model import (
    InferenceFramework,
    InlineVlmOptions,
    ResponseFormat,
)
from docling.datamodel.stage_model_specs import VLM_CONVERT_GRANITE_DOCLING
from docling.datamodel.vlm_engine_options import MlxVlmEngineOptions
from docling.models.inference_engines.vlm.base import VlmEngineInput, VlmEngineType
from docling.models.inference_engines.vlm.mlx_engine import MlxVlmEngine
from docling.models.utils.generation_utils import (
    EmptyPictureRunStopper,
    GenerationStopper,
    TailRepetitionStopper,
)
from docling.models.vlm_pipeline_models.mlx_model import HuggingFaceMlxModel


@pytest.mark.parametrize(
    ("options", "expected_repetition_kwargs"),
    [
        (MlxVlmEngineOptions(), {}),
        (
            MlxVlmEngineOptions(
                repetition_penalty=1.15,
                repetition_context_size=64,
            ),
            {"repetition_penalty": 1.15, "repetition_context_size": 64},
        ),
    ],
)
def test_mlx_engine_forwards_optional_repetition_settings(
    options: MlxVlmEngineOptions,
    expected_repetition_kwargs: dict[str, Any],
) -> None:
    captured: dict[str, Any] = {}
    engine = MlxVlmEngine(options=options, artifacts_path=None)
    engine._initialized = True
    engine.vlm_model = object()
    engine.processor = object()
    engine.config = object()
    engine.apply_chat_template = lambda *args, **kwargs: "prompt"

    def stream_generate(*args, **kwargs):
        captured.update(kwargs)
        yield SimpleNamespace(text="ok")

    engine.stream_generate = stream_generate

    (output,) = engine.predict_batch(
        [
            VlmEngineInput(
                image=Image.new("RGB", (8, 8), "white"),
                prompt="Prompt",
                max_new_tokens=10,
                temperature=0.2,
            )
        ]
    )

    assert output.text == "ok"
    assert captured["max_tokens"] == 10
    assert captured["verbose"] is False
    assert captured["temperature"] == 0.2
    assert {
        key: value
        for key, value in captured.items()
        if key in {"repetition_penalty", "repetition_context_size"}
    } == expected_repetition_kwargs


class _ClosingTagStopper(GenerationStopper):
    """Stops when the model closes the document: a valid end of the output."""

    def should_stop(self, s: str) -> bool:
        return "</doctag>" in s

    def is_valid_end(self) -> bool:
        return True


@pytest.mark.parametrize(
    ("stopper", "tokens", "expected_text", "expected_stop_reason"),
    [
        pytest.param(
            _ClosingTagStopper(),
            ["<doctag>", "text", "</doctag>", "not read"],
            "<doctag>text</doctag>",
            VlmStopReason.END_OF_SEQUENCE,
            id="valid-end",
        ),
        pytest.param(
            TailRepetitionStopper(min_repeats=4, min_span=8, max_unit=4),
            ["<doctag>"] + ["ab"] * 30,
            "<doctag>abababab",
            VlmStopReason.STOP_SEQUENCE,
            id="repetition-cut",
        ),
    ],
)
def test_mlx_engine_reports_the_stop_reason_declared_by_the_stopper(
    stopper: GenerationStopper,
    tokens: list[str],
    expected_text: str,
    expected_stop_reason: VlmStopReason,
) -> None:
    engine = MlxVlmEngine(options=MlxVlmEngineOptions(), artifacts_path=None)
    engine._initialized = True
    engine.vlm_model = object()
    engine.processor = object()
    engine.config = object()
    engine.apply_chat_template = lambda *args, **kwargs: "prompt"
    engine.stream_generate = lambda *args, **kwargs: (
        SimpleNamespace(text=token) for token in tokens
    )

    (output,) = engine.predict_batch(
        [
            VlmEngineInput(
                image=Image.new("RGB", (8, 8), "white"),
                prompt="Prompt",
                extra_generation_config={"custom_stopping_criteria": [stopper]},
            )
        ]
    )

    assert output.text == expected_text
    assert output.stop_reason == expected_stop_reason.value


def _mlx_vlm_generation_kwargs() -> set[str]:
    types = pytest.importorskip(
        "mlx_vlm.generate.types", reason="mlx-vlm is only installed on Apple silicon"
    )
    return set(types.GenerateKwargs.__annotations__)


def test_mlx_engine_only_forwards_kwargs_known_to_mlx_vlm() -> None:
    # mlx-vlm accepts any keyword and silently drops the ones it does not know,
    # so a misspelled sampling option has no effect and raises no error.
    known_kwargs = _mlx_vlm_generation_kwargs()
    captured: dict[str, Any] = {}
    engine = MlxVlmEngine(
        options=MlxVlmEngineOptions(
            repetition_penalty=1.15, repetition_context_size=64
        ),
        artifacts_path=None,
    )
    engine._initialized = True
    engine.vlm_model = object()
    engine.processor = object()
    engine.config = object()
    engine.apply_chat_template = lambda *args, **kwargs: "prompt"

    def stream_generate(*args, **kwargs):
        captured.update(kwargs)
        yield SimpleNamespace(text="ok")

    engine.stream_generate = stream_generate

    engine.predict_batch(
        [
            VlmEngineInput(
                image=Image.new("RGB", (8, 8), "white"),
                prompt="Prompt",
                max_new_tokens=10,
                temperature=0.2,
            )
        ]
    )

    assert captured["temperature"] == 0.2
    assert set(captured) <= known_kwargs


def test_legacy_mlx_model_only_forwards_kwargs_known_to_mlx_vlm() -> None:
    known_kwargs = _mlx_vlm_generation_kwargs()
    captured: dict[str, Any] = {}
    model: Any = HuggingFaceMlxModel(
        enabled=False,
        artifacts_path=None,
        accelerator_options=AcceleratorOptions(),
        vlm_options=InlineVlmOptions(
            repo_id="test/model",
            prompt="Prompt",
            response_format=ResponseFormat.MARKDOWN,
            inference_framework=InferenceFramework.MLX,
            temperature=0.2,
            max_new_tokens=10,
        ),
    )
    model.vlm_model = object()
    model.processor = object()
    model.config = object()
    model.apply_chat_template = lambda *args, **kwargs: "prompt"

    def stream_generate(*args, **kwargs):
        captured.update(kwargs)
        yield SimpleNamespace(text="ok", token=0, logprobs=np.zeros(1))

    model.stream_generate = stream_generate

    (prediction,) = model.process_images([Image.new("RGB", (8, 8), "white")], "Prompt")

    assert prediction.text == "ok"
    assert captured["temperature"] == 0.2
    assert set(captured) <= known_kwargs


GRANITE_MLX_CONFIG = (
    VLM_CONVERT_GRANITE_DOCLING.model_spec.get_runtime_input_extra_config(
        VlmEngineType.MLX
    )
)
RETRY_PENALTY = GRANITE_MLX_CONFIG["retry_repetition_penalty"]
LOOP_PICTURES = "".join(
    f"<picture><loc_259><loc_{y}><loc_261><loc_{y + 5}></picture>\n"
    for y in range(223, 343, 10)
)
RETRIED_TEXT = "<doctag><text>after the figure</text></doctag>"
PAST_THE_END = "<text>past the end</text>"


def _engine(
    calls: list[dict[str, Any]],
    chunked: bool = False,
    options: MlxVlmEngineOptions | None = None,
    loop_suffix: str = "",
    retried_text: str = RETRIED_TEXT,
) -> MlxVlmEngine:
    engine = MlxVlmEngine(options=options or MlxVlmEngineOptions(), artifacts_path=None)
    engine._initialized = True
    engine.vlm_model = object()
    engine.processor = object()
    engine.config = object()
    engine.apply_chat_template = lambda *args, **kwargs: "prompt"

    def stream_generate(*args, **kwargs):
        calls.append(kwargs)
        text = (
            retried_text
            if kwargs.get("repetition_penalty") == RETRY_PENALTY
            else LOOP_PICTURES + loop_suffix
        )
        # The streaming detokenizer can hold text back until generation ends.
        pieces = ["", text] if chunked else list(text)
        for piece in pieces:
            yield SimpleNamespace(text=piece)

    engine.stream_generate = stream_generate
    return engine


def _input(
    extra_generation_config: dict[str, Any],
    stop_strings: list[str] | None = None,
) -> VlmEngineInput:
    return VlmEngineInput(
        image=Image.new("RGB", (8, 8), "white"),
        prompt="convert",
        temperature=0.0,
        max_new_tokens=512,
        stop_strings=["</doctag>"] if stop_strings is None else stop_strings,
        extra_generation_config=extra_generation_config,
    )


def test_granite_mlx_retries_looping_page_with_repetition_penalty() -> None:
    calls: list[dict[str, Any]] = []

    output = _engine(calls).predict_batch([_input(GRANITE_MLX_CONFIG)])[0]

    assert [c.get("repetition_penalty") for c in calls] == [None, RETRY_PENALTY]
    assert output.text == RETRIED_TEXT
    assert output.stop_reason == "stop_string"


def test_granite_mlx_retries_loop_seen_only_in_finished_output() -> None:
    calls: list[dict[str, Any]] = []

    output = _engine(calls, chunked=True).predict_batch([_input(GRANITE_MLX_CONFIG)])[0]

    assert [c.get("repetition_penalty") for c in calls] == [None, RETRY_PENALTY]
    assert output.text == RETRIED_TEXT
    # Two streamed pieces for the looping run and two for the retry.
    assert output.metadata["num_tokens"] == 4


def test_retry_penalty_replaces_engine_penalty_and_keeps_context_size() -> None:
    calls: list[dict[str, Any]] = []
    options = MlxVlmEngineOptions(repetition_penalty=1.03, repetition_context_size=64)

    _engine(calls, options=options).predict_batch([_input(GRANITE_MLX_CONFIG)])

    assert [
        (c.get("repetition_penalty"), c.get("repetition_context_size")) for c in calls
    ] == [(1.03, 64), (RETRY_PENALTY, 64)]


def test_loop_ended_by_stop_string_in_one_piece_is_retried() -> None:
    calls: list[dict[str, Any]] = []

    output = _engine(calls, chunked=True, loop_suffix="</doctag>").predict_batch(
        [_input(GRANITE_MLX_CONFIG)]
    )[0]

    assert [c.get("repetition_penalty") for c in calls] == [None, RETRY_PENALTY]
    assert output.text == RETRIED_TEXT


def test_retry_keeps_valid_end_stoppers() -> None:
    calls: list[dict[str, Any]] = []
    config = {**GRANITE_MLX_CONFIG, "custom_stopping_criteria": [_ClosingTagStopper]}

    output = _engine(calls, retried_text=RETRIED_TEXT + PAST_THE_END).predict_batch(
        [_input(config, stop_strings=[])]
    )[0]

    assert [c.get("repetition_penalty") for c in calls] == [None, RETRY_PENALTY]
    assert output.text == RETRIED_TEXT
    assert output.stop_reason == VlmStopReason.END_OF_SEQUENCE.value


class _FailingStopper(GenerationStopper):
    def should_stop(self, s: str) -> bool:
        raise ValueError("boom")


def test_failing_stopper_is_skipped() -> None:
    calls: list[dict[str, Any]] = []
    config = {"custom_stopping_criteria": [_FailingStopper]}

    output = _engine(calls).predict_batch([_input(config)])[0]

    assert [c.get("repetition_penalty") for c in calls] == [None]
    assert output.text == LOOP_PICTURES
    assert output.stop_reason == VlmStopReason.UNSPECIFIED.value


def test_failing_stopper_is_skipped_by_the_loop_check() -> None:
    # The stop string ends the run before the stoppers see the loop, so only the
    # check on the finished output finds it.
    calls: list[dict[str, Any]] = []
    config = {**GRANITE_MLX_CONFIG, "custom_stopping_criteria": [_FailingStopper]}

    output = _engine(calls, chunked=True, loop_suffix="</doctag>").predict_batch(
        [_input(config)]
    )[0]

    assert [c.get("repetition_penalty") for c in calls] == [None, RETRY_PENALTY]
    assert output.text == RETRIED_TEXT


def test_looping_page_without_retry_config_keeps_stopped_output() -> None:
    calls: list[dict[str, Any]] = []
    config = {"custom_stopping_criteria": [EmptyPictureRunStopper]}

    output = _engine(calls).predict_batch([_input(config)])[0]

    assert [c.get("repetition_penalty") for c in calls] == [None]
    assert output.stop_reason == VlmStopReason.STOP_SEQUENCE.value
