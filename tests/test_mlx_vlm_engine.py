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
from docling.datamodel.vlm_engine_options import MlxVlmEngineOptions
from docling.models.inference_engines.vlm.base import VlmEngineInput
from docling.models.inference_engines.vlm.mlx_engine import MlxVlmEngine
from docling.models.utils.generation_utils import (
    GenerationStopper,
    TailRepetitionStopper,
)
from docling.models.vlm_pipeline_models.mlx_model import HuggingFaceMlxModel


@pytest.mark.parametrize(
    ("options", "input_config", "expected_repetition_kwargs"),
    [
        (MlxVlmEngineOptions(), {}, {}),
        (
            MlxVlmEngineOptions(),
            {"repetition_penalty": 1.1, "repetition_context_size": 0},
            {"repetition_penalty": 1.1, "repetition_context_size": 0},
        ),
        (
            MlxVlmEngineOptions(
                repetition_penalty=1.15,
                repetition_context_size=64,
            ),
            {"repetition_penalty": 1.1, "repetition_context_size": 0},
            {"repetition_penalty": 1.15, "repetition_context_size": 64},
        ),
    ],
)
def test_mlx_engine_forwards_optional_repetition_settings(
    options: MlxVlmEngineOptions,
    input_config: dict[str, Any],
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
                extra_generation_config=input_config,
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
