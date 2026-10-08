# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from docling.datamodel.vlm_engine_options import MlxVlmEngineOptions
from docling.models.inference_engines.vlm.base import VlmEngineInput
from docling.models.inference_engines.vlm.mlx_engine import MlxVlmEngine


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
    assert captured["temp"] == 0.2
    assert {
        key: value
        for key, value in captured.items()
        if key in {"repetition_penalty", "repetition_context_size"}
    } == expected_repetition_kwargs
