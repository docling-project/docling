# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Falcon-OCR compatibility on the Transformers VLM engine (see #3278).

Falcon-OCR (``tiiuae/Falcon-OCR``) trips the generic Transformers path in
three independent ways: it does not support SDPA yet, so it must load with
eager attention; it ships no ``generation_config.json``; and it ships no
usable chat template, so its native ``generate`` API must be used instead of
the chat-template processor flow. The load path itself is unchanged for
every model: only the ``_EAGER_ATTN_REQUIRED_REPO_IDS`` entry selects eager
attention for Falcon-OCR, and the native path is selected by the loaded
model's ``config.model_type``, not by its repo id.

A real conversion with the pinned revision is the remaining check for this
behavior; it needs the model weights and cannot run in a unit test.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

import docling.models.inference_engines.vlm.transformers_engine as tf_engine
from docling.datamodel.pipeline_options_vlm_model import TransformersModelType
from docling.datamodel.stage_model_specs import EngineModelConfig
from docling.datamodel.vlm_engine_options import TransformersVlmEngineOptions
from docling.models.inference_engines.vlm.base import VlmEngineInput
from docling.models.inference_engines.vlm.transformers_engine import (
    TransformersVlmEngine,
    _falcon_ocr_generate,
    _falcon_ocr_outputs,
)

_FALCON_REPO_ID = "tiiuae/Falcon-OCR"


class _FakeConfig:
    """Stand-in for a loaded model's config, carrying only ``model_type``."""

    def __init__(self, model_type: str) -> None:
        self.model_type = model_type


class _FakeModel:
    """Stand-in for FalconOCRForCausalLM, with the native generate API."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.generate_calls: list[dict[str, Any]] = []

    def generate(
        self,
        images: Any,
        *,
        category: str | list[str] = "plain",
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        top_k: int | None = None,
        min_dimension: int = 64,
        max_dimension: int = 1024,
        compile: bool = True,
        seed: int | None = 42,
    ) -> list[str]:
        self.generate_calls.append(
            {
                "images": images,
                "category": category,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "min_dimension": min_dimension,
                "max_dimension": max_dimension,
                "compile": compile,
                "seed": seed,
            }
        )
        return [f"ocr text {index}" for index in range(len(images))]


class _FakeProcessor:
    """Processor whose chat-template path must never be used for Falcon."""

    def __init__(self) -> None:
        self.padding_side = "right"

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> str:
        raise AssertionError("Falcon-OCR must not go through the chat-template path")


class _FakeGenerationConfig:
    """Generation config that is missing from the Falcon repo."""

    from_model_config_result: Any = "derived-generation-config"

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> Any:
        return "pretrained-generation-config"

    @classmethod
    def from_model_config(cls, config: Any) -> Any:
        return cls.from_model_config_result


def _model_config(**extra_config: Any) -> EngineModelConfig:
    return EngineModelConfig(
        repo_id=_FALCON_REPO_ID,
        revision="main",
        torch_dtype="bfloat16",
        extra_config={
            "transformers_model_type": TransformersModelType.AUTOMODEL_CAUSALLM,
            **extra_config,
        },
    )


def _predict_engine(model_type: str = "falcon_ocr") -> TransformersVlmEngine:
    engine = TransformersVlmEngine.__new__(TransformersVlmEngine)
    engine._initialized = True
    engine.options = TransformersVlmEngineOptions(
        trust_remote_code=True,
        compile_model=False,
    )
    engine.device = "cpu"
    # Tests that build engines with __new__ set the attributes predict_batch
    # reads, including the loaded model's model_type.
    engine.model_type = model_type
    engine.model_config = _model_config()
    engine.vlm_model = _FakeModel(_FakeConfig(model_type))  # type: ignore[assignment]
    engine.processor = _FakeProcessor()
    return engine


def test_falcon_predict_batch_uses_native_generate() -> None:
    """Inference bypasses the chat-template flow for Falcon's generate API."""
    engine = _predict_engine()
    model = engine.vlm_model
    assert isinstance(model, _FakeModel)
    image = Image.new("RGB", (8, 8), "white")

    outputs = engine.predict_batch(
        [
            VlmEngineInput(image=image, prompt=""),
            VlmEngineInput(
                image=image,
                prompt="Extract the table content from this image.",
            ),
        ]
    )

    assert [output.text for output in outputs] == ["ocr text 0", "ocr text 1"]
    assert [output.stop_reason for output in outputs] == [
        "unspecified",
        "unspecified",
    ]
    assert len(model.generate_calls) == 1
    call = model.generate_calls[0]
    # The category comes from the preset (default "plain"); the prompt text
    # must not select a category, so "table" in the prompt still reads plain.
    assert call["category"] == "plain"
    assert call["compile"] is False
    assert call["max_new_tokens"] == 4096
    assert call["temperature"] == 0.0
    assert call["min_dimension"] == 64
    assert call["max_dimension"] == 1024


def test_falcon_category_comes_from_preset() -> None:
    """The preset supplies the OCR category as data for the native path."""
    engine = _predict_engine()
    engine.model_config = _model_config(falcon_ocr_category="table")
    model = engine.vlm_model
    assert isinstance(model, _FakeModel)
    image = Image.new("RGB", (8, 8), "white")

    engine.predict_batch([VlmEngineInput(image=image, prompt="")])

    assert len(model.generate_calls) == 1
    assert model.generate_calls[0]["category"] == "table"


def test_non_falcon_model_type_uses_generic_path() -> None:
    """A model with another model_type takes the generic path, unchanged."""
    # The repo id stays Falcon-OCR: selection is by model_type, not repo id.
    engine = _predict_engine(model_type="qwen2_vl")
    engine.processor = None

    with pytest.raises(RuntimeError, match="Processor not loaded"):
        engine.predict_batch(
            [VlmEngineInput(image=Image.new("RGB", (8, 8), "white"), prompt="")]
        )


def test_falcon_generate_passes_compile_model_through() -> None:
    """The engine's compile_model option reaches Falcon's generate()."""
    model = _FakeModel(_FakeConfig("falcon_ocr"))
    image = Image.new("RGB", (8, 8), "white")

    _falcon_ocr_generate(model, [image], compile_model=True)

    assert len(model.generate_calls) == 1
    assert model.generate_calls[0]["compile"] is True


def test_generation_config_falls_back_when_file_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repo without generation_config.json derives it from the model."""
    monkeypatch.setattr(tf_engine, "GenerationConfig", _FakeGenerationConfig)
    engine = TransformersVlmEngine.__new__(TransformersVlmEngine)
    engine.vlm_model = SimpleNamespace(config=_FakeConfig("falcon_ocr"))
    engine.model_config = _model_config()

    assert (
        engine._load_generation_config(artifacts_path=tmp_path, revision="main")
        == "derived-generation-config"
    )


def test_generation_config_loads_when_file_is_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repo that ships generation_config.json loads it as before."""
    (tmp_path / "generation_config.json").write_text("{}")
    monkeypatch.setattr(tf_engine, "GenerationConfig", _FakeGenerationConfig)
    engine = TransformersVlmEngine.__new__(TransformersVlmEngine)
    engine.vlm_model = SimpleNamespace(config=_FakeConfig("falcon_ocr"))
    engine.model_config = _model_config()

    assert (
        engine._load_generation_config(artifacts_path=tmp_path, revision="main")
        == "pretrained-generation-config"
    )


def test_falcon_outputs_require_one_text_per_input() -> None:
    """A short or long result never attaches text to the wrong image."""
    with pytest.raises(RuntimeError, match="1 texts for 2 images"):
        _falcon_ocr_outputs(["only one"], 2)


def test_falcon_outputs_use_unspecified_stop_reason() -> None:
    """The native call reports no token counts, so the stop reason is unknown."""
    outputs = _falcon_ocr_outputs(["ocr text"], 1)

    assert outputs[0].text == "ocr text"
    assert outputs[0].stop_reason == "unspecified"
