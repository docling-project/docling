# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Falcon-OCR compatibility on the Transformers VLM engine (see #3278).

Falcon-OCR (``tiiuae/Falcon-OCR``) trips the generic Transformers path in
four independent ways: it does not support SDPA yet, so it must load with
eager attention on the actual config object; it ships no
``generation_config.json``; and it ships no usable chat template, so its
native ``generate`` API must be used instead of the chat-template processor
flow. These tests drive the load and predict paths with stand-in model
classes, so no model weights are downloaded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

import docling.models.inference_engines.vlm.transformers_engine as tf_engine
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options_vlm_model import TransformersModelType
from docling.datamodel.stage_model_specs import EngineModelConfig
from docling.datamodel.vlm_engine_options import TransformersVlmEngineOptions
from docling.models.inference_engines.vlm.base import VlmEngineInput
from docling.models.inference_engines.vlm.transformers_engine import (
    TransformersVlmEngine,
    _falcon_ocr_category_from_prompt,
)

_FALCON_REPO_ID = "tiiuae/Falcon-OCR"


class _FakeConfig:
    """Stand-in for the remote-code FalconOCRConfig."""

    def __init__(self) -> None:
        self._attn_implementation: str | None = None


class _FakeAutoConfig:
    """Records how the model config is preloaded."""

    last_kwargs: dict[str, Any] | None = None

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> _FakeConfig:
        cls.last_kwargs = dict(kwargs)
        return _FakeConfig()


class _FakeModel:
    """Stand-in for FalconOCRForCausalLM, with the native generate API."""

    instances: list[_FakeModel] = []
    load_kwargs: dict[str, Any] | None = None

    def __init__(self, config: Any) -> None:
        self.config = config
        self.generate_calls: list[dict[str, Any]] = []
        self.instances.append(self)

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> _FakeModel:
        cls.load_kwargs = dict(kwargs)
        return cls(kwargs.get("config") or _FakeConfig())

    def eval(self) -> _FakeModel:
        return self

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

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> _FakeProcessor:
        return cls()

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> str:
        raise AssertionError("Falcon-OCR must not go through the chat-template path")


class _FakeGenerationConfig:
    """Generation config that is missing from the Falcon repo."""

    from_pretrained_error: OSError | None = OSError(
        "tiiuae/Falcon-OCR does not appear to have a file named generation_config.json."
    )
    from_model_config_result: Any = "derived-generation-config"

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> Any:
        if cls.from_pretrained_error is not None:
            raise cls.from_pretrained_error
        return "pretrained-generation-config"

    @classmethod
    def from_model_config(cls, config: Any) -> Any:
        return cls.from_model_config_result


@pytest.fixture(autouse=True)
def _reset_fakes(monkeypatch: pytest.MonkeyPatch):
    _FakeModel.instances.clear()
    _FakeModel.load_kwargs = None
    _FakeAutoConfig.last_kwargs = None
    _FakeGenerationConfig.from_pretrained_error = OSError(
        "tiiuae/Falcon-OCR does not appear to have a file named generation_config.json."
    )
    monkeypatch.setattr(tf_engine, "AutoConfig", _FakeAutoConfig)
    monkeypatch.setattr(tf_engine, "AutoModelForCausalLM", _FakeModel)
    monkeypatch.setattr(tf_engine, "AutoProcessor", _FakeProcessor)
    monkeypatch.setattr(tf_engine, "GenerationConfig", _FakeGenerationConfig)


def _artifacts_path(tmp_path: Path) -> Path:
    artifacts = tmp_path / "artifacts"
    (artifacts / "tiiuae--Falcon-OCR").mkdir(parents=True)
    return artifacts


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


def _load_engine(
    tmp_path: Path, model_config: EngineModelConfig
) -> TransformersVlmEngine:
    return TransformersVlmEngine(
        options=TransformersVlmEngineOptions(
            trust_remote_code=True,
            compile_model=False,
        ),
        accelerator_options=AcceleratorOptions(device="cpu"),
        artifacts_path=_artifacts_path(tmp_path),
        model_config=model_config,
    )


def test_falcon_ocr_loads_with_eager_attention_and_preloaded_config(
    tmp_path: Path,
) -> None:
    """Eager attention lands on the config object and the load call."""
    engine = _load_engine(tmp_path, _model_config())

    assert _FakeModel.load_kwargs is not None
    assert "attn_implementation" in _FakeModel.load_kwargs
    assert "_attn_implementation" not in _FakeModel.load_kwargs
    assert _FakeModel.load_kwargs["attn_implementation"] == "eager"
    assert _FakeModel.load_kwargs["dtype"] == "bfloat16"

    config = _FakeModel.load_kwargs["config"]
    assert config is not None
    assert config._attn_implementation == "eager"
    assert _FakeAutoConfig.last_kwargs is not None
    assert _FakeAutoConfig.last_kwargs["trust_remote_code"] is True
    assert engine.vlm_model is not None


def test_explicit_attn_implementation_override_wins(tmp_path: Path) -> None:
    """An explicit public override beats the Falcon eager default."""
    _load_engine(tmp_path, _model_config(attn_implementation="sdpa"))

    assert _FakeModel.load_kwargs is not None
    assert _FakeModel.load_kwargs["attn_implementation"] == "sdpa"
    assert _FakeModel.load_kwargs["config"]._attn_implementation == "sdpa"


def test_legacy_private_attn_key_is_normalized(tmp_path: Path) -> None:
    """The legacy ``_attn_implementation`` extra-config key still works."""
    _load_engine(tmp_path, _model_config(_attn_implementation="sdpa"))

    assert _FakeModel.load_kwargs is not None
    assert _FakeModel.load_kwargs["attn_implementation"] == "sdpa"
    assert "_attn_implementation" not in _FakeModel.load_kwargs


def test_generation_config_falls_back_when_file_is_missing(
    tmp_path: Path,
) -> None:
    """A repo without generation_config.json derives it from the model."""
    engine = _load_engine(tmp_path, _model_config())

    assert engine.generation_config == "derived-generation-config"


def test_other_generation_config_errors_still_raise(tmp_path: Path) -> None:
    """Only the missing-file error falls back; other load errors surface."""
    _FakeGenerationConfig.from_pretrained_error = OSError("disk read failed")
    with pytest.raises(OSError, match="disk read failed"):
        _load_engine(tmp_path, _model_config())


@pytest.mark.parametrize(
    ("prompt", "category"),
    [
        ("", "plain"),
        ("Extract the text content from this image.", "plain"),
        ("Extract the table content from this image.", "table"),
        ("Extract the formula content from this image.", "formula"),
        ("Extract the title content from this image.", "title"),
        ("Extract the caption content from this image.", "caption"),
    ],
)
def test_falcon_ocr_category_from_prompt(prompt: str, category: str) -> None:
    assert _falcon_ocr_category_from_prompt(prompt) == category


def _predict_engine() -> TransformersVlmEngine:
    engine = TransformersVlmEngine(
        options=TransformersVlmEngineOptions(
            trust_remote_code=True,
            compile_model=False,
        ),
        accelerator_options=AcceleratorOptions(device="cpu"),
        artifacts_path=None,
    )
    engine._initialized = True
    engine.device = "cpu"
    engine.model_config = _model_config()
    engine.vlm_model = _FakeModel(_FakeConfig())  # type: ignore[assignment]
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
    assert len(model.generate_calls) == 1
    call = model.generate_calls[0]
    assert call["category"] == ["plain", "table"]
    assert call["compile"] is False
    assert call["max_new_tokens"] == 4096
    assert call["temperature"] == 0.0
    assert call["min_dimension"] == 64
    assert call["max_dimension"] == 1024


def test_non_falcon_predict_batch_still_requires_processor() -> None:
    """The native path is Falcon-only; other models keep the generic path."""
    engine = _predict_engine()
    engine.model_config = EngineModelConfig(repo_id="other/model")
    engine.processor = None

    with pytest.raises(RuntimeError, match="Processor not loaded"):
        engine.predict_batch(
            [VlmEngineInput(image=Image.new("RGB", (8, 8), "white"), prompt="")]
        )
