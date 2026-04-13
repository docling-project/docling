"""Runtime adapters for Transformers-backed VLM models."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from transformers import AutoConfig, PreTrainedModel

from docling.models.inference_engines.vlm.base import VlmEngineInput, VlmEngineOutput

if TYPE_CHECKING:
    from docling.datamodel.stage_model_specs import EngineModelConfig
    from docling.datamodel.vlm_engine_options import TransformersVlmEngineOptions

_FALCON_OCR_DEFAULT_PROMPT = "Extract the text content from this image."
_FALCON_OCR_CATEGORY_BY_PROMPT_SUBSTRING = (
    ("formula", "formula"),
    ("table", "table"),
    ("caption", "caption"),
    ("footnote", "footnote"),
    ("list-item", "list-item"),
    ("page-footer", "page-footer"),
    ("page-header", "page-header"),
    ("section-header", "section-header"),
    ("title", "title"),
)


class TransformersRuntimeAdapter(Protocol):
    """Protocol for model-specific Transformers runtime extensions."""

    @staticmethod
    def build_model_config(
        *,
        artifacts_path: Path,
        revision: str,
        options: TransformersVlmEngineOptions,
        model_config: EngineModelConfig | None,
        attn_implementation: str,
    ) -> Any | None:
        """Return a config object to pass to ``from_pretrained()``."""

    @staticmethod
    def predict_batch(
        *,
        model: PreTrainedModel,
        input_batch: list[VlmEngineInput],
    ) -> list[VlmEngineOutput]:
        """Run model-specific batch inference."""


def falcon_ocr_build_prompt(prompt: str, *, page: Any | None = None) -> str:
    """Normalize Falcon-OCR prompts while still allowing raw user overrides."""
    del page

    normalized_prompt = prompt.strip() or _FALCON_OCR_DEFAULT_PROMPT
    if "<|image|>" not in normalized_prompt:
        normalized_prompt = f"<|image|>{normalized_prompt}"
    if "<|OCR_PLAIN|>" not in normalized_prompt:
        normalized_prompt = f"{normalized_prompt.rstrip()}\n<|OCR_PLAIN|>"

    return normalized_prompt


def _falcon_ocr_category_from_prompt(prompt: str) -> str:
    normalized_prompt = prompt.lower()
    for prompt_substring, category in _FALCON_OCR_CATEGORY_BY_PROMPT_SUBSTRING:
        if prompt_substring in normalized_prompt:
            return category
    return "plain"


def _falcon_ocr_generation_kwargs(first_input: VlmEngineInput) -> dict[str, Any]:
    extra_generation_config = dict(first_input.extra_generation_config or {})
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": first_input.max_new_tokens,
        "temperature": first_input.temperature,
        "top_k": extra_generation_config.get("top_k"),
        "min_dimension": extra_generation_config.get("min_dimension", 64),
        "max_dimension": extra_generation_config.get("max_dimension", 1024),
    }
    if "seed" in extra_generation_config:
        generation_kwargs["seed"] = extra_generation_config["seed"]

    return generation_kwargs


class FalconOCRTransformersAdapter:
    """Adapter for Falcon-OCR remote-code checkpoints.

    Falcon-OCR does not follow the standard ``processor(text=..., images=...)``
    path. Different remote-code revisions have exposed either a private
    ``_generate_batch(...)`` helper or a public ``generate(images, category=...)``
    method, so the adapter probes both variants here instead of leaking those
    details into the generic engine.
    """

    @staticmethod
    def build_model_config(
        *,
        artifacts_path: Path,
        revision: str,
        options: TransformersVlmEngineOptions,
        model_config: EngineModelConfig | None,
        attn_implementation: str,
    ) -> Any | None:
        del model_config

        config = AutoConfig.from_pretrained(
            artifacts_path,
            trust_remote_code=options.trust_remote_code,
            revision=revision,
            attn_implementation=attn_implementation,
        )
        config._attn_implementation = attn_implementation
        config._attn_implementation_internal = attn_implementation

        return config

    @staticmethod
    def predict_batch(
        *,
        model: PreTrainedModel,
        input_batch: list[VlmEngineInput],
    ) -> list[VlmEngineOutput]:
        if not input_batch:
            return []

        first_input = input_batch[0]
        generation_kwargs = _falcon_ocr_generation_kwargs(first_input)
        metadata = {"transformers_runtime_adapter": "falcon_ocr"}

        # Falcon remote-code revisions have shipped either a private
        # ``_generate_batch`` helper or only the public ``generate`` API.
        # Probe both shapes here so the generic engine can stay model-agnostic.
        generate_batch = getattr(model, "_generate_batch", None)
        if callable(generate_batch):
            image_prompt_pairs = [
                (input_data.image, input_data.prompt) for input_data in input_batch
            ]
            generated_texts = generate_batch(
                image_prompt_pairs,
                **generation_kwargs,
            )
        else:
            generate_kwargs = {
                "category": [
                    _falcon_ocr_category_from_prompt(input_data.prompt)
                    for input_data in input_batch
                ],
                **generation_kwargs,
            }

            if "compile" in inspect.signature(model.generate).parameters:
                generate_kwargs["compile"] = False

            generated_texts = model.generate(
                [input_data.image for input_data in input_batch],
                **generate_kwargs,
            )

        return [
            VlmEngineOutput(text=text, metadata=dict(metadata))
            for text in generated_texts
        ]
