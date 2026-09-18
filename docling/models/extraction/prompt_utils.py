# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Prompt construction utilities for the extraction pipeline.

Call-owned target preparation and ordered-content local preprocessing.
"""

import json
import warnings
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from docling.datamodel.extraction import (
    ContentItem,
    ExtractionTarget,
    ExtractionTemplateType,
    ImageContentItem,
    TextContentItem,
)
from docling.datamodel.extraction_options import (
    ExtractionPromptStyle,
    ExtractionVlmModelSpec,
    _merge_chat_options,
    _reject_request_fields,
)
from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions
from docling.models.extraction.template_utils import (
    _schema_to_nuextract,
    _vllm_constraint_schema,
    normalize_target,
    schema_validator,
)
from docling.models.inference_engines.vlm.base import VlmEngineType

if TYPE_CHECKING:
    from jsonschema.protocols import Validator


@dataclass(frozen=True)
class _PreparedTarget:
    """Call-owned guidance and validator, ready for either inference adapter."""

    target: ExtractionTarget | None
    validator: "Validator | None"
    prompt: str
    chat_template_kwargs: dict[str, Any]
    processor_kwargs: dict[str, Any]
    constraint_schema: dict[str, Any] | None = None
    request_timeout: float | None = None


_OBJECT_INSTRUCTIONS = "Return ONLY a valid JSON object, with no other text."
_MISSING_INSTRUCTIONS = (
    "For unavailable information, use null only where the schema permits null. "
    "Optional properties may be omitted. Never invent a required non-nullable "
    "value; an unavailable required value makes the output invalid."
)


def prepare_target(
    target: ExtractionTarget, model_spec: ExtractionVlmModelSpec
) -> _PreparedTarget:
    """Normalize once and prepare guidance without invoking an engine or model."""
    return _prepare_normalized_target(normalize_target(target), model_spec)


def _prepare_normalized_target(
    owned: ExtractionTarget, model_spec: ExtractionVlmModelSpec
) -> _PreparedTarget:
    schema = owned.output_schema
    if model_spec.requires_output_schema and schema is None:
        raise ValueError(f"{model_spec.name} extraction requires an output schema")
    validator = schema_validator(schema) if schema is not None else None
    chat_kwargs = _merge_chat_options(model_spec.extra_chat_template_kwargs)
    processor_kwargs = deepcopy(model_spec.extra_processor_kwargs)
    guidance = [_OBJECT_INSTRUCTIONS]
    if model_spec.prompt:
        guidance.append(model_spec.prompt)
    if owned.instructions:
        guidance.append(owned.instructions)
    if schema is not None:
        guidance.extend(
            [
                _MISSING_INSTRUCTIONS,
                "Output contract (JSON Schema Draft 2020-12):\n"
                + json.dumps(schema, indent=2),
            ]
        )

    template = owned.template
    if model_spec.preparation == "nuextract":
        if chat_kwargs.get("mode", "structured") != "structured":
            raise ValueError("NuExtract extraction requires structured mode")
        if template is not None:
            if template.format != "nuextract":
                raise ValueError(
                    "NuExtract requires a nuextract native template, not example_json"
                )
            native = template.value
        else:
            assert schema is not None  # ExtractionTarget requires schema or template.
            native = _schema_to_nuextract(schema)
        chat_kwargs["template"] = json.dumps(native, indent=2)
        chat_kwargs["instructions"] = "\n\n".join(guidance)
        prompt = ""
    else:
        if template is not None:
            if template.format != "example_json":
                raise ValueError(
                    "generic chat does not support the nuextract native dialect"
                )
            guidance.append(
                "Example output (illustration only, not a schema or source facts):\n"
                + json.dumps(template.value, indent=2)
            )
        prompt = "\n\n".join(guidance)

    return _PreparedTarget(owned, validator, prompt, chat_kwargs, processor_kwargs)


def normalize_extraction_call(
    template: ExtractionTemplateType | None, target: ExtractionTarget | None
) -> ExtractionTarget | str:
    if (template is None) == (target is None):
        raise ValueError("Provide exactly one of target= or template=")
    if target is not None:
        return normalize_target(target)
    warnings.warn(
        "template= extraction is deprecated; use an explicit ExtractionTarget with target=",
        DeprecationWarning,
        stacklevel=3,
    )
    from docling.datamodel.extraction_options import NUEXTRACT_2B_SPEC

    assert template is not None
    return NUEXTRACT_2B_SPEC.serialize_template(template)


def prepare_legacy_target(
    template: ExtractionTemplateType, model_spec: ExtractionVlmModelSpec
) -> _PreparedTarget:
    """Keep main's sample serialization and prompts separate from explicit targets."""
    if model_spec.requires_output_schema:
        raise ValueError(f"{model_spec.name} requires target= with an output schema")
    prompt = model_spec.build_extraction_prompt(template)
    chat_kwargs = _merge_chat_options(model_spec.extra_chat_template_kwargs)
    if model_spec.prompt_style is ExtractionPromptStyle.NUEXTRACT:
        chat_kwargs["template"] = prompt
        prompt = ""
    return _PreparedTarget(
        None, None, prompt, chat_kwargs, deepcopy(model_spec.extra_processor_kwargs)
    )


def prepare_output_target(
    target: _PreparedTarget, output_mode: str, engine_type: VlmEngineType
) -> _PreparedTarget:
    """Attach an owned, bounded decoder schema only for the explicit vLLM contract."""
    if output_mode == "prompt_only":
        return replace(target, constraint_schema=None)
    if output_mode != "schema_constrained" or engine_type != VlmEngineType.API:
        raise ValueError(
            "schema_constrained requires the explicitly configured vLLM API engine"
        )
    if target.target is None or target.target.output_schema is None:
        raise ValueError("schema_constrained requires an output schema")
    return replace(
        target, constraint_schema=_vllm_constraint_schema(target.target.output_schema)
    )


def prepare_api_request_options(
    target: _PreparedTarget,
    model_spec: ExtractionVlmModelSpec,
    engine_options: ApiVlmEngineOptions,
    params: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Preflight API transport options without constructing a model or engine."""
    model_params = model_spec.get_api_params(engine_options.engine_type)
    engine_params = engine_options.params
    for mapping in (model_params, engine_params):
        _reject_request_fields(mapping)
    dynamic = {
        key: value
        for key, value in target.chat_template_kwargs.items()
        if key in {"template", "instructions"}
    }
    defaults = {
        key: value
        for key, value in target.chat_template_kwargs.items()
        if key not in dynamic
    }
    chat = _merge_chat_options(
        defaults,
        model_params.get("chat_template_kwargs", {}),
        engine_params.get("chat_template_kwargs", {}),
    )
    if (
        model_spec.preparation == "nuextract"
        and chat.get("mode", "structured") != "structured"
    ):
        raise ValueError("NuExtract extraction requires structured mode")
    chat.update(deepcopy(dynamic))
    if chat and engine_options.engine_type != VlmEngineType.API:
        raise ValueError(
            "chat_template_kwargs require the explicitly configured vLLM API engine"
        )
    owned_params = deepcopy(
        params
        if params is not None
        else {
            "temperature": model_spec.temperature,
            "max_tokens": model_spec.max_new_tokens,
            **model_params,
            **engine_params,
        }
    )
    owned_params.pop("chat_template_kwargs", None)
    return owned_params, chat


def build_content_messages(
    content: list[ContentItem], prompt: str
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, TextContentItem):
            items.append({"type": "text", "text": item.text})
        elif isinstance(item, ImageContentItem):
            items.append({"type": "image", "image": item.image})
        else:
            raise ValueError(f"Unsupported content item: {type(item)}")
    if prompt:
        items.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": items}]


def prepared_image_prompt(
    prompt: str, model_spec: ExtractionVlmModelSpec
) -> _PreparedTarget:
    """Main's image wrapper receives final prompts; never render the schema wrapper twice."""
    if model_spec.requires_output_schema:
        raise ValueError(f"{model_spec.name} requires target= with an output schema")
    chat = _merge_chat_options(model_spec.extra_chat_template_kwargs)
    if model_spec.preparation == "nuextract":
        chat["template"] = prompt
        prompt = ""
    return _PreparedTarget(
        None, None, prompt, chat, deepcopy(model_spec.extra_processor_kwargs)
    )


def build_content_inputs(
    processor: Any,
    requests: list[list[ContentItem]],
    targets: list[_PreparedTarget],
    model_spec: ExtractionVlmModelSpec,
    device: str,
) -> dict[str, Any]:
    processor_kwargs = deepcopy(targets[0].processor_kwargs)
    _reject_request_fields(processor_kwargs)
    _reject_request_fields(
        processor_kwargs, reserved={"text", "images", "padding", "return_tensors"}
    )
    messages = [
        build_content_messages(req, target.prompt)
        for req, target in zip(requests, targets)
    ]
    texts = []
    for conversation, target in zip(messages, targets):
        renderer = (
            processor.tokenizer
            if model_spec.local_preprocessing == "tokenizer_qwen"
            else processor
        )
        texts.append(
            renderer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
                **deepcopy(target.chat_template_kwargs),
            )
        )
    images = [
        item.image
        for req in requests
        for item in req
        if isinstance(item, ImageContentItem)
    ]
    # NuExtract 2 uses legacy Qwen preprocessing; newer processors own image sizing.
    image_inputs = (
        _process_all_vision_info(messages)
        if images and model_spec.local_preprocessing == "tokenizer_qwen"
        else images or None
    )
    inputs = processor(
        text=texts,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
        **processor_kwargs,
    )
    return {key: value.to(device) for key, value in inputs.items()}


def _process_all_vision_info(messages: list, examples: list | None = None) -> Any:
    """Process vision info from messages using qwen-vl-utils.

    Adapted from NuExtract source code.
    """
    from qwen_vl_utils import fetch_image, process_vision_info

    def extract_example_images(example_item: Any) -> list:
        if not example_item:
            return []
        examples_to_process = (
            example_item if isinstance(example_item, list) else [example_item]
        )
        images = []
        for example in examples_to_process:
            if (
                isinstance(example.get("input"), dict)
                and example["input"].get("type") == "image"
            ):
                images.append(fetch_image(example["input"]))
        return images

    is_batch = messages and isinstance(messages[0], list)
    messages_batch = messages if is_batch else [messages]
    is_batch_examples = (
        examples
        and isinstance(examples, list)
        and (isinstance(examples[0], list) or examples[0] is None)
    )
    examples_batch = (
        examples
        if is_batch_examples
        else ([examples] if examples is not None else None)
    )

    if examples and examples_batch is not None:
        if len(examples_batch) != len(messages_batch):
            if not is_batch and len(examples_batch) == 1:
                pass
            else:
                raise ValueError(
                    "Examples batch length must match messages batch length"
                )

    all_images = []
    for i, message_group in enumerate(messages_batch):
        if examples and examples_batch is not None and i < len(examples_batch):
            all_images.extend(extract_example_images(examples_batch[i]))
        input_message_images = process_vision_info(message_group)[0] or []
        all_images.extend(input_message_images)

    return all_images if all_images else None
