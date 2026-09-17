# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Prompt construction utilities for the extraction pipeline.

Each function takes a processor, images, and templates and returns
tokenized inputs ready for model.generate().
"""

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from PIL.Image import Image

from docling.datamodel.extraction import (
    ContentItem,
    ExtractionTarget,
    ExtractionTemplateType,
    ImageContentItem,
    TextContentItem,
)

# Re-exported: the schema-instruction wrapper now lives with the model spec in datamodel.
from docling.datamodel.extraction_options import (
    ExtractionPromptStyle,
    ExtractionVlmModelSpec,
    _build_extraction_prompt,
)
from docling.models.extraction.template_utils import (
    _schema_to_nuextract,
    normalize_target,
    schema_validator,
)

if TYPE_CHECKING:
    from jsonschema.protocols import Validator

__all__ = [
    "_build_extraction_prompt",
    "build_granite_vision_inputs",
    "build_nuextract_content_inputs",
]


@dataclass(frozen=True)
class _PreparedTarget:
    """Call-owned guidance and validator, ready for either inference adapter."""

    target: ExtractionTarget | None
    validator: "Validator | None"
    prompt: str
    chat_template_kwargs: dict[str, Any]
    processor_kwargs: dict[str, Any]
    constraint_schema: dict[str, Any] | None = None


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
    owned = normalize_target(target)
    schema = owned.output_schema
    validator = schema_validator(schema) if schema is not None else None
    chat_kwargs = deepcopy(model_spec.extra_chat_template_kwargs)
    for key in (
        "template",
        "instructions",
        "messages",
        "response_format",
        "structured_outputs",
    ):
        if key in chat_kwargs:
            raise ValueError(f"chat-template option {key!r} is request-owned")
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


def prepare_legacy_target(
    template: ExtractionTemplateType, model_spec: ExtractionVlmModelSpec
) -> _PreparedTarget:
    """Keep main's sample serialization and prompts separate from explicit targets."""
    prompt = model_spec.build_extraction_prompt(template)
    chat_kwargs = deepcopy(model_spec.extra_chat_template_kwargs)
    if model_spec.prompt_style is ExtractionPromptStyle.NUEXTRACT:
        chat_kwargs["template"] = prompt
        prompt = ""
    return _PreparedTarget(
        None, None, prompt, chat_kwargs, deepcopy(model_spec.extra_processor_kwargs)
    )


def _content_item_to_nuextract(item: ContentItem) -> dict[str, Any]:
    """Map a ContentItem to NuExtract's native content dict."""
    if isinstance(item, TextContentItem):
        return {"type": "text", "text": item.text}
    if isinstance(item, ImageContentItem):
        return {"type": "image", "image": item.image}
    raise ValueError(f"Unsupported content item: {type(item)}")


def build_nuextract_content_inputs(
    processor: Any,
    requests: list[list[ContentItem]],
    templates: list[str],
    device: str,
    extra_processor_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Build NuExtract inputs from ordered content-item requests.

    Each request is a ``list[ContentItem]`` (image and/or text). The template
    rides the model's own ``template=`` chat kwarg, not the content. Requires
    qwen-vl-utils only when an image is present.
    """
    messages = [
        [{"role": "user", "content": [_content_item_to_nuextract(i) for i in req]}]
        for req in requests
    ]

    texts = [
        processor.tokenizer.apply_chat_template(
            messages[idx],
            template=template,
            tokenize=False,
            add_generation_prompt=True,
        )
        for idx, template in enumerate(templates)
    ]

    has_image = any(isinstance(i, ImageContentItem) for req in requests for i in req)
    image_inputs = _process_all_vision_info(messages) if has_image else None

    processor_inputs = processor(
        text=texts,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
        **extra_processor_kwargs,
    )
    return {k: v.to(device) for k, v in processor_inputs.items()}


def build_granite_vision_inputs(
    processor: Any,
    images: list[Image],
    prompts: list[str],
    device: str,
) -> dict[str, Any]:
    """Build inputs using standard chat conversation format.

    ``prompts`` are the final, ready-to-send prompt strings. The schema-instruction
    wrapper (:func:`_build_extraction_prompt`) is applied upstream in
    the extraction pipeline so that every engine (transformers/api/vllm) shares
    one prompt-construction path; do not wrap again here.
    """
    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        for prompt in prompts
    ]
    texts = [
        processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        for conv in conversations
    ]
    processor_inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        do_pad=True,
    )
    return {k: v.to(device) for k, v in processor_inputs.items()}


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
