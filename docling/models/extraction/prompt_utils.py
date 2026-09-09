# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Prompt construction utilities for the extraction pipeline.

Each function takes a processor, images, and templates and returns
tokenized inputs ready for model.generate().
"""

from typing import Any

from PIL.Image import Image

from docling.datamodel.extraction import (
    ContentItem,
    ImageContentItem,
    TextContentItem,
)

# Re-exported: the schema-instruction wrapper now lives with the model spec in datamodel.
from docling.datamodel.extraction_options import _build_extraction_prompt

__all__ = [
    "_build_extraction_prompt",
    "build_granite_vision_inputs",
    "build_nuextract_content_inputs",
    "build_nuextract_inputs",
]


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


def build_nuextract_inputs(
    processor: Any,
    images: list[Image],
    templates: list[str],
    device: str,
    extra_processor_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Image-only adapter over :func:`build_nuextract_content_inputs`."""
    requests: list[list[ContentItem]] = [
        [ImageContentItem(image=img)] for img in images
    ]
    return build_nuextract_content_inputs(
        processor=processor,
        requests=requests,
        templates=templates,
        device=device,
        extra_processor_kwargs=extra_processor_kwargs,
    )


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
