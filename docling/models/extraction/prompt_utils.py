"""Prompt construction utilities for the extraction pipeline.

Each function takes a processor, images, and templates and returns
tokenized inputs ready for model.generate().
"""

from typing import Any

from PIL.Image import Image


def build_nuextract_inputs(
    processor: Any,
    images: list[Image],
    templates: list[str],
    device: str,
    extra_processor_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Build inputs using the NuExtract-specific template format.

    Requires qwen-vl-utils for vision processing.
    """
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        raise ImportError(
            "qwen-vl-utils is required for NuExtract extraction. "
            "Please install it with: pip install qwen-vl-utils"
        )

    inputs = []
    for pil_img, template in zip(images, templates):
        inputs.append(
            {
                "document": {"type": "image", "image": pil_img},
                "template": template,
            }
        )

    messages = [[{"role": "user", "content": [x["document"]]}] for x in inputs]

    texts = [
        processor.tokenizer.apply_chat_template(
            messages[i],
            template=x["template"],
            tokenize=False,
            add_generation_prompt=True,
        )
        for i, x in enumerate(inputs)
    ]

    image_inputs = _process_all_vision_info(messages)

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
    templates: list[str],
    device: str,
) -> dict[str, Any]:
    """Build inputs using standard chat conversation format with extraction prompt."""
    extraction_prompts = [_build_extraction_prompt(t) for t in templates]

    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": ep},
                ],
            }
        ]
        for ep in extraction_prompts
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


def _build_extraction_prompt(template: str) -> str:
    return (
        "Extract structured data from this document image.\n"
        "Return a JSON object matching this schema:\n\n"
        f"{template}\n\n"
        "Return null for fields you cannot find in the document.\n"
        "Return ONLY valid JSON, no other text."
    )


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
