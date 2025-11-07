import logging
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from PIL.Image import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig

from docling.datamodel.accelerator_options import (
    AcceleratorOptions,
)
from docling.datamodel.base_models import VlmPrediction, VlmStopReason
from docling.datamodel.pipeline_options_vlm_model import InlineVlmOptions
from docling.models.base_model import BaseVlmModel
from docling.models.utils.hf_model_download import (
    HuggingFaceModelDownloadMixin,
)
from docling.utils.accelerator_utils import decide_device

_log = logging.getLogger(__name__)


# Source code from https://huggingface.co/numind/NuExtract-2.0-8B
def process_all_vision_info(messages, examples=None):
    """
    Process vision information from both messages and in-context examples, supporting batch processing.

    Args:
        messages: List of message dictionaries (single input) OR list of message lists (batch input)
        examples: Optional list of example dictionaries (single input) OR list of example lists (batch)

    Returns:
        A flat list of all images in the correct order:
        - For single input: example images followed by message images
        - For batch input: interleaved as (item1 examples, item1 input, item2 examples, item2 input, etc.)
        - Returns None if no images were found
    """
    try:
        from qwen_vl_utils import fetch_image, process_vision_info
    except ImportError:
        raise ImportError(
            "qwen-vl-utils is required for NuExtractTransformersModel. "
            "Please install it with: pip install qwen-vl-utils"
        )

    from qwen_vl_utils import fetch_image, process_vision_info

    # Helper function to extract images from examples
    def extract_example_images(example_item):
        if not example_item:
            return []

        # Handle both list of examples and single example
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

    # Normalize inputs to always be batched format
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

    # Ensure examples batch matches messages batch if provided
    if examples and len(examples_batch) != len(messages_batch):
        if not is_batch and len(examples_batch) == 1:
            # Single example set for a single input is fine
            pass
        else:
            raise ValueError("Examples batch length must match messages batch length")

    # Process all inputs, maintaining correct order
    all_images = []
    for i, message_group in enumerate(messages_batch):
        # Get example images for this input
        if examples and i < len(examples_batch):
            input_example_images = extract_example_images(examples_batch[i])
            all_images.extend(input_example_images)

        # Get message images for this input
        input_message_images = process_vision_info(message_group)[0] or []
        all_images.extend(input_message_images)

    return all_images if all_images else None


class NuExtractTransformersModel(BaseVlmModel, HuggingFaceModelDownloadMixin):
    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        accelerator_options: AcceleratorOptions,
        vlm_options: InlineVlmOptions,
    ):
        self.enabled = enabled
        self.vlm_options = vlm_options

        if self.enabled:
            import torch

            self.device = decide_device(
                accelerator_options.device,
                supported_devices=vlm_options.supported_devices,
            )
            _log.debug(f"Available device for NuExtract VLM: {self.device}")

            self.max_new_tokens = vlm_options.max_new_tokens
            self.temperature = vlm_options.temperature

            repo_cache_folder = vlm_options.repo_id.replace("/", "--")

            if artifacts_path is None:
                artifacts_path = self.download_models(
                    repo_id=self.vlm_options.repo_id,
                    revision=self.vlm_options.revision,
                )
            elif (artifacts_path / repo_cache_folder).exists():
                artifacts_path = artifacts_path / repo_cache_folder

            self.processor = AutoProcessor.from_pretrained(
                artifacts_path,
                trust_remote_code=vlm_options.trust_remote_code,
                use_fast=True,
            )
            self.processor.tokenizer.padding_side = "left"

            self.vlm_model = AutoModelForImageTextToText.from_pretrained(
                artifacts_path,
                device_map=self.device,
                dtype=self.vlm_options.torch_dtype,
                _attn_implementation=(
                    "flash_attention_2"
                    if self.device.startswith("cuda")
                    and accelerator_options.cuda_use_flash_attention2
                    else "sdpa"
                ),
                trust_remote_code=vlm_options.trust_remote_code,
            )
            if sys.version_info < (3, 14):
                self.vlm_model = torch.compile(self.vlm_model)  # type: ignore
            else:
                self.vlm_model.eval()

            # Load generation config
            self.generation_config = GenerationConfig.from_pretrained(artifacts_path)

    def process_images(
        self,
        image_batch: Iterable[Union[Image, np.ndarray]],
        prompt: Union[str, list[str]],
    ) -> Iterable[VlmPrediction]:
        """
        Batched inference for NuExtract VLM using the specialized input format.

        Args:
            image_batch: Iterable of PIL Images or numpy arrays
            prompt: Either:
                - str: Single template used for all images
                - list[str]: List of templates (one per image, must match image count)
        """
        import torch
        from PIL import Image as PILImage

        # Normalize images to RGB PIL
        pil_images: list[Image] = []
        for img in image_batch:
            if isinstance(img, np.ndarray):
                if img.ndim == 3 and img.shape[2] in (3, 4):
                    pil_img = PILImage.fromarray(img.astype(np.uint8))
                elif img.ndim == 2:
                    pil_img = PILImage.fromarray(img.astype(np.uint8), mode="L")
                else:
                    raise ValueError(f"Unsupported numpy array shape: {img.shape}")
            else:
                pil_img = img
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            pil_images.append(pil_img)

        if not pil_images:
            return

        # Normalize templates (1 per image)
        if isinstance(prompt, str):
            templates = [prompt] * len(pil_images)
        else:
            if len(prompt) != len(pil_images):
                raise ValueError(
                    f"Number of templates ({len(prompt)}) must match number of images ({len(pil_images)})"
                )
            templates = prompt

        # Construct NuExtract input format
        inputs = []
        for pil_img, template in zip(pil_images, templates):
            input_item = {
                "document": {"type": "image", "image": pil_img},
                "template": template,
            }
            inputs.append(input_item)

        # Create messages structure for batch processing
        messages = [
            [
                {
                    "role": "user",
                    "content": [x["document"]],
                }
            ]
            for x in inputs
        ]

        # Apply chat template to each example individually
        texts = [
            self.processor.tokenizer.apply_chat_template(
                messages[i],
                template=x["template"],
                tokenize=False,
                add_generation_prompt=True,
            )
            for i, x in enumerate(inputs)
        ]

        # Process vision inputs using qwen-vl-utils
        image_inputs = process_all_vision_info(messages)

        # Process with the processor
        processor_inputs = self.processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
            **self.vlm_options.extra_processor_kwargs,
        )
        processor_inputs = {k: v.to(self.device) for k, v in processor_inputs.items()}

        # Generate
        gen_kwargs = {
            **processor_inputs,
            "max_new_tokens": self.max_new_tokens,
            "generation_config": self.generation_config,
            **self.vlm_options.extra_generation_config,
        }
        if self.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature
        else:
            gen_kwargs["do_sample"] = False

        start_time = time.time()
        with torch.inference_mode():
            generated_ids = self.vlm_model.generate(**gen_kwargs)
        generation_time = time.time() - start_time

        # Trim generated sequences
        input_len = processor_inputs["input_ids"].shape[1]
        trimmed_sequences = generated_ids[:, input_len:]

        # Decode with the processor/tokenizer
        decoded_texts: list[str] = self.processor.batch_decode(
            trimmed_sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # Optional logging
        num_tokens = None
        if generated_ids.shape[0] > 0:  # type: ignore
            # Todo: confirm num tokens is actually from first item, code was already like this
            num_tokens = int(generated_ids[0].shape[0])
            _log.debug(
                f"Generated {num_tokens} tokens in {generation_time:.2f}s "
                f"for batch size {generated_ids.shape[0]}."  # type: ignore
            )

        for text in decoded_texts:
            # Apply decode_response to the output text
            decoded_text = self.vlm_options.decode_response(text)
            yield VlmPrediction(
                text=decoded_text,
                generation_time=generation_time,
                num_tokens=num_tokens,
                stop_reason=VlmStopReason.UNSPECIFIED,
            )
