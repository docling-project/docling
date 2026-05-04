import logging
import sys
import time
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional, Union, cast

import numpy as np
import torch
from PIL.Image import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import VlmPrediction, VlmStopReason
from docling.datamodel.extraction_options import ExtractionPromptStyle
from docling.datamodel.pipeline_options_vlm_model import InlineVlmOptions
from docling.models.base_model import BaseVlmModel
from docling.models.extraction.prompt_utils import (
    build_granite_vision_inputs,
    build_nuextract_inputs,
)
from docling.models.utils.hf_model_download import HuggingFaceModelDownloadMixin
from docling.utils.accelerator_utils import decide_device

_log = logging.getLogger(__name__)


class TransformersExtractionModel(BaseVlmModel, HuggingFaceModelDownloadMixin):
    """Unified extraction model supporting multiple prompt styles."""

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        accelerator_options: AcceleratorOptions,
        vlm_options: InlineVlmOptions,
        prompt_style: ExtractionPromptStyle = ExtractionPromptStyle.NUEXTRACT,
    ):
        self.enabled = enabled
        self.vlm_options = vlm_options
        self.prompt_style = prompt_style

        if self.enabled:
            self.device = decide_device(
                accelerator_options.device,
                supported_devices=vlm_options.supported_devices,
            )
            _log.debug(
                f"Available device for extraction VLM ({prompt_style.value}): "
                f"{self.device}"
            )

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

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*torch_dtype.*deprecated.*",
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message=".*incorrect regex pattern.*",
                    category=UserWarning,
                )
                self.processor = AutoProcessor.from_pretrained(
                    artifacts_path,
                    trust_remote_code=vlm_options.trust_remote_code,
                    use_fast=True,
                )
                self.vlm_model = AutoModelForImageTextToText.from_pretrained(
                    artifacts_path,
                    device_map=self.device,
                    dtype=vlm_options.torch_dtype or torch.bfloat16,
                    _attn_implementation=(
                        "flash_attention_2"
                        if self.device.startswith("cuda")
                        and accelerator_options.cuda_use_flash_attention2
                        else "sdpa"
                    ),
                    trust_remote_code=vlm_options.trust_remote_code,
                )

            if hasattr(self.vlm_model, "merge_lora_adapters"):
                cast(Any, self.vlm_model).merge_lora_adapters()

            if prompt_style == ExtractionPromptStyle.NUEXTRACT and sys.version_info < (
                3,
                14,
            ):
                self.vlm_model = torch.compile(self.vlm_model)  # type: ignore
            else:
                self.vlm_model.eval()

            self.generation_config: Optional[GenerationConfig] = None
            if prompt_style == ExtractionPromptStyle.NUEXTRACT:
                self.processor.tokenizer.padding_side = "left"
                self.generation_config = GenerationConfig.from_pretrained(
                    artifacts_path
                )

    def process_images(
        self,
        image_batch: Iterable[Union[Image, np.ndarray]],
        prompt: Union[str, list[str]],
    ) -> Iterable[VlmPrediction]:
        from PIL import Image as PILImage

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

        if isinstance(prompt, str):
            templates = [prompt] * len(pil_images)
        else:
            if len(prompt) != len(pil_images):
                raise ValueError(
                    f"Number of prompts ({len(prompt)}) must match "
                    f"number of images ({len(pil_images)})"
                )
            templates = prompt

        # Build tokenized inputs based on prompt style
        if self.prompt_style == ExtractionPromptStyle.NUEXTRACT:
            processor_inputs = build_nuextract_inputs(
                processor=self.processor,
                images=pil_images,
                templates=templates,
                device=self.device,
                extra_processor_kwargs=self.vlm_options.extra_processor_kwargs,
            )
        else:
            processor_inputs = build_granite_vision_inputs(
                processor=self.processor,
                images=pil_images,
                templates=templates,
                device=self.device,
            )

        # Generate
        gen_kwargs: dict[str, Any] = {
            **processor_inputs,
            "max_new_tokens": self.max_new_tokens,
        }
        if self.generation_config is not None:
            gen_kwargs["generation_config"] = self.generation_config
            gen_kwargs.update(self.vlm_options.extra_generation_config)
        else:
            gen_kwargs["use_cache"] = True

        if self.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature
        else:
            gen_kwargs["do_sample"] = False

        start_time = time.time()
        with torch.inference_mode():
            generated_ids = cast(Any, self.vlm_model).generate(**gen_kwargs)
        generation_time = time.time() - start_time

        # Decode
        input_len = processor_inputs["input_ids"].shape[1]
        trimmed_sequences = generated_ids[:, input_len:]

        decoded_texts: list[str] = self.processor.batch_decode(
            trimmed_sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        num_tokens = None
        if generated_ids.shape[0] > 0:
            num_tokens = int(generated_ids[0].shape[0])
            _log.debug(
                f"Generated {num_tokens} tokens in {generation_time:.2f}s "
                f"for batch size {generated_ids.shape[0]}."
            )

        for text in decoded_texts:
            decoded_text = self.vlm_options.decode_response(text)
            yield VlmPrediction(
                text=decoded_text,
                generation_time=generation_time,
                num_tokens=num_tokens,
                stop_reason=VlmStopReason.UNSPECIFIED,
            )
