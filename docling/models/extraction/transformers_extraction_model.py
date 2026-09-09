# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import logging
import sys
import time
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from packaging import version
from PIL.Image import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    GenerationConfig,
)

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import VlmPrediction, VlmStopReason
from docling.datamodel.extraction import ContentItem
from docling.datamodel.extraction_options import (
    ExtractionPromptStyle,
    ExtractionVlmOptions,
)
from docling.datamodel.pipeline_options_vlm_model import TransformersModelType
from docling.datamodel.vlm_engine_options import TransformersVlmEngineOptions
from docling.models.base_model import BaseVlmModel
from docling.models.extraction.prompt_utils import (
    build_granite_vision_inputs,
    build_nuextract_content_inputs,
    build_nuextract_inputs,
)
from docling.models.utils.generation_utils import build_generation_config
from docling.models.utils.hf_model_download import HuggingFaceModelDownloadMixin
from docling.utils.accelerator_utils import decide_device
from docling.utils.vlm_utils import strip_stop_strings

_log = logging.getLogger(__name__)


class TransformersExtractionModel(BaseVlmModel, HuggingFaceModelDownloadMixin):
    """Unified extraction model supporting multiple prompt styles."""

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Path | None,
        accelerator_options: AcceleratorOptions,
        vlm_options: ExtractionVlmOptions,
    ):
        self.enabled = enabled
        self.model_spec = vlm_options.model_spec
        engine_options = vlm_options.engine_options
        assert isinstance(engine_options, TransformersVlmEngineOptions)
        self.engine_options = engine_options
        self.prompt_style = self.model_spec.prompt_style

        if self.enabled:
            if (
                self.model_spec.transformers_model_type
                != TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT
            ):
                raise ValueError("Extraction supports only AutoModelForImageTextToText")
            self.device = decide_device(
                engine_options.device or accelerator_options.device,
                supported_devices=self.model_spec.supported_devices,
            )
            _log.debug(
                f"Available device for extraction VLM ({self.prompt_style.value}): "
                f"{self.device}"
            )

            self.max_new_tokens = self.model_spec.max_new_tokens
            self.temperature = self.model_spec.temperature

            repo_id = self.model_spec.get_repo_id(engine_options.engine_type)
            revision = self.model_spec.get_revision(engine_options.engine_type)
            repo_cache_folder = repo_id.replace("/", "--")

            if artifacts_path is None:
                artifacts_path = self.download_models(
                    repo_id=repo_id,
                    revision=revision,
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
                    trust_remote_code=engine_options.trust_remote_code,
                    revision=revision,
                    use_fast=True,
                )
                quantization_config = None
                if engine_options.quantized:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=engine_options.load_in_8bit,
                        llm_int8_threshold=engine_options.llm_int8_threshold,
                    )
                self.vlm_model = AutoModelForImageTextToText.from_pretrained(
                    artifacts_path,
                    device_map=self.device,
                    dtype=(
                        engine_options.torch_dtype
                        or self.model_spec.torch_dtype
                        or torch.bfloat16
                    ),
                    _attn_implementation=(
                        "flash_attention_2"
                        if self.device.startswith("cuda")
                        and accelerator_options.cuda_use_flash_attention2
                        else "sdpa"
                    ),
                    trust_remote_code=engine_options.trust_remote_code,
                    revision=revision,
                    quantization_config=quantization_config,
                )

            # Granite's remote model code optionally exposes adapter merging.
            if hasattr(self.vlm_model, "merge_lora_adapters"):
                cast(Any, self.vlm_model).merge_lora_adapters()

            self.vlm_model.eval()
            if engine_options.compile_model:
                if sys.version_info < (3, 14) or version.parse(
                    torch.__version__
                ) >= version.parse("2.10"):
                    self.vlm_model = cast(Any, torch.compile(self.vlm_model))
                else:
                    _log.warning(
                        "Model compilation requires Python < 3.14 or torch >= 2.10"
                    )

            self.generation_config: GenerationConfig | None = None
            if self.prompt_style == ExtractionPromptStyle.NUEXTRACT:
                self.processor.tokenizer.padding_side = "left"
                self.generation_config = GenerationConfig.from_pretrained(
                    artifacts_path, revision=revision
                )

    def process_images(
        self,
        image_batch: Iterable[Image | np.ndarray],
        prompt: str | list[str],
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

        if self.prompt_style == ExtractionPromptStyle.NUEXTRACT:
            processor_inputs = build_nuextract_inputs(
                processor=self.processor,
                images=pil_images,
                templates=templates,
                device=self.device,
                extra_processor_kwargs=self.model_spec.extra_processor_kwargs,
            )
        else:
            processor_inputs = build_granite_vision_inputs(
                processor=self.processor,
                images=pil_images,
                prompts=templates,
                device=self.device,
            )

        yield from self._generate_and_decode(processor_inputs)

    def process(
        self,
        requests: Iterable[list[ContentItem]],
        template: str,
    ) -> Iterable[VlmPrediction]:
        """Run NuExtract inference over content-item requests."""
        if self.prompt_style != ExtractionPromptStyle.NUEXTRACT:
            raise ValueError(
                f"process() with content items is only supported for the "
                f"NuExtract prompt style, not {self.prompt_style.value}."
            )

        request_list = [list(req) for req in requests]
        if not request_list:
            return

        processor_inputs = build_nuextract_content_inputs(
            processor=self.processor,
            requests=request_list,
            templates=[template] * len(request_list),
            device=self.device,
            extra_processor_kwargs=self.model_spec.extra_processor_kwargs,
        )
        yield from self._generate_and_decode(processor_inputs)

    def _generate_and_decode(
        self, processor_inputs: dict[str, Any]
    ) -> Iterable[VlmPrediction]:
        tokenizer = self.processor.tokenizer
        generation_config = build_generation_config(
            self.generation_config,
            overrides=self.model_spec.extra_generation_config,
            max_new_tokens=self.max_new_tokens,
            use_cache=self.engine_options.use_kv_cache,
            do_sample=self.temperature > 0,
            temperature=self.temperature if self.temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        gen_kwargs: dict[str, Any] = {
            **processor_inputs,
            "generation_config": generation_config,
        }

        max_input_tokens = self.model_spec.max_input_tokens
        if max_input_tokens is not None:
            input_tokens = processor_inputs["input_ids"].shape[1]
            if input_tokens > max_input_tokens:
                raise ValueError(
                    f"Input is {input_tokens} tokens, exceeding the configured "
                    f"context limit of {max_input_tokens} for model "
                    f"'{self.model_spec.name}'. Reduce the page range or input "
                    f"size, or raise max_input_tokens."
                )

        start_time = time.time()
        with torch.inference_mode():
            generated_ids = cast(Any, self.vlm_model).generate(**gen_kwargs)
        generation_time = time.time() - start_time

        input_len = processor_inputs["input_ids"].shape[1]
        trimmed_sequences = generated_ids[:, input_len:]

        decoded_texts: list[str] = self.processor.batch_decode(
            trimmed_sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        if generated_ids.shape[0] > 0:
            _log.debug(
                f"Generated up to {trimmed_sequences.shape[1]} tokens "
                f"in {generation_time:.2f}s "
                f"for batch size {generated_ids.shape[0]}."
            )

        eos_token_id = generation_config.eos_token_id
        eos_token_ids = (
            set(eos_token_id)
            if isinstance(eos_token_id, list)
            else {eos_token_id}
            if eos_token_id is not None
            else set()
        )
        for text, sequence in zip(decoded_texts, trimmed_sequences):
            token_ids = sequence.tolist()
            if any(stop in text for stop in self.model_spec.stop_strings):
                stop_reason = VlmStopReason.STOP_SEQUENCE
            elif eos_token_ids.intersection(token_ids):
                stop_reason = VlmStopReason.END_OF_SEQUENCE
            elif len(token_ids) >= self.max_new_tokens:
                stop_reason = VlmStopReason.LENGTH
            else:
                stop_reason = VlmStopReason.UNSPECIFIED
            yield VlmPrediction(
                text=strip_stop_strings([text], self.model_spec.stop_strings)[0],
                generation_time=generation_time,
                num_tokens=len(token_ids),
                stop_reason=stop_reason,
            )
