import logging
import time
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional, Union, cast

import numpy as np
import torch
from PIL.Image import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import VlmPrediction, VlmStopReason
from docling.datamodel.pipeline_options_vlm_model import InlineVlmOptions
from docling.models.base_model import BaseVlmModel
from docling.models.utils.hf_model_download import HuggingFaceModelDownloadMixin
from docling.utils.accelerator_utils import decide_device

_log = logging.getLogger(__name__)


class GraniteVisionExtractionModel(BaseVlmModel, HuggingFaceModelDownloadMixin):
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
            self.device = decide_device(
                accelerator_options.device,
                supported_devices=vlm_options.supported_devices,
            )
            _log.debug(f"Available device for Granite Vision extraction: {self.device}")

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
                )
                self.model_max_length = self.processor.tokenizer.model_max_length
                self.vlm_model = AutoModelForImageTextToText.from_pretrained(
                    artifacts_path,
                    device_map=self.device,
                    dtype=torch.bfloat16,
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
            self.vlm_model.eval()

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
            prompts = [prompt] * len(pil_images)
        else:
            if len(prompt) != len(pil_images):
                raise ValueError(
                    f"Number of prompts ({len(prompt)}) must match number of images ({len(pil_images)})"
                )
            prompts = prompt

        extraction_prompts = [self._build_extraction_prompt(p) for p in prompts]

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
            self.processor.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True
            )
            for conv in conversations
        ]
        inputs = self.processor(
            text=texts,
            images=pil_images,
            return_tensors="pt",
            padding=True,
            do_pad=True,
        ).to(self.device)

        gen_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": self.max_new_tokens or self.model_max_length,
            "use_cache": True,
        }
        if self.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature
        else:
            gen_kwargs["do_sample"] = False

        start_time = time.time()
        with torch.inference_mode():
            output_ids = cast(Any, self.vlm_model).generate(**gen_kwargs)
        generation_time = time.time() - start_time

        input_len = inputs["input_ids"].shape[1]
        decoded_texts: list[str] = [
            self.processor.decode(
                output_ids[i, input_len:],
                skip_special_tokens=True,
            )
            for i in range(len(pil_images))
        ]

        num_tokens = None
        if output_ids.shape[0] > 0:
            num_tokens = int(output_ids[0].shape[0])
            _log.debug(
                f"Generated {num_tokens} tokens in {generation_time:.2f}s "
                f"for batch size {output_ids.shape[0]}."
            )

        for text in decoded_texts:
            yield VlmPrediction(
                text=text,
                generation_time=generation_time,
                num_tokens=num_tokens,
                stop_reason=VlmStopReason.UNSPECIFIED,
            )

    @staticmethod
    def _build_extraction_prompt(template: str) -> str:
        return (
            "Extract structured data from this document image.\n"
            "Return a JSON object matching this schema:\n\n"
            f"{template}\n\n"
            "Return null for fields you cannot find in the document.\n"
            "Return ONLY valid JSON, no other text."
        )
