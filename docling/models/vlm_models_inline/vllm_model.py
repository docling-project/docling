import logging
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from PIL.Image import Image

from docling.datamodel.accelerator_options import (
    AcceleratorOptions,
)
from docling.datamodel.base_models import Page, VlmPrediction
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options_vlm_model import (
    InlineVlmOptions,
    TransformersPromptStyle,
)
from docling.models.base_model import BaseVlmPageModel
from docling.models.utils.hf_model_download import (
    HuggingFaceModelDownloadMixin,
)
from docling.utils.accelerator_utils import decide_device
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)


class VllmVlmModel(BaseVlmPageModel, HuggingFaceModelDownloadMixin):
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
            from transformers import AutoProcessor
            from vllm import LLM, SamplingParams

            self.device = decide_device(
                accelerator_options.device,
                supported_devices=vlm_options.supported_devices,
            )
            _log.debug(f"Available device for VLM: {self.device}")

            self.max_new_tokens = vlm_options.max_new_tokens
            self.temperature = vlm_options.temperature

            repo_cache_folder = vlm_options.repo_id.replace("/", "--")

            if artifacts_path is None:
                artifacts_path = self.download_models(self.vlm_options.repo_id)
            elif (artifacts_path / repo_cache_folder).exists():
                artifacts_path = artifacts_path / repo_cache_folder

            # Initialize VLLM LLM
            llm_kwargs: Dict[str, Any] = {
                "model": str(artifacts_path),
                "limit_mm_per_prompt": {"image": 1},
                "trust_remote_code": vlm_options.trust_remote_code,
                "model_impl": "transformers",
                "gpu_memory_utilization": 0.3,  # hardcoded for now, leaves room for ~3 different models.
            }

            # Add device-specific configurations

            if self.device == "cpu":
                llm_kwargs["device"] = "cpu"

            # Add quantization if specified
            if vlm_options.quantized:
                if vlm_options.load_in_8bit:
                    llm_kwargs["quantization"] = "bitsandbytes"

            self.llm = LLM(**llm_kwargs)

            # Initialize processor for prompt formatting
            self.processor = AutoProcessor.from_pretrained(
                artifacts_path,
                trust_remote_code=vlm_options.trust_remote_code,
            )

            # Set up sampling parameters
            self.sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                stop=vlm_options.stop_strings if vlm_options.stop_strings else None,
                **vlm_options.extra_generation_config,
            )

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        page_list = list(page_batch)
        if not page_list:
            return

        valid_pages = []
        invalid_pages = []

        for page in page_list:
            assert page._backend is not None
            if not page._backend.is_valid():
                invalid_pages.append(page)
            else:
                valid_pages.append(page)

        # Process valid pages in batch
        if valid_pages:
            with TimeRecorder(conv_res, "vlm"):
                # Prepare images and prompts for batch processing
                images = []
                user_prompts = []
                pages_with_images = []

                for page in valid_pages:
                    assert page.size is not None
                    hi_res_image = page.get_image(
                        scale=self.vlm_options.scale, max_size=self.vlm_options.max_size
                    )

                    # Only process pages with valid images
                    if hi_res_image is not None:
                        images.append(hi_res_image)

                        # Define prompt structure
                        if callable(self.vlm_options.prompt):
                            user_prompt = self.vlm_options.prompt(page.parsed_page)
                        else:
                            user_prompt = self.vlm_options.prompt

                        user_prompts.append(user_prompt)
                        pages_with_images.append(page)

                # Use process_images for the actual inference
                if images:  # Only if we have valid images
                    predictions = list(self.process_images(images, user_prompts))

                    # Attach results to pages
                    for page, prediction in zip(pages_with_images, predictions):
                        page.predictions.vlm_response = prediction

        # Yield all pages (valid and invalid)
        for page in invalid_pages:
            yield page
        for page in valid_pages:
            yield page

    def process_images(
        self,
        image_batch: Iterable[Union[Image, np.ndarray]],
        prompt: Union[str, list[str]],
    ) -> Iterable[VlmPrediction]:
        """Process raw images without page metadata in a single batched inference call.

        Args:
            image_batch: Iterable of PIL Images or numpy arrays
            prompt: Either:
                - str: Single prompt used for all images
                - list[str]: List of prompts (one per image, must match image count)

        Raises:
            ValueError: If prompt list length doesn't match image count.
        """
        pil_images: list[Image] = []

        for img in image_batch:
            # Convert numpy array to PIL Image if needed
            if isinstance(img, np.ndarray):
                if img.ndim == 3 and img.shape[2] in [3, 4]:
                    from PIL import Image as PILImage

                    pil_img = PILImage.fromarray(img.astype(np.uint8))
                elif img.ndim == 2:
                    from PIL import Image as PILImage

                    pil_img = PILImage.fromarray(img.astype(np.uint8), mode="L")
                else:
                    raise ValueError(f"Unsupported numpy array shape: {img.shape}")
            else:
                pil_img = img

            # Ensure image is in RGB mode (handles RGBA, L, etc.)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")

            pil_images.append(pil_img)

        if len(pil_images) == 0:
            return

        # Handle prompt parameter
        if isinstance(prompt, str):
            # Single prompt for all images
            user_prompts = [prompt] * len(pil_images)
        elif isinstance(prompt, list):
            # List of prompts (one per image)
            if len(prompt) != len(pil_images):
                raise ValueError(
                    f"Number of prompts ({len(prompt)}) must match number of images ({len(pil_images)})"
                )
            user_prompts = prompt
        else:
            raise ValueError(f"prompt must be str or list[str], got {type(prompt)}")

        # Format prompts individually
        prompts: list[str] = [
            self.formulate_prompt(user_prompt) for user_prompt in user_prompts
        ]

        # Prepare VLLM inputs
        llm_inputs = []
        for prompt, image in zip(prompts, pil_images):
            llm_inputs.append({"prompt": prompt, "multi_modal_data": {"image": image}})

        start_time = time.time()
        outputs = self.llm.generate(llm_inputs, sampling_params=self.sampling_params)  # type: ignore
        generation_time = time.time() - start_time

        # Logging tokens count for the first sample as a representative metric
        if len(outputs) > 0:
            num_tokens = len(outputs[0].outputs[0].token_ids)
            _log.debug(
                f"Generated {num_tokens} tokens in time {generation_time:.2f} seconds."
            )

        for output in outputs:
            # Apply decode_response to the output text
            decoded_text = self.vlm_options.decode_response(output.outputs[0].text)
            yield VlmPrediction(text=decoded_text, generation_time=generation_time)
