from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Union

import numpy as np
from PIL.Image import Image

from docling.datamodel.base_models import Page, VlmPrediction, VlmStopReason
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions
from docling.exceptions import OperationNotAllowed
from docling.models.base_model import BaseVlmPageModel
from docling.models.utils.generation_utils import GenerationStopper
from docling.utils.api_image_request import (
    api_image_request,
    api_image_request_streaming,
)
from docling.utils.profiling import TimeRecorder


class ApiVlmModel(BaseVlmPageModel):
    # Override the vlm_options type annotation from BaseVlmPageModel
    vlm_options: ApiVlmOptions  # type: ignore[assignment]

    def __init__(
        self,
        enabled: bool,
        enable_remote_services: bool,
        vlm_options: ApiVlmOptions,
    ):
        self.enabled = enabled
        self.vlm_options = vlm_options
        if self.enabled:
            if not enable_remote_services:
                raise OperationNotAllowed(
                    "Connections to remote services is only allowed when set explicitly. "
                    "pipeline_options.enable_remote_services=True, or using the CLI "
                    "--enable-remote-services."
                )

            self.timeout = self.vlm_options.timeout
            self.concurrency = self.vlm_options.concurrency
            self.params = {
                **self.vlm_options.params,
                "temperature": self.vlm_options.temperature,
            }

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        page_list = list(page_batch)
        if not page_list:
            return

        original_order = page_list[:]
        valid_pages = []

        for page in page_list:
            assert page._backend is not None
            if page._backend.is_valid():
                valid_pages.append(page)

        # Process valid pages in batch
        if valid_pages:
            with TimeRecorder(conv_res, "vlm"):
                # Prepare images and prompts for batch processing
                images = []
                prompts = []
                pages_with_images = []

                for page in valid_pages:
                    assert page.size is not None
                    hi_res_image = page.get_image(
                        scale=self.vlm_options.scale, max_size=self.vlm_options.max_size
                    )

                    # Only process pages with valid images
                    if hi_res_image is not None:
                        images.append(hi_res_image)
                        prompt = self._build_prompt_safe(page)
                        prompts.append(prompt)
                        pages_with_images.append(page)

                # Use process_images for the actual inference
                if images:  # Only if we have valid images
                    with TimeRecorder(conv_res, "vlm_inference"):
                        predictions = list(self.process_images(images, prompts))

                    # Attach results to pages
                    for page, prediction in zip(pages_with_images, predictions):
                        page.predictions.vlm_response = prediction

        # Yield pages preserving original order
        for page in original_order:
            yield page

    def process_images(
        self,
        image_batch: Iterable[Union[Image, np.ndarray]],
        prompt: Union[str, list[str]],
    ) -> Iterable[VlmPrediction]:
        """Process raw images without page metadata."""
        images = list(image_batch)

        # Handle prompt parameter
        if isinstance(prompt, str):
            prompts = [prompt] * len(images)
        elif isinstance(prompt, list):
            if len(prompt) != len(images):
                raise ValueError(
                    f"Prompt list length ({len(prompt)}) must match image count ({len(images)})"
                )
            prompts = prompt

        def _process_single_image(image_prompt_pair):
            image, prompt_text = image_prompt_pair

            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                if image.ndim == 3 and image.shape[2] in [3, 4]:
                    from PIL import Image as PILImage

                    image = PILImage.fromarray(image.astype(np.uint8))
                elif image.ndim == 2:
                    from PIL import Image as PILImage

                    image = PILImage.fromarray(image.astype(np.uint8), mode="L")
                else:
                    raise ValueError(f"Unsupported numpy array shape: {image.shape}")

            # Ensure image is in RGB mode
            if image.mode != "RGB":
                image = image.convert("RGB")

            stop_reason = VlmStopReason.UNSPECIFIED

            if self.vlm_options.custom_stopping_criteria:
                # Instantiate any GenerationStopper classes before passing to streaming
                instantiated_stoppers = []
                for criteria in self.vlm_options.custom_stopping_criteria:
                    if isinstance(criteria, GenerationStopper):
                        instantiated_stoppers.append(criteria)
                    elif isinstance(criteria, type) and issubclass(
                        criteria, GenerationStopper
                    ):
                        instantiated_stoppers.append(criteria())
                    # Skip non-GenerationStopper criteria (should have been caught in validation)

                # Streaming path with early abort support
                page_tags, num_tokens = api_image_request_streaming(
                    image=image,
                    prompt=prompt_text,
                    url=self.vlm_options.url,
                    timeout=self.timeout,
                    headers=self.vlm_options.headers,
                    generation_stoppers=instantiated_stoppers,
                    **self.params,
                )
            else:
                # Non-streaming fallback (existing behavior)
                page_tags, num_tokens, stop_reason = api_image_request(
                    image=image,
                    prompt=prompt_text,
                    url=self.vlm_options.url,
                    timeout=self.timeout,
                    headers=self.vlm_options.headers,
                    **self.params,
                )

            page_tags = self.vlm_options.decode_response(page_tags)
            input_prompt = prompt_text if self.vlm_options.track_input_prompt else None
            return VlmPrediction(
                text=page_tags,
                num_tokens=num_tokens,
                stop_reason=stop_reason,
                input_prompt=input_prompt,
            )

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            yield from executor.map(_process_single_image, zip(images, prompts))
