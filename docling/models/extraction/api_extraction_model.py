# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""OpenAI-compatible extraction model."""

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from PIL import Image as PILImage
from PIL.Image import Image

from docling.datamodel.base_models import VlmPrediction, VlmStopReason
from docling.datamodel.extraction import ContentItem, ImageContentItem
from docling.datamodel.extraction_options import (
    ExtractionPromptStyle,
    ExtractionVlmOptions,
)
from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions
from docling.exceptions import OperationNotAllowed
from docling.models.base_model import BaseVlmModel
from docling.utils.api_image_request import api_image_request
from docling.utils.api_nuextract_request import api_nuextract_request


class ApiExtractionVlmModel(BaseVlmModel):
    """Run structured extraction through an OpenAI-compatible API."""

    def __init__(
        self,
        enabled: bool,
        enable_remote_services: bool,
        vlm_options: ExtractionVlmOptions,
    ):
        self.enabled = enabled
        self.model_spec = vlm_options.model_spec
        engine_options = vlm_options.engine_options
        assert isinstance(engine_options, ApiVlmEngineOptions)
        self.engine_options = engine_options
        if self.enabled:
            if not enable_remote_services:
                raise OperationNotAllowed(
                    "Connections to remote services is only allowed when set "
                    "explicitly. pipeline_options.enable_remote_services=True, or "
                    "using the CLI --enable-remote-services."
                )
            self.timeout = engine_options.timeout
            self.concurrency = engine_options.concurrency
            self.params: dict[str, Any] = {
                "temperature": self.model_spec.temperature,
                "max_tokens": self.model_spec.max_new_tokens,
                **vlm_options.get_api_params(),
            }

    def process(
        self,
        requests: Iterable[list[ContentItem]],
        template: str,
    ) -> Iterable[VlmPrediction]:
        if self.model_spec.prompt_style != ExtractionPromptStyle.NUEXTRACT:
            raise ValueError("Content extraction is supported only by NuExtract")
        request_list = [list(req) for req in requests]

        def _run(content_items: list[ContentItem]) -> VlmPrediction:
            resp = api_nuextract_request(
                content_items=content_items,
                template=template,
                url=self.engine_options.url,
                timeout=self.timeout,
                headers=self.engine_options.headers,
                **self.params,
            )
            if not resp.text.strip():
                raise RuntimeError("Extraction API returned no content")
            return VlmPrediction(
                text=resp.text,
                num_tokens=resp.num_tokens,
                usage=resp.usage,
                stop_reason=resp.stop_reason,
            )

        if not request_list:
            return
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            yield from executor.map(_run, request_list)

    def process_images(
        self,
        image_batch: Iterable[Image | np.ndarray],
        prompt: str | list[str],
    ) -> Iterable[VlmPrediction]:
        images = list(image_batch)
        if isinstance(prompt, list):
            if len(prompt) != len(images):
                raise ValueError(
                    f"Number of prompts ({len(prompt)}) must match number of "
                    f"images ({len(images)})"
                )
            if (
                self.model_spec.prompt_style == ExtractionPromptStyle.NUEXTRACT
                and len(set(prompt)) > 1
            ):
                raise ValueError(
                    "Remote NuExtract requires a single shared template per batch."
                )
            prompts = prompt
        else:
            prompts = [prompt] * len(images)

        pil_images: list[Image] = []
        for image in images:
            img = image
            if isinstance(img, np.ndarray):
                if img.ndim == 3 and img.shape[2] in (3, 4):
                    img = PILImage.fromarray(img.astype(np.uint8))
                elif img.ndim == 2:
                    img = PILImage.fromarray(img.astype(np.uint8), mode="L")
                else:
                    raise ValueError(f"Unsupported numpy array shape: {img.shape}")
            pil_images.append(img)

        if self.model_spec.prompt_style == ExtractionPromptStyle.NUEXTRACT:
            requests: list[list[ContentItem]] = [
                [ImageContentItem(image=img)] for img in pil_images
            ]
            yield from self.process(requests, prompts[0] if prompts else "")
            return

        def _run(image_prompt: tuple[Image, str]) -> VlmPrediction:
            image, prompt_text = image_prompt
            resp = api_image_request(
                image=image,
                prompt=prompt_text,
                url=self.engine_options.url,
                timeout=self.timeout,
                headers=self.engine_options.headers,
                **self.params,
            )
            if not resp.text.strip():
                raise RuntimeError("Extraction API returned no content")
            return VlmPrediction(
                text=resp.text,
                num_tokens=resp.num_tokens,
                usage=resp.usage,
                stop_reason=resp.stop_reason,
            )

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            yield from executor.map(_run, zip(pil_images, prompts))
