# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Remote NuExtract extraction model (dim 2).

Routes NuExtract-style API specs to :func:`api_nuextract_request` (content
array + out-of-band template), rather than the plain image-request shape used
by :class:`ApiVlmModel`.
"""

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Union

import numpy as np
from PIL.Image import Image

from docling.datamodel.base_models import VlmPrediction, VlmStopReason
from docling.datamodel.extraction import ContentItem, ImageContentItem
from docling.datamodel.extraction_options import ApiExtractionVlmOptions
from docling.exceptions import OperationNotAllowed
from docling.models.base_model import BaseVlmModel
from docling.utils.api_nuextract_request import api_nuextract_request


class ApiExtractionVlmModel(BaseVlmModel):
    """Remote extraction model for the NuExtract prompt style."""

    def __init__(
        self,
        enabled: bool,
        enable_remote_services: bool,
        vlm_options: ApiExtractionVlmOptions,
    ):
        self.enabled = enabled
        self.vlm_options = vlm_options
        if self.enabled:
            if not enable_remote_services:
                raise OperationNotAllowed(
                    "Connections to remote services is only allowed when set "
                    "explicitly. pipeline_options.enable_remote_services=True, or "
                    "using the CLI --enable-remote-services."
                )
            self.timeout = vlm_options.timeout
            self.concurrency = vlm_options.concurrency
            self.params = {
                **vlm_options.params,
                "temperature": vlm_options.temperature,
            }

    def process(
        self,
        requests: Iterable[list[ContentItem]],
        template: str,
    ) -> Iterable[VlmPrediction]:
        request_list = [list(req) for req in requests]

        def _run(content_items: list[ContentItem]) -> VlmPrediction:
            resp = api_nuextract_request(
                content_items=content_items,
                template=template,
                url=self.vlm_options.url,
                timeout=self.timeout,
                headers=self.vlm_options.headers,
                **self.params,
            )
            text = self.vlm_options.decode_response(resp.text)
            return VlmPrediction(
                text=text,
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
        image_batch: Iterable[Union[Image, np.ndarray]],
        prompt: Union[str, list[str]],
    ) -> Iterable[VlmPrediction]:
        """Image-only adapter over :meth:`process` (prompt is the template)."""
        images = list(image_batch)
        if isinstance(prompt, list):
            if len(prompt) != len(images):
                raise ValueError(
                    f"Number of prompts ({len(prompt)}) must match number of "
                    f"images ({len(images)})"
                )
            # Per-image templates are not supported by the out-of-band template
            # channel; require a single shared template.
            if len(set(prompt)) > 1:
                raise ValueError(
                    "Remote NuExtract requires a single shared template per batch."
                )
            template = prompt[0] if prompt else ""
        else:
            template = prompt

        from PIL import Image as PILImage

        requests: list[list[ContentItem]] = []
        for img in images:
            if isinstance(img, np.ndarray):
                img = PILImage.fromarray(img.astype(np.uint8))
            requests.append([ImageContentItem(image=img)])
        yield from self.process(requests, template)
