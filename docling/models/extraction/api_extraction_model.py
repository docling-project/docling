# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""OpenAI-compatible extraction model."""

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any

import numpy as np
from PIL import Image as PILImage
from PIL.Image import Image

from docling.datamodel.base_models import VlmPrediction
from docling.datamodel.extraction import ContentItem, ImageContentItem
from docling.datamodel.extraction_options import (
    ExtractionVlmOptions,
    _merge_chat_options,
    _reject_request_fields,
)
from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions
from docling.exceptions import OperationNotAllowed
from docling.models.base_model import BaseVlmModel
from docling.models.extraction.prompt_utils import (
    _PreparedTarget,
    prepare_output_target,
    prepared_image_prompt,
)
from docling.models.inference_engines.vlm.base import VlmEngineType
from docling.utils.api_extraction_request import api_extraction_request


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
        self.output_mode = vlm_options.output_mode
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
        target: _PreparedTarget,
    ) -> Iterable[VlmPrediction]:
        target = prepare_output_target(
            target, self.output_mode, self.engine_options.engine_type
        )
        params, chat = self._request_options(target)
        request_list = [list(req) for req in requests]
        if not request_list:
            return
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            yield from executor.map(
                lambda content: self._run(content, target, params, chat), request_list
            )

    def _request_options(
        self, target: _PreparedTarget
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        model_params = self.model_spec.get_api_params(self.engine_options.engine_type)
        engine_params = self.engine_options.params
        for mapping in (model_params, engine_params):
            _reject_request_fields(mapping)
        dynamic = {
            key: value
            for key, value in target.chat_template_kwargs.items()
            if key in {"template", "instructions"}
        }
        defaults = {
            key: value
            for key, value in target.chat_template_kwargs.items()
            if key not in dynamic
        }
        chat = _merge_chat_options(
            defaults,
            model_params.get("chat_template_kwargs", {}),
            engine_params.get("chat_template_kwargs", {}),
        )
        if (
            self.model_spec.preparation == "nuextract"
            and chat.get("mode", "structured") != "structured"
        ):
            raise ValueError("NuExtract extraction requires structured mode")
        chat.update(deepcopy(dynamic))
        if chat and self.engine_options.engine_type != VlmEngineType.API:
            raise ValueError(
                "chat_template_kwargs require the explicitly configured vLLM API engine"
            )
        params = deepcopy(self.params)
        params.pop("chat_template_kwargs", None)
        return params, chat

    def _run(
        self,
        content: list[ContentItem],
        target: _PreparedTarget,
        params: dict[str, Any],
        chat: dict[str, Any],
    ) -> VlmPrediction:
        resp = api_extraction_request(
            content_items=content,
            prompt=target.prompt,
            chat_template_kwargs=deepcopy(chat),
            constraint_schema=target.constraint_schema,
            url=self.engine_options.url,
            timeout=self.timeout,
            headers=self.engine_options.headers,
            **deepcopy(params),
        )
        if not resp.text.strip():
            raise RuntimeError("Extraction API returned no content")
        return VlmPrediction(
            text=resp.text,
            num_tokens=resp.num_tokens,
            usage=resp.usage,
            stop_reason=resp.stop_reason,
        )

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

        for image, text in zip(pil_images, prompts):
            yield from self.process(
                [[ImageContentItem(image=image)]],
                prepared_image_prompt(text, self.model_spec),
            )
