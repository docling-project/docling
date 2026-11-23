# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from functools import partial
from pathlib import Path
from threading import TIMEOUT_MAX
from typing import Optional, Type, Union

from docling_core.types.doc import DoclingDocument, NodeItem
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import (
    ApiImageRequestResult,
    ItemAndImageEnrichmentElement,
)
from docling.datamodel.pipeline_options import (
    PictureDescriptionApiOptions,
    PictureDescriptionBaseOptions,
)
from docling.exceptions import OperationNotAllowed
from docling.models.base_model import EnrichmentTimeoutError
from docling.models.picture_description_base_model import PictureDescriptionBaseModel
from docling.utils.api_image_request import api_image_request


class PictureDescriptionApiModel(PictureDescriptionBaseModel):
    # elements_batch_size = 4

    @classmethod
    def get_options_type(cls) -> Type[PictureDescriptionBaseOptions]:
        return PictureDescriptionApiOptions

    def __init__(
        self,
        enabled: bool,
        enable_remote_services: bool,
        artifacts_path: Optional[Union[Path, str]],
        options: PictureDescriptionApiOptions,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            enable_remote_services=enable_remote_services,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: PictureDescriptionApiOptions
        self.concurrency = self.options.concurrency

        if self.enabled:
            if not enable_remote_services:
                raise OperationNotAllowed(
                    "Connections to remote services is only allowed when set explicitly. "
                    "pipeline_options.enable_remote_services=True."
                )

    def enrich_batch(
        self,
        doc: DoclingDocument,
        element_batch: Iterable[ItemAndImageEnrichmentElement],
        *,
        deadline: float | None,
    ) -> Iterable[NodeItem]:
        if deadline is None:
            return super().enrich_batch(doc, element_batch, deadline=deadline)
        return self._describe_images(
            doc, element_batch, partial(self._annotate_images, deadline=deadline)
        )

    def _annotate_images(
        self, images: Iterable[Image.Image], *, deadline: float | None = None
    ) -> Iterable[ApiImageRequestResult]:
        def remaining_time() -> float | None:
            if deadline is None:
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise EnrichmentTimeoutError
            return remaining

        # Note: technically we could make a batch request here,
        # but not all APIs will allow for it. For example, vllm won't allow more than 1.
        def _api_request(image: Image.Image) -> ApiImageRequestResult:
            remaining = remaining_time()
            timeout = self.options.timeout
            if remaining is not None:
                timeout = min(timeout, remaining)
            return api_image_request(
                image=image,
                prompt=self.options.prompt,
                url=self.options.url,
                timeout=timeout,
                headers=self.options.headers,
                usage_response_key=self.options.usage_response_key,
                **self.options.params,
            )

        executor = ThreadPoolExecutor(max_workers=self.concurrency)
        timed_out = False
        try:
            remaining = remaining_time()
            wait_timeout = (
                remaining if remaining is not None and remaining < TIMEOUT_MAX else None
            )
            for result in executor.map(_api_request, images, timeout=wait_timeout):
                remaining_time()
                yield result
        except FutureTimeoutError as exc:
            if deadline is None:
                raise
            timed_out = True
            raise EnrichmentTimeoutError from exc
        except EnrichmentTimeoutError:
            timed_out = True
            raise
        finally:
            # Running requests own only image data. They can finish independently
            # while the caller returns descriptions already added to the document.
            executor.shutdown(wait=not timed_out, cancel_futures=timed_out)
