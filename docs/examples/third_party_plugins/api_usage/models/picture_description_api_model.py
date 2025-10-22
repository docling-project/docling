from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Type, Union

from api_usage.datamodel.utils.api_image_request_with_usage import (
    api_image_request_with_usage,
)
from docling_core.types.doc import DoclingDocument, NodeItem, PictureItem
from docling_core.types.doc.document import (
    DescriptionAnnotation,
)  # TODO: move import to docling_core.types.doc
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import OpenAiResponseUsage
from docling.datamodel.pipeline_options import PictureDescriptionBaseOptions
from docling.exceptions import OperationNotAllowed
from docling.models.base_model import ItemAndImageEnrichmentElement
from docling.models.picture_description_api_model import PictureDescriptionApiModel
from docs.examples.third_party_plugins.api_usage.datamodel.pipeline_options.picture_description_api_options_with_usage import (
    PictureDescriptionApiOptionsWithUsage,
)


class DescriptionAnnotationWithUsage(DescriptionAnnotation):
    """DescriptionAnnotation."""

    usage: Optional[OpenAiResponseUsage] = None


class PictureDescriptionApiModelWithUsage(PictureDescriptionApiModel):
    # elements_batch_size = 4

    @classmethod
    def get_options_type(cls) -> Type[PictureDescriptionBaseOptions]:
        return PictureDescriptionApiOptionsWithUsage

    def __init__(
        self,
        enabled: bool,
        enable_remote_services: bool,
        artifacts_path: Optional[Union[Path, str]],
        options: PictureDescriptionApiOptionsWithUsage,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            enable_remote_services=enable_remote_services,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: PictureDescriptionApiOptionsWithUsage
        self.concurrency = self.options.concurrency

        if self.enabled:
            if not enable_remote_services:
                raise OperationNotAllowed(
                    "Connections to remote services is only allowed when set explicitly. "
                    "pipeline_options.enable_remote_services=True."
                )

    def _annotate_images(self, images: Iterable[Image.Image]) -> Iterable[str]:
        # Note: technically we could make a batch request here,
        # but not all APIs will allow for it. For example, vllm won't allow more than 1.
        def _api_request(image):
            return api_image_request_with_usage(
                image=image,
                prompt=self.options.prompt,
                url=self.options.url,
                timeout=self.options.timeout,
                headers=self.options.headers,
                **self.options.params,
            )

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            yield from executor.map(_api_request, images)

    def __call__(
        self,
        doc: DoclingDocument,
        element_batch: Iterable[ItemAndImageEnrichmentElement],
    ) -> Iterable[NodeItem]:
        if not self.enabled:
            for element in element_batch:
                yield element.item
            return

        images: List[Image.Image] = []
        elements: List[PictureItem] = []
        for el in element_batch:
            assert isinstance(el.item, PictureItem)
            describe_image = True
            # Don't describe the image if it's smaller than the threshold
            if len(el.item.prov) > 0:
                prov = el.item.prov[0]  # PictureItems have at most a single provenance
                page = doc.pages.get(prov.page_no)
                if page is not None:
                    page_area = page.size.width * page.size.height
                    if page_area > 0:
                        area_fraction = prov.bbox.area() / page_area
                        if area_fraction < self.options.picture_area_threshold:
                            describe_image = False
            if describe_image:
                elements.append(el.item)
                images.append(el.image)

        outputs = self._annotate_images(images)

        for item, output in zip(elements, outputs):
            # api_image_request now may return (text, usage) or plain text;
            # normalize to tuple
            if isinstance(output, tuple):
                text, usage = output
            else:
                text, usage = output, None

            item.annotations.append(
                DescriptionAnnotationWithUsage(
                    text=text, provenance=self.provenance, usage=usage
                )
            )
            yield item
