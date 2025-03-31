import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any, Iterable, List, Optional, Type, Union

from docling_core.types.doc import (
    DoclingDocument,
    NodeItem,
    PictureClassificationClass,
    PictureItem,
)
from docling_core.types.doc.document import (  # TODO: move import to docling_core.types.doc
    PictureDescriptionData,
)
from PIL import Image

from docling.datamodel.pipeline_options import (
    AcceleratorOptions,
    PictureDescriptionBaseOptions,
)
from docling.models.base_model import (
    BaseItemAndImageEnrichmentModel,
    BaseModelWithOptions,
    ItemAndImageEnrichmentElement,
)


class PictureDescriptionBaseModel(
    BaseItemAndImageEnrichmentModel, BaseModelWithOptions
):
    images_scale: float = 2.0

    def __init__(
        self,
        *,
        enabled: bool,
        enable_remote_services: bool,
        artifacts_path: Optional[Union[Path, str]],
        options: PictureDescriptionBaseOptions,
        accelerator_options: AcceleratorOptions,
    ):
        self.enabled = enabled
        self.options = options
        self.provenance = "not-implemented"

    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        return self.enabled and isinstance(element, PictureItem)

    def _annotate_images(self, images: Iterable[Image.Image]) -> Iterable[str]:
        raise NotImplementedError

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
            elements.append(el.item)
            images.append(el.image)

        outputs = self._annotate_images(images)

        for item, output in zip(elements, outputs):
            item.annotations.append(
                PictureDescriptionData(text=output, provenance=self.provenance)
            )
            yield item

    @classmethod
    @abstractmethod
    def get_options_type(cls) -> Type[PictureDescriptionBaseOptions]:
        pass
