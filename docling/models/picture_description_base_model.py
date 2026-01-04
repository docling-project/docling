from abc import abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import List, Optional, Type, Union

from docling_core.types.doc import (
    DoclingDocument,
    NodeItem,
    PictureClassificationData,
    PictureItem,
)
from docling_core.types.doc.document import (  # TODO: move import to docling_core.types.doc
    PictureDescriptionData,
)
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import (
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
            if describe_image and not _passes_classification(
                el.item.annotations,
                self.options.classification_allow,
                self.options.classification_deny,
                self.options.classification_min_confidence,
            ):
                describe_image = False
            if describe_image:
                elements.append(el.item)
                images.append(el.image)

        outputs = self._annotate_images(images)

        for item, output in zip(elements, outputs):
            item.annotations.append(
                PictureDescriptionData(text=output, provenance=self.provenance)
            )
            yield item


def _passes_classification(
    annotations: Iterable[object],
    allow: Optional[List[str]],
    deny: Optional[List[str]],
    min_confidence: float,
) -> bool:
    if not allow and not deny:
        return True
    predicted = None
    for annotation in annotations:
        if isinstance(annotation, PictureClassificationData):
            predicted = annotation.predicted_classes
            break
    if not predicted:
        return allow is None
    if deny:
        deny_set = set(deny)
        for entry in predicted:
            if entry.confidence >= min_confidence and entry.class_name in deny_set:
                return False
    if allow:
        allow_set = set(allow)
        return any(
            entry.confidence >= min_confidence and entry.class_name in allow_set
            for entry in predicted
        )
    return True

    @classmethod
    @abstractmethod
    def get_options_type(cls) -> Type[PictureDescriptionBaseOptions]:
        pass
