from abc import abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import List, Optional, Type, Union

from docling_core.types.doc import (
    DescriptionMetaField,
    DoclingDocument,
    NodeItem,
    PictureClassificationLabel,
    PictureItem,
    PictureMeta,
)
from docling_core.types.doc.document import PictureDescriptionData
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
                el.item.meta,
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
            # FIXME: annotations is deprecated, remove once all consumers use meta.classification
            item.annotations.append(
                PictureDescriptionData(text=output, provenance=self.provenance)
            )

            # Store classification in the new meta field
            if item.meta is None:
                item.meta = PictureMeta()
            item.meta.description = DescriptionMetaField(
                text=output,
                created_by=self.provenance,
            )

            yield item

    @classmethod
    @abstractmethod
    def get_options_type(cls) -> Type[PictureDescriptionBaseOptions]:
        pass


def _passes_classification(
    meta: Optional[PictureMeta],
    allow: Optional[List[PictureClassificationLabel]],
    deny: Optional[List[PictureClassificationLabel]],
    min_confidence: float,
) -> bool:
    if not allow and not deny:
        return True
    predicted = None
    if meta and meta.classification:
        predicted = meta.classification.predictions
    if not predicted:
        return allow is None
    if deny:
        deny_set = {_label_value(label) for label in deny}
        for entry in predicted:
            if _meets_confidence(entry.confidence, min_confidence) and (
                entry.class_name in deny_set
            ):
                return False
    if allow:
        allow_set = {_label_value(label) for label in allow}
        return any(
            _meets_confidence(entry.confidence, min_confidence)
            and entry.class_name in allow_set
            for entry in predicted
        )
    return True


def _label_value(label: Union[PictureClassificationLabel, str]) -> str:
    return label.value if isinstance(label, PictureClassificationLabel) else str(label)


def _meets_confidence(confidence: Optional[float], min_confidence: float) -> bool:
    return min_confidence <= 0 or (
        confidence is not None and confidence >= min_confidence
    )
