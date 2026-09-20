# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import threading
from abc import abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any, List, Optional, Type, Union

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
from docling.datamodel.base_models import (
    ApiImageRequestResult,
    DoclingComponentType,
    ErrorItem,
    FailureCategory,
    VlmStopReason,
)
from docling.datamodel.pipeline_options import (
    PictureDescriptionBaseOptions,
)
from docling.models.base_model import (
    BaseItemAndImageEnrichmentModel,
    BaseModelWithOptions,
    ItemAndImageEnrichmentElement,
)

_USAGE_META_NAMESPACE = "docling"
_USAGE_META_FIELD_NAME = "usage"


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
        if options.batch_size < 1:
            raise ValueError("Picture description batch_size must be >= 1")
        if options.scale <= 0:
            raise ValueError("Picture description scale must be > 0")

        self.enabled = enabled
        self.options = options
        self.provenance = "not-implemented"
        self.elements_batch_size = options.batch_size
        self.images_scale = options.scale
        # Failed requests are handed to the pipeline through collect_errors().
        # Kept per thread: pipelines share model instances, and conversions
        # running concurrently must not collect each other's failures.
        self._failures = threading.local()

    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        return self.enabled and isinstance(element, PictureItem)

    def _annotate_images(
        self, images: Iterable[Image.Image]
    ) -> Iterable[str | ApiImageRequestResult]:
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
                images.append(el.image.convert("RGB"))

        outputs = self._annotate_images(images)

        for item, output in zip(elements, outputs):
            if _is_failed_request(output):
                # No description was produced: report the failure to the
                # pipeline instead of storing an empty text as if it were one.
                self._pending_errors().append(self._failure_item(item, output))
                yield item
                continue
            description_text, usage = _normalize_description_output(output)
            # FIXME: annotations is deprecated, remove once all consumers use meta.classification
            if self.options._keep_deprecated_annotations:
                item.annotations.append(
                    PictureDescriptionData(
                        text=description_text, provenance=self.provenance
                    )
                )

            # Store description in the new meta field
            if item.meta is None:
                item.meta = PictureMeta()
            item.meta.description = DescriptionMetaField(
                text=description_text,
                created_by=self.provenance,
            )
            if usage is not None:
                item.meta.description.set_custom_field(
                    namespace=_USAGE_META_NAMESPACE,
                    name=_USAGE_META_FIELD_NAME,
                    value=usage,
                )

            yield item

    def collect_errors(self) -> List[ErrorItem]:
        errors = self._pending_errors()
        self._failures.errors = []
        return errors

    def _pending_errors(self) -> List[ErrorItem]:
        errors: Optional[List[ErrorItem]] = getattr(self._failures, "errors", None)
        if errors is None:
            errors = self._failures.errors = []
        return errors

    def _failure_item(
        self, item: PictureItem, output: ApiImageRequestResult
    ) -> ErrorItem:
        return ErrorItem(
            component_type=DoclingComponentType.MODEL,
            module_name=type(self).__name__,
            error_message="Picture description failed: "
            f"{output.error or 'unknown error'}.",
            category=FailureCategory.INFERENCE_FAILURE,
            page_no=item.prov[0].page_no if item.prov else None,
        )

    @classmethod
    @abstractmethod
    def get_options_type(cls) -> Type[PictureDescriptionBaseOptions]:
        pass


def _is_failed_request(output: str | ApiImageRequestResult) -> bool:
    return (
        isinstance(output, ApiImageRequestResult)
        and output.stop_reason == VlmStopReason.INFERENCE_ERROR
    )


def _normalize_description_output(
    output: str | ApiImageRequestResult,
) -> tuple[str, Any | None]:
    if isinstance(output, ApiImageRequestResult):
        return output.text, output.usage
    return output, None


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
