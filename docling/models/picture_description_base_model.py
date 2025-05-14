import logging
from abc import abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import List, Optional, Tuple, Type, Union

from docling_core.types.doc import (
    DoclingDocument,
    NodeItem,
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

_log = logging.getLogger(__name__)


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

    def _annotate_with_context(
        self, image_context_map: Iterable[Tuple[Image.Image, str]]
    ) -> Iterable[str]:
        """Override this method to support context in concrete implementations."""
        # Extract only the images (keys) from the dict
        images = [image_context_pair[0] for image_context_pair in image_context_map]
        # Default implementation ignores context
        yield from self._annotate_images(images)

    def _get_surrounding_text(
        self, doc: DoclingDocument, picture_item: PictureItem
    ) -> str:
        """Get text context from items before and after the picture."""
        context = []
        text_items_before_picture, text_items_after_picture = [], []
        found_picture = False
        after_count = 0

        _log.debug(
            "Getting surrounding text for picture ref: %s", picture_item.self_ref
        )
        for item, _ in doc.iterate_items():
            if item == picture_item:
                found_picture = True
                continue

            if not found_picture:  # before picture
                if isinstance(item, (str, NodeItem)):
                    text = item if isinstance(item, str) else getattr(item, "text", "")
                    if text and text.strip():
                        # hold all text items before the picture
                        text_items_before_picture.append(text)
            else:  # after picture
                if (
                    isinstance(item, (str, NodeItem))
                    and after_count
                    < self.options.text_context_window_size_after_picture
                ):
                    text = item if isinstance(item, str) else getattr(item, "text", "")
                    if text and text.strip():
                        text_items_after_picture.append(text)
                        after_count += 1

            if after_count >= self.options.text_context_window_size_after_picture:
                # Stop if we have reached the limit of text items after the picture
                break

        # Combine text items before and after the picture
        if self.options.text_context_window_size_before_picture > 0:
            # get only the last N text items before the picture
            context.extend(
                text_items_before_picture[
                    -self.options.text_context_window_size_before_picture :
                ]
            )

        if self.options.text_context_window_size_after_picture > 0:
            context.extend(text_items_after_picture)

        _log.debug("Context before picture: %s", text_items_before_picture)
        _log.debug("Context after picture: %s", text_items_after_picture)
        # Join the context with newlines
        return "\n".join(context)

    def __call__(
        self,
        doc: DoclingDocument,
        element_batch: Iterable[ItemAndImageEnrichmentElement],
    ) -> Iterable[NodeItem]:
        if not self.enabled:
            for element in element_batch:
                yield element.item
            return

        image_context_map: List[Tuple[Image.Image, str]] = []
        pictures: List[PictureItem] = []

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
                pictures.append(el.item)
                context = ""
                if (
                    self.options.text_context_window_size_before_picture > 0
                    or self.options.text_context_window_size_after_picture > 0
                ):
                    # Get the surrounding text context
                    context = self._get_surrounding_text(doc, el.item)
                image_context_map.append((el.image, context))

        if (
            self.options.text_context_window_size_before_picture > 0
            or self.options.text_context_window_size_after_picture > 0
        ):
            picture_descriptions = self._annotate_with_context(image_context_map)
        else:
            picture_descriptions = self._annotate_images(
                image for image, _ in image_context_map
            )

        for picture, description in zip(pictures, picture_descriptions):
            picture.annotations.append(
                PictureDescriptionData(text=description, provenance=self.provenance)
            )
            yield picture

    @classmethod
    @abstractmethod
    def get_options_type(cls) -> Type[PictureDescriptionBaseOptions]:
        pass
