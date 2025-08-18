import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Generic, Optional, Protocol, Type, Union

import numpy as np
from docling_core.types.doc import BoundingBox, DocItem, DoclingDocument, NodeItem
from PIL.Image import Image
from typing_extensions import TypeVar

from docling.datamodel.base_models import (
    ItemAndImageEnrichmentElement,
    Page,
    VlmPrediction,
)
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import BaseOptions
from docling.datamodel.pipeline_options_vlm_model import (
    InlineVlmOptions,
    TransformersPromptStyle,
)
from docling.datamodel.settings import settings


class BaseModelWithOptions(Protocol):
    @classmethod
    def get_options_type(cls) -> Type[BaseOptions]: ...

    def __init__(self, *, options: BaseOptions, **kwargs): ...


class BasePageModel(ABC):
    @abstractmethod
    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        pass


class BaseVlmModel(ABC):
    """Base class for Vision-Language Models that adds image processing capability."""

    @abstractmethod
    def process_images(
        self,
        image_batch: Iterable[Union[Image, np.ndarray]],
        prompt: Union[str, list[str]],
    ) -> Iterable[VlmPrediction]:
        """Process raw images without page metadata.

        Args:
            image_batch: Iterable of PIL Images or numpy arrays
            prompt: Either:
                - str: Single prompt used for all images
                - list[str]: List of prompts (one per image, must match image count)

        Raises:
            ValueError: If prompt list length doesn't match image count.
        """


class BaseVlmPageModel(BasePageModel, BaseVlmModel):
    """Base implementation for VLM models that inherit from BasePageModel.

    Provides a default __call__ implementation that extracts images from pages,
    processes them using process_images, and attaches results back to pages.
    """

    # Type annotations for attributes that subclasses must initialize
    vlm_options: InlineVlmOptions
    processor: Any

    @abstractmethod
    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        """Extract images from pages, process them, and attach results back."""

    def formulate_prompt(self, user_prompt: str) -> str:
        """Formulate a prompt for the VLM."""
        _log = logging.getLogger(__name__)

        if self.vlm_options.transformers_prompt_style == TransformersPromptStyle.RAW:
            return user_prompt

        elif self.vlm_options.repo_id == "microsoft/Phi-4-multimodal-instruct":
            _log.debug("Using specialized prompt for Phi-4")
            # Note: This might need adjustment for VLLM vs transformers
            user_prompt_prefix = "<|user|>"
            assistant_prompt = "<|assistant|>"
            prompt_suffix = "<|end|>"

            prompt = f"{user_prompt_prefix}<|image_1|>{user_prompt}{prompt_suffix}{assistant_prompt}"
            _log.debug(f"prompt for {self.vlm_options.repo_id}: {prompt}")

            return prompt

        elif self.vlm_options.transformers_prompt_style == TransformersPromptStyle.CHAT:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "This is a page from a document.",
                        },
                        {"type": "image"},
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ]
            prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            return prompt

        raise RuntimeError(
            f"Unknown prompt style `{self.vlm_options.transformers_prompt_style}`. Valid values are {', '.join(s.value for s in TransformersPromptStyle)}."
        )


EnrichElementT = TypeVar("EnrichElementT", default=NodeItem)


class GenericEnrichmentModel(ABC, Generic[EnrichElementT]):
    elements_batch_size: int = settings.perf.elements_batch_size

    @abstractmethod
    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        pass

    @abstractmethod
    def prepare_element(
        self, conv_res: ConversionResult, element: NodeItem
    ) -> Optional[EnrichElementT]:
        pass

    @abstractmethod
    def __call__(
        self, doc: DoclingDocument, element_batch: Iterable[EnrichElementT]
    ) -> Iterable[NodeItem]:
        pass


class BaseEnrichmentModel(GenericEnrichmentModel[NodeItem]):
    def prepare_element(
        self, conv_res: ConversionResult, element: NodeItem
    ) -> Optional[NodeItem]:
        if self.is_processable(doc=conv_res.document, element=element):
            return element
        return None


class BaseItemAndImageEnrichmentModel(
    GenericEnrichmentModel[ItemAndImageEnrichmentElement]
):
    images_scale: float
    expansion_factor: float = 0.0

    def prepare_element(
        self, conv_res: ConversionResult, element: NodeItem
    ) -> Optional[ItemAndImageEnrichmentElement]:
        if not self.is_processable(doc=conv_res.document, element=element):
            return None

        assert isinstance(element, DocItem)
        element_prov = element.prov[0]

        bbox = element_prov.bbox
        width = bbox.r - bbox.l
        height = bbox.t - bbox.b

        # TODO: move to a utility in the BoundingBox class
        expanded_bbox = BoundingBox(
            l=bbox.l - width * self.expansion_factor,
            t=bbox.t + height * self.expansion_factor,
            r=bbox.r + width * self.expansion_factor,
            b=bbox.b - height * self.expansion_factor,
            coord_origin=bbox.coord_origin,
        )

        page_ix = element_prov.page_no - conv_res.pages[0].page_no - 1
        cropped_image = conv_res.pages[page_ix].get_image(
            scale=self.images_scale, cropbox=expanded_bbox
        )
        return ItemAndImageEnrichmentElement(item=element, image=cropped_image)
