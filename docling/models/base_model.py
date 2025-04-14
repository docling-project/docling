from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Generic, Optional, Protocol, Type

from docling_core.types.doc import BoundingBox, DocItem, DoclingDocument, NodeItem
from typing_extensions import TypeVar

from docling.datamodel.base_models import ItemAndImageEnrichmentElement, Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import BaseOptions
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

        page_ix = element_prov.page_no - 1
        cropped_image = conv_res.pages[page_ix].get_image(
            scale=self.images_scale, cropbox=expanded_bbox
        )
        return ItemAndImageEnrichmentElement(item=element, image=cropped_image)
