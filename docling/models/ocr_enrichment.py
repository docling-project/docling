from collections.abc import Iterable
from pathlib import Path
from typing import List, Optional, Type, Union

from docling_core.types.doc import (
    DoclingDocument,
    NodeItem,
    PictureItem,
)
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import (
    OcrOptions,
)
from docling.models.base_model import (
    BaseItemAndImageEnrichmentModel,
    ItemAndImageEnrichmentElement,
)
from docling.models.base_ocr_model import BaseOcrModel
from docling.models.factories import get_ocr_factory


class OcrEnrichmentModel(BaseItemAndImageEnrichmentModel):
    images_scale: float = 2.0

    def __init__(
        self,
        *,
        enabled: bool,
        artifacts_path: Optional[Union[Path, str]],
        options: OcrOptions,
        accelerator_options: AcceleratorOptions,
        allow_external_plugins: bool,
    ):
        self.enabled = enabled
        self.options = options

        self._ocr_model: BaseOcrModel

        if self.enabled:
            ocr_factory = get_ocr_factory(allow_external_plugins=allow_external_plugins)
            self._ocr_model = ocr_factory.create_instance(
                options=self.options,
                enabled=True,
                artifacts_path=artifacts_path,
                accelerator_options=accelerator_options,
            )

    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        return self.enabled and isinstance(element, PictureItem)

    def __call__(
        self,
        doc: DoclingDocument,
        element_batch: Iterable[ItemAndImageEnrichmentElement],
    ) -> Iterable[NodeItem]:
        if not self.enabled:
            for element in element_batch:
                yield element.item
            return

        # TODO: call self._ocr_model
        for element in element_batch:
            yield element.item
