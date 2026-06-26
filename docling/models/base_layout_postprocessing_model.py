from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Type

from docling.datamodel.base_models import LayoutPrediction, Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import BaseLayoutPostprocessorOptions
from docling.models.base_model import BaseModelWithOptions, BasePageModel


class BaseLayoutPostprocessingModel(BasePageModel, BaseModelWithOptions, ABC):
    """Shared interface for layout post-processing stages."""

    @classmethod
    @abstractmethod
    def get_options_type(cls) -> Type[BaseLayoutPostprocessorOptions]:
        """Return the options type supported by this post-processing model."""

    @abstractmethod
    def postprocess_layout(
        self,
        conv_res: ConversionResult,
        pages: Sequence[Page],
    ) -> Sequence[LayoutPrediction]:
        """Finalize raw layout predictions for the provided pages."""

    def __call__(
        self,
        conv_res: ConversionResult,
        page_batch: Iterable[Page],
    ) -> Iterable[Page]:
        pages = list(page_batch)
        predictions = self.postprocess_layout(conv_res, pages)

        for page, prediction in zip(pages, predictions):
            page.predictions.layout = prediction
            yield page
