from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Type

from docling.datamodel.base_models import Page, TableStructurePrediction
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import BaseTableStructureOptions
from docling.models.base_model import BaseModelWithOptions, BasePageModel


class BaseTableStructureModel(BasePageModel, BaseModelWithOptions, ABC):
    """Shared interface for table structure models."""

    enabled: bool

    @classmethod
    @abstractmethod
    def get_options_type(cls) -> Type[BaseTableStructureOptions]:
        """Return the options type supported by this table model."""

    @abstractmethod
    def predict_tables(
        self,
        conv_res: ConversionResult,
        pages: Sequence[Page],
    ) -> Sequence[TableStructurePrediction]:
        """Produce table structure predictions for the provided pages."""

    def __call__(
        self,
        conv_res: ConversionResult,
        page_batch: Iterable[Page],
    ) -> Iterable[Page]:
        if not getattr(self, "enabled", True):
            yield from page_batch
            return

        pages = list(page_batch)
        predictions = self.predict_tables(conv_res, pages)

        for page, prediction in zip(pages, predictions):
            page.predictions.tablestructure = prediction
            yield page
