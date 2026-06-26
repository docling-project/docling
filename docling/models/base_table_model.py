from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Type

from docling_core.types.doc import DocItemLabel

from docling.datamodel.base_models import Page, TableStructurePrediction
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import BaseTableStructureOptions
from docling.models.base_model import BaseModelWithOptions, BasePageModel


def table_candidate_labels(try_table_on_picture: bool) -> list[DocItemLabel]:
    """Labels a table backend runs structure recognition on.

    Pictures are included only when ``try_table_on_picture`` is set, to recover
    tables the layout model mislabels as images when their rows embed icons (#3410).
    """
    labels = [DocItemLabel.TABLE, DocItemLabel.DOCUMENT_INDEX]
    if try_table_on_picture:
        labels.append(DocItemLabel.PICTURE)
    return labels


def is_table_like(num_rows: int, num_cols: int) -> bool:
    """Whether a prediction is solid enough to promote a picture to a table.

    Requires a real grid (two or more rows and columns) so an image is not
    relabeled a table just because the model forced some structure onto it.
    """
    return num_rows >= 2 and num_cols >= 2


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
