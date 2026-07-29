from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Type

from docling_core.types.doc.page import TextCell
from PIL import Image, ImageDraw

from docling.datamodel.base_models import Cluster, Page, TableStructurePrediction
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import BaseTableStructureOptions
from docling.models.base_model import BaseModelWithOptions, BasePageModel


class BaseTableStructureModel(BasePageModel, BaseModelWithOptions, ABC):
    """Shared interface for table structure models."""

    enabled: bool
    options: BaseTableStructureOptions
    scale: float

    def _prepare_table_input(
        self,
        page_image: Image.Image,
        *,
        table_cluster: Cluster,
        table_clusters: Sequence[Cluster],
        text_cells: Sequence[TextCell],
    ) -> tuple[Image.Image, list[TextCell], list[Cluster]]:
        coverage_threshold = self.options.rich_cell_element_coverage_threshold
        nested_tables = [
            candidate
            for candidate in table_clusters
            if candidate.id != table_cluster.id
            and candidate.bbox.area() < table_cluster.bbox.area()
            and candidate.bbox.intersection_over_self(table_cluster.bbox)
            >= coverage_threshold
        ]
        if not nested_tables:
            return page_image, list(text_cells), []

        masked_image = page_image.copy()
        draw = ImageDraw.Draw(masked_image)
        for nested_table in nested_tables:
            draw.rectangle(
                nested_table.bbox.scaled(scale=self.scale).as_tuple(),
                fill="white",
            )

        parent_cells = [
            cell
            for cell in text_cells
            if not any(
                cell.rect.to_bounding_box().intersection_over_self(nested_table.bbox)
                >= coverage_threshold
                for nested_table in nested_tables
            )
        ]
        return masked_image, parent_cells, nested_tables

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
