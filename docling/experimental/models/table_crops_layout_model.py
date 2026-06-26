"""Internal TableCrops layout model that marks full pages as table clusters."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from docling_core.types.doc import BoundingBox, DocItemLabel

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Cluster, LayoutPrediction, Page
from docling.datamodel.document import ConversionResult
from docling.experimental.datamodel.table_crops_layout_options import (
    TableCropsLayoutOptions,
)
from docling.models.base_layout_model import BaseLayoutModel

__all__ = ["TableCropsLayoutModel"]


class TableCropsLayoutModel(BaseLayoutModel):
    """Experimental layout model that treats the full page as a table cluster.
    This is useful in cases where a Docling pipeline is applied to images of table crops only.

    This model is internal and not part of the stable public interface.
    """

    def __init__(
        self,
        artifacts_path: Optional[Path],
        accelerator_options: AcceleratorOptions,
        options: TableCropsLayoutOptions,
        enable_remote_services: bool = False,
    ):
        _ = enable_remote_services
        self.options = options
        self.artifacts_path = artifacts_path
        self.accelerator_options = accelerator_options

    @classmethod
    def get_options_type(cls) -> type[TableCropsLayoutOptions]:
        return TableCropsLayoutOptions

    def predict_layout(
        self,
        conv_res: ConversionResult,
        pages: Sequence[Page],
    ) -> Sequence[LayoutPrediction]:
        layout_predictions: list[LayoutPrediction] = []

        for page in pages:
            if page._backend is None or not page._backend.is_valid():
                existing_prediction = page.predictions.layout or LayoutPrediction()
                layout_predictions.append(existing_prediction)
                continue

            clusters = self._build_page_clusters(page)
            prediction = LayoutPrediction(clusters=clusters)

            self._update_confidence(conv_res, page, clusters)

            layout_predictions.append(prediction)

        return layout_predictions

    def _build_page_clusters(self, page: Page) -> list[Cluster]:
        page_size = page.size
        if page_size is None:
            return []

        bbox = BoundingBox(
            l=0.0,
            t=0.0,
            r=page_size.width,
            b=page_size.height,
        )

        cluster = Cluster(
            id=0,
            label=DocItemLabel.TABLE,
            bbox=bbox,
            confidence=1.0,
            cells=[],
        )

        clusters = [cluster]

        if not self.options.skip_cell_assignment:
            page_cells = list(page.cells)
            cluster.cells = page_cells

            if not page_cells and not self.options.keep_empty_clusters:
                clusters = []

        return clusters

    def _update_confidence(
        self, conv_res: ConversionResult, page: Page, clusters: list[Cluster]
    ) -> None:
        """Populate the layout confidence score for the page."""
        conv_res.confidence.pages[page.page_no].layout_score = 1.0
