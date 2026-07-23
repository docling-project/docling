"""Standalone layout post-processing stage.

Runs after the layout (prediction) stage. It always computes the page
``layout_score`` and, unless disabled, finalizes the raw clusters with
``LayoutPostprocessor`` (cell assignment, empty-cluster handling, etc.).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import List, Type

import numpy as np

from docling.datamodel.base_models import Cluster, LayoutPrediction, Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import LayoutPostprocessorOptions
from docling.datamodel.settings import settings
from docling.models.base_layout_postprocessing_model import (
    BaseLayoutPostprocessingModel,
)
from docling.utils.layout_postprocessor import LayoutPostprocessor
from docling.utils.profiling import TimeRecorder
from docling.utils.visualization import draw_clusters_and_cells_side_by_side


class LayoutPostprocessingModel(BaseLayoutPostprocessingModel):
    """Finalize raw layout clusters and set the page ``layout_score``."""

    def __init__(self, *, options: LayoutPostprocessorOptions) -> None:
        self.options = options

    @classmethod
    def get_options_type(cls) -> Type[LayoutPostprocessorOptions]:
        return LayoutPostprocessorOptions

    def postprocess_layout(
        self,
        conv_res: ConversionResult,
        pages: Sequence[Page],
    ) -> Sequence[LayoutPrediction]:
        predictions: list[LayoutPrediction] = []

        for page in pages:
            # Invalid / missing prediction: carry through without scoring.
            if (
                page._backend is None
                or not page._backend.is_valid()
                or page.predictions.layout is None
            ):
                predictions.append(page.predictions.layout or LayoutPrediction())
                continue

            with TimeRecorder(conv_res, "layout_postprocess"):
                raw_clusters = page.predictions.layout.clusters

                if self.options.run_postprocessor:
                    processed_clusters = LayoutPostprocessor(
                        page, raw_clusters, self.options
                    ).postprocess()
                    # Note: LayoutPostprocessor updates page.cells and
                    # page.parsed_page internally.
                    prediction = LayoutPrediction(clusters=processed_clusters)
                    clusters_for_score = processed_clusters

                    if settings.debug.visualize_layout:
                        draw_clusters_and_cells_side_by_side(
                            conv_res.input.file,
                            page,
                            processed_clusters,
                            mode_prefix="postprocessed",
                        )
                else:
                    prediction = page.predictions.layout
                    clusters_for_score = raw_clusters

                self._update_layout_score(conv_res, page, clusters_for_score)

            predictions.append(prediction)

        return predictions

    @staticmethod
    def _update_layout_score(
        conv_res: ConversionResult,
        page: Page,
        clusters: List[Cluster],
    ) -> None:
        conv_res.confidence.pages[page.page_no].layout_score = (
            float(np.mean([c.confidence for c in clusters])) if clusters else 0.0
        )
