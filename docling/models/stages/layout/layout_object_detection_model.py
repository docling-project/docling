"""Layout detection stage backed by object-detection runtimes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from docling_core.types.doc import CoordOrigin, DocItemLabel
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import BoundingBox, Cluster, LayoutPrediction, Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import LayoutObjectDetectionOptions
from docling.models.base_layout_model import BaseLayoutModel
from docling.models.inference_engines.object_detection import (
    BaseObjectDetectionEngine,
    ObjectDetectionEngineInput,
    ObjectDetectionEngineOutput,
    create_object_detection_engine,
)
from docling.utils.layout_postprocessor import LayoutPostprocessor
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)


class LayoutObjectDetectionModel(BaseLayoutModel):
    """Layout detection using the generic object-detection inference engines."""

    def __init__(
        self,
        artifacts_path: Optional[Path],
        accelerator_options: AcceleratorOptions,
        options: LayoutObjectDetectionOptions,
    ) -> None:
        self.options = options

        self.engine: BaseObjectDetectionEngine = create_object_detection_engine(
            options=options.engine_options,
            model_spec=self.options.model_spec,
            artifacts_path=artifacts_path,
            accelerator_options=accelerator_options,
        )
        self.engine.initialize()

        # Convert engine's string labels to DocItemLabel enums
        self._label_map = self._build_label_map()

    def _build_label_map(self) -> Dict[int, DocItemLabel]:
        """Build label mapping from engine's label names to DocItemLabel enums.

        Raises:
            RuntimeError: If labels don't match DocItemLabel enum.
        """
        id_to_label_str = self.engine.get_label_mapping()
        label_map = {}

        for label_id, label_name in id_to_label_str.items():
            # Convert label name to uppercase to match DocItemLabel enum convention
            label_enum_name = label_name.upper()
            try:
                label_map[label_id] = DocItemLabel[label_enum_name]
            except KeyError:
                raise RuntimeError(
                    f"Label '{label_name}' (ID {label_id}) from model config "
                    f"does not match any DocItemLabel enum value."
                )

        return label_map

    @classmethod
    def get_options_type(cls) -> type[LayoutObjectDetectionOptions]:
        return LayoutObjectDetectionOptions

    def predict_layout(
        self,
        conv_res: ConversionResult,
        pages: Sequence[Page],
    ) -> Sequence[LayoutPrediction]:
        pages = list(pages)
        predictions: list[LayoutPrediction] = []

        for page in pages:
            assert page._backend is not None
            if not page._backend.is_valid():
                existing_prediction = page.predictions.layout or LayoutPrediction()
                page.predictions.layout = existing_prediction
                predictions.append(existing_prediction)
                continue

            page_image = page.get_image(scale=1.0)
            if page_image is None:
                empty_prediction = page.predictions.layout or LayoutPrediction()
                page.predictions.layout = empty_prediction
                predictions.append(empty_prediction)
                continue

            with TimeRecorder(conv_res, "layout"):
                engine_input = ObjectDetectionEngineInput(
                    image=page_image,
                    metadata={"page_no": page.page_no},
                )
                engine_output = self.engine.predict(engine_input)

                clusters = self._predictions_to_clusters(
                    page=page,
                    image=page_image,
                    engine_output=engine_output,
                )

                processed_clusters, processed_cells = LayoutPostprocessor(
                    page=page,
                    clusters=clusters,
                    options=self.options,
                ).postprocess()

                layout_prediction = LayoutPrediction(clusters=processed_clusters)
                page.predictions.layout = layout_prediction

                if processed_clusters:
                    layout_scores = [c.confidence for c in processed_clusters]
                    conv_res.confidence.pages[page.page_no].layout_score = float(
                        np.mean(layout_scores)
                    )
                else:
                    conv_res.confidence.pages[page.page_no].layout_score = 0.0

                if processed_cells:
                    ocr_scores = [c.confidence for c in processed_cells if c.from_ocr]
                    if ocr_scores:
                        conv_res.confidence.pages[page.page_no].ocr_score = float(
                            np.mean(ocr_scores)
                        )

                predictions.append(layout_prediction)

        return predictions

    def _predictions_to_clusters(
        self,
        page: Page,
        image: Image.Image,
        engine_output: ObjectDetectionEngineOutput,
    ) -> List[Cluster]:
        assert page.size is not None
        scale_x = page.size.width / image.width
        scale_y = page.size.height / image.height

        clusters: List[Cluster] = []
        for idx, (label_id, score, bbox_coords) in enumerate(
            zip(engine_output.label_ids, engine_output.scores, engine_output.bboxes)
        ):
            label = self._label_map.get(label_id, DocItemLabel.TEXT)
            bbox = BoundingBox(
                l=bbox_coords[0] * scale_x,
                t=bbox_coords[1] * scale_y,
                r=bbox_coords[2] * scale_x,
                b=bbox_coords[3] * scale_y,
                coord_origin=CoordOrigin.TOPLEFT,
            )
            clusters.append(
                Cluster(
                    id=idx,
                    label=label,
                    confidence=score,
                    bbox=bbox,
                    cells=[],
                )
            )
        return clusters
