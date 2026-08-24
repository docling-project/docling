"""Layout detection stage backed by object-detection runtimes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

from docling_core.types.doc import BoundingBox, CoordOrigin, DocItemLabel
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Cluster, LayoutPrediction, Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import LayoutObjectDetectionOptions
from docling.datamodel.settings import settings
from docling.models.base_layout_model import BaseLayoutModel
from docling.models.inference_engines.object_detection import (
    BaseObjectDetectionEngine,
    ObjectDetectionEngineInput,
    ObjectDetectionEngineOutput,
    create_object_detection_engine,
)
from docling.utils.profiling import TimeRecorder
from docling.utils.visualization import draw_clusters_and_cells_side_by_side

_log = logging.getLogger(__name__)


class LayoutObjectDetectionModel(BaseLayoutModel):
    """Layout detection using the generic object-detection inference engines."""

    def __init__(
        self,
        artifacts_path: Optional[Path],
        accelerator_options: AcceleratorOptions,
        options: LayoutObjectDetectionOptions,
        enable_remote_services: bool = False,
    ) -> None:
        self.options = options

        self.engine: BaseObjectDetectionEngine = create_object_detection_engine(
            options=options.engine_options,
            model_spec=self.options.model_spec,
            artifacts_path=artifacts_path,
            enable_remote_services=enable_remote_services,
            accelerator_options=accelerator_options,
        )
        self.engine.initialize()

        # Convert engine's string labels to DocItemLabel enums
        self._label_map = self._build_label_map()
        self._unmapped_label_ids: Set[int] = set()

    def _build_label_map(self) -> Dict[int, DocItemLabel]:
        """Build label mapping from engine's label names to DocItemLabel enums.

        Raises:
            RuntimeError: If labels don't match DocItemLabel enum.
        """
        id_to_label_str = self.engine.get_label_mapping()
        label_map = {}

        for label_id, label_name in id_to_label_str.items():
            # Normalize to the DocItemLabel enum member convention: uppercase with
            # underscores. Model configs emit hyphenated or spaced names such as
            # "List-item", which .upper() alone leaves as "LIST-ITEM" and fails the
            # enum name lookup (the member is LIST_ITEM).
            label_enum_name = label_name.upper().replace("-", "_").replace(" ", "_")
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

        # Collect the pages that can actually be run, keeping their position so
        # results can be zipped back onto the original page order.
        batch: List[ObjectDetectionEngineInput] = []
        batch_indices: List[int] = []
        for idx, page in enumerate(pages):
            assert page._backend is not None
            if not page._backend.is_valid():
                continue

            page_image = page.get_image(scale=1.0)
            if page_image is None:
                continue

            batch.append(
                ObjectDetectionEngineInput(
                    image=page_image,
                    metadata={"page_no": page.page_no},
                )
            )
            batch_indices.append(idx)

        engine_outputs: List[ObjectDetectionEngineOutput] = []
        if batch:
            with TimeRecorder(conv_res, "layout"):
                engine_outputs = self.engine.predict_batch(batch)

        predictions: List[LayoutPrediction] = [
            page.predictions.layout or LayoutPrediction() for page in pages
        ]
        for idx, engine_input, engine_output in zip(
            batch_indices, batch, engine_outputs
        ):
            page = pages[idx]
            clusters = self._predictions_to_clusters(
                page=page,
                image=engine_input.image,
                engine_output=engine_output,
            )

            if settings.debug.visualize_raw_layout:
                draw_clusters_and_cells_side_by_side(
                    conv_res.input.file, page, clusters, mode_prefix="raw"
                )

            # Emit raw clusters; post-processing and layout_score are
            # handled by the downstream LayoutPostprocessingModel stage.
            predictions[idx] = LayoutPrediction(clusters=clusters)

        for page, prediction in zip(pages, predictions):
            page.predictions.layout = prediction

        return predictions

    def _predictions_to_clusters(
        self,
        page: Page,
        image: Image.Image,
        engine_output: ObjectDetectionEngineOutput,
    ) -> List[Cluster]:
        assert page.size is not None
        page_width = page.size.width
        page_height = page.size.height
        scale_x = page_width / image.width
        scale_y = page_height / image.height

        clusters: List[Cluster] = []
        for idx, (label_id, score, bbox_coords) in enumerate(
            zip(engine_output.label_ids, engine_output.scores, engine_output.bboxes)
        ):
            label = self._label_map.get(label_id)
            if label is None:
                if label_id not in self._unmapped_label_ids:
                    self._unmapped_label_ids.add(label_id)
                    _log.warning(
                        "Dropping detections with label id %s: the model emitted an "
                        "id that is absent from its own config id2label map.",
                        label_id,
                    )
                continue

            # Detections can overshoot the page; downstream area ratios in
            # LayoutPostprocessor assume page-bounded boxes.
            bbox = BoundingBox(
                l=min(max(bbox_coords[0] * scale_x, 0.0), page_width),
                t=min(max(bbox_coords[1] * scale_y, 0.0), page_height),
                r=min(max(bbox_coords[2] * scale_x, 0.0), page_width),
                b=min(max(bbox_coords[3] * scale_y, 0.0), page_height),
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
