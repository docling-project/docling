import logging
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from docling_core.types.doc import BoundingBox, DocItemLabel
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Cluster, LayoutPrediction, Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.layout_model_specs import DOCLING_LAYOUT_V2, LayoutModelConfig
from docling.datamodel.pipeline_options import LayoutOptions
from docling.datamodel.settings import settings
from docling.models.base_layout_model import BaseLayoutModel
from docling.models.utils.hf_model_download import download_hf_model
from docling.utils.accelerator_utils import decide_device
from docling.utils.profiling import TimeRecorder
from docling.utils.visualization import draw_clusters_and_cells_side_by_side

_log = logging.getLogger(__name__)


class LayoutModel(BaseLayoutModel):
    TEXT_ELEM_LABELS = [
        DocItemLabel.TEXT,
        DocItemLabel.FOOTNOTE,
        DocItemLabel.CAPTION,
        DocItemLabel.CHECKBOX_UNSELECTED,
        DocItemLabel.CHECKBOX_SELECTED,
        DocItemLabel.SECTION_HEADER,
        DocItemLabel.PAGE_HEADER,
        DocItemLabel.PAGE_FOOTER,
        DocItemLabel.CODE,
        DocItemLabel.LIST_ITEM,
        DocItemLabel.FORMULA,
    ]
    PAGE_HEADER_LABELS = [DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER]

    TABLE_LABELS = [DocItemLabel.TABLE, DocItemLabel.DOCUMENT_INDEX]
    FIGURE_LABEL = DocItemLabel.PICTURE
    FORMULA_LABEL = DocItemLabel.FORMULA
    CONTAINER_LABELS = [DocItemLabel.FORM, DocItemLabel.KEY_VALUE_REGION]

    def __init__(
        self,
        artifacts_path: Optional[Path],
        accelerator_options: AcceleratorOptions,
        options: LayoutOptions,
        enable_remote_services: bool = False,
    ):
        from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor

        _ = enable_remote_services
        self.options = options

        device = decide_device(accelerator_options.device)
        layout_model_config = options.model_spec
        model_repo_folder = layout_model_config.model_repo_folder
        model_path = layout_model_config.model_path

        if artifacts_path is None:
            artifacts_path = (
                self.download_models(layout_model_config=layout_model_config)
                / model_path
            )
        else:
            if (artifacts_path / model_repo_folder).exists():
                artifacts_path = artifacts_path / model_repo_folder / model_path
            elif (artifacts_path / model_path).exists():
                warnings.warn(
                    "The usage of artifacts_path containing directly "
                    f"{model_path} is deprecated. Please point "
                    "the artifacts_path to the parent containing "
                    f"the {model_repo_folder} folder.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                artifacts_path = artifacts_path / model_path

        self.layout_predictor = LayoutPredictor(
            artifact_path=str(artifacts_path),
            device=device,
            num_threads=accelerator_options.num_threads,
        )

    @classmethod
    def get_options_type(cls) -> type[LayoutOptions]:
        return LayoutOptions

    @staticmethod
    def download_models(
        local_dir: Optional[Path] = None,
        force: bool = False,
        progress: bool = False,
        layout_model_config: LayoutModelConfig = LayoutOptions().model_spec,  # use default
    ) -> Path:
        return download_hf_model(
            repo_id=layout_model_config.repo_id,
            revision=layout_model_config.revision,
            local_dir=local_dir,
            force=force,
            progress=progress,
        )

    def predict_layout(
        self,
        conv_res: ConversionResult,
        pages: Sequence[Page],
    ) -> Sequence[LayoutPrediction]:
        # Convert to list to ensure predictable iteration
        pages = list(pages)

        # Separate valid and invalid pages
        valid_pages = []
        valid_page_images: List[Union[Image.Image, np.ndarray]] = []

        for page in pages:
            assert page._backend is not None
            if not page._backend.is_valid():
                continue

            assert page.size is not None
            page_image = page.get_image(scale=1.0)
            assert page_image is not None

            valid_pages.append(page)
            valid_page_images.append(page_image)

        # Process all valid pages with batch prediction
        batch_predictions = []
        if valid_page_images:
            with TimeRecorder(conv_res, "layout"):
                batch_predictions = self.layout_predictor.predict_batch(  # type: ignore[attr-defined]
                    valid_page_images
                )

        # Process each page with its predictions
        layout_predictions: list[LayoutPrediction] = []
        valid_page_idx = 0
        for page in pages:
            assert page._backend is not None
            if not page._backend.is_valid():
                existing_prediction = page.predictions.layout or LayoutPrediction()
                page.predictions.layout = existing_prediction
                layout_predictions.append(existing_prediction)
                continue

            page_predictions = batch_predictions[valid_page_idx]
            valid_page_idx += 1

            clusters = []
            for ix, pred_item in enumerate(page_predictions):
                label = DocItemLabel(
                    pred_item["label"].lower().replace(" ", "_").replace("-", "_")
                )  # Temporary, until docling-ibm-model uses docling-core types
                cluster = Cluster(
                    id=ix,
                    label=label,
                    confidence=pred_item["confidence"],
                    bbox=BoundingBox.model_validate(pred_item),
                    cells=[],
                )
                clusters.append(cluster)

            if settings.debug.visualize_raw_layout:
                draw_clusters_and_cells_side_by_side(
                    conv_res.input.file, page, clusters, mode_prefix="raw"
                )

            # Emit raw clusters; post-processing (cell assignment, layout_score)
            # is performed by the downstream LayoutPostprocessingModel stage.
            prediction = LayoutPrediction(clusters=clusters)
            page.predictions.layout = prediction

            layout_predictions.append(prediction)

        return layout_predictions
