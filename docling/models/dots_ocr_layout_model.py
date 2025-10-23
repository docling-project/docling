import io
import base64
import requests
import copy
import logging
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

import numpy as np
from docling_core.types.doc import DocItemLabel
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import BoundingBox, Cluster, LayoutPrediction, Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.layout_model_specs import DOCLING_LAYOUT_V2, LayoutModelConfig
from docling.datamodel.pipeline_options import LayoutOptions
from docling.datamodel.settings import settings
from docling.models.base_model import BasePageModel
from docling.models.utils.hf_model_download import download_hf_model
from docling.utils.accelerator_utils import decide_device

# from docling.utils.layout_postprocessor import LayoutPostprocessor
from docling.utils.dotsocr_postprocessor import LayoutPostprocessor
from docling.utils.profiling import TimeRecorder
from docling.utils.visualization import draw_clusters
from docling.datamodel.pipeline_options import PdfPipelineOptions


import requests

_log = logging.getLogger(__name__)


class DotsOCRLayoutModel(BasePageModel):
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

    def __init__(self, pipeline_options: PdfPipelineOptions) -> None:
        self.request = "http://127.0.0.1:8000/api/get_reading_order/"
        self.pipeline_options = pipeline_options
        self.options = pipeline_options.layout_options

    def draw_clusters_and_cells_side_by_side(
        self, conv_res, page, clusters, mode_prefix: str, show: bool = False
    ):
        """
        Draws a page image side by side with clusters filtered into two categories:
        - Left: Clusters excluding FORM, KEY_VALUE_REGION, and PICTURE.
        - Right: Clusters including FORM, KEY_VALUE_REGION, and PICTURE.
        Includes label names and confidence scores for each cluster.
        """
        scale_x = page.image.width / page.size.width
        scale_y = page.image.height / page.size.height

        # Filter clusters for left and right images
        exclude_labels = {
            DocItemLabel.FORM,
            DocItemLabel.KEY_VALUE_REGION,
            DocItemLabel.PICTURE,
        }
        left_clusters = [c for c in clusters if c.label not in exclude_labels]
        right_clusters = [c for c in clusters if c.label in exclude_labels]
        # Create a deep copy of the original image for both sides
        left_image = copy.deepcopy(page.image)
        right_image = copy.deepcopy(page.image)

        # Draw clusters on both images
        draw_clusters(left_image, left_clusters, scale_x, scale_y)
        draw_clusters(right_image, right_clusters, scale_x, scale_y)
        # Combine the images side by side
        combined_width = left_image.width * 2
        combined_height = left_image.height
        combined_image = Image.new("RGB", (combined_width, combined_height))
        combined_image.paste(left_image, (0, 0))
        combined_image.paste(right_image, (left_image.width, 0))
        if show:
            combined_image.show()
        else:
            out_path: Path = (
                Path(settings.debug.debug_output_path)
                / f"debug_{conv_res.input.file.stem}"
            )
            out_path.mkdir(parents=True, exist_ok=True)
            out_file = out_path / f"{mode_prefix}_layout_page_{page.page_no:05}.png"
            combined_image.save(str(out_file), format="png")

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        """
        여기서 conv_res.pages[0].predictions.layout.clusters 를 만듬
        나중에는
        conv_res.pages[0].predictions.layout.clusters -> conv_res.pages[0].assembled
        conv_res.pages[0].assembled -> conv_res.document
        """

        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
            else:
                with TimeRecorder(conv_res, "layout"):
                    assert page.size is not None
                    page_image = page.get_image(
                        scale=self.pipeline_options.images_scale
                    )
                    assert page_image is not None

                    buffer = io.BytesIO()
                    page_image.save(
                        buffer, format="PNG"
                    )  # PNG 형식으로 저장 (필요에 따라 JPEG 등 변경 가능)
                    buffer.seek(0)

                    # 바이트 스트림을 base64로 인코딩
                    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    response = requests.post(
                        url=self.request, json={"img": base64_image}
                    )
                    response = response.json()
                    result = response["result"]

                    clusters = []
                    for idx, pred_item in enumerate(result):
                        label = DocItemLabel(
                            pred_item["category"]
                            .lower()
                            .replace(" ", "_")
                            .replace("-", "_")
                        )  # Temporary, until docling-ibm-model uses docling-core types

                        bbox = {
                            "l": pred_item["bbox"][0],
                            "t": pred_item["bbox"][1],
                            "r": pred_item["bbox"][2],
                            "b": pred_item["bbox"][3],
                        }
                        cluster = Cluster(
                            id=idx,
                            label=label,
                            # confidence=pred_item["confidence"],
                            confidence=1.0,
                            # bbox=BoundingBox.model_validate(pred_item),
                            bbox=BoundingBox.model_validate(bbox),
                            cells=[],
                        )
                        clusters.append(cluster)

                    if settings.debug.visualize_raw_layout:
                        self.draw_clusters_and_cells_side_by_side(
                            conv_res, page, clusters, mode_prefix="raw"
                        )

                    # processed_clusters

                    processed_clusters, processed_cells = LayoutPostprocessor(
                        page, clusters, self.options
                    ).postprocess()

                    # Note: LayoutPostprocessor updates page.cells and page.parsed_page internally

                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            "Mean of empty slice|invalid value encountered in scalar divide",
                            RuntimeWarning,
                            "numpy",
                        )

                        conv_res.confidence.pages[page.page_no].layout_score = float(
                            np.mean([c.confidence for c in processed_clusters])
                        )

                        conv_res.confidence.pages[page.page_no].ocr_score = float(
                            np.mean(
                                [c.confidence for c in processed_cells if c.from_ocr]
                            )
                        )

                    page.predictions.layout = LayoutPrediction(
                        clusters=processed_clusters
                    )

                yield page
