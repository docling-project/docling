"""Remote layout stage backed by a KServe v2 HTTP endpoint."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from docling_core.types.doc import CoordOrigin, DocItemLabel

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import BoundingBox, Cluster, LayoutPrediction, Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.kserve_transport_utils import resolve_kserve_transport_base_url
from docling.datamodel.pipeline_options import KserveV2LayoutOptions
from docling.exceptions import OperationNotAllowed
from docling.models.base_layout_model import BaseLayoutModel
from docling.models.inference_engines.common import KserveV2HttpClient
from docling.utils.layout_postprocessor import LayoutPostprocessor
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)


class KserveV2LayoutModel(BaseLayoutModel):
    """Layout stage using a dedicated remote HTTP layout service."""

    def __init__(
        self,
        artifacts_path: Optional[Path],
        accelerator_options: AcceleratorOptions,
        options: KserveV2LayoutOptions,
        enable_remote_services: bool = False,
    ):
        _ = artifacts_path, accelerator_options

        self.options = options
        self._kserve_client: Optional[KserveV2HttpClient] = None

        if not enable_remote_services:
            raise OperationNotAllowed(
                "Connections to remote services are only allowed when set explicitly. "
                "pipeline_options.enable_remote_services=True."
            )

        self._initialize_client()

    @classmethod
    def get_options_type(cls) -> type[KserveV2LayoutOptions]:
        return KserveV2LayoutOptions

    def _initialize_client(self) -> None:
        base_url = resolve_kserve_transport_base_url(
            url=self.options.url,
            transport=self.options.transport,
        )
        self._kserve_client = KserveV2HttpClient(
            base_url=base_url,
            model_name=self.options.model_name,
            model_version=self.options.model_version,
            timeout=self.options.timeout,
            headers=self.options.headers,
        )
        _log.info(
            "KServe v2 layout client initialized: url=%s, model=%s",
            self.options.url,
            self.options.model_name,
        )

    def close(self) -> None:
        if self._kserve_client is None:
            return
        self._kserve_client.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _preprocess_image(self, page_image) -> np.ndarray:
        image_array = np.asarray(page_image.convert("RGB"), dtype=np.uint8)
        return np.expand_dims(image_array, axis=0)

    def _predictions_to_clusters(
        self,
        *,
        page: Page,
        page_image,
        label_names: np.ndarray,
        scores: np.ndarray,
        boxes: np.ndarray,
    ) -> list[Cluster]:
        assert page.size is not None

        scale_x = page.size.width / page_image.width
        scale_y = page.size.height / page_image.height

        clusters: list[Cluster] = []
        for idx, (label_name, score, bbox_coords) in enumerate(
            zip(label_names, scores, boxes)
        ):
            if isinstance(label_name, bytes):
                label_name_str = label_name.decode("utf-8")
            else:
                label_name_str = str(label_name)

            label = DocItemLabel[
                label_name_str.upper().replace(" ", "_").replace("-", "_")
            ]
            bbox = BoundingBox(
                l=float(bbox_coords[0]) * scale_x,
                t=float(bbox_coords[1]) * scale_y,
                r=float(bbox_coords[2]) * scale_x,
                b=float(bbox_coords[3]) * scale_y,
                coord_origin=CoordOrigin.TOPLEFT,
            )
            clusters.append(
                Cluster(
                    id=idx,
                    label=label,
                    confidence=float(score),
                    bbox=bbox,
                    cells=[],
                )
            )

        return clusters

    def predict_layout(
        self,
        conv_res: ConversionResult,
        pages: Sequence[Page],
    ) -> Sequence[LayoutPrediction]:
        if self._kserve_client is None:
            raise RuntimeError("KServe v2 layout client is not initialized.")

        pages = list(pages)
        predictions: list[LayoutPrediction] = []

        for page in pages:
            assert page._backend is not None
            if not page._backend.is_valid():
                existing_prediction = page.predictions.layout or LayoutPrediction()
                page.predictions.layout = existing_prediction
                predictions.append(existing_prediction)
                continue

            page_image = page.get_image(scale=self.options.scale)
            if page_image is None:
                empty_prediction = page.predictions.layout or LayoutPrediction()
                page.predictions.layout = empty_prediction
                predictions.append(empty_prediction)
                continue

            with TimeRecorder(conv_res, "layout"):
                outputs = self._kserve_client.infer(
                    inputs={
                        self.options.image_input_name: self._preprocess_image(
                            page_image
                        )
                    },
                    output_names=[
                        self.options.label_output_name,
                        self.options.box_output_name,
                        self.options.score_output_name,
                    ],
                    request_parameters=self.options.request_parameters,
                )

                label_names = outputs[self.options.label_output_name]
                boxes = outputs[self.options.box_output_name]
                scores = outputs[self.options.score_output_name]

                clusters = self._predictions_to_clusters(
                    page=page,
                    page_image=page_image,
                    label_names=label_names,
                    scores=scores,
                    boxes=boxes,
                )

                processed_clusters, processed_cells = LayoutPostprocessor(
                    page=page,
                    clusters=clusters,
                    options=self.options,
                ).postprocess()

                layout_prediction = LayoutPrediction(clusters=processed_clusters)
                page.predictions.layout = layout_prediction

                if processed_clusters:
                    conv_res.confidence.pages[page.page_no].layout_score = float(
                        np.mean([cluster.confidence for cluster in processed_clusters])
                    )
                else:
                    conv_res.confidence.pages[page.page_no].layout_score = 0.0

                ocr_scores = [cell.confidence for cell in processed_cells if cell.from_ocr]
                if ocr_scores:
                    conv_res.confidence.pages[page.page_no].ocr_score = float(
                        np.mean(ocr_scores)
                    )

                predictions.append(layout_prediction)

        return predictions
