"""KServe v2-based OCR model implementation."""

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import List, Optional, Type

import numpy as np
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, TextCell
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.kserve_transport_utils import resolve_kserve_transport_base_url
from docling.datamodel.pipeline_options import KserveV2OcrOptions, OcrOptions
from docling.datamodel.settings import settings
from docling.models.base_ocr_model import BaseOcrModel
from docling.models.inference_engines.common import KserveV2Client, KserveV2HttpClient
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)


class KserveV2OcrModel(BaseOcrModel):
    """OCR model using KServe v2 API (Triton Inference Server, KServe, etc.).

    This OCR engine connects to a remote KServe v2-compatible inference server
    to perform OCR via gRPC or HTTP. It handles custom preprocessing to match
    the expected input format of typical OCR models deployed on such servers.

    The preprocessing converts PIL images to numpy arrays with shape (1, H, W, C)
    in UINT8 format, matching the requirements of typical OCR models on Triton.

    Attributes:
        options: Configuration options including KServe v2 connection settings.
        _kserve_client: Client for communicating with the KServe v2 endpoint.
    """

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: KserveV2OcrOptions,
        accelerator_options: AcceleratorOptions,
    ):
        """Initialize the KServe v2 OCR model.

        Args:
            enabled: Whether OCR is enabled.
            artifacts_path: Path to model artifacts (not used for remote inference).
            options: KServe v2 OCR configuration options.
            accelerator_options: Accelerator configuration (not used for remote inference).
        """
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: KserveV2OcrOptions
        self._kserve_client: Optional[KserveV2Client] = None

        if self.enabled:
            self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the KServe v2 client for remote inference."""
        base_url = resolve_kserve_transport_base_url(
            url=self.options.url,
            transport=self.options.transport,
        )

        if self.options.transport == "http":
            self._kserve_client = KserveV2HttpClient(
                base_url=base_url,
                model_name=self.options.model_name,
                model_version=self.options.model_version,
                timeout=self.options.timeout,
                headers=self.options.headers,
            )
        else:
            from docling.models.inference_engines.common.kserve_v2_grpc import (
                KserveV2GrpcClient,
            )

            self._kserve_client = KserveV2GrpcClient(
                base_url=base_url,
                model_name=self.options.model_name,
                model_version=self.options.model_version,
                timeout=self.options.timeout,
                metadata=self.options.grpc_metadata,
                use_tls=self.options.grpc_use_tls,
                max_message_bytes=self.options.grpc_max_message_bytes,
                use_binary_data=self.options.grpc_use_binary_data,
            )

        _log.info(
            "KServe v2 OCR client initialized: url=%s, model=%s, transport=%s",
            self.options.url,
            self.options.model_name,
            self.options.transport,
        )

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for KServe v2 OCR inference.

        Converts PIL image to numpy array with shape (1, H, W, C) in UINT8 format,
        matching the expected input format for RapidOCR models on Triton with batching.

        Args:
            image: PIL Image to preprocess.

        Returns:
            Preprocessed numpy array with shape (1, H, W, C) and dtype UINT8.
        """
        # Convert to RGB and then to numpy array (H, W, C) with UINT8
        image_array = np.array(image.convert("RGB"), dtype=np.uint8)

        # Add batch dimension (1, H, W, C) as required by model with max_batch_size > 0
        batch_input = np.expand_dims(image_array, axis=0)

        return batch_input

    def _create_text_cells(
        self,
        boxes: np.ndarray,
        txts: np.ndarray,
        scores: np.ndarray,
        ocr_rect: BoundingBox,
    ) -> List[TextCell]:
        """Convert KServe v2 OCR outputs to TextCell objects.

        Args:
            boxes: Array of bounding boxes, shape (N, 4, 2) with format [[x0,y0], [x1,y1], [x2,y2], [x3,y3]].
            txts: Array of text strings or bytes.
            scores: Array of confidence scores.
            ocr_rect: The OCR rectangle coordinates for offset adjustment.

        Returns:
            List of TextCell objects with proper coordinate transformation.
        """
        cells: List[TextCell] = []

        for idx, (box, txt, score) in enumerate(zip(boxes, txts, scores)):
            # box format: [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
            # Convert to bounding box coordinates
            x_coords = [pt[0] for pt in box]
            y_coords = [pt[1] for pt in box]

            # Scale back and offset by ocr_rect
            x_min = (min(x_coords) / self.options.scale) + ocr_rect.l
            y_min = (min(y_coords) / self.options.scale) + ocr_rect.t
            x_max = (max(x_coords) / self.options.scale) + ocr_rect.l
            y_max = (max(y_coords) / self.options.scale) + ocr_rect.t

            bbox = BoundingBox.from_tuple(
                coord=(x_min, y_min, x_max, y_max),
                origin=CoordOrigin.TOPLEFT,
            )

            # Handle both bytes and string text
            text_str = txt.decode("utf-8") if isinstance(txt, bytes) else str(txt)

            cell = TextCell(
                index=idx,
                text=text_str,
                orig=text_str,
                confidence=float(score),
                from_ocr=True,
                rect=BoundingRectangle.from_bounding_box(bbox),
            )
            cells.append(cell)

        return cells

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        """Process pages with KServe v2 OCR.

        Args:
            conv_res: Conversion result object for tracking.
            page_batch: Iterable of pages to process.

        Yields:
            Processed pages with OCR results.
        """
        if not self.enabled or self._kserve_client is None:
            yield from page_batch
            return

        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
                continue

            with TimeRecorder(conv_res, "ocr"):
                # Get OCR rectangles (inherited from BaseOcrModel)
                ocr_rects = self.get_ocr_rects(page)

                all_ocr_cells: List[TextCell] = []
                for rect_idx, ocr_rect in enumerate(ocr_rects):
                    # Skip zero area boxes
                    if ocr_rect.area() == 0:
                        continue

                    # Get high-res image
                    high_res_image = page._backend.get_page_image(
                        scale=self.options.scale, cropbox=ocr_rect
                    )

                    # Custom preprocessing for KServe v2
                    image_array = self._preprocess_image(high_res_image)

                    # Call KServe v2 endpoint
                    try:
                        outputs = self._kserve_client.infer(
                            inputs={"image": image_array},
                            output_names=["boxes", "txts", "scores"],
                            request_parameters=self.options.request_parameters,
                        )

                        # Extract results
                        boxes = outputs["boxes"]
                        txts = outputs["txts"]
                        scores = outputs["scores"]

                        # Convert to TextCells
                        cells = self._create_text_cells(boxes, txts, scores, ocr_rect)
                        all_ocr_cells.extend(cells)

                    except Exception as e:
                        _log.error(
                            "KServe v2 OCR inference failed for page %d rect %d: %s",
                            page.page_no,
                            rect_idx,
                            str(e),
                        )
                        # Continue processing other rectangles

                    finally:
                        del high_res_image

                # Post-process the cells (inherited from BaseOcrModel)
                self.post_process_cells(all_ocr_cells, page)

            # DEBUG code:
            if settings.debug.visualize_ocr:
                self.draw_ocr_rects_and_cells(conv_res, page, ocr_rects)

            yield page

    @classmethod
    def get_options_type(cls) -> Type[OcrOptions]:
        """Get the options type for this OCR model.

        Returns:
            The KserveV2OcrOptions class.
        """
        return KserveV2OcrOptions

    def close(self) -> None:
        """Close the KServe v2 client connection."""
        if self._kserve_client is None:
            return
        self._kserve_client.close()

    def __del__(self) -> None:
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass


# Made with Bob
