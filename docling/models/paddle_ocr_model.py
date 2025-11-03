import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Type

import numpy
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    OcrOptions,
    PaddleOcrOptions,
)
from docling.datamodel.settings import settings
from docling.models.base_ocr_model import BaseOcrModel
from docling.utils.accelerator_utils import decide_device
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)

import itertools, io
import grpc
import docling.models.ocr_pb2 as ocr_pb2
import docling.models.ocr_pb2_grpc as ocr_pb2_grpc
import io
from PIL import Image

import base64
import numpy as np
import requests

class PaddleOcrModel(BaseOcrModel):
    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: PaddleOcrOptions,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: PaddleOcrOptions

        self.scale = 1

        if self.enabled:

            if self.options.ocr_endpoint == "":
                raise ValueError("PaddleOcrOptions.ocr_endpoint must be set when using PaddleOcrModel.")

            # Decide the accelerator devices
            device = decide_device(accelerator_options.device)
            use_cuda = str(AcceleratorDevice.CUDA.value).lower() in device
            use_dml = accelerator_options.device == AcceleratorDevice.AUTO
            intra_op_num_threads = accelerator_options.num_threads

            self.ocr_endpoint = self.options.ocr_endpoint
            self.timeout = self.options.timeout

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        if not self.enabled:
            yield from page_batch
            return

        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
            else:
                with TimeRecorder(conv_res, "ocr"):
                    ocr_rects = self.get_ocr_rects(page)

                    all_ocr_cells = []
                    for ocr_rect in ocr_rects:
                        # Skip zero area boxes
                        if ocr_rect.area() == 0:
                            continue
                        high_res_image = page._backend.get_page_image(
                            scale=self.scale, cropbox=ocr_rect
                        )
                        im = numpy.array(high_res_image)

                        result = self.ocr_numpy_image(im, timeout=self.timeout)
                        rec_texts, rec_scores, rec_boxes = self.extract_ocr_fields(result)

                        del high_res_image
                        del im

                        if len(rec_texts) > 0:
                            cells = [
                                TextCell(
                                    index=ix,
                                    text=t,
                                    orig=t,
                                    confidence=s,
                                    from_ocr=True,
                                    rect=BoundingRectangle.from_bounding_box(
                                        BoundingBox.from_tuple(
                                            coord=(
                                                (b[0] / self.scale)
                                                + ocr_rect.l,
                                                (b[1] / self.scale)
                                                + ocr_rect.t,
                                                (b[2] / self.scale)
                                                + ocr_rect.l,
                                                (b[3] / self.scale)
                                                + ocr_rect.t,
                                            ),
                                            origin=CoordOrigin.TOPLEFT,
                                        )
                                    ),
                                )
                                for ix, (t, s, b) in enumerate(zip(rec_texts, rec_scores, rec_boxes))
                            ]
                            all_ocr_cells.extend(cells)

                    # Post-process the cells
                    self.post_process_cells(all_ocr_cells, page)

                # DEBUG code:
                if settings.debug.visualize_ocr:
                    self.draw_ocr_rects_and_cells(conv_res, page, ocr_rects)

                yield page


    def numpy_to_image(self, arr: np.ndarray) -> Image.Image:
        if arr.dtype != np.uint8:
            a_min, a_max = float(arr.min()), float(arr.max())
            arr = ((arr - a_min) / (a_max - a_min) * 255.0).astype(np.uint8) if a_max > a_min else np.zeros_like(arr, dtype=np.uint8)
        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L")
        if arr.ndim == 3 and arr.shape[2] == 3:
            return Image.fromarray(arr, mode="RGB")
        if arr.ndim == 3 and arr.shape[2] == 4:
            return Image.fromarray(arr, mode="RGBA")
        raise ValueError(f"Unsupported array shape: {arr.shape}")

    def pil_to_bytes(self, img: Image.Image) -> tuple[bytes]:
        """
        이미지를 PNG로 직렬화.
        """
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    def post_ocr_bytes(self, img_bytes: bytes, timeout=60) -> dict:
        HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}
        payload = {"file": base64.b64encode(img_bytes).decode("ascii"), "fileType": 1, "visualize": False}
        r = requests.post(self.ocr_endpoint, json=payload, headers=HEADERS, timeout=timeout)
        if not r.ok:
            # 진단에 도움되도록 본문 일부 출력
            raise RuntimeError(f"OCR HTTP {r.status_code}: {r.text[:500]}")
        return r.json()

    def ocr_numpy_image(self, arr: np.ndarray, timeout=60) -> dict:
        img = self.numpy_to_image(arr)
        img_bytes = self.pil_to_bytes(img)
        return self.post_ocr_bytes(img_bytes, timeout=timeout)

    def extract_ocr_fields(self, resp: dict):
        """
        resp: 위와 같은 OCR 응답 JSON(dict)
        return: (rec_texts, rec_scores, rec_boxes) — 모두 list
        """
        if resp is None:
            return [], [], []

        # 최상위 상태 체크
        if resp.get("errorCode") not in (0, None):
            return [], [], []

        ocr_results = (
            resp.get("result", {})
                .get("ocrResults", [])
        )
        if not ocr_results:
            return [], [], []

        pruned = (
            ocr_results[0]
            .get("prunedResult", {})
        )
        if not pruned:
            return [], [], []

        rec_texts  = pruned.get("rec_texts", [])   # list[str]
        rec_scores = pruned.get("rec_scores", [])  # list[float]
        rec_boxes  = pruned.get("rec_boxes", [])   # list[[x1,y1,x2,y2]]

        # 길이 불일치 방어: 최소 길이에 맞춰 자르기
        n = min(len(rec_texts), len(rec_scores), len(rec_boxes))
        return rec_texts[:n], rec_scores[:n], rec_boxes[:n]

    @classmethod
    def get_options_type(cls) -> Type[OcrOptions]:
        return PaddleOcrOptions
