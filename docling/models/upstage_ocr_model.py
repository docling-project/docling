import io
import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Type

import requests
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, TextCell
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    OcrOptions,
    UpstageOcrOptions,
)
from docling.datamodel.settings import settings
from docling.models.base_ocr_model import BaseOcrModel
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)


class UpstageOcrModel(BaseOcrModel):
    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: UpstageOcrOptions,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: UpstageOcrOptions

        self.scale = 1

        if self.enabled:
            api_key = self.options.api_key or os.getenv("UPSTAGE_API_KEY", "")
            if not api_key:
                raise ValueError(
                    "UpstageOcrOptions.api_key 또는 UPSTAGE_API_KEY 환경변수가 설정되어야 합니다."
                )

            self.api_key = api_key
            self.api_endpoint = self.options.api_endpoint
            self.model = self.options.model
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
                        if ocr_rect.area() == 0:
                            continue
                        high_res_image = page._backend.get_page_image(
                            scale=self.scale, cropbox=ocr_rect
                        )

                        try:
                            result = self._call_upstage(high_res_image)
                        except (requests.RequestException, RuntimeError) as e:
                            # 네트워크 장애 / 4xx-5xx HTTP / JSON decode 실패만 skip.
                            # _extract_cells 안의 버그 (KeyError, TypeError 등) 는
                            # 그대로 전파해서 가려지지 않도록 한다.
                            _log.warning(f"Upstage OCR call failed: {e}")
                            continue

                        cells = self._extract_cells(
                            result, ocr_rect, high_res_image.size
                        )

                        del high_res_image

                        if cells:
                            all_ocr_cells.extend(cells)

                    self.post_process_cells(all_ocr_cells, page)

                if settings.debug.visualize_ocr:
                    self.draw_ocr_rects_and_cells(conv_res, page, ocr_rects)

                yield page

    def _call_upstage(self, image: Image.Image) -> dict:
        buf = io.BytesIO()
        image.save(buf, format="PNG", optimize=True)
        buf.seek(0)

        headers = {"Authorization": f"Bearer {self.api_key}"}
        files = {"document": ("page.png", buf, "image/png")}
        data = {"model": self.model}

        r = requests.post(
            self.api_endpoint,
            headers=headers,
            files=files,
            data=data,
            timeout=self.timeout,
        )
        if not r.ok:
            raise RuntimeError(
                f"Upstage OCR HTTP {r.status_code}: {r.text[:500]}"
            )
        return r.json()

    def _extract_cells(self, resp: dict, ocr_rect, sent_image_size):
        """Convert Upstage OCR JSON response to docling TextCell list.

        Coordinates use TOPLEFT origin and Upstage returns 4 vertices per word.
        We compute an axis-aligned bbox via min/max. If the response page
        width/height differ from the image we sent, scale coordinates back to
        sent-image pixel space (defensive — Upstage currently returns 1:1).
        """
        if not resp:
            return []

        pages = resp.get("pages") or []
        if not pages:
            return []
        resp_page = pages[0] or {}

        sent_w, sent_h = sent_image_size
        resp_w = resp_page.get("width") or sent_w
        resp_h = resp_page.get("height") or sent_h
        sx = float(sent_w) / float(resp_w) if resp_w else 1.0
        sy = float(sent_h) / float(resp_h) if resp_h else 1.0

        words = resp_page.get("words") or []
        cells = []
        ix = 0
        for word in words:
            text = word.get("text", "")
            confidence = float(word.get("confidence", 1.0))
            if not text:
                continue
            if confidence < self.options.text_score:
                continue

            verts = (word.get("boundingBox") or {}).get("vertices") or []
            if len(verts) < 2:
                continue

            xs = [float(v.get("x", 0.0)) * sx for v in verts]
            ys = [float(v.get("y", 0.0)) * sy for v in verts]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            rect = BoundingRectangle.from_bounding_box(
                BoundingBox.from_tuple(
                    coord=(
                        (min_x / self.scale) + ocr_rect.l,
                        (min_y / self.scale) + ocr_rect.t,
                        (max_x / self.scale) + ocr_rect.l,
                        (max_y / self.scale) + ocr_rect.t,
                    ),
                    origin=CoordOrigin.TOPLEFT,
                )
            )
            cells.append(
                TextCell(
                    index=ix,
                    text=text,
                    orig=text,
                    confidence=confidence,
                    from_ocr=True,
                    rect=rect,
                )
            )
            ix += 1

        return cells

    @classmethod
    def get_options_type(cls) -> Type[OcrOptions]:
        return UpstageOcrOptions
