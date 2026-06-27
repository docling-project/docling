"""Mistral OCR API model implementation."""

import base64
import logging
import os
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from typing import Any, Type

import requests
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, TextCell
from PIL import Image
from pydantic import BaseModel, ConfigDict
from requests import Response

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import (
    DoclingComponentType,
    ErrorItem,
    FailureCategory,
    Page,
)
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import MistralOcrOptions, OcrOptions
from docling.datamodel.settings import settings
from docling.models.base_ocr_model import BaseOcrModel
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)


class _MistralBlock(BaseModel):
    model_config = ConfigDict(extra="ignore")

    top_left_x: float
    top_left_y: float
    bottom_right_x: float
    bottom_right_y: float
    content: str | None = None
    type: str | None = None


class _MistralWordConfidence(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str
    confidence: float
    start_index: int


class _MistralConfidenceScores(BaseModel):
    model_config = ConfigDict(extra="ignore")

    word_confidence_scores: list[_MistralWordConfidence] = []
    average_page_confidence_score: float | None = None


class _MistralPage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    markdown: str = ""
    blocks: list[_MistralBlock] = []
    confidence_scores: _MistralConfidenceScores | None = None


class _MistralOcrResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    pages: list[_MistralPage] = []


class MistralOcrModel(BaseOcrModel):
    """OCR model using the Mistral OCR API."""

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Path | None,
        options: MistralOcrOptions,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: MistralOcrOptions
        self.scale = options.scale
        self._session = requests.Session()
        self._api_key: str | None = None

        if self.enabled:
            self._api_key = self._resolve_api_key()

    def _resolve_api_key(self) -> str:
        if self.options.api_key is not None:
            return self.options.api_key.get_secret_value()

        api_key = os.getenv(self.options.api_key_env_var)
        if api_key:
            return api_key

        raise RuntimeError(
            "Mistral OCR requires an API key. Set "
            f"`{self.options.api_key_env_var}` or pass `MistralOcrOptions(api_key=...)`."
        )

    @staticmethod
    def _image_data_uri(image: Image.Image) -> str:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{image_b64}"

    def _request_payload(self, image: Image.Image) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.options.model,
            "document": {
                "type": "image_url",
                "image_url": self._image_data_uri(image),
            },
            "include_blocks": True,
            "include_image_base64": self.options.include_image_base64,
            "confidence_scores_granularity": (
                self.options.confidence_scores_granularity
            ),
            "table_format": self.options.table_format,
        }

        if self.options.image_limit is not None:
            payload["image_limit"] = self.options.image_limit
        if self.options.image_min_size is not None:
            payload["image_min_size"] = self.options.image_min_size

        return payload

    def _request_ocr(self, image: Image.Image) -> _MistralOcrResponse:
        assert self._api_key is not None
        response = self._session.post(
            self.options.url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=self._request_payload(image),
            timeout=self.options.timeout,
        )
        self._raise_for_response(response)
        return _MistralOcrResponse.model_validate(response.json())

    @staticmethod
    def _raise_for_response(response: Response) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text[:500]
            raise requests.HTTPError(
                f"Mistral OCR request failed with status "
                f"{response.status_code}: {detail}"
            ) from exc

    @staticmethod
    def _block_confidence(block: _MistralBlock, page: _MistralPage) -> float:
        scores = page.confidence_scores
        if scores is None:
            return 1.0

        content = (block.content or "").strip()
        if content and scores.word_confidence_scores:
            start = page.markdown.find(content)
            if start >= 0:
                end = start + len(content)
                block_scores = [
                    word.confidence
                    for word in scores.word_confidence_scores
                    if start <= word.start_index < end
                ]
                if block_scores:
                    return sum(block_scores) / len(block_scores)

        if scores.average_page_confidence_score is not None:
            return scores.average_page_confidence_score
        return 1.0

    @staticmethod
    def _block_to_cell(
        block: _MistralBlock,
        *,
        index: int,
        ocr_rect: BoundingBox,
        scale: float,
        confidence: float,
    ) -> TextCell | None:
        content = (block.content or "").strip()
        if not content:
            return None
        if block.type is not None and block.type.lower() == "image":
            return None

        bbox = BoundingBox.from_tuple(
            coord=(
                block.top_left_x / scale + ocr_rect.l,
                block.top_left_y / scale + ocr_rect.t,
                block.bottom_right_x / scale + ocr_rect.l,
                block.bottom_right_y / scale + ocr_rect.t,
            ),
            origin=CoordOrigin.TOPLEFT,
        )

        return TextCell(
            index=index,
            text=content,
            orig=content,
            confidence=confidence,
            from_ocr=True,
            rect=BoundingRectangle.from_bounding_box(bbox),
        )

    def _response_to_cells(
        self,
        response: _MistralOcrResponse,
        *,
        ocr_rect: BoundingBox,
        start_index: int,
    ) -> list[TextCell]:
        cells: list[TextCell] = []
        next_index = start_index
        for page in response.pages:
            for block in page.blocks:
                cell = self._block_to_cell(
                    block,
                    index=next_index,
                    ocr_rect=ocr_rect,
                    scale=self.scale,
                    confidence=self._block_confidence(block, page),
                )
                if cell is None:
                    continue
                cells.append(cell)
                next_index += 1
        return cells

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
                continue

            with TimeRecorder(conv_res, "ocr"):
                ocr_rects = self.get_ocr_rects(page)
                all_ocr_cells: list[TextCell] = []
                for rect_idx, ocr_rect in enumerate(ocr_rects):
                    if ocr_rect.area() == 0:
                        continue

                    high_res_image = page._backend.get_page_image(
                        scale=self.scale, cropbox=ocr_rect
                    )
                    try:
                        response = self._request_ocr(high_res_image)
                        cells = self._response_to_cells(
                            response,
                            ocr_rect=ocr_rect,
                            start_index=len(all_ocr_cells),
                        )
                        all_ocr_cells.extend(cells)
                    except Exception as exc:
                        _log.error(
                            "Mistral OCR inference failed for page %d rect %d: %s",
                            page.page_no,
                            rect_idx,
                            str(exc),
                        )
                        conv_res.errors.append(
                            ErrorItem(
                                component_type=DoclingComponentType.MODEL,
                                module_name=type(self).__name__,
                                error_message=str(exc) or exc.__class__.__name__,
                                category=FailureCategory.INFERENCE_FAILURE,
                                page_no=page.page_no,
                            )
                        )
                    finally:
                        del high_res_image

                self.post_process_cells(all_ocr_cells, page)

            if settings.debug.visualize_ocr:
                self.draw_ocr_rects_and_cells(conv_res, page, ocr_rects)

            yield page

    @classmethod
    def get_options_type(cls) -> Type[OcrOptions]:
        return MistralOcrOptions
