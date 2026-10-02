# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""PP-OCRv6 tiny OCR using PaddleOCR's Transformers/safetensors engine."""

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from docling_core.types.doc import CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import OcrOptions, PpOcrv6Options
from docling.datamodel.settings import settings
from docling.exceptions import OcrLanguageNotSupportedError
from docling.models.base_ocr_model import BaseOcrModel
from docling.models.utils.hf_model_download import download_hf_model
from docling.utils.accelerator_utils import decide_device
from docling.utils.ocr_language import OcrLanguage, OcrLanguageSupport
from docling.utils.profiling import TimeRecorder

_DET_MODEL = "PP-OCRv6_tiny_det"
_REC_MODEL = "PP-OCRv6_tiny_rec"
_MODEL_FOLDER = "PaddleOCR"

# PaddleOCR's documented PP-OCRv6 vocabulary, excluding Japanese, which the
# tiny recognizer does not support. These codes share one multilingual model.
_LANGUAGES = frozenset(
    "ch chinese_cht en af az bs ca cs cy da de es et eu fi fr ga gl hr hu "
    "id is it ku la lb lt lv mi ms mt nl no oc pl pt qu rm ro rs_latin "
    "sk sl sq sv sw tl tr uz vi".split()
)
_ISO_TO_NATIVE = {
    "zh-Hans": "ch",
    "zh-Hant": "chinese_cht",
    "sr-Latn": "rs_latin",
    "fil-Latn": "tl",
}


def _model_dirs(artifacts_path: Path) -> tuple[Path, Path]:
    root = artifacts_path / _MODEL_FOLDER
    dirs = (
        root / f"{_DET_MODEL}_safetensors",
        root / f"{_REC_MODEL}_safetensors",
    )
    for directory in dirs:
        if not (directory / "model.safetensors").is_file():
            raise FileNotFoundError(
                f"Missing {directory / 'model.safetensors'}. Download the "
                "corresponding PaddlePaddle Hugging Face model snapshot into "
                "this directory, or omit artifacts_path for PaddleOCR-managed downloads."
            )
    return dirs


def _cell_from_polygon(
    index: int,
    text: str,
    score: float,
    polygon: Sequence[Sequence[float]],
    *,
    scale: float,
    offset_x: float,
    offset_y: float,
) -> TextCell:
    if len(polygon) != 4 or any(len(point) != 2 for point in polygon):
        raise ValueError("PaddleOCR recognition polygon must contain four xy points")
    points = [
        (float(point[0]) / scale + offset_x, float(point[1]) / scale + offset_y)
        for point in polygon
    ]
    return TextCell(
        index=index,
        text=text,
        orig=text,
        confidence=score,
        from_ocr=True,
        rect=BoundingRectangle(
            r_x0=points[0][0],
            r_y0=points[0][1],
            r_x1=points[1][0],
            r_y1=points[1][1],
            r_x2=points[2][0],
            r_y2=points[2][1],
            r_x3=points[3][0],
            r_y3=points[3][1],
            coord_origin=CoordOrigin.TOPLEFT,
        ),
    )


class PpOcrv6Model(BaseOcrModel):
    """Detect and recognize text with the fixed PP-OCRv6 tiny model pair."""

    _model_repo_folder = _MODEL_FOLDER

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Path | None,
        options: PpOcrv6Options,
        accelerator_options: AcceleratorOptions,
    ) -> None:
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: PpOcrv6Options
        self.scale = options.scale
        if not enabled:
            return

        self.resolve_ocr_languages()
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise ImportError(
                "PaddleOCR is not installed. Install docling[paddleocr] or "
                "docling-slim[feat-ocr-paddleocr]."
            ) from exc

        model_dirs: dict[str, str] = {}
        if artifacts_path is not None:
            det_dir, rec_dir = _model_dirs(artifacts_path)
            model_dirs = {
                "text_detection_model_dir": str(det_dir),
                "text_recognition_model_dir": str(rec_dir),
            }

        # PaddleX does not accept "mps", even with the Transformers engine.
        device = decide_device(
            accelerator_options.device,
            supported_devices=[
                AcceleratorDevice.CPU,
                AcceleratorDevice.CUDA,
                AcceleratorDevice.XPU,
            ],
        )
        if device.startswith("cuda"):
            device = device.replace("cuda", "gpu", 1)
        self.reader = PaddleOCR(
            text_detection_model_name=_DET_MODEL,
            text_recognition_model_name=_REC_MODEL,
            engine="transformers",
            device=device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            **model_dirs,
        )

    def supported_ocr_languages(self) -> OcrLanguageSupport:
        return OcrLanguageSupport(native=sorted(_LANGUAGES | {"auto"}))

    def resolve_ocr_languages(self) -> list[str]:
        if "auto" in self.options.lang and self.options.lang != ["auto"]:
            raise ValueError("PaddleOCR 'auto' cannot be combined with a language")
        if not self.languages or self.options.lang == ["auto"]:
            return []
        return super().resolve_ocr_languages()

    @classmethod
    def download_models(
        cls,
        local_dir: Path,
        force: bool = False,
        progress: bool = False,
    ) -> Path:
        """Prefetch both safetensors repositories for offline artifacts_path use."""
        for model_name in (_DET_MODEL, _REC_MODEL):
            repo_name = f"{model_name}_safetensors"
            download_hf_model(
                repo_id=f"PaddlePaddle/{repo_name}",
                local_dir=local_dir / repo_name,
                force=force,
                progress=progress,
            )
        return local_dir

    def map_ocr_language(self, language: OcrLanguage) -> str:
        if language.is_passthrough():
            code = language.native
        else:
            code = _ISO_TO_NATIVE.get(language.bcp47(), language.bcp47_language)
            if (
                not language.has_default_script()
                and language.bcp47() not in _ISO_TO_NATIVE
            ):
                code = None
        if code not in _LANGUAGES:
            raise OcrLanguageNotSupportedError(
                self._engine_name,
                language.tag(),
                supported=self.supported_ocr_languages(),
            )
        return code

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
                cells: list[TextCell] = []
                for ocr_rect in ocr_rects:
                    if ocr_rect.area() == 0:
                        continue
                    image = np.asarray(
                        page._backend.get_page_image(scale=self.scale, cropbox=ocr_rect)
                    )
                    for result in self.reader.predict(image):
                        payload: dict[str, Any] = result.json["res"]
                        texts = payload["rec_texts"]
                        scores = payload["rec_scores"]
                        polygons = payload["rec_polys"]
                        if not (len(texts) == len(scores) == len(polygons)):
                            raise ValueError(
                                "PaddleOCR returned mismatched recognition fields"
                            )
                        for text, score, polygon in zip(texts, scores, polygons):
                            if text:
                                cells.append(
                                    _cell_from_polygon(
                                        len(cells),
                                        text,
                                        float(score),
                                        polygon,
                                        scale=self.scale,
                                        offset_x=ocr_rect.l,
                                        offset_y=ocr_rect.t,
                                    )
                                )
                self.post_process_cells(cells, page, conv_res)

            if settings.debug.visualize_ocr:
                self.draw_ocr_rects_and_cells(conv_res, page, ocr_rects)
            yield page

    @classmethod
    def get_options_type(cls) -> type[OcrOptions]:
        return PpOcrv6Options
