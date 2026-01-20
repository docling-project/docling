import logging
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Type

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    OcrAutoOptions,
    OcrMacOptions,
    OcrOptions,
    RapidOcrOptions,
)
from docling.models.base_ocr_model import BaseOcrModel
from docling.models.stages.ocr.easyocr_model import EasyOcrModel
from docling.models.stages.ocr.ocr_mac_model import OcrMacModel
from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel

_log = logging.getLogger(__name__)


class OcrAutoModel(BaseOcrModel):
    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: OcrAutoOptions,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: OcrAutoOptions

        self._engine: Optional[BaseOcrModel] = None
        if self.enabled:
            if "darwin" == sys.platform:
                try:
                    from ocrmac import ocrmac

                    self._engine = OcrMacModel(
                        enabled=self.enabled,
                        artifacts_path=artifacts_path,
                        options=OcrMacOptions(
                            bitmap_area_threshold=self.options.bitmap_area_threshold,
                            force_full_page_ocr=self.options.force_full_page_ocr,
                        ),
                        accelerator_options=accelerator_options,
                    )
                    _log.info("Auto OCR model selected ocrmac.")
                except ImportError:
                    _log.info("ocrmac cannot be used because ocrmac is not installed.")

            if self._engine is None:
                try:
                    import onnxruntime
                    from rapidocr import EngineType, RapidOCR  # type: ignore

                    self._engine = RapidOcrModel(
                        enabled=self.enabled,
                        artifacts_path=artifacts_path,
                        options=RapidOcrOptions(
                            backend="onnxruntime",
                            bitmap_area_threshold=self.options.bitmap_area_threshold,
                            force_full_page_ocr=self.options.force_full_page_ocr,
                        ),
                        accelerator_options=accelerator_options,
                    )
                    _log.info("Auto OCR model selected rapidocr with onnxruntime.")
                except ImportError:
                    _log.info(
                        "rapidocr cannot be used because onnxruntime is not installed."
                    )

            if self._engine is None:
                try:
                    import easyocr

                    self._engine = EasyOcrModel(
                        enabled=self.enabled,
                        artifacts_path=artifacts_path,
                        options=EasyOcrOptions(
                            bitmap_area_threshold=self.options.bitmap_area_threshold,
                            force_full_page_ocr=self.options.force_full_page_ocr,
                        ),
                        accelerator_options=accelerator_options,
                    )
                    _log.info("Auto OCR model selected easyocr.")
                except ImportError:
                    _log.info("easyocr cannot be used because it is not installed.")

            if self._engine is None:
                try:
                    import torch
                    from rapidocr import EngineType, RapidOCR  # type: ignore

                    self._engine = RapidOcrModel(
                        enabled=self.enabled,
                        artifacts_path=artifacts_path,
                        options=RapidOcrOptions(
                            backend="torch",
                            bitmap_area_threshold=self.options.bitmap_area_threshold,
                            force_full_page_ocr=self.options.force_full_page_ocr,
                        ),
                        accelerator_options=accelerator_options,
                    )
                    _log.info("Auto OCR model selected rapidocr with torch.")
                except ImportError:
                    _log.info(
                        "rapidocr cannot be used because rapidocr or torch is not installed."
                    )

            if self._engine is None:
                _log.warning("No OCR engine found. Please review the install details.")

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        if not self.enabled or self._engine is None:
            yield from page_batch
            return
        yield from self._engine(conv_res, page_batch)

    @classmethod
    def get_options_type(cls) -> Type[OcrOptions]:
        return OcrAutoOptions
