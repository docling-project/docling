import logging
import warnings
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import List, Optional, Type

import numpy
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    OcrOptions,
)
from docling.datamodel.settings import settings
from docling.models.base_ocr_model import BaseOcrModel
from docling.utils.accelerator_utils import decide_device
from docling.utils.profiling import TimeRecorder
from docling.utils.utils import download_url_with_progress

_log = logging.getLogger(__name__)


class EasyOcrModel(BaseOcrModel):
    _model_repo_folder = "EasyOcr"

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: EasyOcrOptions,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: EasyOcrOptions

        self.scale = 3  # multiplier for 72 dpi == 216 dpi.

        if self.enabled:
            try:
                import easyocr
            except ImportError:
                raise ImportError(
                    "EasyOCR is not installed. Please install it via `pip install easyocr` to use this OCR engine. "
                    "Alternatively, Docling has support for other OCR engines. See the documentation."
                )

            if self.options.use_gpu is None:
                device = decide_device(accelerator_options.device)
                # Enable easyocr GPU if running on CUDA, MPS
                use_gpu = any(
                    device.startswith(x)
                    for x in [
                        AcceleratorDevice.CUDA.value,
                        AcceleratorDevice.MPS.value,
                    ]
                )
            else:
                warnings.warn(
                    "Deprecated field. Better to set the `accelerator_options.device` in `pipeline_options`. "
                    "When `use_gpu and accelerator_options.device == AcceleratorDevice.CUDA` the GPU is used "
                    "to run EasyOCR. Otherwise, EasyOCR runs in CPU."
                )
                use_gpu = self.options.use_gpu

            download_enabled = self.options.download_enabled
            model_storage_directory = self.options.model_storage_directory
            if artifacts_path is not None and model_storage_directory is None:
                download_enabled = False
                model_storage_directory = str(artifacts_path / self._model_repo_folder)

            self.reader = easyocr.Reader(
                lang_list=self.options.lang,
                gpu=use_gpu,
                model_storage_directory=model_storage_directory,
                recog_network=self.options.recog_network,
                download_enabled=download_enabled,
                verbose=False,
            )

    @staticmethod
    def download_models(
        detection_models: List[str] = ["craft"],
        recognition_models: List[str] = ["english_g2", "latin_g2"],
        local_dir: Optional[Path] = None,
        force: bool = False,
        progress: bool = False,
    ) -> Path:
        # Models are located in https://github.com/JaidedAI/EasyOCR/blob/master/easyocr/config.py
        from easyocr.config import (
            detection_models as det_models_dict,
            recognition_models as rec_models_dict,
        )

        if local_dir is None:
            local_dir = settings.cache_dir / "models" / EasyOcrModel._model_repo_folder

        local_dir.mkdir(parents=True, exist_ok=True)

        # Collect models to download
        download_list = []
        for model_name in detection_models:
            if model_name in det_models_dict:
                download_list.append(det_models_dict[model_name])
        for model_name in recognition_models:
            if model_name in rec_models_dict["gen2"]:
                download_list.append(rec_models_dict["gen2"][model_name])

        # Download models
        for model_details in download_list:
            buf = download_url_with_progress(model_details["url"], progress=progress)
            with zipfile.ZipFile(buf, "r") as zip_ref:
                zip_ref.extractall(local_dir)

        return local_dir

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
                        result = self.reader.readtext(im)

                        del high_res_image
                        del im

                        cells = [
                            TextCell(
                                index=ix,
                                text=line[1],
                                orig=line[1],
                                from_ocr=True,
                                confidence=line[2],
                                rect=BoundingRectangle.from_bounding_box(
                                    BoundingBox.from_tuple(
                                        coord=(
                                            (line[0][0][0] / self.scale) + ocr_rect.l,
                                            (line[0][0][1] / self.scale) + ocr_rect.t,
                                            (line[0][2][0] / self.scale) + ocr_rect.l,
                                            (line[0][2][1] / self.scale) + ocr_rect.t,
                                        ),
                                        origin=CoordOrigin.TOPLEFT,
                                    )
                                ),
                            )
                            for ix, line in enumerate(result)
                            if line[2] >= self.options.confidence_threshold
                        ]
                        all_ocr_cells.extend(cells)

                    # Post-process the cells
                    page.cells = self.post_process_cells(all_ocr_cells, page.cells)

                # DEBUG code:
                if settings.debug.visualize_ocr:
                    self.draw_ocr_rects_and_cells(conv_res, page, ocr_rects)

                yield page

    @classmethod
    def get_options_type(cls) -> Type[OcrOptions]:
        return EasyOcrOptions
