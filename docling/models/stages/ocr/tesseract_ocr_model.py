# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Type

from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import TextCell

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    OcrOptions,
    TesseractOcrOptions,
)
from docling.datamodel.settings import settings
from docling.exceptions import OcrLanguageNotSupportedError
from docling.models.base_ocr_model import BaseOcrModel
from docling.models.stages.ocr.tesseract_utils import (
    installed_tesseract_languages,
    language_to_tesseract_code,
    osd_script_to_tesseract_code,
    parse_tesseract_orientation,
    tesseract_box_to_bounding_rectangle,
    tesseract_vocabulary,
)
from docling.utils.ocr_language import (
    OcrLanguage,
    OcrLanguageSupport,
)
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)


class TesseractOcrModel(BaseOcrModel):
    multiple_languages = True

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: TesseractOcrOptions,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: TesseractOcrOptions
        # No languages requested: Tesseract runs orientation and script
        # detection per page and picks a `script/` reader from the result.
        self._auto_script: bool = not self.languages
        # multiplier for 72 dpi; the default 3.0 == 216 dpi.
        self.scale = self.options.scale
        self._reader = None
        self._script_readers: dict[str, tesserocr.PyTessBaseAPI] = {}
        self._tesseract_vocabulary: list[str] = []
        self._native_codes: list[str] = []

        if self.enabled:
            install_errmsg = (
                "tesserocr is not correctly installed. "
                "Please install it via `pip install tesserocr` to use this OCR engine. "
                "Note that tesserocr might have to be manually compiled for working with "
                "your Tesseract installation. The Docling documentation provides examples for it. "
                "Alternatively, Docling has support for other OCR engines. See the documentation: "
                "https://docling-project.github.io/docling/installation/"
            )
            missing_langs_errmsg = (
                "tesserocr is not correctly configured. No language models have been detected. "
                "Please ensure that the TESSDATA_PREFIX envvar points to tesseract languages dir. "
                "You can find more information how to setup other OCR engines in Docling "
                "documentation: "
                "https://docling-project.github.io/docling/installation/"
            )

            try:
                import tesserocr
            except ImportError:
                raise ImportError(install_errmsg)
            try:
                tesseract_version = tesserocr.tesseract_version()
            except Exception:
                raise ImportError(install_errmsg)

            _, codes = tesserocr.get_languages()
            self._tesseract_vocabulary = tesseract_vocabulary(codes)
            if not self._tesseract_vocabulary:
                raise ImportError(missing_langs_errmsg)

            # Initialize the tesseractAPI
            _log.debug("Initializing TesserOCR: %s", tesseract_version)

            if self._auto_script and "osd" not in self._tesseract_vocabulary:
                raise ImportError(
                    "An empty OCR language list runs Tesseract's orientation and "
                    "script detection, which needs the 'osd' traineddata. Install "
                    "it (e.g. the tesseract-ocr-osd package) or name a language "
                    "explicitly in `ocr_options.lang`."
                )

            # Needs the installed language list and the prefix, so it runs here.
            self._native_codes = self.resolve_ocr_languages()

            tesserocr_kwargs = {
                "init": True,
                "oem": tesserocr.OEM.DEFAULT,
            }

            self._osd_reader = None

            if self.options.path is not None:
                tesserocr_kwargs["path"] = self.options.path

            # Set main OCR reader with configurable PSM
            main_psm = (
                self.options.psm if self.options.psm is not None else tesserocr.PSM.AUTO
            )
            if self._auto_script:
                # No `lang`: the per-page OSD pass picks a script reader instead.
                self._reader = tesserocr.PyTessBaseAPI(psm=main_psm, **tesserocr_kwargs)
            else:
                self._reader = tesserocr.PyTessBaseAPI(
                    lang="+".join(self._native_codes),
                    psm=main_psm,
                    **tesserocr_kwargs,
                )
            # OSD reader must use PSM.OSD_ONLY for orientation detection
            self._osd_reader = tesserocr.PyTessBaseAPI(
                lang="osd", psm=tesserocr.PSM.OSD_ONLY, **tesserocr_kwargs
            )
            self._reader_RIL = tesserocr.RIL

    def supported_ocr_languages(self) -> OcrLanguageSupport:
        return installed_tesseract_languages(self._tesseract_vocabulary)

    def map_ocr_language(self, language: OcrLanguage) -> str | list[str]:
        name = language_to_tesseract_code(language)
        if name is None or name not in self._tesseract_vocabulary:
            raise OcrLanguageNotSupportedError(
                self._engine_name,
                language.tag,
                supported=self.supported_ocr_languages(),
                detail=(
                    f"No traineddata file {name!r} is installed."
                    if name is not None
                    else "Tesseract has no traineddata for it."
                ),
            )
        return name

    def __del__(self):
        if self._reader is not None:
            # Finalize the tesseractAPI
            self._reader.End()
        for reader in self._script_readers.values():
            reader.End()

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        if not self.enabled:
            yield from page_batch
            return

        for page_i, page in enumerate(page_batch):
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
            else:
                with TimeRecorder(conv_res, "ocr"):
                    assert self._reader is not None
                    assert self._osd_reader is not None
                    assert self._tesseract_vocabulary is not None

                    ocr_rects = self.get_ocr_rects(page)

                    all_ocr_cells = []
                    for ocr_rect_i, ocr_rect in enumerate(ocr_rects):
                        # Skip zero area boxes
                        if ocr_rect.area() == 0:
                            continue
                        high_res_image = page._backend.get_page_image(
                            scale=self.scale, cropbox=ocr_rect
                        )

                        local_reader = self._reader
                        self._osd_reader.SetImage(high_res_image)

                        doc_orientation = 0
                        osd = self._osd_reader.DetectOrientationScript()

                        # No text, or Orientation and Script detection failure
                        if osd is None:
                            _log.error(
                                "OSD failed for doc (doc %s, page: %s, "
                                "OCR rectangle: %s)",
                                conv_res.input.file,
                                page_i,
                                ocr_rect_i,
                            )
                            # Skipping if OSD fail when in auto mode, otherwise proceed
                            # to OCR in the hope OCR will succeed while OSD failed
                            if self._auto_script:
                                continue
                        else:
                            doc_orientation = parse_tesseract_orientation(
                                osd["orient_deg"]
                            )
                            if doc_orientation != 0:
                                high_res_image = high_res_image.rotate(
                                    -doc_orientation, expand=True
                                )
                        if self._auto_script:
                            script = osd["script_name"]
                            lang = osd_script_to_tesseract_code(script)

                            # Check if the detected language is present in the system
                            if lang not in self._tesseract_vocabulary:
                                msg = f"Tesseract detected the script '{script}' and language '{lang}'."
                                msg += " However this language is not installed in your system and will be ignored."
                                _log.warning(msg)
                            else:
                                if lang not in self._script_readers:
                                    import tesserocr

                                    self._script_readers[lang] = (
                                        tesserocr.PyTessBaseAPI(
                                            path=self._reader.GetDatapath(),
                                            lang=lang,
                                            psm=self.options.psm
                                            if self.options.psm is not None
                                            else tesserocr.PSM.AUTO,
                                            init=True,
                                            oem=tesserocr.OEM.DEFAULT,
                                        )
                                    )
                                local_reader = self._script_readers[lang]

                        local_reader.SetImage(high_res_image)
                        boxes = local_reader.GetComponentImages(
                            self._reader_RIL.TEXTLINE, True
                        )

                        cells = []
                        for ix, (im, box, _, _) in enumerate(boxes):
                            # Set the area of interest. Tesseract uses Bottom-Left for the origin
                            local_reader.SetRectangle(
                                box["x"], box["y"], box["w"], box["h"]
                            )

                            # Extract text within the bounding box
                            text = local_reader.GetUTF8Text().strip()
                            confidence = local_reader.MeanTextConf()
                            left, top = box["x"], box["y"]
                            right = left + box["w"]
                            bottom = top + box["h"]
                            bbox = BoundingBox(
                                l=left,
                                t=top,
                                r=right,
                                b=bottom,
                                coord_origin=CoordOrigin.TOPLEFT,
                            )
                            rect = tesseract_box_to_bounding_rectangle(
                                bbox,
                                original_offset=ocr_rect,
                                scale=self.scale,
                                orientation=doc_orientation,
                                im_size=high_res_image.size,
                            )
                            cells.append(
                                TextCell(
                                    index=ix,
                                    text=text,
                                    orig=text,
                                    from_ocr=True,
                                    confidence=confidence,
                                    rect=rect,
                                )
                            )

                        # del high_res_image
                        all_ocr_cells.extend(cells)

                    # Post-process the cells
                    self.post_process_cells(all_ocr_cells, page, conv_res)

                # DEBUG code:
                if settings.debug.visualize_ocr:
                    self.draw_ocr_rects_and_cells(conv_res, page, ocr_rects)

                yield page

    @classmethod
    def get_options_type(cls) -> Type[OcrOptions]:
        return TesseractOcrOptions
