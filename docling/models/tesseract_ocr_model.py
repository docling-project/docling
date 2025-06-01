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
from docling.models.base_ocr_model import BaseOcrModel
from docling.utils.ocr_utils import (
    map_tesseract_script,
    parse_tesseract_orientation,
    tesseract_box_to_bounding_rectangle,
)
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)


class TesseractOcrModel(BaseOcrModel):
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
        self._is_auto: bool = "auto" in self.options.lang
        self.scale = 3  # multiplier for 72 dpi == 216 dpi.
        self.reader = None
        self.script_readers: dict[str, tesserocr.PyTessBaseAPI] = {}

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

            _, self._tesserocr_languages = tesserocr.get_languages()
            if not self._tesserocr_languages:
                raise ImportError(missing_langs_errmsg)

            # Initialize the tesseractAPI
            _log.debug("Initializing TesserOCR: %s", tesseract_version)
            lang = "+".join(self.options.lang)

            if any(lang.startswith("script/") for lang in self._tesserocr_languages):
                self.script_prefix = "script/"
            else:
                self.script_prefix = ""

            tesserocr_kwargs = {
                "psm": tesserocr.PSM.AUTO,
                "init": True,
                "oem": tesserocr.OEM.DEFAULT,
            }

            self.osd_reader = None

            if self.options.path is not None:
                tesserocr_kwargs["path"] = self.options.path

            if lang == "auto":
                self.reader = tesserocr.PyTessBaseAPI(**tesserocr_kwargs)
            else:
                self.reader = tesserocr.PyTessBaseAPI(
                    **{"lang": lang} | tesserocr_kwargs,
                )
            self.osd_reader = tesserocr.PyTessBaseAPI(
                **{"lang": "osd", "psm": tesserocr.PSM.OSD_ONLY} | tesserocr_kwargs
            )
            self.reader_RIL = tesserocr.RIL

    def __del__(self):
        if self.reader is not None:
            # Finalize the tesseractAPI
            self.reader.End()
        for script in self.script_readers:
            self.script_readers[script].End()

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
                    assert self.reader is not None
                    assert self.osd_reader is not None
                    assert self._tesserocr_languages is not None

                    ocr_rects = self.get_ocr_rects(page)

                    all_ocr_cells = []
                    for ocr_rect_i, ocr_rect in enumerate(ocr_rects):
                        # Skip zero area boxes
                        if ocr_rect.area() == 0:
                            continue
                        high_res_image = page._backend.get_page_image(
                            scale=self.scale, cropbox=ocr_rect
                        )

                        local_reader = self.reader
                        self.osd_reader.SetImage(high_res_image)
                        osd = self.osd_reader.DetectOrientationScript()
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
                            if self._is_auto:
                                continue
                        doc_orientation = parse_tesseract_orientation(osd["orient_deg"])
                        if doc_orientation != 0:
                            high_res_image = high_res_image.rotate(
                                -doc_orientation, expand=True
                            )
                        if self._is_auto:
                            script = osd["script_name"]
                            script = map_tesseract_script(script)
                            lang = f"{self.script_prefix}{script}"

                            # Check if the detected language is present in the system
                            if lang not in self._tesserocr_languages:
                                msg = f"Tesseract detected the script '{script}' and language '{lang}'."
                                msg += " However this language is not installed in your system and will be ignored."
                                _log.warning(msg)
                            else:
                                if script not in self.script_readers:
                                    import tesserocr

                                    self.script_readers[script] = (
                                        tesserocr.PyTessBaseAPI(
                                            path=self.reader.GetDatapath(),
                                            lang=lang,
                                            psm=tesserocr.PSM.AUTO,
                                            init=True,
                                            oem=tesserocr.OEM.DEFAULT,
                                        )
                                    )
                                local_reader = self.script_readers[script]

                        local_reader.SetImage(high_res_image)
                        boxes = local_reader.GetComponentImages(
                            self.reader_RIL.TEXTLINE, True
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
                    page.cells = self.post_process_cells(all_ocr_cells, page.cells)

                # DEBUG code:
                if settings.debug.visualize_ocr:
                    self.draw_ocr_rects_and_cells(conv_res, page, ocr_rects)

                yield page

    @classmethod
    def get_options_type(cls) -> Type[OcrOptions]:
        return TesseractOcrOptions
