import logging
from pathlib import Path
from typing import Iterable, Optional, Type

import numpy
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    OcrOptions,
    OnnxtrOcrOptions,
)
from docling.datamodel.settings import settings
from docling.models.base_ocr_model import BaseOcrModel
from docling.utils.accelerator_utils import decide_device
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)


class OnnxtrOcrModel(BaseOcrModel):
    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: OnnxtrOcrOptions,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: OnnxtrOcrOptions

        self.scale = 3  # multiplier for 72 dpi == 216 dpi.

        if self.enabled:
            try:
                from onnxtr.models import ocr_predictor, EngineConfig, from_hub  # type: ignore
            except ImportError:
                raise ImportError(
                    "OnnxTR is not installed. Please install it via `pip install 'onnxtr[gpu]'` to use this OCR engine. "
                    "Alternatively, Docling has support for other OCR engines. See the documentation."
                )

            if options.auto_correct_orientation:
                config = {
                    "assume_straight_pages": False,
                    "straighten_pages": True,
                    "export_as_straight_boxes": True,
                    # Disable crop orientation because we straighten the pages already
                    "disable_crop_orientation": True,
                    "disable_page_orientation": False,
                }
            else:
                config = {
                    "assume_straight_pages": True,
                    "straighten_pages": False,
                    "export_as_straight_boxes": True,
                    "disable_crop_orientation": False,
                    "disable_page_orientation": False,
                }

            self.reader = ocr_predictor(
                det_arch=(
                    from_hub(self.options.det_arch)
                    if self.options.det_arch.count("/") == 1
                    else self.options.det_arch
                ),
                reco_arch=(
                    from_hub(self.options.reco_arch)
                    if self.options.reco_arch.count("/") == 1
                    else self.options.reco_arch
                ),
                preserve_aspect_ratio=self.options.preserve_aspect_ratio,
                symmetric_pad=self.options.symmetric_pad,
                paragraph_break=self.options.paragraph_break,
                load_in_8_bit=self.options.load_in_8_bit,
                **config,
            )

    def _to_absolute_and_docling_format(
        self, geom: list[list[float]], img_shape: tuple[int, int]
    ) -> tuple[int, int, int, int]:
        """
        Convert a bounding box or polygon from relative to absolute coordinates and return in [x1, y1, x2, y2] format.

        Args:
            geom (list): Either [[xmin, ymin], [xmax, ymax]] or [[x1, y1], ..., [x4, y4]]
            img_shape (tuple[int, int]): (height, width) of the image

        Returns:
            tuple: (x1, y1, x2, y2)
        """
        h, w = img_shape
        scale_inv = 1 / self.scale  # Precompute inverse for efficiency

        def scale_point(x: float, y: float) -> tuple[int, int]:
            """Scale and round a point to absolute coordinates."""
            return int(round(x * w * scale_inv)), int(round(y * h * scale_inv))

        if len(geom) == 2:
            (xmin, ymin), (xmax, ymax) = geom
            x1, y1 = scale_point(xmin, ymin)
            x2, y2 = scale_point(xmax, ymax)
        elif len(geom) == 4:
            abs_points = [scale_point(*point) for point in geom]
            x1, y1 = min(p[0] for p in abs_points), min(p[1] for p in abs_points)
            x2, y2 = max(p[0] for p in abs_points), max(p[1] for p in abs_points)
        else:
            raise ValueError(
                f"Invalid geometry format: {geom}. Expected either 2 or 4 points."
            )

        return x1, y1, x2, y2

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
                all_ocr_cells = []

                for ocr_rect in ocr_rects:
                    if ocr_rect.area() == 0:
                        continue

                    with page._backend.get_page_image(
                        scale=self.scale, cropbox=ocr_rect
                    ) as high_res_image:
                        im_width, im_height = high_res_image.size
                        result = self.reader([numpy.array(high_res_image)])

                    if result is not None:
                        for p in result.pages:
                            for ix, word in enumerate(
                                word
                                for block in p.blocks
                                for line in block.lines
                                for word in line.words
                            ):
                                all_ocr_cells.append(
                                    TextCell(
                                        index=ix,
                                        text=word.value,
                                        orig=word.value,
                                        from_ocr=True,
                                        confidence=word.confidence,
                                        rect=BoundingRectangle.from_bounding_box(
                                            BoundingBox.from_tuple(
                                                self._to_absolute_and_docling_format(
                                                    word.geometry,
                                                    img_shape=(im_height, im_width),
                                                ),
                                                origin=CoordOrigin.TOPLEFT,
                                            )
                                        ),
                                    )
                                )

                # Post-process the cells
                page.cells = self.post_process_cells(all_ocr_cells, page.cells)

            # DEBUG code:
            if settings.debug.visualize_ocr:
                self.draw_ocr_rects_and_cells(conv_res, page, ocr_rects)

            yield page

    @classmethod
    def get_options_type(cls) -> Type[OcrOptions]:
        return OnnxtrOcrOptions
