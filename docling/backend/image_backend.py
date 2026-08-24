import logging
import math
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from typing import SupportsFloat

from docling_core.types.doc import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.page import (
    BoundingRectangle,
    PdfPageBoundaryType,
    PdfPageGeometry,
    SegmentedPdfPage,
    TextCell,
)
from PIL import Image

from docling.backend.abstract_backend import AbstractDocumentBackend
from docling.backend.pdf_backend import PdfDocumentBackend, PdfPageBackend
from docling.datamodel.backend_options import PdfBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.exceptions import DocumentLoadError

_log = logging.getLogger(__name__)

_POINTS_PER_INCH = 72.0
_DEFAULT_DPI = (_POINTS_PER_INCH, _POINTS_PER_INCH)


def _validate_dpi(dpi: tuple[SupportsFloat, SupportsFloat]) -> tuple[float, float]:
    if not isinstance(dpi, tuple) or len(dpi) != 2:
        raise ValueError(f"Invalid image DPI metadata: {dpi!r}")

    try:
        dpi_x, dpi_y = float(dpi[0]), float(dpi[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid image DPI metadata: {dpi!r}") from exc

    if not math.isfinite(dpi_x) or not math.isfinite(dpi_y):
        raise ValueError(f"Invalid image DPI metadata: {dpi!r}")
    if dpi_x <= 0 or dpi_y <= 0:
        raise ValueError(f"Invalid image DPI metadata: {dpi!r}")

    return dpi_x, dpi_y


def _get_frame_dpi(image: Image.Image) -> tuple[float, float]:
    dpi = image.info.get("dpi")
    return _validate_dpi(_DEFAULT_DPI if dpi in (None, (1, 1)) else dpi)


class _ImagePageBackend(PdfPageBackend):
    def __init__(
        self, image: Image.Image, page_no: int, dpi: tuple[float, float]
    ) -> None:
        self._image: Image.Image | None = image
        self._page_no = page_no
        self._dpi = dpi
        self.valid: bool = self._image is not None

    @property
    def page_no(self) -> int:
        return self._page_no + 1

    def is_valid(self) -> bool:
        return self.valid

    def get_text_in_rect(self, bbox: BoundingBox) -> str:
        # No text extraction from raw images without OCR
        return ""

    def get_segmented_page(self) -> SegmentedPdfPage:
        # Return empty segmented page with proper dimensions for raw images
        assert self._image is not None
        page_size = self.get_size()
        bbox = BoundingBox(
            l=0.0,
            t=0.0,
            r=float(page_size.width),
            b=float(page_size.height),
            coord_origin=CoordOrigin.BOTTOMLEFT,
        )
        dimension = PdfPageGeometry(
            angle=0.0,
            rect=BoundingRectangle.from_bounding_box(bbox),
            boundary_type=PdfPageBoundaryType.CROP_BOX,
            art_bbox=bbox,
            bleed_bbox=bbox,
            crop_bbox=bbox,
            media_bbox=bbox,
            trim_bbox=bbox,
        )
        return SegmentedPdfPage(
            dimension=dimension,
            char_cells=[],
            word_cells=[],
            textline_cells=[],
            has_chars=False,
            has_words=False,
            has_lines=False,
        )

    def get_text_cells(self) -> Iterable[TextCell]:
        # No text cells on raw images
        return []

    def get_bitmap_rects(self, scale: float = 1) -> Iterable[BoundingBox]:
        # For raw images, the entire page is a bitmap
        assert self._image is not None
        page_size = self.get_size()
        full_page_bbox = BoundingBox(
            l=0.0,
            t=0.0,
            r=float(page_size.width),
            b=float(page_size.height),
            coord_origin=CoordOrigin.TOPLEFT,
        )
        if scale != 1:
            full_page_bbox = full_page_bbox.scaled(scale=scale)
        yield full_page_bbox

    def get_page_image(
        self, scale: float = 1, cropbox: BoundingBox | None = None
    ) -> Image.Image:
        assert self._image is not None
        page_size = self.get_size()

        if cropbox is None:
            left, top, right, bottom = 0.0, 0.0, page_size.width, page_size.height
        else:
            if cropbox.coord_origin != CoordOrigin.TOPLEFT:
                cropbox = cropbox.to_top_left_origin(page_size.height)
            left, top, right, bottom = cropbox.as_tuple()
            left = min(max(0.0, left), page_size.width)
            top = min(max(0.0, top), page_size.height)
            right = min(max(0.0, right), page_size.width)
            bottom = min(max(0.0, bottom), page_size.height)

        target_size = (
            max(1, round((right - left) * scale)),
            max(1, round((bottom - top) * scale)),
        )
        source_box = (
            left * self._dpi[0] / _POINTS_PER_INCH,
            top * self._dpi[1] / _POINTS_PER_INCH,
            right * self._dpi[0] / _POINTS_PER_INCH,
            bottom * self._dpi[1] / _POINTS_PER_INCH,
        )
        if target_size == self._image.size and source_box == (
            0.0,
            0.0,
            float(self._image.width),
            float(self._image.height),
        ):
            return self._image

        return self._image.resize(
            target_size,
            resample=Image.Resampling.LANCZOS,
            box=source_box,
        )

    def get_size(self) -> Size:
        assert self._image is not None
        return Size(
            width=self._image.width * _POINTS_PER_INCH / self._dpi[0],
            height=self._image.height * _POINTS_PER_INCH / self._dpi[1],
        )

    def unload(self):
        # Help GC and free memory
        self._image = None


class ImageDocumentBackend(PdfDocumentBackend):
    """Image-native backend that bypasses pypdfium2.

    Notes:
        - Subclasses PdfDocumentBackend to satisfy pipeline type checks.
        - Intentionally avoids calling PdfDocumentBackend.__init__ to skip
          the image→PDF conversion and any pypdfium2 usage.
        - Handles multi-page TIFF by extracting frames eagerly to separate
          Image objects to keep thread-safety when pages process in parallel.
    """

    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: BytesIO | Path,
        options: PdfBackendOptions | None = None,
    ) -> None:
        if options is None:
            options = PdfBackendOptions()
        # Bypass PdfDocumentBackend.__init__ to avoid image→PDF conversion
        AbstractDocumentBackend.__init__(self, in_doc, path_or_stream, options)
        self.options: PdfBackendOptions = options

        if self.input_format not in {InputFormat.IMAGE}:
            raise RuntimeError(
                f"Incompatible file format {self.input_format} was passed to ImageDocumentBackend."
            )

        # Load frames eagerly for thread-safety across pages
        self._frames: list[Image.Image] = []
        self._frame_dpi: list[tuple[float, float]] = []
        try:
            with Image.open(self.path_or_stream) as img:  # type: ignore[arg-type]
                # Handle multi-frame and single-frame images
                # - multiframe formats: TIFF, GIF, ICO
                # - singleframe formats: JPEG (.jpg, .jpeg), PNG (.png), BMP, WEBP (unless animated), HEIC
                frame_count = getattr(img, "n_frames", 1)

                if frame_count > 1:
                    for i in range(frame_count):
                        img.seek(i)
                        self._frame_dpi.append(_get_frame_dpi(img))
                        self._frames.append(img.copy().convert("RGB"))
                else:
                    self._frame_dpi.append(_get_frame_dpi(img))
                    self._frames.append(img.convert("RGB"))
        except Exception as e:
            for frame in self._frames:
                frame.close()
            self._frames = []
            self._frame_dpi = []
            raise DocumentLoadError(
                f"Could not load image for document {self.file}"
            ) from e

    def is_valid(self) -> bool:
        return len(self._frames) > 0

    def page_count(self) -> int:
        return len(self._frames)

    def load_page(self, page_no: int) -> _ImagePageBackend:
        if not (0 <= page_no < len(self._frames)):
            raise IndexError(f"Page index out of range: {page_no}")
        return _ImagePageBackend(
            self._frames[page_no], page_no, self._frame_dpi[page_no]
        )

    @classmethod
    def supported_formats(cls) -> set[InputFormat]:
        # Only IMAGE here; PDF handling remains in PDF-oriented backends
        return {InputFormat.IMAGE}

    @classmethod
    def supports_pagination(cls) -> bool:
        return True

    def unload(self):
        for frame in self._frames:
            frame.close()
        self._frames = []
        self._frame_dpi = []
        super().unload()
