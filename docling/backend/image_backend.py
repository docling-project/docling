import logging
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Optional, Union

from docling_core.types.doc import BoundingBox, CoordOrigin
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
from docling.datamodel.base_models import InputFormat, Size
from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)


class _ImagePageBackend(PdfPageBackend):
    def __init__(self, image: Image.Image):
        self._image: Optional[Image.Image] = image
        self.valid: bool = self._image is not None

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
        self, scale: float = 1, cropbox: Optional[BoundingBox] = None
    ) -> Image.Image:
        assert self._image is not None
        img = self._image

        if cropbox is not None:
            # Expected cropbox comes in TOPLEFT coords in our pipeline
            if cropbox.coord_origin != CoordOrigin.TOPLEFT:
                # Convert to TOPLEFT relative to current image height
                cropbox = cropbox.to_top_left_origin(img.height)
            left, top, right, bottom = cropbox.as_tuple()
            left = max(0, round(left))
            top = max(0, round(top))
            right = min(img.width, round(right))
            bottom = min(img.height, round(bottom))
            img = img.crop((left, top, right, bottom))

        if scale != 1:
            new_w = max(1, round(img.width * scale))
            new_h = max(1, round(img.height * scale))
            img = img.resize((new_w, new_h))

        return img

    def get_size(self) -> Size:
        assert self._image is not None
        return Size(width=self._image.width, height=self._image.height)

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
        path_or_stream: Union[BytesIO, Path],
        options: PdfBackendOptions = PdfBackendOptions(),
    ):
        # Bypass PdfDocumentBackend.__init__ to avoid image→PDF conversion
        AbstractDocumentBackend.__init__(self, in_doc, path_or_stream, options)
        self.options: PdfBackendOptions = options

        if self.input_format not in {InputFormat.IMAGE}:
            raise RuntimeError(
                f"Incompatible file format {self.input_format} was passed to ImageDocumentBackend."
            )

        # Load frames eagerly for thread-safety across pages
        self._frames: List[Image.Image] = []
        try:
            img = Image.open(self.path_or_stream)  # type: ignore[arg-type]

            # Handle multi-frame and single-frame images
            # - multiframe formats: TIFF, GIF, ICO
            # - singleframe formats: JPEG (.jpg, .jpeg), PNG (.png), BMP, WEBP (unless animated), HEIC
            frame_count = getattr(img, "n_frames", 1)

            if frame_count > 1:
                for i in range(frame_count):
                    img.seek(i)
                    self._frames.append(img.copy().convert("RGB"))
            else:
                self._frames.append(img.convert("RGB"))
        except Exception as e:
            raise RuntimeError(f"Could not load image for document {self.file}") from e

    def is_valid(self) -> bool:
        return len(self._frames) > 0

    def page_count(self) -> int:
        return len(self._frames)

    def load_page(self, page_no: int) -> _ImagePageBackend:
        if not (0 <= page_no < len(self._frames)):
            raise IndexError(f"Page index out of range: {page_no}")
        return _ImagePageBackend(self._frames[page_no])

    @classmethod
    def supported_formats(cls) -> set[InputFormat]:
        # Only IMAGE here; PDF handling remains in PDF-oriented backends
        return {InputFormat.IMAGE}

    @classmethod
    def supports_pagination(cls) -> bool:
        return True

    def unload(self):
        super().unload()
        self._frames = []
