from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from docling_core.types.doc import BoundingBox, CoordOrigin
from PIL import Image
from pypdfium2._helpers.misc import PdfiumError

from docling.backend.pdf_backend import PdfDocumentBackend, PdfPageBackend
from docling.datamodel.backend_options import PdfBackendOptions
from docling.utils.locks import pypdfium2_lock

if TYPE_CHECKING:
    from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)


class ManagedPdfiumDocumentBackend(PdfDocumentBackend, ABC):
    """Shared lifecycle management for PDFium-backed document backends."""

    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: Union[BytesIO, Path],
        options: Optional[PdfBackendOptions] = None,
    ) -> None:
        if options is None:
            options = PdfBackendOptions()
        super().__init__(in_doc, path_or_stream, options)
        self._closed = False

    @abstractmethod
    def _close_native_document(self) -> None:
        pass

    def unload(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_native_document()
        super().unload()


class ManagedPdfiumPageBackend(PdfPageBackend, ABC):
    """Shared page lifecycle and rendering for PDFium-backed page backends."""

    # Maximum Form XObject nesting depth considered when classifying a page
    # as raster-only.
    _RASTER_ONLY_MAX_DEPTH = 16

    def __init__(self, supersample_factor: float = 1.5) -> None:
        self._closed = False
        self.supersample_factor = supersample_factor
        self._raster_only: Optional[bool] = None

    @abstractmethod
    def _require_page(self) -> pdfium.PdfPage:
        pass

    @abstractmethod
    def _close_native_page(self) -> None:
        pass

    def _is_raster_only(self) -> bool:
        """Whether the rendered page content consists solely of image objects.

        This is the common case for scanned documents, where each page is a
        single full-page image with no text or vector content. Pages carrying
        annotations are not considered raster-only, since annotations are
        drawn on rendering but are not part of the page content objects.
        """
        if self._raster_only is None:
            max_depth = self._RASTER_ONLY_MAX_DEPTH
            has_image = False
            raster_only = True
            try:
                with pypdfium2_lock:
                    page = self._require_page()
                    if pdfium_c.FPDFPage_GetAnnotCount(page) > 0:
                        raster_only = False
                    else:
                        for obj in page.get_objects(max_depth=max_depth):
                            if obj.type == pdfium_c.FPDF_PAGEOBJ_IMAGE:
                                has_image = True
                            elif obj.type == pdfium_c.FPDF_PAGEOBJ_FORM:
                                if obj.level >= max_depth - 1:
                                    # Content nested beyond the recursion limit
                                    # was not inspected; assume it is not raster.
                                    raster_only = False
                                    break
                            else:
                                raster_only = False
                                break
            except PdfiumError:
                _log.info(
                    f"Failed to inspect the page objects of page {self.page_no}.",
                    exc_info=True,
                )
                raster_only = False
            self._raster_only = has_image and raster_only
        return self._raster_only

    def get_page_image(
        self, scale: float = 1, cropbox: Optional[BoundingBox] = None
    ) -> Image.Image:
        page_size = self.get_size()

        if not cropbox:
            cropbox = BoundingBox(
                l=0,
                r=page_size.width,
                t=0,
                b=page_size.height,
                coord_origin=CoordOrigin.TOPLEFT,
            )
            padbox = BoundingBox(
                l=0, r=0, t=0, b=0, coord_origin=CoordOrigin.BOTTOMLEFT
            )
        else:
            padbox = cropbox.to_bottom_left_origin(page_size.height).model_copy()
            padbox.r = page_size.width - padbox.r
            padbox.t = page_size.height - padbox.t

        # Supersampling (rendering at a higher scale, then downsizing) sharpens
        # vector content, but for raster-only (scanned) pages it is a lossy
        # resample round-trip of already-rasterized pixels that can destroy
        # borderline OCR text detections. Render those pages directly instead.
        factor = self.supersample_factor
        if factor != 1.0 and self._is_raster_only():
            factor = 1.0

        with pypdfium2_lock:
            bitmap = self._require_page().render(
                scale=scale * factor,
                rotation=0,  # no additional rotation
                crop=padbox.as_tuple(),
            )
            image = bitmap.to_pil().copy()
            bitmap.close()

        target_size = (round(cropbox.width * scale), round(cropbox.height * scale))
        if image.size != target_size:
            if (
                factor == 1.0
                and image.width >= target_size[0]
                and image.height >= target_size[1]
            ):
                # pdfium sizes bitmaps with ceil() per side; trim the rounding
                # excess at the right/bottom edges instead of resampling.
                image = image.crop((0, 0, *target_size))
            else:
                # Downsize the supersampled render to the requested scale. Only
                # reached on the direct-render path if per-side crop rounding
                # made the bitmap smaller than the target (max 1-2 px).
                image = image.resize(size=target_size)

        return image

    def unload(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_native_page()
