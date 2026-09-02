# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Optional, Set, Union

from docling_core.types.doc import BoundingBox, Size
from docling_core.types.doc.page import SegmentedPdfPage, TextCell
from PIL import Image

from docling.backend.abstract_backend import PaginatedDocumentBackend
from docling.datamodel.backend_options import PdfBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.utils.pdf_outline import _PdfOutlineItem

if TYPE_CHECKING:
    from docling_parse.pdf_parser import PdfStructure

    from docling.datamodel.base_models import TaggedTextCell


class PdfPageBackend(ABC):
    @property
    @abstractmethod
    def page_no(self) -> int:
        pass

    @abstractmethod
    def get_text_in_rect(self, bbox: BoundingBox) -> str:
        pass

    @abstractmethod
    def get_segmented_page(self) -> Optional[SegmentedPdfPage]:
        pass

    @abstractmethod
    def get_text_cells(self) -> Iterable[TextCell]:
        pass

    def get_marked_content(self) -> list["TaggedTextCell"]:
        """Text cells with their tagged-PDF marked-content linkage.

        Empty by default: only backends that read the content stream's
        BDC/EMC structure (docling-parse) can say which /MCID or /Artifact
        sequence a glyph was drawn in.
        """
        return []

    def get_visible_text_cells(self) -> Optional[list[TextCell]]:
        """Return the subset of `get_text_cells()` that actually paints ink on the page.

        Text drawn in a rendering mode that paints nothing (PDF 32000 modes 3 and 7) is left out
        `None` means this backend cannot tell visible from invisible text at all
        """
        return None

    @abstractmethod
    def get_bitmap_rects(self, scale: float = 1) -> Iterable[BoundingBox]:
        """Return bitmap bounds in 72-DPI document coordinates, scaled by ``scale``."""

    def has_content_in(
        self,
        *,
        bbox: BoundingBox,
        chars: bool = False,
        shapes: bool = True,
        bitmaps: bool = True,
    ) -> Optional[bool]:
        """`True` if any visible element of an enabled category overlaps bbox, else `False`

        `None` means this backend cannot answer the query at all, as distinct from `False`, which
        means it looked and found no intersecting content.
        """
        return None

    def get_shape_lines(
        self,
        *,
        horizontal: bool = True,
        vertical: bool = True,
        tolerance: float = 1e-3,
    ) -> Optional[list[BoundingBox]]:
        """Return the visible horizontal and/or vertical stroked shape segments.

        Boxes use top-left origin.
        Segments are returned as degenerate (zero-height or zero-width) boxes with top-left
        origin. `None` means this backend cannot answer the query at all
        """
        return None

    def get_connected_shape_bounding_boxes(
        self, *, tolerance: float = 0.0
    ) -> Optional[list[BoundingBox]]:
        """Return the bboxes of visible shapes merged by overlapping bboxes.

        Boxes use top-left origin. `None` means this backend cannot answer the query at
        all, as distinct from an empty list, which means it looked and found no shapes.
        """
        return None

    @abstractmethod
    def get_page_image(
        self, scale: float = 1, cropbox: Optional[BoundingBox] = None
    ) -> Image.Image:
        """Render a logical page region at ``scale`` pixels per document point."""

    @abstractmethod
    def get_size(self) -> Size:
        """Return the page size in 72-DPI document points."""

    @abstractmethod
    def is_valid(self) -> bool:
        pass

    @abstractmethod
    def unload(self):
        pass


class PdfDocumentBackend(PaginatedDocumentBackend):
    supports_random_page_access: ClassVar[bool] = True

    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: Union[BytesIO, Path],
        options: Optional[PdfBackendOptions] = None,
    ):
        if options is None:
            options = PdfBackendOptions()
        super().__init__(in_doc, path_or_stream, options)
        self.options: PdfBackendOptions

        if self.input_format not in self.supported_formats():
            raise RuntimeError(
                f"Incompatible file format {self.input_format} was passed to a PdfDocumentBackend. Valid format are {','.join(self.supported_formats())}."
            )

    @abstractmethod
    def load_page(self, page_no: int) -> PdfPageBackend:
        pass

    @abstractmethod
    def page_count(self) -> int:
        pass

    def iter_pages(self) -> Iterator[PdfPageBackend]:
        for page_index in range(self.page_count()):
            yield self.load_page(page_index)

    def get_document_outline(self) -> list[_PdfOutlineItem]:
        """Return the PDF bookmark / table-of-contents outline.

        A flat, document-ordered list where each entry carries its own depth (``level``). The
        default returns an empty list; PDFium-backed backends override this with the real
        outline. Backends without an embedded outline (e.g. OCR/image) keep the default.
        """
        return []

    def get_structure(self) -> Optional["PdfStructure"]:
        """The tagged-PDF logical structure tree (ISO 32000-2, 14.7), or None.

        The default returns None; the docling-parse backend overrides it. The
        structure is the authored counterpart of the layout model's
        prediction and is consumed by the tagged-structure stage.
        """
        return None

    @classmethod
    def supported_formats(cls) -> Set[InputFormat]:
        return {InputFormat.PDF}

    @classmethod
    def supports_pagination(cls) -> bool:
        return True


def iter_pdf_page_backends(
    backend: PdfDocumentBackend, page_nos: Iterable[int]
) -> Iterator[PdfPageBackend]:
    """Yield requested page backends, identified by absolute one-based page number.

    The caller owns each yielded backend. Unrequested pages and a yielded page
    abandoned by closing the iterator are unloaded here.
    """
    if backend.supports_random_page_access:
        for page_no in page_nos:
            page_backend = backend.load_page(page_no - 1)
            try:
                yield page_backend
            except GeneratorExit:
                page_backend.unload()
                raise
        return

    remaining_page_nos = set(page_nos)
    if not remaining_page_nos:
        return
    for page_backend in backend.iter_pages():
        if page_backend.page_no not in remaining_page_nos:
            page_backend.unload()
            continue

        remaining_page_nos.remove(page_backend.page_no)
        try:
            yield page_backend
        except GeneratorExit:
            page_backend.unload()
            raise
        if not remaining_page_nos:
            return
