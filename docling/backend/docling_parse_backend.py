import logging
from collections.abc import Iterable, Iterator
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import pypdfium2 as pdfium
from docling_core.types.doc import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.page import SegmentedPdfPage, TextCell
from docling_parse.pdf_parser import (
    ContentConfig,
    ContentLevel,
    DecodeConfig,
    DoclingPdfParser,
    DoclingThreadedPdfParser,
    PageParseResult,
    PdfDocument,
    RenderConfig,
    ThreadedPdfParserConfig,
)
from PIL import Image
from pypdfium2 import PdfPage

from docling.backend.managed_pdfium_backend import (
    ManagedPdfiumDocumentBackend,
    ManagedPdfiumPageBackend,
)
from docling.backend.pdf_backend import PdfDocumentBackend, PdfPageBackend
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.backend_options import (
    PdfBackendOptions,
    ThreadedDoclingParseBackendOptions,
)
from docling.datamodel.settings import DEFAULT_PAGE_RANGE
from docling.exceptions import DocumentLoadError
from docling.utils.locks import pypdfium2_lock
from docling.utils.pdf_outline import (
    _PdfOutlineItem,
    extract_outline_from_docling_parse,
)

if TYPE_CHECKING:
    from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)


def _make_docling_parse_decode_config(
    *,
    enforce_same_font: bool = True,
    release_native_memory_every_n_pages: int | None = None,
) -> DecodeConfig:
    config = DecodeConfig(enforce_same_font=enforce_same_font)

    if release_native_memory_every_n_pages is not None:
        config.release_native_memory_every_n_pages = release_native_memory_every_n_pages

    return config


def _make_docling_parse_page_content_config(
    *,
    create_words: bool,
    create_textlines: bool,
) -> ContentConfig:
    compute = ContentLevel.COMPUTE
    materialize = ContentLevel.COMPUTE_AND_MATERIALIZE
    skip = ContentLevel.SKIP

    return ContentConfig(
        char_cells_content_level=compute
        if (create_words or create_textlines)
        else skip,
        word_cells_content_level=materialize if create_words else skip,
        line_cells_content_level=materialize if create_textlines else skip,
        shapes_content_level=skip,
        bitmaps_content_level=materialize,
        include_bitmap_bytes=False,  # only need bitmap rectangles for OCR
    )


def _render_page_image_via_pdfium(
    ppage: PdfPage,
    page_size: Size,
    scale: float,
    cropbox: Optional[BoundingBox],
) -> Image.Image:
    """Render a page raster with pypdfium2.

    Shared by the serial and threaded docling-parse page backends so both feed
    the downstream layout model an identical raster. Diverging renderers (e.g.
    the docling-parse rasterizer) produce subtly different images and can flip
    layout detection (see issue #3512).
    """
    if not cropbox:
        cropbox = BoundingBox(
            l=0,
            r=page_size.width,
            t=0,
            b=page_size.height,
            coord_origin=CoordOrigin.TOPLEFT,
        )
        padbox = BoundingBox(l=0, r=0, t=0, b=0, coord_origin=CoordOrigin.BOTTOMLEFT)
    else:
        padbox = cropbox.to_bottom_left_origin(page_size.height).model_copy()
        padbox.r = page_size.width - padbox.r
        padbox.t = page_size.height - padbox.t

    with pypdfium2_lock:
        bitmap = ppage.render(
            scale=scale * 1.5,
            rotation=0,  # no additional rotation
            crop=padbox.as_tuple(),
        )
        image = bitmap.to_pil().copy()
        bitmap.close()
    # We resize the image from 1.5x the given scale to make it sharper.
    return image.resize(
        size=(round(cropbox.width * scale), round(cropbox.height * scale))
    )


class DoclingParsePageBackend(ManagedPdfiumPageBackend):
    def __init__(
        self,
        *,
        dp_doc: PdfDocument,
        page_obj: PdfPage,
        page_no: int,
        create_words: bool = True,
        create_textlines: bool = True,
        keep_chars: bool = False,
        keep_lines: bool = False,
        keep_images: bool = True,
    ):
        super().__init__()
        self._ppage = page_obj
        self._dp_doc: Optional[PdfDocument] = dp_doc
        self._page_no = page_no

        self._create_words = create_words
        self._create_textlines = create_textlines

        self._keep_chars = keep_chars
        self._keep_lines = keep_lines
        self._keep_images = keep_images

        self._dpage: Optional[SegmentedPdfPage] = None
        self._unloaded = False
        self.valid = (self._ppage is not None) and (self._dp_doc is not None)

    @property
    def page_no(self) -> int:
        return self._page_no + 1

    def _require_page(self) -> PdfPage:
        assert self._ppage is not None, "Page backend was unloaded."
        return self._ppage

    def _ensure_parsed(self) -> None:
        if self._dpage is not None:
            return

        content_config = _make_docling_parse_page_content_config(
            create_words=self._create_words,
            create_textlines=self._create_textlines,
        )

        assert self._dp_doc is not None
        seg_page = self._dp_doc.get_page(
            self._page_no + 1,
            content_config=content_config,
        )

        # In Docling, all TextCell instances are expected with top-left origin.
        [
            tc.to_top_left_origin(seg_page.dimension.height)
            for tc in seg_page.textline_cells
        ]
        [tc.to_top_left_origin(seg_page.dimension.height) for tc in seg_page.char_cells]
        [tc.to_top_left_origin(seg_page.dimension.height) for tc in seg_page.word_cells]

        self._dpage = seg_page

    def is_valid(self) -> bool:
        return self.valid

    def get_text_in_rect(self, bbox: BoundingBox) -> str:
        self._ensure_parsed()
        assert self._dpage is not None

        # Find intersecting cells on the page
        text_piece = ""
        page_size = self.get_size()

        scale = (
            1  # FIX - Replace with param in get_text_in_rect across backends (optional)
        )

        for i, cell in enumerate(self._dpage.textline_cells):
            cell_bbox = (
                cell.rect.to_bounding_box()
                .to_top_left_origin(page_height=page_size.height)
                .scaled(scale)
            )

            overlap_frac = cell_bbox.intersection_over_self(bbox)

            if overlap_frac > 0.5:
                if len(text_piece) > 0:
                    text_piece += " "
                text_piece += cell.text

        return text_piece

    def get_segmented_page(self) -> Optional[SegmentedPdfPage]:
        self._ensure_parsed()
        return self._dpage

    def get_text_cells(self) -> Iterable[TextCell]:
        self._ensure_parsed()
        assert self._dpage is not None

        return self._dpage.textline_cells

    def get_bitmap_rects(self, scale: float = 1) -> Iterable[BoundingBox]:
        self._ensure_parsed()
        assert self._dpage is not None

        AREA_THRESHOLD = 0  # 32 * 32

        images = self._dpage.bitmap_resources

        for img in images:
            cropbox = img.rect.to_bounding_box().to_top_left_origin(
                self.get_size().height
            )

            if cropbox.area() > AREA_THRESHOLD:
                cropbox = cropbox.scaled(scale=scale)

                yield cropbox

    def get_page_image(
        self, scale: float = 1, cropbox: Optional[BoundingBox] = None
    ) -> Image.Image:
        return _render_page_image_via_pdfium(
            self._require_page(), self.get_size(), scale, cropbox
        )

    def get_size(self) -> Size:
        with pypdfium2_lock:
            page = self._require_page()
            return Size(width=page.get_width(), height=page.get_height())

        # TODO: Take width and height from docling-parse.
        # return Size(
        #    width=self._dpage.dimension.width,
        #    height=self._dpage.dimension.height,
        # )

    def _close_native_page(self) -> None:
        if not self._unloaded and self._dp_doc is not None:
            self._dp_doc.unload_pages((self._page_no + 1, self._page_no + 2))
            self._unloaded = True

        with pypdfium2_lock:
            if self._ppage is not None:
                self._ppage.close()

        self._ppage = None
        self._dpage = None
        self._dp_doc = None


class DoclingParseDocumentBackend(ManagedPdfiumDocumentBackend):
    def __init__(
        self,
        in_doc: "InputDocument",
        path_or_stream: Union[BytesIO, Path],
        options: Optional[PdfBackendOptions] = None,
    ):
        if options is None:
            options = PdfBackendOptions()
        super().__init__(in_doc, path_or_stream, options)

        password = (
            self.options.password.get_secret_value() if self.options.password else None
        )
        self.dp_doc: Optional[PdfDocument]
        try:
            with pypdfium2_lock:
                self._pdoc = pdfium.PdfDocument(self.path_or_stream, password=password)
            self.parser = DoclingPdfParser(loglevel="fatal")
            decode_config = _make_docling_parse_decode_config(
                enforce_same_font=self.options.enforce_same_font,
            )
            self.dp_doc = self.parser.load(
                path_or_stream=self.path_or_stream,
                password=password,
                decode_config=decode_config,
            )
        except RuntimeError as e:
            # pypdfium2 (PdfiumError) and docling-parse both signal unreadable
            # bytes by raising RuntimeError; tag it as a load failure.
            detail = str(e).strip()
            if detail:
                raise DocumentLoadError(
                    f"docling-parse could not load document {self.document_hash}: {detail}"
                ) from e
            raise DocumentLoadError(
                f"docling-parse could not load document {self.document_hash}."
            ) from e

        if self.dp_doc is None:
            raise DocumentLoadError(
                f"docling-parse could not load document {self.document_hash}."
            )

    def page_count(self) -> int:
        # return len(self._pdoc)  # To be replaced with docling-parse API

        len_1 = len(self._pdoc)
        assert self.dp_doc is not None
        len_2 = self.dp_doc.number_of_pages()

        if len_1 != len_2:
            _log.error(f"Inconsistent number of pages: {len_1}!={len_2}")

        return len_2

    def load_page(
        self, page_no: int, create_words: bool = True, create_textlines: bool = True
    ) -> DoclingParsePageBackend:
        assert self.dp_doc is not None
        with pypdfium2_lock:
            ppage = self._pdoc[page_no]

        return DoclingParsePageBackend(
            dp_doc=self.dp_doc,
            page_obj=ppage,
            page_no=page_no,
            create_words=create_words,
            create_textlines=create_textlines,
        )

    def is_valid(self) -> bool:
        return self.page_count() > 0

    def get_document_outline(self) -> list[_PdfOutlineItem]:
        """Extract the outline via docling-parse's native table-of-contents (no pypdfium2)."""
        if self.dp_doc is None:
            return []
        return extract_outline_from_docling_parse(self.dp_doc)

    def _close_native_document(self) -> None:
        if self.dp_doc is not None:
            self.dp_doc.unload()
            self.dp_doc = None

        if self._pdoc is not None:
            with pypdfium2_lock:
                try:
                    self._pdoc.close()
                except Exception:
                    pass
            self._pdoc = None


def _resolve_threaded_page_numbers(
    path_or_stream: Union[BytesIO, Path],
    password: Optional[str],
    page_range: tuple[int, int],
) -> list[int] | None:
    start_page, end_page = page_range

    if page_range == DEFAULT_PAGE_RANGE:
        return None

    with pypdfium2_lock:
        pdoc = pdfium.PdfDocument(path_or_stream, password=password)
        try:
            page_count = len(pdoc)
        finally:
            pdoc.close()

    clipped_end_page = min(end_page, page_count)
    if start_page > clipped_end_page:
        return []

    return list(range(start_page, clipped_end_page + 1))


class ThreadedDoclingParsePageBackend(PdfPageBackend):
    def __init__(
        self,
        result: PageParseResult,
        doc_backend: Optional["ThreadedDoclingParseDocumentBackend"] = None,
    ):
        self._result = result
        self._seg_page: Optional[SegmentedPdfPage] = None
        # The owning document backend lazily provides a shared pypdfium2 document
        # used to render page rasters with the same engine as the serial backend,
        # so the downstream layout model sees an identical image (issue #3512).
        self._doc_backend = doc_backend
        self._ppage: Optional[PdfPage] = None

    @property
    def page_no(self) -> int:
        return self._result.page_number

    def is_valid(self) -> bool:
        return self._result.success

    def get_text_in_rect(self, bbox: BoundingBox) -> str:
        segmented_page = self.get_segmented_page()
        if segmented_page is None:
            return ""

        text_piece = ""
        for cell in segmented_page.textline_cells:
            cell_bbox = cell.rect.to_bounding_box()
            overlap_frac = cell_bbox.intersection_over_self(bbox)
            if overlap_frac > 0.5:
                if text_piece:
                    text_piece += " "
                text_piece += cell.text

        return text_piece

    def get_segmented_page(self) -> Optional[SegmentedPdfPage]:
        if not self.is_valid():
            return None
        if self._seg_page is None:
            seg_page = self._result.get_page()
            page_height = seg_page.dimension.height
            for tc in seg_page.textline_cells:
                tc.to_top_left_origin(page_height)
            for tc in seg_page.char_cells:
                tc.to_top_left_origin(page_height)
            for tc in seg_page.word_cells:
                tc.to_top_left_origin(page_height)
            self._seg_page = seg_page
        return self._seg_page

    def get_text_cells(self) -> Iterable[TextCell]:
        segmented_page = self.get_segmented_page()
        if segmented_page is None:
            return []
        return segmented_page.textline_cells

    def get_bitmap_rects(self, scale: float = 1) -> Iterable[BoundingBox]:
        segmented_page = self.get_segmented_page()
        if segmented_page is None:
            return []

        page_height = self.get_size().height
        cropboxes: list[BoundingBox] = []
        for image_resource in segmented_page.bitmap_resources:
            cropbox = image_resource.rect.to_bounding_box().to_top_left_origin(
                page_height
            )
            if cropbox.area() > 0:
                cropboxes.append(cropbox.scaled(scale=scale))
        return cropboxes

    def get_page_image(
        self, scale: float = 1, cropbox: Optional[BoundingBox] = None
    ) -> Image.Image:
        if self._doc_backend is not None:
            pdoc = self._doc_backend._ensure_pdoc()
            with pypdfium2_lock:
                if self._ppage is None:
                    self._ppage = pdoc[self._result.page_number - 1]
            return _render_page_image_via_pdfium(
                self._ppage, self.get_size(), scale, cropbox
            ).convert("RGB")
        # Fallback: docling-parse rasterizer (renders a slightly different image).
        return self._result.get_image(scale=scale, cropbox=cropbox).convert("RGB")

    def get_size(self) -> Size:
        return Size(width=self._result.page_width, height=self._result.page_height)

    def unload(self) -> None:
        if self._ppage is not None:
            with pypdfium2_lock:
                self._ppage.close()
            self._ppage = None


class ThreadedDoclingParseDocumentBackend(PdfDocumentBackend):
    supports_random_page_access = False

    def __init__(
        self,
        in_doc: "InputDocument",
        path_or_stream: Union[BytesIO, Path],
        options: Optional[PdfBackendOptions] = None,
    ):
        if options is None:
            options = PdfBackendOptions()
        super().__init__(in_doc, path_or_stream, options)
        self.options: PdfBackendOptions
        self._closed = False

        password = (
            self.options.password.get_secret_value() if self.options.password else None
        )
        requested_page_numbers = _resolve_threaded_page_numbers(
            self.path_or_stream,
            password,
            in_doc.limits.page_range,
        )

        parser_threads = (
            self.options.parser_threads
            if isinstance(self.options, ThreadedDoclingParseBackendOptions)
            and self.options.parser_threads is not None
            else AcceleratorOptions().num_threads
        )
        render_config = RenderConfig()
        render_config.scale = 1.0
        native_memory_release_interval = (
            self.options.release_native_memory_every_n_pages
            if isinstance(self.options, ThreadedDoclingParseBackendOptions)
            else 128
        )
        decode_config = _make_docling_parse_decode_config(
            enforce_same_font=self.options.enforce_same_font,
            release_native_memory_every_n_pages=native_memory_release_interval,
        )
        content_config = _make_docling_parse_page_content_config(
            create_words=True,
            create_textlines=True,
        )

        self.parser = DoclingThreadedPdfParser(
            parser_config=ThreadedPdfParserConfig(
                loglevel="fatal",
                threads=parser_threads,
                render_config=render_config,
                page_content_config=content_config,
            ),
            decode_config=decode_config,
        )
        self.doc_key = self.parser.load(
            self.path_or_stream,
            password=password,
            page_numbers=requested_page_numbers,
        )

        self._password = password
        # pypdfium2 document, opened lazily on the first page render. Page rasters
        # are rendered with pypdfium2 (same engine as the serial backend) so the
        # downstream layout model receives an identical image; the docling-parse
        # rasterizer produces a subtly different image that can change layout
        # detection and drop tables (issue #3512). Kept lazy so that documents
        # whose pages are never rendered incur no pypdfium2 load.
        self._pdoc: Optional[pdfium.PdfDocument] = None

    def _ensure_pdoc(self) -> pdfium.PdfDocument:
        with pypdfium2_lock:
            if self._pdoc is None:
                if isinstance(self.path_or_stream, BytesIO):
                    self.path_or_stream.seek(0)
                self._pdoc = pdfium.PdfDocument(
                    self.path_or_stream, password=self._password
                )
            return self._pdoc

    def is_valid(self) -> bool:
        return not self._closed and self.page_count() > 0

    def page_count(self) -> int:
        return self.parser.page_count(self.doc_key)

    def get_document_outline(self) -> list[_PdfOutlineItem]:
        """Extract the outline via docling-parse.

        The threaded parser exposes no table-of-contents accessor, so a lightweight lazy
        docling-parse document is loaded purely to read the (cheap, structure-only) outline.
        The pypdfium2 document used for page rendering has no outline accessor either, so it
        is not reused here.
        """
        password = (
            self.options.password.get_secret_value() if self.options.password else None
        )
        if isinstance(self.path_or_stream, BytesIO):
            self.path_or_stream.seek(0)
        dp_doc = DoclingPdfParser(loglevel="fatal").load(
            path_or_stream=self.path_or_stream, lazy=True, password=password
        )
        if dp_doc is None:
            return []
        try:
            return extract_outline_from_docling_parse(dp_doc)
        finally:
            dp_doc.unload()

    def load_page(self, page_no: int) -> PdfPageBackend:
        raise NotImplementedError(
            "ThreadedDoclingParseDocumentBackend only supports iter_pages()."
        )

    def iter_pages(self) -> Iterator[ThreadedDoclingParsePageBackend]:
        for result in self.parser.iterate_results():
            yield ThreadedDoclingParsePageBackend(result, self)

    def unload(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.parser.unload(self.doc_key)
        if self._pdoc is not None:
            with pypdfium2_lock:
                self._pdoc.close()
            self._pdoc = None
        super().unload()
