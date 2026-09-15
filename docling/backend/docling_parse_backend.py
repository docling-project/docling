# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import warnings
from collections.abc import Iterable, Iterator
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from docling_core.types.doc import BoundingBox, Size
from docling_core.types.doc.page import (
    PdfCellRenderingMode,
    PdfTextCell,
    SegmentedPdfPage,
    TextCell,
)
from docling_parse.pdf_parser import (
    ContentConfig,
    ContentLevel,
    DecodeConfig,
    DoclingThreadedPdfParser,
    PageParseResult,
    RenderConfig,
    ThreadedPdfParserConfig,
)
from PIL import Image

from docling.backend.pdf_backend import PdfDocumentBackend, PdfPageBackend
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.backend_options import (
    PdfBackendOptions,
    ThreadedDoclingParseBackendOptions,
)
from docling.exceptions import DocumentLoadError
from docling.utils.pdf_outline import (
    _PdfOutlineItem,
    extract_outline_from_docling_parse,
)

if TYPE_CHECKING:
    from docling.datamodel.document import InputDocument

# PDF 32000 text rendering modes that paint no ink. docling-parse applies the same filter
# natively when answering `intersects_with()`, so the cell-level view has to match it.
_INVISIBLE_RENDERING_MODES = frozenset(
    {PdfCellRenderingMode.INVISIBLE, PdfCellRenderingMode.ONLY_CLIPPING}
)


def _visible_text_cells(cells: Iterable[TextCell]) -> list[TextCell]:
    """Keep only the cells that paint ink on the page"""
    return [
        cell
        for cell in cells
        if not isinstance(cell, PdfTextCell)
        or cell.rendering_mode not in _INVISIBLE_RENDERING_MODES
    ]


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
    materialize_char_cells: bool = False,
    compute_shapes: bool = True,
    include_bitmap_bytes: bool = False,
) -> ContentConfig:
    compute = ContentLevel.COMPUTE
    materialize = ContentLevel.COMPUTE_AND_MATERIALIZE
    skip = ContentLevel.SKIP

    return ContentConfig(
        char_cells_content_level=(
            materialize
            if materialize_char_cells
            else compute
            if (create_words or create_textlines)
            else skip
        ),
        word_cells_content_level=materialize if create_words else skip,
        line_cells_content_level=materialize if create_textlines else skip,
        # The threaded parser renders the page image from this same decode, so
        # shapes must be computed there or the render loses all vector content.
        shapes_content_level=compute if compute_shapes else skip,
        bitmaps_content_level=materialize,
        # Decoding the bitmap bytes is only needed when the embedded images are
        # consumed as such; OCR gets by with the bitmap rectangles alone.
        include_bitmap_bytes=include_bitmap_bytes,
    )


class ThreadedDoclingParsePageBackend(PdfPageBackend):
    def __init__(self, result: PageParseResult, rendered: bool = True):
        self._result = result
        self._rendered = rendered
        self._seg_page: Optional[SegmentedPdfPage] = None

    @property
    def page_no(self) -> int:
        return self._result.page_number

    def is_valid(self) -> bool:
        return self._result.success

    def get_error_message(self) -> str:
        return self._result.error_message

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

    def get_visible_text_cells(self) -> Optional[list[TextCell]]:
        segmented_page = self.get_segmented_page()
        if segmented_page is None:
            return []
        return _visible_text_cells(segmented_page.textline_cells)

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

    def has_content_in(
        self,
        *,
        bbox: BoundingBox,
        chars: bool = False,
        shapes: bool = True,
        bitmaps: bool = True,
    ) -> Optional[bool]:
        if not self.is_valid():
            return False
        return self._result.intersects_with(
            bbox=bbox, chars=chars, shapes=shapes, bitmaps=bitmaps
        )

    def get_shape_lines(
        self,
        *,
        horizontal: bool = True,
        vertical: bool = True,
        tolerance: float = 1e-3,
    ) -> Optional[list[BoundingBox]]:
        if not self.is_valid():
            return []

        page_height = self.get_size().height
        return [
            bbox.to_top_left_origin(page_height)
            for bbox in self._result.get_shape_lines(
                horizontal=horizontal, vertical=vertical, tolerance=tolerance
            )
        ]

    def get_connected_shape_bounding_boxes(
        self, *, tolerance: float = 0.0
    ) -> Optional[list[BoundingBox]]:
        if not self.is_valid():
            return []

        page_height = self.get_size().height
        return [
            bbox.to_top_left_origin(page_height)
            for bbox in self._result.get_connected_shape_bounding_boxes(
                tolerance=tolerance
            )
        ]

    def get_page_image(
        self, scale: float = 1, cropbox: Optional[BoundingBox] = None
    ) -> Image.Image:
        if not self._rendered:
            raise RuntimeError(
                "This backend was configured with render_pages=False, so page "
                f"{self.page_no} was parsed but never rendered and no page image exists."
            )
        return self._result.get_image(scale=scale, cropbox=cropbox).convert("RGB")

    def get_size(self) -> Size:
        return Size(width=self._result.page_width, height=self._result.page_height)

    def unload(self) -> None:
        return None


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
        self._iterating = False

        password = (
            self.options.password.get_secret_value() if self.options.password else None
        )
        threaded_options = (
            self.options
            if isinstance(self.options, ThreadedDoclingParseBackendOptions)
            else ThreadedDoclingParseBackendOptions()
        )
        parser_threads = (
            threaded_options.parser_threads
            if threaded_options.parser_threads is not None
            else AcceleratorOptions().num_threads
        )
        self._render_pages = threaded_options.render_pages
        render_config: RenderConfig | None = None
        if self._render_pages:
            render_config = RenderConfig()
            render_config.scale = threaded_options.render_scale
        decode_config = _make_docling_parse_decode_config(
            enforce_same_font=self.options.enforce_same_font,
            release_native_memory_every_n_pages=(
                threaded_options.release_native_memory_every_n_pages
            ),
        )
        content_config = _make_docling_parse_page_content_config(
            create_words=True,
            create_textlines=True,
            materialize_char_cells=self.options._materialize_char_cells,
            # Shapes only matter for the render; skip them when nothing is rendered.
            compute_shapes=self._render_pages,
            include_bitmap_bytes=self.options.include_bitmap_images,
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
        try:
            # The threaded parser derives its document key by hashing from the current
            # stream offset, so the stream has to be rewound for that key to cover the
            # whole document.
            if isinstance(self.path_or_stream, BytesIO):
                self.path_or_stream.seek(0)
            self.doc_key = self.parser.load(
                self.path_or_stream,
                password=password,
                page_range=in_doc.limits.page_range,
            )
        except (RuntimeError, ValueError) as e:
            # The threaded parser surfaces native parse failures on unreadable bytes
            # as RuntimeError or ValueError. Tag both as load failures.
            detail = str(e).strip()
            if detail:
                raise DocumentLoadError(
                    f"docling-parse could not load document {self.document_hash}: {detail}"
                ) from e
            raise DocumentLoadError(
                f"docling-parse could not load document {self.document_hash}."
            ) from e

    def is_valid(self) -> bool:
        return not self._closed and self.page_count() > 0

    def page_count(self) -> int:
        return self.parser.page_count(self.doc_key)

    def get_document_outline(self) -> list[_PdfOutlineItem]:
        """Extract the outline from the threaded parser's document annotations."""
        annotations = self.parser.get_annotations(self.doc_key)
        toc = annotations.table_of_contents if annotations is not None else None
        return extract_outline_from_docling_parse(toc)

    def load_page(self, page_no: int) -> PdfPageBackend:
        raise NotImplementedError(
            "ThreadedDoclingParseDocumentBackend only supports iter_pages()."
        )

    def iter_pages(self) -> Iterator[ThreadedDoclingParsePageBackend]:
        self._iterating = True
        for result in self.parser.iterate_results():
            yield ThreadedDoclingParsePageBackend(result, rendered=self._render_pages)
        self._iterating = False

    def unload(self) -> None:
        if self._closed:
            return
        self._closed = True
        # The parser cannot unload while an iteration is active. Drain the raw
        # tasks rather than creating page backends for work a consumer abandoned.
        while self.parser.has_tasks():
            self.parser.get_task()
        self._iterating = False
        self.parser.unload(self.doc_key)
        super().unload()


class DoclingParseDocumentBackend(ThreadedDoclingParseDocumentBackend):
    """Deprecated alias for :class:`ThreadedDoclingParseDocumentBackend`."""

    def __init__(
        self,
        in_doc: "InputDocument",
        path_or_stream: Union[BytesIO, Path],
        options: Optional[PdfBackendOptions] = None,
    ):
        warnings.warn(
            "DoclingParseDocumentBackend is deprecated; use "
            "ThreadedDoclingParseDocumentBackend instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(in_doc, path_or_stream, options)
