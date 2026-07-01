import logging
import multiprocessing as mp
import queue
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import pypdfium2 as pdfium
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import SegmentedPdfPage, TextCell
from docling_parse.pdf_parser import DoclingPdfParser, PdfDocument
from PIL import Image
from pypdfium2 import PdfPage

from docling.backend.pdf_backend import PdfDocumentBackend, PdfPageBackend
from docling.datamodel.base_models import Size
from docling.utils.locks import pypdfium2_lock

if TYPE_CHECKING:
    from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)

_PAGE_LOAD_TIMEOUT_SECS = 120  # 2 minutes per page


def _page_worker_main(
    req_queue: mp.Queue,
    resp_queue: mp.Queue,
    path_or_stream: Union[str, bytes],
) -> None:
    """Persistent worker process: loads PDF once, serves page requests until sentinel."""
    try:
        from docling_parse.pdf_parser import DoclingPdfParser

        parser = DoclingPdfParser(loglevel="fatal")
        target = BytesIO(path_or_stream) if isinstance(path_or_stream, bytes) else Path(path_or_stream)
        doc = parser.load(path_or_stream=target)
        resp_queue.put(("ready", None))
    except Exception as e:
        resp_queue.put(("init_error", str(e)))
        return

    while True:
        request = req_queue.get()
        if request is None:  # sentinel — shut down
            break
        page_no, create_words, create_textlines = request
        try:
            seg_page = doc.get_page(page_no + 1, create_words=create_words, create_textlines=create_textlines)
            resp_queue.put(("ok", seg_page))
        except Exception as e:
            resp_queue.put(("error", str(e)))


class DoclingParseV4PageBackend(PdfPageBackend):
    def __init__(self, parsed_page: SegmentedPdfPage, page_obj: PdfPage):
        self._ppage = page_obj
        self._dpage = parsed_page
        self.valid = parsed_page is not None

    def is_valid(self) -> bool:
        return self.valid

    def get_text_in_rect(self, bbox: BoundingBox) -> str:
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
        return self._dpage

    def get_text_cells(self) -> Iterable[TextCell]:
        return self._dpage.textline_cells

    def get_bitmap_rects(self, scale: float = 1) -> Iterable[BoundingBox]:
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

        with pypdfium2_lock:
            image = (
                self._ppage.render(
                    scale=scale * 1.5,
                    rotation=0,  # no additional rotation
                    crop=padbox.as_tuple(),
                )
                .to_pil()
                .resize(
                    size=(round(cropbox.width * scale), round(cropbox.height * scale))
                )
            )  # We resize the image from 1.5x the given scale to make it sharper.

        return image

    def get_size(self) -> Size:
        with pypdfium2_lock:
            return Size(width=self._ppage.get_width(), height=self._ppage.get_height())

        # TODO: Take width and height from docling-parse.
        # return Size(
        #    width=self._dpage.dimension.width,
        #    height=self._dpage.dimension.height,
        # )

    def unload(self):
        self._ppage = None
        self._dpage = None


class DoclingParseV4DocumentBackend(PdfDocumentBackend):
    def __init__(self, in_doc: "InputDocument", path_or_stream: Union[BytesIO, Path]):
        super().__init__(in_doc, path_or_stream)

        with pypdfium2_lock:
            self._pdoc = pdfium.PdfDocument(self.path_or_stream)
        self.parser = DoclingPdfParser(loglevel="fatal")
        self.dp_doc: PdfDocument = self.parser.load(path_or_stream=self.path_or_stream)
        success = self.dp_doc is not None

        if not success:
            raise RuntimeError(
                f"docling-parse v4 could not load document {self.document_hash}."
            )

        worker_target: Union[str, bytes] = (
            self.path_or_stream.getvalue()
            if isinstance(self.path_or_stream, BytesIO)
            else str(self.path_or_stream)
        )
        self._req_queue: mp.Queue = mp.Queue()
        self._resp_queue: mp.Queue = mp.Queue()
        self._worker = mp.Process(
            target=_page_worker_main,
            args=(self._req_queue, self._resp_queue, worker_target),
            daemon=True,
        )
        self._worker.start()
        try:
            status, msg = self._resp_queue.get(timeout=30)
            if status != "ready":
                raise RuntimeError(f"docling-parse worker init failed: {msg}")
        except queue.Empty:
            self._worker.kill()
            raise RuntimeError("docling-parse worker timed out during init.")
        self._worker_alive = True

    def page_count(self) -> int:
        # return len(self._pdoc)  # To be replaced with docling-parse API

        len_1 = len(self._pdoc)
        len_2 = self.dp_doc.number_of_pages()

        if len_1 != len_2:
            _log.error(f"Inconsistent number of pages: {len_1}!={len_2}")

        return len_2

    def _kill_worker(self) -> None:
        self._worker_alive = False
        if self._worker.is_alive():
            self._worker.terminate()
            self._worker.join(timeout=5)
            if self._worker.is_alive():
                self._worker.kill()
                self._worker.join()

    def load_page(
        self, page_no: int, create_words: bool = True, create_textlines: bool = True
    ) -> PdfPageBackend:
        from docling.backend.pypdfium2_backend import PyPdfiumPageBackend

        if not self._worker_alive:
            with pypdfium2_lock:
                return PyPdfiumPageBackend(self._pdoc, self.document_hash, page_no)

        self._req_queue.put((page_no, create_words, create_textlines))
        try:
            status, result = self._resp_queue.get(timeout=_PAGE_LOAD_TIMEOUT_SECS)
        except queue.Empty:
            _log.warning(
                f"docling-parse worker timed out on page {page_no + 1} "
                f"(>{_PAGE_LOAD_TIMEOUT_SECS}s). Killing worker, switching to pdfium."
            )
            self._kill_worker()
            with pypdfium2_lock:
                return PyPdfiumPageBackend(self._pdoc, self.document_hash, page_no)

        if status == "error":
            _log.warning(
                f"docling-parse worker error on page {page_no + 1}: {result}. "
                "Falling back to pdfium backend."
            )
            with pypdfium2_lock:
                return PyPdfiumPageBackend(self._pdoc, self.document_hash, page_no)

        seg_page: SegmentedPdfPage = result

        # In Docling, all TextCell instances are expected with top-left origin.
        [tc.to_top_left_origin(seg_page.dimension.height) for tc in seg_page.textline_cells]
        [tc.to_top_left_origin(seg_page.dimension.height) for tc in seg_page.char_cells]
        [tc.to_top_left_origin(seg_page.dimension.height) for tc in seg_page.word_cells]

        with pypdfium2_lock:
            return DoclingParseV4PageBackend(seg_page, self._pdoc[page_no])

    def is_valid(self) -> bool:
        return self.page_count() > 0

    def unload(self):
        super().unload()
        # Shut down the persistent worker process
        if self._worker_alive:
            try:
                self._req_queue.put(None)  # sentinel
                self._worker.join(timeout=5)
            except Exception:
                pass
            self._kill_worker()
        # Unload docling-parse document first
        if self.dp_doc is not None:
            self.dp_doc.unload()
            self.dp_doc = None

        # Then close pypdfium2 document with proper locking
        if self._pdoc is not None:
            with pypdfium2_lock:
                try:
                    self._pdoc.close()
                except Exception:
                    # Ignore cleanup errors
                    pass
            self._pdoc = None
