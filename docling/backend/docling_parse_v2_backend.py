import logging
import random
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

import pypdfium2 as pdfium
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import (
    BoundingRectangle,
    PdfPageBoundaryType,
    PdfPageGeometry,
    SegmentedPdfPage,
    TextCell,
)
from docling_parse.pdf_parsers import pdf_parser_v2
from PIL import Image, ImageDraw
from pypdfium2 import PdfPage

from docling.backend.pdf_backend import PdfDocumentBackend, PdfPageBackend
from docling.backend.pypdfium2_backend import get_pdf_page_geometry
from docling.datamodel.base_models import Size
from docling.utils.locks import pypdfium2_lock

if TYPE_CHECKING:
    from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)


class DoclingParseV2PageBackend(PdfPageBackend):
    def __init__(
        self, parser: pdf_parser_v2, document_hash: str, page_no: int, page_obj: PdfPage
    ):
        self._ppage = page_obj
        parsed_page = parser.parse_pdf_from_key_on_page(document_hash, page_no)

        self.valid = "pages" in parsed_page and len(parsed_page["pages"]) == 1
        if self.valid:
            self._dpage = parsed_page["pages"][0]
        else:
            _log.info(
                f"An error occurred when loading page {page_no} of document {document_hash}."
            )

    def is_valid(self) -> bool:
        return self.valid

    def _compute_text_cells(self) -> List[TextCell]:
        """Compute text cells from docling-parse v2 data."""
        cells: List[TextCell] = []
        cell_counter = 0

        if not self.valid:
            return cells

        page_size = self.get_size()

        parser_width = self._dpage["sanitized"]["dimension"]["width"]
        parser_height = self._dpage["sanitized"]["dimension"]["height"]

        cells_data = self._dpage["sanitized"]["cells"]["data"]
        cells_header = self._dpage["sanitized"]["cells"]["header"]

        for i, cell_data in enumerate(cells_data):
            x0 = cell_data[cells_header.index("x0")]
            y0 = cell_data[cells_header.index("y0")]
            x1 = cell_data[cells_header.index("x1")]
            y1 = cell_data[cells_header.index("y1")]

            if x1 < x0:
                x0, x1 = x1, x0
            if y1 < y0:
                y0, y1 = y1, y0

            text_piece = cell_data[cells_header.index("text")]
            cells.append(
                TextCell(
                    index=cell_counter,
                    text=text_piece,
                    orig=text_piece,
                    from_ocr=False,
                    rect=BoundingRectangle.from_bounding_box(
                        BoundingBox(
                            l=x0 * page_size.width / parser_width,
                            b=y0 * page_size.height / parser_height,
                            r=x1 * page_size.width / parser_width,
                            t=y1 * page_size.height / parser_height,
                            coord_origin=CoordOrigin.BOTTOMLEFT,
                        )
                    ).to_top_left_origin(page_size.height),
                )
            )
            cell_counter += 1

        return cells

    def get_text_in_rect(self, bbox: BoundingBox) -> str:
        if not self.valid:
            return ""
        # Find intersecting cells on the page
        text_piece = ""
        page_size = self.get_size()

        parser_width = self._dpage["sanitized"]["dimension"]["width"]
        parser_height = self._dpage["sanitized"]["dimension"]["height"]

        scale = (
            1  # FIX - Replace with param in get_text_in_rect across backends (optional)
        )

        cells_data = self._dpage["sanitized"]["cells"]["data"]
        cells_header = self._dpage["sanitized"]["cells"]["header"]

        for i, cell_data in enumerate(cells_data):
            x0 = cell_data[cells_header.index("x0")]
            y0 = cell_data[cells_header.index("y0")]
            x1 = cell_data[cells_header.index("x1")]
            y1 = cell_data[cells_header.index("y1")]

            cell_bbox = BoundingBox(
                l=x0 * scale * page_size.width / parser_width,
                b=y0 * scale * page_size.height / parser_height,
                r=x1 * scale * page_size.width / parser_width,
                t=y1 * scale * page_size.height / parser_height,
                coord_origin=CoordOrigin.BOTTOMLEFT,
            ).to_top_left_origin(page_height=page_size.height * scale)

            overlap_frac = cell_bbox.intersection_over_self(bbox)

            if overlap_frac > 0.5:
                if len(text_piece) > 0:
                    text_piece += " "
                text_piece += cell_data[cells_header.index("text")]

        return text_piece

    def get_segmented_page(self) -> Optional[SegmentedPdfPage]:
        if not self.valid:
            return None

        text_cells = self._compute_text_cells()

        # Get the PDF page geometry from pypdfium2
        dimension = get_pdf_page_geometry(self._ppage)

        # Create SegmentedPdfPage
        return SegmentedPdfPage(
            dimension=dimension,
            textline_cells=text_cells,
            char_cells=[],
            word_cells=[],
            has_textlines=len(text_cells) > 0,
            has_words=False,
            has_chars=False,
        )

    def get_text_cells(self) -> Iterable[TextCell]:
        return self._compute_text_cells()

    def get_bitmap_rects(self, scale: float = 1) -> Iterable[BoundingBox]:
        AREA_THRESHOLD = 0  # 32 * 32

        images = self._dpage["sanitized"]["images"]["data"]
        images_header = self._dpage["sanitized"]["images"]["header"]

        for row in images:
            x0 = row[images_header.index("x0")]
            y0 = row[images_header.index("y0")]
            x1 = row[images_header.index("x1")]
            y1 = row[images_header.index("y1")]

            cropbox = BoundingBox.from_tuple(
                (x0, y0, x1, y1), origin=CoordOrigin.BOTTOMLEFT
            ).to_top_left_origin(self.get_size().height)

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

    def unload(self):
        self._ppage = None
        self._dpage = None


class DoclingParseV2DocumentBackend(PdfDocumentBackend):
    def __init__(self, in_doc: "InputDocument", path_or_stream: Union[BytesIO, Path]):
        super().__init__(in_doc, path_or_stream)

        with pypdfium2_lock:
            self._pdoc = pdfium.PdfDocument(self.path_or_stream)
            self.parser = pdf_parser_v2("fatal")

            success = False
            if isinstance(self.path_or_stream, BytesIO):
                success = self.parser.load_document_from_bytesio(
                    self.document_hash, self.path_or_stream
                )
            elif isinstance(self.path_or_stream, Path):
                success = self.parser.load_document(
                    self.document_hash, str(self.path_or_stream)
                )

            if not success:
                raise RuntimeError(
                    f"docling-parse v2 could not load document {self.document_hash}."
                )

    def page_count(self) -> int:
        # return len(self._pdoc)  # To be replaced with docling-parse API

        len_1 = len(self._pdoc)
        len_2 = self.parser.number_of_pages(self.document_hash)

        if len_1 != len_2:
            _log.error(f"Inconsistent number of pages: {len_1}!={len_2}")

        return len_2

    def load_page(self, page_no: int) -> DoclingParseV2PageBackend:
        with pypdfium2_lock:
            return DoclingParseV2PageBackend(
                self.parser, self.document_hash, page_no, self._pdoc[page_no]
            )

    def is_valid(self) -> bool:
        return self.page_count() > 0

    def unload(self):
        super().unload()
        self.parser.unload_document(self.document_hash)
        with pypdfium2_lock:
            self._pdoc.close()
            self._pdoc = None
