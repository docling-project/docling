import logging
import random
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Union

import pypdfium2 as pdfium
from docling_core.types.doc import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.page import BoundingRectangle, SegmentedPdfPage, TextCell
from docling_parse.pdf_parsers import pdf_parser_v1
from PIL import Image, ImageDraw
from pypdfium2 import PdfPage

from docling.backend.pdf_backend import PdfDocumentBackend, PdfPageBackend
from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)


class DoclingParsePageBackend(PdfPageBackend):
    def __init__(
        self, parser: pdf_parser_v1, document_hash: str, page_no: int, page_obj: PdfPage
    ):
        self._ppage = page_obj
        parsed_page = parser.parse_pdf_from_key_on_page(document_hash, page_no)

        self.valid = "pages" in parsed_page
        if self.valid:
            self._dpage = parsed_page["pages"][0]
        else:
            _log.info(
                f"An error occurred when loading page {page_no} of document {document_hash}."
            )

    def is_valid(self) -> bool:
        return self.valid

    def get_text_in_rect(self, bbox: BoundingBox) -> str:
        if not self.valid:
            return ""
        # Find intersecting cells on the page
        text_piece = ""
        page_size = self.get_size()
        parser_width = self._dpage["width"]
        parser_height = self._dpage["height"]

        scale = (
            1  # FIX - Replace with param in get_text_in_rect across backends (optional)
        )

        for i in range(len(self._dpage["cells"])):
            rect = self._dpage["cells"][i]["box"]["device"]
            x0, y0, x1, y1 = rect
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
                text_piece += self._dpage["cells"][i]["content"]["rnormalized"]

        return text_piece

    def get_segmented_page(self) -> Optional[SegmentedPdfPage]:
        return None

    def get_text_cells(self) -> Iterable[TextCell]:
        cells: List[TextCell] = []
        cell_counter = 0

        if not self.valid:
            return cells

        page_size = self.get_size()

        parser_width = self._dpage["width"]
        parser_height = self._dpage["height"]

        for i in range(len(self._dpage["cells"])):
            rect = self._dpage["cells"][i]["box"]["device"]
            x0, y0, x1, y1 = rect

            if x1 < x0:
                x0, x1 = x1, x0
            if y1 < y0:
                y0, y1 = y1, y0

            text_piece = self._dpage["cells"][i]["content"]["rnormalized"]
            cells.append(
                TextCell(
                    index=cell_counter,
                    text=text_piece,
                    orig=text_piece,
                    from_ocr=False,
                    rect=BoundingRectangle.from_bounding_box(
                        BoundingBox(
                            # l=x0, b=y0, r=x1, t=y1,
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

        def draw_clusters_and_cells():
            image = (
                self.get_page_image()
            )  # make new image to avoid drawing on the saved ones
            draw = ImageDraw.Draw(image)
            for c in cells:
                x0, y0, x1, y1 = c.rect.to_bounding_box().as_tuple()
                cell_color = (
                    random.randint(30, 140),
                    random.randint(30, 140),
                    random.randint(30, 140),
                )
                draw.rectangle([(x0, y0), (x1, y1)], outline=cell_color)
            image.show()

        # before merge:
        # draw_clusters_and_cells()

        # cells = merge_horizontal_cells(cells)

        # after merge:
        # draw_clusters_and_cells()

        return cells

    def get_bitmap_rects(self, scale: float = 1) -> Iterable[BoundingBox]:
        AREA_THRESHOLD = 0  # 32 * 32

        for i in range(len(self._dpage["images"])):
            bitmap = self._dpage["images"][i]
            cropbox = BoundingBox.from_tuple(
                bitmap["box"], origin=CoordOrigin.BOTTOMLEFT
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

        image = (
            self._ppage.render(
                scale=scale * 1.5,
                rotation=0,  # no additional rotation
                crop=padbox.as_tuple(),
            )
            .to_pil()
            .resize(size=(round(cropbox.width * scale), round(cropbox.height * scale)))
        )  # We resize the image from 1.5x the given scale to make it sharper.

        return image

    def get_size(self) -> Size:
        return Size(width=self._ppage.get_width(), height=self._ppage.get_height())

    def unload(self):
        self._ppage = None
        self._dpage = None


class DoclingParseDocumentBackend(PdfDocumentBackend):
    def __init__(self, in_doc: "InputDocument", path_or_stream: Union[BytesIO, Path]):
        super().__init__(in_doc, path_or_stream)

        self._pdoc = pdfium.PdfDocument(self.path_or_stream)
        self.parser = pdf_parser_v1()

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
                f"docling-parse could not load document with hash {self.document_hash}."
            )

    def page_count(self) -> int:
        return len(self._pdoc)  # To be replaced with docling-parse API

    def load_page(self, page_no: int) -> DoclingParsePageBackend:
        return DoclingParsePageBackend(
            self.parser, self.document_hash, page_no, self._pdoc[page_no]
        )

    def is_valid(self) -> bool:
        return self.page_count() > 0

    def unload(self):
        super().unload()
        self.parser.unload_document(self.document_hash)
        self._pdoc.close()
        self._pdoc = None
