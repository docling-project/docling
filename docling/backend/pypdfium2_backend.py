import logging
import random
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from docling_core.types.doc import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.page import (
    BoundingRectangle,
    PdfPageBoundaryType,
    PdfPageGeometry,
    SegmentedPdfPage,
    TextCell,
)
from PIL import Image, ImageDraw
from pypdfium2 import PdfTextPage
from pypdfium2._helpers.misc import PdfiumError

from docling.backend.pdf_backend import PdfDocumentBackend, PdfPageBackend
from docling.utils.locks import pypdfium2_lock


def get_pdf_page_geometry(
    ppage: pdfium.PdfPage,
    angle: float = 0.0,
    boundary_type: PdfPageBoundaryType = PdfPageBoundaryType.CROP_BOX,
) -> PdfPageGeometry:
    """
    Create PdfPageGeometry from a pypdfium2 PdfPage object.

    Args:
        ppage: pypdfium2 PdfPage object
        angle: Page rotation angle in degrees (default: 0.0)
        boundary_type: The boundary type for the page (default: CROP_BOX)

    Returns:
        PdfPageGeometry with all the different bounding boxes properly set
    """
    with pypdfium2_lock:
        # Get the main bounding box (intersection of crop_box and media_box)
        bbox_tuple = ppage.get_bbox()
        bbox = BoundingBox.from_tuple(bbox_tuple, CoordOrigin.BOTTOMLEFT)

        # Get all the different page boxes from pypdfium2
        media_box_tuple = ppage.get_mediabox()
        crop_box_tuple = ppage.get_cropbox()
        art_box_tuple = ppage.get_artbox()
        bleed_box_tuple = ppage.get_bleedbox()
        trim_box_tuple = ppage.get_trimbox()

        # Convert to BoundingBox objects using existing from_tuple method
        # pypdfium2 returns (x0, y0, x1, y1) in PDF coordinate system (bottom-left origin)
        # Use bbox as fallback when specific box types are not defined
        media_bbox = (
            BoundingBox.from_tuple(media_box_tuple, CoordOrigin.BOTTOMLEFT)
            if media_box_tuple
            else bbox
        )
        crop_bbox = (
            BoundingBox.from_tuple(crop_box_tuple, CoordOrigin.BOTTOMLEFT)
            if crop_box_tuple
            else bbox
        )
        art_bbox = (
            BoundingBox.from_tuple(art_box_tuple, CoordOrigin.BOTTOMLEFT)
            if art_box_tuple
            else bbox
        )
        bleed_bbox = (
            BoundingBox.from_tuple(bleed_box_tuple, CoordOrigin.BOTTOMLEFT)
            if bleed_box_tuple
            else bbox
        )
        trim_bbox = (
            BoundingBox.from_tuple(trim_box_tuple, CoordOrigin.BOTTOMLEFT)
            if trim_box_tuple
            else bbox
        )

        return PdfPageGeometry(
            angle=angle,
            rect=BoundingRectangle.from_bounding_box(bbox),
            boundary_type=boundary_type,
            art_bbox=art_bbox,
            bleed_bbox=bleed_bbox,
            crop_bbox=crop_bbox,
            media_bbox=media_bbox,
            trim_bbox=trim_bbox,
        )


if TYPE_CHECKING:
    from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)


class PyPdfiumPageBackend(PdfPageBackend):
    def __init__(
        self, pdfium_doc: pdfium.PdfDocument, document_hash: str, page_no: int
    ):
        # Note: lock applied by the caller
        self.valid = True  # No better way to tell from pypdfium.
        try:
            self._ppage: pdfium.PdfPage = pdfium_doc[page_no]
        except PdfiumError:
            _log.info(
                f"An exception occurred when loading page {page_no} of document {document_hash}.",
                exc_info=True,
            )
            self.valid = False
        self.text_page: Optional[PdfTextPage] = None

    def is_valid(self) -> bool:
        return self.valid

    def _compute_text_cells(self) -> List[TextCell]:
        """Compute text cells from pypdfium."""
        with pypdfium2_lock:
            if not self.text_page:
                self.text_page = self._ppage.get_textpage()

        cells = []
        cell_counter = 0

        page_size = self.get_size()

        with pypdfium2_lock:
            for i in range(self.text_page.count_rects()):
                rect = self.text_page.get_rect(i)
                text_piece = self.text_page.get_text_bounded(*rect)
                x0, y0, x1, y1 = rect
                cells.append(
                    TextCell(
                        index=cell_counter,
                        text=text_piece,
                        orig=text_piece,
                        from_ocr=False,
                        rect=BoundingRectangle.from_bounding_box(
                            BoundingBox(
                                l=x0,
                                b=y0,
                                r=x1,
                                t=y1,
                                coord_origin=CoordOrigin.BOTTOMLEFT,
                            )
                        ).to_top_left_origin(page_size.height),
                    )
                )
                cell_counter += 1

        # PyPdfium2 produces very fragmented cells, with sub-word level boundaries, in many PDFs.
        # The cell merging code below is to clean this up.
        def merge_horizontal_cells(
            cells: List[TextCell],
            horizontal_threshold_factor: float = 1.0,
            vertical_threshold_factor: float = 0.5,
        ) -> List[TextCell]:
            if not cells:
                return []

            def group_rows(cells: List[TextCell]) -> List[List[TextCell]]:
                rows = []
                current_row = [cells[0]]
                row_top = cells[0].rect.to_bounding_box().t
                row_bottom = cells[0].rect.to_bounding_box().b
                row_height = cells[0].rect.to_bounding_box().height

                for cell in cells[1:]:
                    vertical_threshold = row_height * vertical_threshold_factor
                    if (
                        abs(cell.rect.to_bounding_box().t - row_top)
                        <= vertical_threshold
                        and abs(cell.rect.to_bounding_box().b - row_bottom)
                        <= vertical_threshold
                    ):
                        current_row.append(cell)
                        row_top = min(row_top, cell.rect.to_bounding_box().t)
                        row_bottom = max(row_bottom, cell.rect.to_bounding_box().b)
                        row_height = row_bottom - row_top
                    else:
                        rows.append(current_row)
                        current_row = [cell]
                        row_top = cell.rect.to_bounding_box().t
                        row_bottom = cell.rect.to_bounding_box().b
                        row_height = cell.rect.to_bounding_box().height

                if current_row:
                    rows.append(current_row)

                return rows

            def merge_row(row: List[TextCell]) -> List[TextCell]:
                merged = []
                current_group = [row[0]]

                for cell in row[1:]:
                    prev_cell = current_group[-1]
                    avg_height = (
                        prev_cell.rect.height + cell.rect.to_bounding_box().height
                    ) / 2
                    if (
                        cell.rect.to_bounding_box().l
                        - prev_cell.rect.to_bounding_box().r
                        <= avg_height * horizontal_threshold_factor
                    ):
                        current_group.append(cell)
                    else:
                        merged.append(merge_group(current_group))
                        current_group = [cell]

                if current_group:
                    merged.append(merge_group(current_group))

                return merged

            def merge_group(group: List[TextCell]) -> TextCell:
                if len(group) == 1:
                    return group[0]

                merged_bbox = BoundingBox(
                    l=min(cell.rect.to_bounding_box().l for cell in group),
                    t=min(cell.rect.to_bounding_box().t for cell in group),
                    r=max(cell.rect.to_bounding_box().r for cell in group),
                    b=max(cell.rect.to_bounding_box().b for cell in group),
                )

                assert self._ppage is not None
                self.text_page = self._ppage.get_textpage()
                bbox = merged_bbox.to_bottom_left_origin(page_size.height)
                merged_text = self.text_page.get_text_bounded(*bbox.as_tuple())

                return TextCell(
                    index=group[0].index,
                    text=merged_text,
                    orig=merged_text,
                    rect=BoundingRectangle.from_bounding_box(merged_bbox),
                    from_ocr=False,
                )

            rows = group_rows(cells)
            merged_cells = [cell for row in rows for cell in merge_row(row)]

            for i, cell in enumerate(merged_cells, 1):
                cell.index = i

            return merged_cells

        return merge_horizontal_cells(cells)

    def get_bitmap_rects(self, scale: float = 1) -> Iterable[BoundingBox]:
        AREA_THRESHOLD = 0  # 32 * 32
        page_size = self.get_size()
        with pypdfium2_lock:
            for obj in self._ppage.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE]):
                pos = obj.get_pos()
                cropbox = BoundingBox.from_tuple(
                    pos, origin=CoordOrigin.BOTTOMLEFT
                ).to_top_left_origin(page_height=page_size.height)

                if cropbox.area() > AREA_THRESHOLD:
                    cropbox = cropbox.scaled(scale=scale)

                    yield cropbox

    def get_text_in_rect(self, bbox: BoundingBox) -> str:
        with pypdfium2_lock:
            if not self.text_page:
                self.text_page = self._ppage.get_textpage()

        if bbox.coord_origin != CoordOrigin.BOTTOMLEFT:
            bbox = bbox.to_bottom_left_origin(self.get_size().height)

        with pypdfium2_lock:
            text_piece = self.text_page.get_text_bounded(*bbox.as_tuple())

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
        self.text_page = None


class PyPdfiumDocumentBackend(PdfDocumentBackend):
    def __init__(self, in_doc: "InputDocument", path_or_stream: Union[BytesIO, Path]):
        super().__init__(in_doc, path_or_stream)

        try:
            with pypdfium2_lock:
                self._pdoc = pdfium.PdfDocument(self.path_or_stream)
        except PdfiumError as e:
            raise RuntimeError(
                f"pypdfium could not load document with hash {self.document_hash}"
            ) from e

    def page_count(self) -> int:
        with pypdfium2_lock:
            return len(self._pdoc)

    def load_page(self, page_no: int) -> PyPdfiumPageBackend:
        with pypdfium2_lock:
            return PyPdfiumPageBackend(self._pdoc, self.document_hash, page_no)

    def is_valid(self) -> bool:
        return self.page_count() > 0

    def unload(self):
        super().unload()
        with pypdfium2_lock:
            self._pdoc.close()
            self._pdoc = None
