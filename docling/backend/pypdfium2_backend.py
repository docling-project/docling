import logging
import random
from collections.abc import Iterable
from importlib.metadata import version
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
from rtree import index

from docling.backend.managed_pdfium_backend import (
    ManagedPdfiumDocumentBackend,
    ManagedPdfiumPageBackend,
)
from docling.datamodel.backend_options import PdfBackendOptions
from docling.exceptions import DocumentLoadError
from docling.utils.locks import pypdfium2_lock
from docling.utils.pdf_outline import _PdfOutlineItem, extract_outline_from_pdfium


def _merge_overlapping_boxes(
    boxes: List[BoundingBox], tolerance: float
) -> List[BoundingBox]:
    """Merge boxes that overlap (within ``tolerance``) into their connected components.

    All boxes must share the top-left origin. An R-tree keeps this near-linear: pages of
    vector art routinely carry thousands of path objects.
    """
    if not boxes:
        return []

    def _query(bbox: BoundingBox) -> tuple[float, float, float, float]:
        return (
            bbox.l - tolerance,
            bbox.t - tolerance,
            bbox.r + tolerance,
            bbox.b + tolerance,
        )

    prop = index.Property()
    prop.dimension = 2
    tree = index.Index(properties=prop)
    for i, bbox in enumerate(boxes):
        tree.insert(i, (bbox.l, bbox.t, bbox.r, bbox.b))

    merged: List[BoundingBox] = []
    visited: set[int] = set()
    for start in range(len(boxes)):
        if start in visited:
            continue

        visited.add(start)
        stack = [start]
        left, top, right, bottom = (
            boxes[start].l,
            boxes[start].t,
            boxes[start].r,
            boxes[start].b,
        )
        while stack:
            current = boxes[stack.pop()]
            left = min(left, current.l)
            top = min(top, current.t)
            right = max(right, current.r)
            bottom = max(bottom, current.b)
            for neighbor in tree.intersection(_query(current)):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        merged.append(
            BoundingBox(
                l=left, t=top, r=right, b=bottom, coord_origin=CoordOrigin.TOPLEFT
            )
        )

    return merged


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


def _rect_to_display_frame(
    rect: tuple[float, float, float, float],
    rotation: int,
    page_size: Size,
) -> tuple[float, float, float, float]:
    """Map a rect from the page's unrotated frame to its rotated display frame.

    PDFium reports page-object and text coordinates in the unrotated (MediaBox)
    frame, ignoring the page's ``/Rotate`` entry. ``PdfPage.get_size()`` and the
    rendered page bitmap, on the other hand, are already in the rotated display
    frame. Applying the rotation here puts everything the backend returns into
    that single frame.

    Args:
        rect: ``(x0, y0, x1, y1)`` in the unrotated frame, bottom-left origin.
        rotation: page rotation in degrees (``PdfPage.get_rotation()``).
        page_size: page size in the display frame (``get_size()``).

    Returns:
        ``(x0, y0, x1, y1)`` in the display frame, bottom-left origin.
    """
    x0, y0, x1, y1 = rect
    if rotation == 90:
        return (y0, page_size.height - x1, y1, page_size.height - x0)
    elif rotation == 180:
        return (
            page_size.width - x1,
            page_size.height - y1,
            page_size.width - x0,
            page_size.height - y0,
        )
    elif rotation == 270:
        return (page_size.width - y1, x0, page_size.width - y0, x1)
    return (x0, y0, x1, y1)


def _rect_to_pdf_frame(
    rect: tuple[float, float, float, float],
    rotation: int,
    page_size: Size,
) -> tuple[float, float, float, float]:
    """Map a rect from the rotated display frame back to the unrotated frame.

    Inverse of :func:`_rect_to_display_frame`, needed whenever coordinates are
    handed back to PDFium (e.g. ``PdfTextPage.get_text_bounded()``), which only
    understands the unrotated frame.

    Args:
        rect: ``(x0, y0, x1, y1)`` in the display frame, bottom-left origin.
        rotation: page rotation in degrees (``PdfPage.get_rotation()``).
        page_size: page size in the display frame (``get_size()``).

    Returns:
        ``(x0, y0, x1, y1)`` in the unrotated frame, bottom-left origin.
    """
    x0, y0, x1, y1 = rect
    if rotation == 90:
        return (page_size.height - y1, x0, page_size.height - y0, x1)
    elif rotation == 180:
        return (
            page_size.width - x1,
            page_size.height - y1,
            page_size.width - x0,
            page_size.height - y0,
        )
    elif rotation == 270:
        return (y0, page_size.width - x1, y1, page_size.width - x0)
    return (x0, y0, x1, y1)


if TYPE_CHECKING:
    from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)


# Resolve pypdfium2 major version
# pypdfium2 5.x renamed PdfObject.get_pos() -> get_bounds()
_PYPDFIUM2_MAJOR_VERSION = int(version("pypdfium2").split(".")[0])

# PDF 32000 text rendering modes that paint no ink, matching the filter docling-parse
# applies natively when answering content-intersection queries.
_INVISIBLE_TEXT_RENDER_MODES = frozenset(
    {pdfium_c.FPDF_TEXTRENDERMODE_INVISIBLE, pdfium_c.FPDF_TEXTRENDERMODE_CLIP}
)


class PyPdfiumPageBackend(ManagedPdfiumPageBackend):
    def __init__(
        self,
        pdfium_doc: pdfium.PdfDocument,
        document_hash: str,
        page_no: int,
    ):
        super().__init__()
        self._page_no = page_no
        # Note: lock applied by the caller
        self.valid = True  # No better way to tell from pypdfium.
        self._ppage: pdfium.PdfPage | None = None
        try:
            self._ppage = pdfium_doc[page_no]
        except PdfiumError:
            _log.info(
                f"An exception occurred when loading page {page_no} of document {document_hash}.",
                exc_info=True,
            )
            self.valid = False
        self.text_page: Optional[PdfTextPage] = None

    def is_valid(self) -> bool:
        return self.valid

    @property
    def page_no(self) -> int:
        return self._page_no + 1

    def _require_page(self) -> pdfium.PdfPage:
        assert self._ppage is not None, "Page backend was unloaded."
        return self._ppage

    def _compute_text_cells(self) -> List[TextCell]:
        """Compute text cells from pypdfium."""
        with pypdfium2_lock:
            page = self._require_page()
            if not self.text_page:
                self.text_page = page.get_textpage()
            rotation = page.get_rotation()

        cells = []
        cell_counter = 0

        page_size = self.get_size()

        with pypdfium2_lock:
            for i in range(self.text_page.count_rects()):
                rect = self.text_page.get_rect(i)
                text_piece = self.text_page.get_text_bounded(*rect)
                # `rect` is in the unrotated frame, `page_size` in the rotated
                # display frame: bring the rect over before converting origin.
                x0, y0, x1, y1 = _rect_to_display_frame(rect, rotation, page_size)
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

                assert self.text_page is not None
                bbox = merged_bbox.to_bottom_left_origin(page_size.height)
                # Cells are stored in the display frame; PDFium only understands
                # the unrotated one, so undo the rotation before querying it.
                pdf_rect = _rect_to_pdf_frame(bbox.as_tuple(), rotation, page_size)
                with pypdfium2_lock:
                    merged_text = self.text_page.get_text_bounded(*pdf_rect)

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

    def _object_rects(
        self, obj_type: int, *, skip_invisible_text: bool = False
    ) -> Iterable[BoundingBox]:
        """Yield the bboxes of the page objects of ``obj_type``, in top-left origin.

        With ``skip_invisible_text``, text objects drawn in a rendering mode that paints no
        ink are left out, matching what docling-parse does natively.
        """
        page_size = self.get_size()

        with pypdfium2_lock:
            page = self._require_page()
            rotation = page.get_rotation()
            for obj in page.get_objects(filter=[obj_type]):
                if (
                    skip_invisible_text
                    and obj_type == pdfium_c.FPDF_PAGEOBJ_TEXT
                    and pdfium_c.FPDFTextObj_GetTextRenderMode(obj.raw)
                    in _INVISIBLE_TEXT_RENDER_MODES
                ):
                    continue

                if _PYPDFIUM2_MAJOR_VERSION >= 5:
                    pos = obj.get_bounds()  # pypdfium2 >= 5.x
                else:
                    pos = obj.get_pos()  # pypdfium2 <= 4.x
                pos = _rect_to_display_frame(pos, rotation, page_size)

                yield BoundingBox.from_tuple(
                    pos, origin=CoordOrigin.BOTTOMLEFT
                ).to_top_left_origin(page_height=page_size.height)

    def get_bitmap_rects(self, scale: float = 1) -> Iterable[BoundingBox]:
        AREA_THRESHOLD = 0  # 32 * 32

        for cropbox in self._object_rects(pdfium_c.FPDF_PAGEOBJ_IMAGE):
            if cropbox.area() > AREA_THRESHOLD:
                yield cropbox.scaled(scale=scale)

    def intersects_with(
        self,
        *,
        bbox: BoundingBox,
        chars: bool = False,
        shapes: bool = True,
        bitmaps: bool = True,
    ) -> Optional[bool]:
        """Best-effort content-intersection test built from page-object bboxes.

        pypdfium2 exposes no clip state, so this approximates the docling-parse query: an
        object counts as intersecting when its bounding box does, even if the object is
        clipped away or fully transparent. Text rendering mode is the one visibility signal
        it does expose, and invisible text is skipped just like docling-parse does.
        """
        if not self.valid:
            return False

        page_size = self.get_size()
        probe = bbox.to_top_left_origin(page_height=page_size.height)

        obj_types = []
        if shapes:
            obj_types.append(pdfium_c.FPDF_PAGEOBJ_PATH)
        if bitmaps:
            obj_types.append(pdfium_c.FPDF_PAGEOBJ_IMAGE)
        if chars:
            obj_types.append(pdfium_c.FPDF_PAGEOBJ_TEXT)

        for obj_type in obj_types:
            for rect in self._object_rects(obj_type, skip_invisible_text=True):
                # Plain overlap, so that degenerate (zero-area) rules still count.
                if (
                    rect.l <= probe.r
                    and probe.l <= rect.r
                    and rect.t <= probe.b
                    and probe.t <= rect.b
                ):
                    return True

        return False

    def get_connected_shape_bounding_boxes(
        self, *, tolerance: float = 0.0
    ) -> Optional[List[BoundingBox]]:
        """Best-effort connected shape regions, merged from path-object bboxes.

        Unlike the docling-parse implementation this sees neither clip state nor stroke
        width, so the regions are the union of raw path bounding boxes.
        """
        if not self.valid:
            return []

        return _merge_overlapping_boxes(
            list(self._object_rects(pdfium_c.FPDF_PAGEOBJ_PATH)), tolerance
        )

    def get_text_in_rect(self, bbox: BoundingBox) -> str:
        with pypdfium2_lock:
            page = self._require_page()
            if not self.text_page:
                self.text_page = page.get_textpage()
            rotation = page.get_rotation()

        page_size = self.get_size()

        if bbox.coord_origin != CoordOrigin.BOTTOMLEFT:
            bbox = bbox.to_bottom_left_origin(page_size.height)

        # `bbox` is expressed in the rotated display frame, PDFium expects the
        # unrotated one.
        pdf_rect = _rect_to_pdf_frame(bbox.as_tuple(), rotation, page_size)

        with pypdfium2_lock:
            text_piece = self.text_page.get_text_bounded(*pdf_rect)

        return text_piece

    def get_segmented_page(self) -> Optional[SegmentedPdfPage]:
        if not self.valid:
            return None

        text_cells = self._compute_text_cells()

        # Get the PDF page geometry from pypdfium2
        dimension = get_pdf_page_geometry(self._require_page())

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
            bitmap = self._require_page().render(
                scale=scale * 1.5,
                rotation=0,  # no additional rotation
                crop=padbox.as_tuple(),
            )
            image = bitmap.to_pil().copy()
            bitmap.close()
        # We resize the image from 1.5x the given scale to make it sharper.
        image = image.resize(
            size=(round(cropbox.width * scale), round(cropbox.height * scale))
        )

        return image

    def get_size(self) -> Size:
        with pypdfium2_lock:
            page = self._require_page()
            return Size(width=page.get_width(), height=page.get_height())

    def _close_native_page(self) -> None:
        with pypdfium2_lock:
            if self.text_page is not None:
                self.text_page.close()
            if self._ppage is not None:
                self._ppage.close()

        self.text_page = None
        self._ppage = None


class PyPdfiumDocumentBackend(ManagedPdfiumDocumentBackend):
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
        try:
            with pypdfium2_lock:
                self._pdoc = pdfium.PdfDocument(self.path_or_stream, password=password)
        except PdfiumError as e:
            raise DocumentLoadError(
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

    def get_document_outline(self) -> list[_PdfOutlineItem]:
        """Extract the PDF outline from the pypdfium2 document (title, depth, page, position)."""
        if self._pdoc is None:
            return []
        return extract_outline_from_pdfium(self._pdoc)

    def _close_native_document(self) -> None:
        with pypdfium2_lock:
            if self._pdoc is not None:
                self._pdoc.close()
                self._pdoc = None
