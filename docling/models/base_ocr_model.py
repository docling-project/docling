# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import copy
import logging
from abc import abstractmethod
from collections.abc import Iterable
from enum import Enum
from pathlib import Path

import numpy as np
from docling_core.types.doc import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.page import (
    BoundingRectangle,
    PdfCellRenderingMode,
    PdfPageGeometry,
    PdfTextCell,
    SegmentedPdfPage,
    TextCell,
)
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, find_objects, label

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import OcrMode, OcrOptions
from docling.datamodel.settings import settings
from docling.datamodel.spatial import BoundingBoxSpatialIndex
from docling.models.base_model import BaseModelWithOptions, BasePageModel

_log = logging.getLogger(__name__)


def _empty_segmented_page(page: Page) -> SegmentedPdfPage:
    """A minimal SegmentedPdfPage for pages whose native parse was skipped
    (PagePreprocessingOptions.skip_cell_extraction), sized from the page."""
    width, height = page.size.width, page.size.height
    # CoordOrigin.BOTTOMLEFT by convention: a real backend produces the
    # segmented page in bottom-left coordinates.
    rect = BoundingRectangle(
        r_x0=0,  # lower-left
        r_y0=0,
        r_x1=width,  # lower-right
        r_y1=0,
        r_x2=width,  # upper-right
        r_y2=height,
        r_x3=0,  # upper-left
        r_y3=height,
        coord_origin=CoordOrigin.BOTTOMLEFT,
    )
    bbox = BoundingBox(l=0, t=height, r=width, b=0, coord_origin=CoordOrigin.BOTTOMLEFT)
    return SegmentedPdfPage(
        dimension=PdfPageGeometry(
            angle=0.0,
            rect=rect,
            boundary_type="crop_box",
            art_bbox=bbox,
            bleed_bbox=bbox,
            crop_bbox=bbox,
            media_bbox=bbox,
            trim_bbox=bbox,
        ),
        char_cells=[],
        word_cells=[],
        textline_cells=[],
    )


try:
    import cv2

    CV2_INSTALLED = True
except ImportError:
    CV2_INSTALLED = False


class _MergeCellsPriority(str, Enum):
    # Take the OCR cells ONLY if they do not overlap with any PDF cell
    PDF_FIRST = "pdf_cells_first"

    # Take the PDF cells ONLY if they do not overlap with any OCR cell
    OCR_FIRST = "ocr_cells_first"


# PDF 32000 text rendering modes that paint no ink
_INVISIBLE_RENDERING_MODES = frozenset(
    {PdfCellRenderingMode.INVISIBLE, PdfCellRenderingMode.ONLY_CLIPPING}
)


# Flag to control if the OCR cells post-processing will use all PDF cells or only the visible ones
_PDF_CELLS_POST_PROCESSING_SEGREGATION = True


def _segregate_by_visibility(
    cells: Iterable[TextCell],
) -> tuple[list[TextCell], list[TextCell]]:
    """Split cells into (visible, invisible), preserving the order within each group.

    Cells carrying no rendering mode (plain `TextCell`, e.g. OCR output) count as visible,
    as does `UNKNOWN` (-1): no `Tr` operator means the PDF default mode 0 (fill).
    """
    visible: list[TextCell] = []
    invisible: list[TextCell] = []
    for cell in cells:
        if (
            isinstance(cell, PdfTextCell)
            and cell.rendering_mode in _INVISIBLE_RENDERING_MODES
        ):
            invisible.append(cell)
        else:
            visible.append(cell)
    return visible, invisible


class BaseOcrModel(BasePageModel, BaseModelWithOptions):
    r"""
    Base class for all OCR models.
    It offers common OCR functionalities
    """

    DEFAULT_DILATION_SIZE = 20

    def __init__(
        self,
        *,
        enabled: bool,
        artifacts_path: Path | None,
        options: OcrOptions,
        accelerator_options: AcceleratorOptions,
    ):
        self.enabled = enabled
        self.options = options

    def get_ocr_rects(self, page: Page) -> list[BoundingBox]:
        r"""
        Produce the input rects for the OCR according to the logic for each OcrMode
        """
        assert page.size is not None

        # Compute the OCR rects according to the mode
        ocr_rects: list[BoundingBox]

        # Both DEFAULT and PDF_AWARE_LAYOUT_REGIONS make OCR input as layout detections eliminated by PDF cells
        if (
            self.options.mode == OcrMode.DEFAULT
            or self.options.mode == OcrMode.PDF_AWARE_LAYOUT_REGIONS
        ):
            ocr_rects = self._find_pdf_aware_layout_ocr_rects(page)
        elif self.options.mode == OcrMode.LAYOUT_REGIONS:
            ocr_rects = self._find_layout_ocr_rects(page)
        elif self.options.mode == OcrMode.FULL_PAGE:
            # A big bbox covering the entire page
            ocr_rects = [
                BoundingBox(
                    l=0,
                    t=0,
                    r=page.size.width,
                    b=page.size.height,
                    coord_origin=CoordOrigin.TOPLEFT,
                )
            ]
        return ocr_rects

    def _find_layout_ocr_rects(self, page: Page) -> list[BoundingBox]:
        r"""
        1. Collect the bboxes of all layout clusters.
        2. Deduplicate the candidate ocr_rects.
        """
        if page.predictions.layout is None:
            return []

        # Use every layout detection bbox as an initial ocr_rect
        ocr_rects = [c.bbox for c in page.predictions.layout.clusters]

        # Deduplicate the ocr_rects
        _, ocr_rects = self._deduplicate_rects(
            page.size, ocr_rects, dilation_size=BaseOcrModel.DEFAULT_DILATION_SIZE
        )
        return ocr_rects

    def _find_pdf_aware_layout_ocr_rects(self, page: Page) -> list[BoundingBox]:
        r"""
        Compute the OCR rects from the layout clusters of a programmatic PDF.

        1. Start from the layout clusters.
        2. Eliminate clusters that intersect exclusively with programmatic text PDF cells
           The following clusters therefore remain:
           - Clusters without any overlapping PDF cell.
           - Clusters with at least one overlapping non-text region (e.g. bitmap, shape).
        3. Deduplicate the remaining cluster bboxes.
        """
        if page.predictions.layout is None:
            return []
        if page._backend is None:
            return self._find_layout_ocr_rects(page)

        assert page.size is not None
        backend = page._backend

        # Probe the backend to decide if `has_content_in()` is available or indexing is needed
        page_bbox = BoundingBox(
            l=0,
            t=0,
            r=page.size.width,
            b=page.size.height,
            coord_origin=CoordOrigin.TOPLEFT,
        )
        use_backend_queries = backend.has_content_in(bbox=page_bbox) is not None

        text_index: BoundingBoxSpatialIndex | None = None
        non_text_index: BoundingBoxSpatialIndex | None = None
        if not use_backend_queries:
            # Index for the text PDF cells
            text_cells = backend.get_visible_text_cells()
            if text_cells is None:
                text_cells = backend.get_text_cells()

            text_index = BoundingBoxSpatialIndex()
            for i, text_cell in enumerate(text_cells):
                text_index.insert(i, text_cell.rect.to_bounding_box())

            # Index for the non-text PDF cells: bitmaps, and shapes when available
            non_text_boxes = list(backend.get_bitmap_rects())
            shape_boxes = backend.get_connected_shape_bounding_boxes()
            if shape_boxes is not None:
                non_text_boxes.extend(shape_boxes)

            non_text_index = BoundingBoxSpatialIndex()
            for i, bbox in enumerate(non_text_boxes):
                non_text_index.insert(i, bbox)

        # Collect the non-eliminated cluster bboxes
        ocr_rects: list[BoundingBox] = []
        for cluster in page.predictions.layout.clusters:
            cluster_bbox = cluster.bbox

            if use_backend_queries:
                has_non_text = backend.has_content_in(
                    bbox=cluster_bbox, chars=False, shapes=True, bitmaps=True
                )
            else:
                assert non_text_index is not None
                has_non_text = any(
                    True for _ in non_text_index.intersection(cluster_bbox)
                )

            if has_non_text:
                ocr_rects.append(cluster_bbox)
                continue

            # Of the rest, only the clusters without any programmatic text need OCR.
            if use_backend_queries:
                has_text = backend.has_content_in(
                    bbox=cluster_bbox, chars=True, shapes=False, bitmaps=False
                )
            else:
                assert text_index is not None
                has_text = any(True for _ in text_index.intersection(cluster_bbox))

            if not has_text:
                ocr_rects.append(cluster_bbox)

        # Deduplicate the surviving cluster bboxes.
        _, ocr_rects = self._deduplicate_rects(
            page.size, ocr_rects, dilation_size=BaseOcrModel.DEFAULT_DILATION_SIZE
        )

        return ocr_rects

    def _deduplicate_rects(
        self, size: Size, rects: Iterable[BoundingBox], dilation_size=0
    ) -> tuple[float, list[BoundingBox]]:
        r"""
        Deduplicate the given rects and compute the coverage ratio defined as sum(rects)/image_size

        1. Rasterize the rects into a blank binary black-white image.
           - The background is black and the rects are white.
        2. Optionally apply a small binary dilation on the rects.
        3. Identify the bounding boxes around the "white" regions of the binary image.
        4. Compute the coverage as the ratio of white pixels in the image to the page area.
        5. Return the coverage and the discovered bboxes.
        """
        image = Image.new(
            "1", (round(size.width), round(size.height))
        )  # '1' mode is binary

        # Draw all bitmap rects into a binary image
        draw = ImageDraw.Draw(image)
        for rect in rects:
            x0, y0, x1, y1 = rect.as_tuple()
            x0, y0, x1, y1 = round(x0), round(y0), round(x1), round(y1)
            draw.rectangle([(x0, y0), (x1, y1)], fill=1)

        np_image = np.array(image)

        if dilation_size > 0:
            # Grow the rects by dilation_size / 2 pixels in all directions.
            kernel = np.ones((dilation_size, dilation_size), dtype=np.uint8)
            if CV2_INSTALLED:
                np_image = cv2.dilate(
                    (np_image > 0).astype(np.uint8), kernel, iterations=1
                )
            else:
                np_image = binary_dilation(np_image > 0, structure=kernel)

        # Find the connected components
        labeled_image, _ = label(np_image > 0)  # Label white regions

        # Find enclosing bounding boxes for each connected component.
        slices = find_objects(labeled_image)
        bounding_boxes = [
            BoundingBox(
                l=slc[1].start,
                t=slc[0].start,
                r=slc[1].stop - 1,
                b=slc[0].stop - 1,
                coord_origin=CoordOrigin.TOPLEFT,
            )
            for slc in slices
        ]

        # Compute area fraction on page covered by bitmaps
        area_frac = np.sum(np_image > 0) / (size.width * size.height)
        return (area_frac, bounding_boxes)  # fraction covered  # boxes

    def post_process_cells(
        self,
        ocr_cells: list[TextCell],
        page: Page,
        conv_res: ConversionResult,
        priority: _MergeCellsPriority | None = None,
    ) -> None:
        r"""
        Post-process the OCR cells and update the page object according to the algorithm:

        - If FULL_PAGE: Any existing PDF cells are ignored and only the OCR cells are used.
        - If LAYOUT_REGIONS or PDF_AWARE_LAYOUT_REGIONS and the priority parameter is None,
          the priority is auto-selected based on the OcrMode:
              - OCR_FIRST when LAYOUT_REGIONS
              - PDF_FIRST when PDF_AWARE_LAYOUT_REGIONS
          Check the comments on _MergeCellsPriority for the semantic of each priority value
        """
        # Get existing cells from the read-only property
        existing_cells = page.cells

        # Combine existing and OCR cells with overlap filtering
        if self.options.mode == OcrMode.FULL_PAGE:
            final_cells = ocr_cells
        else:
            if priority is None:
                priority = (
                    _MergeCellsPriority.OCR_FIRST
                    if self.options.mode == OcrMode.LAYOUT_REGIONS
                    else _MergeCellsPriority.PDF_FIRST
                )
            final_cells = self._merge_ocr_and_pdf_cells(
                ocr_cells,
                existing_cells,
                priority,
                segregate_pdf_cells=_PDF_CELLS_POST_PROCESSING_SEGREGATION,
            )

        # Re-index in-place
        for i, cell in enumerate(final_cells):
            cell.index = i

        if page.parsed_page is None:
            # No native parse ran (e.g. skip_cell_extraction): create an empty
            # SegmentedPdfPage so the OCR output has somewhere to live.
            page.parsed_page = _empty_segmented_page(page)

        # Update parsed_page.textline_cells directly
        page.parsed_page.textline_cells = final_cells
        page.parsed_page.has_lines = len(final_cells) > 0

        # In OcrMode.FULL_PAGE, PDF-extracted word/char cells are unreliable. Keep only OCR cells
        if self.options.mode == OcrMode.FULL_PAGE:
            page.parsed_page.word_cells = [
                c for c in page.parsed_page.word_cells if c.from_ocr
            ]
            page.parsed_page.char_cells = [
                c for c in page.parsed_page.char_cells if c.from_ocr
            ]
            page.parsed_page.has_words = len(page.parsed_page.word_cells) > 0
            page.parsed_page.has_chars = len(page.parsed_page.char_cells) > 0

        ocr_confidences = [c.confidence for c in final_cells if c.from_ocr]
        if ocr_confidences:
            conv_res.confidence.pages[page.page_no].ocr_score = float(
                np.mean(ocr_confidences)
            )

    def _merge_ocr_and_pdf_cells(
        self,
        ocr_cells: list[TextCell],
        pdf_cells: list[TextCell],
        priority: _MergeCellsPriority,
        segregate_pdf_cells: bool = True,
    ) -> list[TextCell]:
        r"""
        Merge PDF and OCR cells, resolving overlaps according to `priority`.

        When `segregate_pdf_cells` is True, the OCR cells are merged only with the visible PDF
        cells and the invisible ones are put back afterwards. Otherwise, the OCR cells are merged
        with all the PDF cells indiscriminately.
        """
        visible_cells: list[TextCell]
        invisible_cells: list[TextCell]
        if segregate_pdf_cells:
            visible_cells, invisible_cells = _segregate_by_visibility(pdf_cells)
        else:
            visible_cells, invisible_cells = list(pdf_cells), []

        # The prioritized cells are always kept
        # the secondary cells are added only where they don't overlap a prioritized cell.
        if priority == _MergeCellsPriority.PDF_FIRST:
            prioritized_cells, secondary_cells = visible_cells, ocr_cells
        else:
            prioritized_cells, secondary_cells = ocr_cells, visible_cells

        merged_cells = self._merge_cells_by_priority(prioritized_cells, secondary_cells)

        # Put the invisible cells back
        if invisible_cells:
            merged_cells = self._merge_cells_by_priority(merged_cells, invisible_cells)

        return merged_cells

    def _merge_cells_by_priority(
        self,
        prioritized_cells: list[TextCell],
        secondary_cells: list[TextCell],
    ) -> list[TextCell]:
        r"""
        Keep every prioritized cell, plus the secondary cells that overlap none of them.
        """
        idx = BoundingBoxSpatialIndex()

        # The R-tree bbox intersection is a weak criterion but it works.
        merged_cells = list(prioritized_cells)

        if len(prioritized_cells) <= len(secondary_cells):
            # Index the (smaller) prioritized cells; keep each secondary cell that
            # doesn't overlap any of them.
            for i, cell in enumerate(prioritized_cells):
                idx.insert(i, cell.rect.to_bounding_box())
            for cell in secondary_cells:
                overlaps = any(
                    True for _ in idx.intersection(cell.rect.to_bounding_box())
                )
                if not overlaps:
                    merged_cells.append(cell)
        else:
            # Index the (smaller) secondary cells; drop the ones overlapping any
            # prioritized cell and keep the rest.
            for i, cell in enumerate(secondary_cells):
                idx.insert(i, cell.rect.to_bounding_box())
            overlapping_ids: set[int] = set()
            for cell in prioritized_cells:
                overlapping_ids.update(idx.intersection(cell.rect.to_bounding_box()))
            merged_cells.extend(
                cell
                for i, cell in enumerate(secondary_cells)
                if i not in overlapping_ids
            )

        return merged_cells

    def draw_ocr_rects_and_cells(self, conv_res, page, ocr_rects, show: bool = False):
        r"""
        - OCR input rects: Yellow panes
        - OCR detected text: Magenta bboxes
        - PDF text: Gray bboxes
        """
        image = copy.deepcopy(page.image)
        scale_x = image.width / page.size.width
        scale_y = image.height / page.size.height

        draw = ImageDraw.Draw(image, "RGBA")

        # Draw OCR rectangles as yellow filled rect
        for rect in ocr_rects:
            x0, y0, x1, y1 = rect.as_tuple()
            y0 *= scale_y
            y1 *= scale_y
            x0 *= scale_x
            x1 *= scale_x

            shade_color = (255, 255, 0, 40)  # transparent yellow
            draw.rectangle([(x0, y0), (x1, y1)], fill=shade_color, outline=None)

        # Draw OCR and programmatic cells
        for tc in page.cells:
            x0, y0, x1, y1 = tc.rect.to_bounding_box().as_tuple()
            y0 *= scale_y
            y1 *= scale_y
            x0 *= scale_x
            x1 *= scale_x

            if y1 <= y0:
                y1, y0 = y0, y1

            color = "magenta" if tc.from_ocr else "gray"

            draw.rectangle([(x0, y0), (x1, y1)], outline=color)

        if show:
            image.show()
        else:
            out_path: Path = (
                Path(settings.debug.debug_output_path)
                / f"debug_{conv_res.input.file.stem}"
            )
            out_path.mkdir(parents=True, exist_ok=True)

            out_file = out_path / f"ocr_page_{page.page_no:05}.png"
            image.save(str(out_file), format="png")

    @abstractmethod
    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        pass

    @classmethod
    @abstractmethod
    def get_options_type(cls) -> type[OcrOptions]:
        pass
