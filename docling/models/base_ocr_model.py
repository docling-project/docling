import copy
import logging
from abc import abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Type

import numpy as np
from docling_core.types.doc import BoundingBox, CoordOrigin, DocItemLabel
from docling_core.types.doc.page import TextCell
from PIL import Image, ImageDraw
from rtree import index

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import OcrMode, OcrOptions
from docling.datamodel.settings import settings
from docling.models.base_model import BaseModelWithOptions, BasePageModel

_log = logging.getLogger(__name__)


class BaseOcrModel(BasePageModel, BaseModelWithOptions):
    OCR_CLUSTER_LABELS = [
        DocItemLabel.CHART,
        DocItemLabel.PICTURE,
        DocItemLabel.HANDWRITTEN_TEXT,
    ]

    def __init__(
        self,
        *,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: OcrOptions,
        accelerator_options: AcceleratorOptions,
    ):
        # Make sure any delay/error from import occurs on ocr model init and not first use
        from scipy.ndimage import binary_dilation, find_objects, label

        self.enabled = enabled
        self.options = options

    def get_ocr_rects(self, page: Page) -> List[BoundingBox]:
        r""" """
        ocr_bboxes: List[BoundingBox]
        if self.options.mode == OcrMode.LAYOUT_ONLY:
            ocr_bboxes = self._get_cluster_ocr_rects(page)
        elif self.options.mode == OcrMode.PDF_ONLY:
            ocr_bboxes = self._get_pdf_ocr_rects(page)
        elif self.options.mode == OcrMode.LAYOUT_OR_PDF:
            ocr_bboxes = self._get_cluster_ocr_rects(page)
            if len(ocr_bboxes) == 0:
                ocr_bboxes = self._get_pdf_ocr_rects(page)
        elif self.options.mode == OcrMode.LAYOUT_AND_PDF:
            ocr_bboxes = self._filter_by_intersection(
                self._get_cluster_ocr_rects(page), self._get_pdf_ocr_rects(page)
            )
        return ocr_bboxes

    def _filter_by_intersection(
        self, layout_bboxes: List[BoundingBox], pdf_bboxes: List[BoundingBox]
    ) -> List[BoundingBox]:
        r"""
        Keep only the layout bboxes that intersect with at least one PDF bbox.
        A layout bbox that does not intersect with any PDF bbox is dropped.
        """
        ocr_bboxes: List[BoundingBox] = [
            layout_bbox
            for layout_bbox in layout_bboxes
            if any(
                layout_bbox.intersection_area_with(pdf_bbox) > 0
                for pdf_bbox in pdf_bboxes
            )
        ]
        return ocr_bboxes

    def _get_cluster_ocr_rects(self, page: Page) -> List[BoundingBox]:
        r"""
        Compute OCR rectangles from the layout clusters of a page.

        Clusters labeled as charts, pictures, or handwritten text are turned into
        OCR regions, using the cluster bounding box directly. Returns an empty list
        if the page has no layout prediction yet.
        """
        if page.predictions.layout is None:
            return []

        cluster_bboxes = [
            cluster.bbox
            for cluster in page.predictions.layout.clusters
            if cluster.label in self.OCR_CLUSTER_LABELS
        ]
        return cluster_bboxes

    def _get_pdf_ocr_rects(self, page: Page) -> List[BoundingBox]:
        r"""
        Compute the rectangles that should be OCRed on a given page.

        The bitmap rects of the page (the bboxes with the pictures inside the page
        of a programmatic PDF; empty if the page does not come from a programmatic
        PDF) are turned into candidate OCR regions and their page coverage via the
        following algorithm:

        1. Rasterize the bitmap rects into a blank binary black-white image.
           - The background is black and the rects are white.
        2. Apply a small binary dilation on the rects.
        3. Identify the bounding boxes around the "white" regions of the binary image.
        4. Compute the coverage as the ratio of white pixels in the dilated image to
           the page area.
        5. Return the coverage and the discovered bboxes.

        The coverage then decides which rectangles are returned:
        - If ``force_full_page_ocr`` is set, or the coverage exceeds both the
          bitmap-coverage threshold and ``bitmap_area_threshold``, a single
          full-page rectangle is returned.
        - Else if the coverage exceeds ``bitmap_area_threshold``, the individual
          discovered bboxes are returned.
        - Otherwise the bitmap coverage is too low and no rectangles are returned.
        """
        from scipy.ndimage import binary_dilation, find_objects, label

        BITMAP_COVERAGE_THRESHOLD = 0.75
        assert page.size is not None

        def find_ocr_rects(size, bitmap_rects):
            image = Image.new(
                "1", (round(size.width), round(size.height))
            )  # '1' mode is binary

            # Draw all bitmap rects into a binary image
            draw = ImageDraw.Draw(image)
            for rect in bitmap_rects:
                x0, y0, x1, y1 = rect.as_tuple()
                x0, y0, x1, y1 = round(x0), round(y0), round(x1), round(y1)
                draw.rectangle([(x0, y0), (x1, y1)], fill=1)

            #######################################################################################
            # Debug: Dump the image as a file
            # enable `pipeline_options.do_ocr = True` in tests/test_e2e_conversion.py
            # from datetime import datetime

            # viz_root = Path(
            #     "/Users/nli/docling/ocr_layout_pipelines_refactoring/viz_ocr_rect/"
            # )
            # tmp_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            # tmp_fn = viz_root / tmp_filename
            # image.save(
            #     str(tmp_fn),
            #     format="png",
            # )
            #######################################################################################

            np_image = np.array(image)

            # Dilate the image by 10 pixels to merge nearby bitmap rectangles
            structure = np.ones(
                (20, 20)
            )  # Create a 20x20 structure element (10 pixels in all directions)
            np_image = binary_dilation(np_image > 0, structure=structure)

            # Find the connected components
            labeled_image, num_features = label(np_image > 0)  # Label white regions

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

        if page._backend is not None:
            bitmap_rects = page._backend.get_bitmap_rects()
        else:
            bitmap_rects = []

        coverage, ocr_rects = find_ocr_rects(page.size, bitmap_rects)

        # return full-page rectangle if page is dominantly covered with bitmaps
        if self.options.force_full_page_ocr or coverage > max(
            BITMAP_COVERAGE_THRESHOLD, self.options.bitmap_area_threshold
        ):
            return [
                BoundingBox(
                    l=0,
                    t=0,
                    r=page.size.width,
                    b=page.size.height,
                    coord_origin=CoordOrigin.TOPLEFT,
                )
            ]
        # return individual rectangles if the bitmap coverage is above the threshold
        elif coverage > self.options.bitmap_area_threshold:
            return ocr_rects
        else:  # overall coverage of bitmaps is too low, drop all bitmap rectangles.
            return []

    # Filters OCR cells by dropping any OCR cell that intersects with an existing programmatic cell.
    def _filter_ocr_cells(
        self, ocr_cells: List[TextCell], programmatic_cells: List[TextCell]
    ) -> List[TextCell]:
        # Create R-tree index for programmatic cells
        p = index.Property()
        p.dimension = 2
        idx = index.Index(properties=p)
        for i, cell in enumerate(programmatic_cells):
            idx.insert(i, cell.rect.to_bounding_box().as_tuple())

        def is_overlapping_with_existing_cells(ocr_cell):
            # Query the R-tree to get overlapping rectangles
            possible_matches_index = list(
                idx.intersection(ocr_cell.rect.to_bounding_box().as_tuple())
            )

            return (
                len(possible_matches_index) > 0
            )  # this is a weak criterion but it works.

        filtered_ocr_cells = [
            rect for rect in ocr_cells if not is_overlapping_with_existing_cells(rect)
        ]

        return filtered_ocr_cells

    def post_process_cells(
        self,
        ocr_cells: List[TextCell],
        page: Page,
        conv_res: ConversionResult,
    ) -> None:
        r"""
        Post-process the OCR cells and update the page object.
        Updates parsed_page.textline_cells directly since page.cells is now read-only.
        """
        # Get existing cells from the read-only property
        existing_cells = page.cells

        # Combine existing and OCR cells with overlap filtering
        final_cells = self._combine_cells(existing_cells, ocr_cells)

        assert page.parsed_page is not None

        # Update parsed_page.textline_cells directly
        page.parsed_page.textline_cells = final_cells
        page.parsed_page.has_lines = len(final_cells) > 0

        # When force_full_page_ocr is used, PDF-extracted word/char cells are
        # unreliable. Filter out cells where from_ocr=False, keeping any OCR-
        # generated cells. This ensures downstream components (e.g., table
        # structure model) fall back to OCR-extracted textline cells.
        if self.options.force_full_page_ocr:
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

    def _combine_cells(
        self, existing_cells: List[TextCell], ocr_cells: List[TextCell]
    ) -> List[TextCell]:
        """Combine existing and OCR cells with filtering and re-indexing."""
        if self.options.force_full_page_ocr:
            combined = ocr_cells
        else:
            filtered_ocr_cells = self._filter_ocr_cells(ocr_cells, existing_cells)
            combined = list(existing_cells) + filtered_ocr_cells

        # Re-index in-place
        for i, cell in enumerate(combined):
            cell.index = i

        return combined

    def draw_ocr_rects_and_cells(self, conv_res, page, ocr_rects, show: bool = False):
        # ToDecide: If we want to have all drawing functions in docling/utils/visualization.py
        #           or even inside docling-core
        image = copy.deepcopy(page.image)
        scale_x = image.width / page.size.width
        scale_y = image.height / page.size.height

        draw = ImageDraw.Draw(image, "RGBA")

        # Draw OCR rectangles as yellow filled rect
        for rect in ocr_rects:
            x0, y0, x1, y1 = rect.as_tuple()
            y0 *= scale_x
            y1 *= scale_y
            x0 *= scale_x
            x1 *= scale_x

            shade_color = (255, 255, 0, 40)  # transparent yellow
            draw.rectangle([(x0, y0), (x1, y1)], fill=shade_color, outline=None)

        # Draw OCR and programmatic cells
        for tc in page.cells:
            x0, y0, x1, y1 = tc.rect.to_bounding_box().as_tuple()
            y0 *= scale_x
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
    def get_options_type(cls) -> Type[OcrOptions]:
        pass
