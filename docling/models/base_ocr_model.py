import copy
import logging
from abc import abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import List, Optional, Type

import numpy as np
from docling_core.types.doc import BoundingBox, CoordOrigin, DocItemLabel, Size
from docling_core.types.doc.page import TextCell
from PIL import Image, ImageDraw
from rtree import index
from scipy.ndimage import binary_dilation, find_objects, label

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Cluster, Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import OcrMode, OcrOptions
from docling.datamodel.settings import settings
from docling.models.base_model import BaseModelWithOptions, BasePageModel

_log = logging.getLogger(__name__)


class BaseOcrModel(BasePageModel, BaseModelWithOptions):
    MAXOUT_COVERAGE_THRESHOLD = 0.75
    SPARSE_LABELS = [
        DocItemLabel.PICTURE,
        DocItemLabel.CHART,
        DocItemLabel.TABLE,
        DocItemLabel.DOCUMENT_INDEX,
        DocItemLabel.KEY_VALUE_REGION,
        DocItemLabel.FORM,
    ]
    DENSE_LABELS = [
        DocItemLabel.CAPTION,
        DocItemLabel.FOOTNOTE,
        DocItemLabel.LIST_ITEM,
        DocItemLabel.PAGE_FOOTER,
        DocItemLabel.PAGE_HEADER,
        DocItemLabel.SECTION_HEADER,
        DocItemLabel.TEXT,
        DocItemLabel.TITLE,
        DocItemLabel.CHECKBOX_SELECTED,
        DocItemLabel.CHECKBOX_UNSELECTED,
        DocItemLabel.HANDWRITTEN_TEXT,
        DocItemLabel.PARAGRAPH,
        DocItemLabel.REFERENCE,
        DocItemLabel.FIELD_REGION,
        DocItemLabel.FIELD_HEADING,
        DocItemLabel.FIELD_ITEM,
        DocItemLabel.FIELD_KEY,
        DocItemLabel.FIELD_VALUE,
        DocItemLabel.FIELD_HINT,
        DocItemLabel.MARKER,
        # DocItemLabel.FORMULA
        # DocItemLabel.CODE
    ]

    def __init__(
        self,
        *,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: OcrOptions,
        accelerator_options: AcceleratorOptions,
    ):
        self.enabled = enabled
        self.options = options

    def get_ocr_rects(self, page: Page) -> List[BoundingBox]:
        r"""
        Produce the input rects for the OCR according to the logic for each OcrMode
        If `force_full_page_ocr` is set, return a big bbox covering the entire page
        """
        assert page.size is not None

        # If `force_full_page_ocr` is set, return a big bbox covering the entire page
        if self.options.force_full_page_ocr:
            return [
                BoundingBox(
                    l=0,
                    t=0,
                    r=page.size.width,
                    b=page.size.height,
                    coord_origin=CoordOrigin.TOPLEFT,
                )
            ]

        # Compute the OCR rects according to the mode
        ocr_rects: List[BoundingBox]
        if self.options.mode == OcrMode.PDF_BITMAPS_ONLY:
            ocr_rects = self._find_pdf_ocr_rects(page)
        elif self.options.mode == OcrMode.LAYOUT_DETECTIONS:
            ocr_rects = self._find_layout_ocr_rects(page)
        elif self.options.mode == OcrMode.LAYOUT_DETECTIONS_WITHOUT_PDF_TEXT:
            ocr_rects = self._find_layout_ocr_rects_without_pdf_text(page)
        return ocr_rects

    def _find_pdf_ocr_rects(self, page: Page) -> List[BoundingBox]:
        r"""
        Compute the OCR rectangles coming ONLY from the programmatic PDF cells

        1. Deduplicate the bitmap rects.
        2. If coverage > MAXOUT_COVERAGE_THRESHOLD, return a single bbox covering the entire page.
        3. Else if coverage > `bitmap_area_threshold`, return the deduplicated rects.
        4. Otherwise return an empty list.
        """
        if page._backend is None:
            return []

        # Get the programmatic PDF cells and deduplicate them
        bitmap_rects = page._backend.get_bitmap_rects()
        coverage, ocr_rects = self._deduplicate_rects(page.size, bitmap_rects, 20)

        # return full-page rectangle if page is dominantly covered with bitmaps
        if coverage > max(
            BaseOcrModel.MAXOUT_COVERAGE_THRESHOLD, self.options.bitmap_area_threshold
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

        # Overall coverage of bitmaps is too low, drop all bitmap rectangles.
        return []

    def _find_layout_ocr_rects(self, page: Page) -> List[BoundingBox]:
        r"""
        1. Filter the layout clusters accoring to the dense/sparse logic.
        2. Deduplicate.
        """
        if page.predictions.layout is None:
            return []

        # Filter the layout detections to get the initial ocr_rects
        ocr_rects = [
            c.bbox for c in self._filter_clusters(page.predictions.layout.clusters)
        ]

        # Deduplicate the ocr_rects
        _, ocr_rects = self._deduplicate_rects(page.size, ocr_rects)

        return ocr_rects

    def _find_layout_ocr_rects_without_pdf_text(self, page: Page) -> List[BoundingBox]:
        r"""
        Compute OCR rectangles from the layout detections, dropping detections that
        are already sufficiently covered by text-bearing programmatic PDF cells.

        1. Filter the layout clusters accoring to the dense/sparse logic.
        2. Build the ocr_rects out of:
           a. The dense clusters that do not overlap with any PDF cell with text.
           b. The filtered sparse clusters that are covered by cells less than a threshold.
        """
        # If there is no page.backend, this equals to _find_layout_ocr_rects()
        if page._backend is None:
            return self._find_layout_ocr_rects(page)
        if page.predictions.layout is None:
            return []

        # Create an R-tree index with the text-bearing PDF cell bboxes
        p = index.Property()
        p.dimension = 2
        cells_index = index.Index(properties=p)
        cell_bboxes: List[BoundingBox] = []
        for cell in page._backend.get_text_cells():
            txt = cell.text
            if txt is None or txt.strip() == "":
                continue
            cell_bbox = cell.rect.to_bounding_box()
            cells_index.insert(len(cell_bboxes), cell_bbox.as_tuple())
            cell_bboxes.append(cell_bbox)

        # Iterate over the clusters to pick up the OCR rects
        ocr_rects: List[BoundingBox] = []
        for cluster in self._filter_clusters(page.predictions.layout.clusters):
            candidate_cell_bboxes = [
                cell_bboxes[i]
                for i in cells_index.intersection(cluster.bbox.as_tuple())
            ]
            if cluster.label in self.SPARSE_LABELS:
                coverage = self._compute_coverage(cluster.bbox, candidate_cell_bboxes)
                if coverage < self.options.sparse_cell_coverage_threshold:
                    ocr_rects.append(cluster.bbox)
            elif len(candidate_cell_bboxes) == 0:
                ocr_rects.append(cluster.bbox)

        # Deduplicate the ocr_rects
        _, ocr_rects = self._deduplicate_rects(page.size, ocr_rects)
        return ocr_rects

    def _filter_clusters(self, clusters: List[Cluster]) -> List[Cluster]:
        r"""
        - Keep all clusters with "dense" labels.
        - Keep a "sparse" cluster only if no "dense" cluster overlaps.
        """
        # Build an index for the dense bboxes
        p = index.Property()
        p.dimension = 2
        dense_idx = index.Index(properties=p)
        idx_id = 0
        for cluster in clusters:
            if cluster.label in self.SPARSE_LABELS:
                continue
            tuple_bbox = cluster.bbox.as_tuple()
            dense_idx.insert(idx_id, tuple_bbox)
            idx_id += 1

        # Select only the non-overlapping sparse bboxes
        filtered_clusters: List[Cluster] = []
        for cluster in clusters:
            if cluster.label in self.DENSE_LABELS:
                filtered_clusters.append(cluster)
                continue
            tuple_bbox = cluster.bbox.as_tuple()
            overlapping_dense_bboxes = list(dense_idx.intersection(tuple_bbox))
            if len(overlapping_dense_bboxes) == 0:
                filtered_clusters.append(cluster)

        return filtered_clusters

    def _compute_coverage(
        self, bbox: BoundingBox, text_cell_bboxes: List[BoundingBox]
    ) -> float:
        r"""
        Compute the fraction of `bbox`'s area that is covered by the given
        text-bearing PDF cell bboxes.

        The clipped intersections are rasterized into a binary mask so that
        overlapping cells are counted once (and the result stays within [0, 1]).
        All bboxes must share `bbox`'s coordinate origin.
        """
        if not text_cell_bboxes or bbox.area() <= 0:
            return 0.0

        width = max(round(bbox.width), 1)
        height = max(round(bbox.height), 1)

        mask = Image.new("1", (width, height))  # '1' mode is binary
        draw = ImageDraw.Draw(mask)
        for cell_bbox in text_cell_bboxes:
            intersection = bbox.get_intersection_bbox(cell_bbox)
            if intersection is None:
                continue
            # Translate the intersection into the mask's local top-left coordinates.
            x0 = round(intersection.l - bbox.l)
            y0 = round(intersection.t - bbox.t)
            x1 = round(intersection.r - bbox.l)
            y1 = round(intersection.b - bbox.t)
            draw.rectangle([(x0, y0), (x1, y1)], fill=1)

        covered_pixels = int(np.count_nonzero(np.asarray(mask)))
        return covered_pixels / (width * height)

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
            structure = np.ones((dilation_size, dilation_size))
            np_image = binary_dilation(np_image > 0, structure=structure)

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

        # When force_full_page_ocr is used, PDF-extracted word/char cells are unreliable.
        # Filter out cells where from_ocr=False, keeping any OCR generated cells.
        # This ensures downstream components (e.g., table structure model) fall back to
        # OCR-extracted textline cells.
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
        r"""Combine existing and OCR cells with filtering and re-indexing."""
        if self.options.force_full_page_ocr:
            combined = ocr_cells
        else:
            filtered_ocr_cells = self._filter_ocr_cells(ocr_cells, existing_cells)
            combined = list(existing_cells) + filtered_ocr_cells

        # Re-index in-place
        for i, cell in enumerate(combined):
            cell.index = i

        return combined

    def _filter_ocr_cells(
        self, ocr_cells: List[TextCell], programmatic_cells: List[TextCell]
    ) -> List[TextCell]:
        r"""
        Filter OCR cells by dropping any OCR cell that intersects with a programmatic PDF cell
        """
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
    def get_options_type(cls) -> Type[OcrOptions]:
        pass
