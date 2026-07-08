import copy
import logging
from abc import abstractmethod
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from docling_core.types.doc import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.page import TextCell
from PIL import Image, ImageDraw
from rtree import index
from scipy.ndimage import binary_dilation, find_objects, label

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import OcrMode, OcrOptions
from docling.datamodel.settings import settings
from docling.models.base_model import BaseModelWithOptions, BasePageModel

_log = logging.getLogger(__name__)

try:
    import cv2

    CV2_INSTALLED = True
except ImportError:
    CV2_INSTALLED = False


class BaseOcrModel(BasePageModel, BaseModelWithOptions):
    MAXOUT_COVERAGE_THRESHOLD = 0.75

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
        if self.options.mode == OcrMode.FULL_PAGE_OCR:
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
        elif self.options.mode == OcrMode.CLUSTER_OCR:
            ocr_rects = self._find_cluster_ocr_rects(page)
        elif self.options.mode == OcrMode.PDF_CLUSTER_OCR:
            ocr_rects = self._find_pdf_clusters_ocr_rects(page)
        # elif self.options.mode == OcrMode.PDF_BITMAPS_ONLY:
        #     ocr_rects = self._find_pdf_ocr_rects(page)
        return ocr_rects

    def _find_cluster_ocr_rects(self, page: Page) -> list[BoundingBox]:
        r"""
        1. Collect the bboxes of all layout clusters.
        2. Deduplicate the candidate ocr_rects.
        """
        if page.predictions.layout is None:
            return []

        # Use every layout detection bbox as an initial ocr_rect
        ocr_rects = [c.bbox for c in page.predictions.layout.clusters]

        # Deduplicate the ocr_rects
        _, ocr_rects = self._deduplicate_rects(page.size, ocr_rects)

        return ocr_rects

    def _find_pdf_clusters_ocr_rects(self, page: Page) -> list[BoundingBox]:
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
            return self._find_cluster_ocr_rects(page)

        # Create index for the text PDF cells
        p = index.Property()
        p.dimension = 2
        text_index = index.Index(properties=p)
        for i, text_cell in enumerate(page._backend.get_text_cells()):
            text_index.insert(i, text_cell.rect.to_bounding_box().as_tuple())

        # Create index for the non-text PDF cells
        non_text_index = index.Index(properties=p)
        for i, bbox in enumerate(page._backend.get_bitmap_rects()):
            non_text_index.insert(i, bbox.as_tuple())

        # Collect the non-eliminated cluster bboxes
        ocr_rects: list[BoundingBox] = []
        for cluster in page.predictions.layout.clusters:
            cluster_bbox_tuple = cluster.bbox.as_tuple()
            text_overlaps = list(text_index.intersection(cluster_bbox_tuple))
            non_text_overlaps = list(non_text_index.intersection(cluster_bbox_tuple))

            # Get the clusters that overlap with non-txt PDF cells
            if len(non_text_overlaps) > 0:
                ocr_rects.append(cluster.bbox)
            # And the ones that don't overlap with any PDF cells
            elif len(text_overlaps) == 0:
                ocr_rects.append(cluster.bbox)

        # Deduplicate the surviving cluster bboxes.
        _, ocr_rects = self._deduplicate_rects(page.size, ocr_rects)

        return ocr_rects

    # def _find_pdf_ocr_rects(self, page: Page) -> list[BoundingBox]:
    #     r"""
    #     Compute the OCR rectangles coming ONLY from the programmatic PDF cells
    #
    #     1. Deduplicate the bitmap rects.
    #     2. If coverage > MAXOUT_COVERAGE_THRESHOLD, return a single bbox covering the entire page.
    #     3. Else if coverage > `bitmap_area_threshold`, return the deduplicated rects.
    #     4. Otherwise return an empty list.
    #     """
    #     if page._backend is None:
    #         return []
    #
    #     # Get the programmatic PDF cells and deduplicate them
    #     bitmap_rects = page._backend.get_bitmap_rects()
    #     coverage, ocr_rects = self._deduplicate_rects(page.size, bitmap_rects, 20)
    #
    #     # return full-page rectangle if page is dominantly covered with bitmaps
    #     if coverage > max(
    #         BaseOcrModel.MAXOUT_COVERAGE_THRESHOLD, self.options.bitmap_area_threshold
    #     ):
    #         return [
    #             BoundingBox(
    #                 l=0,
    #                 t=0,
    #                 r=page.size.width,
    #                 b=page.size.height,
    #                 coord_origin=CoordOrigin.TOPLEFT,
    #             )
    #         ]
    #     # return individual rectangles if the bitmap coverage is above the threshold
    #     elif coverage > self.options.bitmap_area_threshold:
    #         return ocr_rects
    #
    #     # Overall coverage of bitmaps is too low, drop all bitmap rectangles.
    #     return []

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

    def _deduplicate_rects2(
        self, size: Size, rects: Iterable[BoundingBox], dilation_size=0
    ) -> tuple[float, list[BoundingBox]]:
        r"""
        Instead of rasterizing the page into a 1-bit image and running connected
        components over the pixels, this works directly on the rectangle
        geometry, which is O(page_pixels)-independent and typically much cheaper
        when the number of rects is small:

        1. Normalize each rect to an axis-aligned box, grow it by
           ``dilation_size / 2`` on every side (mirroring the binary dilation)
           and clip it to the page bounds.
        2. Merge overlapping/touching rects into connected components using an
           R-tree spatial index (O(N log N + K) neighbor queries) plus union-find
           for transitive grouping.
        3. For each component, return the bounding box of the union of its rects.
        4. Compute the coverage as the exact area of the union of all rects
           (sweep-line / Klee's algorithm) divided by the page area.
        """
        page_w = round(size.width)
        page_h = round(size.height)
        page_area = float(page_w * page_h)

        # 1. Normalize, dilate and clip the input rects.
        pad = dilation_size / 2.0
        boxes: list[tuple[float, float, float, float]] = []
        for rect in rects:
            x0, y0, x1, y1 = rect.as_tuple()
            # Corners may be unordered depending on the coord origin: normalize.
            left, right = (x0, x1) if x0 <= x1 else (x1, x0)
            top, bottom = (y0, y1) if y0 <= y1 else (y1, y0)
            # Grow (dilation) and clip to the page bounds.
            left = max(0.0, left - pad)
            top = max(0.0, top - pad)
            right = min(float(page_w), right + pad)
            bottom = min(float(page_h), bottom + pad)
            if right <= left or bottom <= top:
                continue  # degenerate after clipping
            boxes.append((left, top, right, bottom))

        if not boxes or page_area <= 0:
            return (0.0, [])

        arr = np.array(boxes, dtype=float)  # columns: left, top, right, bottom
        n = len(boxes)

        # 2. Connected components: R-tree neighbor queries + union-find.
        parent = list(range(n))

        def find(i: int) -> int:
            root = i
            while parent[root] != root:
                root = parent[root]
            while parent[i] != root:  # path compression
                parent[i], i = root, parent[i]
            return root

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        spatial = index.Index()
        for i, box in enumerate(boxes):
            spatial.insert(i, box)
        for i, box in enumerate(boxes):
            for j in spatial.intersection(box):
                if j > i:
                    union(i, j)

        # 3. Union bounding box per connected component.
        components: dict[int, list[int]] = {}
        for i in range(n):
            components.setdefault(find(i), []).append(i)

        bounding_boxes = [
            BoundingBox(
                l=float(arr[members, 0].min()),
                t=float(arr[members, 1].min()),
                r=float(arr[members, 2].max()),
                b=float(arr[members, 3].max()),
                coord_origin=CoordOrigin.TOPLEFT,
            )
            for members in components.values()
        ]

        # 4. Exact area of the union of all rects via a vertical sweep line.
        union_area = 0.0
        x_edges = np.unique(arr[:, [0, 2]])
        for k in range(len(x_edges) - 1):
            x_lo = x_edges[k]
            x_hi = x_edges[k + 1]
            dx = x_hi - x_lo
            if dx <= 0:
                continue
            # Rects spanning the whole [x_lo, x_hi] slab contribute here.
            active = arr[(arr[:, 0] <= x_lo) & (arr[:, 2] >= x_hi)]
            if active.size == 0:
                continue
            # Union length of the active y-intervals.
            intervals = active[:, [1, 3]]
            intervals = intervals[intervals[:, 0].argsort()]
            cur_top, cur_bottom = intervals[0]
            y_len = 0.0
            for y_top, y_bottom in intervals[1:]:
                if y_top > cur_bottom:
                    y_len += cur_bottom - cur_top
                    cur_top, cur_bottom = y_top, y_bottom
                else:
                    cur_bottom = max(cur_bottom, y_bottom)
            y_len += cur_bottom - cur_top
            union_area += dx * y_len

        area_frac = union_area / page_area
        return (area_frac, bounding_boxes)

    def post_process_cells(
        self,
        ocr_cells: list[TextCell],
        page: Page,
        conv_res: ConversionResult,
    ) -> None:
        r"""
        Post-process the OCR cells and update the page object.
        Updates parsed_page.textline_cells directly.
        """
        # Get existing cells from the read-only property
        existing_cells = page.cells

        # Combine existing and OCR cells with overlap filtering
        if self.options.mode == OcrMode.FULL_PAGE_OCR:
            final_cells = ocr_cells
        else:
            filtered_ocr_cells = self._filter_ocr_cells(ocr_cells, existing_cells)
            final_cells = list(existing_cells) + filtered_ocr_cells

        # Re-index in-place
        for i, cell in enumerate(final_cells):
            cell.index = i

        assert page.parsed_page is not None

        # Update parsed_page.textline_cells directly
        page.parsed_page.textline_cells = final_cells
        page.parsed_page.has_lines = len(final_cells) > 0

        # In OcrMode.FULL_PAGE_OCR, PDF-extracted word/char cells are unreliable.
        # Filter out cells where from_ocr=False, keeping any OCR generated cells.
        # This ensures downstream components (e.g., table structure model) fall back to
        # OCR-extracted textline cells.
        if self.options.mode == OcrMode.FULL_PAGE_OCR:
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

    def _filter_ocr_cells(
        self, ocr_cells: list[TextCell], programmatic_cells: list[TextCell]
    ) -> list[TextCell]:
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
    def get_options_type(cls) -> type[OcrOptions]:
        pass
