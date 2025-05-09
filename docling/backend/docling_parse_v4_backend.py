import logging
from collections.abc import Iterable
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union, List, Dict
from concurrent.futures import ThreadPoolExecutor
import time

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

# Constants
AREA_THRESHOLD = 0  # Threshold for bitmap area processing
OVERLAP_THRESHOLD = 0.5  # Threshold for text cell overlap
IMAGE_SCALE_FACTOR = 1.5  # Scale factor for rendering images
MAX_WORKERS = 4  # Maximum number of worker threads for parallel processing
PAGE_CACHE_SIZE = 5  # Number of pages to keep in memory


class DoclingParseV4PageBackend(PdfPageBackend):
    def __init__(self, parsed_page: SegmentedPdfPage, page_obj: PdfPage):
        self._ppage = page_obj
        self._dpage = parsed_page
        self.valid = parsed_page is not None
        self._page_size: Optional[Size] = None
        self._textline_cells_top_left: Optional[List[TextCell]] = None
        self._creation_time = time.time()
        self._last_access_time = self._creation_time

    def is_valid(self) -> bool:
        self._last_access_time = time.time()
        return self.valid

    @lru_cache(maxsize=1)
    def get_size(self) -> Size:
        """Get page size with caching for better performance."""
        self._last_access_time = time.time()
        if self._page_size is None:
            with pypdfium2_lock:
                self._page_size = Size(
                    width=self._ppage.get_width(), height=self._ppage.get_height()
                )
        return self._page_size

    def _get_textline_cells_top_left(self) -> List[TextCell]:
        """Get text cells converted to top-left origin (with caching)."""
        self._last_access_time = time.time()
        if self._textline_cells_top_left is None:
            page_size = self.get_size()
            self._textline_cells_top_left = []
            
            for cell in self._dpage.textline_cells:
                # Create a copy of the cell with top-left origin
                cell_copy = cell.model_copy(deep=True)
                cell_copy.rect = cell_copy.rect.to_top_left_origin(page_size.height)
                self._textline_cells_top_left.append(cell_copy)
                
        return self._textline_cells_top_left

    def get_text_in_rect(self, bbox: BoundingBox) -> str:
        """Extract text from cells that overlap with the given bounding box."""
        self._last_access_time = time.time()
        if not self.valid:
            return ""
            
        text_pieces = []
        page_size = self.get_size()
        scale = 1

        # Ensure bbox is in top-left origin
        if bbox.coord_origin != CoordOrigin.TOPLEFT:
            bbox = bbox.to_top_left_origin(page_height=page_size.height)

        for cell in self._get_textline_cells_top_left():
            cell_bbox = cell.rect.to_bounding_box().scaled(scale)
            
            # Calculate intersection area
            overlap_frac = cell_bbox.intersection_area_with(bbox) / cell_bbox.area()
            
            if overlap_frac > OVERLAP_THRESHOLD:
                text_pieces.append(cell.text)

        return " ".join(text_pieces)

    def get_text_in_rects(self, bboxes: List[BoundingBox]) -> List[str]:
        """Extract text from multiple regions in a single pass for better performance."""
        self._last_access_time = time.time()
        if not self.valid or not bboxes:
            return [""] * len(bboxes)
            
        page_size = self.get_size()
        results = [""] * len(bboxes)
        text_pieces = [[] for _ in range(len(bboxes))]
        
        # Ensure all bboxes are in top-left origin
        normalized_bboxes = []
        for bbox in bboxes:
            if bbox.coord_origin != CoordOrigin.TOPLEFT:
                normalized_bboxes.append(bbox.to_top_left_origin(page_height=page_size.height))
            else:
                normalized_bboxes.append(bbox)
        
        # Process all cells once
        for cell in self._get_textline_cells_top_left():
            cell_bbox = cell.rect.to_bounding_box()
            
            # Check against all bboxes
            for i, bbox in enumerate(normalized_bboxes):
                overlap_frac = cell_bbox.intersection_area_with(bbox) / cell_bbox.area()
                if overlap_frac > OVERLAP_THRESHOLD:
                    text_pieces[i].append(cell.text)
                    
        # Join text pieces for each bbox
        for i in range(len(bboxes)):
            results[i] = " ".join(text_pieces[i])
                
        return results

    def get_segmented_page(self) -> Optional[SegmentedPdfPage]:
        self._last_access_time = time.time()
        return self._dpage

    def get_text_cells(self) -> Iterable[TextCell]:
        """Return text cells in top-left origin coordinates."""
        self._last_access_time = time.time()
        if not self.valid:
            return []
            
        return self._get_textline_cells_top_left()

    def get_bitmap_rects(self, scale: float = 1) -> Iterable[BoundingBox]:
        """Get bounding boxes for bitmap images on the page."""
        self._last_access_time = time.time()
        if not self.valid:
            return []
            
        page_height = self.get_size().height
        
        for img in self._dpage.bitmap_resources:
            cropbox = img.rect.to_bounding_box().to_top_left_origin(page_height)

            if cropbox.area() > AREA_THRESHOLD:
                yield cropbox.scaled(scale=scale)

    def get_page_image(
        self, scale: float = 1, cropbox: Optional[BoundingBox] = None
    ) -> Image.Image:
        """Render the page as an image, optionally cropped to a specific region."""
        self._last_access_time = time.time()
        page_size = self.get_size()

        # Skip rendering if page is invalid
        if not self.valid:
            # Return blank image of appropriate size
            width = round(page_size.width * scale)
            height = round(page_size.height * scale)
            return Image.new('RGB', (width, height), color='white')

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
                    scale=scale * IMAGE_SCALE_FACTOR,
                    rotation=0,
                    crop=padbox.as_tuple(),
                )
                .to_pil()
                .resize(
                    size=(round(cropbox.width * scale), round(cropbox.height * scale))
                )
            )

        return image

    def unload(self):
        """Clean up resources."""
        self._ppage = None
        self._dpage = None
        self._page_size = None
        self._textline_cells_top_left = None


class DoclingParseV4DocumentBackend(PdfDocumentBackend):
    def __init__(self, in_doc: "InputDocument", path_or_stream: Union[BytesIO, Path]):
        """Initialize the document backend with error handling."""
        super().__init__(in_doc, path_or_stream)
        self._page_cache: Dict[int, DoclingParseV4PageBackend] = {}
        self._executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

        try:
            with pypdfium2_lock:
                self._pdoc = pdfium.PdfDocument(self.path_or_stream)
                
            self.parser = DoclingPdfParser(loglevel="fatal")
            self.dp_doc: PdfDocument = self.parser.load(path_or_stream=self.path_or_stream)
            
            if self.dp_doc is None:
                raise RuntimeError(f"Failed to load document with docling-parse v4")
                
        except Exception as e:
            _log.error(f"Error initializing DoclingParseV4: {str(e)}")
            raise RuntimeError(
                f"docling-parse v4 could not load document {self.document_hash}: {str(e)}"
            )

    def page_count(self) -> int:
        """Get the number of pages in the document with validation."""
        len_1 = len(self._pdoc)
        len_2 = self.dp_doc.number_of_pages()

        if len_1 != len_2:
            _log.warning(f"Inconsistent number of pages: {len_1}!={len_2}")

        return len_2

    def _manage_cache(self):
        """Manage page cache size by removing least recently used pages."""
        if len(self._page_cache) > PAGE_CACHE_SIZE:
            # Sort pages by last access time
            sorted_pages = sorted(
                self._page_cache.items(), 
                key=lambda item: item[1]._last_access_time
            )
            # Remove oldest pages
            pages_to_remove = len(self._page_cache) - PAGE_CACHE_SIZE
            for i in range(pages_to_remove):
                page_no, page = sorted_pages[i]
                page.unload()
                del self._page_cache[page_no]

    def load_page(
        self, page_no: int, create_words: bool = True, create_textlines: bool = True
    ) -> DoclingParseV4PageBackend:
        """Load a specific page with error handling and caching."""
        # Check cache first
        if page_no in self._page_cache:
            self._page_cache[page_no]._last_access_time = time.time()
            return self._page_cache[page_no]
            
        try:
            with pypdfium2_lock:
                page_backend = DoclingParseV4PageBackend(
                    self.dp_doc.get_page(
                        page_no + 1,
                        create_words=create_words,
                        create_textlines=create_textlines,
                    ),
                    self._pdoc[page_no],
                )
            
            # Add to cache
            self._page_cache[page_no] = page_backend
            self._manage_cache()
            
            return page_backend
        except Exception as e:
            _log.error(f"Error loading page {page_no}: {str(e)}")
            # Return an invalid page backend instead of raising an exception
            with pypdfium2_lock:
                return DoclingParseV4PageBackend(None, self._pdoc[page_no])

    def load_pages_in_parallel(
        self, page_numbers: List[int], create_words: bool = True, create_textlines: bool = True
    ) -> List[DoclingParseV4PageBackend]:
        """Load multiple pages in parallel for better performance."""
        # Check which pages are already in cache
        pages_to_load = []
        results = [None] * len(page_numbers)
        
        for i, page_no in enumerate(page_numbers):
            if page_no in self._page_cache:
                self._page_cache[page_no]._last_access_time = time.time()
                results[i] = self._page_cache[page_no]
            else:
                pages_to_load.append((i, page_no))
        
        if not pages_to_load:
            return results
            
        # Define a function to load a single page
        def load_single_page(idx_page_tuple):
            idx, page_no = idx_page_tuple
            try:
                with pypdfium2_lock:
                    page = self.dp_doc.get_page(
                        page_no + 1,
                        create_words=create_words,
                        create_textlines=create_textlines,
                    )
                    ppage = self._pdoc[page_no]
                    
                page_backend = DoclingParseV4PageBackend(page, ppage)
                return idx, page_no, page_backend
            except Exception as e:
                _log.error(f"Error loading page {page_no} in parallel: {str(e)}")
                with pypdfium2_lock:
                    return idx, page_no, DoclingParseV4PageBackend(None, self._pdoc[page_no])
        
        # Load pages in parallel
        for idx, page_no, page_backend in self._executor.map(load_single_page, pages_to_load):
            results[idx] = page_backend
            self._page_cache[page_no] = page_backend
        
        self._manage_cache()
        return results

    def is_valid(self) -> bool:
        """Check if the document is valid."""
        return self.dp_doc is not None and self.page_count() > 0

    def unload(self):
        """Clean up resources properly."""
        super().unload()
        
        # Unload all cached pages
        for page in self._page_cache.values():
            page.unload()
        self._page_cache.clear()
        
        # Shutdown executor
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
        
        if hasattr(self, 'dp_doc') and self.dp_doc is not None:
            self.dp_doc.unload()
        
        if hasattr(self, '_pdoc') and self._pdoc is not None:
            with pypdfium2_lock:
                self._pdoc.close()
                
        self._pdoc = None
        self.dp_doc = None
        self.parser = None
