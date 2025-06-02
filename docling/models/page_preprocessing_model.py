import re
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import ImageDraw
from pydantic import BaseModel

from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.settings import settings
from docling.models.base_model import BasePageModel
from docling.utils.profiling import TimeRecorder


class PagePreprocessingOptions(BaseModel):
    images_scale: Optional[float]
    create_parsed_page: bool


class PagePreprocessingModel(BasePageModel):
    def __init__(self, options: PagePreprocessingOptions):
        self.options = options

        # Pre-compiled regex patterns for efficiency
        self.GLYPH_RE = re.compile(r"GLYPH<[0-9A-Fa-f]+>")
        self.SLASH_G_RE = re.compile(r"(?:/G\d+){2,}")
        self.FRAG_RE = re.compile(r"\b[A-Za-z](?:/[a-z]{1,3}\.[a-z]{1,3}){2,}\b")
        self.SLASH_NUMBER_GARBAGE_RE = re.compile(
            r"(?:/\w+\s*){2,}"
        )  # Two or more "/token " sequences

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
            else:
                with TimeRecorder(conv_res, "page_parse"):
                    page = self._populate_page_images(page)
                    page = self._parse_page_cells(conv_res, page)
                yield page

    # Generate the page image and store it in the page object
    def _populate_page_images(self, page: Page) -> Page:
        # default scale
        page.get_image(
            scale=1.0
        )  # puts the page image on the image cache at default scale

        images_scale = self.options.images_scale
        # user requested scales
        if images_scale is not None:
            page._default_image_scale = images_scale
            page.get_image(
                scale=images_scale
            )  # this will trigger storing the image in the internal cache

        return page

    # Extract and populate the page cells and store it in the page object
    def _parse_page_cells(self, conv_res: ConversionResult, page: Page) -> Page:
        assert page._backend is not None

        page.cells = list(page._backend.get_text_cells())

        if self.options.create_parsed_page:
            page.parsed_page = page._backend.get_segmented_page()

        # Rate the text quality from the PDF parser, and aggregate on page
        text_scores = []
        for c in page.cells:
            score = self.rate_text_quality(c.text)
            text_scores.append(score)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "Mean of empty slice", RuntimeWarning, "numpy"
            )
            conv_res.confidence.pages[page.page_no].parse_score = float(
                np.nanquantile(
                    text_scores, q=0.10
                )  # To emphasise problems in the parse_score, we take the 10% percentile score of all text cells.
            )

        # DEBUG code:
        def draw_text_boxes(image, cells, show: bool = False):
            draw = ImageDraw.Draw(image)
            for c in cells:
                x0, y0, x1, y1 = (
                    c.to_bounding_box().l,
                    c.to_bounding_box().t,
                    c.to_bounding_box().r,
                    c.to_bounding_box().b,
                )

                draw.rectangle([(x0, y0), (x1, y1)], outline="red")
            if show:
                image.show()
            else:
                out_path: Path = (
                    Path(settings.debug.debug_output_path)
                    / f"debug_{conv_res.input.file.stem}"
                )
                out_path.mkdir(parents=True, exist_ok=True)

                out_file = out_path / f"cells_page_{page.page_no:05}.png"
                image.save(str(out_file), format="png")

        if settings.debug.visualize_cells:
            draw_text_boxes(page.get_image(scale=1.0), page.cells)

        return page

    def rate_text_quality(self, text: str) -> float:
        # Hard errors: if any of these patterns are found, return 0.0 immediately.
        blacklist_chars = ["�"]
        if (
            any(text.find(c) >= 0 for c in blacklist_chars)
            or self.GLYPH_RE.search(text)
            or self.SLASH_G_RE.search(text)
            or self.SLASH_NUMBER_GARBAGE_RE.match(
                text
            )  # Check if text is mostly slash-number pattern
        ):
            return 0.0

        penalty = 0.0

        # Apply a penalty only if the fragmented words pattern occurs at least three times.
        frag_matches = self.FRAG_RE.findall(text)
        if len(frag_matches) >= 3:
            penalty += 0.1 * len(frag_matches)

        # Additional heuristic: if the average token length is below 2, add a penalty.
        # tokens = text.split()
        # if tokens and (sum(map(len, tokens)) / len(tokens)) < 2:
        #    penalty += 0.2

        return max(1.0 - penalty, 0.0)
