import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from docling_core.types.doc import BoundingBox
from pydantic import AnyUrl, BaseModel, ValidationError

from docling.datamodel.base_models import (
    AssembledUnit,
    ContainerElement,
    FigureElement,
    Page,
    PageElement,
    Table,
    TextElement,
)
from docling.datamodel.document import ConversionResult
from docling.models.base_model import BasePageModel
from docling.models.stages.layout.layout_model import LayoutModel
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)

# Ligature normalization map.
# Covers the Unicode Alphabetic Presentation Forms block (U+FB00-U+FB06),
# the Latin capital/small ligature IJ (U+0132/U+0133), and the Private-Use
# Area character U+F0A0 (emitted by some PDF fonts as a spurious glyph).
_LIGATURE_MAP: Dict[str, str] = {
    "\ufb00": "ff",  # ﬀ Latin small ligature ff
    "\ufb01": "fi",  # ﬁ Latin small ligature fi
    "\ufb02": "fl",  # ﬂ Latin small ligature fl
    "\ufb03": "ffi",  # ﬃ Latin small ligature ffi
    "\ufb04": "ffl",  # ﬄ Latin small ligature ffl
    "\ufb05": "st",  # ﬅ Latin small ligature long s t
    "\ufb06": "st",  # ﬆ Latin small ligature st
    "\u0132": "IJ",  # Ĳ Latin capital ligature IJ (used in Dutch)
    "\u0133": "ij",  # ĳ Latin small ligature ij (used in Dutch)
    "\uf0a0": "",  # Private-use glyph emitted by some PDF fonts; discard
}
# Matches any ligature character in the map, optionally followed by a spurious
# space before a word character (to absorb spaces inserted by PDF parsers
# between a ligature glyph and the rest of the word, e.g. "ﬁ eld" → "field").
# Note: U+0132/U+0133 and U+F0A0 are listed as alternates (not a range) to
# avoid an invalid descending range with U+FB06.
_LIGATURE_RE = re.compile(r"([\ufb00-\ufb06]|\u0132|\u0133|\uf0a0)( (?=\w))?")


class PageAssembleOptions(BaseModel):
    pass


class PageAssembleModel(BasePageModel):
    # Minimum fraction of a cluster's area that a hyperlink rect must cover
    # to be considered a match (avoids false positives from adjacent links).
    _HYPERLINK_COVERAGE_THRESHOLD = 0.5

    def __init__(self, options: PageAssembleOptions):
        self.options = options

    @staticmethod
    def _match_hyperlink(
        cluster_bbox: BoundingBox,
        page: Page,
    ) -> Optional[Union[AnyUrl, Path]]:
        """Pick the hyperlink annotation with the highest spatial overlap on cluster_bbox.

        Hyperlink rects are BOTTOMLEFT-origin; cluster bboxes are TOPLEFT-origin.
        """
        if page.parsed_page is None or not page.parsed_page.hyperlinks:
            return None

        if page.size is None:
            return None

        page_height = page.size.height

        # Accumulate coverage per URI — a single hyperlink may span multiple
        # annotation rectangles (e.g. a URL that wraps across lines).
        coverage_by_uri: Dict[str, float] = {}

        for hl in page.parsed_page.hyperlinks:
            if hl.uri is None:
                continue

            uri_str = str(hl.uri)
            hl_bbox = hl.rect.to_bounding_box().to_top_left_origin(page_height)
            coverage_by_uri[uri_str] = coverage_by_uri.get(
                uri_str, 0.0
            ) + cluster_bbox.intersection_over_self(hl_bbox)

        if not coverage_by_uri:
            return None

        best_uri = max(coverage_by_uri.items(), key=lambda x: x[1])[0]
        if coverage_by_uri[best_uri] < PageAssembleModel._HYPERLINK_COVERAGE_THRESHOLD:
            return None

        try:
            return AnyUrl(best_uri)
        except ValidationError:
            return Path(best_uri)

    def sanitize_text(self, lines):
        if len(lines) == 0:
            return ""

        for ix, line in enumerate(lines[1:]):
            prev_line = lines[ix]

            if prev_line.endswith("-"):
                prev_words = re.findall(r"\b[\w]+\b", prev_line)
                line_words = re.findall(r"\b[\w]+\b", line)

                if (
                    len(prev_words)
                    and len(line_words)
                    and prev_words[-1].isalnum()
                    and line_words[0].isalnum()
                ):
                    lines[ix] = prev_line[:-1]
            else:
                lines[ix] += " "

        sanitized_text = "".join(lines)

        # Text normalization
        sanitized_text = sanitized_text.replace("⁄", "/")  # noqa: RUF001
        sanitized_text = sanitized_text.replace("’", "'")  # noqa: RUF001
        sanitized_text = sanitized_text.replace("‘", "'")  # noqa: RUF001
        sanitized_text = sanitized_text.replace("“", '"')
        sanitized_text = sanitized_text.replace("”", '"')
        sanitized_text = sanitized_text.replace("•", "·")
        # Ligature expansion: replace ligature characters with their ASCII equivalents.
        # For the traditional fb00-fb06 ligatures (e.g. ﬁ, ﬂ), any spurious space
        # inserted by the PDF parser between the glyph and the rest of the word is
        # absorbed (e.g. "ﬁ eld" → "field").
        # For U+0132/U+0133 (Dutch IJ/ij) and U+F0A0 (PUA discard glyph), any
        # captured trailing space is re-emitted so that real word boundaries are
        # preserved (e.g. "Ĳ is" → "IJ is", "hello\uf0a0 world" → "hello world").
        sanitized_text = _LIGATURE_RE.sub(
            lambda m: _LIGATURE_MAP[m.group(1)]
            + ("" if "\ufb00" <= m.group(1) <= "\ufb06" else (m.group(2) or "")),
            sanitized_text,
        )

        return sanitized_text.strip()  # Strip any leading or trailing whitespace

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
            else:
                with TimeRecorder(conv_res, "page_assemble"):
                    assert page.predictions.layout is not None

                    # assembles some JSON output page by page.

                    elements: List[PageElement] = []
                    headers: List[PageElement] = []
                    body: List[PageElement] = []

                    for cluster in page.predictions.layout.clusters:
                        # _log.info("Cluster label seen:", cluster.label)
                        if cluster.label in LayoutModel.TEXT_ELEM_LABELS:
                            textlines = [
                                cell.text.replace("\x02", "-").strip()
                                for cell in cluster.cells
                                if len(cell.text.strip()) > 0
                            ]
                            text = self.sanitize_text(textlines)
                            hyperlink = self._match_hyperlink(cluster.bbox, page)
                            text_el = TextElement(
                                label=cluster.label,
                                id=cluster.id,
                                text=text,
                                hyperlink=hyperlink,
                                page_no=page.page_no,
                                cluster=cluster,
                            )
                            elements.append(text_el)

                            if cluster.label in LayoutModel.PAGE_HEADER_LABELS:
                                headers.append(text_el)
                            else:
                                body.append(text_el)
                        elif cluster.label in LayoutModel.TABLE_LABELS:
                            tbl = None
                            if page.predictions.tablestructure:
                                tbl = page.predictions.tablestructure.table_map.get(
                                    cluster.id, None
                                )
                            if not tbl:  # fallback: add table without structure, if it isn't present
                                tbl = Table(
                                    label=cluster.label,
                                    id=cluster.id,
                                    text="",
                                    otsl_seq=[],
                                    table_cells=[],
                                    cluster=cluster,
                                    page_no=page.page_no,
                                )

                            elements.append(tbl)
                            body.append(tbl)
                        elif cluster.label == LayoutModel.FIGURE_LABEL:
                            fig = None
                            if page.predictions.figures_classification:
                                fig = page.predictions.figures_classification.figure_map.get(
                                    cluster.id, None
                                )
                            if not fig:  # fallback: add figure without classification, if it isn't present
                                fig = FigureElement(
                                    label=cluster.label,
                                    id=cluster.id,
                                    text="",
                                    data=None,
                                    cluster=cluster,
                                    page_no=page.page_no,
                                )
                            elements.append(fig)
                            body.append(fig)
                        elif cluster.label in LayoutModel.CONTAINER_LABELS:
                            container_el = ContainerElement(
                                label=cluster.label,
                                id=cluster.id,
                                page_no=page.page_no,
                                cluster=cluster,
                            )
                            elements.append(container_el)
                            body.append(container_el)

                    page.assembled = AssembledUnit(
                        elements=elements, headers=headers, body=body
                    )

                yield page
