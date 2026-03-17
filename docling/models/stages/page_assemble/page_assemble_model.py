import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from docling_core.types.doc import BoundingBox, DocItemLabel
from pydantic import AnyUrl, BaseModel, ValidationError

from docling.datamodel.base_models import (
    AssembledUnit,
    Cluster,
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

# Ligature normalization map (Unicode Alphabetic Presentation Forms block U+FB00-U+FB06)
_LIGATURE_MAP: Dict[str, str] = {
    "\ufb00": "ff",  # ﬀ Latin small ligature ff
    "\ufb01": "fi",  # ﬁ Latin small ligature fi
    "\ufb02": "fl",  # ﬂ Latin small ligature fl
    "\ufb03": "ffi",  # ﬃ Latin small ligature ffi
    "\ufb04": "ffl",  # ﬄ Latin small ligature ffl
    "\ufb05": "st",  # ﬅ Latin small ligature long s t
    "\ufb06": "st",  # ﬆ Latin small ligature st
}
# Matches a ligature character optionally followed by a space before a word character
# (to absorb spurious spaces inserted by PDF parsers between a ligature glyph and the
# rest of the word, e.g. "ﬁ eld" → "field")
_LIGATURE_RE = re.compile(r"([\ufb00-\ufb06])( (?=\w))?")


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
        matched_indices: Optional[set[int]] = None,
    ) -> Optional[Union[AnyUrl, Path]]:
        """Pick the hyperlink annotation with the highest spatial overlap on cluster_bbox.

        Hyperlink rects are BOTTOMLEFT-origin; cluster bboxes are TOPLEFT-origin.

        If *matched_indices* is provided, the indices of every hyperlink that
        contributed to the winning URI are added to the set so callers can track
        which hyperlinks have been consumed.
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

        best_uri = max(coverage_by_uri, key=coverage_by_uri.get)  # type: ignore[arg-type]
        if coverage_by_uri[best_uri] < PageAssembleModel._HYPERLINK_COVERAGE_THRESHOLD:
            return None

        # Only mark hyperlinks as consumed when the cluster actually matched
        # (coverage >= threshold).  Below-threshold overlaps must remain
        # unconsumed so they become fallback REFERENCE items rather than
        # being silently dropped.
        if matched_indices is not None:
            overlapping_matches: List[tuple[int, float]] = []
            for idx, hl in enumerate(page.parsed_page.hyperlinks):
                if hl.uri is None:
                    continue
                if str(hl.uri) != best_uri:
                    continue
                hl_bbox = hl.rect.to_bounding_box().to_top_left_origin(page_height)
                overlap = cluster_bbox.intersection_over_self(hl_bbox)
                if overlap > 0:
                    overlapping_matches.append((idx, overlap))

            overlapping_matches.sort(key=lambda item: item[1], reverse=True)

            consumed_overlap = 0.0
            cutoff_overlap = None
            for idx, overlap in overlapping_matches:
                matched_indices.add(idx)
                consumed_overlap += overlap
                cutoff_overlap = overlap
                if consumed_overlap >= PageAssembleModel._HYPERLINK_COVERAGE_THRESHOLD:
                    break

            if cutoff_overlap is not None:
                for idx, overlap in overlapping_matches:
                    if idx in matched_indices:
                        continue
                    if overlap == cutoff_overlap:
                        matched_indices.add(idx)

        try:
            return AnyUrl(best_uri)
        except ValidationError:
            return Path(best_uri)

    @staticmethod
    def _collect_unmatched_hyperlinks(
        page: Page,
        matched_indices: set[int],
        next_cluster_id: int,
        text_cluster_bboxes: List[BoundingBox],
    ) -> List[TextElement]:
        """Create synthetic REFERENCE TextElements for hyperlinks not matched to any cluster.

        Only hyperlinks that still overlap a text cluster are materialized. This
        preserves recovery for missed text links without inventing visible URL
        text for non-text annotations such as linked figures.

        Each unmatched hyperlink annotation becomes its own element (no
        deduplication) so that repeated URLs at different page positions are
        preserved with correct bounding boxes for reading-order placement.
        """
        if page.parsed_page is None or not page.parsed_page.hyperlinks:
            return []

        if page.size is None or not text_cluster_bboxes:
            return []

        page_height = page.size.height

        elements: List[TextElement] = []
        cid = next_cluster_id

        for idx, hl in enumerate(page.parsed_page.hyperlinks):
            if idx in matched_indices:
                continue
            if hl.uri is None:
                continue

            uri_str = str(hl.uri)
            bbox = hl.rect.to_bounding_box().to_top_left_origin(page_height)
            if not any(
                text_bbox.intersection_over_self(bbox) > 0
                for text_bbox in text_cluster_bboxes
            ):
                continue

            try:
                hyperlink: Union[AnyUrl, Path] = AnyUrl(uri_str)
            except ValidationError:
                hyperlink = Path(uri_str)

            cluster = Cluster(
                id=cid,
                label=DocItemLabel.REFERENCE,
                bbox=bbox,
            )
            elements.append(
                TextElement(
                    label=DocItemLabel.REFERENCE,
                    id=cid,
                    text=uri_str,
                    hyperlink=hyperlink,
                    page_no=page.page_no,
                    cluster=cluster,
                )
            )
            cid += 1

        return elements

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
        # Ligature expansion: replace ligature characters with their ASCII equivalents,
        # absorbing any spurious space inserted by the PDF parser between the ligature
        # glyph and the following word characters (e.g. "ﬁ eld" → "field").
        sanitized_text = _LIGATURE_RE.sub(
            lambda m: _LIGATURE_MAP[m.group(1)], sanitized_text
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
                    matched_indices: set[int] = set()
                    text_cluster_bboxes: List[BoundingBox] = []

                    for cluster in page.predictions.layout.clusters:
                        # _log.info("Cluster label seen:", cluster.label)
                        if cluster.label in LayoutModel.TEXT_ELEM_LABELS:
                            text_cluster_bboxes.append(cluster.bbox)
                            textlines = [
                                cell.text.replace("\x02", "-").strip()
                                for cell in cluster.cells
                                if len(cell.text.strip()) > 0
                            ]
                            text = self.sanitize_text(textlines)
                            hyperlink = self._match_hyperlink(
                                cluster.bbox, page, matched_indices
                            )
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

                    # Propagate unmatched hyperlinks as REFERENCE items
                    # so they are not silently lost.
                    max_cluster_id = max(
                        (c.id for c in page.predictions.layout.clusters),
                        default=-1,
                    )
                    unmatched = self._collect_unmatched_hyperlinks(
                        page,
                        matched_indices,
                        max_cluster_id + 1,
                        text_cluster_bboxes,
                    )
                    elements.extend(unmatched)
                    body.extend(unmatched)

                    page.assembled = AssembledUnit(
                        elements=elements, headers=headers, body=body
                    )

                yield page
