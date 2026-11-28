# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from docling_core.types.doc import BoundingBox, DocItemLabel
from docling_core.types.doc.page import TextCell
from pydantic import AnyUrl, BaseModel, ValidationError

from docling.datamodel.base_models import (
    AssembledUnit,
    Cluster,
    ContainerElement,
    FieldRegionElement,
    FigureElement,
    FormFieldPrediction,
    Page,
    PageElement,
    Table,
    TextElement,
)
from docling.datamodel.document import ConversionResult
from docling.models.base_layout_model import (
    CONTAINER_LABELS,
    FIGURE_LABEL,
    PAGE_HEADER_LABELS,
    TABLE_LABELS,
    TEXT_ELEM_LABELS,
)
from docling.models.base_model import BasePageModel
from docling.utils.profiling import TimeRecorder
from docling.utils.text_normalization import (
    cell_text_lines,
    normalize_extracted_text,
)

_log = logging.getLogger(__name__)

# A drop cap is a single uppercase letter at least this many times as tall as
# the median text line of the page.
_DROP_CAP_HEIGHT_RATIO = 2.0

# Labels whose text can hold a drop cap or the paragraph it opens. Captions,
# footnotes, page furniture, code and formulas are left out to avoid merging
# unrelated text.
_DROP_CAP_TEXT_LABELS = {
    DocItemLabel.TEXT,
    DocItemLabel.SECTION_HEADER,
    DocItemLabel.LIST_ITEM,
}


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

    def sanitize_text(self, lines: List[str]) -> str:
        """Join the text lines of a cluster into a single normalized string.

        Lines are joined with a space, except across a hyphen that splits a
        word over the line break ("algo-" / "rithms"), which is dropped so
        that the two halves close up. A hyphen only splits a word when it is
        attached to the word it splits; one that follows whitespace is a
        literal character -- a separator dash, a bullet marker, or a wrapped
        hyphen-prefixed token -- so it is kept and its lines are joined like
        any other line break. An attached hyphen is also kept where no
        hyphenation rule could have placed it: after a single-character token
        ("n-" / "octanal"), before a continuation opening on a digit
        ("6-methyl-" / "5-hepten-2-one"), or after an uppercase abbreviation
        when the continuation is lowercase ("DNA-" / "binding"). There the
        hyphen belongs to the word.

        The joined text is then normalized: a few typographic characters are
        replaced by ASCII equivalents and ligatures are expanded.
        """
        if len(lines) == 0:
            return ""

        for ix, line in enumerate(lines[1:]):
            prev_line = lines[ix]

            if prev_line.endswith("-"):
                prev_words = re.findall(r"\b[\w]+\b", prev_line)
                line_words = re.findall(r"\b[\w]+\b", line)

                hyphen_attached_to_word = len(prev_line) > 1 and prev_line[-2].isalnum()
                hyphen_after_uppercase_abbrev = (
                    len(prev_words)
                    and len(line_words)
                    and prev_words[-1][-1].isupper()
                    and line_words[0][0].islower()
                )

                if (
                    hyphen_attached_to_word
                    and len(prev_words)
                    and len(line_words)
                    and prev_words[-1].isalnum()
                    and len(prev_words[-1]) > 1
                    and line_words[0].isalnum()
                    and not line_words[0][0].isdigit()
                    and not hyphen_after_uppercase_abbrev
                ):
                    lines[ix] = prev_line[:-1]
                elif not hyphen_attached_to_word:
                    lines[ix] += " "
            else:
                lines[ix] += " "

        sanitized_text = "".join(lines)

        sanitized_text = normalize_extracted_text(sanitized_text)

        return sanitized_text.strip()  # Strip any leading or trailing whitespace

    @staticmethod
    def _is_drop_cap(cell: TextCell, median_height: float) -> bool:
        """Whether a cell holds a drop cap: one oversized uppercase letter."""
        text = cell.text.strip()
        return (
            len(text) == 1
            and text.isalpha()
            and text.isupper()
            and cell.rect.to_bounding_box().height
            >= _DROP_CAP_HEIGHT_RATIO * median_height
        )

    @staticmethod
    def _opens_after_drop_cap(
        drop_cap: TextCell, cell: TextCell, median_height: float
    ) -> bool:
        """Whether a cell is the first line of the paragraph a drop cap opens.

        The line must continue the word the drop cap starts (open on a
        lowercase letter) and sit right of the drop cap: its left edge close to
        the drop cap's right edge, its top aligned with the drop cap's top and
        its baseline above the drop cap's baseline.
        """
        cap_box = drop_cap.rect.to_bounding_box()
        line_box = cell.rect.to_bounding_box()
        gap = line_box.l - cap_box.r
        return (
            cell.text.strip()[:1].islower()
            and -median_height <= gap <= max(cap_box.width, median_height)
            and abs(line_box.t - cap_box.t) <= median_height
            and cap_box.b - line_box.b >= 0.5 * median_height
        )

    @classmethod
    def _median_line_height(cls, clusters: List[Cluster]) -> Optional[float]:
        """Median height of the text lines of the page, drop caps aside."""
        heights = [
            cell.rect.to_bounding_box().height
            for cluster in clusters
            if cluster.label in _DROP_CAP_TEXT_LABELS
            for cell in cluster.cells
            if len(cell.text.strip()) > 1
        ]
        if not heights:
            return None
        return float(np.median(heights))

    @classmethod
    def _merge_drop_cap_clusters(
        cls, clusters: List[Cluster], median_height: float
    ) -> List[Cluster]:
        """Fold clusters that hold only a drop cap into the paragraph they open.

        Layout analysis can return a drop cap as a cluster of its own. Its cell
        is moved into the cluster holding the first line of the paragraph, whose
        bbox is widened to cover it; the drop-cap cluster is dropped.
        """
        merged_ids: set[int] = set()
        targets: Dict[int, Cluster] = {}
        for cap_cluster in clusters:
            if (
                cap_cluster.label not in _DROP_CAP_TEXT_LABELS
                or len(cap_cluster.cells) != 1
                or not cls._is_drop_cap(cap_cluster.cells[0], median_height)
            ):
                continue
            drop_cap = cap_cluster.cells[0]
            for cluster in clusters:
                if (
                    cluster.id == cap_cluster.id
                    or cluster.id in merged_ids
                    or cluster.label not in _DROP_CAP_TEXT_LABELS
                ):
                    continue
                target = targets.get(cluster.id, cluster)
                if any(
                    cls._opens_after_drop_cap(drop_cap, cell, median_height)
                    for cell in target.cells
                ):
                    targets[cluster.id] = target.model_copy(
                        update={
                            "cells": [drop_cap, *target.cells],
                            "bbox": BoundingBox.enclosing_bbox(
                                [target.bbox, cap_cluster.bbox]
                            ),
                        }
                    )
                    merged_ids.add(cap_cluster.id)
                    break
        return [
            targets.get(cluster.id, cluster)
            for cluster in clusters
            if cluster.id not in merged_ids
        ]

    @classmethod
    def _cluster_text_lines(
        cls, cluster: Cluster, median_height: Optional[float]
    ) -> List[str]:
        """Return the text lines of a cluster, each drop cap joined to its word.

        A drop cap cell is prepended to the first line of the paragraph it
        opens ("R" / "ealized" gives "Realized"), instead of being a line of
        its own that would be joined with a space.
        """
        if median_height is None or cluster.label not in _DROP_CAP_TEXT_LABELS:
            return cell_text_lines(cluster.cells)

        cells = list(cluster.cells)
        for drop_cap in [c for c in cells if cls._is_drop_cap(c, median_height)]:
            for ix, cell in enumerate(cells):
                if cell is not drop_cap and cls._opens_after_drop_cap(
                    drop_cap, cell, median_height
                ):
                    cells[ix] = cell.model_copy(
                        update={"text": drop_cap.text.strip() + cell.text.strip()}
                    )
                    cells = [c for c in cells if c is not drop_cap]
                    break
        return cell_text_lines(cells)

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
                    form_fields = page.predictions.form_fields or FormFieldPrediction()

                    clusters = page.predictions.layout.clusters
                    median_height = self._median_line_height(clusters)
                    if median_height is not None:
                        clusters = self._merge_drop_cap_clusters(
                            clusters, median_height
                        )

                    for cluster in clusters:
                        # _log.info("Cluster label seen:", cluster.label)
                        if cluster.label in TEXT_ELEM_LABELS:
                            text = self.sanitize_text(
                                self._cluster_text_lines(cluster, median_height)
                            )
                            hyperlink = self._match_hyperlink(cluster.bbox, page)
                            # A paragraph that inlines AcroForm widgets carries a
                            # single keyed field item; attach it so it materializes
                            # as a field_item in place (see TextElement.field_item)
                            # instead of being pulled into a separate field_region.
                            field_item = form_fields.inline_items.get(cluster.id)
                            text_el = TextElement(
                                label=cluster.label,
                                id=cluster.id,
                                text=text,
                                hyperlink=hyperlink,
                                page_no=page.page_no,
                                cluster=cluster,
                                field_item=field_item,
                            )
                            elements.append(text_el)

                            if cluster.label in PAGE_HEADER_LABELS:
                                headers.append(text_el)
                            else:
                                body.append(text_el)
                        elif cluster.label in TABLE_LABELS:
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
                        elif cluster.label == FIGURE_LABEL:
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
                        elif cluster.label in CONTAINER_LABELS:
                            field_items = form_fields.form_items.get(cluster.id)
                            if field_items is None:
                                container_el = ContainerElement(
                                    label=cluster.label,
                                    id=cluster.id,
                                    page_no=page.page_no,
                                    cluster=cluster,
                                )
                            else:
                                container_el = FieldRegionElement(
                                    label=DocItemLabel.FIELD_REGION,
                                    id=cluster.id,
                                    page_no=page.page_no,
                                    cluster=cluster,
                                    items=field_items,
                                )
                            elements.append(container_el)
                            body.append(container_el)

                    page.assembled = AssembledUnit(
                        elements=elements, headers=headers, body=body
                    )

                yield page
