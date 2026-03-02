import logging
import re
from collections.abc import Iterable
from typing import Dict, List

import numpy as np
from pydantic import BaseModel

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
    def __init__(self, options: PageAssembleOptions):
        self.options = options

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

                    for cluster in page.predictions.layout.clusters:
                        # _log.info("Cluster label seen:", cluster.label)
                        if cluster.label in LayoutModel.TEXT_ELEM_LABELS:
                            textlines = [
                                cell.text.replace("\x02", "-").strip()
                                for cell in cluster.cells
                                if len(cell.text.strip()) > 0
                            ]
                            text = self.sanitize_text(textlines)
                            text_el = TextElement(
                                label=cluster.label,
                                id=cluster.id,
                                text=text,
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
