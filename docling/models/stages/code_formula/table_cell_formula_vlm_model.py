# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Formula recognition over PDF table cells.

The stage takes the ``TableItem`` that ``iterate_items()`` already yields and descends into
``data.table_cells`` itself; ``BaseItemAndImageEnrichmentModel`` cannot be reused because a
``TableCell`` has a ``bbox`` but no ``prov``.

It mutates the document mid-iteration, which is safe only because ``is_processable`` accepts
nothing but a ``TableItem``: the group it appends to ``table.children`` is visited when the
walk resumes, so widening that gate would cause re-enrichment or non-termination.
"""

import logging
import re
from collections.abc import Iterable
from typing import ClassVar, NamedTuple, Optional

from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DoclingDocument,
    GroupLabel,
    NodeItem,
    ProvenanceItem,
    RichTableCell,
    Size,
    TableCell,
    TableItem,
)
from PIL import Image
from pydantic import BaseModel, ConfigDict

from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.models.base_model import GenericEnrichmentModel
from docling.models.inference_engines.vlm import VlmEngineInput
from docling.models.stages.code_formula.code_formula_vlm_model import (
    CodeFormulaVlmModel,
)
from docling.utils.utils import chunkify

_log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- pre-filter

# PDFs typesetting maths with Symbol-style fonts emit PUA codepoints for glyph pieces (the
# segments of a large brace, say). Prose essentially never contains them, so this is the
# highest-precision signal there is -- and the only one for such a cell.
_PUA_RE = re.compile("[\ue000-\uf8ff\U000f0000-\U000ffffd\U00100000-\U0010fffd]")

# Unicode blocks that are unambiguously mathematical notation. Degree and micro signs, the
# dash block and the bullet blocks are excluded: they mark units and ranges, not maths.
_MATH_CHAR_RE = re.compile(
    "["
    "\u00b1\u00d7\u00f7"  # plus-minus, multiplication, division
    "\u0370-\u03ff"  # Greek and Coptic (sigma, mu, alpha, ...)
    "\u1f00-\u1fff"  # Greek Extended
    "\u2070-\u209f"  # Superscripts and Subscripts
    "\u2100-\u214f"  # Letterlike Symbols
    "\u2150-\u218f"  # Number Forms (vulgar fractions)
    "\u2190-\u21ff"  # Arrows
    "\u2200-\u22ff"  # Mathematical Operators (sum, integral, <=, ~=, sqrt)
    "\u2300-\u23ff"  # Miscellaneous Technical (big brackets and braces)
    "\u27c0-\u27ef"  # Miscellaneous Mathematical Symbols-A
    "\u27f0-\u27ff"  # Supplemental Arrows-A
    "\u2900-\u297f"  # Supplemental Arrows-B
    "\u2980-\u29ff"  # Miscellaneous Mathematical Symbols-B
    "\u2a00-\u2aff"  # Supplemental Mathematical Operators
    "\ufe35-\ufe44"  # vertical bracket presentation forms
    "\U0001d400-\U0001d7ff"  # Mathematical Alphanumeric Symbols
    "]"
)

# Content that is already LaTeX, or TeX-flavoured.
_LATEX_RE = re.compile(r"\$[^$]+\$|\\\(|\\\[|\\[A-Za-z]{2,}")

# Sub/superscript notation that survived text extraction, e.g. "a^2", "sigma_SF".
_SCRIPT_RE = re.compile(r"[\w)\]}]\s*\^|_\s*[\w({]")

# Named functions and arithmetic adjacent to a digit. ASCII '-' is excluded as ambiguous with
# hyphenation; U+2212 MINUS is already covered above.
_EXPRESSION_RE = re.compile(
    r"\b(?:log|ln|exp|sin|cos|tan|sqrt|min|max|mod)\s*[_(\d]"
    r"|\d\s*[+*/]\s*[\w(.]"
    r"|[\w).]\s*[+*/]\s*\d"
)

_RELATION_RE = re.compile(r"[=<>]")
_WORD_RE = re.compile(r"\w")

# Anchored, so it vetoes only a cell that is *entirely* numeric. The garbled "4 SF = <sigma>"
# that TR 38.901 emits for sigma_SF = 4 has more structure, and reaches the positive tests.
_PLAIN_NUMBER_RE = re.compile(
    r"""^
    [+\-\u2212]?\d+(?:[.,]\d+)?(?:\s*[eE][+\-]?\d+)?
    (?:\s*[-\u2013\u2014\u2212/]\s*[+\-\u2212]?\d+(?:[.,]\d+)?)?
    (?:\s*(?:%|\u00b0|[A-Za-z][A-Za-z/]{0,4}\d?))?
    $""",
    re.VERBOSE,
)


def cell_text_may_contain_formula(text: Optional[str]) -> bool:
    """Cheap gate deciding whether a table cell is worth a VLM call.

    The layout model emits no ``formula`` clusters inside table regions, so the garbled cell
    text is the only signal available. Tuned for recall over precision, because the feature is
    opt-in: a false positive costs one cheap call, a false negative drops content silently.
    Override :meth:`TableCellFormulaVlmModel._may_contain_formula` to retune.
    """
    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    # First because it is the strongest signal. Ordering against the numeric veto is only
    # defensive: that pattern is ^-anchored over ASCII, so PUA can never reach it.
    if _PUA_RE.search(stripped):
        return True
    if _PLAIN_NUMBER_RE.match(stripped):
        return False
    if _MATH_CHAR_RE.search(stripped):
        return True
    if _LATEX_RE.search(stripped):
        return True
    if _SCRIPT_RE.search(stripped):
        return True
    if _EXPRESSION_RE.search(stripped):
        return True
    return bool(_RELATION_RE.search(stripped) and _WORD_RE.search(stripped))


# --------------------------------------------------------------------------- data model


class TableFormulaEnrichmentElement(BaseModel):
    """A table together with the cell images that may contain formulas.

    ``cell_crops`` keys on the index into ``item.data.table_cells``, not a grid position: a
    spanned cell appears once there but at every position it covers in ``TableData.grid``, and
    must be transcribed once. A repeated index means several ordered crops for one cell.
    """

    # Spelled out because it is load-bearing, though it is also the default: ``item`` must stay
    # the same object as the node in the document, or every mutation here lands on a copy.
    model_config = ConfigDict(
        arbitrary_types_allowed=True, revalidate_instances="never"
    )

    item: TableItem
    cell_crops: list[tuple[int, Image.Image]]


class _CropRequest(NamedTuple):
    table: TableItem
    cell_index: int
    image: Image.Image


# ---------------------------------------------------------------------------- the stage


class TableCellFormulaVlmModel(GenericEnrichmentModel[TableFormulaEnrichmentElement]):
    """Runs formula recognition over PDF table cells, emitting rich table cells."""

    # Aligned with CodeFormulaVlmModel so crops match the model's training data.
    images_scale: ClassVar[float] = 1.67  # = 120 dpi
    expansion_factor: ClassVar[float] = 0.18

    # One table per batch: crops are materialised eagerly, so batching tables only multiplies
    # live PIL images -- `__call__` re-chunks to `vlm_batch_size` anyway.
    elements_batch_size = 1

    #: Crops per VLM forward pass. Matches ``CodeFormulaVlmModel.elements_batch_size``.
    vlm_batch_size: ClassVar[int] = 5

    #: Reject crops thinner than this, in PDF points, on either axis.
    min_crop_size: ClassVar[float] = 1.0

    def __init__(
        self, *, enabled: bool, code_formula_model: CodeFormulaVlmModel
    ) -> None:
        """Initialize the stage.

        Args:
            enabled: Whether ``do_table_cell_formula_enrichment`` was requested.
            code_formula_model: The already-constructed formula stage, whose engine, prompt and
                post-processing are reused -- no second copy of the weights, and cell
                transcriptions stay identical to page-level ones.
        """
        self._code_formula_model = code_formula_model
        self.enabled = bool(
            enabled
            and code_formula_model.engine is not None
            and code_formula_model.options.extract_formulas
        )
        if enabled and not self.enabled:
            _log.warning(
                "`do_table_cell_formula_enrichment` is enabled but formula recognition is "
                "not available: `do_formula_enrichment` must be enabled as well. Table cell "
                "formula enrichment will be skipped."
            )

    # Deliberately no __del__: `code_formula_model` owns the shared engine and already calls
    # engine.cleanup() in its own.

    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        return self.enabled and isinstance(element, TableItem)

    def _may_contain_formula(self, text: Optional[str]) -> bool:
        """Override point for the pre-filter."""
        return cell_text_may_contain_formula(text)

    # ------------------------------------------------------------------------ preparing

    def prepare_element(
        self, conv_res: ConversionResult, element: NodeItem
    ) -> Optional[TableFormulaEnrichmentElement]:
        if not self.is_processable(doc=conv_res.document, element=element):
            return None
        assert isinstance(element, TableItem)

        if not element.prov:
            return None  # no page geometry: nothing to crop

        page = self._find_page(conv_res, element.prov[0].page_no)
        if page is None or page.size is None:
            return None
        if page._backend is None:
            # Without a backend, get_image() ignores `cropbox` and returns the whole page,
            # which would attach a page-sized transcription to one cell.
            _log.debug(
                "No page backend for page %s; skipping table cell formula enrichment for %s.",
                element.prov[0].page_no,
                element.self_ref,
            )
            return None

        crops: list[tuple[int, Image.Image]] = []
        for idx, cell in enumerate(element.data.table_cells):
            if isinstance(cell, RichTableCell):
                continue  # picture cell, DOCX/HTML source, or a re-run: never overwrite a ref
            if cell.bbox is None:
                continue
            if not self._may_contain_formula(cell.text):
                continue
            bbox = self._clamp_to_page(
                cell.bbox.expand_by_scale(self.expansion_factor, self.expansion_factor),
                page.size,
            )
            if bbox is None:
                continue
            image = page.get_image(scale=self.images_scale, cropbox=bbox)
            if image is None or image.width < 1 or image.height < 1:
                continue
            crops.append((idx, image))

        if not crops:
            return None
        return TableFormulaEnrichmentElement(item=element, cell_crops=crops)

    @staticmethod
    def _find_page(conv_res: ConversionResult, page_no: int) -> Optional[Page]:
        # By page number, not `page_no - pages[0].page_no`: that arithmetic raises on an empty
        # page list and picks the wrong page when the list is sparse (page_range, failed pages).
        return next((p for p in conv_res.pages if p.page_no == page_no), None)

    @classmethod
    def _clamp_to_page(
        cls, bbox: BoundingBox, page_size: Size
    ) -> Optional[BoundingBox]:
        """Clamp an expanded cell bbox to the page, or reject it if degenerate.

        Origin-agnostic: ``width``/``height`` are absolute, so clamping each coordinate works
        without knowing the origin.
        """
        clamped = BoundingBox(
            l=max(0.0, min(bbox.l, page_size.width)),
            r=max(0.0, min(bbox.r, page_size.width)),
            t=max(0.0, min(bbox.t, page_size.height)),
            b=max(0.0, min(bbox.b, page_size.height)),
            coord_origin=bbox.coord_origin,
        )
        if clamped.width < cls.min_crop_size or clamped.height < cls.min_crop_size:
            return None
        return clamped

    # ------------------------------------------------------------------------ inference

    def __call__(
        self,
        doc: DoclingDocument,
        element_batch: Iterable[TableFormulaEnrichmentElement],
    ) -> Iterable[NodeItem]:
        elements = list(element_batch)
        if not self.enabled or self._code_formula_model.engine is None:
            yield from (el.item for el in elements)
            return

        # The loop's unit is the table, the engine's is the crop.
        queue = [
            _CropRequest(table=el.item, cell_index=idx, image=image)
            for el in elements
            for idx, image in el.cell_crops
        ]

        texts: list[str] = []
        for sub_batch in chunkify(queue, self.vlm_batch_size):
            texts.extend(self._transcribe(list(sub_batch)))

        # Crop order preserved, so several crops for one cell become ordered children.
        per_cell: dict[tuple[str, int], list[str]] = {}
        for request, text in zip(queue, texts):
            per_cell.setdefault(
                (request.table.self_ref, request.cell_index), []
            ).append(text)

        tables = {el.item.self_ref: el.item for el in elements}
        for (table_ref, cell_index), cell_texts in per_cell.items():
            try:
                self._write_rich_cell(
                    doc=doc,
                    table=tables[table_ref],
                    cell_index=cell_index,
                    texts=cell_texts,
                )
            except Exception:
                # One bad cell must not lose the rest of the table.
                _log.warning(
                    "Failed to attach recognised formula to cell %s of %s",
                    cell_index,
                    table_ref,
                    exc_info=True,
                )

        yield from (el.item for el in elements)

    def _transcribe(self, requests: list[_CropRequest]) -> list[str]:
        engine = self._code_formula_model.engine
        assert engine is not None
        prompt = self._code_formula_model._get_prompt("formula")
        try:
            outputs = engine.predict_batch(
                [
                    VlmEngineInput(
                        image=request.image,
                        prompt=prompt,
                        temperature=0.0,
                        max_new_tokens=2048,
                        extra_generation_config={
                            # Keep special tokens, so _post_process can strip them.
                            "skip_special_tokens": False,
                        },
                    )
                    for request in requests
                ]
            )
        except Exception as exc:
            # Mirrors CodeFormulaVlmModel: enrichment degrades, it never fails a conversion.
            _log.error(f"Error processing table cell formula batch: {exc}")
            return [""] * len(requests)
        return self._code_formula_model._post_process([out.text for out in outputs])

    # ---------------------------------------------------------------- rich-cell writing

    def _write_rich_cell(
        self,
        *,
        doc: DoclingDocument,
        table: TableItem,
        cell_index: int,
        texts: list[str],
    ) -> None:
        latex = [text.strip() for text in texts if text and text.strip()]
        if not latex:
            return  # nothing recognised: no empty group that would serialize to nothing

        cell = table.data.table_cells[cell_index]
        if isinstance(cell, RichTableCell):
            return  # became rich in the meantime; never overwrite an existing ref

        content_layer = table.content_layer
        # `add_group(parent=table)` sets group.parent AND appends to table.children, the two
        # halves validate_tree requires of a rich-cell ref. Not `add_table_cell`: that appends.
        group = doc.add_group(
            label=GroupLabel.UNSPECIFIED,
            # This table's index, not len(doc.tables) -- that is the table *count*, so every
            # table would name its (col, row) groups identically.
            name=(
                f"rich_cell_group_{table.self_ref.rsplit('/', 1)[-1]}_"
                f"{cell.start_col_offset_idx}_{cell.start_row_offset_idx}"
            ),
            parent=table,
            content_layer=content_layer,
        )
        bbox = self._cell_prov_bbox(doc=doc, table=table, cell=cell)
        page_no = table.prov[0].page_no if table.prov else None
        for text in latex:
            prov = (
                ProvenanceItem(page_no=page_no, bbox=bbox, charspan=(0, len(text)))
                if bbox is not None and page_no is not None
                else None
            )
            doc.add_formula(
                text=text,
                orig=cell.text or text,  # garbled source text, kept for auditability
                prov=prov,
                parent=group,
                content_layer=content_layer,
            )

        # Mirrors the reading-order stage's rich-cell replacement; `exclude` keeps it
        # re-entrant, and the spread carries every present and future TableCell field.
        table.data.table_cells[cell_index] = RichTableCell(
            **cell.model_dump(exclude={"ref"}),
            ref=group.get_ref(),
        )

    @staticmethod
    def _cell_prov_bbox(
        *, doc: DoclingDocument, table: TableItem, cell: TableCell
    ) -> Optional[BoundingBox]:
        """The cell bbox in whatever coordinate origin the table's provenance uses.

        Derived rather than assumed: the two table-structure paths disagree, one rescaling cell
        bboxes without converting the origin and the other preserving it.
        """
        if not table.prov or cell.bbox is None:
            return None
        page = doc.pages.get(table.prov[0].page_no)
        if page is None or page.size is None:
            return None
        target = table.prov[0].bbox.coord_origin
        if cell.bbox.coord_origin == target:
            return cell.bbox
        if target == CoordOrigin.BOTTOMLEFT:
            return cell.bbox.to_bottom_left_origin(page_height=page.size.height)
        return cell.bbox.to_top_left_origin(page_height=page.size.height)
