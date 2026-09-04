# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Formula recognition over PDF table cells.

``TableItem`` is yielded by ``DoclingDocument.iterate_items()``, so this stage receives the
table and descends into ``data.table_cells`` itself. No change to
``BasePipeline._enrich_document`` and no widening of the enrichment contract is required.

Cell crops cannot reuse ``BaseItemAndImageEnrichmentModel``: a ``TableCell`` carries a ``bbox``
but no ``prov``, and is not a ``DocItem``.

Why this exists: in technical standards, normative maths frequently lives *inside* a table
(3GPP TR 38.901 Table 7.4.1-1 is the canonical case). The PDF/TableFormer backend emits those
cells as plain text in visual glyph order, so an equation arrives scrambled -- and partly as
Private Use Area font glyphs that carry no Unicode meaning at all, which is why a consumer
cannot reconstruct it afterwards from the text. See docling#3828.

Mutating the document while the enrichment loop iterates is safe here, but only narrowly so.
``_enrich_document`` walks a live generator, and ``_iterate_items_with_stack`` yields a node
*before* walking its children, so the group this stage appends to ``table.children`` -- and the
``FormulaItem``s under it -- are visited when iteration resumes. That is harmless only because
:meth:`TableCellFormulaVlmModel.is_processable` rejects everything that is not a ``TableItem``.
Widening that gate would cause re-enrichment, or non-termination.
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

# Private Use Area, including the two supplementary PUA planes. PDFs that typeset maths with
# Symbol/Wingdings-style fonts emit PUA codepoints for the glyph pieces -- for example the
# segments of a large piecewise brace. Ordinary prose essentially never contains them, so this
# is both the highest-precision signal available from garbled cell text and the *only* signal
# for cells whose text is otherwise unrecognisable.
_PUA_RE = re.compile("[\ue000-\uf8ff\U000f0000-\U000ffffd\U00100000-\U0010fffd]")

# Unicode blocks that are unambiguously mathematical notation.
#
# Deliberately excluded, because they mark units and ranges rather than maths: U+00B0 DEGREE
# SIGN and U+00B5 MICRO SIGN ("30°", "µs"), the dash block U+2010-U+2015 (numeric ranges), and
# Geometric Shapes / Dingbats (list bullets).
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

# Named functions and arithmetic adjacent to a digit. ASCII '-' is excluded on purpose: it is
# ambiguous with hyphenation and numeric ranges, and U+2212 MINUS SIGN is already covered by
# _MATH_CHAR_RE.
_EXPRESSION_RE = re.compile(
    r"\b(?:log|ln|exp|sin|cos|tan|sqrt|min|max|mod)\s*[_(\d]"
    r"|\d\s*[+*/]\s*[\w(.]"
    r"|[\w).]\s*[+*/]\s*\d"
)

_RELATION_RE = re.compile(r"[=<>]")
_WORD_RE = re.compile(r"\w")

# Anchored, so it vetoes only a cell that is *entirely* a number, a numeric range or ratio, a
# percentage, or a number with a unit. Anything with more structure -- notably the garbled
# "4 SF = <sigma>" form that TR 38.901 emits for sigma_SF = 4 -- reaches the positive tests.
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

    The layout model labels a maths-heavy table as a single ``table`` cluster and emits *zero*
    ``formula`` clusters inside it (measured on 3GPP TR 38.901 Table 7.4.1-1, PDF page 28), so
    cluster-overlap detection is not available and the already-garbled ``TableCell.text`` is the
    only signal there is.

    This predicate is deliberately tuned for **recall over precision**. The feature is opt-in,
    so a false positive costs one extra cheap call that yields a correct if redundant formula
    node, whereas a false negative silently drops normative content -- which is the defect the
    stage exists to fix. Override :meth:`TableCellFormulaVlmModel._may_contain_formula` to
    retune it.
    """
    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    # Checked first because it is the strongest signal available. The ordering against the
    # numeric veto below is defensive rather than load-bearing: that pattern is ^-anchored and
    # its unit class is ASCII letters, so it can never match a string containing a PUA glyph.
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

    ``cell_crops`` holds ``(index into item.data.table_cells, crop)`` pairs. Keying on the cell
    *index* rather than on a grid position is load-bearing: a spanned cell appears once in
    ``table_cells`` but at every position it covers in ``TableData.grid``, and must be
    transcribed once. The same index may legitimately appear more than once, in which case the
    crops stay ordered and the cell becomes a group with that many ordered children.
    """

    # ``revalidate_instances="never"`` is pydantic's default and is spelled out because it is
    # load-bearing: ``item`` must remain the *same object* as the node in the document, or every
    # mutation this stage makes would land on a copy and be silently lost.
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

    # The enrichment loop batches *tables*, and one table can contribute dozens of crops which
    # `prepare_element` materialises eagerly. Batching more than one table therefore only
    # multiplies the number of live PIL images, with no inference benefit -- `__call__`
    # re-chunks to `vlm_batch_size` regardless. Holding this at 1 also keeps the
    # mutate-during-iteration window uniform: `__call__` always runs while `iterate_items` is
    # parked at this table's own yield point.
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
                post-processing are reused. Sharing it means no second copy of the weights, and
                it keeps a formula transcribed inside a cell byte-identical to the same formula
                transcribed as a page-level ``FormulaItem``.
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

    # NOTE: deliberately no __del__. The engine belongs to `code_formula_model`, whose own
    # __del__ calls engine.cleanup(); a second one here would clean up a shared engine twice.

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
            # Page.get_image() ignores `cropbox` when the backend is gone and returns the whole
            # page from its image cache. Feeding a full page to the formula model would attach
            # a page-sized transcription to one cell, so refuse instead. `keep_backend` should
            # prevent this being reached.
            _log.debug(
                "No page backend for page %s; skipping table cell formula enrichment for %s.",
                element.prov[0].page_no,
                element.self_ref,
            )
            return None

        crops: list[tuple[int, Image.Image]] = []
        for idx, cell in enumerate(element.data.table_cells):
            if isinstance(cell, RichTableCell):
                # Already rich: a cell holding a picture (see the reading-order stage), a
                # DOCX/HTML source, or a re-run. Never overwrite an existing ref.
                continue
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
        # Looked up by page number rather than `page_no - conv_res.pages[0].page_no`: that
        # arithmetic raises IndexError on an empty page list, and silently selects the wrong
        # page when the list is sparse or unsorted (page_range, failed pages).
        return next((p for p in conv_res.pages if p.page_no == page_no), None)

    @classmethod
    def _clamp_to_page(
        cls, bbox: BoundingBox, page_size: Size
    ) -> Optional[BoundingBox]:
        """Clamp an expanded cell bbox to the page, or reject it if degenerate.

        Origin-agnostic: ``expand_by_scale`` grows the box correctly for both origins, and
        ``BoundingBox.width``/``height`` are absolute, so clamping each coordinate into the page
        works without knowing which origin is in use.
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

        # Flatten every crop in the batch into one inference queue: the enrichment loop's unit
        # is the table, but the engine's is the crop.
        queue = [
            _CropRequest(table=el.item, cell_index=idx, image=image)
            for el in elements
            for idx, image in el.cell_crops
        ]

        texts: list[str] = []
        for sub_batch in chunkify(queue, self.vlm_batch_size):
            texts.extend(self._transcribe(list(sub_batch)))

        # Regrouped per (table, cell) with crop order preserved, so a cell that produced several
        # crops becomes a group with several ordered children.
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
                # One bad cell must not lose the rest of the table, let alone fail the
                # conversion.
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
            # Mirrors CodeFormulaVlmModel: an engine failure degrades to "no enrichment", it
            # never fails the conversion.
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
            # Nothing recognised. Leave the plain cell alone rather than leaving behind an empty
            # group that serializes to nothing.
            return

        cell = table.data.table_cells[cell_index]
        if isinstance(cell, RichTableCell):
            return  # became rich in the meantime; never overwrite an existing ref

        content_layer = table.content_layer
        # `add_group(parent=table)` sets group.parent AND appends the group to table.children,
        # which are the two halves of what DoclingDocument.validate_tree requires of every
        # RichTableCell ref. `add_table_cell` is deliberately not used: it appends a cell,
        # whereas an existing plain cell has to be replaced in place.
        group = doc.add_group(
            label=GroupLabel.UNSPECIFIED,
            # Named after THIS table's index rather than len(doc.tables): the latter is the
            # table *count*, so every table in a document would name its (col, row) groups
            # identically. Group names are not identity in a DoclingDocument, so that is not
            # corrupting, but it makes the names useless for telling two tables apart.
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
                # The garbled source glyph text, kept so the transcription stays auditable
                # against what was actually on the page.
                orig=cell.text or text,
                prov=prov,
                parent=group,
                content_layer=content_layer,
            )

        # `exclude={"ref"}` mirrors the reading-order stage's rich-cell replacement and keeps
        # this re-entrant. Spreading model_dump() carries every present and future TableCell
        # field without hand-copying a dozen of them.
        table.data.table_cells[cell_index] = RichTableCell(
            **cell.model_dump(exclude={"ref"}),
            ref=group.get_ref(),
        )

    @staticmethod
    def _cell_prov_bbox(
        *, doc: DoclingDocument, table: TableItem, cell: TableCell
    ) -> Optional[BoundingBox]:
        """The cell bbox in whatever coordinate origin the table's provenance uses.

        Derived from the table's prov rather than assumed, because the two table-structure code
        paths do not agree: one rescales cell bboxes without converting the origin, the other
        reconstructs them preserving it.
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
