# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Mark superscript and subscript text in documents converted from PDF.

A text item holding such text keeps its place, and its text moves into an inline group of
runs with ``formatting.script``, which the item holds.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
)
from docling_core.types.doc.document import (
    Formatting,
    InlineGroup,
    ListItem,
    NodeItem,
    Script,
    TextItem,
)

from docling.datamodel.backend_options import PdfBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import ScriptPositionOptions
from docling.utils.pdf_char_geometry import (
    PdfChar,
    PdfPageChars,
    open_pdf,
    pypdfium2_available,
    read_page_chars,
)
from docling.utils.profiling import ProfilingScope, TimeRecorder
from docling.utils.script_position import (
    ScriptRun,
    script_positions,
    script_runs,
)

_log = logging.getLogger(__name__)

# List items are handled too: they keep their item and get the runs as a child.
_TEXT_LABELS = frozenset(
    {DocItemLabel.TEXT, DocItemLabel.PARAGRAPH, DocItemLabel.FOOTNOTE}
)
# Slack (points) when testing whether a character's centre lies in an item's box.
_BOX_TOLERANCE = 0.5


class _Page:
    """A page's characters, with their centres in Docling's page frame."""

    def __init__(self, page: PdfPageChars):
        self.page = page
        self.chars = page.chars
        boxes = [self._display_box(char) for char in page.chars]
        self.centres = np.array(
            [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes], dtype=float
        ).reshape(-1, 2)

    def indexes_in(self, box: BoundingBox) -> list[int]:
        """Return the characters whose centre is in a bottom-left-origin box."""
        x, y = self.centres[:, 0], self.centres[:, 1]
        inside = (
            (x >= box.l - _BOX_TOLERANCE)
            & (x <= box.r + _BOX_TOLERANCE)
            & (y >= box.b - _BOX_TOLERANCE)
            & (y <= box.t + _BOX_TOLERANCE)
        )
        return np.flatnonzero(inside).tolist()

    def box_of(self, indexes: list[int]) -> BoundingBox:
        boxes = [self._display_box(self.chars[i]) for i in indexes]
        return BoundingBox(
            l=min(b[0] for b in boxes),
            b=min(b[1] for b in boxes),
            r=max(b[2] for b in boxes),
            t=max(b[3] for b in boxes),
            coord_origin=CoordOrigin.BOTTOMLEFT,
        )

    def _display_box(self, char: PdfChar) -> tuple[float, float, float, float]:
        return self.page.to_display_frame(
            (char.left, char.bottom, char.right, char.top)
        )


# Returns the characters of a 1-based page, or None when they cannot be read.
PageLookup = Callable[[int], "_Page | None"]


@dataclass
class _Rewrite:
    """The runs computed for one text item, applied once every item is computed."""

    item: TextItem
    runs: list[ScriptRun]
    # The provenance of each run, one per page it is on.
    provs: list[list[ProvenanceItem]]


class ScriptPositionModel:
    def __init__(self, options: ScriptPositionOptions):
        self.options = options

    def __call__(self, conv_res: ConversionResult) -> DoclingDocument:
        document = conv_res.document
        backend = conv_res.input._backend
        # The PDF pipeline also converts images, which have no PDF characters to read.
        if (
            not self.options.enabled
            or conv_res.input.format != InputFormat.PDF
            or backend.path_or_stream is None
        ):
            return document
        if not pypdfium2_available():
            _log.warning(
                "Superscript and subscript recovery needs pypdfium2, which is not "
                "installed; skipping it."
            )
            return document
        password = (
            backend.options.password
            if isinstance(backend.options, PdfBackendOptions)
            else None
        )

        with TimeRecorder(conv_res, "script_position", scope=ProfilingScope.DOCUMENT):
            # The step is optional: a failure keeps the document as it is.
            try:
                with open_pdf(
                    backend.path_or_stream,
                    password.get_secret_value() if password else None,
                ) as pdf:
                    rewrites = self.find_scripts(document, _PageCache(pdf).get)
            except Exception as exc:
                _log.warning(f"Superscript and subscript recovery failed: {exc}")
                return document
            self.apply_scripts(document, rewrites)
        return document

    def apply_scripts(
        self, document: DoclingDocument, rewrites: list[_Rewrite]
    ) -> None:
        """Move the text of each rewritten item into an inline group of its runs."""
        for rewrite in rewrites:
            _apply(document, rewrite)

    def find_scripts(
        self, document: DoclingDocument, page_for: PageLookup
    ) -> list[_Rewrite]:
        """Compute the runs of every text item that holds superscript or subscript text."""
        referenced = {
            ref.cref
            for item in [*document.tables, *document.pictures]
            for ref in [*item.captions, *item.footnotes, *item.references]
        }
        rewrites = []
        for item, _ in document.iterate_items(with_groups=False):
            if isinstance(item, TextItem) and _is_candidate(document, item, referenced):
                rewrite = _rewrite_for(document, item, page_for)
                if rewrite is not None:
                    rewrites.append(rewrite)
        return rewrites


class _PageCache:
    """Read each page's characters once, when an item on it is first looked at."""

    def __init__(self, pdf):
        self.pdf = pdf
        self.pages: dict[int, _Page | None] = {}

    def get(self, page_no: int) -> _Page | None:
        if page_no not in self.pages:
            # PDFium reports a page it cannot read only by raising.
            try:
                self.pages[page_no] = _Page(read_page_chars(self.pdf, page_no - 1))
            except Exception as exc:
                _log.debug(f"Cannot read characters of page {page_no}: {exc}")
                self.pages[page_no] = None
        return self.pages[page_no]


def _is_candidate(
    document: DoclingDocument, item: TextItem, referenced: set[str]
) -> bool:
    if not item.prov or not item.text:
        return False
    # Captions and footnotes of tables and pictures are referenced from them.
    if item.self_ref in referenced:
        return False
    if item.parent is not None and isinstance(
        item.parent.resolve(document), InlineGroup
    ):
        return False
    return isinstance(item, ListItem) or item.label in _TEXT_LABELS


def _rewrite_for(
    document: DoclingDocument, item: TextItem, page_for: PageLookup
) -> _Rewrite | None:
    chars, owners = _item_chars(document, item, page_for)
    # Cheap check first: matching the text to the characters costs more.
    if all(p == Script.BASELINE for p in script_positions(chars)):
        return None
    runs = script_runs(item.text, chars)
    if runs is None:
        return None
    return _Rewrite(item, runs, [_run_provs(item, r, owners, page_for) for r in runs])


def _item_chars(
    document: DoclingDocument, item: TextItem, page_for: PageLookup
) -> tuple[list[PdfChar], list[tuple[int, int]]]:
    """Return the PDF characters in the item's boxes, with (page, index) of each."""
    chars: list[PdfChar] = []
    owners: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for prov in item.prov:
        page = page_for(prov.page_no)
        if page is None or prov.page_no not in document.pages:
            continue
        box = prov.bbox.to_bottom_left_origin(document.pages[prov.page_no].size.height)
        for index in page.indexes_in(box):
            if (prov.page_no, index) not in seen:
                seen.add((prov.page_no, index))
                chars.append(page.chars[index])
                owners.append((prov.page_no, index))
    return chars, owners


def _run_provs(
    item: TextItem, run: ScriptRun, owners: list[tuple[int, int]], page_for: PageLookup
) -> list[ProvenanceItem]:
    """Box the run's characters on each page they are on, else use the item's first box.

    A run can cross a page break, as the text after a script at the end of a page.
    """
    # Per page, the position in the run's text and the index of each character.
    by_page: dict[int, list[tuple[int, int]]] = {}
    for k, offset in zip(run.chars, run.offsets, strict=True):
        page_no, index = owners[k]
        by_page.setdefault(page_no, []).append((offset, index))
    pages = sorted(by_page, key=lambda p: min(o for o, _ in by_page[p]))
    provs = []
    for n, page_no in enumerate(pages):
        page = page_for(page_no)
        if page is None:
            continue
        start = 0 if n == 0 else min(o for o, _ in by_page[page_no])
        end = (
            len(run.text)
            if n == len(pages) - 1
            else min(o for o, _ in by_page[pages[n + 1]])
        )
        provs.append(
            ProvenanceItem(
                page_no=page_no,
                bbox=page.box_of([i for _, i in by_page[page_no]]),
                charspan=(start, end),
            )
        )
    if provs:
        return provs
    return [
        ProvenanceItem(
            page_no=item.prov[0].page_no,
            bbox=item.prov[0].bbox,
            charspan=(0, len(run.text)),
        )
    ]


def _apply(document: DoclingDocument, rewrite: _Rewrite) -> None:
    """Move an item's text into an inline group of its runs, held by the item.

    The item keeps its place, so a paragraph stays a paragraph block in the export and
    a list item keeps its marker, as in the Word backend.
    """
    item = rewrite.item
    if item.children:
        group = document.insert_inline_group(
            sibling=item.children[0].resolve(document),
            content_layer=item.content_layer,
            after=False,
        )
    else:
        group = document.add_inline_group(parent=item, content_layer=item.content_layer)
    label = DocItemLabel.TEXT if isinstance(item, ListItem) else item.label

    for run, provs in zip(rewrite.runs, rewrite.provs, strict=True):
        formatting = None
        if run.script != Script.BASELINE or item.formatting is not None:
            formatting = (item.formatting or Formatting()).model_copy(
                update={"script": run.script}
            )
        text_item = document.add_text(
            label=label,
            text=run.text,
            orig=run.text,
            prov=provs[0],
            parent=group,
            formatting=formatting,
            hyperlink=item.hyperlink,
            content_layer=item.content_layer,
        )
        text_item.prov.extend(provs[1:])

    item.text = ""
    item.orig = ""
