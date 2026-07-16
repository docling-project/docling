"""Ordered/unordered list-item inference for the PDF/image pipeline.

The layout model only flags a region as ``LIST_ITEM``; it carries no notion of whether the item
belongs to a numbered (ordered) or a bulleted (unordered) list. That distinction has to be
recovered from the item text. At item-creation time the reading-order stage already runs
``docling_ibm_models``' :class:`ListItemMarkerProcessor`, which strips *simple* leading markers
(``1.``, ``a)``, ``[3]``, ``•`` ...) into ``ListItem.marker`` and sets ``ListItem.enumerated``.

**Compound / hierarchical markers** such as ``9a.``, ``3.a.`` or ``1.2.1`` are not in the marker
processor's pattern set. The marker stays fused inside ``text`` and ``marker`` is left empty.
When such an item sits inside an otherwise-enumerated list, the Markdown serializer sees the
empty marker and prepends a *position-based* number of its own, yielding a doubled, wrong marker
(e.g. ``7. 9a. Compute ...`` instead of ``9a. Compute ...``).

:class:`ListNormalizationModel` runs right after the reading-order model and, per list group,
recovers those compound ordered markers into ``marker`` (removing them from ``text`` and setting
``enumerated``). Recovery is applied only in a numbered context, so stray dotted-decimal
*content* (``1.1 million ...``) in a bullet list is left untouched.

It never adds, removes or reorders items, and only rewrites items whose marker was previously
missed -- items the marker processor already handled are untouched. The core
(:meth:`ListNormalizationModel.normalize`) works on a bare ``DoclingDocument`` so it can be
reused outside the pipeline.
"""

import logging
import re

from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import ListGroup, ListItem

from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import ListNormalizationOptions

_log = logging.getLogger(__name__)

# Leading compound/hierarchical ordered marker that the simple marker processor misses:
#   dotted decimal   -> 1.1  1.1.1  (1.2)
#   digit + letter   -> 9a  9a.  9a)  3.a  3.a.
# followed by whitespace and the item content. Simple markers (``1.``, ``a)`` ...) are
# intentionally excluded here -- they are already handled at item-creation time.
_COMPOUND_ORDERED = re.compile(
    r"^\s*(?P<marker>\(?\d+(?:\.\d+)+[.)\]]?|\(?\d+\.?[A-Za-z][.)\]]?)\s+(?P<content>\S.*)$",
    re.DOTALL,
)

# A short alphabetic/roman ordinal token with a delimiter, e.g. ``b.`` ``iv)`` ``(A)``.
_ALPHA_ORDINAL = re.compile(r"^\(?[A-Za-z]{1,4}[.)\]]$|^\([A-Za-z]{1,4}\)$")


def _marker_is_numbered(marker: str) -> bool:
    """Whether ``marker`` denotes an ordered position (digit, or alpha/roman ordinal)."""
    m = marker.strip()
    if not m:
        return False
    if any(ch.isdigit() for ch in m):
        return True
    return _ALPHA_ORDINAL.match(m) is not None


class ListNormalizationModel:
    """Infer ordered vs. unordered list items after reading-order assembly.

    Enabled via :class:`ListNormalizationOptions`. When disabled (the default) the input
    document is returned unchanged.
    """

    def __init__(self, options: ListNormalizationOptions):
        self.options = options

    def __call__(self, conv_res: ConversionResult) -> DoclingDocument:
        document = conv_res.document
        if not self.options.enabled:
            return document
        try:
            return self.normalize(document)
        except Exception:  # never let list normalization break a conversion
            _log.warning("List normalization failed; leaving list items unchanged.")
            return document

    def normalize(self, document: DoclingDocument) -> DoclingDocument:
        """Normalize every list group in ``document`` in place and return it."""
        for item, _level in document.iterate_items(with_groups=True):
            if isinstance(item, ListGroup):
                self._normalize_group(item, document)
        return document

    def _normalize_group(self, group: ListGroup, document: DoclingDocument) -> None:
        items = [
            child
            for ref in group.children
            if isinstance((child := ref.resolve(document)), ListItem)
        ]
        if not items:
            return

        self._recover_ordered_markers(items)

    @staticmethod
    def _recover_ordered_markers(items: list[ListItem]) -> None:
        """Extract compound ordered markers left fused in ``text`` into ``marker``.

        Applied only when the group already has a numbered context -- either an item with a
        recognized numbered marker, or at least two items that themselves look compound-ordered
        -- so that a lone dotted-decimal value in prose is not mistaken for a marker.
        """
        candidates: dict[int, re.Match[str]] = {}
        for it in items:
            if it.marker == "":
                match = _COMPOUND_ORDERED.match(it.text)
                if match is not None:
                    candidates[id(it)] = match
        if not candidates:
            return

        has_numbered_context = (
            any(_marker_is_numbered(it.marker) for it in items) or len(candidates) >= 2
        )
        if not has_numbered_context:
            return

        for it in items:
            match = candidates.get(id(it))
            if match is not None:
                it.marker = match.group("marker")
                it.text = match.group("content")
                it.enumerated = True
