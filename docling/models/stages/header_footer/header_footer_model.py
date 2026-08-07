"""Recovery of page headers/footers that are actually body content.

The layout model classifies regions as ``PAGE_HEADER``/``PAGE_FOOTER`` from per-page position
only -- it has no notion of whether the text repeats across pages. A one-off heading sitting at
the top of a page therefore gets tagged as a page header, moved to the ``FURNITURE`` content
layer by the reading-order model and silently dropped from the default export (which only emits
``BODY``). See issue #3693.

:class:`HeaderFooterModel` runs just *before* the reading-order model -- the first stage with
whole-document context -- and demotes header/footer detections that are not genuine running
furniture back into the body, relabeling the assembled elements in place so the reading-order
and heading-hierarchy stages then treat them as ordinary content.

A detection is left untouched (kept as furniture) when **either**:

1. **repetition** -- its normalized text (digits stripped, so ``"Page 3 of 10"`` matches
   ``"Page 4 of 10"``) appears on at least ``min_repeat_pages`` pages, i.e. a consistent
   running header/footer; or
2. **coverage** -- the side it belongs to (header vs footer) is present on at least
   ``min_coverage`` of the pages. A real running header/footer appears on (almost) every page,
   so a side that only shows up on a few scattered pages is most likely mis-detected content
   (issue #3693 has no genuine headers, only sparse mis-detections), whereas a header whose
   text legitimately changes per section is still trusted because it covers most pages.

Otherwise the detection is reclassified: a non-repeating header becomes ``SECTION_HEADER`` (so
the heading-hierarchy step can level it) when ``promote_headers_to_section`` is set, a
non-repeating footer becomes plain ``TEXT``. The step never drops or reorders items and never
touches genuine, repeating headers/footers.
"""

import re
from collections import defaultdict
from typing import Protocol

from docling_core.types.doc import DocItemLabel

from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import HeaderFooterOptions

_DIGITS_RE = re.compile(r"\d+")
_WS_RE = re.compile(r"\s+")

_HEADER_FOOTER_LABELS = (DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER)


class _Candidate(Protocol):
    """Minimal interface the heuristic needs from an assembled element."""

    label: DocItemLabel
    page_no: int
    text: str | None


def _normalize_signature(text: str | None) -> str:
    """Normalize header/footer text for cross-page comparison.

    Digits are stripped so that running headers/footers differing only by a page number or
    date ("Page 3 of 10" vs "Page 4 of 10") collapse to the same signature; whitespace is
    folded and the result is lower-cased.
    """
    if not text:
        return ""
    return _WS_RE.sub(" ", _DIGITS_RE.sub("", text.lower())).strip()


def select_header_footer_reclassifications(
    elements: list[_Candidate],
    page_count: int,
    options: HeaderFooterOptions,
) -> dict[int, DocItemLabel]:
    """Decide which header/footer elements should be demoted into the body.

    Pure function over the assembled elements: it only reads ``label``, ``page_no`` and
    ``text`` and returns a mapping ``{index_in_elements: new_label}`` for the detections to
    reclassify. Everything not in the mapping keeps its current label.
    """
    if not options.enabled or page_count < options.min_pages:
        return {}

    result: dict[int, DocItemLabel] = {}
    for label in _HEADER_FOOTER_LABELS:
        idxs = [i for i, el in enumerate(elements) if el.label == label]
        if not idxs:
            continue

        # Coverage: fraction of pages on which this side appears at all (issue #3693 idea).
        pages_on_side = {elements[i].page_no for i in idxs}
        side_is_genuine = len(pages_on_side) / page_count >= options.min_coverage

        # Repetition: pages each normalized text appears on.
        sig_pages: dict[str, set[int]] = defaultdict(set)
        for i in idxs:
            sig_pages[_normalize_signature(elements[i].text)].add(elements[i].page_no)

        if label == DocItemLabel.PAGE_HEADER and options.promote_headers_to_section:
            body_label = DocItemLabel.SECTION_HEADER
        else:
            body_label = DocItemLabel.TEXT

        for i in idxs:
            sig = _normalize_signature(elements[i].text)
            if not sig:
                # Empty/blank detection carries no content to recover; leave as furniture.
                continue
            repeats = len(sig_pages[sig]) >= options.min_repeat_pages
            if repeats or side_is_genuine:
                continue
            result[i] = body_label

    return result


class HeaderFooterModel:
    """Demotes mis-detected page headers/footers back into the document body.

    Mutates ``conv_res.assembled`` in place (so the reading-order model that runs next sees the
    corrected labels) and returns ``conv_res`` for call-site symmetry with the other stages.
    """

    def __init__(self, options: HeaderFooterOptions):
        self.options = options

    def __call__(self, conv_res: ConversionResult) -> ConversionResult:
        if not self.options.enabled or conv_res.assembled is None:
            return conv_res

        elements = conv_res.assembled.elements
        page_count = len(conv_res.pages)
        if page_count == 0:
            return conv_res

        reclassifications = select_header_footer_reclassifications(
            elements, page_count, self.options
        )
        if not reclassifications:
            return conv_res

        reclassified = set()
        for idx, new_label in reclassifications.items():
            element = elements[idx]
            element.label = new_label
            # Keep the backing cluster label aligned with the element label.
            element.cluster.label = new_label
            reclassified.add(id(element))

        # Keep the headers/body buckets consistent: a recovered header/footer is no longer
        # furniture, so move it out of the headers bucket and into the body bucket.
        assembled = conv_res.assembled
        moved = [el for el in assembled.headers if id(el) in reclassified]
        if moved:
            assembled.headers = [
                el for el in assembled.headers if id(el) not in reclassified
            ]
            assembled.body = assembled.body + moved

        return conv_res
