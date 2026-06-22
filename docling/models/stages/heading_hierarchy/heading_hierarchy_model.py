"""Section-header level inference for the PDF/image pipeline.

The layout model classifies regions as ``SECTION_HEADER`` without a level, so every heading
produced by the PDF path defaults to ``level=1`` and the document hierarchy is flattened
(Roman-numeral parts and Arabic-numeral subsections collapse to the same depth).

:class:`HeadingHierarchyModel` runs right after the reading-order model and assigns
``SectionHeaderItem.level`` using -- in precedence order:

1. **numbering** -- legal/outline numbering such as ``PART I -> 1. -> 1.1 -> (a) -> (i)``.
   This is the primary signal: on legal/regulatory documents numbering is far more reliable
   than styling, which is often uniform.
2. **style** -- font size approximated from the parsed PDF cells, used only for headings
   that have no recognizable numbering.

Bookmark / PDF-outline inference (the most authoritative signal) requires new backend
plumbing and is intentionally left as a follow-up; :func:`_infer_from_bookmarks` is the
extension point and currently a no-op.

The model only ever rewrites heading levels -- it never adds, removes or reorders items. The
core (:meth:`HeadingHierarchyModel.assign_heading_levels`) works on a bare
``DoclingDocument`` so it can be reused outside the pipeline; the font-based fallback simply
needs the parsed pages passed in.
"""

import re
from dataclasses import dataclass
from statistics import median

from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import SectionHeaderItem
from docling_core.types.doc.page import SegmentedPdfPage

from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import HeadingHierarchyOptions

# Default precedence of numbering schemes, highest hierarchy level first. ``dotted`` shares
# the ``arabic`` rank and is ordered below it by its segment depth (1.1 below 1.).
_DEFAULT_FAMILY_ORDER = [
    "part",  # PART I / TITLE I / BOOK I
    "chapter",  # CHAPTER 1
    "article",  # ARTICLE 1 / SECTION 1 / Clause / § 1
    "roman_u",  # I. II. III.
    "arabic",  # 1. 2. 3.  (and dotted 1.1, 1.1.1 by depth)
    "alpha_u",  # A. B. C.
    "alpha_l",  # (a) (b) (c)
    "roman_l",  # (i) (ii) (iii)
]

# Single characters that are valid Roman numerals and therefore ambiguous with alpha markers
# (e.g. "I." may be Roman "1" or alpha "9"). Resolved in :func:`_resolve_ambiguous`.
_ROMAN_SINGLES = set("IVXLCDMivxlcdm")

# Canonical Roman-numeral validator (1..3999), case-insensitive via ``.upper()``.
_ROMAN_RE = re.compile(
    r"^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$", re.IGNORECASE
)

_KW_PART = re.compile(r"^(part|title|book)\b", re.IGNORECASE)
_KW_CHAPTER = re.compile(r"^(chapter)\b", re.IGNORECASE)
_KW_ARTICLE = re.compile(
    r"^(article|section|clause|schedule|annex|appendix|rule)\b", re.IGNORECASE
)
_SECTION_SYMBOL = re.compile(r"^§+\s*\d")  # § 1 / §§ 1.2
# Dotted decimal outline (1.1, 1.1.1, ...), terminated by space/end/punctuation.
_DOTTED = re.compile(r"^(\d+(?:\.\d+)+)(?:[.)\]\s]|$)")
# Single Arabic index (1. / 2)).
_ARABIC = re.compile(r"^(\d+)[.)]")
# Single/multi letter marker, optionally parenthesized: (a) / A. / (iv) / IV.
_LETTER = re.compile(r"^\(?\s*([A-Za-z]+)\s*[).]")


@dataclass
class _Marker:
    """A parsed leading numbering marker for a heading."""

    family: str  # canonical scheme family (see _DEFAULT_FAMILY_ORDER)
    depth: int = 1  # dotted-decimal segment count; 1 for everything else
    token: str | None = None  # raw alpha/roman token, kept for ambiguity resolution
    ambiguous: bool = False  # single-letter Roman/alpha that needs context to resolve


def _is_roman(token: str) -> bool:
    return bool(token) and _ROMAN_RE.fullmatch(token) is not None


def _classify_letter(token: str) -> _Marker | None:
    """Classify a bare alpha/Roman token (``A``, ``iv``, ``i`` ...) into a marker."""
    upper = token.isupper()
    if len(token) == 1:
        if token in _ROMAN_SINGLES:
            # Ambiguous: could be Roman or the Nth letter; resolved later by context.
            return _Marker(
                family="roman_u" if upper else "roman_l", token=token, ambiguous=True
            )
        return _Marker(family="alpha_u" if upper else "alpha_l", token=token)
    # Multi-letter tokens only count as numbering if they are valid Roman numerals;
    # otherwise they are plain words (e.g. "Summary."), not a numbered heading.
    if _is_roman(token):
        return _Marker(family="roman_u" if upper else "roman_l", token=token)
    return None


def _parse_marker(text: str) -> _Marker | None:
    """Extract the leading numbering marker from a heading, or None if unnumbered."""
    s = (text or "").strip()
    if not s:
        return None

    if _KW_PART.match(s):
        return _Marker(family="part")
    if _KW_CHAPTER.match(s):
        return _Marker(family="chapter")
    if _KW_ARTICLE.match(s) or _SECTION_SYMBOL.match(s):
        return _Marker(family="article")

    m = _DOTTED.match(s)
    if m:
        return _Marker(family="dotted", depth=m.group(1).count(".") + 1)
    if _ARABIC.match(s):
        return _Marker(family="arabic")

    m = _LETTER.match(s)
    if m:
        return _classify_letter(m.group(1))
    return None


def _resolve_ambiguous(markers: list[_Marker | None]) -> None:
    """Resolve single-letter Roman/alpha markers in place using document-wide evidence.

    A lone ``I.`` is Roman when the document also contains unambiguous Roman markers
    (``II``, ``III`` ...) and alpha when it contains unambiguous alpha markers (``B``, ``F``
    ...). When evidence is absent or conflicting, ``I``/``i`` default to Roman (the common
    legal case) and other letters to alpha.
    """

    def _has(family: str) -> bool:
        return any(
            m is not None and not m.ambiguous and m.family == family for m in markers
        )

    upper_roman, upper_alpha = _has("roman_u"), _has("alpha_u")
    lower_roman, lower_alpha = _has("roman_l"), _has("alpha_l")

    for m in markers:
        if m is None or not m.ambiguous or m.token is None:
            continue
        upper = m.token.isupper()
        has_roman = upper_roman if upper else lower_roman
        has_alpha = upper_alpha if upper else lower_alpha
        if has_roman and not has_alpha:
            roman = True
        elif has_alpha and not has_roman:
            roman = False
        else:
            roman = m.token in ("I", "i")
        if roman:
            m.family = "roman_u" if upper else "roman_l"
        else:
            m.family = "alpha_u" if upper else "alpha_l"
        m.ambiguous = False


def _family_rank(family: str, order: list[str]) -> int:
    key = "arabic" if family == "dotted" else family
    try:
        return order.index(key)
    except ValueError:
        return len(order)  # unknown scheme -> lowest priority


def _infer_from_numbering(
    headings: list[SectionHeaderItem], options: HeadingHierarchyOptions
) -> dict[int, int]:
    """Map heading index -> level from numbering markers (relative, compressed levels)."""
    order = options.numbering_schemes or _DEFAULT_FAMILY_ORDER
    markers = [_parse_marker(h.text) for h in headings]
    _resolve_ambiguous(markers)

    keys: dict[int, tuple[int, int]] = {}
    for i, m in enumerate(markers):
        if m is None:
            continue
        keys[i] = (_family_rank(m.family, order), m.depth)
    if not keys:
        return {}

    # Compress the distinct (rank, depth) keys actually present into contiguous levels, so a
    # document that starts at "1." is not forced to start at depth 2.
    key_to_level = {
        key: lvl for lvl, key in enumerate(sorted(set(keys.values())), start=1)
    }
    return {i: key_to_level[key] for i, key in keys.items()}


def _heading_font_size(
    item: SectionHeaderItem, parsed_pages: dict[int, SegmentedPdfPage]
) -> float | None:
    """Median height of parsed PDF cells overlapping the heading, as a font-size proxy."""
    if not item.prov:
        return None
    prov = item.prov[0]
    parsed = parsed_pages.get(prov.page_no)
    if parsed is None:
        return None

    page_height = parsed.dimension.height
    hbox = prov.bbox.to_top_left_origin(page_height)
    heights: list[float] = []
    for cell in parsed.textline_cells:
        if not cell.text or not cell.text.strip():
            continue
        cbox = cell.rect.to_bounding_box().to_top_left_origin(page_height)
        if hbox.overlaps(cbox):
            heights.append(cell.rect.height)
    if not heights:
        return None
    return median(heights)


def _infer_from_style(
    headings: list[SectionHeaderItem],
    parsed_pages: dict[int, SegmentedPdfPage],
    options: HeadingHierarchyOptions,
) -> dict[int, int]:
    """Map heading index -> level from font size buckets (larger size = higher level)."""
    if not parsed_pages:
        return {}
    sizes: dict[int, float] = {}
    for i, heading in enumerate(headings):
        size = _heading_font_size(heading, parsed_pages)
        if size is not None:
            sizes[i] = size
    if not sizes:
        return {}

    # Bucket by rounded point size; the largest size becomes level 1.
    rounded = {i: round(size) for i, size in sizes.items()}
    ranked = {
        size: lvl
        for lvl, size in enumerate(sorted(set(rounded.values()), reverse=True), start=1)
    }
    return {i: ranked[size] for i, size in rounded.items()}


def _infer_from_bookmarks(
    headings: list[SectionHeaderItem], options: HeadingHierarchyOptions
) -> dict[int, int]:
    """Reserved: map heading index -> level from the PDF outline/bookmarks.

    Authoritative when present, but the PDF backends do not yet surface the document
    outline. This is the extension point for that follow-up; it is currently a no-op.
    """
    return {}


class HeadingHierarchyModel:
    """Assigns ``SectionHeaderItem.level`` on an already-assembled ``DoclingDocument``.

    Intended to run right after the reading-order model. It accepts a ``ConversionResult``
    and returns the (in-place modified) ``DoclingDocument``. The core logic is exposed via
    :meth:`assign_heading_levels`, which works on a bare ``DoclingDocument`` (plus optional
    parsed pages for the style fallback) so it can be reused outside the pipeline.
    """

    def __init__(self, options: HeadingHierarchyOptions):
        self.options = options

    def __call__(self, conv_res: ConversionResult) -> DoclingDocument:
        document = conv_res.document
        if not self.options.enabled:
            return document

        parsed_pages: dict[int, SegmentedPdfPage] = {}
        if self.options.use_style:
            parsed_pages = {
                page.page_no: page.parsed_page
                for page in conv_res.pages
                if page.parsed_page is not None
            }
        return self.assign_heading_levels(document, parsed_pages=parsed_pages)

    def assign_heading_levels(
        self,
        document: DoclingDocument,
        parsed_pages: dict[int, SegmentedPdfPage] | None = None,
    ) -> DoclingDocument:
        """Assign heading levels in place from the configured signals.

        Numbering wins over style; bookmarks (once implemented) win over both. Headings with
        no applicable signal keep their existing level. ``parsed_pages`` (page number ->
        parsed page) is only needed for the style fallback.
        """
        headings = [
            item for item in document.texts if isinstance(item, SectionHeaderItem)
        ]
        if not headings:
            return document

        levels: dict[int, int] = {}
        if self.options.use_numbering:
            levels.update(_infer_from_numbering(headings, self.options))
        if self.options.use_style and parsed_pages:
            for i, level in _infer_from_style(
                headings, parsed_pages, self.options
            ).items():
                levels.setdefault(i, level)  # do not override a numbering-derived level
        if self.options.use_bookmarks:
            levels.update(
                _infer_from_bookmarks(headings, self.options)
            )  # authoritative

        for i, heading in enumerate(headings):
            level = levels.get(i)
            if level is not None:
                heading.level = max(1, min(int(level), self.options.max_level))
        return document
