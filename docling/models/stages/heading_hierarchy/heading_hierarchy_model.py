# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Section-header level inference for the PDF/image pipeline.

The layout model classifies regions as ``SECTION_HEADER`` without a level, so every heading
produced by the PDF path defaults to ``level=1`` and the document hierarchy is flattened
(Roman-numeral parts and Arabic-numeral subsections collapse to the same depth).

:class:`HeadingHierarchyModel` runs right after the reading-order model and assigns
``SectionHeaderItem.level`` using -- in precedence order:

1. **bookmarks** -- the PDF outline / table-of-contents, when present. This is the most
   authoritative signal: the outline is the document's own declared hierarchy. Bookmarks are
   fuzzily matched (title + page) to detected headings; a confidently matched heading takes
   the bookmark's depth, and a confidently matched *list-item* is promoted to a heading
   (layout models often mis-classify a heading as a list-item).
2. **numbering** -- legal/outline numbering such as ``PART I -> 1. -> 1.1 -> (a) -> (i)``.
   The primary signal for headings without a bookmark match: on legal/regulatory documents
   numbering is far more reliable than styling, which is often uniform.
3. **style** -- the heading's visual style, read from the parsed PDF cells, used only for
   headings that have neither a bookmark match nor recognizable numbering. Headings are ranked
   by font size first -- with near-equal sizes merged, since the size is measured from the cells
   and the same font measures taller on a heading that has descenders -- and then by font weight,
   slant and letter case, so that headings sharing a size are still separated (bold above
   regular, upright above italic, all-caps above mixed).

With ``demote_unmatched_headings`` (off by default), a detected heading that matches no bookmark
is demoted to plain text when the outline looks complete -- see
:func:`_demote_unmatched_headings` for the completeness checks.

Apart from promoting a confidently bookmark-matched list-item and the opt-in demotion above, the
model only rewrites heading levels -- it never adds, removes or reorders items. The core
(:meth:`HeadingHierarchyModel.assign_heading_levels`) works on a bare ``DoclingDocument`` so it
can be reused outside the pipeline; the outline drives bookmark inference and the parsed pages
drive the style fallback.
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from statistics import median

from docling_core.types.doc import DocItemLabel, DoclingDocument
from docling_core.types.doc.document import ListItem, SectionHeaderItem, TextItem
from docling_core.types.doc.page import PdfTextCell, SegmentedPdfPage, TextCell

from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import HeadingHierarchyOptions
from docling.utils.font_style import parse_font_style, weight_class
from docling.utils.pdf_outline import _PdfOutlineItem

_log = logging.getLogger(__name__)

# Default precedence of numbering schemes, highest hierarchy level first. ``dotted`` shares
# the ``arabic`` rank and is ordered below it by its segment depth (1.1 below 1.).
_DEFAULT_FAMILY_ORDER = [
    "part",  # PART I / TITLE I / BOOK I
    "chapter",  # CHAPTER 1
    "division",  # DIVISION I / Division 1 / Division A
    "subdivision",  # SUBDIVISION A / Subdivision a
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

# Part/chapter/article keywords are ordinary words too ("Rule Definition", "Part-Year
# Method"), so they only count as numbering when they stand alone ("APPENDIX", "Annex:")
# or the next token is an enumerator; see :func:`_keyword_marker`.
_KW_TAIL = r"(?:\s+([^\s.,:;)\]\u2013\u2014-]+)|\s*(?:[.:\u2013\u2014]|-\s|$))"
_KW_PART = re.compile(r"^(?:part|title|book)" + _KW_TAIL, re.IGNORECASE)
# Skip statute citations like "Chapter A-15" (letter-dash-digit); real chapters still match.
_KW_CHAPTER = re.compile(r"^chapter(?!\s+[a-z]-\d)" + _KW_TAIL, re.IGNORECASE)
_KW_ARTICLE = re.compile(
    r"^(?:article|section|clause|schedule|annex|appendix|rule)" + _KW_TAIL,
    re.IGNORECASE,
)
_NUMBER_WORDS = frozenset(
    "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
    " fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy"
    " eighty ninety first second third fourth fifth sixth seventh eighth ninth tenth"
    " eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth"
    " nineteenth twentieth".split()
)
# "Division"/"Subdivision" are stricter: they only count as numbering when an explicit
# enumerator (an Arabic index, a Roman numeral, or a single letter) follows. The enumerator
# must be terminated by whitespace, end-of-string, or separator punctuation, so a French
# elision such as "Division d'appel" (a letter followed by an apostrophe) is not read as a
# marker.
_KW_ENUM = r"\s+(\d+|[a-z]|[ivxlcdm]+)(?=\s|$|[.,:;)\]\u2013\u2014-])"
_KW_DIVISION = re.compile(r"^division" + _KW_ENUM, re.IGNORECASE)
_KW_SUBDIVISION = re.compile(r"^subdivision" + _KW_ENUM, re.IGNORECASE)
_SECTION_SYMBOL = re.compile(r"^§+\s*\d")  # § 1 / §§ 1.2
_SEP = r"(?:[)\]]|[:\-\u2013\u2014](?=\s|$))"
# Dotted decimal outline (1.1, 1.1.1, ...), terminated by punctuation/space/end.
_DOTTED = re.compile(r"^\(?\s*(\d+(?:\.\d+)+)(?:[.)\]\s]|[:\-\u2013\u2014](?=\s|$)|$)")
# Single Arabic index (1. / 1) / (1) / 1: / 1 -).
_ARABIC = re.compile(r"^\(?\s*(\d+)\s*(?:\.(?!\d)|" + _SEP + r")")
# A bare index is only accepted with document-wide sequence evidence.
_BARE_ARABIC = re.compile(r"^(\d+)\s+\S")
# Single/multi letter marker, optionally parenthesized: (a) / A. / (iv) / IV. / A: / A -
_LETTER = re.compile(r"^\(?\s*([A-Za-z]+)\s*(?:\.|" + _SEP + r")")


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


def _keyword_enumerator(m: re.Match[str] | None) -> bool:
    """True when a keyword match carries a real enumerator (digit, letter, or Roman).

    Multi-letter tokens are only accepted when they are valid Roman numerals, so
    "Division Civil Remedies" (all Roman-numeral letters) is not mistaken for a marker.
    """
    if m is None:
        return False
    tok = m.group(1)
    return tok.isdigit() or len(tok) == 1 or _is_roman(tok)


def _keyword_marker(m: re.Match[str] | None) -> bool:
    """True when a part/chapter/article keyword stands alone or precedes an enumerator.

    The enumerator may contain a digit ("Chapter 2D", "Rule 12(b)"), or be a single letter,
    a valid Roman numeral, or a number word ("CHAPTER ONE", "ARTICLE FIRST").
    """
    if m is None:
        return False
    tok = m.group(1)
    return (
        tok is None
        or any(c.isdigit() for c in tok)
        or len(tok) == 1
        or _is_roman(tok)
        or tok.lower() in _NUMBER_WORDS
    )


def _parse_marker(text: str) -> _Marker | None:
    """Extract the leading numbering marker from a heading, or None if unnumbered."""
    s = (text or "").strip()
    if not s:
        return None

    if _keyword_marker(_KW_PART.match(s)):
        return _Marker(family="part")
    if _keyword_enumerator(_KW_DIVISION.match(s)):
        return _Marker(family="division")
    if _keyword_enumerator(_KW_SUBDIVISION.match(s)):
        return _Marker(family="subdivision")
    if _keyword_marker(_KW_CHAPTER.match(s)):
        return _Marker(family="chapter")
    if _keyword_marker(_KW_ARTICLE.match(s)) or _SECTION_SYMBOL.match(s):
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


def _resolve_bare_arabic(
    headings: list[SectionHeaderItem], markers: list[_Marker | None]
) -> None:
    """Accept bare Arabic indices in consecutive runs starting at 1, with at least two entries.

    A leading number alone may be a year or quantity. Require a chapter sequence before treating
    it as numbering, allowing explicit Arabic markers to provide evidence too. Dotted sections
    and unnumbered headings may intervene. A gap such as ``1 Intro``, ``2 Methods``, ``4 Results``
    ends the run, leaving the bare ``4`` unrecognized. A 1-based run of incidental headings, such
    as ``1 January``, ``2 February`` or ``1 kg``, ``2 kg``, can still be misclassified.
    """
    sequence: list[int] = []
    for i, heading in enumerate(headings):
        marker = markers[i]
        if marker is not None and marker.family != "arabic":
            continue
        text = heading.text.strip()
        match = _ARABIC.match(text) if marker is not None else _BARE_ARABIC.match(text)
        if match is None:
            continue
        number = int(match.group(1))
        if number == 1:
            sequence = [i]
        elif sequence and number == len(sequence) + 1:
            sequence.append(i)
            # Confirm the first entry when 2 arrives, then only the newly extended pair.
            for index in sequence[-2:]:
                if markers[index] is None:
                    markers[index] = _Marker(family="arabic")
        else:
            sequence = []


def _infer_from_numbering(
    headings: list[SectionHeaderItem], options: HeadingHierarchyOptions
) -> dict[int, int]:
    """Map heading index -> level from numbering markers (relative, compressed levels)."""
    order = options.numbering_schemes or _DEFAULT_FAMILY_ORDER
    markers = [_parse_marker(h.text) for h in headings]
    _resolve_ambiguous(markers)
    _resolve_bare_arabic(headings, markers)

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


@dataclass(frozen=True)
class _HeadingStyle:
    """The visual style of a heading, used as the ranking key of the style fallback."""

    size: float  # median height of the underlying cells, as a font-size proxy
    weight_cls: int = 0  # 0 light/regular, 1 medium/semibold, 2 bold and above
    italic: bool = False
    caps: bool = False


# Share of the heading's characters that must be italic for the heading to count as italic.
_ITALIC_RATIO = 0.6

# Layout labels whose text lines the style signal reads.
_STYLE_LABELS = frozenset({DocItemLabel.SECTION_HEADER, DocItemLabel.LIST_ITEM})


def _is_all_caps(text: str) -> bool:
    """Whether the text is written in capitals (ignoring digits and punctuation)."""
    letters = [char for char in text if char.isalpha()]
    return len(letters) >= 4 and all(char.isupper() for char in letters)


def _heading_cells(
    item: SectionHeaderItem, parsed_pages: dict[int, SegmentedPdfPage]
) -> list[TextCell]:
    """The parsed PDF cells with text that overlap the heading's first provenance box."""
    if not item.prov:
        return []
    prov = item.prov[0]
    parsed = parsed_pages.get(prov.page_no)
    if parsed is None:
        return []

    page_height = parsed.dimension.height
    hbox = prov.bbox.to_top_left_origin(page_height)
    return [
        cell
        for cell in parsed.textline_cells
        if cell.text
        and cell.text.strip()
        and hbox.overlaps(cell.rect.to_bounding_box().to_top_left_origin(page_height))
    ]


def _heading_style(
    item: SectionHeaderItem,
    parsed_pages: dict[int, SegmentedPdfPage],
    options: HeadingHierarchyOptions,
) -> _HeadingStyle | None:
    """Style of a heading, derived from the parsed PDF cells overlapping it.

    Weight and slant are a character-weighted vote across those cells, since a heading can mix
    fonts (a regular "1.1 " in front of a bold title). Cells without font information -- OCR
    output, the pypdfium2 backend, or names that carry no recognizable styling -- contribute
    their size but leave the heading at the regular weight, so the ranking degrades to font size.
    """
    heights: list[float] = []
    weights: Counter[int] = Counter()
    styled_chars = 0
    italic_chars = 0
    for cell in _heading_cells(item, parsed_pages):
        heights.append(cell.rect.height)

        if not options.use_font_style or not isinstance(cell, PdfTextCell):
            continue
        style = parse_font_style(cell.font_name)
        if not style.known:
            continue
        chars = len(cell.text.strip())
        weights[weight_class(style.weight)] += chars
        styled_chars += chars
        if style.italic:
            italic_chars += chars

    if not heights:
        return None
    size = median(heights)
    if not options.use_font_style:
        return _HeadingStyle(size=size)
    return _HeadingStyle(
        size=size,
        # On a tie, the heavier class wins: emphasis is what makes a heading stand out.
        weight_cls=max(weights, key=lambda cls: (weights[cls], cls), default=0),
        italic=bool(styled_chars) and italic_chars / styled_chars >= _ITALIC_RATIO,
        caps=_is_all_caps(item.text),
    )


def _cluster_sizes(sizes: set[float], tolerance: float) -> dict[float, int]:
    """Group font sizes into clusters, largest first, and map each size to its cluster index.

    The size of a heading is the median height of its cells, which measures the glyphs that are
    actually on the line rather than the font size: a heading with descenders ("Securing and
    protecting") measures a couple of points taller than one without ("Contents") in the very same
    font. Treating every distinct height as its own level therefore invents levels and, because
    each level then holds a single heading, leaves no headings for weight or slant to separate.
    Consecutive sizes within ``tolerance`` (relative) are merged to absorb that measurement noise.
    """
    clusters: dict[float, int] = {}
    index = 0
    previous: float | None = None
    for size in sorted(sizes, reverse=True):
        if previous is not None and (previous - size) > tolerance * previous:
            index += 1
        clusters[size] = index
        previous = size
    return clusters


def _infer_from_style(
    headings: list[SectionHeaderItem],
    parsed_pages: dict[int, SegmentedPdfPage],
    options: HeadingHierarchyOptions,
) -> dict[int, int]:
    """Map heading index -> level from the heading styles (most prominent style = level 1)."""
    if not parsed_pages:
        return {}
    styles: dict[int, _HeadingStyle] = {}
    for i, heading in enumerate(headings):
        style = _heading_style(heading, parsed_pages, options)
        if style is not None:
            styles[i] = style
    if not styles:
        return {}

    clusters = _cluster_sizes(
        {style.size for style in styles.values()}, options.style_size_tolerance
    )
    # Order by size first, then by how much the heading stands out within its size: heavier before
    # lighter, upright before italic (italic sits lighter on the page), capitals before mixed case.
    keys = {
        i: (clusters[style.size], -style.weight_cls, style.italic, not style.caps)
        for i, style in styles.items()
    }
    # Compress the distinct keys actually present into contiguous levels, so a signal that does
    # not vary across the document's headings (e.g. all of them bold) adds no levels.
    ranked = {key: lvl for lvl, key in enumerate(sorted(set(keys.values())), start=1)}
    return {i: ranked[key] for i, key in keys.items()}


# --------------------------------------------------------------------------------- bookmarks

# Leading numbering marker stripped before fuzzy-matching a title, so a bookmark "Definitions"
# matches an on-page heading "1.1 Definitions" (and vice-versa).
_LEADING_MARKER = re.compile(
    r"^\s*(?:"
    r"(?:part|title|book|division|subdivision|chapter|article|section|clause|schedule|annex|appendix|rule)"
    r"\b[\s.:]*[0-9ivxlcdm]*"
    r"|§+\s*[0-9.]+"
    r"|\(?[0-9]+(?:\.[0-9]+)*[.)\]]?"
    r"|\(?[A-Za-z]{1,2}(?:[.)\]]|(?=\s*[:\-\u2013\u2014]))"
    r")[\s.:)\-\u2013\u2014]*",
    re.IGNORECASE,
)


def _norm(text: str) -> str:
    """Lower-case, collapse whitespace and trim outer punctuation for matching."""
    s = re.sub(r"\s+", " ", (text or "").lower()).strip()
    # Trim leading/trailing punctuation (incl. unicode dash variants) and underscores.
    return re.sub(r"^[\W_]+|[\W_]+$", "", s)


def _strip_marker(text: str) -> str:
    return _LEADING_MARKER.sub("", text or "", count=1)


def _match_score(cand_text: str, bm_title: str) -> float:
    """Fuzzy similarity in 0..1 between a detected heading and a bookmark title.

    Both strings are compared with and without their leading numbering marker (bookmarks often
    omit or carry a different marker), and containment of one normalized title in the other
    boosts the score (bookmarks are frequently truncated).
    """
    variants_a = {_norm(cand_text), _norm(_strip_marker(cand_text))} - {""}
    variants_b = {_norm(bm_title), _norm(_strip_marker(bm_title))} - {""}
    best = 0.0
    for a in variants_a:
        for b in variants_b:
            best = max(best, SequenceMatcher(None, a, b).ratio())
            if len(a) >= 4 and len(b) >= 4 and (a in b or b in a):
                best = max(best, 0.92)
    return best


def _item_page_and_top(
    item: SectionHeaderItem | ListItem, document: DoclingDocument
) -> tuple[int | None, float | None]:
    """Return ``(1-based page, top-left-origin top)`` for a text item, when derivable."""
    if not item.prov:
        return None, None
    prov = item.prov[0]
    page = document.pages.get(prov.page_no)
    if page is not None and page.size is not None:
        return prov.page_no, prov.bbox.to_top_left_origin(page.size.height).t
    return prov.page_no, None


def _infer_from_bookmarks(
    document: DoclingDocument,
    outline: list[_PdfOutlineItem],
    options: HeadingHierarchyOptions,
) -> dict[int, int]:
    """Match the PDF outline to detected headings/list-items and return authoritative levels.

    Returns ``id(item) -> level`` for every confidently matched heading (existing or promoted).
    A matched list-item is promoted to a ``SectionHeaderItem`` in place. The mapping is keyed by
    object identity so it survives the ``texts`` reshuffling that promotion causes.
    """
    candidates: list[SectionHeaderItem | ListItem] = [
        item
        for item in document.texts
        if isinstance(item, (SectionHeaderItem, ListItem))
    ]
    if not candidates:
        return {}

    info = [(item, *_item_page_and_top(item, document)) for item in candidates]
    claimed: set[int] = set()
    matches: list[tuple[SectionHeaderItem | ListItem, int]] = []

    for bm in outline:
        title = (bm.title or "").strip()
        if not title:
            continue
        threshold = options.bookmark_match_threshold
        if bm.page_no is None:
            threshold = min(1.0, threshold + 0.1)  # cross-page match must be stronger

        best_idx: int | None = None
        best_score = 0.0
        best_dist = float("inf")
        for idx, (item, page_no, top) in enumerate(info):
            if idx in claimed:
                continue
            if bm.page_no is not None and page_no is not None and page_no != bm.page_no:
                continue
            score = _match_score(item.text, title)
            if score < threshold:
                continue
            dist = (
                abs(top - bm.y_top)
                if (top is not None and bm.y_top is not None)
                else float("inf")
            )
            if score > best_score + 1e-6 or (
                abs(score - best_score) <= 1e-6 and dist < best_dist
            ):
                best_idx, best_score, best_dist = idx, score, dist
        if best_idx is not None:
            claimed.add(best_idx)
            matches.append((info[best_idx][0], bm.level))

    if not matches:
        return {}

    # Compress the raw bookmark depths actually used into contiguous 1-based levels.
    used_levels = sorted({lvl for _, lvl in matches})
    level_map = {lvl: i + 1 for i, lvl in enumerate(used_levels)}

    result: dict[int, int] = {}
    for item, raw_level in matches:
        level = level_map[raw_level]
        if isinstance(item, ListItem):
            # Promote a mis-classified list-item: same content/position, now a heading.
            heading = SectionHeaderItem(
                self_ref=item.self_ref,  # reassigned by replace_item's insert
                parent=item.parent,
                content_layer=item.content_layer,
                prov=item.prov,
                orig=item.orig,
                text=item.text,
                formatting=item.formatting,
                hyperlink=item.hyperlink,
                level=level,
            )
            document.replace_item(new_item=heading, old_item=item)
            result[id(heading)] = level
        else:
            result[id(item)] = level
    return result


# Unnumbered headings that documents commonly leave out of their bookmarks. They are never demoted.
_STANDARD_HEADING = re.compile(
    r"^(abstract|keywords|acknowledge?ments?|references|bibliography|appendix|appendices)\b",
    re.IGNORECASE,
)

# Share of the detected headings that must match a bookmark for the outline to count as complete.
_COMPLETE_MATCH_SHARE = 0.5

# Subset-embedded fonts are prefixed with six uppercase letters and a plus (PDF 32000-1 9.6.4).
_FONT_SUBSET_PREFIX = re.compile(r"^[A-Z]{6}\+")


# A word set with letter spacing, such as ``A B S T R A C T``: three or more single letters.
_LETTER_SPACED = re.compile(r"^\s*(?:[^\W\d_]\s+){2,}[^\W\d_]\s*$")

# The section number at the start of a heading or bookmark title: ``4``, ``4.``, ``2.1``, ``IV.``,
# ``A.``.
_SECTION_NUMBER = re.compile(
    r"^\s*\(?(\d+(?:\.\d+)*|[IVXLC]+(?=[.)\s])|[A-Z](?=[.)]))[.)]?(?=\s|$)"
)


def _is_standard_heading(text: str) -> bool:
    unspaced = re.sub(r"\s+", "", text) if _LETTER_SPACED.match(text) else text
    return any(
        _STANDARD_HEADING.match(_norm(variant))
        for variant in (text, _strip_marker(text), unspaced)
    )


def _section_number(text: str) -> str | None:
    match = _SECTION_NUMBER.match(text or "")
    return match.group(1).upper() if match else None


def _body_size(parsed_pages: dict[int, SegmentedPdfPage]) -> float | None:
    """The median text-line height of the document, weighted by characters.

    Cell heights follow the glyphs on the line (descenders, superscripts), so body lines spread
    over a range of heights; the weighted median is a stabler body-size proxy than the mode.
    """
    weighted = sorted(
        (cell.rect.height, len(text))
        for parsed in parsed_pages.values()
        for cell in parsed.textline_cells
        if (text := (cell.text or "").strip())
    )
    total = sum(chars for _, chars in weighted)
    seen = 0
    for height, chars in weighted:
        seen += chars
        if 2 * seen >= total:
            return height
    return None


def _face(cell: TextCell) -> str | None:
    """The font face of a digital cell (its font name without the subset tag), else None."""
    if not isinstance(cell, PdfTextCell):
        return None
    name = _FONT_SUBSET_PREFIX.sub("", cell.font_name.lstrip("/"), count=1)
    return name if name and name != "null" else cell.font_key or None


def _main_face(cells: list[TextCell]) -> str | None:
    """The font face that carries most characters of ``cells``."""
    chars: Counter[str] = Counter()
    for cell in cells:
        face = _face(cell)
        if face is not None:
            chars[face] += len(cell.text.strip())
    return chars.most_common(1)[0][0] if chars else None


def _share_a_heading_style(
    headings: list[SectionHeaderItem],
    parsed_pages: dict[int, SegmentedPdfPage],
    options: HeadingHierarchyOptions,
) -> bool:
    """Whether two or more of ``headings`` share a heading-like style.

    A style is heading-like when it is larger than the body text, bold, or set in a font face
    other than the body face. The face check catches bold and italic fonts whose names carry no
    readable style, such as the TeX fonts ``CMBX10`` and ``CMTI10``. A family of unmatched
    headings in such a style means that the outline omits a level (often the deepest one), so
    those headings are real and the outline is not complete.
    """
    body_size = _body_size(parsed_pages)
    if body_size is None:
        return True  # no evidence either way: treat the outline as incomplete
    body_face = _main_face(
        [cell for parsed in parsed_pages.values() for cell in parsed.textline_cells]
    )
    # Weight is read from the font names even when the style fallback ranks by size only.
    style_options = options.model_copy(update={"use_font_style": True})
    keys: list[tuple[float, str | None, bool]] = []
    for heading in headings:
        style = _heading_style(heading, parsed_pages, style_options)
        if style is None:
            continue
        face = _main_face(_heading_cells(heading, parsed_pages))
        larger = style.size > body_size * (1 + options.style_size_tolerance)
        bold = style.weight_cls > 0
        if larger or bold or (face is not None and face != body_face):
            keys.append((style.size, face, bold))
    clusters = _cluster_sizes(
        {size for size, _, _ in keys}, options.style_size_tolerance
    )
    counts = Counter((clusters[size], face, bold) for size, face, bold in keys)
    return any(count >= 2 for count in counts.values())


def _outline_looks_complete(
    headings: list[SectionHeaderItem],
    unmatched: list[SectionHeaderItem],
    demotable: list[SectionHeaderItem],
    outline: list[_PdfOutlineItem],
    parsed_pages: dict[int, SegmentedPdfPage],
    options: HeadingHierarchyOptions,
) -> bool:
    """Whether the outline appears to list every real heading of the document.

    All of these must hold: the outline has more than one level (a flat outline usually lists only
    the top level); every detected dotted-numbered heading (``2.1 ...``) matches a bookmark; at
    least half of the detected headings match a bookmark; and no two of the headings that would be
    demoted share a heading-like style.
    """
    if not parsed_pages:
        return False
    if len({bm.level for bm in outline if (bm.title or "").strip()}) < 2:
        return False
    if any(_DOTTED.match(heading.text.strip()) for heading in unmatched):
        return False
    if len(headings) - len(unmatched) < _COMPLETE_MATCH_SHARE * len(headings):
        return False
    return not _share_a_heading_style(demotable, parsed_pages, options)


def _demote_unmatched_headings(
    document: DoclingDocument,
    matched: set[int],
    outline: list[_PdfOutlineItem],
    parsed_pages: dict[int, SegmentedPdfPage],
    options: HeadingHierarchyOptions,
) -> int:
    """Demote headings that match no bookmark to text, when the outline looks complete.

    ``matched`` holds the ``id`` of every bookmark-matched heading. Standard unnumbered headings
    (Abstract, References, ...) and headings with children are kept. A heading also counts as
    matched when its section number belongs to a bookmark that matched no heading: the bookmark
    title can differ from the printed one (for example, unresolved LaTeX references in hyperref
    bookmarks). Returns the number of demoted headings.
    """
    headings = [item for item in document.texts if isinstance(item, SectionHeaderItem)]
    matched_numbers = {
        _section_number(heading.text) for heading in headings if id(heading) in matched
    }
    free_numbers = {
        number
        for bm in outline
        if (number := _section_number(bm.title or "")) is not None
        and number not in matched_numbers
    }
    unmatched = [
        heading
        for heading in headings
        if id(heading) not in matched
        and _section_number(heading.text) not in free_numbers
    ]
    demotable = [
        heading
        for heading in unmatched
        if not heading.children and not _is_standard_heading(heading.text)
    ]
    if not demotable or not _outline_looks_complete(
        headings, unmatched, demotable, outline, parsed_pages, options
    ):
        return 0

    for heading in demotable:
        text_item = TextItem(
            label=DocItemLabel.TEXT,
            self_ref=heading.self_ref,  # reassigned by the insert
            parent=heading.parent,
            content_layer=heading.content_layer,
            meta=heading.meta,
            prov=heading.prov,
            source=heading.source,
            comments=heading.comments,
            orig=heading.orig,
            text=heading.text,
            formatting=heading.formatting,
            hyperlink=heading.hyperlink,
        )
        document.insert_item_after_sibling(new_item=text_item, sibling=heading)
    # One deletion for all headings: each call re-indexes the whole document.
    document.delete_items(node_items=list(demotable))
    _log.debug("Demoted %d headings that match no bookmark.", len(demotable))
    return len(demotable)


class HeadingHierarchyModel:
    """Assigns ``SectionHeaderItem.level`` on an already-assembled ``DoclingDocument``.

    Intended to run right after the reading-order model. It accepts a ``ConversionResult``
    and returns the (in-place modified) ``DoclingDocument``. The core logic is exposed via
    :meth:`assign_heading_levels`, which works on a bare ``DoclingDocument`` (plus optional
    parsed pages for the style fallback) so it can be reused outside the pipeline.
    """

    def __init__(self, options: HeadingHierarchyOptions):
        self.options = options

    def retain_parsed_lines(self, page: Page) -> SegmentedPdfPage | None:
        """Return the part of a finished page's ``parsed_page`` that this stage reads later.

        The pipeline releases the parsed pages before assembly to save memory, but the style
        signal reads the text lines of the headings during assembly. Only the text lines that
        overlap a section header, or a list item that a bookmark can promote, are kept. With
        ``demote_unmatched_headings``, all text lines are kept, because its style check compares
        the headings with the body text.
        """
        parsed = page.parsed_page
        layout = page.predictions.layout
        if not self.options.enabled or parsed is None:
            return None
        if self.options.demote_unmatched_headings and self.options.use_bookmarks:
            return SegmentedPdfPage(
                dimension=parsed.dimension,
                char_cells=[],
                word_cells=[],
                textline_cells=list(parsed.textline_cells),
                has_lines=bool(parsed.textline_cells),
            )
        if not self.options.use_style or layout is None:
            return None
        height = parsed.dimension.height
        boxes = [
            cluster.bbox.to_top_left_origin(height)
            for cluster in layout.clusters
            if cluster.label in _STYLE_LABELS
        ]
        lines = [
            cell
            for cell in parsed.textline_cells
            if any(
                box.overlaps(cell.rect.to_bounding_box().to_top_left_origin(height))
                for box in boxes
            )
        ]
        return SegmentedPdfPage(
            dimension=parsed.dimension,
            char_cells=[],
            word_cells=[],
            textline_cells=lines,
            has_lines=bool(lines),
        )

    def __call__(self, conv_res: ConversionResult) -> DoclingDocument:
        document = conv_res.document
        if not self.options.enabled:
            return document

        parsed_pages: dict[int, SegmentedPdfPage] = {}
        if self.options.use_style or self.options.demote_unmatched_headings:
            parsed_pages = {
                page.page_no: page.parsed_page
                for page in conv_res.pages
                if page.parsed_page is not None
            }
        outline = conv_res._pdf_outline if self.options.use_bookmarks else None
        try:
            return self.assign_heading_levels(
                document, parsed_pages=parsed_pages, outline=outline
            )
        finally:
            # Release the transient outline once consumed.
            conv_res._pdf_outline = None

    def assign_heading_levels(
        self,
        document: DoclingDocument,
        parsed_pages: dict[int, SegmentedPdfPage] | None = None,
        outline: list[_PdfOutlineItem] | None = None,
    ) -> DoclingDocument:
        """Assign heading levels in place from the configured signals.

        Precedence: bookmarks (when confidently matched) > numbering > style. The bookmark pass
        may promote a mis-classified list-item to a heading and, with
        ``demote_unmatched_headings``, demote headings that a complete outline does not list.
        Headings with no applicable signal keep their existing level. ``parsed_pages`` (page
        number -> parsed page) is only needed for the style fallback and the demotion;
        ``outline`` (PDF bookmarks) for bookmark inference.
        """
        # Bookmark pass first: it may promote list-items and demote headings, changing the set
        # of headings, so it has to run before the heading list is collected.
        bookmark_levels: dict[int, int] = {}
        if self.options.use_bookmarks and outline:
            bookmark_levels = _infer_from_bookmarks(document, outline, self.options)
            if self.options.demote_unmatched_headings and bookmark_levels:
                _demote_unmatched_headings(
                    document,
                    matched=set(bookmark_levels),
                    outline=outline,
                    parsed_pages=parsed_pages or {},
                    options=self.options,
                )

        headings = [
            item for item in document.texts if isinstance(item, SectionHeaderItem)
        ]
        if not headings:
            return document

        levels: dict[int, int] = {}
        # Bookmarks are authoritative: seed levels first so numbering/style cannot override them.
        for i, heading in enumerate(headings):
            level = bookmark_levels.get(id(heading))
            if level is not None:
                levels[i] = level
        if self.options.use_numbering:
            for i, level in _infer_from_numbering(headings, self.options).items():
                levels.setdefault(i, level)  # do not override a bookmark-derived level
        if self.options.use_style and parsed_pages:
            for i, level in _infer_from_style(
                headings, parsed_pages, self.options
            ).items():
                levels.setdefault(i, level)  # do not override bookmark/numbering levels

        for i, heading in enumerate(headings):
            level = levels.get(i)
            if level is not None:
                heading.level = max(1, min(int(level), self.options.max_level))
        return document
