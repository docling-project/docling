# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""The content a Pages document holds, however its container spells it.

Both container generations describe the same things — paragraphs made of runs,
lists, tables, pictures, page furniture, comments — so they are modelled once
here and read into that model by :mod:`docling.backend.iwork.pages_iwa` and
:mod:`docling.backend.iwork.pages_xml`. Turning the result into a
:class:`~docling_core.types.doc.DoclingDocument` is the backend's job, which is
what keeps the two readers from having to agree on anything else.
"""

import re
from typing import NamedTuple, TypeVar

from docling_core.types.doc import (
    DocItemLabel,
    Formatting,
    Script,
    TableData,
)

_T = TypeVar("_T")


class Run(NamedTuple):
    """A stretch of text sharing one character style and one link."""

    text: str
    formatting: Formatting | None
    hyperlink: str | None = None


class ListLabel(NamedTuple):
    """How Pages labels one list item: its depth, and the marker it shows."""

    depth: int
    enumerated: bool
    marker: str


class ListStyle(NamedTuple):
    """A Pages list style: what each nesting depth is labelled with.

    Every field is a parallel array indexed by depth, so a style describes the
    whole ladder of nine levels at once rather than one level at a time.
    """

    label_types: tuple[int, ...]
    strings: tuple[str, ...]

    def label(self, depth: int) -> ListLabel | None:
        """Return how a paragraph at ``depth`` is labelled, or None if plain.

        Args:
            depth: The paragraph's nesting depth, counted from zero.

        Returns:
            The label, or None when this style leaves the depth unlabelled —
            which is what Pages' "None" style does at every depth, and is how a
            paragraph that merely inherits a list style stays body text.
        """
        if depth >= len(self.label_types):
            return None
        label_type = self.label_types[depth]
        if label_type == LABEL_TYPE_NONE:
            return None
        if label_type == LABEL_TYPE_NUMBER:
            return ListLabel(depth, True, "")
        marker = self.strings[depth] if depth < len(self.strings) else ""
        # An image bullet has no text to show, so it falls back to the marker
        # docling uses for an unlabelled item.
        return ListLabel(depth, False, marker or "-")


class Comment(NamedTuple):
    """One comment thread, and the identifier of the text it annotates."""

    text: str
    anchor: str


class Paragraph(NamedTuple):
    """One block of body text with the label its Pages style implies.

    A paragraph is kept as runs rather than a single string because Pages
    applies character styles to arbitrary stretches of it, and a bold phrase in
    the middle of a sentence has to stay attached to that phrase.
    """

    runs: tuple[Run, ...]
    label: DocItemLabel
    level: int | None
    list_label: ListLabel | None = None
    anchors: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        """The paragraph's full text, with its runs joined back together."""
        return "".join(run.text for run in self.runs)


class Picture(NamedTuple):
    """An image anchored in the text flow.

    ``data`` is None when the image's bytes are not in the container — Pages
    writes a placeholder for media it has not downloaded — so the picture is
    still placed, just without an image.
    """

    data: bytes | None
    name: str


class StorageRuns(NamedTuple):
    """The run tables of one ``TSWP.StorageArchive``.

    Each table pairs a character index with the value that applies from there
    until the next entry, so they are read together when the storage is split
    into paragraphs.
    """

    styles: list[tuple[int, str | None]] = []
    characters: list[tuple[int, Formatting | None]] = []
    lists: list[tuple[int, ListStyle | None]] = []
    depths: list[tuple[int, int]] = []
    links: list[tuple[int, str | None]] = []


Block = Paragraph | Picture | TableData
"""One piece of document content, in the order Pages lays it out."""


class Content(NamedTuple):
    """Everything one Pages document holds.

    Page furniture is kept apart from the body flow rather than interleaved with
    it: a header belongs to every page a page master covers, not to one point in
    the text, so there is no position in ``blocks`` that would be right for it.
    """

    blocks: list[Block]
    headers: list[Paragraph] = []
    footers: list[Paragraph] = []
    footnotes: list[Paragraph] = []
    comments: list[Comment] = []


SCRIPTS = {1: Script.SUPER, 2: Script.SUB}
"""``SuperscriptType`` values of a character style, as ``Script`` values.

Zero means neither, and is the value Pages writes for ordinary text.
"""

LABEL_TYPE_NONE = 0
"""``kNone``: the depth is unlabelled, which is what plain body text carries."""

LABEL_TYPE_STRING = 2
"""``kString``: the depth draws a fixed marker.

It is the entry at that depth of the style's ``strings`` — a bullet character,
usually. ``kImage`` (1) draws a picture instead and is treated the same way,
since there is no text in it to show.
"""

LABEL_TYPE_NUMBER = 3
"""``kNumber``: the depth is numbered, so the list is an ordered one."""

HEADING_PATTERN = re.compile(r"^heading\s*(\d+)?$", re.IGNORECASE)
"""Matches Pages' built-in heading styles, e.g. "Heading 1" or bare "Heading"."""

# Apple marks inline attachments (images, footnote anchors) with U+FFFC inside
# the text run. There is no text there to emit.
OBJECT_REPLACEMENT = "￼"


def clean(text: str) -> str:
    """Drop the placeholders Apple writes where an inline attachment sits.

    Whitespace is deliberately left alone: a run boundary can fall mid-sentence,
    so the space on either side of a formatted phrase belongs to the paragraph
    and is trimmed once, by :func:`trim`, rather than at every boundary.
    """
    return text.replace(OBJECT_REPLACEMENT, "")


def trim(runs: list[Run]) -> tuple[Run, ...]:
    """Trim a paragraph's outer whitespace without disturbing its run boundaries.

    Args:
        runs: The paragraph's runs, in document order.

    Returns:
        The runs with leading and trailing whitespace removed and empty ones
        dropped, which is empty when the paragraph holds nothing but whitespace.
    """
    kept = [run for run in runs if run.text]

    while kept:
        head = kept[0]._replace(text=kept[0].text.lstrip())
        if head.text:
            kept[0] = head
            break
        kept.pop(0)

    while kept:
        tail = kept[-1]._replace(text=kept[-1].text.rstrip())
        if tail.text:
            kept[-1] = tail
            break
        kept.pop()

    return tuple(kept)


def split_paragraphs(text: str, runs: StorageRuns) -> list[Paragraph]:
    """Split a TSWP text run into labelled paragraphs of formatted runs.

    Apple separates paragraphs with newlines and pads empty ones, so blank
    results are dropped rather than emitted as empty text items. Every run table
    is keyed by character index into ``text``, and each entry stays in force
    until the next one begins.

    Args:
        text: The concatenated text of the storage.
        runs: The storage's run tables.

    Returns:
        The non-empty paragraphs, each labelled and carrying its runs.
    """
    paragraphs: list[Paragraph] = []
    offset = 0

    for line in text.split("\n"):
        pieces = runs_for(line, offset, runs)
        if pieces:
            label, level = label_for_style(value_at(runs.styles, offset))
            paragraphs.append(
                Paragraph(pieces, label, level, list_label_at(runs, offset))
            )
        offset += len(line) + 1  # + 1 for the newline that split consumed

    return paragraphs


def list_label_at(runs: StorageRuns, offset: int) -> ListLabel | None:
    """Return how the paragraph starting at ``offset`` is labelled as a list item."""
    style = value_at(runs.lists, offset)
    if style is None:
        return None
    return style.label(value_at(runs.depths, offset) or 0)


def runs_for(line: str, start: int, runs: StorageRuns) -> tuple[Run, ...]:
    """Cut one line into runs at the style and link boundaries inside it.

    Args:
        line: One paragraph of the storage's text.
        start: The paragraph's character index into the storage.
        runs: The storage's run tables.

    Returns:
        The paragraph's runs, trimmed, in document order.
    """
    if not runs.characters and not runs.links:
        return trim([Run(clean(line), None)])

    # Boundaries are absolute character indices; keep the ones inside this line.
    inside = {
        index
        for table in (runs.characters, runs.links)
        for index, _ in table
        if start < index < start + len(line)
    }
    boundaries = [start, *sorted(inside)]
    pieces: list[Run] = []

    for position, begin in enumerate(boundaries):
        end = (
            boundaries[position + 1]
            if position + 1 < len(boundaries)
            else start + len(line)
        )
        text = clean(line[begin - start : end - start])
        if text:
            pieces.append(
                Run(
                    text,
                    value_at(runs.characters, begin),
                    value_at(runs.links, begin),
                )
            )

    return trim(pieces)


def value_at(table: list[tuple[int, _T]], index: int) -> _T | None:
    """Return the value a run table puts in force at ``index``.

    Args:
        table: Character index and value pairs, in document order.
        index: The character index to look up.

    Returns:
        The value of the last entry at or before ``index``, or None when the
        table starts after it.
    """
    current: _T | None = None
    for position, value in table:
        if position > index:
            break
        current = value
    return current


def label_for_style(style_name: str | None) -> tuple[DocItemLabel, int | None]:
    """Map an iWork paragraph style name onto a Docling label.

    Pages names its built-in styles the same way in both container generations
    ("Title", "Heading 1", "Subheading", "Body"), so one mapping serves the IWA
    and XML readers alike. Custom styles are unknown to us and stay body text.

    Args:
        style_name: The paragraph style name, or None when the run inherits one.

    Returns:
        The label to use, and the heading level when the label is a section
        header.
    """
    if not style_name:
        return DocItemLabel.TEXT, None

    name = style_name.strip()
    lowered = name.casefold()

    if lowered == "title":
        return DocItemLabel.TITLE, None
    if lowered in {"subtitle", "subheading"}:
        return DocItemLabel.SECTION_HEADER, 2

    match = HEADING_PATTERN.match(name)
    if match:
        # A bare "Heading" is the top level: Pages' Layout template pairs it
        # with "Subheading" rather than numbering them.
        level = int(match.group(1)) if match.group(1) else 1
        return DocItemLabel.SECTION_HEADER, min(level, 6)

    return DocItemLabel.TEXT, None


def authored(author: str | None, text: str) -> str:
    """Prefix a comment with its author, the way the Word backend renders one."""
    return f"[author: {author}]: {text}" if author else text


def unique_paragraphs(paragraphs: list[Paragraph]) -> list[Paragraph]:
    """Drop repeats, keeping the first of each, without reordering."""
    seen: set[str] = set()
    unique: list[Paragraph] = []
    for paragraph in paragraphs:
        if paragraph.text in seen:
            continue
        seen.add(paragraph.text)
        unique.append(paragraph)
    return unique


def build_formatting(active: set[str], script: Script | None) -> Formatting | None:
    """Build a ``Formatting`` from the character properties in force.

    Args:
        active: The names of the boolean properties that are set.
        script: The script setting, or None for ordinary text.

    Returns:
        The formatting, or None when the style says nothing Docling records.
    """
    if not active and script is None:
        return None
    return Formatting(
        bold="bold" in active,
        italic="italic" in active,
        underline="underline" in active,
        strikethrough="strikethrough" in active,
        script=script or Script.BASELINE,
    )
