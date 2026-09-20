# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Reader for the ``index.apxl`` of an iWork '09 (and earlier) Keynote document.

Keynote wrote a plain XML tree before 2013, in the ``key`` namespace over the
``sf`` one every iWork app shared, and it describes the same presentation the
modern container keeps in an object graph: a slide list, and a slide holding a
title placeholder, a body placeholder, presenter notes and a page whose layers
hold everything else.

Only the elements Keynote adds live here; the ``sf`` vocabulary underneath them
— runs, character and list styles, tables, placed images — is shared with the
other iWork apps and is read by :mod:`docling.backend.iwork.legacy`.
"""

import zipfile
from typing import NamedTuple
from xml.etree.ElementTree import Element

from docling_core.types.doc import DocItemLabel, Formatting

from docling.backend.iwork.content import (
    Block,
    Comment,
    Geometry,
    ListStyle,
    Paragraph,
)
from docling.backend.iwork.keynote_content import (
    DEFAULT_SLIDE_HEIGHT,
    DEFAULT_SLIDE_WIDTH,
    Presentation,
    Slide,
    reading_order,
)
from docling.backend.iwork.legacy import (
    SF_CHARACTER_STYLE,
    SF_MEDIA_ELEMENTS,
    SF_NAMESPACE,
    SF_PARAGRAPH,
    SF_TABULAR_MODEL,
    SFA_ATTR_H,
    SFA_ATTR_ID,
    SFA_ATTR_IDREF,
    SFA_ATTR_W,
    float_attr,
    legacy_formatting,
    legacy_geometry,
    legacy_inherited_lists,
    legacy_list_label,
    legacy_list_styles,
    legacy_picture,
    legacy_runs,
    legacy_styles,
    legacy_table,
    parse_index,
)

KEYNOTE_KIND = "Keynote"
"""What Keynote calls its documents, for the error messages of a shared reader."""

KEY_NAMESPACE = "http://developer.apple.com/namespaces/keynote2"

KEY_SIZE = f"{{{KEY_NAMESPACE}}}size"

KEY_SLIDE_LIST = f"{{{KEY_NAMESPACE}}}slide-list"

KEY_SLIDE = f"{{{KEY_NAMESPACE}}}slide"

KEY_PAGE = f"{{{KEY_NAMESPACE}}}page"

KEY_NOTES = f"{{{KEY_NAMESPACE}}}notes"
"""The iWork '09 vocabulary for a presentation.

The slide list holds the slides in presentation order, and each of them holds
its page — where everything placed on it lives — and its presenter notes. The
theme keeps its master slides elsewhere under a tag of its own, so reading the
slide list is enough to leave them out.
"""

KEY_TITLE_PLACEHOLDER = f"{{{KEY_NAMESPACE}}}title-placeholder"

KEY_BODY_PLACEHOLDER = f"{{{KEY_NAMESPACE}}}body-placeholder"
"""The placeholders a slide reserves for its title and its body.

Both are written whether or not the author typed into them, and neither is
labelled by its paragraph style: Keynote names its theme styles in the language
the theme was written in, so the placeholder is what says which is which.
"""

PLACEHOLDER_LABELS = {
    KEY_TITLE_PLACEHOLDER: DocItemLabel.TITLE,
    KEY_BODY_PLACEHOLDER: DocItemLabel.TEXT,
}
"""What the text of each placeholder is labelled as."""

SF_LAYERS = f"{{{SF_NAMESPACE}}}layers"

SF_LAYER = f"{{{SF_NAMESPACE}}}layer"

SF_DRAWABLES = f"{{{SF_NAMESPACE}}}drawables"
"""Where a slide's page keeps what is placed on it.

A page has several layers, and one of them is an ``sf:proxy-master-layer``
pointing at the master slide's. That one is deliberately not followed: what the
master draws belongs to every slide using it, not to this one.
"""

SF_TITLE_PLACEHOLDER_REF = f"{{{SF_NAMESPACE}}}title-placeholder-ref"

SF_BODY_PLACEHOLDER_REF = f"{{{SF_NAMESPACE}}}body-placeholder-ref"

PLACEHOLDER_REFS = frozenset({SF_TITLE_PLACEHOLDER_REF, SF_BODY_PLACEHOLDER_REF})
"""How the drawables of a page refer back to the slide's own placeholders.

A reference carries no geometry of its own, so the placeholder it names is what
is placed and what is positioned.
"""

SF_SHAPE = f"{{{SF_NAMESPACE}}}shape"
"""A text box placed on a slide, which holds its text the way a placeholder does."""

SF_STICKY_NOTE = f"{{{SF_NAMESPACE}}}sticky-note"
"""A comment, which Keynote draws as a note stuck to the slide.

It sits among the drawables rather than annotating a stretch of text the way a
Pages comment does, and it records no author, so only its text is recovered.
"""

SF_TABULAR_INFO = f"{{{SF_NAMESPACE}}}tabular-info"
"""A table placed on a slide, wrapping the ``sf:tabular-model`` that holds it."""


class Styles(NamedTuple):
    """The style lookups a presentation's text is read against.

    They are built once for the whole document and carried down to every
    paragraph, since a slide's text is styled from the theme rather than from
    anything the slide itself holds.
    """

    characters: dict[str | None, Formatting | None]
    lists: dict[str, ListStyle]
    inherited: dict[str, str]


def read_content(
    archive: zipfile.ZipFile,
    member: str,
    max_total_bytes: int,
    document_hash: str,
) -> Presentation:
    """Read an iWork '09 presentation out of its ``index.apxl``.

    Args:
        archive: The open ``.key`` container, which also holds any image data.
        member: The name of its index member.
        max_total_bytes: The largest index this is willing to decompress to.
        document_hash: The document's hash, for error messages.

    Returns:
        Every slide the presentation holds, and the size they are laid out at.

    Raises:
        DocumentLoadError: If the index cannot be decompressed or parsed.
    """
    root = parse_index(archive, member, max_total_bytes, document_hash, KEYNOTE_KIND)
    styles = Styles(
        characters=legacy_styles(root, SF_CHARACTER_STYLE, legacy_formatting),
        lists=legacy_list_styles(root),
        inherited=legacy_inherited_lists(root),
    )

    slides = [read_slide(slide, archive, styles) for slide in iter_slides(root)]
    width, height = slide_size(root)
    return Presentation(slides=slides, width=width, height=height)


def iter_slides(root: Element) -> list[Element]:
    """Collect the presentation's slides, in presentation order."""
    slides: list[Element] = []
    for child in root:
        if child.tag == KEY_SLIDE_LIST:
            slides.extend(slide for slide in child if slide.tag == KEY_SLIDE)
    return slides


def slide_size(root: Element) -> tuple[float, float]:
    """Read the size the presentation's slides are laid out at, in points."""
    for child in root:
        if child.tag != KEY_SIZE:
            continue
        width = float_attr(child, SFA_ATTR_W)
        height = float_attr(child, SFA_ATTR_H)
        if width and height and width > 0 and height > 0:
            return width, height
    return DEFAULT_SLIDE_WIDTH, DEFAULT_SLIDE_HEIGHT


def read_slide(slide: Element, archive: zipfile.ZipFile, styles: Styles) -> Slide:
    """Read one slide: what is placed on it, its comments and its notes.

    Args:
        slide: A ``key:slide`` element.
        archive: The open ``.key`` container, for any image data.
        styles: The document's style lookups.

    Returns:
        The slide.
    """
    placed = slide_drawables(slide)

    blocks: list[Block] = []
    comments: list[Comment] = []
    for position in reading_order([geometry for _, _, geometry in placed]):
        element, label, _ = placed[position]
        found, said = drawable_blocks(element, label, archive, styles)
        blocks.extend(found)
        comments.extend(said)

    return Slide(blocks=blocks, notes=slide_notes(slide, styles), comments=comments)


def slide_drawables(
    slide: Element,
) -> list[tuple[Element, DocItemLabel | None, Geometry | None]]:
    """Collect what is placed on a slide, with the label and position of each.

    The page's drawables refer back to the slide's placeholders rather than
    holding them, and a slide sometimes leaves a placeholder out of that list
    altogether, so the placeholders are added afterwards as well — the same
    merge the 2013 reader makes between a slide's two accounts of its drawables.

    Args:
        slide: A ``key:slide`` element.

    Returns:
        Each drawable, the label its text takes if it is a placeholder, and
        where it sits, in the order the slide stores them.
    """
    placeholders = slide_placeholders(slide)
    placed: list[tuple[Element, DocItemLabel | None, Geometry | None]] = []
    seen: set[int] = set()

    def add(element: Element, label: DocItemLabel | None) -> None:
        if id(element) in seen:
            return
        seen.add(id(element))
        placed.append((element, label, legacy_geometry(element)))

    for drawable in iter_drawables(slide):
        if drawable.tag in PLACEHOLDER_REFS:
            target = placeholders.get(drawable.get(SFA_ATTR_IDREF) or "")
            if target is not None:
                add(*target)
            continue
        add(drawable, None)

    for element, label in placeholders.values():
        add(element, label)

    return placed


def slide_placeholders(slide: Element) -> dict[str, tuple[Element, DocItemLabel]]:
    """Map a slide's title and body placeholders by the identifier they answer to."""
    found: dict[str, tuple[Element, DocItemLabel]] = {}
    for child in slide:
        label = PLACEHOLDER_LABELS.get(child.tag)
        identifier = child.get(SFA_ATTR_ID)
        if label is not None and identifier:
            found[identifier] = (child, label)
    return found


def iter_drawables(slide: Element) -> list[Element]:
    """Collect the drawables of a slide's own layers, leaving the master's alone."""
    drawables: list[Element] = []
    for page in slide:
        if page.tag != KEY_PAGE:
            continue
        for layers in page:
            if layers.tag != SF_LAYERS:
                continue
            for layer in layers:
                if layer.tag != SF_LAYER:
                    continue
                for group in layer:
                    if group.tag == SF_DRAWABLES:
                        drawables.extend(group)
    return drawables


def drawable_blocks(
    element: Element,
    label: DocItemLabel | None,
    archive: zipfile.ZipFile,
    styles: Styles,
) -> tuple[list[Block], list[Comment]]:
    """Read whichever kind of drawable is placed on a slide.

    Args:
        element: The drawable to read.
        label: The label its text takes, or None for anything but a placeholder.
        archive: The open ``.key`` container, for any image data.
        styles: The document's style lookups.

    Returns:
        The blocks it contributes to the slide, and the comments it holds.
    """
    if element.tag == SF_STICKY_NOTE:
        text = sticky_note_text(element, styles)
        return [], [Comment(text, "")] if text else []

    if element.tag == SF_TABULAR_INFO:
        model = next(iter(element.iter(SF_TABULAR_MODEL)), None)
        table = legacy_table(model) if model is not None else None
        return ([table] if table is not None else []), []

    if element.tag in SF_MEDIA_ELEMENTS:
        picture = legacy_picture(element, archive)
        return ([picture] if picture is not None else []), []

    return list(element_paragraphs(element, label or DocItemLabel.TEXT, styles)), []


def element_paragraphs(
    element: Element, label: DocItemLabel, styles: Styles
) -> list[Paragraph]:
    """Read the paragraphs of a placeholder, a text box or the presenter notes.

    Args:
        element: The element holding the text.
        label: The label to give every paragraph of it.
        styles: The document's style lookups.

    Returns:
        The non-empty paragraphs, in document order.
    """
    paragraphs: list[Paragraph] = []
    for para in element.iter(SF_PARAGRAPH):
        runs = legacy_runs(para, styles.characters)
        if not runs:
            continue
        list_label = legacy_list_label(para, styles.lists, styles.inherited)
        paragraphs.append(Paragraph(runs, label, None, list_label))
    return paragraphs


def sticky_note_text(element: Element, styles: Styles) -> str:
    """Read the text of a sticky note, with its paragraphs run together."""
    return " ".join(
        "".join(run.text for run in legacy_runs(para, styles.characters))
        for para in element.iter(SF_PARAGRAPH)
    ).strip()


def slide_notes(slide: Element, styles: Styles) -> list[Paragraph]:
    """Read the presenter notes of one slide."""
    notes: list[Paragraph] = []
    for child in slide:
        if child.tag == KEY_NOTES:
            notes.extend(element_paragraphs(child, DocItemLabel.TEXT, styles))
    return notes
