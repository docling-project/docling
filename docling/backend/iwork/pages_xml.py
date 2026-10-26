# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Reader for the ``index.xml`` of an iWork '09 (and earlier) Pages document.

Pages wrote a plain XML tree before 2013, in the ``sf`` namespace, and the
content the modern container keeps in an object graph is spelled out in elements
and attributes instead. Page furniture and comments each carry their own
``sf:text-body``, so the body walk prunes them and they are read separately.

Only the elements Pages adds to that tree live here; the ``sf`` vocabulary
underneath them is shared with the other iWork apps and is read by
:mod:`docling.backend.iwork.legacy`.
"""

import zipfile
from xml.etree.ElementTree import Element

from docling_core.types.doc import Formatting

from docling.backend.iwork.content import (
    Block,
    Comment,
    Content,
    Paragraph,
    label_for_style,
    unique_paragraphs,
)
from docling.backend.iwork.legacy import (
    SF_ATTR_NAME,
    SF_ATTR_STYLE,
    SF_CHARACTER_STYLE,
    SF_MEDIA_ELEMENTS,
    SF_NAMESPACE,
    SF_PARAGRAPH,
    SF_PARAGRAPH_STYLE,
    SF_TABULAR_MODEL,
    SFA_ATTR_ID,
    legacy_formatting,
    legacy_list_label,
    legacy_list_styles,
    legacy_picture,
    legacy_runs,
    legacy_styles,
    legacy_table,
    parse_index,
)

PAGES_KIND = "Pages"
"""What Pages calls its documents, for the error messages of a shared reader."""

SF_HEADER = f"{{{SF_NAMESPACE}}}header"

SF_FOOTER = f"{{{SF_NAMESPACE}}}footer"

SF_FOOTNOTES = f"{{{SF_NAMESPACE}}}footnotes"

SF_ANNOTATIONS = f"{{{SF_NAMESPACE}}}annotations"

SF_FURNITURE = frozenset({SF_HEADER, SF_FOOTER, SF_FOOTNOTES, SF_ANNOTATIONS})
"""Elements whose paragraphs are not body content.

Each carries its own ``sf:text-body``, so they have to be pruned from the body
walk by element rather than by looking for the document's body, and read
separately afterwards.
"""

SF_ANNOTATION = f"{{{SF_NAMESPACE}}}annotation"

SF_ANNOTATION_FIELD = f"{{{SF_NAMESPACE}}}annotation-field"

SF_ATTR_TARGET = f"{{{SF_NAMESPACE}}}target"
"""The iWork '09 vocabulary for comments.

An ``sf:annotation`` names the ``sf:annotation-field`` it targets, and that
field wraps the stretch of body text being commented on.
"""


def read_content(
    archive: zipfile.ZipFile,
    member: str,
    max_total_bytes: int,
    document_hash: str,
) -> Content:
    """Read the content of an iWork '09 document out of its ``index.xml``.

    Args:
        archive: The open ``.pages`` container, which also holds any image data.
        member: The name of its index member.
        max_total_bytes: The largest index this is willing to decompress to.
        document_hash: The document's hash, for error messages.

    Returns:
        Everything the document holds.

    Raises:
        DocumentLoadError: If the index cannot be decompressed or parsed.
    """
    root = parse_index(archive, member, max_total_bytes, document_hash, PAGES_KIND)
    style_names = legacy_styles(
        root, SF_PARAGRAPH_STYLE, lambda element: element.get(SF_ATTR_NAME)
    )
    character_styles = legacy_styles(root, SF_CHARACTER_STYLE, legacy_formatting)

    list_styles = legacy_list_styles(root)

    blocks: list[Block] = []
    for element in iter_body_elements(root):
        if element.tag == SF_TABULAR_MODEL:
            table = legacy_table(element)
            if table is not None:
                blocks.append(table)
            continue
        if element.tag in SF_MEDIA_ELEMENTS:
            picture = legacy_picture(element, archive)
            if picture is not None:
                blocks.append(picture)
            continue

        runs = legacy_runs(element, character_styles)
        if not runs:
            continue
        style = element.get(SF_ATTR_STYLE)
        label, level = label_for_style(style_names.get(style))
        anchors = tuple(
            field.get(SFA_ATTR_ID) or "" for field in element.iter(SF_ANNOTATION_FIELD)
        )
        blocks.append(
            Paragraph(
                runs,
                label,
                level,
                legacy_list_label(element, list_styles),
                tuple(anchor for anchor in anchors if anchor),
            )
        )

    def furniture(tag: str) -> list[Paragraph]:
        return legacy_furniture(root, tag, style_names, character_styles)

    return Content(
        blocks=blocks,
        headers=furniture(SF_HEADER),
        footers=furniture(SF_FOOTER),
        footnotes=furniture(SF_FOOTNOTES),
        comments=legacy_comments(root),
    )


def iter_body_elements(root: Element) -> list[Element]:
    """Collect the body content of an '09 document, skipping page furniture.

    Headers, footers and footnotes each hold their own ``sf:text-body``, so a
    plain ``root.iter()`` would pull their paragraphs into the body flow. They
    are pruned instead, which matches the IWA reader: it follows
    ``TP.DocumentArchive`` to the body storage and never sees them.

    A table and an image are not descended into once found, so the paragraphs
    inside a table cell stay in the table rather than reappearing as body text.

    Args:
        root: The parsed ``index.xml`` root element.

    Returns:
        The paragraph, table and image elements of the body, in document order.
    """
    elements: list[Element] = []
    # Explicit stack, for the same reason the text walk uses one: nesting depth
    # is attacker-controlled.
    stack: list[Element] = [root]

    while stack:
        node = stack.pop()
        if node.tag == SF_PARAGRAPH or node.tag == SF_TABULAR_MODEL:
            elements.append(node)
            continue
        if node.tag in SF_MEDIA_ELEMENTS:
            elements.append(node)
            continue
        for child in reversed(list(node)):
            if child.tag not in SF_FURNITURE:
                stack.append(child)

    return elements


def legacy_furniture(
    root: Element,
    tag: str,
    style_names: dict[str | None, str | None],
    character_styles: dict[str | None, Formatting | None],
) -> list[Paragraph]:
    """Read the paragraphs of one kind of '09 page furniture.

    Pages writes a first-page, an even-page and an odd-page variant of every
    header and footer whether or not the author filled them in, so identical
    text is emitted once.

    Args:
        root: The parsed ``index.xml`` root element.
        tag: The furniture element to collect, one of :data:`SF_FURNITURE`.
        style_names: Paragraph style names, keyed by style identifier.
        character_styles: Character style formatting, keyed by style identifier.

    Returns:
        The furniture's non-empty paragraphs, in document order.
    """
    paragraphs: list[Paragraph] = []
    for element in root.iter(tag):
        for para in element.iter(SF_PARAGRAPH):
            runs = legacy_runs(para, character_styles)
            if not runs:
                continue
            label, level = label_for_style(style_names.get(para.get(SF_ATTR_STYLE)))
            paragraphs.append(Paragraph(runs, label, level))
    return unique_paragraphs(paragraphs)


def legacy_comments(root: Element) -> list[Comment]:
    """Read the comments of an '09 document, with the text each one annotates.

    An ``sf:annotation`` holds its text in a storage of its own and names the
    ``sf:annotation-field`` in the body that it targets.

    Args:
        root: The parsed ``index.xml`` root element.

    Returns:
        One comment per annotation, in document order.
    """
    comments: list[Comment] = []
    for annotation in root.iter(SF_ANNOTATION):
        text = " ".join(
            "".join(run.text for run in legacy_runs(para, {}))
            for para in annotation.iter(SF_PARAGRAPH)
        ).strip()
        if text:
            comments.append(Comment(text, annotation.get(SF_ATTR_TARGET) or ""))
    return comments
