# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Reader for the ``index.xml`` of an iWork '09 (and earlier) document.

Pages wrote a plain XML tree before 2013, in the ``sf`` namespace, and the
content the modern container keeps in an object graph is spelled out in elements
and attributes instead. Page furniture and comments each carry their own
``sf:text-body``, so the body walk prunes them and they are read separately.
"""

import logging
import zipfile
from collections.abc import Callable
from typing import TypeVar
from xml.etree.ElementTree import Element

from docling_core.types.doc import (
    Formatting,
    Script,
    TableCell,
    TableData,
)

from docling.backend.iwork.content import (
    LABEL_TYPE_NONE,
    LABEL_TYPE_NUMBER,
    LABEL_TYPE_STRING,
    SCRIPTS,
    Block,
    Comment,
    Content,
    ListLabel,
    ListStyle,
    Paragraph,
    Picture,
    Run,
    build_formatting,
    clean,
    label_for_style,
    trim,
    unique_paragraphs,
)

_log = logging.getLogger(__name__)

_T = TypeVar("_T")

# An index.xml.gz can expand enormously relative to its stored size, so the
# legacy path decompresses incrementally against this ceiling rather than
# trusting the member size that max_total_bytes is computed from.
MAX_LEGACY_XML_BYTES = 100 * 1024 * 1024

SF_NAMESPACE = "http://developer.apple.com/namespaces/sf"

SF_PARAGRAPH = f"{{{SF_NAMESPACE}}}p"

# iWork '09 placeholder text. It is what the template shows before the author
# types anything, so it must never be emitted as document content.
SF_GHOST_TEXT = f"{{{SF_NAMESPACE}}}ghost-text"

SF_PARAGRAPH_STYLE = f"{{{SF_NAMESPACE}}}paragraphstyle"

SFA_NAMESPACE = "http://developer.apple.com/namespaces/sfa"

SF_ATTR_IDENT = f"{{{SF_NAMESPACE}}}ident"

SF_ATTR_NAME = f"{{{SF_NAMESPACE}}}name"

SF_ATTR_STYLE = f"{{{SF_NAMESPACE}}}style"

SF_ATTR_NUMCOLS = f"{{{SF_NAMESPACE}}}numcols"

SF_ATTR_NUMROWS = f"{{{SF_NAMESPACE}}}numrows"

SF_ATTR_HEADER_ROWS = f"{{{SF_NAMESPACE}}}num-header-rows"

SFA_ATTR_STRING = f"{{{SFA_NAMESPACE}}}s"

SF_TABULAR_MODEL = f"{{{SF_NAMESPACE}}}tabular-model"

SF_GRID = f"{{{SF_NAMESPACE}}}grid"

SF_CELL_TEXT = f"{{{SF_NAMESPACE}}}ct"

SF_SPAN = f"{{{SF_NAMESPACE}}}span"

SF_CHARACTER_STYLE = f"{{{SF_NAMESPACE}}}characterstyle"

SFA_ATTR_NUMBER = "{http://developer.apple.com/namespaces/sfa}number"

SF_MEDIA = f"{{{SF_NAMESPACE}}}media"

SF_IMAGE = f"{{{SF_NAMESPACE}}}image"

SF_DATA = f"{{{SF_NAMESPACE}}}data"

SF_ATTR_PATH = "path"

SF_MEDIA_ELEMENTS = frozenset({SF_MEDIA, SF_IMAGE})
"""Elements that place an image in an iWork '09 document.

Both wrap an ``sf:data`` naming the container member that holds the bytes, and
neither is descended into once found: the renditions Pages keeps below them all
name the same picture.
"""

SF_LIST_STYLE = f"{{{SF_NAMESPACE}}}liststyle"

SF_LIST_LABEL_TYPE = f"{{{SF_NAMESPACE}}}list-label-typeinfo"

SF_TEXT_LABEL = f"{{{SF_NAMESPACE}}}text-label"

SF_ATTR_TYPE = f"{{{SF_NAMESPACE}}}type"

SF_ATTR_FORMAT = f"{{{SF_NAMESPACE}}}format"

SF_ATTR_LIST_LEVEL = f"{{{SF_NAMESPACE}}}list-level"

SF_ATTR_LIST_STYLE = f"{{{SF_NAMESPACE}}}list-style"
"""The iWork '09 vocabulary for lists.

An ``sf:liststyle`` holds one ``sf:list-label-typeinfo`` per nesting level, and
a paragraph joins the list by naming the style and its own ``sf:list-level``,
which counts from one.
"""

SF_LABEL_TYPE_NONE = "none"
"""``sf:list-label-typeinfo`` type that leaves a level unlabelled."""

SF_BULLET_LABEL_TYPES = frozenset({"bullet", "image", "string", "text"})
"""``sf:text-label`` types that draw a fixed marker rather than a number."""

SF_SUPERSCRIPT = f"{{{SF_NAMESPACE}}}superscript"
"""Property-map entry of an '09 character style holding its script setting.

Its number matches the modern ``SuperscriptType``: one raises the text and two
lowers it.
"""

SF_LINK = f"{{{SF_NAMESPACE}}}link"

SF_ATTR_HREF = "href"
"""The iWork '09 vocabulary for hyperlinks.

``href`` is one of the few attributes iWork writes unqualified, so it is read
through :func:`sf_attr` rather than by namespaced name alone.
"""

SF_PROPERTY_LABELS = {
    f"{{{SF_NAMESPACE}}}bold": "bold",
    f"{{{SF_NAMESPACE}}}italic": "italic",
    f"{{{SF_NAMESPACE}}}underline": "underline",
    f"{{{SF_NAMESPACE}}}strikethru": "strikethrough",
}
"""Property-map entries of an iWork '09 character style, as ``Formatting`` names."""

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

SFA_ATTR_ID = f"{{{SFA_NAMESPACE}}}ID"
"""The iWork '09 vocabulary for comments.

An ``sf:annotation`` names the ``sf:annotation-field`` it targets, and that
field wraps the stretch of body text being commented on.
"""


def read_content(root: Element, archive: zipfile.ZipFile) -> Content:
    """Read the content of an iWork '09 document out of its parsed ``index.xml``.

    Args:
        root: The parsed ``index.xml`` root element.
        archive: The open ``.pages`` container, which holds any image data.

    Returns:
        Everything the document holds.
    """
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


def legacy_table(model: Element) -> TableData | None:
    """Build table data from one ``sf:tabular-model`` of an '09 document.

    Cells are stored flat in ``sf:datasource``, in row-major order, so the grid
    dimensions on ``sf:grid`` are what give them their positions.

    Args:
        model: An ``sf:tabular-model`` element.

    Returns:
        The table, or None when its grid or its cells are missing.
    """
    grid = next(iter(model.iter(SF_GRID)), None)
    if grid is None:
        return None

    num_cols = int_attr(grid, SF_ATTR_NUMCOLS)
    num_rows = int_attr(grid, SF_ATTR_NUMROWS)
    header_rows = int_attr(model, SF_ATTR_HEADER_ROWS) or 0
    if not num_cols or not num_rows:
        return None

    values = [
        clean(cell.get(SFA_ATTR_STRING) or "".join(cell.itertext())).strip()
        for cell in model.iter(SF_CELL_TEXT)
    ]
    if not values:
        return None

    cells: list[TableCell] = []
    for index, text in enumerate(values[: num_cols * num_rows]):
        row, col = divmod(index, num_cols)
        cells.append(
            TableCell(
                text=text,
                start_row_offset_idx=row,
                end_row_offset_idx=row + 1,
                start_col_offset_idx=col,
                end_col_offset_idx=col + 1,
                column_header=row < header_rows,
            )
        )

    return TableData(num_rows=num_rows, num_cols=num_cols, table_cells=cells)


def legacy_picture(media: Element, archive: zipfile.ZipFile) -> Picture | None:
    """Read an '09 image, whose bytes are a member of the container.

    Args:
        media: An ``sf:media`` or ``sf:image`` element.
        archive: The open ``.pages`` container.

    Returns:
        The picture, or None when the element names no stored data.
    """
    for data in media.iter(SF_DATA):
        path = sf_attr(data, SF_ATTR_PATH)
        if not path:
            continue
        try:
            return Picture(archive.read(path), path)
        except KeyError:
            _log.debug("Pages image data member %s is missing", path)
            return Picture(None, path)
    return None


def sf_attr(element: Element, name: str) -> str | None:
    """Read an attribute iWork '09 may or may not have qualified.

    Most attributes carry the ``sf`` namespace, but a few — ``href`` on
    ``sf:link`` among them — are written unqualified, and which spelling a
    document uses varies with the release that wrote it.

    Args:
        element: The element to read.
        name: The local name of the attribute.

    Returns:
        The attribute's value under either spelling, or None.
    """
    return element.get(f"{{{SF_NAMESPACE}}}{name}") or element.get(name)


def int_attr(element: Element, name: str) -> int | None:
    """Read an integer attribute, tolerating absent or malformed values."""
    raw = element.get(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


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


def legacy_runs(
    paragraph: Element, character_styles: dict[str | None, Formatting | None]
) -> tuple[Run, ...]:
    """Build the runs of an iWork '09 paragraph.

    ``sf:span`` carries the character style, so the paragraph is walked span by
    span rather than flattened. Template placeholder text is skipped, as
    ``itertext()`` would otherwise emit what the template displays before the
    author types anything.

    Walked with an explicit stack: nesting depth is attacker-controlled, and a
    recursive walk exhausts the interpreter stack on a deeply nested document.

    Args:
        paragraph: An ``sf:p`` element.
        character_styles: Character style formatting, keyed by style identifier.

    Returns:
        The paragraph's non-empty runs, in document order.
    """
    runs: list[Run] = []
    # (element, formatting in force, link in force, whether this emits the tail)
    stack: list[tuple[Element, Formatting | None, str | None, bool]] = [
        (paragraph, None, None, False)
    ]

    while stack:
        element, formatting, link, want_tail = stack.pop()

        if want_tail:
            if element.tail:
                runs.append(Run(clean(element.tail), formatting, link))
            continue

        if element.text:
            runs.append(Run(clean(element.text), formatting, link))

        # Push in reverse so children pop in document order. A child's tail sits
        # outside it, so it keeps the parent's formatting.
        for child in reversed(list(element)):
            stack.append((child, formatting, link, True))
            if child.tag == SF_GHOST_TEXT:
                continue
            inherited = formatting
            if child.tag == SF_SPAN:
                inherited = character_styles.get(child.get(SF_ATTR_STYLE), formatting)
            nested = link
            if child.tag == SF_LINK:
                nested = sf_attr(child, SF_ATTR_HREF) or link
            stack.append((child, inherited, nested, False))

    return trim(runs)


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


def legacy_styles(
    root: Element, tag: str, decode: Callable[[Element], _T]
) -> dict[str | None, _T]:
    """Read one kind of iWork '09 style, keyed by every name it answers to.

    A paragraph or a span names its style through ``sf:style``, and what it puts
    there is sometimes the style's ``sf:ident`` and sometimes its ``sfa:ID``.
    Both are indexed so a reference resolves either way; a style that carries
    neither cannot be referenced at all and is skipped.

    Args:
        root: The parsed ``index.xml`` root element.
        tag: The style element to collect.
        decode: Reads one style element into the value to key.

    Returns:
        The decoded styles, keyed by identifier.
    """
    styles: dict[str | None, _T] = {}
    for element in root.iter(tag):
        keys = [element.get(SF_ATTR_IDENT), element.get(SFA_ATTR_ID)]
        if not any(keys):
            continue
        value = decode(element)
        for key in keys:
            if key:
                styles.setdefault(key, value)
    return styles


def legacy_list_styles(root: Element) -> dict[str, ListStyle]:
    """Read the ``sf:liststyle`` definitions of an '09 document by identifier.

    Args:
        root: The parsed ``index.xml`` root element.

    Returns:
        The label ladder of every named list style, keyed by its identifier.
    """
    styles: dict[str, ListStyle] = {}

    for element in root.iter(SF_LIST_STYLE):
        keys = [element.get(SF_ATTR_IDENT), element.get(SFA_ATTR_ID)]
        if not any(key and key not in styles for key in keys):
            continue

        label_types: list[int] = []
        strings: list[str] = []
        for level in element.iter(SF_LIST_LABEL_TYPE):
            if level.get(SF_ATTR_TYPE) == SF_LABEL_TYPE_NONE:
                label_types.append(LABEL_TYPE_NONE)
                strings.append("")
                continue
            text_label = next(iter(level.iter(SF_TEXT_LABEL)), None)
            kind = text_label.get(SF_ATTR_TYPE) if text_label is not None else None
            if kind is not None and kind not in SF_BULLET_LABEL_TYPES:
                # Anything else names a numbering sequence: decimal, upper-roman,
                # lower-alpha and the rest, which Pages counts rather than draws.
                label_types.append(LABEL_TYPE_NUMBER)
                strings.append("")
                continue
            label_types.append(LABEL_TYPE_STRING)
            strings.append(
                (text_label.get(SF_ATTR_FORMAT) or "") if text_label is not None else ""
            )

        style = ListStyle(tuple(label_types), tuple(strings))
        for key in keys:
            if key:
                styles.setdefault(key, style)

    return styles


def legacy_list_label(
    paragraph: Element, list_styles: dict[str, ListStyle]
) -> ListLabel | None:
    """Return how an '09 paragraph is labelled as a list item, if it is one.

    Args:
        paragraph: An ``sf:p`` element.
        list_styles: The document's list styles, keyed by identifier.

    Returns:
        The label, or None when the paragraph names no list style or the style
        leaves its level unlabelled.
    """
    style = list_styles.get(paragraph.get(SF_ATTR_LIST_STYLE) or "")
    if style is None:
        return None
    # sf:list-level counts from one, unlike the depth the IWA reader works in.
    level = int_attr(paragraph, SF_ATTR_LIST_LEVEL) or 1
    return style.label(max(level - 1, 0))


def legacy_formatting(style: Element) -> Formatting | None:
    """Read an iWork '09 character style's property map as a ``Formatting``."""
    active: set[str] = set()
    script: Script | None = None

    for element in style.iter():
        number = next(
            (
                child.get(SFA_ATTR_NUMBER)
                for child in element
                if child.get(SFA_ATTR_NUMBER) is not None
            ),
            None,
        )
        if number in (None, "0"):
            continue

        label = SF_PROPERTY_LABELS.get(element.tag)
        if label is not None:
            active.add(label)
        elif element.tag == SF_SUPERSCRIPT and number is not None:
            script = SCRIPTS.get(as_int(number))

    return build_formatting(active, script)


def as_int(number: str) -> int:
    """Read an iWork property number, which may be written as a float."""
    try:
        return int(float(number))
    except ValueError:
        return 0
