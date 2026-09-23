# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Readers for the ``sf`` vocabulary an iWork '09 (and earlier) document is written in.

Before 2013 every iWork app wrote a plain XML tree rather than an object graph,
and the parts of that tree which describe content — paragraphs and their runs,
character and list styles, tables, placed images — are spelled the same way in a
Pages document and in a Keynote presentation. They are read once here, leaving
each app's module with only the elements its own namespace adds.
"""

import logging
import zipfile
import zlib
from collections.abc import Callable
from typing import TypeVar
from xml.etree.ElementTree import Element

import defusedxml.ElementTree as ET
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
    Geometry,
    ListLabel,
    ListStyle,
    Picture,
    Run,
    build_formatting,
    clean,
    trim,
)
from docling.exceptions import DocumentLoadError

_log = logging.getLogger(__name__)

_T = TypeVar("_T")


# An index.xml.gz can expand enormously relative to its stored size, so the
# legacy path decompresses incrementally against this ceiling rather than
# trusting the member size that max_total_bytes is computed from.
MAX_LEGACY_XML_BYTES = 100 * 1024 * 1024

SF_NAMESPACE = "http://developer.apple.com/namespaces/sf"

SF_PARAGRAPH = f"{{{SF_NAMESPACE}}}p"

SF_GHOST_TEXT = f"{{{SF_NAMESPACE}}}ghost-text"
SF_GHOST_TEXT_REF = f"{{{SF_NAMESPACE}}}ghost-text-ref"

SF_PLACEHOLDER_TEXT = frozenset({SF_GHOST_TEXT, SF_GHOST_TEXT_REF})
"""Elements holding iWork '09 placeholder text.

It is what the template shows before the author types anything, so it must never
be emitted as document content. A template defines each placeholder once as an
``sf:ghost-text`` and every later paragraph that reuses it holds an
``sf:ghost-text-ref``, which names the original by ``sfa:IDREF`` but carries its
own inline copy of the text — so both have to be pruned, not just the first.
"""

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

SF_DATASOURCE = f"{{{SF_NAMESPACE}}}datasource"

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

SF_GEOMETRY = f"{{{SF_NAMESPACE}}}geometry"

SF_POSITION = f"{{{SF_NAMESPACE}}}position"

SF_SIZE = f"{{{SF_NAMESPACE}}}size"

SFA_ATTR_X = f"{{{SFA_NAMESPACE}}}x"

SFA_ATTR_Y = f"{{{SFA_NAMESPACE}}}y"

SFA_ATTR_W = f"{{{SFA_NAMESPACE}}}w"

SFA_ATTR_H = f"{{{SFA_NAMESPACE}}}h"
"""The iWork '09 vocabulary for where a drawable sits.

An ``sf:geometry`` gives a position and a size, both in points, and the
natural size beside them is the drawable's own rather than the one it was
scaled to, so it is not read.
"""

SF_LIST_STYLE = f"{{{SF_NAMESPACE}}}liststyle"

SF_LIST_LABEL_TYPE = f"{{{SF_NAMESPACE}}}list-label-typeinfo"

SF_TEXT_LABEL = f"{{{SF_NAMESPACE}}}text-label"

SF_ATTR_TYPE = f"{{{SF_NAMESPACE}}}type"

SF_ATTR_FORMAT = f"{{{SF_NAMESPACE}}}format"

SF_ATTR_LIST_LEVEL = f"{{{SF_NAMESPACE}}}list-level"

SF_ATTR_LIST_STYLE = f"{{{SF_NAMESPACE}}}list-style"
"""The iWork '09 vocabulary for lists.

An ``sf:liststyle`` holds one ``sf:list-label-typeinfo`` per rung of a ladder
nine deep, and a paragraph joins the list by naming the style and the rung it
sits on. Rung zero is the unlabelled one ordinary body text sits on, so a
paragraph that names no level is not a list item.
"""

SF_ATTR_PARENT_IDENT = f"{{{SF_NAMESPACE}}}parent-ident"

SF_LIST_STYLE_PROPERTY = f"{{{SF_NAMESPACE}}}listStyle"

SF_LIST_STYLE_REF = f"{{{SF_NAMESPACE}}}liststyle-ref"
"""How a paragraph style hands a list style down to the paragraphs using it.

Pages names the list style on the paragraph itself, but Keynote leaves it to
the theme: a bulleted paragraph names an empty style whose ``sf:parent-ident``
leads, sometimes through several more, to the one carrying the list style in
its property map. Without following that chain a Keynote '09 deck loses every
bullet it has.
"""

MAX_STYLE_INHERITANCE = 8
"""How far to follow ``sf:parent-ident`` before giving up.

A real chain is two or three long. The bound is what keeps a document whose
styles inherit from each other in a circle from looping forever.
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

SFA_ATTR_ID = f"{{{SFA_NAMESPACE}}}ID"
"""The identifier iWork '09 gives an element it may refer back to elsewhere."""

SFA_ATTR_IDREF = f"{{{SFA_NAMESPACE}}}IDREF"
"""The identifier one iWork '09 element refers to another by."""


def parse_index(
    archive: zipfile.ZipFile,
    member: str,
    max_total_bytes: int,
    document_hash: str,
    kind: str,
) -> Element:
    """Decompress and parse the index of an iWork '09 document.

    Args:
        archive: The open container.
        member: The name of its index member.
        max_total_bytes: The largest index this is willing to decompress to.
        document_hash: The document's hash, for error messages.
        kind: What the app calls its documents, for error messages.

    Returns:
        The parsed root element.

    Raises:
        DocumentLoadError: If the member cannot be decompressed or parsed.
    """
    raw = archive.read(member)
    if member.endswith(".gz"):
        # max_total_bytes only counts the stored size of a gzipped member, so a
        # small index.xml.gz could otherwise expand without bound. Cap the
        # output instead of using gzip.decompress, which has no limit.
        limit = min(MAX_LEGACY_XML_BYTES, max_total_bytes)
        try:
            decompressor = zlib.decompressobj(wbits=31)
            raw = decompressor.decompress(raw, limit)
            if decompressor.unconsumed_tail:
                raise DocumentLoadError(
                    f"'{member}' in {kind} document with hash {document_hash} "
                    f"expands beyond the {limit} byte limit."
                )
        except zlib.error as exc:
            raise DocumentLoadError(
                f"Could not decompress '{member}' in {kind} document with hash "
                f"{document_hash}."
            ) from exc

    try:
        return ET.fromstring(raw)
    except Exception as exc:
        raise DocumentLoadError(
            f"Could not parse '{member}' in {kind} document with hash {document_hash}."
        ) from exc


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

    source = next(iter(grid.iter(SF_DATASOURCE)), None)
    if source is None:
        return None

    values = [legacy_cell_text(cell) for cell in source]
    if not any(values):
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


def legacy_cell_text(cell: Element) -> str:
    """Read one cell of an '09 table, which is empty unless it holds text.

    The datasource holds one element per cell of the grid, in row-major order,
    named for what the cell turned out to hold: ``sf:t`` for text, ``sf:n`` for
    a number, and so on. Only text is recovered — a number, a date or a formula
    result is left empty rather than guessed at from the stored value and the
    format beside it — but every cell still takes its place, or everything after
    the first non-text one shifts along a column.

    Args:
        cell: One child of an ``sf:datasource``.

    Returns:
        The cell's text, or an empty string when it holds none.
    """
    text = next(iter(cell.iter(SF_CELL_TEXT)), None)
    if text is None:
        return ""
    return clean(text.get(SFA_ATTR_STRING) or "".join(text.itertext())).strip()


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
            _log.debug("iWork image data member %s is missing", path)
            return Picture(None, path)
    return None


def legacy_geometry(element: Element) -> Geometry | None:
    """Read where an iWork '09 drawable sits, from its ``sf:geometry``.

    Args:
        element: The drawable to read.

    Returns:
        The geometry, or None when the drawable carries none — which is what
        an element that merely refers to a positioned one looks like.
    """
    geometry = next(iter(element.iter(SF_GEOMETRY)), None)
    if geometry is None:
        return None

    position = next(iter(geometry.iter(SF_POSITION)), None)
    if position is None:
        return None
    left = float_attr(position, SFA_ATTR_X)
    top = float_attr(position, SFA_ATTR_Y)
    if left is None or top is None:
        return None

    size = next(iter(geometry.iter(SF_SIZE)), None)
    width = float_attr(size, SFA_ATTR_W) if size is not None else None
    height = float_attr(size, SFA_ATTR_H) if size is not None else None
    return Geometry(left, top, width or 0.0, height or 0.0)


def float_attr(element: Element, name: str) -> float | None:
    """Read a measurement attribute, tolerating absent or malformed values."""
    raw = element.get(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
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
            if child.tag in SF_PLACEHOLDER_TEXT:
                continue
            inherited = formatting
            if child.tag == SF_SPAN:
                inherited = character_styles.get(child.get(SF_ATTR_STYLE), formatting)
            nested = link
            if child.tag == SF_LINK:
                nested = sf_attr(child, SF_ATTR_HREF) or link
            stack.append((child, inherited, nested, False))

    return trim(runs)


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


def legacy_inherited_lists(root: Element) -> dict[str, str]:
    """Map each paragraph style to the list style it ends up carrying.

    A style that does not carry one inherits whatever its parent carries, so
    the chains are walked once here and flattened into a single lookup rather
    than followed again for every paragraph.

    Args:
        root: The parsed index root element.

    Returns:
        The identifier of the list style in force, keyed by every name the
        paragraph style using it answers to.
    """
    styles: dict[str, Element] = {}
    for element in root.iter(SF_PARAGRAPH_STYLE):
        for key in (element.get(SF_ATTR_IDENT), element.get(SFA_ATTR_ID)):
            if key:
                styles.setdefault(key, element)

    resolved: dict[str, str] = {}
    for key, element in styles.items():
        current: Element | None = element
        for _ in range(MAX_STYLE_INHERITANCE):
            if current is None:
                break
            named = own_list_style(current)
            if named is not None:
                resolved[key] = named
                break
            parent = current.get(SF_ATTR_PARENT_IDENT)
            current = styles.get(parent) if parent else None
    return resolved


def own_list_style(style: Element) -> str | None:
    """Read the list style one paragraph style names, ignoring what it inherits."""
    for named in style.iter(SF_LIST_STYLE_PROPERTY):
        for reference in named.iter(SF_LIST_STYLE_REF):
            identifier = reference.get(SFA_ATTR_IDREF)
            if identifier:
                return identifier
    return None


def legacy_list_label(
    paragraph: Element,
    list_styles: dict[str, ListStyle],
    inherited: dict[str, str] | None = None,
) -> ListLabel | None:
    """Return how an '09 paragraph is labelled as a list item, if it is one.

    Args:
        paragraph: An ``sf:p`` element.
        list_styles: The document's list styles, keyed by identifier.
        inherited: The list style each paragraph style ends up carrying, for the
            apps that leave it off the paragraph. None means only what the
            paragraph names itself is considered.

    Returns:
        The label, or None when the paragraph reaches no list style or the style
        leaves its rung unlabelled.
    """
    named = paragraph.get(SF_ATTR_LIST_STYLE)
    if named is None and inherited is not None:
        named = inherited.get(paragraph.get(SF_ATTR_STYLE) or "")

    style = list_styles.get(named or "")
    if style is None:
        return None

    # sf:list-level is the rung of the style's ladder, counted from the
    # unlabelled one ordinary body text sits on; a paragraph that names no level
    # is on that rung. Docling counts nesting from the first labelled rung
    # instead, so the depth is one less.
    rung = int_attr(paragraph, SF_ATTR_LIST_LEVEL) or 0
    label = style.label(rung)
    if label is None:
        return None
    return label._replace(depth=max(rung - 1, 0))


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
