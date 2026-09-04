# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Reader for the object graph of a Pages 5+ (2013 onwards) document.

The container is a set of ``Index/*.iwa`` archives whose schemas Apple has never
published, so what is read here is the object graph itself: a document archive
referencing a body text storage, run tables keyed by character index for
everything applied to that text, and drawables reached either from the
attachments anchored in the text or from the document's own list of them.

Only the message and field numbers are format knowledge; the container layer
lives in :mod:`docling.backend.iwork.iwa`.
"""

import logging
import zipfile
from collections.abc import Callable
from typing import NamedTuple, TypeVar

from docling_core.types.doc import (
    Formatting,
    TableCell,
    TableData,
)

from docling.backend.iwork import tables
from docling.backend.iwork.content import (
    SCRIPTS,
    Block,
    Comment,
    Content,
    ListStyle,
    Paragraph,
    Picture,
    StorageRuns,
    authored,
    build_formatting,
    label_for_style,
    list_label_at,
    runs_for,
    split_paragraphs,
    unique_paragraphs,
    value_at,
)
from docling.backend.iwork.iwa import (
    IWAObject,
    iter_objects,
    read_fields,
    read_reference,
)
from docling.exceptions import DocumentLoadError

_log = logging.getLogger(__name__)

_T = TypeVar("_T")


MAX_REFERENCE_DEPTH = 4
"""How far to descend when collecting references from a message."""

REFERENCE_MAX_BYTES = 6
"""Longest a ``TSP.Reference`` can be; anything larger is a nested message."""

TSWP_CHARACTER_STYLE = 2021
"""Message type of ``TSWP.CharacterStyleArchive``."""

STORAGE_CHARACTER_STYLE_FIELD = 8
"""Field of ``TSWP.StorageArchive`` holding the character style run table."""

STYLE_PROPERTIES_FIELD = 11
"""Field of a character style holding its property map."""

STYLE_SCRIPT_FIELD = 10
"""Field of a character style holding its superscript or subscript setting."""

CHARACTER_PROPERTY_LABELS = {
    1: "bold",
    2: "italic",
    11: "underline",
    12: "strikethrough",
}
"""Property fields of a character style, as they map onto ``Formatting``.

Established by correlating style *names* with their properties across three real
Apple documents: "Emphasis"/"Bold" set field 1, "Italic" field 2, "Underline"
and "Link" field 11, and "Strikethrough" field 12. Fields carrying anything else
— colours, fonts, capitalisation — have no equivalent here and are ignored.
"""

TSWP_SHAPE_INFO = 2011
"""Message type of ``TSWP.ShapeInfoArchive``, a shape that holds text."""

DOCUMENT_DRAWABLES_FIELD = 20
"""Field of ``TP.DocumentArchive`` referencing the document's floating drawables.

Text boxes hang off this rather than off the body storage. Reaching them by
ownership matters: scanning every ``TSWP.StorageArchive`` in the document would
also pick up headers, footers and footnotes, which are deliberately excluded.
"""

TP_DOCUMENT_ARCHIVE = 10000
"""Message type of ``TP.DocumentArchive``, the root object of a Pages document."""

TSWP_STORAGE_ARCHIVE = 2001
"""Message type of ``TSWP.StorageArchive``, which holds a run of text.

TSWP is Apple's shared text engine, so the same archive appears in Numbers and
Keynote documents.
"""

TSWP_PARAGRAPH_STYLE = 2022
"""Message type of ``TSWP.ParagraphStyleArchive``, named by its ``TSS`` super."""

DOCUMENT_BODY_FIELD = 4
"""Field of ``TP.DocumentArchive`` referencing the body ``TSWP.StorageArchive``."""

STORAGE_TEXT_FIELD = 3
"""Field of ``TSWP.StorageArchive`` holding the text itself."""

STYLE_SUPER_FIELD = 1
"""Field of ``TSWP.ParagraphStyleArchive`` holding its ``TSS.StyleArchive`` super."""

STYLE_NAME_FIELD = 1
"""Field of ``TSS.StyleArchive`` holding the style's human-facing name."""

TST_TABLE_MODEL = 6001
"""Message type of ``TST.TableModelArchive``, the root of one table."""

TST_TABULAR_INFO = 6000
"""Message type of ``TST.TableInfoArchive``, the drawable a table sits in."""

TABULAR_INFO_MODEL_FIELD = 2
"""Field of ``TST.TableInfoArchive`` referencing its ``TST.TableModelArchive``."""

TSD_IMAGE = 3005
"""Message type of ``TSD.ImageArchive``, one placed image."""

TSD_GROUP = 3008
"""Message type of ``TSD.GroupArchive``, several drawables grouped together."""

GROUP_CHILDREN_FIELD = 2
"""Field of ``TSD.GroupArchive`` referencing the drawables it holds."""

IMAGE_DATA_FIELDS = (15, 13, 11, 12)
"""Fields of ``TSD.ImageArchive`` that may carry the image's bytes.

Pages keeps several renditions of a placed image and does not always write all
of them, so they are tried in descending order of fidelity: the adjusted image
first, then the original, then the placed data, then the thumbnail.
"""

TSWP_DRAWABLE_ATTACHMENT = 2003
"""Message type of ``TSWP.DrawableAttachmentArchive``.

This is what a U+FFFC in the text resolves to: a drawable — an image, a table,
a text box — anchored at that character.
"""

ATTACHMENT_DRAWABLE_FIELD = 1
"""Field of ``TSWP.DrawableAttachmentArchive`` referencing the anchored drawable."""

STORAGE_ATTACHMENT_FIELD = 9
"""Field of ``TSWP.StorageArchive`` holding the attachment run table."""

STORAGE_COMMENT_FIELD = 23
"""Field of ``TSWP.StorageArchive`` holding the comment run table.

Its entries mark the stretch of text a comment is attached to, which is how
Pages 5 records a comment: as a highlight over the words being commented on
rather than as a character in the text.
"""

TSWP_COMMENT_FIELD = 2013
"""Message type of ``TSWP.CommentFieldArchive``, the anchor of one comment."""

COMMENT_FIELD_STORAGE_FIELD = 1
"""Field of ``TSWP.CommentFieldArchive`` referencing the comment itself."""

TSD_COMMENT_STORAGE = 3056
"""Message type of ``TSD.CommentStorageArchive``, one comment or one reply."""

COMMENT_TEXT_FIELD = 1

COMMENT_AUTHOR_FIELD = 3

COMMENT_REPLIES_FIELD = 4
"""Fields of ``TSD.CommentStorageArchive``.

Replies are comments in their own right, so a thread is a chain to be followed.
"""

TSK_ANNOTATION_AUTHOR = 212
"""Message type of ``TSK.AnnotationAuthorArchive``, who wrote a comment."""

AUTHOR_NAME_FIELD = 1
"""Field of ``TSK.AnnotationAuthorArchive`` holding the author's name."""

TSWP_NOTE = 2008
"""Message type of ``TSWP.NoteArchive``, one footnote or endnote."""

NOTE_STORAGE_FIELD = 2
"""Field of ``TSWP.NoteArchive`` referencing the storage holding the note's text."""

STORAGE_FOOTNOTE_FIELD = 16
"""Field of ``TSWP.StorageArchive`` holding the footnote run table.

Its entries anchor a note at the character the footnote mark occupies, which is
one of the U+FFFC placeholders in the text.
"""

STORAGE_PAGE_MASTER_FIELD = 17
"""Field of ``TSWP.StorageArchive`` holding the page master run table.

Headers and footers hang off the page master that covers a stretch of the
document rather than off the text itself, so this is the way in to them.
"""

TP_PAGE_MASTER = 10011
"""Message type of ``TP.PageMasterArchive``, the page layout of one section."""

PAGE_MASTER_HEADER_FOOTER_FIELDS = (23, 24, 25)
"""Fields of ``TP.PageMasterArchive`` referencing its headers and footers.

Pages keeps three sets — first page, even pages, odd pages — and writes all of
them whether or not the author filled them in.
"""

TP_HEADERS_AND_FOOTERS = 10143
"""Message type of ``TP.HeadersAndFootersArchive``."""

HEADERS_FIELD = 1

FOOTERS_FIELD = 2
"""Fields of ``TP.HeadersAndFootersArchive``, each a list of text storages."""

TSP_PACKAGE_METADATA = 11006
"""Message type of ``TSP.PackageMetadata``, which names the container's data files."""

PACKAGE_DATAS_FIELD = 4
"""Field of ``TSP.PackageMetadata`` listing one ``TSP.DataInfo`` per data file."""

DATA_INFO_IDENTIFIER_FIELD = 1

DATA_INFO_PREFERRED_NAME_FIELD = 3

DATA_INFO_NAME_FIELD = 4
"""Fields of ``TSP.DataInfo``.

An image references a data file by identifier; the file itself is a ``Data/``
member of the container, named by ``file_name`` when Pages renamed it on import
and by ``preferred_file_name`` otherwise.
"""

DATA_MEMBER_PREFIX = "Data/"

STORAGE_PARAGRAPH_STYLE_FIELD = 5
"""Field of ``TSWP.StorageArchive`` holding the paragraph style run table.

Each entry pairs a character index with a reference to the style that applies
from there. Entries without a reference leave the preceding style in force.

"""

TSWP_LINK_FIELD = 2032
"""Message type of ``TSWP.LinkFieldArchive``, one hyperlink."""

LINK_URL_FIELD = 2
"""Field of ``TSWP.LinkFieldArchive`` holding the address it points at."""

STORAGE_SMART_FIELD = 11
"""Field of ``TSWP.StorageArchive`` holding the smart field run table.

Pages calls a hyperlink a smart field, alongside placeholders and date fields,
so this table is read for links and anything else in it is left alone.
"""

TSWP_LIST_STYLE = 2023
"""Message type of ``TSWP.ListStyleArchive``, which labels a list's levels."""

STORAGE_LIST_DEPTH_FIELD = 6
"""Field of ``TSWP.StorageArchive`` holding each paragraph's nesting depth.

Its entries carry two numbers rather than a reference; the first is the depth,
counted from zero, and a document with no nesting carries the single entry
``(0, 0)``.
"""

STORAGE_LIST_STYLE_FIELD = 7
"""Field of ``TSWP.StorageArchive`` holding the list style run table.

This, not the depth, is what makes a paragraph a list item: Pages leaves a list
style in force over plain paragraphs too, and the style's label type for the
paragraph's depth is what says whether a marker is drawn.
"""

LIST_LABEL_TYPES_FIELD = 11

LIST_STRINGS_FIELD = 16
"""Fields of ``TSWP.ListStyleArchive``, one entry per nesting depth."""


def read_content(
    archive: zipfile.ZipFile,
    infos: list[zipfile.ZipInfo],
    max_file_bytes: int,
    document_hash: str,
) -> Content:
    """Read the content of a Pages 5+ document out of its IWA object graph.

    Args:
        archive: The open ``.pages`` container.
        infos: Its members.
        max_file_bytes: The largest member this is willing to decompress.
        document_hash: The document's hash, for error messages.

    Returns:
        Everything the document holds.

    Raises:
        DocumentLoadError: If a member is too large, or the object graph has no
            document archive or no body text storage.
    """
    objects: dict[int, IWAObject] = {}
    for info in infos:
        if not info.filename.endswith(".iwa"):
            continue
        if info.file_size > max_file_bytes:
            raise DocumentLoadError(
                f"Pages archive member {info.filename} is {info.file_size} "
                f"bytes, exceeding the max_file_bytes limit of "
                f"{max_file_bytes}."
            )
        for obj in iter_objects(archive.read(info)):
            objects[obj.identifier] = obj

    document = next(
        (o for o in objects.values() if o.message_type == TP_DOCUMENT_ARCHIVE),
        None,
    )
    if document is None:
        raise DocumentLoadError(
            f"Pages document with hash {document_hash} has no "
            "TP.DocumentArchive; the container may be corrupt or "
            "password-protected."
        )

    body_ref = read_fields(document.payload).get(DOCUMENT_BODY_FIELD, [None])[0]
    target = read_reference(body_ref) if isinstance(body_ref, bytes) else None
    storage = objects.get(target) if target is not None else None
    if storage is None or storage.message_type != TSWP_STORAGE_ARCHIVE:
        raise DocumentLoadError(
            f"Pages document with hash {document_hash} does not reference "
            "a body text storage."
        )

    reader = IWAReader(archive, objects)
    blocks = reader.storage_blocks(storage)
    blocks.extend(reader.floating_blocks(document))
    headers, footers = reader.page_furniture(storage)
    return Content(
        blocks=blocks,
        headers=headers,
        footers=footers,
        footnotes=reader.footnotes(storage),
        comments=reader.comments(storage),
    )


def iwa_style_name(payload: bytes) -> str | None:
    """Read a paragraph style's name out of its ``TSS`` super message.

    ``TSWP.ParagraphStyleArchive`` wraps a ``TSS.StyleArchive`` that carries the
    human-facing name ("Body", "Heading 1"). Anonymous styles — the ones Pages
    creates for ad-hoc formatting — have no name and are treated as body text.

    Args:
        payload: The encoded ``TSWP.ParagraphStyleArchive``.

    Returns:
        The style name, or None when the style is anonymous.
    """
    super_message = read_fields(payload).get(STYLE_SUPER_FIELD, [None])[0]
    if not isinstance(super_message, bytes):
        return None
    name = read_fields(super_message).get(STYLE_NAME_FIELD, [None])[0]
    if not isinstance(name, bytes):
        return None
    try:
        return name.decode("utf-8")
    except UnicodeDecodeError:
        return None


def iwa_storage_text(fields: dict[int, list[int | bytes]]) -> str:
    """Join the text pieces of a ``TSWP.StorageArchive``."""
    return "".join(
        value.decode("utf-8", errors="replace")
        for value in fields.get(STORAGE_TEXT_FIELD, [])
        if isinstance(value, bytes)
    )


def iwa_storage_runs(
    fields: dict[int, list[int | bytes]], objects: dict[int, IWAObject]
) -> StorageRuns:
    """Resolve every run table a ``TSWP.StorageArchive`` carries.

    Args:
        fields: Decoded fields of the storage.
        objects: Every object in the document, keyed by identifier.

    Returns:
        The tables, each sorted by character index.
    """
    return StorageRuns(
        styles=iwa_object_runs(
            fields,
            STORAGE_PARAGRAPH_STYLE_FIELD,
            objects,
            TSWP_PARAGRAPH_STYLE,
            iwa_style_name,
        ),
        characters=iwa_object_runs(
            fields,
            STORAGE_CHARACTER_STYLE_FIELD,
            objects,
            TSWP_CHARACTER_STYLE,
            iwa_formatting,
        ),
        lists=iwa_object_runs(
            fields,
            STORAGE_LIST_STYLE_FIELD,
            objects,
            TSWP_LIST_STYLE,
            iwa_list_style,
        ),
        depths=iwa_depth_runs(fields),
        links=iwa_object_runs(
            fields, STORAGE_SMART_FIELD, objects, TSWP_LINK_FIELD, iwa_link
        ),
    )


def iwa_object_runs(
    fields: dict[int, list[int | bytes]],
    field: int,
    objects: dict[int, IWAObject],
    message_type: int,
    decode: Callable[[bytes], _T | None],
) -> list[tuple[int, _T | None]]:
    """Resolve one ``TSWP.ObjectAttributeTable`` to (character index, value) pairs.

    Every run table of a storage has this shape: entries pairing a character
    index with a reference to the object that applies from there. An entry
    without a reference clears the value from that character on, which is how
    Pages ends a bold phrase or leaves a list.

    Args:
        fields: Decoded fields of the storage.
        field: The storage field holding the table.
        objects: Every object in the document, keyed by identifier.
        message_type: The message type the referenced objects must have.
        decode: Reads one referenced object's payload into a value.

    Returns:
        Character index and value pairs, in document order.
    """
    table = fields.get(field, [])
    if not table or not isinstance(table[0], bytes):
        return []

    runs: list[tuple[int, _T | None]] = []
    for entry in tables.safe_fields(table[0]).get(1, []):
        if not isinstance(entry, bytes):
            continue
        parsed = tables.safe_fields(entry)
        index = parsed.get(1, [None])[0]
        if not isinstance(index, int):
            continue

        reference = parsed.get(2, [None])[0]
        value: _T | None = None
        if isinstance(reference, bytes):
            target = read_reference(reference)
            referenced = objects.get(target) if target is not None else None
            if referenced is not None and referenced.message_type == message_type:
                value = decode(referenced.payload)
        runs.append((index, value))

    runs.sort(key=lambda run: run[0])
    return runs


def iwa_depth_runs(fields: dict[int, list[int | bytes]]) -> list[tuple[int, int]]:
    """Resolve the list depth table, whose entries hold numbers, not references."""
    table = fields.get(STORAGE_LIST_DEPTH_FIELD, [])
    if not table or not isinstance(table[0], bytes):
        return []

    runs: list[tuple[int, int]] = []
    for entry in tables.safe_fields(table[0]).get(1, []):
        if not isinstance(entry, bytes):
            continue
        parsed = tables.safe_fields(entry)
        index = parsed.get(1, [None])[0]
        depth = parsed.get(2, [None])[0]
        if isinstance(index, int) and isinstance(depth, int):
            runs.append((index, depth))

    runs.sort(key=lambda run: run[0])
    return runs


def iwa_link(payload: bytes) -> str | None:
    """Read the address a ``TSWP.LinkFieldArchive`` points at."""
    url = tables.safe_fields(payload).get(LINK_URL_FIELD, [None])[0]
    if not isinstance(url, bytes):
        return None
    return url.decode("utf-8", errors="replace").strip() or None


def iwa_list_style(payload: bytes) -> ListStyle:
    """Read a ``TSWP.ListStyleArchive`` as its per-depth label ladder."""
    fields = tables.safe_fields(payload)
    label_types = tuple(
        value
        for value in fields.get(LIST_LABEL_TYPES_FIELD, [])
        if isinstance(value, int)
    )
    strings = tuple(
        value.decode("utf-8", errors="replace")
        for value in fields.get(LIST_STRINGS_FIELD, [])
        if isinstance(value, bytes)
    )
    return ListStyle(label_types, strings)


def iwa_formatting(payload: bytes) -> Formatting | None:
    """Read a character style's property map as a :class:`Formatting`."""
    properties = tables.safe_fields(payload).get(STYLE_PROPERTIES_FIELD, [None])[0]
    if not isinstance(properties, bytes):
        return None

    decoded = tables.safe_fields(properties)
    active = {
        label
        for field, label in CHARACTER_PROPERTY_LABELS.items()
        if any(isinstance(value, int) and value for value in decoded.get(field, []))
    }
    script = decoded.get(STYLE_SCRIPT_FIELD, [None])[0]
    return build_formatting(
        active, SCRIPTS.get(script) if isinstance(script, int) else None
    )


class IWAReader:
    """Reads content out of the object graph of a Pages 5+ document.

    Drawables are reached twice over — once from the attachment table of the
    text they are anchored in, and once from the document's own list of floating
    ones — so every drawable this has already emitted is remembered. That also
    bounds the walk: an object graph may contain cycles.
    """

    def __init__(self, archive: zipfile.ZipFile, objects: dict[int, IWAObject]) -> None:
        self._archive = archive
        self._objects = objects
        self._data_files = iwa_data_files(objects)
        self._emitted: set[int] = set()

    def storage_blocks(self, storage: IWAObject) -> list[Block]:
        """Read one ``TSWP.StorageArchive`` as paragraphs and anchored drawables.

        Apple marks the anchor of a drawable with U+FFFC inside the text, and the
        storage's attachment table says which drawable each one is. The drawable
        is emitted straight after the paragraph it is anchored in, which is where
        it belongs in the reading order.

        Args:
            storage: The storage to read.

        Returns:
            The storage's blocks, in document order.
        """
        fields = read_fields(storage.payload)
        text = iwa_storage_text(fields)
        runs = iwa_storage_runs(fields, self._objects)
        attachments = iwa_attachment_runs(fields)
        comments = iwa_attachment_runs(fields, STORAGE_COMMENT_FIELD)

        blocks: list[Block] = []
        offset = 0
        for line in text.split("\n"):
            end = offset + len(line) + 1
            pieces = runs_for(line, offset, runs)
            if pieces:
                label, level = label_for_style(value_at(runs.styles, offset))
                anchors = tuple(
                    str(field) for index, field in comments if offset <= index < end
                )
                blocks.append(
                    Paragraph(
                        pieces, label, level, list_label_at(runs, offset), anchors
                    )
                )
            for index, identifier in attachments:
                if offset <= index < end:
                    blocks.extend(self._drawable_blocks(identifier))
            offset = end  # the + 1 above covers the newline that split consumed

        return blocks

    def floating_blocks(self, document: IWAObject) -> list[Block]:
        """Read the drawables the document owns rather than anchors in its text.

        Reaching them by ownership matters: scanning every ``TSWP.StorageArchive``
        in the document would also pick up headers, footers and footnotes, which
        belong to the page rather than to the body flow.

        Args:
            document: The ``TP.DocumentArchive`` of the document.

        Returns:
            The blocks of every drawable not already emitted from the text.
        """
        drawables = read_fields(document.payload).get(DOCUMENT_DRAWABLES_FIELD, [None])[
            0
        ]
        if not isinstance(drawables, bytes):
            return []

        container = read_reference(drawables)
        root = self._objects.get(container) if container is not None else None
        if root is None:
            return []

        blocks: list[Block] = []
        for identifier in sorted(iwa_referenced_ids(root.payload)):
            blocks.extend(self._drawable_blocks(identifier))
        return blocks

    def footnotes(self, storage: IWAObject) -> list[Paragraph]:
        """Read the notes anchored in one storage.

        The footnote run table anchors a note at the character its mark occupies
        — one of the U+FFFC placeholders the text carries — and the note holds
        its own storage of text.

        Args:
            storage: The storage whose footnote table to read.

        Returns:
            The notes' paragraphs, in the order they are anchored.
        """
        fields = read_fields(storage.payload)
        paragraphs: list[Paragraph] = []
        for _, identifier in iwa_attachment_runs(fields, STORAGE_FOOTNOTE_FIELD):
            note = self._objects.get(identifier)
            if note is None or note.message_type != TSWP_NOTE:
                continue
            text_id = tables.reference_field(note.payload, NOTE_STORAGE_FIELD)
            paragraphs.extend(self._storage_paragraphs(text_id))
        return paragraphs

    def page_furniture(
        self, storage: IWAObject
    ) -> tuple[list[Paragraph], list[Paragraph]]:
        """Read the headers and footers of the page masters a storage runs under.

        Pages writes three sets per master — first page, even pages, odd pages —
        whether or not the author filled them in, and a document with several
        sections repeats them per master, so identical text is emitted once.

        Args:
            storage: The body storage, which names its page masters.

        Returns:
            The header paragraphs and the footer paragraphs.
        """
        fields = read_fields(storage.payload)
        headers: list[Paragraph] = []
        footers: list[Paragraph] = []

        for _, identifier in iwa_attachment_runs(fields, STORAGE_PAGE_MASTER_FIELD):
            master = self._objects.get(identifier)
            if master is None or master.message_type != TP_PAGE_MASTER:
                continue
            for field in PAGE_MASTER_HEADER_FOOTER_FIELDS:
                pair = tables.reference_field(master.payload, field)
                bundle = self._objects.get(pair) if pair is not None else None
                if bundle is None or bundle.message_type != TP_HEADERS_AND_FOOTERS:
                    continue
                for source, target in (
                    (HEADERS_FIELD, headers),
                    (FOOTERS_FIELD, footers),
                ):
                    for text_id in tables.reference_list(bundle.payload, source):
                        target.extend(self._storage_paragraphs(text_id))

        return unique_paragraphs(headers), unique_paragraphs(footers)

    def comments(self, storage: IWAObject) -> list[Comment]:
        """Read the comments attached to the text of one storage.

        Pages 5 records a comment as a highlight over the words being commented
        on rather than as a character in the text, so the run table gives the
        stretch it covers. Each entry names a comment field, which holds the
        comment; replies are comments in their own right and are followed as a
        chain.

        Args:
            storage: The storage whose comment table to read.

        Returns:
            One comment per thread entry, anchored by the field's identifier.
        """
        fields = read_fields(storage.payload)
        comments: list[Comment] = []

        for _, identifier in iwa_attachment_runs(fields, STORAGE_COMMENT_FIELD):
            field = self._objects.get(identifier)
            if field is None or field.message_type != TSWP_COMMENT_FIELD:
                continue
            head = tables.reference_field(field.payload, COMMENT_FIELD_STORAGE_FIELD)
            comments.extend(
                Comment(text, str(identifier)) for text in self._thread(head)
            )

        return comments

    def _thread(self, identifier: int | None) -> list[str]:
        """Read one comment and its replies, as text prefixed by their authors."""
        texts: list[str] = []
        pending = [identifier]
        seen: set[int] = set()

        while pending:
            current = pending.pop(0)
            if current is None or current in seen:
                continue
            seen.add(current)
            comment = self._objects.get(current)
            if comment is None or comment.message_type != TSD_COMMENT_STORAGE:
                continue

            fields = tables.safe_fields(comment.payload)
            raw = fields.get(COMMENT_TEXT_FIELD, [None])[0]
            if isinstance(raw, bytes):
                text = raw.decode("utf-8", errors="replace").strip()
                if text:
                    texts.append(authored(self._author(comment.payload), text))
            pending.extend(
                tables.reference_list(comment.payload, COMMENT_REPLIES_FIELD)
            )

        return texts

    def _author(self, payload: bytes) -> str | None:
        """Read the name of whoever wrote a comment."""
        identifier = tables.reference_field(payload, COMMENT_AUTHOR_FIELD)
        author = self._objects.get(identifier) if identifier is not None else None
        if author is None or author.message_type != TSK_ANNOTATION_AUTHOR:
            return None
        name = tables.safe_fields(author.payload).get(AUTHOR_NAME_FIELD, [None])[0]
        if not isinstance(name, bytes):
            return None
        return name.decode("utf-8", errors="replace").strip() or None

    def _storage_paragraphs(self, identifier: int | None) -> list[Paragraph]:
        """Read one storage's paragraphs, ignoring anything anchored in it."""
        storage = self._objects.get(identifier) if identifier is not None else None
        if storage is None or storage.message_type != TSWP_STORAGE_ARCHIVE:
            return []
        fields = read_fields(storage.payload)
        return split_paragraphs(
            iwa_storage_text(fields), iwa_storage_runs(fields, self._objects)
        )

    def _drawable_blocks(self, identifier: int) -> list[Block]:
        """Read whichever kind of drawable ``identifier`` names."""
        if identifier in self._emitted:
            return []
        self._emitted.add(identifier)

        drawable = self._objects.get(identifier)
        if drawable is None:
            return []

        if drawable.message_type == TSWP_DRAWABLE_ATTACHMENT:
            anchored = tables.reference_field(
                drawable.payload, ATTACHMENT_DRAWABLE_FIELD
            )
            return self._drawable_blocks(anchored) if anchored is not None else []

        if drawable.message_type == TSD_IMAGE:
            return [self._picture(drawable)]

        if drawable.message_type == TST_TABULAR_INFO:
            model = tables.reference_field(drawable.payload, TABULAR_INFO_MODEL_FIELD)
            table = self._objects.get(model) if model is not None else None
            if table is None or table.message_type != TST_TABLE_MODEL:
                return []
            data = tables.table(table, self._objects)
            return [data] if data is not None else []

        if drawable.message_type == TSD_GROUP:
            blocks: list[Block] = []
            for child in tables.reference_list(drawable.payload, GROUP_CHILDREN_FIELD):
                blocks.extend(self._drawable_blocks(child))
            return blocks

        if drawable.message_type == TSWP_SHAPE_INFO:
            blocks = []
            for storage_id in sorted(iwa_referenced_ids(drawable.payload)):
                storage = self._objects.get(storage_id)
                if storage is not None and storage.message_type == TSWP_STORAGE_ARCHIVE:
                    blocks.extend(self.storage_blocks(storage))
            return blocks

        return []

    def _picture(self, image: IWAObject) -> Picture:
        """Read a ``TSD.ImageArchive`` and the container member holding its bytes."""
        fields = tables.safe_fields(image.payload)
        named = ""
        for field in IMAGE_DATA_FIELDS:
            reference = fields.get(field, [None])[0]
            if not isinstance(reference, bytes):
                continue
            data_id = read_reference(reference)
            member = self._data_files.get(data_id) if data_id is not None else None
            if member is None:
                continue
            named = named or member
            try:
                return Picture(self._archive.read(member), member)
            except KeyError:
                # Pages names every rendition it knows of, including ones it did
                # not write into this container, so keep trying the rest.
                _log.debug("Pages image data member %s is missing", member)
        return Picture(None, named)


def iwa_attachment_runs(
    fields: dict[int, list[int | bytes]],
    field: int = STORAGE_ATTACHMENT_FIELD,
) -> list[tuple[int, int]]:
    """Resolve an anchoring run table to (character index, object id) pairs.

    Unlike the style tables, an entry here anchors an object at one character
    rather than putting a value in force from it, so entries without a reference
    carry nothing and are dropped. Attachments, footnotes and page masters all
    use this shape.

    Args:
        fields: Decoded fields of the storage.
        field: The storage field holding the table.

    Returns:
        Character index and object identifier pairs, in document order.
    """
    table = fields.get(field, [])
    if not table or not isinstance(table[0], bytes):
        return []

    runs: list[tuple[int, int]] = []
    for entry in tables.safe_fields(table[0]).get(1, []):
        if not isinstance(entry, bytes):
            continue
        parsed = tables.safe_fields(entry)
        index = parsed.get(1, [None])[0]
        reference = parsed.get(2, [None])[0]
        if not isinstance(index, int) or not isinstance(reference, bytes):
            continue
        target = read_reference(reference)
        if target is not None:
            runs.append((index, target))

    runs.sort(key=lambda run: run[0])
    return runs


def iwa_data_files(objects: dict[int, IWAObject]) -> dict[int, str]:
    """Map each data identifier to the container member that holds its bytes.

    Args:
        objects: Every object in the document, keyed by identifier.

    Returns:
        Data identifiers and the ``Data/`` member names they name.
    """
    metadata = next(
        (o for o in objects.values() if o.message_type == TSP_PACKAGE_METADATA), None
    )
    if metadata is None:
        return {}

    files: dict[int, str] = {}
    for entry in tables.safe_fields(metadata.payload).get(PACKAGE_DATAS_FIELD, []):
        if not isinstance(entry, bytes):
            continue
        info = tables.safe_fields(entry)
        identifier = info.get(DATA_INFO_IDENTIFIER_FIELD, [None])[0]
        name = info.get(DATA_INFO_NAME_FIELD, [None])[0]
        if not isinstance(name, bytes):
            name = info.get(DATA_INFO_PREFERRED_NAME_FIELD, [None])[0]
        if isinstance(identifier, int) and isinstance(name, bytes):
            files[identifier] = DATA_MEMBER_PREFIX + name.decode(
                "utf-8", errors="replace"
            )
    return files


def iwa_referenced_ids(payload: bytes, depth: int = 0) -> set[int]:
    """Collect the object identifiers a message references, at any nesting.

    Args:
        payload: The encoded message to scan.
        depth: Current recursion depth, bounded to keep a hostile document from
            driving this arbitrarily deep.

    Returns:
        Every identifier reachable from the message.
    """
    if depth > MAX_REFERENCE_DEPTH:
        return set()

    found: set[int] = set()
    for values in tables.safe_fields(payload).values():
        for value in values:
            if not isinstance(value, bytes):
                continue
            if len(value) <= REFERENCE_MAX_BYTES:
                try:
                    target = read_reference(value)
                except DocumentLoadError:
                    continue
                if isinstance(target, int):
                    found.add(target)
            else:
                found |= iwa_referenced_ids(value, depth + 1)
    return found
