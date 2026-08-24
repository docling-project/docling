"""Backends for Apple iWork documents.

Currently limited to Pages (``.pages``). A ``.pages`` file is a ZIP container, but
what is inside changed completely with Pages 5:

* **Pages 5 and later (2013 onwards)** store the document as ``Index/*.iwa`` —
  Snappy-framed protobuf whose schemas Apple has never published. This is what
  essentially every Pages document in circulation looks like.
* **iWork '09 and earlier** stored it as a plain ``index.xml``, optionally
  gzipped, alongside a ``QuickLook/Preview.pdf`` render that Apple stopped
  writing after that release.

Both generations are read for their text here, so the backend is declarative: it
builds a :class:`~docling_core.types.doc.DoclingDocument` directly rather than
rendering pages and running layout analysis over them.

Paragraph styles carry the document outline in both generations, so titles and
headings are recovered from them, lists from the list styles they point at, and
tables from the structures each generation uses for them.
"""

import logging
import mimetypes
import re
import zipfile
import zlib
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import NamedTuple, TypeVar
from xml.etree.ElementTree import Element

import defusedxml.ElementTree as ET
from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    Formatting,
    TableCell,
    TableData,
)
from docling_core.types.doc.items.group import ListGroup
from typing_extensions import override

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.backend.iwork.iwa import (
    IWAObject,
    is_encrypted,
    iter_objects,
    read_fields,
    read_reference,
)
from docling.datamodel.backend_options import IWorkBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.exceptions import DocumentLoadError

_log = logging.getLogger(__name__)

_T = TypeVar("_T")


class _Run(NamedTuple):
    """A stretch of text sharing one character style."""

    text: str
    formatting: Formatting | None


class _ListLabel(NamedTuple):
    """How Pages labels one list item: its depth, and the marker it shows."""

    depth: int
    enumerated: bool
    marker: str


class _ListStyle(NamedTuple):
    """A Pages list style: what each nesting depth is labelled with.

    Every field is a parallel array indexed by depth, so a style describes the
    whole ladder of nine levels at once rather than one level at a time.
    """

    label_types: tuple[int, ...]
    strings: tuple[str, ...]

    def label(self, depth: int) -> _ListLabel | None:
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
        if label_type == _LABEL_TYPE_NONE:
            return None
        if label_type == _LABEL_TYPE_NUMBER:
            return _ListLabel(depth, True, "")
        marker = self.strings[depth] if depth < len(self.strings) else ""
        # An image bullet has no text to show, so it falls back to the marker
        # docling uses for an unlabelled item.
        return _ListLabel(depth, False, marker or "-")


class _Paragraph(NamedTuple):
    """One block of body text with the label its Pages style implies.

    A paragraph is kept as runs rather than a single string because Pages
    applies character styles to arbitrary stretches of it, and a bold phrase in
    the middle of a sentence has to stay attached to that phrase.
    """

    runs: tuple[_Run, ...]
    label: DocItemLabel
    level: int | None
    list_label: _ListLabel | None = None

    @property
    def text(self) -> str:
        """The paragraph's full text, with its runs joined back together."""
        return "".join(run.text for run in self.runs)


class _StorageRuns(NamedTuple):
    """The run tables of one ``TSWP.StorageArchive``.

    Each table pairs a character index with the value that applies from there
    until the next entry, so they are read together when the storage is split
    into paragraphs.
    """

    styles: list[tuple[int, str | None]] = []
    characters: list[tuple[int, Formatting | None]] = []
    lists: list[tuple[int, _ListStyle | None]] = []
    depths: list[tuple[int, int]] = []


_PAGES_MIMETYPE = "application/vnd.apple.pages"

# DocumentOrigin only accepts a mimetype that the stdlib knows or that
# docling-core allow-lists, and Python ships no mapping for ".pages". Teaching
# the stdlib the real Apple type keeps the origin honest without waiting on a
# docling-core release; it also makes mimetypes.guess_type() correct for callers.
mimetypes.add_type(_PAGES_MIMETYPE, ".pages")

_MODERN_INDEX_PREFIX = "Index/"
_LEGACY_INDEX_MEMBERS = ("index.xml", "index.xml.gz")

# An index.xml.gz can expand enormously relative to its stored size, so the
# legacy path decompresses incrementally against this ceiling rather than
# trusting the member size that max_total_bytes is computed from.
_MAX_LEGACY_XML_BYTES = 100 * 1024 * 1024

_MAX_REFERENCE_DEPTH = 4
"""How far to descend when collecting references from a message."""

_REFERENCE_MAX_BYTES = 6
"""Longest a ``TSP.Reference`` can be; anything larger is a nested message."""

_TSWP_CHARACTER_STYLE = 2021
"""Message type of ``TSWP.CharacterStyleArchive``."""

_STORAGE_CHARACTER_STYLE_FIELD = 8
"""Field of ``TSWP.StorageArchive`` holding the character style run table."""

_STYLE_PROPERTIES_FIELD = 11
"""Field of a character style holding its property map."""

_CHARACTER_PROPERTY_LABELS = {
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

_TSWP_SHAPE_INFO = 2011
"""Message type of ``TSWP.ShapeInfoArchive``, a shape that holds text."""

_DOCUMENT_DRAWABLES_FIELD = 20
"""Field of ``TP.DocumentArchive`` referencing the document's floating drawables.

Text boxes hang off this rather than off the body storage. Reaching them by
ownership matters: scanning every ``TSWP.StorageArchive`` in the document would
also pick up headers, footers and footnotes, which are deliberately excluded.
"""

_TP_DOCUMENT_ARCHIVE = 10000
"""Message type of ``TP.DocumentArchive``, the root object of a Pages document."""

_TSWP_STORAGE_ARCHIVE = 2001
"""Message type of ``TSWP.StorageArchive``, which holds a run of text.

TSWP is Apple's shared text engine, so the same archive appears in Numbers and
Keynote documents.
"""

_TSWP_PARAGRAPH_STYLE = 2022
"""Message type of ``TSWP.ParagraphStyleArchive``, named by its ``TSS`` super."""

_DOCUMENT_BODY_FIELD = 4
"""Field of ``TP.DocumentArchive`` referencing the body ``TSWP.StorageArchive``."""

_STORAGE_TEXT_FIELD = 3
"""Field of ``TSWP.StorageArchive`` holding the text itself."""

_STYLE_SUPER_FIELD = 1
"""Field of ``TSWP.ParagraphStyleArchive`` holding its ``TSS.StyleArchive`` super."""

_STYLE_NAME_FIELD = 1
"""Field of ``TSS.StyleArchive`` holding the style's human-facing name."""

_TST_TABLE_MODEL = 6001
"""Message type of ``TST.TableModelArchive``, the root of one table."""

_TST_TILE = 6002
"""Message type of ``TST.Tile``, which lays a table's cells out into rows."""

_TST_DATA_LIST = 6005
"""Message type of ``TST.TableDataList``, a table's shared value table.

Cells reference their contents by key rather than holding them, so two cells
with the same text share a single entry.
"""

_TABLE_ROWS_FIELD = 6
_TABLE_COLS_FIELD = 7
_TABLE_HEADER_ROWS_FIELD = 9
_TABLE_DATA_STORE_FIELD = 4
"""Fields of ``TST.TableModelArchive``: geometry, header rows, and data store."""

_STORE_TILES_FIELD = 3
_STORE_STRINGS_FIELD = 4
"""Fields of a table's data store: its tiles, and its string data list."""

_TILE_ROWS_FIELD = 5
_ROW_INDEX_FIELD = 1
_ROW_STORAGE_FIELD = 3
_ROW_OFFSETS_FIELD = 4
"""Fields of ``TST.Tile`` and of one of its rows.

A row holds a packed cell buffer plus one ``int16`` offset per column, where a
negative offset marks a column with no cell.
"""

_CELL_VERSION = 4
_CELL_TYPE_TEXT = 3
_CELL_KEY_OFFSET = 16
"""Layout of a packed cell.

Byte 0 is the storage version and byte 1 the value type; a text cell holds the
key of its string in the four bytes at ``_CELL_KEY_OFFSET``. Only this
combination is decoded, so an unrecognised cell yields no text rather than
misread bytes.
"""

_STORAGE_PARAGRAPH_STYLE_FIELD = 5
"""Field of ``TSWP.StorageArchive`` holding the paragraph style run table.

Each entry pairs a character index with a reference to the style that applies
from there. Entries without a reference leave the preceding style in force.
"""

_TSWP_LIST_STYLE = 2023
"""Message type of ``TSWP.ListStyleArchive``, which labels a list's levels."""

_STORAGE_LIST_DEPTH_FIELD = 6
"""Field of ``TSWP.StorageArchive`` holding each paragraph's nesting depth.

Its entries carry two numbers rather than a reference; the first is the depth,
counted from zero, and a document with no nesting carries the single entry
``(0, 0)``.
"""

_STORAGE_LIST_STYLE_FIELD = 7
"""Field of ``TSWP.StorageArchive`` holding the list style run table.

This, not the depth, is what makes a paragraph a list item: Pages leaves a list
style in force over plain paragraphs too, and the style's label type for the
paragraph's depth is what says whether a marker is drawn.
"""

_LIST_LABEL_TYPES_FIELD = 11
_LIST_STRINGS_FIELD = 16
"""Fields of ``TSWP.ListStyleArchive``, one entry per nesting depth."""

_LABEL_TYPE_NONE = 0
_LABEL_TYPE_STRING = 2
_LABEL_TYPE_NUMBER = 3
"""Label types of ``TSWP.ListStyleArchive``.

``kNone`` leaves the depth unlabelled and ``kNumber`` numbers it; ``kImage``
and ``kString`` both draw a fixed marker, which for a string label is the entry
at that depth of the style's ``strings``.
"""

_SF_NAMESPACE = "http://developer.apple.com/namespaces/sf"
_SF_PARAGRAPH = f"{{{_SF_NAMESPACE}}}p"
# iWork '09 placeholder text. It is what the template shows before the author
# types anything, so it must never be emitted as document content.
_SF_GHOST_TEXT = f"{{{_SF_NAMESPACE}}}ghost-text"
_SF_PARAGRAPH_STYLE = f"{{{_SF_NAMESPACE}}}paragraphstyle"
_SFA_NAMESPACE = "http://developer.apple.com/namespaces/sfa"
_SF_ATTR_IDENT = f"{{{_SF_NAMESPACE}}}ident"
_SF_ATTR_NAME = f"{{{_SF_NAMESPACE}}}name"
_SF_ATTR_STYLE = f"{{{_SF_NAMESPACE}}}style"
_SF_ATTR_NUMCOLS = f"{{{_SF_NAMESPACE}}}numcols"
_SF_ATTR_NUMROWS = f"{{{_SF_NAMESPACE}}}numrows"
_SF_ATTR_HEADER_ROWS = f"{{{_SF_NAMESPACE}}}num-header-rows"
_SFA_ATTR_STRING = f"{{{_SFA_NAMESPACE}}}s"
_SF_TABULAR_MODEL = f"{{{_SF_NAMESPACE}}}tabular-model"
_SF_GRID = f"{{{_SF_NAMESPACE}}}grid"
_SF_CELL_TEXT = f"{{{_SF_NAMESPACE}}}ct"
_SF_SPAN = f"{{{_SF_NAMESPACE}}}span"
_SF_CHARACTER_STYLE = f"{{{_SF_NAMESPACE}}}characterstyle"
_SFA_ATTR_NUMBER = "{http://developer.apple.com/namespaces/sfa}number"

_SF_LIST_STYLE = f"{{{_SF_NAMESPACE}}}liststyle"
_SF_LIST_LABEL_TYPE = f"{{{_SF_NAMESPACE}}}list-label-typeinfo"
_SF_TEXT_LABEL = f"{{{_SF_NAMESPACE}}}text-label"
_SF_ATTR_TYPE = f"{{{_SF_NAMESPACE}}}type"
_SF_ATTR_FORMAT = f"{{{_SF_NAMESPACE}}}format"
_SF_ATTR_LIST_LEVEL = f"{{{_SF_NAMESPACE}}}list-level"
_SF_ATTR_LIST_STYLE = f"{{{_SF_NAMESPACE}}}list-style"
"""The iWork '09 vocabulary for lists.

An ``sf:liststyle`` holds one ``sf:list-label-typeinfo`` per nesting level, and
a paragraph joins the list by naming the style and its own ``sf:list-level``,
which counts from one.
"""

_SF_LABEL_TYPE_NONE = "none"
"""``sf:list-label-typeinfo`` type that leaves a level unlabelled."""

_SF_BULLET_LABEL_TYPES = frozenset({"bullet", "image", "string", "text"})
"""``sf:text-label`` types that draw a fixed marker rather than a number."""

_SF_PROPERTY_LABELS = {
    f"{{{_SF_NAMESPACE}}}bold": "bold",
    f"{{{_SF_NAMESPACE}}}italic": "italic",
    f"{{{_SF_NAMESPACE}}}underline": "underline",
    f"{{{_SF_NAMESPACE}}}strikethru": "strikethrough",
}
"""Property-map entries of an iWork '09 character style, as ``Formatting`` names."""

_SF_FURNITURE = frozenset(
    {
        f"{{{_SF_NAMESPACE}}}header",
        f"{{{_SF_NAMESPACE}}}footer",
        f"{{{_SF_NAMESPACE}}}footnotes",
    }
)
"""Elements whose paragraphs are page furniture rather than body content.

Each carries its own ``sf:text-body``, so they have to be pruned by element
rather than by looking for the document's body. The IWA reader only ever sees
the body storage, so skipping them keeps both generations in agreement about
what the document contains.
"""

_HEADING_PATTERN = re.compile(r"^heading\s*(\d+)?$", re.IGNORECASE)
"""Matches Pages' built-in heading styles, e.g. "Heading 1" or bare "Heading"."""

# Apple marks inline attachments (images, footnote anchors) with U+FFFC inside
# the text run. There is no text there to emit.
_OBJECT_REPLACEMENT = "￼"


class IWorkPagesDocumentBackend(DeclarativeDocumentBackend):
    """Extract text from Apple Pages documents of either generation.

    Known limitations:
        * Only text cells are read from a table. A cell holding a number, a
          date or a formula result is left empty rather than guessed at.
        * Bold, italic, underline and strikethrough are recovered; other
          character properties, such as colour or capitalisation, have no
          equivalent here.
        * A list item whose runs differ in formatting keeps its text but loses
          the formatting, since a list item carries a single one.
        * Text boxes are read from Pages 5+ documents, where they are floating
          drawables owned by the document. An iWork '09 document keeps them in
          the body flow, so they already appear there.
        * Headers, footers, footnotes and comments are not included in either
          generation.
        * Password-protected documents cannot be read.
        * ``.pages`` bundles saved as a *directory* package rather than a single
          file are not recognised; the converter cannot address a directory as an
          input document.
    """

    @override
    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: BytesIO | Path,
        options: IWorkBackendOptions | None = None,
    ):
        if options is None:
            options = IWorkBackendOptions()
        super().__init__(in_doc, path_or_stream, options)
        self.options: IWorkBackendOptions = options

        self._paragraphs: list[_Paragraph] = []
        self._tables: list[TableData] = []
        self._valid = False

        try:
            with zipfile.ZipFile(path_or_stream) as archive:
                self._paragraphs, self._tables = self._read_document(archive)
        except DocumentLoadError:
            raise
        except RecursionError as exc:
            # RecursionError subclasses RuntimeError, so it must be caught first;
            # otherwise deeply nested XML would be reported as an encryption
            # problem, hiding a real robustness failure behind benign advice.
            raise DocumentLoadError(
                f"Pages document with hash {self.document_hash} is nested too "
                "deeply to parse."
            ) from exc
        except (NotImplementedError, RuntimeError) as exc:
            # Encryption is normally detected up front from the member table.
            # Anything reaching here is an unreadable member for some other
            # reason (an unknown compression method, a missing codec module), so
            # the message stays about the container rather than about passwords.
            raise DocumentLoadError(
                f"Could not read Pages document with hash {self.document_hash}: "
                f"the archive contains a member Docling cannot decompress ({exc})."
            ) from exc
        except (zipfile.BadZipFile, OSError) as exc:
            raise DocumentLoadError(
                f"Could not open Pages document with hash {self.document_hash}: "
                "the file is not a readable ZIP container."
            ) from exc

        self._valid = True

    def _read_document(
        self, archive: zipfile.ZipFile
    ) -> tuple[list[_Paragraph], list[TableData]]:
        """Dispatch to the reader for whichever generation wrote the container."""
        infos = archive.infolist()
        if len(infos) > self.options.max_member_count:
            raise DocumentLoadError(
                f"Pages archive has {len(infos)} members, exceeding the "
                f"max_member_count limit of {self.options.max_member_count}."
            )
        total_bytes = sum(info.file_size for info in infos)
        if total_bytes > self.options.max_total_bytes:
            raise DocumentLoadError(
                f"Pages archive expands to {total_bytes} bytes, exceeding the "
                f"max_total_bytes limit of {self.options.max_total_bytes}."
            )

        if any(is_encrypted(info) for info in infos):
            raise DocumentLoadError(
                f"Pages document with hash {self.document_hash} is "
                "password-protected; Docling cannot read encrypted iWork "
                "documents. Remove the password in Pages and save again."
            )

        names = {info.filename for info in infos}
        if any(name.startswith(_MODERN_INDEX_PREFIX) for name in names):
            return self._read_iwa_document(archive, infos)

        legacy = next((n for n in _LEGACY_INDEX_MEMBERS if n in names), None)
        if legacy is not None:
            return self._read_legacy_document(archive, legacy)

        raise DocumentLoadError(
            f"Document with hash {self.document_hash} is a ZIP archive but does "
            "not look like a Pages document: it has neither an Index/ directory "
            "nor an index.xml."
        )

    def _read_iwa_document(
        self, archive: zipfile.ZipFile, infos: list[zipfile.ZipInfo]
    ) -> tuple[list[_Paragraph], list[TableData]]:
        """Read body text from the IWA object graph of a Pages 5+ document."""
        objects: dict[int, IWAObject] = {}
        for info in infos:
            if not info.filename.endswith(".iwa"):
                continue
            if info.file_size > self.options.max_file_bytes:
                raise DocumentLoadError(
                    f"Pages archive member {info.filename} is {info.file_size} "
                    f"bytes, exceeding the max_file_bytes limit of "
                    f"{self.options.max_file_bytes}."
                )
            for obj in iter_objects(archive.read(info)):
                objects[obj.identifier] = obj

        document = next(
            (o for o in objects.values() if o.message_type == _TP_DOCUMENT_ARCHIVE),
            None,
        )
        if document is None:
            raise DocumentLoadError(
                f"Pages document with hash {self.document_hash} has no "
                "TP.DocumentArchive; the container may be corrupt or "
                "password-protected."
            )

        body_ref = read_fields(document.payload).get(_DOCUMENT_BODY_FIELD, [None])[0]
        target = read_reference(body_ref) if isinstance(body_ref, bytes) else None
        storage = objects.get(target) if target is not None else None
        if storage is None or storage.message_type != _TSWP_STORAGE_ARCHIVE:
            raise DocumentLoadError(
                f"Pages document with hash {self.document_hash} does not reference "
                "a body text storage."
            )

        fields = read_fields(storage.payload)
        paragraphs = _split_paragraphs(
            _iwa_storage_text(fields), _iwa_storage_runs(fields, objects)
        )
        paragraphs.extend(_iwa_text_box_paragraphs(document, objects))
        return paragraphs, _iwa_tables(objects)

    def _read_legacy_document(
        self, archive: zipfile.ZipFile, member: str
    ) -> tuple[list[_Paragraph], list[TableData]]:
        """Read body text from the ``index.xml`` of an iWork '09 document."""
        raw = archive.read(member)
        if member.endswith(".gz"):
            # max_total_bytes only counts the stored size of a gzipped member, so
            # a small index.xml.gz could otherwise expand without bound. Cap the
            # output instead of using gzip.decompress, which has no limit.
            limit = min(_MAX_LEGACY_XML_BYTES, self.options.max_total_bytes)
            try:
                decompressor = zlib.decompressobj(wbits=31)
                raw = decompressor.decompress(raw, limit)
                if decompressor.unconsumed_tail:
                    raise DocumentLoadError(
                        f"'{member}' in Pages document with hash "
                        f"{self.document_hash} expands beyond the {limit} byte "
                        "limit."
                    )
            except zlib.error as exc:
                raise DocumentLoadError(
                    f"Could not decompress '{member}' in Pages document with hash "
                    f"{self.document_hash}."
                ) from exc

        try:
            root = ET.fromstring(raw)
        except Exception as exc:
            raise DocumentLoadError(
                f"Could not parse '{member}' in Pages document with hash "
                f"{self.document_hash}."
            ) from exc

        style_names = {
            element.get(_SF_ATTR_IDENT): element.get(_SF_ATTR_NAME)
            for element in root.iter(_SF_PARAGRAPH_STYLE)
            if element.get(_SF_ATTR_IDENT)
        }
        character_styles = {
            element.get(_SF_ATTR_IDENT): _legacy_formatting(element)
            for element in root.iter(_SF_CHARACTER_STYLE)
            if element.get(_SF_ATTR_IDENT)
        }

        list_styles = _legacy_list_styles(root)

        paragraphs: list[_Paragraph] = []
        for para in _iter_body_paragraphs(root):
            runs = _legacy_runs(para, character_styles)
            if not runs:
                continue
            style = para.get(_SF_ATTR_STYLE)
            label, level = _label_for_style(style_names.get(style))
            paragraphs.append(
                _Paragraph(runs, label, level, _legacy_list_label(para, list_styles))
            )

        return paragraphs, _read_legacy_tables(root)

    @override
    def is_valid(self) -> bool:
        return self._valid

    @classmethod
    @override
    def supports_pagination(cls) -> bool:
        return False

    @classmethod
    @override
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.IWORK_PAGES}

    @override
    def convert(self) -> DoclingDocument:
        if not self.is_valid():
            raise RuntimeError(
                f"Cannot convert Pages document with hash {self.document_hash} "
                "because the backend failed to init."
            )

        origin = DocumentOrigin(
            filename=self.file.name or "file",
            mimetype=_PAGES_MIMETYPE,
            binary_hash=self.document_hash,
        )
        doc = DoclingDocument(name=self.file.stem or "file", origin=origin)

        lists = _ListStack(doc)
        for paragraph in self._paragraphs:
            _add_paragraph(doc, paragraph, lists)

        # Pages keeps tables outside the body text flow, so they cannot be
        # interleaved with the paragraphs and are appended instead.
        for table in self._tables:
            doc.add_table(data=table)

        return doc


def _iter_text_excluding_ghosts(element: Element) -> list[str]:
    """Collect text under ``element``, skipping ``sf:ghost-text`` subtrees.

    Walked with an explicit stack rather than recursively: nesting depth in the
    XML is attacker-controlled, and a recursive walk exhausts the interpreter
    stack on a deeply nested document.
    """
    parts: list[str] = []
    # Each entry is (node, want_tail): want_tail entries emit the node's trailing
    # text after its subtree has been visited.
    stack: list[tuple[Element, bool]] = [(element, False)]

    while stack:
        node, want_tail = stack.pop()
        if want_tail:
            if node.tail:
                parts.append(node.tail)
            continue

        if node.text:
            parts.append(node.text)

        # Push in reverse so children pop in document order, each followed by its
        # own tail. A ghost-text child is skipped but still contributes its tail.
        for child in reversed(list(node)):
            stack.append((child, True))
            if child.tag != _SF_GHOST_TEXT:
                stack.append((child, False))

    return parts


def _clean(text: str) -> str:
    """Drop the placeholders Apple writes where an inline attachment sits.

    Whitespace is deliberately left alone: a run boundary can fall mid-sentence,
    so the space on either side of a formatted phrase belongs to the paragraph
    and is trimmed once, by :func:`_trim`, rather than at every boundary.
    """
    return text.replace(_OBJECT_REPLACEMENT, "")


def _trim(runs: list[_Run]) -> tuple[_Run, ...]:
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


def _iwa_style_name(payload: bytes) -> str | None:
    """Read a paragraph style's name out of its ``TSS`` super message.

    ``TSWP.ParagraphStyleArchive`` wraps a ``TSS.StyleArchive`` that carries the
    human-facing name ("Body", "Heading 1"). Anonymous styles — the ones Pages
    creates for ad-hoc formatting — have no name and are treated as body text.

    Args:
        payload: The encoded ``TSWP.ParagraphStyleArchive``.

    Returns:
        The style name, or None when the style is anonymous.
    """
    super_message = read_fields(payload).get(_STYLE_SUPER_FIELD, [None])[0]
    if not isinstance(super_message, bytes):
        return None
    name = read_fields(super_message).get(_STYLE_NAME_FIELD, [None])[0]
    if not isinstance(name, bytes):
        return None
    try:
        return name.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _split_paragraphs(text: str, runs: _StorageRuns) -> list[_Paragraph]:
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
    paragraphs: list[_Paragraph] = []
    offset = 0

    for line in text.split("\n"):
        pieces = _runs_for(line, offset, runs.characters)
        if pieces:
            label, level = _label_for_style(_value_at(runs.styles, offset))
            paragraphs.append(
                _Paragraph(pieces, label, level, _list_label_at(runs, offset))
            )
        offset += len(line) + 1  # + 1 for the newline that split consumed

    return paragraphs


def _list_label_at(runs: _StorageRuns, offset: int) -> _ListLabel | None:
    """Return how the paragraph starting at ``offset`` is labelled as a list item."""
    style = _value_at(runs.lists, offset)
    if style is None:
        return None
    return style.label(_value_at(runs.depths, offset) or 0)


def _runs_for(
    line: str, start: int, character_runs: list[tuple[int, Formatting | None]]
) -> tuple[_Run, ...]:
    """Cut one line into runs at the character style boundaries inside it."""
    if not character_runs:
        return _trim([_Run(_clean(line), None)])

    # Boundaries are absolute character indices; keep the ones inside this line.
    boundaries = [start] + [
        index for index, _ in character_runs if start < index < start + len(line)
    ]
    runs: list[_Run] = []

    for position, begin in enumerate(boundaries):
        end = (
            boundaries[position + 1]
            if position + 1 < len(boundaries)
            else start + len(line)
        )
        piece = _clean(line[begin - start : end - start])
        if piece:
            runs.append(_Run(piece, _value_at(character_runs, begin)))

    return _trim(runs)


def _value_at(table: list[tuple[int, _T]], index: int) -> _T | None:
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


def _iwa_storage_text(fields: dict[int, list[int | bytes]]) -> str:
    """Join the text pieces of a ``TSWP.StorageArchive``."""
    return "".join(
        value.decode("utf-8", errors="replace")
        for value in fields.get(_STORAGE_TEXT_FIELD, [])
        if isinstance(value, bytes)
    )


def _iwa_storage_runs(
    fields: dict[int, list[int | bytes]], objects: dict[int, IWAObject]
) -> _StorageRuns:
    """Resolve every run table a ``TSWP.StorageArchive`` carries.

    Args:
        fields: Decoded fields of the storage.
        objects: Every object in the document, keyed by identifier.

    Returns:
        The tables, each sorted by character index.
    """
    return _StorageRuns(
        styles=_iwa_object_runs(
            fields,
            _STORAGE_PARAGRAPH_STYLE_FIELD,
            objects,
            _TSWP_PARAGRAPH_STYLE,
            _iwa_style_name,
        ),
        characters=_iwa_object_runs(
            fields,
            _STORAGE_CHARACTER_STYLE_FIELD,
            objects,
            _TSWP_CHARACTER_STYLE,
            _iwa_formatting,
        ),
        lists=_iwa_object_runs(
            fields,
            _STORAGE_LIST_STYLE_FIELD,
            objects,
            _TSWP_LIST_STYLE,
            _iwa_list_style,
        ),
        depths=_iwa_depth_runs(fields),
    )


def _iwa_object_runs(
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
    for entry in _safe_fields(table[0]).get(1, []):
        if not isinstance(entry, bytes):
            continue
        parsed = _safe_fields(entry)
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


def _iwa_depth_runs(fields: dict[int, list[int | bytes]]) -> list[tuple[int, int]]:
    """Resolve the list depth table, whose entries hold numbers, not references."""
    table = fields.get(_STORAGE_LIST_DEPTH_FIELD, [])
    if not table or not isinstance(table[0], bytes):
        return []

    runs: list[tuple[int, int]] = []
    for entry in _safe_fields(table[0]).get(1, []):
        if not isinstance(entry, bytes):
            continue
        parsed = _safe_fields(entry)
        index = parsed.get(1, [None])[0]
        depth = parsed.get(2, [None])[0]
        if isinstance(index, int) and isinstance(depth, int):
            runs.append((index, depth))

    runs.sort(key=lambda run: run[0])
    return runs


def _iwa_list_style(payload: bytes) -> _ListStyle:
    """Read a ``TSWP.ListStyleArchive`` as its per-depth label ladder."""
    fields = _safe_fields(payload)
    label_types = tuple(
        value
        for value in fields.get(_LIST_LABEL_TYPES_FIELD, [])
        if isinstance(value, int)
    )
    strings = tuple(
        value.decode("utf-8", errors="replace")
        for value in fields.get(_LIST_STRINGS_FIELD, [])
        if isinstance(value, bytes)
    )
    return _ListStyle(label_types, strings)


def _iwa_formatting(payload: bytes) -> Formatting | None:
    """Read a character style's property map as a :class:`Formatting`."""
    properties = _safe_fields(payload).get(_STYLE_PROPERTIES_FIELD, [None])[0]
    if not isinstance(properties, bytes):
        return None

    flags = {}
    for field, label in _CHARACTER_PROPERTY_LABELS.items():
        values = _safe_fields(properties).get(field, [])
        if any(isinstance(v, int) and v for v in values):
            flags[label] = True

    return Formatting(**flags) if flags else None


def _read_legacy_tables(root: Element) -> list[TableData]:
    """Build table data from the ``sf:tabular-model`` elements of an '09 document.

    Cells are stored flat in ``sf:datasource``, in row-major order, so the grid
    dimensions on ``sf:grid`` are what give them their positions.

    Args:
        root: The parsed ``index.xml`` root element.

    Returns:
        One :class:`TableData` per table, in document order.
    """
    tables: list[TableData] = []

    for model in root.iter(_SF_TABULAR_MODEL):
        grid = next(iter(model.iter(_SF_GRID)), None)
        if grid is None:
            continue

        num_cols = _int_attr(grid, _SF_ATTR_NUMCOLS)
        num_rows = _int_attr(grid, _SF_ATTR_NUMROWS)
        header_rows = _int_attr(model, _SF_ATTR_HEADER_ROWS) or 0
        if not num_cols or not num_rows:
            continue

        values = [
            _clean(cell.get(_SFA_ATTR_STRING) or "".join(cell.itertext())).strip()
            for cell in model.iter(_SF_CELL_TEXT)
        ]
        if not values:
            continue

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

        tables.append(
            TableData(num_rows=num_rows, num_cols=num_cols, table_cells=cells)
        )

    return tables


def _int_attr(element: Element, name: str) -> int | None:
    """Read an integer attribute, tolerating absent or malformed values."""
    raw = element.get(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _label_for_style(style_name: str | None) -> tuple[DocItemLabel, int | None]:
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

    match = _HEADING_PATTERN.match(name)
    if match:
        # A bare "Heading" is the top level: Pages' Layout template pairs it
        # with "Subheading" rather than numbering them.
        level = int(match.group(1)) if match.group(1) else 1
        return DocItemLabel.SECTION_HEADER, min(level, 6)

    return DocItemLabel.TEXT, None


def _iter_body_paragraphs(root: Element) -> list[Element]:
    """Collect the body paragraphs of an '09 document, skipping page furniture.

    Headers, footers and footnotes each hold their own ``sf:text-body``, so a
    plain ``root.iter()`` would pull their paragraphs into the body flow. They
    are pruned instead, which matches the IWA reader: it follows
    ``TP.DocumentArchive`` to the body storage and never sees them.

    Args:
        root: The parsed ``index.xml`` root element.

    Returns:
        The body paragraphs, in document order.
    """
    paragraphs: list[Element] = []
    # Explicit stack, for the same reason the text walk uses one: nesting depth
    # is attacker-controlled.
    stack: list[Element] = [root]

    while stack:
        node = stack.pop()
        if node.tag == _SF_PARAGRAPH:
            paragraphs.append(node)
        for child in reversed(list(node)):
            if child.tag not in _SF_FURNITURE:
                stack.append(child)

    return paragraphs


def _iwa_tables(objects: dict[int, IWAObject]) -> list[TableData]:
    """Build table data from the ``TST`` archives of a Pages 5+ document.

    A table keeps its geometry on the model, its cell contents in a shared value
    list, and the placement of those values in tiles. Cells reference their value
    by key, so equal values share one entry — which is why the tiles have to be
    read rather than assuming the value list is already in cell order.

    Args:
        objects: Every object in the document, keyed by identifier.

    Returns:
        One :class:`TableData` per table that could be read, in object order.
    """
    tables: list[TableData] = []

    for model in objects.values():
        if model.message_type != _TST_TABLE_MODEL:
            continue

        fields = _safe_fields(model.payload)
        num_rows = fields.get(_TABLE_ROWS_FIELD, [None])[0]
        num_cols = fields.get(_TABLE_COLS_FIELD, [None])[0]
        store_raw = fields.get(_TABLE_DATA_STORE_FIELD, [None])[0]
        if not isinstance(num_rows, int) or not isinstance(num_cols, int):
            continue
        if not num_rows or not num_cols or not isinstance(store_raw, bytes):
            continue

        header_rows = fields.get(_TABLE_HEADER_ROWS_FIELD, [0])[0]
        store = _safe_fields(store_raw)
        strings = _iwa_string_table(store, objects)

        cells: list[TableCell] = []
        for tile in _iwa_tiles(store, objects):
            cells.extend(
                _iwa_tile_cells(
                    tile,
                    strings,
                    num_cols,
                    header_rows if isinstance(header_rows, int) else 0,
                )
            )

        if cells:
            tables.append(
                TableData(num_rows=num_rows, num_cols=num_cols, table_cells=cells)
            )

    return tables


def _iwa_string_table(
    store: dict[int, list[int | bytes]], objects: dict[int, IWAObject]
) -> dict[int, str]:
    """Read a table's shared strings, keyed as its cells reference them."""
    reference = store.get(_STORE_STRINGS_FIELD, [None])[0]
    target = read_reference(reference) if isinstance(reference, bytes) else None
    data_list = objects.get(target) if target is not None else None
    if data_list is None or data_list.message_type != _TST_DATA_LIST:
        return {}

    strings: dict[int, str] = {}
    for entry in _safe_fields(data_list.payload).get(3, []):
        if not isinstance(entry, bytes):
            continue
        parsed = _safe_fields(entry)
        key = parsed.get(1, [None])[0]
        value = next((v for v in parsed.get(3, []) if isinstance(v, bytes)), None)
        if isinstance(key, int) and value is not None:
            strings[key] = value.decode("utf-8", errors="replace")
    return strings


def _iwa_tiles(
    store: dict[int, list[int | bytes]], objects: dict[int, IWAObject]
) -> list[IWAObject]:
    """Resolve the tiles a table's data store points at."""
    tiles: list[IWAObject] = []
    container = store.get(_STORE_TILES_FIELD, [None])[0]
    if not isinstance(container, bytes):
        return tiles

    for entry in _safe_fields(container).get(1, []):
        if not isinstance(entry, bytes):
            continue
        reference = _safe_fields(entry).get(2, [None])[0]
        target = read_reference(reference) if isinstance(reference, bytes) else None
        tile = objects.get(target) if target is not None else None
        if tile is not None and tile.message_type == _TST_TILE:
            tiles.append(tile)
    return tiles


def _iwa_tile_cells(
    tile: IWAObject, strings: dict[int, str], num_cols: int, header_rows: int
) -> list[TableCell]:
    """Read one tile's cells, placing them by each row's per-column offsets."""
    cells: list[TableCell] = []

    for row_message in _safe_fields(tile.payload).get(_TILE_ROWS_FIELD, []):
        if not isinstance(row_message, bytes):
            continue
        row = _safe_fields(row_message)
        row_index = row.get(_ROW_INDEX_FIELD, [None])[0]
        storage = row.get(_ROW_STORAGE_FIELD, [None])[0]
        offsets = row.get(_ROW_OFFSETS_FIELD, [None])[0]
        if not isinstance(row_index, int):
            continue
        if not isinstance(storage, bytes) or not isinstance(offsets, bytes):
            continue

        for column in range(min(num_cols, len(offsets) // 2)):
            start = int.from_bytes(
                offsets[column * 2 : column * 2 + 2], "little", signed=True
            )
            text = _iwa_cell_text(storage, start, strings)
            if text is None:
                continue
            cells.append(
                TableCell(
                    text=text,
                    start_row_offset_idx=row_index,
                    end_row_offset_idx=row_index + 1,
                    start_col_offset_idx=column,
                    end_col_offset_idx=column + 1,
                    column_header=row_index < header_rows,
                )
            )

    return cells


def _iwa_cell_text(storage: bytes, start: int, strings: dict[int, str]) -> str | None:
    """Read one packed cell, or None when there is nothing readable there.

    Only the text cell layout is decoded. Any other value type — a number, a
    date, a formula result — is skipped rather than guessed at from bytes whose
    meaning has not been established against a real document.
    """
    if start < 0 or start + _CELL_KEY_OFFSET + 4 > len(storage):
        return None
    if storage[start] != _CELL_VERSION or storage[start + 1] != _CELL_TYPE_TEXT:
        return None

    key_at = start + _CELL_KEY_OFFSET
    key = int.from_bytes(storage[key_at : key_at + 4], "little")
    return strings.get(key)


def _safe_fields(payload: bytes) -> dict[int, list[int | bytes]]:
    """Decode a message, treating an unreadable one as empty.

    The table archives carry sub-messages this reader has no need to understand,
    some of which use wire types the fields it does want never use. Failing the
    whole document over one of them would be wrong.
    """
    try:
        return read_fields(payload)
    except DocumentLoadError:
        return {}


def _iwa_text_box_paragraphs(
    document: IWAObject, objects: dict[int, IWAObject]
) -> list[_Paragraph]:
    """Read the text of the document's text boxes.

    Text boxes are floating drawables, so they are reached from
    ``TP.DocumentArchive`` through its drawables field rather than from the body
    storage. Following that ownership path is what keeps headers, footers and
    footnotes out: they hang off other fields entirely.

    Args:
        document: The ``TP.DocumentArchive`` of the document.
        objects: Every object in the document, keyed by identifier.

    Returns:
        The text boxes' paragraphs, ordered by the object graph.
    """
    drawables = read_fields(document.payload).get(_DOCUMENT_DRAWABLES_FIELD, [None])[0]
    if not isinstance(drawables, bytes):
        return []

    container = read_reference(drawables)
    root = objects.get(container) if container is not None else None
    if root is None:
        return []

    paragraphs: list[_Paragraph] = []
    for shape_id in sorted(_iwa_referenced_ids(root.payload)):
        shape = objects.get(shape_id)
        if shape is None or shape.message_type != _TSWP_SHAPE_INFO:
            continue

        for storage_id in sorted(_iwa_referenced_ids(shape.payload)):
            storage = objects.get(storage_id)
            if storage is None or storage.message_type != _TSWP_STORAGE_ARCHIVE:
                continue
            fields = read_fields(storage.payload)
            paragraphs.extend(
                _split_paragraphs(
                    _iwa_storage_text(fields), _iwa_storage_runs(fields, objects)
                )
            )

    return paragraphs


def _iwa_referenced_ids(payload: bytes, depth: int = 0) -> set[int]:
    """Collect the object identifiers a message references, at any nesting.

    Args:
        payload: The encoded message to scan.
        depth: Current recursion depth, bounded to keep a hostile document from
            driving this arbitrarily deep.

    Returns:
        Every identifier reachable from the message.
    """
    if depth > _MAX_REFERENCE_DEPTH:
        return set()

    found: set[int] = set()
    for values in _safe_fields(payload).values():
        for value in values:
            if not isinstance(value, bytes):
                continue
            if len(value) <= _REFERENCE_MAX_BYTES:
                try:
                    target = read_reference(value)
                except DocumentLoadError:
                    continue
                if isinstance(target, int):
                    found.add(target)
            else:
                found |= _iwa_referenced_ids(value, depth + 1)
    return found


class _ListStack:
    """The list groups open while consecutive list items keep arriving.

    Pages records a nesting depth per paragraph rather than opening and closing
    lists, so the groups a :class:`DoclingDocument` needs are inferred here: a
    deeper item opens groups down to its depth, a shallower one closes back to
    it, and any other paragraph ends the list entirely.
    """

    def __init__(self, doc: DoclingDocument) -> None:
        self._doc = doc
        self._groups: list[ListGroup] = []

    def close(self) -> None:
        """End the list, so the next item starts a new one."""
        self._groups.clear()

    def group_for(self, depth: int) -> ListGroup:
        """Return the group a list item at ``depth`` belongs in, opening it if needed.

        Args:
            depth: The item's nesting depth, counted from zero.

        Returns:
            The innermost open group.
        """
        del self._groups[depth + 1 :]
        while len(self._groups) <= depth:
            self._groups.append(
                self._doc.add_list_group(
                    name="list", parent=self._groups[-1] if self._groups else None
                )
            )
        return self._groups[depth]


def _add_paragraph(
    doc: DoclingDocument, paragraph: _Paragraph, lists: _ListStack
) -> None:
    """Add one paragraph, keeping any character formatting attached to its runs.

    ``TextItem`` carries a single ``Formatting``, so a paragraph whose runs
    differ has to become an inline group of items — the same shape the Word and
    HTML backends produce for mixed runs.

    Args:
        doc: The document being built.
        paragraph: The paragraph to add.
        lists: The list groups currently open.
    """
    if paragraph.list_label is None:
        lists.close()
    else:
        _add_list_item(doc, paragraph, paragraph.list_label, lists)
        return

    if paragraph.label == DocItemLabel.TITLE:
        doc.add_title(text=paragraph.text)
        return
    if paragraph.label == DocItemLabel.SECTION_HEADER:
        doc.add_heading(text=paragraph.text, level=paragraph.level or 1)
        return

    runs = [run for run in paragraph.runs if run.text]
    if len(runs) <= 1:
        formatting = runs[0].formatting if runs else None
        doc.add_text(label=paragraph.label, text=paragraph.text, formatting=formatting)
        return

    # Formatting is a model, so compare rather than deduplicate through a set.
    first = runs[0].formatting
    if all(run.formatting == first for run in runs):
        doc.add_text(label=paragraph.label, text=paragraph.text, formatting=first)
        return

    group = doc.add_inline_group()
    for run in runs:
        doc.add_text(
            label=paragraph.label,
            text=run.text,
            formatting=run.formatting,
            parent=group,
        )


def _add_list_item(
    doc: DoclingDocument, paragraph: _Paragraph, label: _ListLabel, lists: _ListStack
) -> None:
    """Add one list item under the group its nesting depth belongs to."""
    group = lists.group_for(label.depth)
    runs = [run for run in paragraph.runs if run.text]
    first = runs[0].formatting if runs else None
    doc.add_list_item(
        text=paragraph.text,
        enumerated=label.enumerated,
        marker=label.marker,
        parent=group,
        formatting=first if all(run.formatting == first for run in runs) else None,
    )


def _legacy_runs(
    paragraph: Element, character_styles: dict[str | None, Formatting | None]
) -> tuple[_Run, ...]:
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
    runs: list[_Run] = []
    # (element, formatting in force, whether this visit emits the tail)
    stack: list[tuple[Element, Formatting | None, bool]] = [(paragraph, None, False)]

    while stack:
        element, formatting, want_tail = stack.pop()

        if want_tail:
            if element.tail:
                runs.append(_Run(_clean(element.tail), formatting))
            continue

        if element.text:
            runs.append(_Run(_clean(element.text), formatting))

        # Push in reverse so children pop in document order. A child's tail sits
        # outside it, so it keeps the parent's formatting.
        for child in reversed(list(element)):
            stack.append((child, formatting, True))
            if child.tag == _SF_GHOST_TEXT:
                continue
            inherited = formatting
            if child.tag == _SF_SPAN:
                inherited = character_styles.get(child.get(_SF_ATTR_STYLE), formatting)
            stack.append((child, inherited, False))

    return _trim(runs)


def _legacy_list_styles(root: Element) -> dict[str, _ListStyle]:
    """Read the ``sf:liststyle`` definitions of an '09 document by identifier.

    Args:
        root: The parsed ``index.xml`` root element.

    Returns:
        The label ladder of every named list style, keyed by its identifier.
    """
    styles: dict[str, _ListStyle] = {}

    for element in root.iter(_SF_LIST_STYLE):
        ident = element.get(_SF_ATTR_IDENT)
        if not ident or ident in styles:
            continue

        label_types: list[int] = []
        strings: list[str] = []
        for level in element.iter(_SF_LIST_LABEL_TYPE):
            if level.get(_SF_ATTR_TYPE) == _SF_LABEL_TYPE_NONE:
                label_types.append(_LABEL_TYPE_NONE)
                strings.append("")
                continue
            text_label = next(iter(level.iter(_SF_TEXT_LABEL)), None)
            kind = text_label.get(_SF_ATTR_TYPE) if text_label is not None else None
            if kind is not None and kind not in _SF_BULLET_LABEL_TYPES:
                # Anything else names a numbering sequence: decimal, upper-roman,
                # lower-alpha and the rest, which Pages counts rather than draws.
                label_types.append(_LABEL_TYPE_NUMBER)
                strings.append("")
                continue
            label_types.append(_LABEL_TYPE_STRING)
            strings.append(
                (text_label.get(_SF_ATTR_FORMAT) or "")
                if text_label is not None
                else ""
            )

        styles[ident] = _ListStyle(tuple(label_types), tuple(strings))

    return styles


def _legacy_list_label(
    paragraph: Element, list_styles: dict[str, _ListStyle]
) -> _ListLabel | None:
    """Return how an '09 paragraph is labelled as a list item, if it is one.

    Args:
        paragraph: An ``sf:p`` element.
        list_styles: The document's list styles, keyed by identifier.

    Returns:
        The label, or None when the paragraph names no list style or the style
        leaves its level unlabelled.
    """
    style = list_styles.get(paragraph.get(_SF_ATTR_LIST_STYLE) or "")
    if style is None:
        return None
    # sf:list-level counts from one, unlike the depth the IWA reader works in.
    level = _int_attr(paragraph, _SF_ATTR_LIST_LEVEL) or 1
    return style.label(max(level - 1, 0))


def _legacy_formatting(style: Element) -> Formatting | None:
    """Read an iWork '09 character style's property map as a ``Formatting``."""
    flags = {}
    for element in style.iter():
        label = _SF_PROPERTY_LABELS.get(element.tag)
        if label is None:
            continue
        number = next(
            (
                child.get(_SFA_ATTR_NUMBER)
                for child in element
                if child.get(_SFA_ATTR_NUMBER) is not None
            ),
            None,
        )
        if number not in (None, "0"):
            flags[label] = True

    return Formatting(**flags) if flags else None
