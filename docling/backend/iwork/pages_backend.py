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
from urllib.parse import urlparse
from xml.etree.ElementTree import Element

import defusedxml.ElementTree as ET
from docling_core.types.doc import (
    ContentLayer,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    Formatting,
    ImageRef,
    Script,
    TableCell,
    TableData,
)
from docling_core.types.doc.items.group import ListGroup
from docling_core.types.doc.items.text import TextItem
from PIL import Image
from pydantic import AnyUrl, ValidationError
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
    """A stretch of text sharing one character style and one link."""

    text: str
    formatting: Formatting | None
    hyperlink: str | None = None


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


class _Comment(NamedTuple):
    """One comment thread, and the identifier of the text it annotates."""

    text: str
    anchor: str


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
    anchors: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        """The paragraph's full text, with its runs joined back together."""
        return "".join(run.text for run in self.runs)


class _CellValues(NamedTuple):
    """A table's shared value lists, keyed as its cells reference them.

    Cells reference their contents by key rather than holding them, so two cells
    with the same text share one entry.
    """

    strings: dict[int, str] = {}
    rich_text: dict[int, str] = {}


class _Picture(NamedTuple):
    """An image anchored in the text flow.

    ``data`` is None when the image's bytes are not in the container — Pages
    writes a placeholder for media it has not downloaded — so the picture is
    still placed, just without an image.
    """

    data: bytes | None
    name: str


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
    links: list[tuple[int, str | None]] = []


_Block = _Paragraph | _Picture | TableData
"""One piece of document content, in the order Pages lays it out."""


class _Content(NamedTuple):
    """Everything one Pages document holds.

    Page furniture is kept apart from the body flow rather than interleaved with
    it: a header belongs to every page a page master covers, not to one point in
    the text, so there is no position in ``blocks`` that would be right for it.
    """

    blocks: list[_Block]
    headers: list[_Paragraph] = []
    footers: list[_Paragraph] = []
    footnotes: list[_Paragraph] = []
    comments: list[_Comment] = []


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

_STYLE_SCRIPT_FIELD = 10
"""Field of a character style holding its superscript or subscript setting."""

_SCRIPTS = {1: Script.SUPER, 2: Script.SUB}
"""``SuperscriptType`` values of a character style, as ``Script`` values.

Zero means neither, and is the value Pages writes for ordinary text.
"""

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

_TST_TABULAR_INFO = 6000
"""Message type of ``TST.TableInfoArchive``, the drawable a table sits in."""

_TABULAR_INFO_MODEL_FIELD = 2
"""Field of ``TST.TableInfoArchive`` referencing its ``TST.TableModelArchive``."""

_TSD_IMAGE = 3005
"""Message type of ``TSD.ImageArchive``, one placed image."""

_TSD_GROUP = 3008
"""Message type of ``TSD.GroupArchive``, several drawables grouped together."""

_GROUP_CHILDREN_FIELD = 2
"""Field of ``TSD.GroupArchive`` referencing the drawables it holds."""

_IMAGE_DATA_FIELDS = (15, 13, 11, 12)
"""Fields of ``TSD.ImageArchive`` that may carry the image's bytes.

Pages keeps several renditions of a placed image and does not always write all
of them, so they are tried in descending order of fidelity: the adjusted image
first, then the original, then the placed data, then the thumbnail.
"""

_TSWP_DRAWABLE_ATTACHMENT = 2003
"""Message type of ``TSWP.DrawableAttachmentArchive``.

This is what a U+FFFC in the text resolves to: a drawable — an image, a table,
a text box — anchored at that character.
"""

_ATTACHMENT_DRAWABLE_FIELD = 1
"""Field of ``TSWP.DrawableAttachmentArchive`` referencing the anchored drawable."""

_STORAGE_ATTACHMENT_FIELD = 9
"""Field of ``TSWP.StorageArchive`` holding the attachment run table."""

_STORAGE_COMMENT_FIELD = 23
"""Field of ``TSWP.StorageArchive`` holding the comment run table.

Its entries mark the stretch of text a comment is attached to, which is how
Pages 5 records a comment: as a highlight over the words being commented on
rather than as a character in the text.
"""

_TSWP_COMMENT_FIELD = 2013
"""Message type of ``TSWP.CommentFieldArchive``, the anchor of one comment."""

_COMMENT_FIELD_STORAGE_FIELD = 1
"""Field of ``TSWP.CommentFieldArchive`` referencing the comment itself."""

_TSD_COMMENT_STORAGE = 3056
"""Message type of ``TSD.CommentStorageArchive``, one comment or one reply."""

_COMMENT_TEXT_FIELD = 1
_COMMENT_AUTHOR_FIELD = 3
_COMMENT_REPLIES_FIELD = 4
"""Fields of ``TSD.CommentStorageArchive``.

Replies are comments in their own right, so a thread is a chain to be followed.
"""

_TSK_ANNOTATION_AUTHOR = 212
"""Message type of ``TSK.AnnotationAuthorArchive``, who wrote a comment."""

_AUTHOR_NAME_FIELD = 1
"""Field of ``TSK.AnnotationAuthorArchive`` holding the author's name."""

_TSWP_NOTE = 2008
"""Message type of ``TSWP.NoteArchive``, one footnote or endnote."""

_NOTE_STORAGE_FIELD = 2
"""Field of ``TSWP.NoteArchive`` referencing the storage holding the note's text."""

_STORAGE_FOOTNOTE_FIELD = 16
"""Field of ``TSWP.StorageArchive`` holding the footnote run table.

Its entries anchor a note at the character the footnote mark occupies, which is
one of the U+FFFC placeholders in the text.
"""

_STORAGE_PAGE_MASTER_FIELD = 17
"""Field of ``TSWP.StorageArchive`` holding the page master run table.

Headers and footers hang off the page master that covers a stretch of the
document rather than off the text itself, so this is the way in to them.
"""

_TP_PAGE_MASTER = 10011
"""Message type of ``TP.PageMasterArchive``, the page layout of one section."""

_PAGE_MASTER_HEADER_FOOTER_FIELDS = (23, 24, 25)
"""Fields of ``TP.PageMasterArchive`` referencing its headers and footers.

Pages keeps three sets — first page, even pages, odd pages — and writes all of
them whether or not the author filled them in.
"""

_TP_HEADERS_AND_FOOTERS = 10143
"""Message type of ``TP.HeadersAndFootersArchive``."""

_HEADERS_FIELD = 1
_FOOTERS_FIELD = 2
"""Fields of ``TP.HeadersAndFootersArchive``, each a list of text storages."""

_TSP_PACKAGE_METADATA = 11006
"""Message type of ``TSP.PackageMetadata``, which names the container's data files."""

_PACKAGE_DATAS_FIELD = 4
"""Field of ``TSP.PackageMetadata`` listing one ``TSP.DataInfo`` per data file."""

_DATA_INFO_IDENTIFIER_FIELD = 1
_DATA_INFO_PREFERRED_NAME_FIELD = 3
_DATA_INFO_NAME_FIELD = 4
"""Fields of ``TSP.DataInfo``.

An image references a data file by identifier; the file itself is a ``Data/``
member of the container, named by ``file_name`` when Pages renamed it on import
and by ``preferred_file_name`` otherwise.
"""

_DATA_MEMBER_PREFIX = "Data/"

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
_STORE_RICH_TEXT_FIELD = 17
"""Fields of a table's data store: its tiles, and its two value lists.

A cell holding plain text references the string list; one holding styled text
references the rich text list instead, whose entries point at a whole
``TSWP.StorageArchive``.
"""

_LIST_ENTRIES_FIELD = 3
_LIST_SEGMENTS_FIELD = 4
"""Fields of ``TST.TableDataList``: its entries, and the segments they spill into.

A list long enough to be split keeps its entries in referenced segments instead,
which have the same entry shape.
"""

_ENTRY_KEY_FIELD = 1
_ENTRY_STRING_FIELD = 3
_ENTRY_RICH_TEXT_FIELD = 9
"""Fields of one value list entry: the key cells reference it by, and its value."""

_TST_TEXT_REF = 6218
"""Message type of the indirection a rich text entry points at.

It holds nothing but a reference to the ``TSWP.StorageArchive`` with the text.
"""

_TILE_ROWS_FIELD = 5
_ROW_INDEX_FIELD = 1
_ROW_STORAGE_FIELD = 3
_ROW_OFFSETS_FIELD = 4
_ROW_WIDE_STORAGE_FIELD = 6
_ROW_WIDE_OFFSETS_FIELD = 7
_ROW_WIDE_OFFSETS_FLAG = 8
"""Fields of ``TST.Tile`` and of one of its rows.

A row holds a packed cell buffer plus one ``int16`` offset per column, where a
negative offset marks a column with no cell. Pages 5.2 moved both to their own
fields and started scaling the offsets by four, keeping the older pair in place
for the benefit of releases that could not read the new one, so the newer pair
is preferred when it is there.
"""

_CELL_VERSION_LEGACY = 4
_CELL_VERSION_CURRENT = 5
"""Storage versions of a packed cell, in byte 0."""

_CELL_TYPE_TEXT = 3
_CELL_TYPE_RICH_TEXT = 9
"""Value types of a packed cell, in byte 1, that carry text."""

_CELL_KEY_OFFSET = 16
"""Where a version 4 cell keeps the key of its string."""

_CELL_FLAGS_OFFSET = 8
_CELL_VALUES_OFFSET = 12
"""Where a version 5 cell keeps its flags, and where its values begin.

The flags say which values are present; each one that is takes a fixed width,
so the position of any of them depends on all the ones before it.
"""

_CELL_FLAG_STRING = 0x8
_CELL_FLAG_RICH_TEXT = 0x10
_CELL_VALUE_WIDTHS = (
    (0x1, 16),
    (0x2, 8),
    (0x4, 8),
    (_CELL_FLAG_STRING, 4),
    (_CELL_FLAG_RICH_TEXT, 4),
)
"""The values a version 5 cell may hold, in the order they are laid out.

A decimal, a double and a duration come first, then the keys of the string and
the rich text a cell may reference. Nothing after the rich text key is needed,
so the walk stops there.
"""

_STORAGE_PARAGRAPH_STYLE_FIELD = 5
"""Field of ``TSWP.StorageArchive`` holding the paragraph style run table.

Each entry pairs a character index with a reference to the style that applies
from there. Entries without a reference leave the preceding style in force.
"""

_TSWP_LINK_FIELD = 2032
"""Message type of ``TSWP.LinkFieldArchive``, one hyperlink."""

_LINK_URL_FIELD = 2
"""Field of ``TSWP.LinkFieldArchive`` holding the address it points at."""

_STORAGE_SMART_FIELD = 11
"""Field of ``TSWP.StorageArchive`` holding the smart field run table.

Pages calls a hyperlink a smart field, alongside placeholders and date fields,
so this table is read for links and anything else in it is left alone.
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

_SF_MEDIA = f"{{{_SF_NAMESPACE}}}media"
_SF_IMAGE = f"{{{_SF_NAMESPACE}}}image"
_SF_DATA = f"{{{_SF_NAMESPACE}}}data"
_SF_ATTR_PATH = "path"

_SF_MEDIA_ELEMENTS = frozenset({_SF_MEDIA, _SF_IMAGE})
"""Elements that place an image in an iWork '09 document.

Both wrap an ``sf:data`` naming the container member that holds the bytes, and
neither is descended into once found: the renditions Pages keeps below them all
name the same picture.
"""

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

_SF_SUPERSCRIPT = f"{{{_SF_NAMESPACE}}}superscript"
"""Property-map entry of an '09 character style holding its script setting.

Its number matches the modern ``SuperscriptType``: one raises the text and two
lowers it.
"""

_SF_LINK = f"{{{_SF_NAMESPACE}}}link"
_SF_ATTR_HREF = "href"
"""The iWork '09 vocabulary for hyperlinks.

``href`` is one of the few attributes iWork writes unqualified, so it is read
through :func:`_sf_attr` rather than by namespaced name alone.
"""

_SF_PROPERTY_LABELS = {
    f"{{{_SF_NAMESPACE}}}bold": "bold",
    f"{{{_SF_NAMESPACE}}}italic": "italic",
    f"{{{_SF_NAMESPACE}}}underline": "underline",
    f"{{{_SF_NAMESPACE}}}strikethru": "strikethrough",
}
"""Property-map entries of an iWork '09 character style, as ``Formatting`` names."""

_SF_HEADER = f"{{{_SF_NAMESPACE}}}header"
_SF_FOOTER = f"{{{_SF_NAMESPACE}}}footer"
_SF_FOOTNOTES = f"{{{_SF_NAMESPACE}}}footnotes"
_SF_ANNOTATIONS = f"{{{_SF_NAMESPACE}}}annotations"

_SF_FURNITURE = frozenset({_SF_HEADER, _SF_FOOTER, _SF_FOOTNOTES, _SF_ANNOTATIONS})
"""Elements whose paragraphs are not body content.

Each carries its own ``sf:text-body``, so they have to be pruned from the body
walk by element rather than by looking for the document's body, and read
separately afterwards.
"""

_SF_ANNOTATION = f"{{{_SF_NAMESPACE}}}annotation"
_SF_ANNOTATION_FIELD = f"{{{_SF_NAMESPACE}}}annotation-field"
_SF_ATTR_TARGET = f"{{{_SF_NAMESPACE}}}target"
_SFA_ATTR_ID = f"{{{_SFA_NAMESPACE}}}ID"
"""The iWork '09 vocabulary for comments.

An ``sf:annotation`` names the ``sf:annotation-field`` it targets, and that
field wraps the stretch of body text being commented on.
"""

_HEADING_PATTERN = re.compile(r"^heading\s*(\d+)?$", re.IGNORECASE)
"""Matches Pages' built-in heading styles, e.g. "Heading 1" or bare "Heading"."""

# Apple marks inline attachments (images, footnote anchors) with U+FFFC inside
# the text run. There is no text there to emit.
_OBJECT_REPLACEMENT = "￼"


class IWorkPagesDocumentBackend(DeclarativeDocumentBackend):
    """Extract text from Apple Pages documents of either generation.

    Known limitations:
        * Only text cells are read from a table, in either of the two storage
          layouts Pages has used. A cell holding a number, a date or a formula
          result is left empty rather than guessed at.
        * A picture is placed where the document anchors it, but its caption,
          its cropping and its accessibility description are not read.
        * Bold, italic, underline, strikethrough, superscript, subscript and
          hyperlinks are recovered; other character properties, such as colour
          or capitalisation, have no equivalent here.
        * A list item whose runs differ in formatting keeps its text but loses
          the formatting, since a list item carries a single one.
        * Text boxes are read from Pages 5+ documents, where they are floating
          drawables owned by the document. An iWork '09 document keeps them in
          the body flow, so they already appear there.
        * Headers, footers and footnotes are recovered into the furniture
          content layer and comments into the notes layer, so all of them stay
          out of the reading order by default.
        * A comment records its author but not the date it was written.
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

        self._content = _Content(blocks=[])
        self._valid = False

        try:
            with zipfile.ZipFile(path_or_stream) as archive:
                self._content = self._read_document(archive)
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

    def _read_document(self, archive: zipfile.ZipFile) -> _Content:
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
    ) -> _Content:
        """Read the content of a Pages 5+ document out of its IWA object graph."""
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

        reader = _IWAReader(archive, objects)
        blocks = reader.storage_blocks(storage)
        blocks.extend(reader.floating_blocks(document))
        headers, footers = reader.page_furniture(storage)
        return _Content(
            blocks=blocks,
            headers=headers,
            footers=footers,
            footnotes=reader.footnotes(storage),
            comments=reader.comments(storage),
        )

    def _read_legacy_document(self, archive: zipfile.ZipFile, member: str) -> _Content:
        """Read the content of an iWork '09 document out of its ``index.xml``."""
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

        style_names = _legacy_styles(
            root, _SF_PARAGRAPH_STYLE, lambda element: element.get(_SF_ATTR_NAME)
        )
        character_styles = _legacy_styles(root, _SF_CHARACTER_STYLE, _legacy_formatting)

        list_styles = _legacy_list_styles(root)

        blocks: list[_Block] = []
        for element in _iter_body_elements(root):
            if element.tag == _SF_TABULAR_MODEL:
                table = _legacy_table(element)
                if table is not None:
                    blocks.append(table)
                continue
            if element.tag in _SF_MEDIA_ELEMENTS:
                picture = _legacy_picture(element, archive)
                if picture is not None:
                    blocks.append(picture)
                continue

            runs = _legacy_runs(element, character_styles)
            if not runs:
                continue
            style = element.get(_SF_ATTR_STYLE)
            label, level = _label_for_style(style_names.get(style))
            anchors = tuple(
                field.get(_SFA_ATTR_ID) or ""
                for field in element.iter(_SF_ANNOTATION_FIELD)
            )
            blocks.append(
                _Paragraph(
                    runs,
                    label,
                    level,
                    _legacy_list_label(element, list_styles),
                    tuple(anchor for anchor in anchors if anchor),
                )
            )

        def furniture(tag: str) -> list[_Paragraph]:
            return _legacy_furniture(root, tag, style_names, character_styles)

        return _Content(
            blocks=blocks,
            headers=furniture(_SF_HEADER),
            footers=furniture(_SF_FOOTER),
            footnotes=furniture(_SF_FOOTNOTES),
            comments=_legacy_comments(root),
        )

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
        annotated: dict[str, TextItem] = {}
        for block in self._content.blocks:
            item = _add_block(doc, block, lists)
            if isinstance(block, _Paragraph) and item is not None:
                for anchor in block.anchors:
                    annotated.setdefault(anchor, item)

        _add_furniture(doc, self._content)
        _add_comments(doc, self._content.comments, annotated)
        return doc


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
        pieces = _runs_for(line, offset, runs)
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


def _runs_for(line: str, start: int, runs: _StorageRuns) -> tuple[_Run, ...]:
    """Cut one line into runs at the style and link boundaries inside it.

    Args:
        line: One paragraph of the storage's text.
        start: The paragraph's character index into the storage.
        runs: The storage's run tables.

    Returns:
        The paragraph's runs, trimmed, in document order.
    """
    if not runs.characters and not runs.links:
        return _trim([_Run(_clean(line), None)])

    # Boundaries are absolute character indices; keep the ones inside this line.
    inside = {
        index
        for table in (runs.characters, runs.links)
        for index, _ in table
        if start < index < start + len(line)
    }
    boundaries = [start, *sorted(inside)]
    pieces: list[_Run] = []

    for position, begin in enumerate(boundaries):
        end = (
            boundaries[position + 1]
            if position + 1 < len(boundaries)
            else start + len(line)
        )
        text = _clean(line[begin - start : end - start])
        if text:
            pieces.append(
                _Run(
                    text,
                    _value_at(runs.characters, begin),
                    _value_at(runs.links, begin),
                )
            )

    return _trim(pieces)


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
        links=_iwa_object_runs(
            fields, _STORAGE_SMART_FIELD, objects, _TSWP_LINK_FIELD, _iwa_link
        ),
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


def _iwa_link(payload: bytes) -> str | None:
    """Read the address a ``TSWP.LinkFieldArchive`` points at."""
    url = _safe_fields(payload).get(_LINK_URL_FIELD, [None])[0]
    if not isinstance(url, bytes):
        return None
    return url.decode("utf-8", errors="replace").strip() or None


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

    decoded = _safe_fields(properties)
    active = {
        label
        for field, label in _CHARACTER_PROPERTY_LABELS.items()
        if any(isinstance(value, int) and value for value in decoded.get(field, []))
    }
    script = decoded.get(_STYLE_SCRIPT_FIELD, [None])[0]
    return _formatting(
        active, _SCRIPTS.get(script) if isinstance(script, int) else None
    )


def _legacy_table(model: Element) -> TableData | None:
    """Build table data from one ``sf:tabular-model`` of an '09 document.

    Cells are stored flat in ``sf:datasource``, in row-major order, so the grid
    dimensions on ``sf:grid`` are what give them their positions.

    Args:
        model: An ``sf:tabular-model`` element.

    Returns:
        The table, or None when its grid or its cells are missing.
    """
    grid = next(iter(model.iter(_SF_GRID)), None)
    if grid is None:
        return None

    num_cols = _int_attr(grid, _SF_ATTR_NUMCOLS)
    num_rows = _int_attr(grid, _SF_ATTR_NUMROWS)
    header_rows = _int_attr(model, _SF_ATTR_HEADER_ROWS) or 0
    if not num_cols or not num_rows:
        return None

    values = [
        _clean(cell.get(_SFA_ATTR_STRING) or "".join(cell.itertext())).strip()
        for cell in model.iter(_SF_CELL_TEXT)
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


def _legacy_picture(media: Element, archive: zipfile.ZipFile) -> _Picture | None:
    """Read an '09 image, whose bytes are a member of the container.

    Args:
        media: An ``sf:media`` or ``sf:image`` element.
        archive: The open ``.pages`` container.

    Returns:
        The picture, or None when the element names no stored data.
    """
    for data in media.iter(_SF_DATA):
        path = _sf_attr(data, _SF_ATTR_PATH)
        if not path:
            continue
        try:
            return _Picture(archive.read(path), path)
        except KeyError:
            _log.debug("Pages image data member %s is missing", path)
            return _Picture(None, path)
    return None


def _sf_attr(element: Element, name: str) -> str | None:
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
    return element.get(f"{{{_SF_NAMESPACE}}}{name}") or element.get(name)


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


def _iter_body_elements(root: Element) -> list[Element]:
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
        if node.tag == _SF_PARAGRAPH or node.tag == _SF_TABULAR_MODEL:
            elements.append(node)
            continue
        if node.tag in _SF_MEDIA_ELEMENTS:
            elements.append(node)
            continue
        for child in reversed(list(node)):
            if child.tag not in _SF_FURNITURE:
                stack.append(child)

    return elements


def _iwa_table(model: IWAObject, objects: dict[int, IWAObject]) -> TableData | None:
    """Build table data from one ``TST.TableModelArchive``.

    A table keeps its geometry on the model, its cell contents in a shared value
    list, and the placement of those values in tiles. Cells reference their value
    by key, so equal values share one entry — which is why the tiles have to be
    read rather than assuming the value list is already in cell order.

    Args:
        model: The table's ``TST.TableModelArchive``.
        objects: Every object in the document, keyed by identifier.

    Returns:
        The table, or None when nothing readable could be placed in it.
    """
    fields = _safe_fields(model.payload)
    num_rows = fields.get(_TABLE_ROWS_FIELD, [None])[0]
    num_cols = fields.get(_TABLE_COLS_FIELD, [None])[0]
    store_raw = fields.get(_TABLE_DATA_STORE_FIELD, [None])[0]
    if not isinstance(num_rows, int) or not isinstance(num_cols, int):
        return None
    if not num_rows or not num_cols or not isinstance(store_raw, bytes):
        return None

    header_rows = fields.get(_TABLE_HEADER_ROWS_FIELD, [0])[0]
    store = _safe_fields(store_raw)
    values = _iwa_cell_values(store, objects)

    cells: list[TableCell] = []
    for tile in _iwa_tiles(store, objects):
        cells.extend(
            _iwa_tile_cells(
                tile,
                values,
                num_cols,
                header_rows if isinstance(header_rows, int) else 0,
            )
        )

    if not cells:
        return None
    return TableData(num_rows=num_rows, num_cols=num_cols, table_cells=cells)


def _iwa_cell_values(
    store: dict[int, list[int | bytes]], objects: dict[int, IWAObject]
) -> _CellValues:
    """Read a table's shared value lists, keyed as its cells reference them.

    Args:
        store: Decoded fields of the table's data store.
        objects: Every object in the document, keyed by identifier.

    Returns:
        The plain strings and the rich text, each keyed by cell reference.
    """
    return _CellValues(
        strings=_iwa_value_list(
            store, _STORE_STRINGS_FIELD, objects, _iwa_entry_string
        ),
        rich_text=_iwa_value_list(
            store,
            _STORE_RICH_TEXT_FIELD,
            objects,
            lambda entry, known: _iwa_entry_rich_text(entry, known),
        ),
    )


def _iwa_value_list(
    store: dict[int, list[int | bytes]],
    field: int,
    objects: dict[int, IWAObject],
    decode: Callable[[dict[int, list[int | bytes]], dict[int, IWAObject]], str | None],
) -> dict[int, str]:
    """Read one ``TST.TableDataList``, following any segments it spills into."""
    reference = store.get(field, [None])[0]
    target = read_reference(reference) if isinstance(reference, bytes) else None
    data_list = objects.get(target) if target is not None else None
    if data_list is None or data_list.message_type != _TST_DATA_LIST:
        return {}

    payloads = [data_list.payload]
    for segment in _iwa_reference_list(data_list.payload, _LIST_SEGMENTS_FIELD):
        spilled = objects.get(segment)
        if spilled is not None:
            payloads.append(spilled.payload)

    values: dict[int, str] = {}
    for payload in payloads:
        for entry in _safe_fields(payload).get(_LIST_ENTRIES_FIELD, []):
            if not isinstance(entry, bytes):
                continue
            parsed = _safe_fields(entry)
            key = parsed.get(_ENTRY_KEY_FIELD, [None])[0]
            value = decode(parsed, objects)
            if isinstance(key, int) and value is not None:
                values[key] = value
    return values


def _iwa_entry_string(
    entry: dict[int, list[int | bytes]], objects: dict[int, IWAObject]
) -> str | None:
    """Read a value list entry that holds its string directly."""
    value = next(
        (v for v in entry.get(_ENTRY_STRING_FIELD, []) if isinstance(v, bytes)), None
    )
    return None if value is None else value.decode("utf-8", errors="replace")


def _iwa_entry_rich_text(
    entry: dict[int, list[int | bytes]], objects: dict[int, IWAObject]
) -> str | None:
    """Read a value list entry that points at a whole text storage."""
    reference = entry.get(_ENTRY_RICH_TEXT_FIELD, [None])[0]
    target = read_reference(reference) if isinstance(reference, bytes) else None
    indirection = objects.get(target) if target is not None else None
    if indirection is None or indirection.message_type != _TST_TEXT_REF:
        return None

    storage_id = _iwa_reference_field(indirection.payload, 1)
    storage = objects.get(storage_id) if storage_id is not None else None
    if storage is None or storage.message_type != _TSWP_STORAGE_ARCHIVE:
        return None
    return _iwa_storage_text(_safe_fields(storage.payload)).strip() or None


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
    tile: IWAObject, values: _CellValues, num_cols: int, header_rows: int
) -> list[TableCell]:
    """Read one tile's cells, placing them by each row's per-column offsets."""
    cells: list[TableCell] = []

    for row_message in _safe_fields(tile.payload).get(_TILE_ROWS_FIELD, []):
        if not isinstance(row_message, bytes):
            continue
        row = _safe_fields(row_message)
        row_index = row.get(_ROW_INDEX_FIELD, [None])[0]
        storage = row.get(_ROW_WIDE_STORAGE_FIELD, [None])[0]
        offsets = row.get(_ROW_WIDE_OFFSETS_FIELD, [None])[0]
        scale = 4 if row.get(_ROW_WIDE_OFFSETS_FLAG, [0])[0] else 1
        if not isinstance(storage, bytes) or not isinstance(offsets, bytes):
            storage = row.get(_ROW_STORAGE_FIELD, [None])[0]
            offsets = row.get(_ROW_OFFSETS_FIELD, [None])[0]
            scale = 1
        if not isinstance(row_index, int):
            continue
        if not isinstance(storage, bytes) or not isinstance(offsets, bytes):
            continue

        for column in range(min(num_cols, len(offsets) // 2)):
            start = int.from_bytes(
                offsets[column * 2 : column * 2 + 2], "little", signed=True
            )
            text = _iwa_cell_text(storage, start * scale, values)
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


def _iwa_cell_text(storage: bytes, start: int, values: _CellValues) -> str | None:
    """Read one packed cell, or None when there is nothing readable there.

    Only the layouts that carry text are decoded. Any other value type — a
    number, a date, a formula result — is skipped rather than guessed at from
    bytes whose meaning has not been established against a real document.

    Args:
        storage: The row's packed cell buffer.
        start: Where in the buffer this cell begins.
        values: The table's shared value lists.

    Returns:
        The cell's text, or None when it holds none.
    """
    if start < 0 or start + _CELL_VALUES_OFFSET > len(storage):
        return None

    version = storage[start]
    if version == _CELL_VERSION_LEGACY:
        if storage[start + 1] != _CELL_TYPE_TEXT:
            return None
        key_at = start + _CELL_KEY_OFFSET
        if key_at + 4 > len(storage):
            return None
        return values.strings.get(_read_uint32(storage, key_at))

    if version != _CELL_VERSION_CURRENT:
        return None
    if storage[start + 1] not in (_CELL_TYPE_TEXT, _CELL_TYPE_RICH_TEXT):
        return None

    flags = _read_uint32(storage, start + _CELL_FLAGS_OFFSET)
    offset = start + _CELL_VALUES_OFFSET
    for flag, width in _CELL_VALUE_WIDTHS:
        if not flags & flag:
            continue
        if offset + width > len(storage):
            return None
        if flag == _CELL_FLAG_STRING:
            return values.strings.get(_read_uint32(storage, offset))
        if flag == _CELL_FLAG_RICH_TEXT:
            return values.rich_text.get(_read_uint32(storage, offset))
        offset += width
    return None


def _read_uint32(buffer: bytes, at: int) -> int:
    """Read a little-endian 32-bit value out of a packed cell buffer."""
    return int.from_bytes(buffer[at : at + 4], "little")


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


class _IWAReader:
    """Reads content out of the object graph of a Pages 5+ document.

    Drawables are reached twice over — once from the attachment table of the
    text they are anchored in, and once from the document's own list of floating
    ones — so every drawable this has already emitted is remembered. That also
    bounds the walk: an object graph may contain cycles.
    """

    def __init__(self, archive: zipfile.ZipFile, objects: dict[int, IWAObject]) -> None:
        self._archive = archive
        self._objects = objects
        self._data_files = _iwa_data_files(objects)
        self._emitted: set[int] = set()

    def storage_blocks(self, storage: IWAObject) -> list[_Block]:
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
        text = _iwa_storage_text(fields)
        runs = _iwa_storage_runs(fields, self._objects)
        attachments = _iwa_attachment_runs(fields)
        comments = _iwa_attachment_runs(fields, _STORAGE_COMMENT_FIELD)

        blocks: list[_Block] = []
        offset = 0
        for line in text.split("\n"):
            end = offset + len(line) + 1
            pieces = _runs_for(line, offset, runs)
            if pieces:
                label, level = _label_for_style(_value_at(runs.styles, offset))
                anchors = tuple(
                    str(field) for index, field in comments if offset <= index < end
                )
                blocks.append(
                    _Paragraph(
                        pieces, label, level, _list_label_at(runs, offset), anchors
                    )
                )
            for index, identifier in attachments:
                if offset <= index < end:
                    blocks.extend(self._drawable_blocks(identifier))
            offset = end  # the + 1 above covers the newline that split consumed

        return blocks

    def floating_blocks(self, document: IWAObject) -> list[_Block]:
        """Read the drawables the document owns rather than anchors in its text.

        Reaching them by ownership matters: scanning every ``TSWP.StorageArchive``
        in the document would also pick up headers, footers and footnotes, which
        belong to the page rather than to the body flow.

        Args:
            document: The ``TP.DocumentArchive`` of the document.

        Returns:
            The blocks of every drawable not already emitted from the text.
        """
        drawables = read_fields(document.payload).get(
            _DOCUMENT_DRAWABLES_FIELD, [None]
        )[0]
        if not isinstance(drawables, bytes):
            return []

        container = read_reference(drawables)
        root = self._objects.get(container) if container is not None else None
        if root is None:
            return []

        blocks: list[_Block] = []
        for identifier in sorted(_iwa_referenced_ids(root.payload)):
            blocks.extend(self._drawable_blocks(identifier))
        return blocks

    def footnotes(self, storage: IWAObject) -> list[_Paragraph]:
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
        paragraphs: list[_Paragraph] = []
        for _, identifier in _iwa_attachment_runs(fields, _STORAGE_FOOTNOTE_FIELD):
            note = self._objects.get(identifier)
            if note is None or note.message_type != _TSWP_NOTE:
                continue
            text_id = _iwa_reference_field(note.payload, _NOTE_STORAGE_FIELD)
            paragraphs.extend(self._storage_paragraphs(text_id))
        return paragraphs

    def page_furniture(
        self, storage: IWAObject
    ) -> tuple[list[_Paragraph], list[_Paragraph]]:
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
        headers: list[_Paragraph] = []
        footers: list[_Paragraph] = []

        for _, identifier in _iwa_attachment_runs(fields, _STORAGE_PAGE_MASTER_FIELD):
            master = self._objects.get(identifier)
            if master is None or master.message_type != _TP_PAGE_MASTER:
                continue
            for field in _PAGE_MASTER_HEADER_FOOTER_FIELDS:
                pair = _iwa_reference_field(master.payload, field)
                bundle = self._objects.get(pair) if pair is not None else None
                if bundle is None or bundle.message_type != _TP_HEADERS_AND_FOOTERS:
                    continue
                for source, target in (
                    (_HEADERS_FIELD, headers),
                    (_FOOTERS_FIELD, footers),
                ):
                    for text_id in _iwa_reference_list(bundle.payload, source):
                        target.extend(self._storage_paragraphs(text_id))

        return _unique_paragraphs(headers), _unique_paragraphs(footers)

    def comments(self, storage: IWAObject) -> list[_Comment]:
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
        comments: list[_Comment] = []

        for _, identifier in _iwa_attachment_runs(fields, _STORAGE_COMMENT_FIELD):
            field = self._objects.get(identifier)
            if field is None or field.message_type != _TSWP_COMMENT_FIELD:
                continue
            head = _iwa_reference_field(field.payload, _COMMENT_FIELD_STORAGE_FIELD)
            comments.extend(
                _Comment(text, str(identifier)) for text in self._thread(head)
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
            if comment is None or comment.message_type != _TSD_COMMENT_STORAGE:
                continue

            fields = _safe_fields(comment.payload)
            raw = fields.get(_COMMENT_TEXT_FIELD, [None])[0]
            if isinstance(raw, bytes):
                text = raw.decode("utf-8", errors="replace").strip()
                if text:
                    texts.append(_authored(self._author(comment.payload), text))
            pending.extend(_iwa_reference_list(comment.payload, _COMMENT_REPLIES_FIELD))

        return texts

    def _author(self, payload: bytes) -> str | None:
        """Read the name of whoever wrote a comment."""
        identifier = _iwa_reference_field(payload, _COMMENT_AUTHOR_FIELD)
        author = self._objects.get(identifier) if identifier is not None else None
        if author is None or author.message_type != _TSK_ANNOTATION_AUTHOR:
            return None
        name = _safe_fields(author.payload).get(_AUTHOR_NAME_FIELD, [None])[0]
        if not isinstance(name, bytes):
            return None
        return name.decode("utf-8", errors="replace").strip() or None

    def _storage_paragraphs(self, identifier: int | None) -> list[_Paragraph]:
        """Read one storage's paragraphs, ignoring anything anchored in it."""
        storage = self._objects.get(identifier) if identifier is not None else None
        if storage is None or storage.message_type != _TSWP_STORAGE_ARCHIVE:
            return []
        fields = read_fields(storage.payload)
        return _split_paragraphs(
            _iwa_storage_text(fields), _iwa_storage_runs(fields, self._objects)
        )

    def _drawable_blocks(self, identifier: int) -> list[_Block]:
        """Read whichever kind of drawable ``identifier`` names."""
        if identifier in self._emitted:
            return []
        self._emitted.add(identifier)

        drawable = self._objects.get(identifier)
        if drawable is None:
            return []

        if drawable.message_type == _TSWP_DRAWABLE_ATTACHMENT:
            anchored = _iwa_reference_field(
                drawable.payload, _ATTACHMENT_DRAWABLE_FIELD
            )
            return self._drawable_blocks(anchored) if anchored is not None else []

        if drawable.message_type == _TSD_IMAGE:
            return [self._picture(drawable)]

        if drawable.message_type == _TST_TABULAR_INFO:
            model = _iwa_reference_field(drawable.payload, _TABULAR_INFO_MODEL_FIELD)
            table = self._objects.get(model) if model is not None else None
            if table is None or table.message_type != _TST_TABLE_MODEL:
                return []
            data = _iwa_table(table, self._objects)
            return [data] if data is not None else []

        if drawable.message_type == _TSD_GROUP:
            blocks: list[_Block] = []
            for child in _iwa_reference_list(drawable.payload, _GROUP_CHILDREN_FIELD):
                blocks.extend(self._drawable_blocks(child))
            return blocks

        if drawable.message_type == _TSWP_SHAPE_INFO:
            blocks = []
            for storage_id in sorted(_iwa_referenced_ids(drawable.payload)):
                storage = self._objects.get(storage_id)
                if (
                    storage is not None
                    and storage.message_type == _TSWP_STORAGE_ARCHIVE
                ):
                    blocks.extend(self.storage_blocks(storage))
            return blocks

        return []

    def _picture(self, image: IWAObject) -> _Picture:
        """Read a ``TSD.ImageArchive`` and the container member holding its bytes."""
        fields = _safe_fields(image.payload)
        named = ""
        for field in _IMAGE_DATA_FIELDS:
            reference = fields.get(field, [None])[0]
            if not isinstance(reference, bytes):
                continue
            data_id = read_reference(reference)
            member = self._data_files.get(data_id) if data_id is not None else None
            if member is None:
                continue
            named = named or member
            try:
                return _Picture(self._archive.read(member), member)
            except KeyError:
                # Pages names every rendition it knows of, including ones it did
                # not write into this container, so keep trying the rest.
                _log.debug("Pages image data member %s is missing", member)
        return _Picture(None, named)


def _iwa_reference_field(payload: bytes, field: int) -> int | None:
    """Read the object identifier a message's reference field points at."""
    reference = _safe_fields(payload).get(field, [None])[0]
    if not isinstance(reference, bytes):
        return None
    return read_reference(reference)


def _iwa_reference_list(payload: bytes, field: int) -> list[int]:
    """Read the object identifiers a message's repeated reference field holds."""
    identifiers = []
    for reference in _safe_fields(payload).get(field, []):
        if not isinstance(reference, bytes):
            continue
        target = read_reference(reference)
        if target is not None:
            identifiers.append(target)
    return identifiers


def _iwa_attachment_runs(
    fields: dict[int, list[int | bytes]],
    field: int = _STORAGE_ATTACHMENT_FIELD,
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
    for entry in _safe_fields(table[0]).get(1, []):
        if not isinstance(entry, bytes):
            continue
        parsed = _safe_fields(entry)
        index = parsed.get(1, [None])[0]
        reference = parsed.get(2, [None])[0]
        if not isinstance(index, int) or not isinstance(reference, bytes):
            continue
        target = read_reference(reference)
        if target is not None:
            runs.append((index, target))

    runs.sort(key=lambda run: run[0])
    return runs


def _iwa_data_files(objects: dict[int, IWAObject]) -> dict[int, str]:
    """Map each data identifier to the container member that holds its bytes.

    Args:
        objects: Every object in the document, keyed by identifier.

    Returns:
        Data identifiers and the ``Data/`` member names they name.
    """
    metadata = next(
        (o for o in objects.values() if o.message_type == _TSP_PACKAGE_METADATA), None
    )
    if metadata is None:
        return {}

    files: dict[int, str] = {}
    for entry in _safe_fields(metadata.payload).get(_PACKAGE_DATAS_FIELD, []):
        if not isinstance(entry, bytes):
            continue
        info = _safe_fields(entry)
        identifier = info.get(_DATA_INFO_IDENTIFIER_FIELD, [None])[0]
        name = info.get(_DATA_INFO_NAME_FIELD, [None])[0]
        if not isinstance(name, bytes):
            name = info.get(_DATA_INFO_PREFERRED_NAME_FIELD, [None])[0]
        if isinstance(identifier, int) and isinstance(name, bytes):
            files[identifier] = _DATA_MEMBER_PREFIX + name.decode(
                "utf-8", errors="replace"
            )
    return files


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


def _authored(author: str | None, text: str) -> str:
    """Prefix a comment with its author, the way the Word backend renders one."""
    return f"[author: {author}]: {text}" if author else text


def _unique_paragraphs(paragraphs: list[_Paragraph]) -> list[_Paragraph]:
    """Drop repeats, keeping the first of each, without reordering."""
    seen: set[str] = set()
    unique: list[_Paragraph] = []
    for paragraph in paragraphs:
        if paragraph.text in seen:
            continue
        seen.add(paragraph.text)
        unique.append(paragraph)
    return unique


def _add_comments(
    doc: DoclingDocument, comments: list[_Comment], annotated: dict[str, TextItem]
) -> None:
    """Add the document's comments, linked to the text they annotate.

    Comments go into the notes content layer, where the Word backend puts them
    too, and each is attached to the item holding the text it was written about
    whenever that text was recovered.

    Args:
        doc: The document being built.
        comments: The comments read from the Pages document.
        annotated: The item each comment anchor was recovered into.
    """
    for comment in comments:
        target = annotated.get(comment.anchor)
        doc.add_comment(text=comment.text, targets=[target] if target else None)


def _add_furniture(doc: DoclingDocument, content: _Content) -> None:
    """Add the document's headers, footers and footnotes.

    They go into the furniture layer, where the Word backend puts a header and a
    footer too, so they are available to callers that ask for it but stay out of
    the reading order by default.

    Args:
        doc: The document being built.
        content: The content read from the Pages document.
    """
    for paragraphs, label in (
        (content.headers, DocItemLabel.PAGE_HEADER),
        (content.footers, DocItemLabel.PAGE_FOOTER),
        (content.footnotes, DocItemLabel.FOOTNOTE),
    ):
        for paragraph in paragraphs:
            _add_runs(doc, paragraph, label, ContentLayer.FURNITURE)


def _add_block(
    doc: DoclingDocument, block: _Block, lists: _ListStack
) -> TextItem | None:
    """Add one block of content, in the order Pages lays the document out.

    Args:
        doc: The document being built.
        block: The block to add.
        lists: The list groups currently open.

    Returns:
        The item a paragraph became, so a comment can be attached to it, or None
        for anything a comment cannot annotate.
    """
    if isinstance(block, _Paragraph):
        return _add_paragraph(doc, block, lists)

    # A table or a picture ends any list it follows, the same as body text.
    lists.close()
    if isinstance(block, _Picture):
        _add_picture(doc, block)
    else:
        doc.add_table(data=block)
    return None


def _add_picture(doc: DoclingDocument, picture: _Picture) -> None:
    """Add one picture, embedding its image when the bytes can be decoded.

    Args:
        doc: The document being built.
        picture: The picture to add.
    """
    image: ImageRef | None = None
    if picture.data is not None:
        try:
            with Image.open(BytesIO(picture.data)) as opened:
                image = ImageRef.from_pil(image=opened.convert("RGB"), dpi=72)
        except (OSError, ValueError) as exc:
            # Pages stores whatever the author placed, including formats Pillow
            # has no decoder for. The picture still belongs in the flow.
            _log.debug("Could not decode Pages image %s: %s", picture.name, exc)

    doc.add_picture(image=image)


def _add_paragraph(
    doc: DoclingDocument, paragraph: _Paragraph, lists: _ListStack
) -> TextItem | None:
    """Add one paragraph, as a heading, a list item or body text.

    Args:
        doc: The document being built.
        paragraph: The paragraph to add.
        lists: The list groups currently open.

    Returns:
        The item the paragraph became, or its first when its runs differ.
    """
    if paragraph.list_label is None:
        lists.close()
    else:
        return _add_list_item(doc, paragraph, paragraph.list_label, lists)

    if paragraph.label == DocItemLabel.TITLE:
        return doc.add_title(text=paragraph.text)
    if paragraph.label == DocItemLabel.SECTION_HEADER:
        return doc.add_heading(text=paragraph.text, level=paragraph.level or 1)

    return _add_runs(doc, paragraph, paragraph.label)


def _add_runs(
    doc: DoclingDocument,
    paragraph: _Paragraph,
    label: DocItemLabel,
    content_layer: ContentLayer | None = None,
) -> TextItem:
    """Add a paragraph's runs, keeping the formatting attached to each one.

    ``TextItem`` carries a single ``Formatting`` and a single hyperlink, so a
    paragraph whose runs differ in either has to become an inline group of items
    — the same shape the Word and HTML backends produce for mixed runs.

    Args:
        doc: The document being built.
        paragraph: The paragraph to add.
        label: The label to give the item, or every item of the group.
        content_layer: The layer to add to, or None for the document's default.

    Returns:
        The item added, or the first of the group. A comment annotates a stretch
        of the paragraph, and the first item is the one that always exists,
        whichever run that stretch began in.
    """
    runs = [run for run in paragraph.runs if run.text]
    if _uniform(runs):
        first = runs[0] if runs else _Run("", None)
        return doc.add_text(
            label=label,
            text=paragraph.text,
            formatting=first.formatting,
            hyperlink=_hyperlink(first.hyperlink),
            content_layer=content_layer,
        )

    group = doc.add_inline_group(content_layer=content_layer)
    items = [
        doc.add_text(
            label=label,
            text=run.text,
            formatting=run.formatting,
            hyperlink=_hyperlink(run.hyperlink),
            parent=group,
            content_layer=content_layer,
        )
        for run in runs
    ]
    return items[0]


def _add_list_item(
    doc: DoclingDocument, paragraph: _Paragraph, label: _ListLabel, lists: _ListStack
) -> TextItem:
    """Add one list item under the group its nesting depth belongs to."""
    group = lists.group_for(label.depth)
    runs = [run for run in paragraph.runs if run.text]
    uniform = runs[0] if runs and _uniform(runs) else _Run("", None)
    return doc.add_list_item(
        text=paragraph.text,
        enumerated=label.enumerated,
        marker=label.marker,
        parent=group,
        formatting=uniform.formatting,
        hyperlink=_hyperlink(uniform.hyperlink),
    )


def _uniform(runs: list[_Run]) -> bool:
    """Report whether every run shares one formatting and one link.

    ``Formatting`` is a model rather than a hashable value, so the runs are
    compared against the first rather than deduplicated through a set.
    """
    if len(runs) <= 1:
        return True
    first = runs[0]
    return all(
        run.formatting == first.formatting and run.hyperlink == first.hyperlink
        for run in runs
    )


def _hyperlink(address: str | None) -> AnyUrl | Path | None:
    """Resolve a link's address to a URL or a local path.

    A Pages document can link to a file next to it as well as to a URL, and an
    address Pydantic will not accept is dropped rather than allowed to fail the
    whole conversion.

    Args:
        address: The address the document recorded, if any.

    Returns:
        The address as a URL or a path, or None when there is none to use.
    """
    if not address:
        return None
    if not urlparse(address).scheme:
        return Path(address)
    try:
        return AnyUrl(address)
    except ValidationError:
        _log.debug("Skipping malformed Pages hyperlink address: %r", address)
        return None


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
    # (element, formatting in force, link in force, whether this emits the tail)
    stack: list[tuple[Element, Formatting | None, str | None, bool]] = [
        (paragraph, None, None, False)
    ]

    while stack:
        element, formatting, link, want_tail = stack.pop()

        if want_tail:
            if element.tail:
                runs.append(_Run(_clean(element.tail), formatting, link))
            continue

        if element.text:
            runs.append(_Run(_clean(element.text), formatting, link))

        # Push in reverse so children pop in document order. A child's tail sits
        # outside it, so it keeps the parent's formatting.
        for child in reversed(list(element)):
            stack.append((child, formatting, link, True))
            if child.tag == _SF_GHOST_TEXT:
                continue
            inherited = formatting
            if child.tag == _SF_SPAN:
                inherited = character_styles.get(child.get(_SF_ATTR_STYLE), formatting)
            nested = link
            if child.tag == _SF_LINK:
                nested = _sf_attr(child, _SF_ATTR_HREF) or link
            stack.append((child, inherited, nested, False))

    return _trim(runs)


def _legacy_furniture(
    root: Element,
    tag: str,
    style_names: dict[str | None, str | None],
    character_styles: dict[str | None, Formatting | None],
) -> list[_Paragraph]:
    """Read the paragraphs of one kind of '09 page furniture.

    Pages writes a first-page, an even-page and an odd-page variant of every
    header and footer whether or not the author filled them in, so identical
    text is emitted once.

    Args:
        root: The parsed ``index.xml`` root element.
        tag: The furniture element to collect, one of :data:`_SF_FURNITURE`.
        style_names: Paragraph style names, keyed by style identifier.
        character_styles: Character style formatting, keyed by style identifier.

    Returns:
        The furniture's non-empty paragraphs, in document order.
    """
    paragraphs: list[_Paragraph] = []
    for element in root.iter(tag):
        for para in element.iter(_SF_PARAGRAPH):
            runs = _legacy_runs(para, character_styles)
            if not runs:
                continue
            label, level = _label_for_style(style_names.get(para.get(_SF_ATTR_STYLE)))
            paragraphs.append(_Paragraph(runs, label, level))
    return _unique_paragraphs(paragraphs)


def _legacy_comments(root: Element) -> list[_Comment]:
    """Read the comments of an '09 document, with the text each one annotates.

    An ``sf:annotation`` holds its text in a storage of its own and names the
    ``sf:annotation-field`` in the body that it targets.

    Args:
        root: The parsed ``index.xml`` root element.

    Returns:
        One comment per annotation, in document order.
    """
    comments: list[_Comment] = []
    for annotation in root.iter(_SF_ANNOTATION):
        text = " ".join(
            "".join(run.text for run in _legacy_runs(para, {}))
            for para in annotation.iter(_SF_PARAGRAPH)
        ).strip()
        if text:
            comments.append(_Comment(text, annotation.get(_SF_ATTR_TARGET) or ""))
    return comments


def _legacy_styles(
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
        keys = [element.get(_SF_ATTR_IDENT), element.get(_SFA_ATTR_ID)]
        if not any(keys):
            continue
        value = decode(element)
        for key in keys:
            if key:
                styles.setdefault(key, value)
    return styles


def _legacy_list_styles(root: Element) -> dict[str, _ListStyle]:
    """Read the ``sf:liststyle`` definitions of an '09 document by identifier.

    Args:
        root: The parsed ``index.xml`` root element.

    Returns:
        The label ladder of every named list style, keyed by its identifier.
    """
    styles: dict[str, _ListStyle] = {}

    for element in root.iter(_SF_LIST_STYLE):
        keys = [element.get(_SF_ATTR_IDENT), element.get(_SFA_ATTR_ID)]
        if not any(key and key not in styles for key in keys):
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

        style = _ListStyle(tuple(label_types), tuple(strings))
        for key in keys:
            if key:
                styles.setdefault(key, style)

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
    active: set[str] = set()
    script: Script | None = None

    for element in style.iter():
        number = next(
            (
                child.get(_SFA_ATTR_NUMBER)
                for child in element
                if child.get(_SFA_ATTR_NUMBER) is not None
            ),
            None,
        )
        if number in (None, "0"):
            continue

        label = _SF_PROPERTY_LABELS.get(element.tag)
        if label is not None:
            active.add(label)
        elif element.tag == _SF_SUPERSCRIPT and number is not None:
            script = _SCRIPTS.get(_as_int(number))

    return _formatting(active, script)


def _as_int(number: str) -> int:
    """Read an iWork property number, which may be written as a float."""
    try:
        return int(float(number))
    except ValueError:
        return 0


def _formatting(active: set[str], script: Script | None) -> Formatting | None:
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
