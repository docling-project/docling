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
headings are recovered from them, and tables are read from the structures each
generation uses for them.
"""

import logging
import mimetypes
import re
import zipfile
import zlib
from io import BytesIO
from pathlib import Path
from typing import NamedTuple
from xml.etree.ElementTree import Element

import defusedxml.ElementTree as ET
from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    TableCell,
    TableData,
)
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


class _Paragraph(NamedTuple):
    """One block of body text with the label its Pages style implies."""

    text: str
    label: DocItemLabel
    level: int | None


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

_STORAGE_PARAGRAPH_STYLE_FIELD = 5
"""Field of ``TSWP.StorageArchive`` holding the paragraph style run table.

Each entry pairs a character index with a reference to the style that applies
from there. Entries without a reference leave the preceding style in force.
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
        * Tables are read from iWork '09 documents only. A Pages 5+ table keeps
          its cell contents in a packed per-row storage buffer (``Index/Tables/
          Tile.iwa``) that still needs decoding; the strings are reachable but
          their grid positions are not, and guessing at the order would silently
          mangle any table containing repeated values.
        * Lists are not recovered.
        * Text boxes, headers, footers, footnotes and comments are not
          included; only the main body storage is read, in both generations.
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
        text = "".join(
            value.decode("utf-8", errors="replace")
            for value in fields.get(_STORAGE_TEXT_FIELD, [])
            if isinstance(value, bytes)
        )
        styles = self._iwa_style_runs(fields, objects)
        return _split_paragraphs(text, styles), []

    @staticmethod
    def _iwa_style_runs(
        fields: dict[int, list[int | bytes]],
        objects: dict[int, IWAObject],
    ) -> list[tuple[int, str | None]]:
        """Resolve the paragraph style run table to (character index, style name).

        Entries without a style reference leave the previous style in force, so
        only the ones that carry a reference are returned.

        Args:
            fields: Decoded fields of the body ``TSWP.StorageArchive``.
            objects: Every object in the document, keyed by identifier.

        Returns:
            Character index and style name pairs, in document order.
        """
        table = fields.get(_STORAGE_PARAGRAPH_STYLE_FIELD, [])
        if not table or not isinstance(table[0], bytes):
            return []

        runs: list[tuple[int, str | None]] = []
        for entry in read_fields(table[0]).get(1, []):
            if not isinstance(entry, bytes):
                continue
            parsed = read_fields(entry)
            index = parsed.get(1, [None])[0]
            reference = parsed.get(2, [None])[0]
            if not isinstance(index, int) or not isinstance(reference, bytes):
                continue
            target = read_reference(reference)
            style = objects.get(target) if target is not None else None
            if style is None or style.message_type != _TSWP_PARAGRAPH_STYLE:
                continue
            runs.append((index, _iwa_style_name(style.payload)))

        runs.sort(key=lambda run: run[0])
        return runs

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

        paragraphs: list[_Paragraph] = []
        for para in _iter_body_paragraphs(root):
            # itertext() would pull in the template placeholder text, which is
            # not document content.
            text = _clean("".join(_iter_text_excluding_ghosts(para)))
            if not text:
                continue
            label, level = _label_for_style(style_names.get(para.get(_SF_ATTR_STYLE)))
            paragraphs.append(_Paragraph(text, label, level))

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

        for paragraph in self._paragraphs:
            if paragraph.label == DocItemLabel.TITLE:
                doc.add_title(text=paragraph.text)
            elif paragraph.label == DocItemLabel.SECTION_HEADER:
                doc.add_heading(text=paragraph.text, level=paragraph.level or 1)
            else:
                doc.add_text(label=paragraph.label, text=paragraph.text)

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
    return text.replace(_OBJECT_REPLACEMENT, "").strip()


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


def _split_paragraphs(
    text: str, style_runs: list[tuple[int, str | None]]
) -> list[_Paragraph]:
    """Split a TSWP text run into labelled paragraphs.

    Apple separates paragraphs with newlines and pads empty ones, so blank
    results are dropped rather than emitted as empty text items. The style runs
    are keyed by character index into ``text``, and each one stays in force until
    the next begins.

    Args:
        text: The concatenated text of the body storage.
        style_runs: Character index and style name pairs, in document order.

    Returns:
        The non-empty paragraphs, each labelled by its style.
    """
    paragraphs: list[_Paragraph] = []
    offset = 0
    run_index = 0
    current: str | None = None

    for line in text.split("\n"):
        # Advance through the style table to whichever run covers this line.
        while run_index < len(style_runs) and style_runs[run_index][0] <= offset:
            current = style_runs[run_index][1]
            run_index += 1

        cleaned = _clean(line)
        if cleaned:
            label, level = _label_for_style(current)
            paragraphs.append(_Paragraph(cleaned, label, level))
        offset += len(line) + 1  # + 1 for the newline that split consumed

    return paragraphs


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
            _clean(cell.get(_SFA_ATTR_STRING) or "".join(cell.itertext()))
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
