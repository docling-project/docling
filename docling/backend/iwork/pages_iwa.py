# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Reader for the object graph of a Pages 5+ (2013 onwards) document.

The container is a set of ``Index/*.iwa`` archives whose schemas Apple has never
published, so what is read here is the object graph itself: a document archive
referencing a body text storage, run tables keyed by character index for
everything applied to that text, and drawables reached either from the
attachments anchored in the text or from the document's own list of them.

Only what ``TP``, Pages' own namespace, adds to a document lives here; the
``TSWP``, ``TST`` and ``TSD`` archives underneath it are shared with the other
iWork apps and are read by :mod:`docling.backend.iwork.archives`.
"""

import zipfile

from docling.backend.iwork.archives import (
    NOTE_STORAGE_FIELD,
    STORAGE_FOOTNOTE_FIELD,
    STORAGE_PAGE_MASTER_FIELD,
    TSWP_NOTE,
    TSWP_STORAGE_ARCHIVE,
    IWAReader,
    iwa_attachment_runs,
    iwa_reference_field,
    iwa_reference_list,
    iwa_referenced_ids,
)
from docling.backend.iwork.content import (
    Block,
    Content,
    Paragraph,
    unique_paragraphs,
)
from docling.backend.iwork.iwa import (
    IWAObject,
    iter_objects,
    read_fields,
    read_reference,
)
from docling.exceptions import DocumentLoadError

DOCUMENT_DRAWABLES_FIELD = 20
"""Field of ``TP.DocumentArchive`` referencing the document's floating drawables.

Text boxes hang off this rather than off the body storage. Reaching them by
ownership matters: scanning every ``TSWP.StorageArchive`` in the document would
also pick up headers, footers and footnotes, which are deliberately excluded.
"""

TP_DOCUMENT_ARCHIVE = 10000
"""Message type of ``TP.DocumentArchive``, the root object of a Pages document."""

DOCUMENT_BODY_FIELD = 4
"""Field of ``TP.DocumentArchive`` referencing the body ``TSWP.StorageArchive``."""

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

    reader = PagesReader(archive, objects)
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


class PagesReader(IWAReader):
    """Reads the parts of a Pages document that ``TP`` adds to the shared ones.

    A Pages document hangs its floating drawables off the document archive, its
    footnotes off the body storage's note table, and its headers and footers off
    the page masters that storage runs under. None of those exist in the other
    iWork apps, so they are read here rather than in
    :class:`~docling.backend.iwork.archives.IWAReader`.
    """

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
            text_id = iwa_reference_field(note.payload, NOTE_STORAGE_FIELD)
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
                pair = iwa_reference_field(master.payload, field)
                bundle = self._objects.get(pair) if pair is not None else None
                if bundle is None or bundle.message_type != TP_HEADERS_AND_FOOTERS:
                    continue
                for source, target in (
                    (HEADERS_FIELD, headers),
                    (FOOTERS_FIELD, footers),
                ):
                    for text_id in iwa_reference_list(bundle.payload, source):
                        target.extend(self._storage_paragraphs(text_id))

        return unique_paragraphs(headers), unique_paragraphs(footers)
