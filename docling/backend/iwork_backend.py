# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Backend for Apple Pages (``.pages``) documents.

A ``.pages`` file is a ZIP container, but what is inside changed completely with
Pages 5:

* **Pages 5 and later (2013 onwards)** store the document as ``Index/*.iwa`` —
  Snappy-framed protobuf whose schemas Apple has never published. This is what
  essentially every Pages document in circulation looks like.
* **iWork '09 and earlier** stored it as a plain ``index.xml``, optionally
  gzipped, alongside a ``QuickLook/Preview.pdf`` render that Apple stopped
  writing after that release.

Both are read into the same model, so the backend is declarative: it builds a
:class:`~docling_core.types.doc.DoclingDocument` directly rather than rendering
pages and running layout analysis over them.
"""

import logging
import mimetypes
import zipfile
import zlib
from io import BytesIO
from pathlib import Path
from typing import TypeVar
from urllib.parse import urlparse
from xml.etree.ElementTree import Element

import defusedxml.ElementTree as ET
from docling_core.types.doc import (
    ContentLayer,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    ImageRef,
)
from docling_core.types.doc.items.group import ListGroup
from docling_core.types.doc.items.text import TextItem
from PIL import Image
from pydantic import AnyUrl, ValidationError
from typing_extensions import override

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.backend.iwork import pages_iwa, pages_xml
from docling.backend.iwork.content import (
    Block,
    Comment,
    Content,
    ListLabel,
    Paragraph,
    Picture,
    Run,
    label_for_style,
)
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

_PAGES_MIMETYPE = "application/vnd.apple.pages"

# DocumentOrigin only accepts a mimetype that the stdlib knows or that
# docling-core allow-lists, and Python ships no mapping for ".pages". Teaching
# the stdlib the real Apple type keeps the origin honest without waiting on a
# docling-core release; it also makes mimetypes.guess_type() correct for callers.
mimetypes.add_type(_PAGES_MIMETYPE, ".pages")

_MODERN_INDEX_PREFIX = "Index/"

_LEGACY_INDEX_MEMBERS = ("index.xml", "index.xml.gz")


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

        self._content = Content(blocks=[])
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

    def _read_document(self, archive: zipfile.ZipFile) -> Content:
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
            return pages_iwa.read_content(
                archive, infos, self.options.max_file_bytes, self.document_hash
            )

        legacy = next((n for n in _LEGACY_INDEX_MEMBERS if n in names), None)
        if legacy is not None:
            root = self._parse_legacy_index(archive, legacy)
            return pages_xml.read_content(root, archive)

        raise DocumentLoadError(
            f"Document with hash {self.document_hash} is a ZIP archive but does "
            "not look like a Pages document: it has neither an Index/ directory "
            "nor an index.xml."
        )

    def _parse_legacy_index(self, archive: zipfile.ZipFile, member: str) -> Element:
        """Decompress and parse the ``index.xml`` of an iWork '09 document.

        Args:
            archive: The open ``.pages`` container.
            member: The name of its index member.

        Returns:
            The parsed root element.

        Raises:
            DocumentLoadError: If the member cannot be decompressed or parsed.
        """
        raw = archive.read(member)
        if member.endswith(".gz"):
            # max_total_bytes only counts the stored size of a gzipped member, so
            # a small index.xml.gz could otherwise expand without bound. Cap the
            # output instead of using gzip.decompress, which has no limit.
            limit = min(pages_xml.MAX_LEGACY_XML_BYTES, self.options.max_total_bytes)
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

        return root

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
            if isinstance(block, Paragraph) and item is not None:
                for anchor in block.anchors:
                    annotated.setdefault(anchor, item)

        _add_furniture(doc, self._content)
        _add_comments(doc, self._content.comments, annotated)
        return doc


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


def _add_comments(
    doc: DoclingDocument, comments: list[Comment], annotated: dict[str, TextItem]
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


def _add_furniture(doc: DoclingDocument, content: Content) -> None:
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
    doc: DoclingDocument, block: Block, lists: _ListStack
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
    if isinstance(block, Paragraph):
        return _add_paragraph(doc, block, lists)

    # A table or a picture ends any list it follows, the same as body text.
    lists.close()
    if isinstance(block, Picture):
        _add_picture(doc, block)
    else:
        doc.add_table(data=block)
    return None


def _add_picture(doc: DoclingDocument, picture: Picture) -> None:
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
    doc: DoclingDocument, paragraph: Paragraph, lists: _ListStack
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
    paragraph: Paragraph,
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
        first = runs[0] if runs else Run("", None)
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
    doc: DoclingDocument, paragraph: Paragraph, label: ListLabel, lists: _ListStack
) -> TextItem:
    """Add one list item under the group its nesting depth belongs to."""
    group = lists.group_for(label.depth)
    runs = [run for run in paragraph.runs if run.text]
    uniform = runs[0] if runs and _uniform(runs) else Run("", None)
    return doc.add_list_item(
        text=paragraph.text,
        enumerated=label.enumerated,
        marker=label.marker,
        parent=group,
        formatting=uniform.formatting,
        hyperlink=_hyperlink(uniform.hyperlink),
    )


def _uniform(runs: list[Run]) -> bool:
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
